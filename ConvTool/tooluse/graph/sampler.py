"""Weighted random walk sampler over the tool-level graph projection.

Produces ordered sequences of tool names that PatternSampler (Layer 2)
converts into typed ToolChain objects.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass, field

import networkx as nx

from tooluse.graph.builder import GraphBuilder

logger = logging.getLogger(__name__)


def _normalize_domain(category: str) -> str:
    """Map a raw ToolBench category to one of ~8 broad domain labels."""
    cat = category.lower()
    if any(k in cat for k in ("finance", "stock", "crypto", "banking", "exchange")):
        return "finance"
    if any(k in cat for k in ("weather", "climate", "forecast", "meteo")):
        return "weather"
    if any(k in cat for k in ("food", "dining", "recipe", "nutrition", "health",
                               "diet", "restaurant", "keto", "cocktail")):
        return "food"
    if any(k in cat for k in ("travel", "hotel", "accommodation", "booking",
                               "flight", "airline", "airport")):
        return "travel"
    if any(k in cat for k in ("geo", "location", "map", "address", "geocod",
                               "postcode", "timezone", "postal")):
        return "location"
    if any(k in cat for k in ("entertainment", "media", "movie", "music", "game",
                               "sport", "streaming", "video")):
        return "entertainment"
    if any(k in cat for k in ("productivity", "calendar", "scheduling", "event")):
        return "productivity"
    return cat.split()[0] if cat.strip() else "general"


@dataclass
class WalkResult:
    tools: list[str]
    total_weight: float
    reasons: list[str] = field(default_factory=list)


class ChainSampler:
    """Weighted random walk over a tool-level projection of the knowledge graph."""

    def __init__(self, graph: GraphBuilder) -> None:
        self._graph = graph
        self._tool_graph: nx.DiGraph | None = None

    @property
    def tool_graph(self) -> nx.DiGraph:
        if self._tool_graph is None:
            self._tool_graph = self._build_tool_graph()
        return self._tool_graph

    def sample_chain(
        self,
        rng: random.Random,
        length: int,
        weighted: bool = True,
        allow_revisit: bool = False,
    ) -> WalkResult | None:
        """Walk the tool graph and return an ordered list of tool names."""
        g = self.tool_graph
        tools = list(g.nodes())
        if not tools:
            return None

        starters = [t for t in tools if g.in_degree(t) == 0]
        if not starters:
            starters = tools

        start = rng.choice(starters)
        chain: list[str] = [start]
        visited: set[str] = {start}
        total_weight = 0.0
        reasons: list[str] = []

        for _ in range(length - 1):
            current = chain[-1]
            neighbors = [
                (n, g[current][n])
                for n in g.successors(current)
                if allow_revisit or n not in visited
            ]

            if not neighbors:
                all_unvisited = (
                    tools if allow_revisit
                    else [t for t in tools if t not in visited]
                )
                if not all_unvisited:
                    break

                # Prefer same-domain tools at dead ends to keep the chain coherent.
                chain_domains = [g.nodes.get(t, {}).get("domain", "") for t in chain]
                chain_domains = [d for d in chain_domains if d]
                if chain_domains:
                    primary_domain = Counter(chain_domains).most_common(1)[0][0]
                    same_domain = [
                        t for t in all_unvisited
                        if g.nodes.get(t, {}).get("domain", "") == primary_domain
                    ]
                    candidates = same_domain if same_domain else all_unvisited
                else:
                    candidates = all_unvisited

                next_tool = rng.choice(candidates)
                chain.append(next_tool)
                visited.add(next_tool)
                total_weight += 0.5
                reasons.append("fallback")
                continue

            if weighted:
                weights = [data.get("weight", 1.0) for _, data in neighbors]
                next_tool = rng.choices(
                    [n for n, _ in neighbors], weights=weights
                )[0]
            else:
                next_tool = rng.choice([n for n, _ in neighbors])

            edge_data = g[current][next_tool]
            chain.append(next_tool)
            visited.add(next_tool)
            total_weight += edge_data.get("weight", 1.0)
            reasons.append(edge_data.get("reason", "adjacent"))

        logger.debug(
            "ChainSampler: walk length=%d, tools=%s, total_weight=%.2f",
            len(chain), chain, total_weight,
        )
        return WalkResult(tools=chain, total_weight=total_weight, reasons=reasons)

    def sample_subgraph(
        self,
        rng: random.Random,
        start_tool: str | None = None,
        hops: int = 2,
    ) -> list[str]:
        """BFS from a seed tool out to `hops` edges; returns all tool names in the cluster."""
        g = self.tool_graph
        tools = list(g.nodes())
        if not tools:
            return []

        if start_tool is None:
            starters = [t for t in tools if g.in_degree(t) == 0] or tools
            start_tool = rng.choice(starters)

        cluster: set[str] = {start_tool}
        frontier: set[str] = {start_tool}
        for _ in range(hops):
            next_frontier: set[str] = set()
            for node in frontier:
                for neighbor in g.successors(node):
                    if neighbor not in cluster:
                        next_frontier.add(neighbor)
            cluster |= next_frontier
            frontier = next_frontier

        return sorted(cluster)

    def _build_tool_graph(self) -> nx.DiGraph:
        """Project the full knowledge graph down to tool-level nodes.

        Edge weights: feeds=2.0 (data-flow match), tag_overlap=1.0 (shared concept).
        """
        raw: nx.DiGraph = self._graph.graph
        g: nx.DiGraph = nx.DiGraph()

        for name in self._graph.get_all_tool_names():
            category = self._graph.get_tool_category(name)
            g.add_node(name, domain=_normalize_domain(category))

        for src, tgt, data in raw.edges(data=True):
            if data.get("rel") != "FEEDS":
                continue
            src_attrs = raw.nodes[src]
            tgt_attrs = raw.nodes[tgt]
            if (
                src_attrs.get("kind") == "Endpoint"
                and tgt_attrs.get("kind") == "Endpoint"
            ):
                src_tool = src_attrs.get("tool_name")
                tgt_tool = tgt_attrs.get("tool_name")
                if src_tool and tgt_tool and src_tool != tgt_tool:
                    if g.has_edge(src_tool, tgt_tool):
                        g[src_tool][tgt_tool]["weight"] = min(
                            g[src_tool][tgt_tool]["weight"] + 0.5, 4.0
                        )
                    else:
                        g.add_edge(src_tool, tgt_tool, weight=2.0, reason="feeds")

        for concept in self._graph.get_all_concepts():
            concept_tools = self._graph.get_tools_in_concept(concept)
            for i, t1 in enumerate(concept_tools):
                for t2 in concept_tools[i + 1:]:
                    if not g.has_edge(t1, t2):
                        g.add_edge(t1, t2, weight=1.0, reason="tag_overlap")
                    if not g.has_edge(t2, t1):
                        g.add_edge(t2, t1, weight=1.0, reason="tag_overlap")

        logger.debug(
            "ChainSampler tool graph: %d tools, %d edges.",
            g.number_of_nodes(), g.number_of_edges(),
        )
        return g
