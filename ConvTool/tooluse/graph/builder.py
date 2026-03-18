"""NetworkX DiGraph knowledge graph for the tool registry.

Node types  : Tool, Endpoint, Parameter, ResponseField, ConceptTag
Edge types  : HAS_ENDPOINT, TAKES_PARAM, RETURNS_FIELD, TAGGED_AS, CO_OCCURS, FEEDS
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from tooluse.registry.models import ToolRegistry

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds and queries the in-process NetworkX DiGraph knowledge graph."""

    def __init__(self) -> None:
        self._g: nx.DiGraph = nx.DiGraph()

    def __enter__(self) -> GraphBuilder:
        return self

    def __exit__(self, *exc) -> None:
        pass

    @property
    def graph(self) -> nx.DiGraph:
        return self._g

    def clear(self) -> None:
        self._g.clear()
        logger.debug("Graph cleared.")

    def create_indexes(self) -> None:
        """No-op: NetworkX provides O(1) lookup by node key."""

    def ingest(self, registry: ToolRegistry) -> None:
        """Populate the graph from a ToolRegistry."""
        self._ingest_nodes(registry)
        self._add_co_occurs_edges(registry)
        self._add_feeds_edges()
        logger.info(
            "Graph built: %d nodes, %d edges (%d endpoint nodes).",
            self._g.number_of_nodes(),
            self._g.number_of_edges(),
            sum(1 for _, a in self._g.nodes(data=True) if a.get("kind") == "Endpoint"),
        )

    def _ingest_nodes(self, registry: ToolRegistry) -> None:
        for tool in registry.tools:
            t_node = f"Tool:{tool.name}"
            self._g.add_node(
                t_node,
                kind="Tool",
                name=tool.name,
                description=tool.description,
                category=tool.category,
            )

            for concept in tool.concepts:
                c_node = f"ConceptTag:{concept.name}"
                self._g.add_node(c_node, kind="ConceptTag", name=concept.name)
                self._g.add_edge(t_node, c_node, rel="TAGGED_AS")

            for ep in tool.endpoints:
                e_node = f"Endpoint:{ep.id}"
                self._g.add_node(
                    e_node,
                    kind="Endpoint",
                    id=ep.id,
                    name=ep.name,
                    description=ep.description,
                    method=ep.method,
                    tool_name=tool.name,
                )
                self._g.add_edge(t_node, e_node, rel="HAS_ENDPOINT")

                for param in ep.parameters:
                    p_id = f"{ep.id}.{param.name}"
                    p_node = f"Parameter:{p_id}"
                    self._g.add_node(
                        p_node,
                        kind="Parameter",
                        id=p_id,
                        name=param.name,
                        type=param.type,
                        required=param.required,
                        description=param.description,
                        endpoint_id=ep.id,
                    )
                    self._g.add_edge(e_node, p_node, rel="TAKES_PARAM")

                for rf in ep.response_fields:
                    rf_id = f"{ep.id}.{rf.name}"
                    rf_node = f"ResponseField:{rf_id}"
                    self._g.add_node(
                        rf_node,
                        kind="ResponseField",
                        id=rf_id,
                        name=rf.name,
                        type=rf.type,
                        endpoint_id=ep.id,
                    )
                    self._g.add_edge(e_node, rf_node, rel="RETURNS_FIELD")

    def _add_co_occurs_edges(self, registry: ToolRegistry) -> None:
        """Link ConceptTag nodes that co-occur on the same tool (bidirectional)."""
        for tool in registry.tools:
            c_nodes = [f"ConceptTag:{c.name}" for c in tool.concepts]
            for i, c1 in enumerate(c_nodes):
                for c2 in c_nodes[i + 1:]:
                    if not self._g.has_edge(c1, c2):
                        self._g.add_edge(c1, c2, rel="CO_OCCURS")
                    if not self._g.has_edge(c2, c1):
                        self._g.add_edge(c2, c1, rel="CO_OCCURS")

    def _add_feeds_edges(self) -> None:
        """Add FEEDS edges between data-flow compatible endpoints.

        Endpoint A FEEDS Endpoint B when A has a ResponseField whose name
        matches a required Parameter of B.  This creates implicit data-flow
        edges so the sampler can discover realistic chaining opportunities.
        """
        required_by: dict[str, list[str]] = defaultdict(list)
        for _, attrs in self._g.nodes(data=True):
            if attrs.get("kind") == "Parameter" and attrs.get("required"):
                required_by[attrs["name"]].append(attrs["endpoint_id"])

        for _, attrs in self._g.nodes(data=True):
            if attrs.get("kind") == "ResponseField":
                src_ep = attrs["endpoint_id"]
                for tgt_ep in required_by.get(attrs["name"], []):
                    if src_ep == tgt_ep:
                        continue
                    src_node = f"Endpoint:{src_ep}"
                    tgt_node = f"Endpoint:{tgt_ep}"
                    if (
                        src_node in self._g
                        and tgt_node in self._g
                        and not self._g.has_edge(src_node, tgt_node)
                    ):
                        self._g.add_edge(src_node, tgt_node, rel="FEEDS")

    def get_all_endpoint_ids(self) -> list[str]:
        """Return all endpoint IDs sorted alphabetically."""
        return sorted(
            attrs["id"]
            for _, attrs in self._g.nodes(data=True)
            if attrs.get("kind") == "Endpoint"
        )

    def get_tool_endpoints(self, tool_name: str) -> list[str]:
        """Return endpoint IDs for a specific tool."""
        t_node = f"Tool:{tool_name}"
        if t_node not in self._g:
            return []
        return sorted(
            self._g.nodes[n]["id"]
            for n in self._g.successors(t_node)
            if self._g.nodes[n].get("kind") == "Endpoint"
        )

    def get_tools_in_concept(self, concept: str) -> list[str]:
        """Return tool names tagged with a given concept."""
        c_node = f"ConceptTag:{concept}"
        if c_node not in self._g:
            return []
        return sorted(
            self._g.nodes[n]["name"]
            for n in self._g.predecessors(c_node)
            if self._g.nodes[n].get("kind") == "Tool"
        )

    def get_related_concepts(self, concept: str) -> list[str]:
        """Return concept names that CO_OCCURS with the given concept."""
        c_node = f"ConceptTag:{concept}"
        if c_node not in self._g:
            return []
        return sorted(
            self._g.nodes[n]["name"]
            for n in self._g.successors(c_node)
            if self._g.nodes[n].get("kind") == "ConceptTag"
        )

    def get_all_concepts(self) -> list[str]:
        """Return all ConceptTag names."""
        return sorted(
            attrs["name"]
            for _, attrs in self._g.nodes(data=True)
            if attrs.get("kind") == "ConceptTag"
        )

    def get_all_tool_names(self) -> list[str]:
        """Return all Tool names."""
        return sorted(
            attrs["name"]
            for _, attrs in self._g.nodes(data=True)
            if attrs.get("kind") == "Tool"
        )

    def get_tool_category(self, tool_name: str) -> str:
        """Return the raw category string for a tool (e.g. ``'Finance'``)."""
        t_node = f"Tool:{tool_name}"
        return self._g.nodes[t_node].get("category", "") if t_node in self._g else ""

    def get_endpoints_sharing_concept(self, endpoint_id: str) -> list[str]:
        """Return endpoint IDs from other tools that share at least one ConceptTag."""
        e_node = f"Endpoint:{endpoint_id}"
        if e_node not in self._g:
            return []

        tool_node = next(
            (
                n for n in self._g.predecessors(e_node)
                if self._g.nodes[n].get("kind") == "Tool"
            ),
            None,
        )
        if tool_node is None:
            return []

        concept_nodes = [
            n for n in self._g.successors(tool_node)
            if self._g.nodes[n].get("kind") == "ConceptTag"
        ]

        result: set[str] = set()
        for c_node in concept_nodes:
            for other_tool in self._g.predecessors(c_node):
                if (
                    self._g.nodes[other_tool].get("kind") == "Tool"
                    and other_tool != tool_node
                ):
                    for ep in self._g.successors(other_tool):
                        if (
                            self._g.nodes[ep].get("kind") == "Endpoint"
                            and ep != e_node
                        ):
                            result.add(self._g.nodes[ep]["id"])

        return sorted(result)

    def get_chainable_endpoints(self, endpoint_id: str) -> list[str]:
        """Return endpoints reachable via FEEDS edges (output field matches required param)."""
        e_node = f"Endpoint:{endpoint_id}"
        if e_node not in self._g:
            return []
        return sorted(
            self._g.nodes[n]["id"]
            for n in self._g.successors(e_node)
            if (
                self._g.nodes[n].get("kind") == "Endpoint"
                and self._g.edges[e_node, n].get("rel") == "FEEDS"
            )
        )
