"""Conversation topology sampler (Layer 2 of the 2-layer system).

ChainSampler selects which tools to use; PatternSampler decides how they are
structured — linear, fan-out, diamond, conditional, etc.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from enum import Enum

from tooluse.graph.builder import GraphBuilder
from tooluse.graph.sampler import ChainSampler
from tooluse.sampler.base import PatternType, ToolChain, ToolChainStep

logger = logging.getLogger(__name__)


class ConvPattern(str, Enum):
    LINEAR      = "linear"
    PIPELINE    = "pipeline"
    FAN_OUT     = "fan_out"
    FAN_IN      = "fan_in"
    DIAMOND     = "diamond"
    CONDITIONAL = "conditional"


_PATTERN_WEIGHTS: list[tuple[ConvPattern, float]] = [
    (ConvPattern.LINEAR,      0.30),
    (ConvPattern.PIPELINE,    0.20),
    (ConvPattern.FAN_OUT,     0.15),
    (ConvPattern.FAN_IN,      0.15),
    (ConvPattern.DIAMOND,     0.10),
    (ConvPattern.CONDITIONAL, 0.10),
]

_PATTERN_TYPE_MAP: dict[ConvPattern, PatternType] = {
    ConvPattern.LINEAR:      PatternType.LINEAR,
    ConvPattern.PIPELINE:    PatternType.PIPELINE,
    ConvPattern.FAN_OUT:     PatternType.FAN_OUT,
    ConvPattern.FAN_IN:      PatternType.FAN_IN,
    ConvPattern.DIAMOND:     PatternType.DIAMOND,
    ConvPattern.CONDITIONAL: PatternType.CONDITIONAL,
}


class PatternSampler:
    """Picks a conversation topology and wires ChainSampler tools into ToolChainSteps."""

    def __init__(self, chain_sampler: ChainSampler, graph: GraphBuilder) -> None:
        self._chain = chain_sampler
        self._graph = graph

    def sample(
        self,
        rng: random.Random,
        seed: int,
        pattern_counts: dict[str, int] | None = None,
    ) -> ToolChain | None:
        """Build a typed ToolChain: pick pattern → walk graph → map to endpoints."""
        pattern = self._pick_pattern(rng, pattern_counts)
        num_tools = self._tools_needed(pattern, rng)

        walk = None
        for _attempt in range(5):
            candidate = self._chain.sample_chain(rng, length=num_tools)
            if candidate is None or len(candidate.tools) < 2:
                continue
            if self._is_domain_coherent(candidate.tools):
                walk = candidate
                break
            walk = candidate  # use last candidate if no coherent walk found

        if walk is None or len(walk.tools) < 2:
            return None

        if len(walk.tools) < num_tools:
            pattern = ConvPattern.LINEAR

        steps = self._build_steps(pattern, walk.tools, rng)
        if steps is None or len(steps) < 3:
            return None

        logger.info(
            "PatternSampler: pattern=%s, tools=%s, steps=%d.",
            pattern.value, walk.tools, len(steps),
        )
        return ToolChain(
            steps=steps,
            pattern_type=_PATTERN_TYPE_MAP[pattern],
            seed=seed,
        )

    def _is_domain_coherent(self, tools: list[str]) -> bool:
        """Return True if every tool in the walk shares the same domain."""
        tool_graph = self._chain.tool_graph
        domains = [
            tool_graph.nodes.get(t, {}).get("domain", "")
            for t in tools
            if t in tool_graph
        ]
        domains = [d for d in domains if d]
        if len(domains) < 2:
            return True
        top_count = Counter(domains).most_common(1)[0][1]
        return top_count == len(domains)

    def _pick_pattern(
        self,
        rng: random.Random,
        pattern_counts: dict[str, int] | None = None,
    ) -> ConvPattern:
        """Select a pattern; down-weight overused patterns when counts are supplied."""
        patterns = [p for p, _ in _PATTERN_WEIGHTS]
        base_weights = [w for _, w in _PATTERN_WEIGHTS]

        if not pattern_counts:
            return rng.choices(patterns, weights=base_weights)[0]

        total = sum(pattern_counts.values())
        avg = (total / len(patterns)) if total > 0 else 1.0
        adjusted = [
            max(bw * avg / (pattern_counts.get(p.value, 0) + 1), 0.02)
            for p, bw in _PATTERN_WEIGHTS
        ]
        return rng.choices(patterns, weights=adjusted)[0]

    def _tools_needed(self, pattern: ConvPattern, rng: random.Random) -> int:
        return {
            ConvPattern.LINEAR:      rng.randint(3, 5),
            ConvPattern.PIPELINE:    rng.randint(4, 5),
            ConvPattern.FAN_OUT:     3,
            ConvPattern.FAN_IN:      3,
            ConvPattern.DIAMOND:     4,
            ConvPattern.CONDITIONAL: rng.randint(3, 4),
        }[pattern]

    def _tool_to_endpoint(
        self, tool_name: str, rng: random.Random
    ) -> str | None:
        """Pick a random endpoint from the given tool."""
        eps = self._graph.get_tool_endpoints(tool_name)
        return rng.choice(eps) if eps else None

    def _build_steps(
        self,
        pattern: ConvPattern,
        tools: list[str],
        rng: random.Random,
    ) -> list[ToolChainStep] | None:
        """Map tool names to endpoints and wire depends_on indices for the topology."""
        eps = [self._tool_to_endpoint(t, rng) for t in tools]
        eps = [e for e in eps if e is not None]

        if len(eps) < 2:
            return None

        if pattern in (ConvPattern.LINEAR, ConvPattern.PIPELINE):
            steps = [ToolChainStep(endpoint_id=eps[0])]
            for i, ep in enumerate(eps[1:], 1):
                steps.append(ToolChainStep(endpoint_id=ep, depends_on=[i - 1]))
            return steps

        if pattern == ConvPattern.FAN_OUT:
            if len(eps) < 3:
                return self._build_steps(ConvPattern.LINEAR, tools, rng)
            steps = [
                ToolChainStep(endpoint_id=eps[0]),
                ToolChainStep(endpoint_id=eps[1], depends_on=[0]),
                ToolChainStep(endpoint_id=eps[2], depends_on=[0]),
            ]
            for i, ep in enumerate(eps[3:], 3):
                steps.append(ToolChainStep(endpoint_id=ep, depends_on=[i - 1]))
            return steps

        if pattern == ConvPattern.FAN_IN:
            if len(eps) < 3:
                return self._build_steps(ConvPattern.LINEAR, tools, rng)
            steps = [
                ToolChainStep(endpoint_id=eps[0]),
                ToolChainStep(endpoint_id=eps[1]),
                ToolChainStep(endpoint_id=eps[2], depends_on=[0, 1]),
            ]
            for i, ep in enumerate(eps[3:], 3):
                steps.append(ToolChainStep(endpoint_id=ep, depends_on=[i - 1]))
            return steps

        if pattern == ConvPattern.DIAMOND:
            if len(eps) < 4:
                return self._build_steps(ConvPattern.FAN_OUT, tools, rng)
            steps = [
                ToolChainStep(endpoint_id=eps[0]),
                ToolChainStep(endpoint_id=eps[1], depends_on=[0]),
                ToolChainStep(endpoint_id=eps[2], depends_on=[0]),
                ToolChainStep(endpoint_id=eps[3], depends_on=[1, 2]),
            ]
            for i, ep in enumerate(eps[4:], 4):
                steps.append(ToolChainStep(endpoint_id=ep, depends_on=[i - 1]))
            return steps

        if pattern == ConvPattern.CONDITIONAL:
            steps = [ToolChainStep(endpoint_id=eps[0])]
            for i, ep in enumerate(eps[1:], 1):
                steps.append(ToolChainStep(endpoint_id=ep, depends_on=[i - 1]))
            return steps

        return None
