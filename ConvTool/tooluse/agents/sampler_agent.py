"""Proposes tool chains from the knowledge graph using the 2-layer sampler."""

from __future__ import annotations

import logging
import random

from tooluse.graph.builder import GraphBuilder
from tooluse.graph.sampler import ChainSampler
from tooluse.graph.patterns import PatternSampler
from tooluse.sampler.base import ToolChain

logger = logging.getLogger(__name__)


class SamplerAgent:
    """Proposes tool chains using the 2-layer graph sampling system."""

    def __init__(self, graph: GraphBuilder) -> None:
        self._chain_sampler = ChainSampler(graph)
        self._pattern_sampler = PatternSampler(self._chain_sampler, graph)

    def propose(
        self,
        rng: random.Random,
        seed: int,
        pattern_counts: dict[str, int] | None = None,
    ) -> ToolChain:
        """Sample a typed ToolChain, retrying up to 5 times to get ≥3 steps."""
        for attempt in range(5):
            attempt_rng = random.Random(rng.randint(0, 2**31))
            chain = self._pattern_sampler.sample(
                attempt_rng, seed, pattern_counts=pattern_counts
            )
            if chain is not None and len(chain.steps) >= 3:
                logger.info(
                    "SamplerAgent: proposed %s chain with %d steps.",
                    chain.pattern_type.value,
                    len(chain.steps),
                )
                return chain
            logger.debug(
                "SamplerAgent: attempt %d failed (chain=%s), retrying.",
                attempt + 1, chain,
            )

        raise RuntimeError(
            "SamplerAgent failed to produce a valid tool chain after 5 attempts."
        )
