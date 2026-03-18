"""Diversity metrics engine for evaluating dataset variety.

Primary metric: pairwise tool-chain Jaccard dissimilarity.
Secondary metric: pattern-type entropy.
"""

from __future__ import annotations

import math
import logging
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DiversityMetrics:
    avg_jaccard_dissimilarity: float
    pattern_entropy: float
    pattern_distribution: dict[str, float]
    num_conversations: int
    corpus_memory_enabled: bool | None
    tool_coverage: float
    avg_tools_per_conversation: float


def compute_metrics(
    records: list[dict[str, Any]],
    total_tool_count: int | None = None,
) -> DiversityMetrics:
    """Compute diversity metrics over a dataset of conversation records."""
    if not records:
        return DiversityMetrics(
            avg_jaccard_dissimilarity=0.0,
            pattern_entropy=0.0,
            pattern_distribution={},
            num_conversations=0,
            corpus_memory_enabled=None,
            tool_coverage=0.0,
            avg_tools_per_conversation=0.0,
        )

    tool_chains = _extract_tool_chains(records)
    jaccard = _pairwise_jaccard_dissimilarity(tool_chains)
    pattern_dist = _pattern_distribution(records)
    entropy = _shannon_entropy(pattern_dist)

    all_tools: set[str] = set()
    total_tools_used = 0
    for chain in tool_chains:
        all_tools.update(chain)
        total_tools_used += len(chain)

    coverage = len(all_tools) / total_tool_count if total_tool_count else 0.0
    avg_tools = total_tools_used / len(records) if records else 0.0

    corpus_flag = None
    for r in records:
        meta = r.get("metadata", {})
        if "corpus_memory_enabled" in meta:
            corpus_flag = meta["corpus_memory_enabled"]
            break

    return DiversityMetrics(
        avg_jaccard_dissimilarity=round(jaccard, 4),
        pattern_entropy=round(entropy, 4),
        pattern_distribution={k: round(v, 4) for k, v in pattern_dist.items()},
        num_conversations=len(records),
        corpus_memory_enabled=corpus_flag,
        tool_coverage=round(coverage, 4),
        avg_tools_per_conversation=round(avg_tools, 2),
    )


def _extract_tool_chains(records: list[dict[str, Any]]) -> list[set[str]]:
    """Extract the set of tool names used in each conversation."""
    chains: list[set[str]] = []
    for record in records:
        tool_calls = record.get("tool_calls", [])
        tools = set()
        for tc in tool_calls:
            eid = tc.get("endpoint_id", "")
            if "." in eid:
                tools.add(eid.split(".")[0])
            else:
                tools.add(eid)
        chains.append(tools)
    return chains


def _pairwise_jaccard_dissimilarity(chains: list[set[str]]) -> float:
    """Compute average pairwise Jaccard dissimilarity across all chains.

    Jaccard dissimilarity = 1 - |A ∩ B| / |A ∪ B|
    Higher values indicate more diverse tool usage.
    """
    if len(chains) < 2:
        return 0.0

    total = 0.0
    count = 0
    for a, b in combinations(chains, 2):
        union = a | b
        if not union:
            continue
        intersection = a & b
        dissimilarity = 1.0 - len(intersection) / len(union)
        total += dissimilarity
        count += 1

    return total / count if count > 0 else 0.0


def _pattern_distribution(records: list[dict[str, Any]]) -> dict[str, float]:
    """Compute the distribution of pattern types across records."""
    counter: Counter[str] = Counter()
    for record in records:
        pattern = record.get("metadata", {}).get("pattern_type", "unknown")
        counter[pattern] += 1
    total = sum(counter.values())
    return {k: v / total for k, v in sorted(counter.items())}


def _shannon_entropy(distribution: dict[str, float]) -> float:
    """Compute Shannon entropy of a probability distribution.

    Higher entropy = more uniform distribution = more diverse patterns.
    """
    entropy = 0.0
    for p in distribution.values():
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy
