"""Data structures shared across the tool-chain sampling layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PatternType(str, Enum):
    """Structural topology of a multi-step tool-calling conversation.

    LINEAR      — A → B → C (3 steps, simple sequential)
    PIPELINE    — A → B → C → D → E (4-5 steps, deeper chain)
    FAN_OUT     — one starter tool feeding two independent branches
    FAN_IN      — two independent branches merging into a final tool
    DIAMOND     — fan-out followed by fan-in (4 steps)
    CONDITIONAL — path through B or C depending on A's result
    """
    LINEAR      = "linear"
    PIPELINE    = "pipeline"
    FAN_OUT     = "fan_out"
    FAN_IN      = "fan_in"
    DIAMOND     = "diamond"
    CONDITIONAL = "conditional"


@dataclass
class ToolChainStep:
    endpoint_id: str
    depends_on: list[int] = field(default_factory=list)


@dataclass
class ToolChain:
    steps: list[ToolChainStep]
    pattern_type: PatternType
    seed: int

    @property
    def endpoint_ids(self) -> list[str]:
        return [s.endpoint_id for s in self.steps]

    @property
    def tool_names(self) -> list[str]:
        seen: list[str] = []
        for s in self.steps:
            tool_name = s.endpoint_id.split(".")[0]
            if tool_name not in seen:
                seen.append(tool_name)
        return seen

    @property
    def num_distinct_tools(self) -> int:
        return len(set(s.endpoint_id.split(".")[0] for s in self.steps))
