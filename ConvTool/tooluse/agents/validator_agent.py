"""Validates generated conversations against structural and semantic requirements."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from tooluse.registry.models import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    num_tool_calls: int = 0
    num_distinct_tools: int = 0
    num_turns: int = 0
    num_clarifications: int = 0


class ValidatorAgent:
    """Validates conversation quality and structural requirements."""

    def __init__(
        self,
        registry: ToolRegistry,
        min_tool_calls: int = 3,
        min_distinct_tools: int = 2,
    ) -> None:
        self._registry = registry
        self._min_tool_calls = min_tool_calls
        self._min_distinct_tools = min_distinct_tools

    def validate(self, conversation: dict[str, Any]) -> ValidationReport:
        errors: list[str] = []
        warnings: list[str] = []

        messages = conversation.get("messages", [])
        tool_calls = conversation.get("tool_calls", [])
        tool_outputs = conversation.get("tool_outputs", [])

        num_turns = len(messages)
        num_clarifications = sum(
            1 for m in messages
            if m.get("role") == "assistant" and "?" in m.get("content", "")
        )

        num_tool_calls = len(tool_calls)
        if num_tool_calls < self._min_tool_calls:
            errors.append(
                f"Too few tool calls: {num_tool_calls} < {self._min_tool_calls}"
            )

        tool_names = set()
        for tc in tool_calls:
            eid = tc.get("endpoint_id", "")
            tool_name = eid.split(".")[0] if "." in eid else eid
            tool_names.add(tool_name)
        num_distinct_tools = len(tool_names)

        if num_distinct_tools < self._min_distinct_tools:
            errors.append(
                f"Too few distinct tools: {num_distinct_tools} < {self._min_distinct_tools}"
            )

        if len(tool_calls) != len(tool_outputs):
            errors.append(
                f"Mismatched tool calls ({len(tool_calls)}) and outputs ({len(tool_outputs)})"
            )

        for i, tc in enumerate(tool_calls):
            endpoint_id = tc.get("endpoint_id", "")
            endpoint = self._registry.get_endpoint(endpoint_id)
            if endpoint is None:
                warnings.append(f"Step {i}: unknown endpoint '{endpoint_id}'")
                continue

            args = tc.get("arguments", {})
            for param in endpoint.required_parameters:
                if param.name not in args:
                    warnings.append(
                        f"Step {i}: missing required param '{param.name}' "
                        f"for {endpoint_id}"
                    )

        if not messages:
            errors.append("Conversation has no messages")
        elif messages[0].get("role") != "user":
            errors.append("Conversation must start with a user message")

        if num_turns < 3:
            warnings.append(f"Very short conversation: {num_turns} turns")

        if len(tool_calls) >= 2:
            chaining_ok = self._check_chaining(tool_calls, tool_outputs)
            if not chaining_ok:
                warnings.append(
                    "No shared argument values detected across steps — "
                    "conversations may not be chaining correctly"
                )

        valid = len(errors) == 0

        report = ValidationReport(
            valid=valid,
            errors=errors,
            warnings=warnings,
            num_tool_calls=num_tool_calls,
            num_distinct_tools=num_distinct_tools,
            num_turns=num_turns,
            num_clarifications=num_clarifications,
        )

        if valid:
            logger.debug("Validation passed: %d tools, %d calls.", num_distinct_tools, num_tool_calls)
        else:
            logger.warning("Validation failed: %s", errors)

        return report

    @staticmethod
    def _check_chaining(
        tool_calls: list[dict[str, Any]],
        tool_outputs: list[dict[str, Any]],
    ) -> bool:
        """Return True if any later step reuses a value (arg or output) from a prior step."""
        prior_values: set[str] = set()

        for step_idx, tc in enumerate(tool_calls):
            current_args = tc.get("arguments", {})

            if step_idx > 0:
                for val in current_args.values():
                    str_val = str(val).strip()
                    if str_val and len(str_val) > 1 and str_val in prior_values:
                        return True

            for val in current_args.values():
                str_val = str(val).strip()
                if str_val and len(str_val) > 1:
                    prior_values.add(str_val)

            if step_idx < len(tool_outputs):
                for val in tool_outputs[step_idx].values():
                    if isinstance(val, (str, int, float)) and not isinstance(val, bool):
                        str_val = str(val).strip()
                        if str_val and len(str_val) > 1:
                            prior_values.add(str_val)

        return False
