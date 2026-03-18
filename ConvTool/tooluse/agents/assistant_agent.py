"""AssistantAgent: produces tool calls and integrates results.

Responsible for:
- Asking clarification questions when required parameters are missing.
- Constructing tool call arguments (with memory grounding).
- Integrating tool outputs into natural language responses.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any

from tooluse.agents.params import generate_param_value
from tooluse.agents.planner_agent import StepPlan
from tooluse.execution.engine import ExecutionEngine, ExecutionResult, SessionState
from tooluse.memory.store import MemoryStore
from tooluse.registry.models import Endpoint, ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    endpoint_id: str
    arguments: dict[str, Any]


@dataclass
class StepResult:
    tool_call: ToolCall
    execution_result: ExecutionResult
    assistant_message: str
    clarification_question: str | None = None
    memory_grounded: bool = False


class AssistantAgent:
    """Generates assistant-side messages, tool calls, and result summaries."""

    def __init__(
        self,
        registry: ToolRegistry,
        engine: ExecutionEngine,
        memory: MemoryStore | None = None,
    ) -> None:
        self._registry = registry
        self._engine = engine
        self._memory = memory

    def ask_clarification(
        self,
        step_plan: StepPlan,
        rng: random.Random,
    ) -> str:
        """Generate a clarification question for missing parameters.

        Uses the parameter's description as the primary label when the raw
        parameter name is cryptic (short, contains double-underscore ORM
        suffixes, or is otherwise not human-readable).  This prevents
        questions like "I need the sname" or "I need the protein in grams  lt".
        """
        if not step_plan.missing_params:
            return ""

        param_name = step_plan.missing_params[0]
        ep = self._registry.get_endpoint(step_plan.endpoint_id)

        readable_name = self._readable_param_label(param_name, ep)

        templates = [
            f"I'd be happy to help. Could you please provide the {readable_name}?",
            f"Before I proceed, I need the {readable_name}. What should it be?",
            f"To continue, I'll need to know the {readable_name}. Can you share that?",
        ]
        question = rng.choice(templates)

        if ep:
            param_schema = next(
                (p for p in ep.parameters if p.name == param_name), None
            )
            if (
                param_schema
                and param_schema.description
                and readable_name.lower() not in param_schema.description.lower()[:60]
            ):
                question += f" ({param_schema.description})"

        return question

    @staticmethod
    def _readable_param_label(param_name: str, endpoint: "Endpoint | None") -> str:
        """Human-readable label for a param: use schema description for cryptic names."""
        is_cryptic = len(param_name) <= 5 or "__" in param_name

        if is_cryptic and endpoint is not None:
            schema = next(
                (p for p in endpoint.parameters if p.name == param_name), None
            )
            if schema and schema.description:
                return schema.description.rstrip(".?! ").lower()

        return param_name.replace("_", " ")

    def execute_step(
        self,
        step_plan: StepPlan,
        step_index: int,
        session: SessionState,
        conversation_id: str,
        provided_args: dict[str, Any],
        rng: random.Random,
    ) -> StepResult:
        """Execute a single tool call step with memory grounding."""
        ep = self._registry.get_endpoint(step_plan.endpoint_id)
        if ep is None:
            raise ValueError(f"Unknown endpoint: {step_plan.endpoint_id}")

        arguments = self._build_arguments(
            ep, step_index, session, conversation_id, provided_args, rng
        )

        memory_grounded = False
        if step_index > 0 and self._memory is not None:
            memory_grounded = self._ground_from_memory(
                ep, arguments, session, conversation_id
            )

        tool_call = ToolCall(endpoint_id=step_plan.endpoint_id, arguments=arguments)
        result = self._engine.execute(step_plan.endpoint_id, arguments, session)

        if self._memory is not None:
            self._memory.add(
                content=json.dumps(result.output),
                scope="session",
                metadata={
                    "conversation_id": conversation_id,
                    "step": step_index,
                    "endpoint": step_plan.endpoint_id,
                },
            )

        assistant_msg = self._summarize_result(ep, result, rng, arguments=arguments)

        return StepResult(
            tool_call=tool_call,
            execution_result=result,
            assistant_message=assistant_msg,
            memory_grounded=memory_grounded,
        )

    def _build_arguments(
        self,
        endpoint: Endpoint,
        step_index: int,
        session: SessionState,
        conversation_id: str,
        provided_args: dict[str, Any],
        rng: random.Random,
    ) -> dict[str, Any]:
        """Construct arguments from provided values, session state, and defaults."""
        args: dict[str, Any] = dict(provided_args)

        for param in endpoint.required_parameters:
            if param.name not in args:
                session_val = session.get(param.name)
                if session_val is not None:
                    args[param.name] = session_val

        for param in endpoint.required_parameters:
            if param.name not in args:
                args[param.name] = generate_param_value(
                    param.name, param.type, rng, endpoint_id=endpoint.id
                )

        for param in endpoint.optional_parameters:
            if param.name not in args and param.default is not None:
                args[param.name] = param.default

        return args

    def _ground_from_memory(
        self,
        endpoint: Endpoint,
        arguments: dict[str, Any],
        session: SessionState,
        conversation_id: str,
    ) -> bool:
        """Attempt to ground arguments in session memory.

        Returns True if at least one memory entry was retrieved.
        """
        query = f"Previous tool outputs for {endpoint.name}"
        results = self._memory.search(query, scope="session", top_k=3)

        if results:
            logger.debug(
                "Memory grounding: %d entries found for %s",
                len(results),
                endpoint.name,
            )
            for entry in results:
                text = entry.get("memory", entry.get("text", ""))
                if text:
                    try:
                        data = json.loads(text)
                        if isinstance(data, dict):
                            for param in endpoint.required_parameters:
                                if param.name not in arguments and param.name in data:
                                    arguments[param.name] = data[param.name]
                    except (json.JSONDecodeError, TypeError):
                        pass
            return True
        return False

    def _summarize_result(
        self,
        endpoint: Endpoint,
        result: ExecutionResult,
        rng: random.Random,
        arguments: dict[str, Any] | None = None,
    ) -> str:
        """Produce a natural-language summary grounded in actual argument values."""
        output = result.output
        args = arguments or {}

        if "error" in output:
            return f"I encountered an issue: {output['error']}."

        tool_name = endpoint.tool_name.replace("_", " ").title()

        # Build a subject hint from the arguments so the summary references
        # what was actually queried (GOOGL, Paris, USD→EUR, etc.).
        subject = _extract_subject(args)

        summary_parts: list[str] = []
        list_reported = False
        for key, value in output.items():
            if isinstance(value, list) and len(value) > 0 and not list_reported:
                list_reported = True
                count = len(value)
                if isinstance(value[0], dict):
                    first = value[0]
                    label = first.get("name", first.get("title", first.get("symbol", "")))
                    if label:
                        summary_parts.append(f"I found {count} option(s), including {label}")
                    else:
                        summary_parts.append(f"I found {count} result(s)")
                else:
                    summary_parts.append(f"I found {count} result(s)")
            elif isinstance(value, dict):
                continue
            elif key == "status" and value not in (None, ""):
                summary_parts.append(f"status: {value}")
            elif key == "booking_reference":
                summary_parts.append(f"your booking reference is {value}")
            elif key == "reservation_id":
                summary_parts.append(f"your reservation ID is {value}")
            elif key == "temperature":
                summary_parts.append(f"the temperature is {value}°")
            elif key == "conditions":
                summary_parts.append(f"conditions are {value}")
            elif key == "humidity":
                summary_parts.append(f"humidity is {value}%")
            elif key == "distance_km":
                summary_parts.append(f"the distance is {value} km")
            elif key == "converted_amount":
                summary_parts.append(f"the converted amount is {value}")
            elif key == "exchange_rate":
                summary_parts.append(f"the rate is {value}")
            elif key == "rating":
                summary_parts.append(f"rated {value}/5")
            elif key == "price_per_night":
                summary_parts.append(f"from ${value}/night")

        if not summary_parts:
            if subject:
                return rng.choice([
                    f"I've retrieved {subject} data from {tool_name}.",
                    f"I looked up {subject} using {tool_name} — all good.",
                    f"{tool_name} returned the details for {subject}.",
                ])
            return rng.choice([
                f"I've retrieved the data from {tool_name}.",
                f"Done — {tool_name} responded successfully.",
                f"I completed the request using {tool_name}.",
            ])

        if len(summary_parts) == 1 and summary_parts[0].startswith("status:"):
            status_val = summary_parts[0].split(": ", 1)[-1].lower()
            if status_val in ("confirmed", "success", "ok", "200", "done"):
                if subject:
                    return rng.choice([
                        f"I retrieved the {subject} data from {tool_name} successfully.",
                        f"Using {tool_name}, I fetched the information for {subject}.",
                        f"{tool_name} confirmed the request for {subject}.",
                    ])
                return rng.choice([
                    f"I've retrieved the data from {tool_name} successfully.",
                    f"Using {tool_name}, the request was confirmed.",
                    f"{tool_name} processed the request successfully.",
                ])

        body = ". ".join(p.capitalize() for p in summary_parts)
        subject_clause = f" for {subject}" if subject else ""

        intro_phrases = [
            f"Here's what I found{subject_clause} using {tool_name}: {body}.",
            f"I've checked {tool_name}{subject_clause}. {body}.",
            f"Using {tool_name}: {body}.",
        ]
        return rng.choice(intro_phrases)


def _extract_subject(args: dict[str, Any]) -> str:
    """Pull a human-readable subject from the tool call arguments.

    Priority order reflects what's most useful to mention in a summary.
    """
    for key in ("symbol", "ticker", "stock_symbol"):
        if key in args:
            return str(args[key])
    if "from_currency" in args and "to_currency" in args:
        return f"{args['from_currency']} to {args['to_currency']}"
    for key in ("location", "city", "address", "origin"):
        if key in args:
            return str(args[key])
    for key in ("artist", "album", "genre", "ingredient", "cuisine"):
        if key in args:
            return str(args[key])
    for key in ("query", "q", "keyword", "term", "search"):
        if key in args:
            return f'"{args[key]}"'
    for key in ("icao", "iata", "airport_code"):
        if key in args:
            return str(args[key])
    return ""
