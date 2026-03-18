"""Wires together all agents to generate a single conversation record."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any

from tooluse.agents.assistant_agent import AssistantAgent, StepResult
from tooluse.agents.planner_agent import (
    PlannerAgent,
    ConversationPlan,
    _API_INTERNAL_PARAMS,
)
from tooluse.agents.sampler_agent import SamplerAgent
from tooluse.agents.user_proxy import UserProxyAgent
from tooluse.agents.validator_agent import ValidatorAgent, ValidationReport
from tooluse.execution.engine import ExecutionEngine, SessionState
from tooluse.graph.builder import GraphBuilder
from tooluse.memory.store import MemoryStore
from tooluse.registry.models import ToolRegistry
from tooluse.sampler.base import PatternType

# Params that persist across steps so the same ticker/city/date is reused throughout.
_STICKY_PARAMS: frozenset[str] = frozenset({
    "symbol", "ticker", "stock_symbol", "location", "city", "address",
    "origin", "destination", "country", "country_code", "from_currency",
    "to_currency", "base_currency", "base", "target", "currency",
    "check_in", "check_out", "date", "start_date", "end_date",
    "ingredient", "cuisine", "artist", "album", "genre",
    "query", "keyword", "term", "q", "search",
    "name", "guest_name", "email",
    "latitude", "longitude", "ip", "icao", "iata",
})

logger = logging.getLogger(__name__)

MAX_RETRIES = 5


@dataclass
class GeneratedConversation:
    messages: list[dict[str, str]]
    tool_calls: list[dict[str, Any]]
    tool_outputs: list[dict[str, Any]]
    metadata: dict[str, Any]
    validation: ValidationReport


class ConversationOrchestrator:
    """Coordinates the multi-agent conversation generation pipeline."""

    def __init__(
        self,
        registry: ToolRegistry,
        graph: GraphBuilder,
        memory: MemoryStore | None = None,
        corpus_memory_enabled: bool = True,
    ) -> None:
        self._registry = registry
        self._graph = graph
        self._memory = memory
        self._corpus_memory_enabled = corpus_memory_enabled

        self._sampler_agent = SamplerAgent(graph)
        self._planner_agent = PlannerAgent(
            registry, memory, corpus_memory_enabled
        )
        self._user_proxy = UserProxyAgent(registry)
        self._engine = ExecutionEngine(registry)
        self._assistant_agent = AssistantAgent(registry, self._engine, memory)
        self._validator = ValidatorAgent(registry)

    def generate(self, seed: int, conversation_index: int) -> GeneratedConversation:
        """Generate a conversation, retrying up to MAX_RETRIES times if validation fails."""
        last_result: GeneratedConversation | None = None

        for attempt in range(MAX_RETRIES):
            attempt_seed = seed + conversation_index * 1000 + attempt
            rng = random.Random(attempt_seed)
            conversation_id = f"conv_{seed}_{conversation_index}_{attempt}"

            try:
                last_result = self._generate_one(rng, attempt_seed, conversation_id)
                if last_result.validation.valid:
                    return last_result
                logger.info(
                    "Attempt %d for conversation %d failed validation: %s",
                    attempt, conversation_index, last_result.validation.errors,
                )
            except Exception:
                logger.exception(
                    "Attempt %d for conversation %d raised an exception.",
                    attempt, conversation_index,
                )

        logger.warning(
            "All retries exhausted for conversation %d, returning last attempt.",
            conversation_index,
        )
        if last_result is None:
            raise RuntimeError(
                f"No result produced for conversation {conversation_index} "
                "after all retries."
            )
        return last_result

    def _generate_one(
        self,
        rng: random.Random,
        seed: int,
        conversation_id: str,
    ) -> GeneratedConversation:
        session = SessionState()
        messages: list[dict[str, str]] = []
        tool_calls_list: list[dict[str, Any]] = []
        tool_outputs_list: list[dict[str, Any]] = []
        step_results: list[StepResult] = []
        memory_grounded_count = 0
        non_first_steps = 0

        # Query corpus memory so PatternSampler can boost underrepresented patterns.
        pattern_counts: dict[str, int] | None = None
        if self._corpus_memory_enabled and self._memory is not None:
            summaries = self._memory.search(
                "conversation pattern domain tools", scope="corpus", top_k=20
            )
            if summaries:
                pattern_counts = {}
                for entry in summaries:
                    text = entry.get("memory", "")
                    for pt in PatternType:
                        if f"Pattern: {pt.value}" in text:
                            pattern_counts[pt.value] = (
                                pattern_counts.get(pt.value, 0) + 1
                            )
                            break

        chain = self._sampler_agent.propose(rng, seed, pattern_counts=pattern_counts)
        plan = self._planner_agent.plan(chain, rng, conversation_id)

        initial_msg = self._user_proxy.generate_initial_message(plan, rng)
        messages.append({"role": "user", "content": initial_msg})

        # Accumulates sticky params (symbol, city, …) so later steps reuse the same values.
        conversation_context: dict[str, Any] = dict(plan.initial_params)

        for step_idx, step_plan in enumerate(plan.steps):
            current_args = {
                k: v for k, v in conversation_context.items()
                if k in _STICKY_PARAMS
            }

            if step_plan.needs_clarification:
                question = self._assistant_agent.ask_clarification(step_plan, rng)
                if question:
                    messages.append({"role": "assistant", "content": question})

                    ep = self._registry.get_endpoint(step_plan.endpoint_id)
                    param_types = {}
                    if ep:
                        param_types = {p.name: p.type for p in ep.parameters}

                    for param_name in step_plan.missing_params:
                        answer, value = self._user_proxy.generate_clarification_response(
                            question, param_name, rng,
                            param_type=param_types.get(param_name, "STRING"),
                        )
                        messages.append({"role": "user", "content": answer})
                        current_args[param_name] = value
                        conversation_context[param_name] = value

            step_result = self._assistant_agent.execute_step(
                step_plan=step_plan,
                step_index=step_idx,
                session=session,
                conversation_id=conversation_id,
                provided_args=current_args,
                rng=rng,
            )
            step_results.append(step_result)

            for k, v in step_result.tool_call.arguments.items():
                if k in _STICKY_PARAMS:
                    conversation_context[k] = v

            if step_idx > 0:
                non_first_steps += 1
                if step_result.memory_grounded:
                    memory_grounded_count += 1

            tool_calls_list.append({
                "endpoint_id": step_result.tool_call.endpoint_id,
                "arguments": step_result.tool_call.arguments,
            })
            tool_outputs_list.append(step_result.execution_result.output)

            messages.append({
                "role": "assistant",
                "content": step_result.assistant_message,
            })

            if step_idx < len(plan.steps) - 1:
                followup = self._user_proxy.generate_followup(
                    step_idx,
                    rng,
                    domain=plan.domain,
                    last_assistant_msg=step_result.assistant_message,
                    conversation_context=conversation_context,
                )
                messages.append({"role": "user", "content": followup})

        if non_first_steps > 0:
            memory_grounding_rate = memory_grounded_count / non_first_steps
        else:
            memory_grounding_rate = None

        conversation_record = {
            "messages": messages,
            "tool_calls": tool_calls_list,
            "tool_outputs": tool_outputs_list,
        }

        validation = self._validator.validate(conversation_record)

        tool_ids_used = list(set(
            tc["endpoint_id"].split(".")[0] for tc in tool_calls_list
        ))

        num_clarifications = sum(
            1 for sp in plan.steps if sp.needs_clarification
        )

        metadata = {
            "seed": seed,
            "tool_ids_used": sorted(tool_ids_used),
            "num_turns": len(messages),
            "num_clarification_questions": num_clarifications,
            "memory_grounding_rate": memory_grounding_rate,
            "corpus_memory_enabled": self._corpus_memory_enabled,
            "pattern_type": chain.pattern_type.value,
            "domain": plan.domain,
        }

        result = GeneratedConversation(
            messages=messages,
            tool_calls=tool_calls_list,
            tool_outputs=tool_outputs_list,
            metadata=metadata,
            validation=validation,
        )

        if (
            self._corpus_memory_enabled
            and self._memory is not None
            and validation.valid
        ):
            summary = (
                f"Tools: {', '.join(sorted(tool_ids_used))}. "
                f"Domain: {plan.domain}. "
                f"Pattern: {chain.pattern_type.value}. "
                f"Scenario: {plan.scenario}."
            )
            self._memory.add(
                content=summary,
                scope="corpus",
                metadata={
                    "conversation_id": conversation_id,
                    "tools": sorted(tool_ids_used),
                    "pattern_type": chain.pattern_type.value,
                },
            )

        return result
