"""PlannerAgent: creates a conversation plan from a tool chain.

The plan determines the user scenario, which parameters will be
provided upfront vs. through clarification, and the overall
conversation structure.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any

from tooluse.agents.params import _API_INTERNAL_PARAMS, _is_api_internal, generate_param_value
from tooluse.memory.store import MemoryStore
from tooluse.registry.models import Endpoint, ToolRegistry
from tooluse.sampler.base import ToolChain

logger = logging.getLogger(__name__)

_DOMAIN_SCENARIOS: dict[str, list[str]] = {
    "weather": [
        "planning a hiking trip next weekend",
        "deciding whether to pack an umbrella for my flight",
        "checking if the concert venue will have bad weather tonight",
        "planning an outdoor birthday party and need to pick the best day",
    ],
    "maps": [
        "finding directions from the airport to my hotel",
        "looking for coffee shops near the conference center",
        "planning a scenic road trip route",
        "finding parking near the stadium before the game",
    ],
    "travel": [
        "booking a hotel for a business trip to Tokyo",
        "comparing accommodation options for a family holiday",
        "finding a last-minute room for tonight near the airport",
        "planning a two-week European itinerary",
    ],
    "dining": [
        "looking for a sushi restaurant for my team dinner tonight",
        "finding a romantic spot for a date night in the city",
        "planning a birthday dinner and need something upscale",
        "hunting for the best pizza place near the hotel",
    ],
    "finance": [
        "preparing a travel budget for my upcoming trip to Japan",
        "checking if it's a good time to convert USD to EUR",
        "tracking crypto prices before deciding to invest",
        "comparing exchange rates for an international wire transfer",
        "reviewing my portfolio before the quarterly earnings call",
        "doing due diligence on a stock before adding it to my watchlist",
        "comparing financial metrics across a few companies",
        "looking up market data ahead of my investment decision",
    ],
    "productivity": [
        "setting up a meeting with a colleague in a different timezone",
        "blocking off time for a project milestone review",
        "organizing reminders for the week ahead",
        "scheduling a team standup across three time zones",
    ],
    "food": [
        "looking for low-carb dinner ideas for the week",
        "finding cocktail recipes for a house party",
        "planning a keto meal prep for the week",
        "discovering new dessert ideas for a dinner party",
    ],
    "entertainment": [
        "building a playlist for a road trip",
        "finding something new to watch on streaming tonight",
        "discovering songs similar to my favourite album",
        "getting book recommendations for a long-haul flight",
    ],
    "location": [
        "verifying the timezone before scheduling an international call",
        "looking up the exact coordinates of a meeting venue",
        "checking the local time in multiple cities for a global event",
        "confirming a mailing address before sending a parcel",
    ],
    "sports": [
        "checking live scores during the match",
        "looking up the schedule for the upcoming season",
        "finding stats for a player before placing a fantasy bet",
        "comparing team standings before the playoffs",
    ],
}

_DEFAULT_SCENARIOS = [
    "researching options before making a booking",
    "comparing data from multiple sources before a decision",
    "checking a few things before my trip",
    "pulling together information for a report",
    "looking up details I need for a presentation tomorrow",
]


@dataclass
class StepPlan:
    endpoint_id: str
    provided_params: list[str]
    missing_params: list[str]
    needs_clarification: bool


@dataclass
class ConversationPlan:
    scenario: str
    domain: str
    steps: list[StepPlan]
    tool_chain: ToolChain
    initial_params: dict[str, Any] = field(default_factory=dict)


class PlannerAgent:
    """Plans conversations by analyzing tool chains and crafting scenarios."""

    def __init__(
        self,
        registry: ToolRegistry,
        memory: MemoryStore | None = None,
        corpus_memory_enabled: bool = True,
    ) -> None:
        self._registry = registry
        self._memory = memory
        self._corpus_memory_enabled = corpus_memory_enabled

    def plan(
        self,
        tool_chain: ToolChain,
        rng: random.Random,
        conversation_id: str,
    ) -> ConversationPlan:
        domain = self._infer_domain(tool_chain)
        corpus_context = (
            self._get_corpus_context(tool_chain)
            if self._corpus_memory_enabled
            else ""
        )
        scenario = self._pick_scenario(domain, rng, corpus_context)
        step_plans = self._plan_steps(tool_chain, rng)
        initial_params = self._generate_initial_params(step_plans, rng)

        plan = ConversationPlan(
            scenario=scenario,
            domain=domain,
            steps=step_plans,
            tool_chain=tool_chain,
            initial_params=initial_params,
        )
        logger.info(
            "PlannerAgent: domain=%s, scenario='%s', %d steps, %d clarifications.",
            domain,
            scenario,
            len(step_plans),
            sum(1 for s in step_plans if s.needs_clarification),
        )
        return plan

    def _infer_domain(self, chain: ToolChain) -> str:
        categories: list[str] = []
        for step in chain.steps:
            ep = self._registry.get_endpoint(step.endpoint_id)
            if ep:
                tool = self._registry.get_tool(ep.tool_name)
                if tool:
                    categories.append(tool.category.lower())
        if categories:
            return max(set(categories), key=categories.count)
        return "general"

    def _get_corpus_context(self, tool_chain: ToolChain) -> str:
        if self._memory is None:
            return ""
        query = f"tools: {', '.join(tool_chain.tool_names)}"
        results = self._memory.search(query, scope="corpus", top_k=5)
        summaries = [
            entry.get("memory", entry.get("text", ""))
            for entry in results
            if entry.get("memory") or entry.get("text")
        ]
        return "\n".join(summaries)

    def _pick_scenario(
        self, domain: str, rng: random.Random, corpus_context: str
    ) -> str:
        scenarios = list(_DOMAIN_SCENARIOS.get(domain, _DEFAULT_SCENARIOS))
        if corpus_context:
            unused = [s for s in scenarios if s not in corpus_context]
            if unused:
                scenarios = unused
        return rng.choice(scenarios)

    def _plan_steps(
        self, chain: ToolChain, rng: random.Random
    ) -> list[StepPlan]:
        """Decide per-step which params are provided vs. need clarification.

        Clarification can happen at any step, not just the first:
        - Step 0: 40% chance (simulates ambiguous user intent)
        - Later steps: 20% chance when ≥2 required params (simulates
          missing context that session state doesn't cover)

        This ensures multi-turn disambiguation appears throughout
        conversations, not only at the start.
        """
        plans: list[StepPlan] = []
        has_clarification = False

        for i, step in enumerate(chain.steps):
            ep = self._registry.get_endpoint(step.endpoint_id)
            if ep is None:
                continue

            required_names = [p.name for p in ep.required_parameters]
            clarifiable = [
                n for n in required_names if not _is_api_internal(n)
            ]
            missing: list[str] = []
            provided: list[str] = list(required_names)

            if i == 0:
                if len(clarifiable) > 1 and rng.random() < 0.4:
                    missing = [rng.choice(clarifiable)]
                    provided = [n for n in required_names if n not in missing]
                    has_clarification = True
            else:
                if len(clarifiable) >= 2 and rng.random() < 0.20:
                    missing = [rng.choice(clarifiable)]
                    provided = [n for n in required_names if n not in missing]
                    has_clarification = True

            # Guarantee at least one clarification in ~60% of conversations
            if not has_clarification and i == len(chain.steps) - 1:
                for j, earlier in enumerate(plans):
                    earlier_ep = self._registry.get_endpoint(earlier.endpoint_id)
                    clarifiable_earlier = [
                        p.name
                        for p in (earlier_ep.required_parameters if earlier_ep else [])
                        if not _is_api_internal(p.name)
                    ]
                    if (
                        earlier_ep
                        and len(clarifiable_earlier) > 1
                        and rng.random() < 0.5
                    ):
                        param = rng.choice(clarifiable_earlier)
                        plans[j] = StepPlan(
                            endpoint_id=earlier.endpoint_id,
                            provided_params=[
                                n for n in earlier.provided_params if n != param
                            ],
                            missing_params=[param],
                            needs_clarification=True,
                        )
                        has_clarification = True
                        break

            plans.append(StepPlan(
                endpoint_id=step.endpoint_id,
                provided_params=provided,
                missing_params=missing,
                needs_clarification=len(missing) > 0,
            ))
        return plans

    def _generate_initial_params(
        self, steps: list[StepPlan], rng: random.Random
    ) -> dict[str, Any]:
        """Generate realistic parameter values for the first step.

        Only includes user-facing parameters — API internals (format, language,
        outputsize, etc.) are filtered out so they never appear in user messages.
        """
        if not steps:
            return {}
        first_ep = self._registry.get_endpoint(steps[0].endpoint_id)
        if first_ep is None:
            return {}
        return {
            param.name: generate_param_value(
                param.name, param.type, rng, endpoint_id=first_ep.id
            )
            for param in first_ep.parameters
            if param.name in steps[0].provided_params
            and not _is_api_internal(param.name)
        }
