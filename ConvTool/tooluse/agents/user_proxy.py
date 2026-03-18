"""UserProxyAgent: generates user-side messages in the conversation.

Produces the initial user request and follow-up messages including
answers to clarification questions.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from tooluse.agents.params import _API_INTERNAL_PARAMS, _is_api_internal, generate_param_value
from tooluse.agents.planner_agent import ConversationPlan
from tooluse.registry.models import ToolRegistry

logger = logging.getLogger(__name__)

_INTENT_TEMPLATES: dict[str, list[str]] = {
    "weather": [
        "I'm {scenario} and need to check the weather in {location}.",
        "Can you tell me what the weather is like in {location}? I'm {scenario}.",
        "What's the weather forecast for {location}? I'm {scenario}.",
    ],
    "maps": [
        "I need help {scenario}. Can you look up {address}?",
        "I'm {scenario}. Could you find directions for me?",
        "Help me with {scenario}. I need to search for places nearby.",
    ],
    "travel": [
        "I'm {scenario}. Can you find hotels in {city}?",
        "I need to book a hotel in {city}. I'm {scenario}.",
        "Help me find accommodation in {city} for my trip.",
    ],
    "dining": [
        "I'm {scenario}. Can you find restaurants in {city}?",
        "I'm looking for a place to eat in {city}. I'm {scenario}.",
        "Help me find a good restaurant. I'm {scenario}.",
    ],
    "finance": [
        "I'm {scenario}. Can you pull up the relevant financial data?",
        "I need some market information. I'm {scenario}.",
        "Can you help me look up financial data? I'm {scenario}.",
        "I need to check some figures. I'm {scenario}.",
        "I'm {scenario}. Help me get the data I need.",
    ],
    "productivity": [
        "I need help {scenario}. Can you check my calendar?",
        "I'm {scenario}. Help me set up an event.",
        "I want to organize something. I'm {scenario}.",
    ],
}

_DEFAULT_TEMPLATES = [
    "I need help with something. I'm {scenario}.",
    "Can you assist me? I'm {scenario}.",
]


class UserProxyAgent:
    """Generates user messages based on the conversation plan."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def generate_initial_message(
        self,
        plan: ConversationPlan,
        rng: random.Random,
    ) -> str:
        """Produce the opening user message based on the plan."""
        templates = _INTENT_TEMPLATES.get(plan.domain, _DEFAULT_TEMPLATES)
        template = rng.choice(templates)

        fill = {
            "scenario": plan.scenario,
            **{k: str(v) for k, v in plan.initial_params.items()},
        }

        # Provide sensible defaults for any unfilled template vars
        for key in ("location", "address", "city"):
            if key not in fill:
                fill[key] = plan.initial_params.get(
                    "location",
                    plan.initial_params.get("city", "the area"),
                )

        try:
            message = template.format_map(_SafeDict(fill))
        except Exception:
            message = f"I need help {plan.scenario}."

        # Append provided parameter details naturally
        param_details = self._format_params(plan)
        if param_details:
            message += " " + param_details

        logger.debug("UserProxyAgent initial: %s", message)
        return message

    def generate_clarification_response(
        self,
        question: str,
        missing_param: str,
        rng: random.Random,
        param_type: str = "STRING",
    ) -> tuple[str, Any]:
        """Answer a clarification question with a realistic, type-correct value."""
        value = generate_param_value(missing_param, param_type, rng)
        response_templates = [
            f"It's {value}.",
            f"Sure, {value}.",
            f"I'd like {value}.",
            f"{value}, please.",
        ]
        response = rng.choice(response_templates)
        return response, value

    def generate_followup(
        self,
        step_index: int,
        rng: random.Random,
        domain: str = "",
        last_assistant_msg: str = "",
        conversation_context: dict[str, Any] | None = None,
    ) -> str:
        """Generate a contextual follow-up message that reacts to the last result.

        Tries to reference the subject (symbol, city, etc.) when available so
        the conversation reads as coherent rather than a series of disconnected
        one-liners.
        """
        ctx = conversation_context or {}

        # Extract a subject the user can naturally refer to.
        subject = ""
        for key in ("symbol", "ticker", "location", "city", "artist", "ingredient"):
            if key in ctx:
                subject = str(ctx[key])
                break

        # ── Domain-specific follow-ups with optional subject interpolation ──
        s = f" for {subject}" if subject else ""
        s_in = f" in {subject}" if subject else ""
        s_to = f" to {subject}" if subject else ""

        domain_pools: dict[str, list[str]] = {
            "finance": [
                f"Got it. Can you pull the historical trend{s} as well?",
                f"Thanks. What does the 52-week range look like{s}?",
                "Good. Now can you check the trading volume?",
                f"Perfect. Can you also fetch the daily OHLC data{s}?",
                "That's helpful. Can you also show the moving average?",
                f"Nice. What's the market cap{s}?",
                "OK. Can you cross-check that with another data source?",
                f"Understood. What's the volatility index like{s}?",
                "Thanks. Now show me the year-to-date performance.",
                f"Great. Can you also get the dividend yield{s}?",
            ],
            "weather": [
                "Thanks. Is there any rain expected this weekend?",
                "Good. What about the 7-day forecast?",
                f"Got it. What are the UV levels like{s_in}?",
                "OK, does the wind look strong enough to affect travel?",
                "Perfect. What's the humidity like?",
                "Thanks. Is there a storm warning in effect?",
                f"Understood. What's the air quality index{s_in}?",
                "Good. What temperature should I expect in the evening?",
                "Thanks. Any chance of snow over the next few days?",
                "Perfect. Would you say it's safe to fly in these conditions?",
            ],
            "travel": [
                f"Great. Are there rooms available{s_in} for those dates?",
                "Thanks. Can you also pull up the guest reviews?",
                f"Good. What flights are going{s_to}?",
                "Perfect. What's the cancellation policy like?",
                "Thanks. Can you filter for places rated above 4 stars?",
                "Understood. Are there any direct flights or only connections?",
                "OK. How much would I be looking at per night on average?",
                f"Nice. Any availability for next weekend{s_in}?",
                "Good. Can you check if breakfast is included?",
                "Thanks. What are the check-in and check-out times?",
            ],
            "food": [
                "Thanks. Can you get the nutritional breakdown for that?",
                "Great. Any similar recipes you'd recommend?",
                "Good. What are the main ingredients I'd need?",
                "OK, does it work as a vegetarian or vegan option?",
                "Perfect. Can you also suggest a matching drink?",
                "Thanks. How long does it take to prepare?",
                "Understood. What's the calorie count per serving?",
                "Good. Can you find a gluten-free version of that?",
                "Nice. Any dessert ideas that would pair well with this?",
                "OK. What's a good substitute for the main ingredient?",
            ],
            "location": [
                "Got it. Can you confirm the timezone as well?",
                "Thanks. What's the current local time there?",
                "Good. Can you also get the currency used in that country?",
                f"Perfect. Are there any travel advisories{s_in}?",
                "Thanks. What's the nearest major airport?",
                "Understood. Can you verify the postal code for that address?",
                f"OK. What other cities are nearby{s_in}?",
                "Good. Can you get the full coordinates for that location?",
                f"Nice. What language is spoken{s_in}?",
                "Thanks. What's the dialling code for that country?",
            ],
            "entertainment": [
                "Great. Can you pull up more details on that?",
                "Thanks. What else is trending in that genre?",
                "Good. Any similar titles you'd recommend?",
                "OK, what's the audience rating like?",
                "Perfect. Where can I watch or listen to it?",
                "Thanks. Who are the main cast or artists involved?",
                "Understood. Is there a sequel or follow-up available?",
                "Good. What year was it released?",
                "Nice. Can you find a trailer or preview?",
                "OK. What streaming platform carries it?",
            ],
            "sports": [
                "Got it. What are the current league standings?",
                "Thanks. Who are the top scorers this season?",
                "Good. When is the next match scheduled?",
                "Perfect. Can you check the head-to-head record?",
                "Thanks. Any injury news for the squad?",
                "Understood. What were the stats from the last game?",
                "OK. Who's the leading goalkeeper this season?",
                "Good. What's the average attendance at home games?",
                "Nice. Can you pull up the transfer news?",
                "Thanks. How are they performing in away fixtures?",
            ],
            "productivity": [
                "Great. Can you check if that time slot is free?",
                "Thanks. Can you also block out an hour before for prep?",
                "Good. Who else should be invited to the meeting?",
                "OK. Can you set a reminder for the day before?",
                "Perfect. What's on the agenda for that session?",
                "Thanks. Can you find a recurring slot that works for everyone?",
                "Understood. What timezone should the invite be sent in?",
                "Good. Can you also add the meeting link?",
            ],
        }

        generic = [
            "Thanks, that's useful. What should I do next?",
            "Got it. Can you continue with the next step?",
            "OK, that makes sense. Please go ahead.",
            "Understood. What else can you find out?",
            "Perfect. What's the follow-up on that?",
            "Good to know. Can you dig a bit deeper?",
            "Thanks. Is there anything else I should be aware of?",
            "OK. What would you recommend as the next step?",
        ]

        pool = domain_pools.get(domain, []) + generic
        return rng.choice(pool)

    def _format_params(self, plan: ConversationPlan) -> str:
        """Append user-facing parameter values to the opening message.

        API-internal params (format, outputsize, etc.) are silently skipped —
        users never mention those.
        """
        if not plan.initial_params:
            return ""
        parts: list[str] = []
        for key, value in plan.initial_params.items():
            if _is_api_internal(key):
                continue
            readable_key = key.replace("_", " ")
            parts.append(f"The {readable_key} is {value}")
        if parts:
            return ". ".join(parts) + "."
        return ""


class _SafeDict(dict):
    """Dict that returns the key name for missing format variables."""

    def __missing__(self, key: str) -> str:
        return key
