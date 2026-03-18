"""Offline tool execution engine.

Validates arguments against endpoint schemas, produces deterministic
mock outputs, and maintains session state so that later tool calls
can reference earlier outputs (IDs, booking references, coordinates, etc.).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

from tooluse.registry.models import Endpoint, ParameterSchema, ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    endpoint_id: str
    arguments: dict[str, Any]
    output: dict[str, Any]
    validation: ValidationResult


class SessionState:
    """Tracks argument values and outputs across tool calls so later steps can reuse them."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._step_outputs: list[dict[str, Any]] = []

    def record_output(
        self,
        endpoint_id: str,
        output: dict[str, Any],
        arguments: dict[str, Any] | None = None,
    ) -> None:
        """Store both input arguments and output fields; outputs take precedence on collision."""
        self._step_outputs.append({"endpoint_id": endpoint_id, **output})
        if arguments:
            for key, value in arguments.items():
                self._store[key] = value
        for key, value in output.items():
            self._store[key] = value

    def get(self, key: str) -> Any | None:
        return self._store.get(key)

    def get_outputs(self) -> list[dict[str, Any]]:
        return list(self._step_outputs)

    @property
    def all_values(self) -> dict[str, Any]:
        return dict(self._store)


_TYPE_VALIDATORS: dict[str, type | tuple[type, ...]] = {
    "STRING": (str,),
    "NUMBER": (int, float),
    "BOOLEAN": (bool,),
    "ARRAY": (list,),
    "OBJECT": (dict,),
    "ENUM": (str,),
}


def _check_type(param: ParameterSchema, value: Any) -> str | None:
    """Return an error string if value doesn't match the param type, else None.

    Lenient: numeric strings are accepted for NUMBER; booleans aren't accepted for STRING.
    """
    expected = param.type.upper()

    if expected == "STRING":
        if isinstance(value, str):
            return None
        if isinstance(value, (int, float, bool)):
            return (
                f"Parameter '{param.name}' expects STRING, "
                f"got {type(value).__name__}: {value!r}"
            )
        return (
            f"Parameter '{param.name}' expects STRING, "
            f"got {type(value).__name__}"
        )

    if expected == "NUMBER":
        if isinstance(value, (int, float)):
            return None
        if isinstance(value, str):
            try:
                float(value)
                return None
            except ValueError:
                pass
        return (
            f"Parameter '{param.name}' expects NUMBER, "
            f"got {type(value).__name__}: {value!r}"
        )

    if expected == "BOOLEAN":
        if isinstance(value, bool):
            return None
        if isinstance(value, str) and value.lower() in ("true", "false", "0", "1"):
            return None
        return (
            f"Parameter '{param.name}' expects BOOLEAN, "
            f"got {type(value).__name__}: {value!r}"
        )

    allowed = _TYPE_VALIDATORS.get(expected)
    if allowed and not isinstance(value, allowed):
        return (
            f"Parameter '{param.name}' expects {expected}, "
            f"got {type(value).__name__}: {value!r}"
        )

    return None


class ExecutionEngine:
    """Simulates tool execution offline with deterministic mocking."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def execute(
        self,
        endpoint_id: str,
        arguments: dict[str, Any],
        session: SessionState,
    ) -> ExecutionResult:
        endpoint = self._registry.get_endpoint(endpoint_id)
        if endpoint is None:
            return ExecutionResult(
                endpoint_id=endpoint_id,
                arguments=arguments,
                output={"error": f"Unknown endpoint: {endpoint_id}"},
                validation=ValidationResult(valid=False, errors=["Unknown endpoint"]),
            )

        enriched_args = self._enrich_from_session(endpoint, arguments, session)
        validation = self._validate(endpoint, enriched_args)
        output = self._mock_response(endpoint, enriched_args)
        session.record_output(endpoint_id, output, arguments=enriched_args)

        return ExecutionResult(
            endpoint_id=endpoint_id,
            arguments=enriched_args,
            output=output,
            validation=validation,
        )

    def _validate(
        self, endpoint: Endpoint, arguments: dict[str, Any]
    ) -> ValidationResult:
        """Check presence of required params, type correctness, and enum constraints."""
        errors: list[str] = []

        for param in endpoint.required_parameters:
            if param.name not in arguments:
                errors.append(f"Missing required parameter: {param.name}")

        all_params = {p.name: p for p in endpoint.parameters}
        for arg_name, arg_value in arguments.items():
            param = all_params.get(arg_name)
            if param is None:
                continue

            type_error = _check_type(param, arg_value)
            if type_error:
                errors.append(type_error)

            if param.enum and arg_value not in param.enum:
                errors.append(
                    f"Parameter '{param.name}' value '{arg_value}' "
                    f"not in allowed values: {param.enum}"
                )

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def _enrich_from_session(
        self,
        endpoint: Endpoint,
        arguments: dict[str, Any],
        session: SessionState,
    ) -> dict[str, Any]:
        """Fill missing required arguments from session state when possible."""
        enriched = dict(arguments)
        for param in endpoint.required_parameters:
            if param.name not in enriched:
                session_value = session.get(param.name)
                if session_value is not None:
                    enriched[param.name] = session_value
        return enriched

    def _mock_response(
        self, endpoint: Endpoint, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate a deterministic mock response based on the endpoint schema."""
        output: dict[str, Any] = {}
        for rf in endpoint.response_fields:
            output[rf.name] = self._mock_field(rf.name, rf.type, endpoint, arguments)
        return output

    def _mock_field(
        self,
        field_name: str,
        field_type: str,
        endpoint: Endpoint,
        arguments: dict[str, Any],
    ) -> Any:
        """Deterministic mock value keyed by endpoint ID + arguments hash."""
        stable_key = f"{endpoint.id}:{sorted(arguments.items())}"
        digest = hashlib.md5(stable_key.encode()).hexdigest()[:8]

        if field_name in arguments:
            return arguments[field_name]

        generators: dict[str, Any] = {
            "hotel_id": f"hotel_{digest}",
            "restaurant_id": f"rest_{digest}",
            "event_id": f"evt_{digest}",
            "reservation_id": f"rsv_{digest}",
            "booking_reference": f"BK-{digest.upper()}",
            "latitude": 40.7128 + int(digest[:4], 16) / 100000,
            "longitude": -74.0060 + int(digest[4:8], 16) / 100000,
            "temperature": 18 + int(digest[:2], 16) % 20,
            "humidity": 40 + int(digest[:2], 16) % 40,
            "wind_speed": 5 + int(digest[:2], 16) % 25,
            "conditions": _WEATHER_CONDITIONS[int(digest[:2], 16) % len(_WEATHER_CONDITIONS)],
            "rating": round(3.0 + (int(digest[:2], 16) % 20) / 10, 1),
            "price_per_night": 80 + int(digest[:2], 16) % 300,
            "total_price": 200 + int(digest[:2], 16) % 800,
            "distance_km": round(1.0 + int(digest[:2], 16) % 50, 1),
            "duration_minutes": 10 + int(digest[:2], 16) % 90,
            "exchange_rate": round(0.5 + (int(digest[:4], 16) % 200) / 100, 4),
            "converted_amount": round(
                float(arguments.get("amount", 100)) * (0.5 + (int(digest[:4], 16) % 200) / 100), 2
            ),
            "status": "confirmed",
            "confirmed_time": arguments.get("time", "12:00"),
            "datetime": f"{arguments.get('date', '2025-01-15')}T{arguments.get('time', '12:00')}:00",
            "formatted_address": f"{arguments.get('address', 'Main Street 1')}, City",
            "address": f"{int(digest[:3], 16) % 999} Oak Avenue",
            "city": arguments.get("city", "New York"),
            "country": "United States",
            "count": 3,
        }

        if field_name in generators:
            return generators[field_name]

        if field_type == "NUMBER":
            return int(digest[:4], 16) % 100
        if field_type == "ARRAY":
            return self._mock_array(field_name, endpoint, arguments, digest)
        if field_type == "OBJECT":
            return self._mock_object(field_name, arguments, digest)
        return f"{field_name}_{digest}"

    def _mock_array(
        self,
        field_name: str,
        endpoint: Endpoint,
        arguments: dict[str, Any],
        digest: str,
    ) -> list[dict[str, Any]]:
        count = 3
        items: list[dict[str, Any]] = []
        city = arguments.get("city", arguments.get("location", "New York"))

        if "hotel" in field_name:
            for i in range(count):
                items.append({
                    "hotel_id": f"hotel_{digest}_{i}",
                    "name": f"{_HOTEL_ADJECTIVES[i % len(_HOTEL_ADJECTIVES)]} {city} Hotel",
                    "rating": round(3.5 + i * 0.3, 1),
                    "price_per_night": 80 + i * 50,
                })
        elif "restaurant" in field_name:
            for i in range(count):
                items.append({
                    "restaurant_id": f"rest_{digest}_{i}",
                    "name": f"{_RESTAURANT_NAMES[i % len(_RESTAURANT_NAMES)]}",
                    "cuisine": arguments.get("cuisine", "italian"),
                    "rating": round(3.8 + i * 0.2, 1),
                })
        elif "event" in field_name:
            for i in range(count):
                items.append({
                    "event_id": f"evt_{digest}_{i}",
                    "title": f"Event {i + 1}",
                    "date": arguments.get("date", "2025-01-15"),
                    "time": f"{9 + i * 3}:00",
                })
        elif "forecast" in field_name:
            for i in range(min(count, int(arguments.get("days", 3)))):
                items.append({
                    "day": i + 1,
                    "high": 15 + int(digest[i:i+2], 16) % 15,
                    "low": 5 + int(digest[i:i+2], 16) % 10,
                    "conditions": _WEATHER_CONDITIONS[(int(digest[:2], 16) + i) % len(_WEATHER_CONDITIONS)],
                })
        elif "place" in field_name:
            category = arguments.get("category", "restaurant")
            for i in range(count):
                items.append({
                    "name": f"{_HOTEL_ADJECTIVES[i % len(_HOTEL_ADJECTIVES)]} {category.title()}",
                    "distance_km": round(0.5 + i * 1.2, 1),
                    "rating": round(3.5 + i * 0.3, 1),
                })
        elif "step" in field_name:
            items = [
                {"instruction": "Head north on Main St", "distance_km": 0.5},
                {"instruction": "Turn right onto Oak Ave", "distance_km": 1.2},
                {"instruction": "Arrive at destination", "distance_km": 0.0},
            ]
        elif "menu" in field_name or "highlight" in field_name:
            items = [
                {"dish": "Margherita Pizza", "price": 12.50},
                {"dish": "Truffle Pasta", "price": 18.00},
                {"dish": "Tiramisu", "price": 9.00},
            ]
        elif "amenities" in field_name or "amenity" in field_name:
            items = ["WiFi", "Pool", "Spa", "Parking"]
            return items
        elif "rate" in field_name:
            return {"EUR": 0.92, "GBP": 0.79, "JPY": 149.50}
        else:
            for i in range(count):
                items.append({"id": f"item_{digest}_{i}", "value": f"value_{i}"})

        return items

    def _mock_object(
        self, field_name: str, arguments: dict[str, Any], digest: str
    ) -> dict[str, Any]:
        if "rate" in field_name:
            return {"EUR": 0.92, "GBP": 0.79, "JPY": 149.50, "CHF": 0.88}
        return {"key": f"value_{digest}"}


_WEATHER_CONDITIONS = ["sunny", "partly cloudy", "cloudy", "rainy", "overcast"]
_HOTEL_ADJECTIVES = ["Grand", "Royal", "Sunset", "Harbor", "Central"]
_RESTAURANT_NAMES = ["Trattoria Roma", "Le Petit Bistro", "Sakura Garden", "Casa Del Sol", "The Golden Spoon"]
