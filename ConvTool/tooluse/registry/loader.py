"""ToolBench loader and registry serialization.

Handles the conversion from raw ToolBench JSON format into the
normalized internal ToolRegistry representation.  Designed to be
resilient against the many inconsistencies present in real ToolBench
data (missing fields, empty strings where dicts are expected, placeholder
bodies, status-code 111 endpoints, duplicate descriptions, etc.).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from tooluse.registry.models import (
    ConceptTag,
    Endpoint,
    ParameterSchema,
    ResponseField,
    Tool,
    ToolRegistry,
)

logger = logging.getLogger(__name__)

_FALLBACK_RESPONSE_FIELDS = [
    ResponseField(name="result", type="STRING"),
    ResponseField(name="status", type="STRING"),
]

_TYPE_NORMALIZATION: dict[str, str] = {
    "string": "STRING",
    "str": "STRING",
    "text": "STRING",
    "number": "NUMBER",
    "num": "NUMBER",
    "int": "NUMBER",
    "integer": "NUMBER",
    "float": "NUMBER",
    "double": "NUMBER",
    "boolean": "BOOLEAN",
    "bool": "BOOLEAN",
    "enum": "ENUM",
    "array": "ARRAY",
    "list": "ARRAY",
    "object": "OBJECT",
    "dict": "OBJECT",
    "json": "OBJECT",
}


def _normalize_type(raw_type: Any) -> str:
    """Normalize a parameter or response-field type to a canonical string."""
    if not raw_type or not isinstance(raw_type, str):
        return "STRING"
    return _TYPE_NORMALIZATION.get(raw_type.strip().lower(), raw_type.upper())


def _parse_parameter(raw: dict | Any, *, required: bool) -> ParameterSchema | None:
    """Parse a single parameter dict, tolerating missing/malformed data."""
    if not isinstance(raw, dict):
        logger.debug("Skipping non-dict parameter entry: %s", type(raw))
        return None

    name = raw.get("name")
    if not name or not isinstance(name, str):
        name = raw.get("param_name", "unknown")
    name = str(name).strip() or "unknown"

    default = raw.get("default")
    if isinstance(default, str) and default.strip() == "":
        default = None

    enum_val = raw.get("enum")
    if isinstance(enum_val, list) and all(isinstance(e, str) for e in enum_val):
        pass
    else:
        enum_val = None

    return ParameterSchema(
        name=name,
        type=_normalize_type(raw.get("type")),
        description=str(raw.get("description", "") or "").strip(),
        required=required,
        default=default,
        enum=enum_val,
    )


def _extract_response_fields_from_body(body: Any) -> list[ResponseField]:
    """Extract response field names from a ToolBench ``body`` value.

    The ``body`` field in ToolBench is highly inconsistent:
    - Empty string ``""`` or ``None`` → no info
    - Placeholder dict ``{"key1": "value", "key2": "value"}`` → no real info
    - Actual example response dict → extract top-level keys as fields
    - Nested dicts → flatten one level
    """
    if not body or not isinstance(body, dict):
        return []

    placeholder_keys = {"key1", "key2"}
    if set(body.keys()) == placeholder_keys:
        return []

    fields: list[ResponseField] = []
    for key, val in body.items():
        if isinstance(val, dict):
            inferred = "OBJECT"
        elif isinstance(val, list):
            inferred = "ARRAY"
        elif isinstance(val, (int, float)):
            inferred = "NUMBER"
        elif isinstance(val, bool):
            inferred = "BOOLEAN"
        else:
            inferred = "STRING"
        fields.append(ResponseField(name=str(key), type=inferred))

    return fields


def _extract_response_fields_from_schema(schema: Any) -> list[ResponseField]:
    """Extract response fields from a JSON-Schema-style ``schema`` value.

    Many ToolBench entries include a ``schema`` dict that follows a
    subset of JSON Schema (``type``, ``properties``).
    """
    if not schema or not isinstance(schema, dict):
        return []

    props = schema.get("properties")
    if not isinstance(props, dict):
        return []

    fields: list[ResponseField] = []
    for key, prop_def in props.items():
        if isinstance(prop_def, dict):
            ftype = _normalize_type(prop_def.get("type"))
        else:
            ftype = "STRING"
        fields.append(ResponseField(name=str(key), type=ftype))

    return fields


def _parse_response_fields(api_entry: dict) -> list[ResponseField]:
    """Best-effort extraction of response fields from a ToolBench API entry.

    Tries ``schema`` first (structured), then ``body`` (example), then
    falls back to generic defaults.
    """
    fields = _extract_response_fields_from_schema(api_entry.get("schema"))
    if fields:
        return fields

    fields = _extract_response_fields_from_body(api_entry.get("body"))
    if fields:
        return fields

    return list(_FALLBACK_RESPONSE_FIELDS)


def _parse_endpoint(tool_name: str, raw: dict) -> Endpoint | None:
    """Parse a single API entry into an Endpoint, handling all known ToolBench quirks."""
    if not isinstance(raw, dict):
        return None

    name = raw.get("name", raw.get("api_name", "unknown"))
    if not isinstance(name, str) or not name.strip():
        name = "unknown"
    name = name.strip()

    endpoint_id = f"{tool_name}.{name}"

    required_params: list[ParameterSchema] = []
    for p in raw.get("required_parameters") or []:
        parsed = _parse_parameter(p, required=True)
        if parsed:
            required_params.append(parsed)

    optional_params: list[ParameterSchema] = []
    for p in raw.get("optional_parameters") or []:
        parsed = _parse_parameter(p, required=False)
        if parsed:
            optional_params.append(parsed)

    response_fields = _parse_response_fields(raw)

    description = raw.get("description", raw.get("api_description", ""))
    if not isinstance(description, str):
        description = ""
    description = description.strip()

    method = raw.get("method", "GET")
    if not isinstance(method, str) or method.strip().upper() not in {
        "GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS",
    }:
        method = "GET"
    else:
        method = method.strip().upper()

    url = raw.get("url", "")
    if not isinstance(url, str):
        url = ""

    return Endpoint(
        id=endpoint_id,
        name=name,
        tool_name=tool_name,
        description=description,
        method=method,
        url=url,
        parameters=required_params + optional_params,
        response_fields=response_fields,
    )


_KEYWORD_CONCEPTS: dict[str, str] = {
    "weather": "weather",
    "forecast": "weather",
    "climate": "weather",
    "temperature": "weather",
    "map": "location",
    "geo": "location",
    "address": "location",
    "postcode": "location",
    "timezone": "location",
    "hotel": "accommodation",
    "booking": "booking",
    "reserv": "booking",
    "restaurant": "dining",
    "recipe": "dining",
    "food": "dining",
    "nutrition": "health",
    "diet": "health",
    "currency": "finance",
    "stock": "finance",
    "crypto": "finance",
    "exchange": "finance",
    "price": "pricing",
    "event": "scheduling",
    "calendar": "scheduling",
    "travel": "travel",
    "flight": "travel",
    "airport": "travel",
    "airline": "travel",
    "direction": "navigation",
    "nearby": "discovery",
    "search": "search",
    "movie": "media",
    "music": "media",
    "game": "gaming",
    "stream": "media",
    "video": "media",
}


def _infer_concepts(tool_name: str, category: str) -> list[ConceptTag]:
    """Derive concept tags from the tool name and category."""
    concepts: set[str] = set()
    normed_category = re.sub(r"[_\-]", " ", category).lower().strip()
    concepts.add(normed_category)

    combined = f"{tool_name} {category}".lower()
    for keyword, concept in _KEYWORD_CONCEPTS.items():
        if keyword in combined:
            concepts.add(concept)

    return [ConceptTag(name=c) for c in sorted(concepts)]


def _sanitize_tool_name(raw_name: str) -> str:
    """Clean up a tool name, removing excess whitespace."""
    if not raw_name or not isinstance(raw_name, str):
        return "unknown_tool"
    return raw_name.strip() or "unknown_tool"


def _parse_tool(raw: dict, category: str) -> Tool | None:
    """Parse a full tool JSON dict into a Tool object.

    ``category`` is supplied externally (from the directory name) since
    ToolBench JSONs do not carry their own category field.
    """
    tool_name = _sanitize_tool_name(
        raw.get("tool_name", raw.get("name", raw.get("title", "unknown_tool")))
    )

    api_list = raw.get("api_list")
    if not isinstance(api_list, list) or not api_list:
        logger.warning("Tool '%s' has no api_list, skipping.", tool_name)
        return None

    endpoints: list[Endpoint] = []
    for api_entry in api_list:
        ep = _parse_endpoint(tool_name, api_entry)
        if ep:
            endpoints.append(ep)

    if not endpoints:
        logger.warning("Tool '%s' yielded zero valid endpoints, skipping.", tool_name)
        return None

    description = raw.get("tool_description", raw.get("description", ""))
    if not isinstance(description, str):
        description = ""

    concepts = _infer_concepts(tool_name, category)

    return Tool(
        name=tool_name,
        description=description.strip(),
        category=category,
        endpoints=endpoints,
        concepts=concepts,
    )


def load_toolbench(data_dir: Path) -> ToolRegistry:
    """Load ToolBench tool definitions from a ``Category/tool.json`` tree.

    Expected layout::

        data_dir/
          Weather/
            open_weather_map.json
            weatherapi_com.json
          Finance/
            alpha_vantage.json
            ...

    Also supports a flat directory of JSON files (legacy / test usage)
    where each file may contain a single tool dict or a list of tools.
    """
    data_dir = Path(data_dir)
    registry = ToolRegistry()

    category_dirs = sorted(
        p for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )

    if category_dirs:
        for cat_dir in category_dirs:
            category = cat_dir.name.replace("_", " ")
            json_files = sorted(cat_dir.glob("*.json"))
            for path in json_files:
                if path.name.startswith("."):
                    continue
                logger.info("Loading %s/%s", category, path.name)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    logger.warning("Skipping malformed JSON %s: %s", path, exc)
                    continue

                if isinstance(data, list):
                    entries = data
                elif isinstance(data, dict):
                    entries = [data]
                else:
                    logger.warning("Unexpected top-level type in %s", path)
                    continue

                for raw in entries:
                    if not isinstance(raw, dict):
                        continue
                    try:
                        tool = _parse_tool(raw, category)
                        if tool:
                            registry.tools.append(tool)
                    except Exception:
                        logger.exception(
                            "Failed to parse tool entry in %s/%s", category, path.name
                        )
    else:
        json_files = sorted(data_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {data_dir}")

        for path in json_files:
            logger.info("Loading ToolBench file: %s", path.name)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            tools_raw = data if isinstance(data, list) else data.get("tools", [data])
            for raw in tools_raw:
                if not isinstance(raw, dict):
                    continue
                try:
                    category = raw.get("category_name", raw.get("category", "General"))
                    tool = _parse_tool(raw, category)
                    if tool:
                        registry.tools.append(tool)
                except Exception:
                    logger.exception("Failed to parse tool entry in %s", path.name)

    total_endpoints = sum(len(t.endpoints) for t in registry.tools)
    logger.info(
        "Registry loaded: %d tools, %d endpoints across %d categories",
        len(registry.tools),
        total_endpoints,
        len({t.category for t in registry.tools}),
    )
    return registry


def save_registry(registry: ToolRegistry, path: Path) -> None:
    """Persist the registry to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry.to_dict(), f, indent=2)
    logger.info("Registry saved to %s", path)


def load_registry(path: Path) -> ToolRegistry:
    """Load a previously saved registry from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ToolRegistry.from_dict(data)
