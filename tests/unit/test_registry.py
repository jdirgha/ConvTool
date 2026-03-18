"""Unit tests for the tool registry loader and models.

Tests cover:
- Real ToolBench directory structure (Category/tool.json)
- Real ToolBench field names (body, schema, statuscode, etc.)
- Missing/inconsistent field handling (empty strings, null, status 111)
- Flat-directory fallback (legacy compatibility)
- Registry round-trip serialization
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tooluse.registry.loader import load_toolbench, save_registry, load_registry
from tooluse.registry.models import (
    ConceptTag,
    Endpoint,
    ParameterSchema,
    ResponseField,
    Tool,
    ToolRegistry,
)


TOOLBENCH_WEATHER_TOOL = {
    "tool_name": "Open Weather Map",
    "tool_description": "Open Weather Map API for current conditions.",
    "title": "Open Weather Map",
    "pricing": "FREE",
    "score": {
        "avgServiceLevel": 100,
        "avgLatency": 296,
        "avgSuccessRate": 0,
        "popularityScore": 0.1,
        "__typename": "Score",
    },
    "home_url": "https://rapidapi.com/example/api/open-weather-map/",
    "host": "open-weather-map.p.rapidapi.com",
    "api_list": [
        {
            "name": "current weather data",
            "url": "https://open-weather-map.p.rapidapi.com/weather",
            "description": "Get current weather for a location.",
            "method": "GET",
            "required_parameters": [
                {
                    "name": "q",
                    "type": "STRING",
                    "description": "City name",
                    "default": "London",
                }
            ],
            "optional_parameters": [
                {
                    "name": "units",
                    "type": "STRING",
                    "description": "Units of measurement",
                    "default": "metric",
                }
            ],
            "code": "import requests\nresponse = requests.get(url)\n",
            "statuscode": 200,
            "body": {
                "coord": {"lon": -0.1257, "lat": 51.5085},
                "weather": [{"id": 800, "main": "Clear"}],
                "main": {"temp": 15.5, "humidity": 72},
            },
            "headers": "",
            "schema": "",
        }
    ],
}

TOOLBENCH_FINANCE_TOOL = {
    "tool_name": "Exchange Rate API",
    "tool_description": "Currency exchange rates.",
    "title": "Exchange Rate API",
    "pricing": "FREE",
    "score": None,
    "home_url": "https://rapidapi.com/example/api/exchange-rate/",
    "host": "exchange-rate.p.rapidapi.com",
    "api_list": [
        {
            "name": "convert",
            "url": "https://exchange-rate.p.rapidapi.com/convert",
            "description": "Convert between currencies.",
            "method": "GET",
            "required_parameters": [
                {"name": "from", "type": "STRING", "description": "Source currency"},
                {"name": "to", "type": "STRING", "description": "Target currency"},
                {"name": "amount", "type": "NUMBER", "description": "Amount"},
            ],
            "optional_parameters": [],
            "statuscode": 200,
            "body": "",
            "schema": {
                "type": "object",
                "properties": {
                    "result": {"type": "number"},
                    "rate": {"type": "number"},
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                },
            },
        },
        {
            "name": "list_currencies",
            "url": "https://exchange-rate.p.rapidapi.com/currencies",
            "description": "List available currencies.",
            "method": "GET",
            "required_parameters": [],
            "optional_parameters": [],
            "statuscode": 111,
            "body": "",
            "headers": "",
            "schema": "",
        },
    ],
}

TOOLBENCH_MESSY_TOOL = {
    "tool_name": "Messy API",
    "tool_description": "",
    "title": "Messy",
    "pricing": "FREE",
    "score": None,
    "home_url": "",
    "host": "",
    "api_list": [
        {
            "name": "do_thing",
            "url": "",
            "description": "do_thing",
            "method": "GET",
            "required_parameters": [],
            "optional_parameters": [
                {"name": "phone", "type": "NUMBER", "description": "", "default": "1"}
            ],
            "statuscode": 111,
            "body": "",
            "headers": "",
            "schema": "",
        },
        {
            "name": "placeholder_body",
            "url": "",
            "description": "Has placeholder body.",
            "method": "POST",
            "required_parameters": [],
            "optional_parameters": [],
            "body": {"key1": "value", "key2": "value"},
            "schema": "",
        },
    ],
}


@pytest.fixture
def toolbench_tree(tmp_path: Path) -> Path:
    """Create a ToolBench-style Category/tool.json directory tree."""
    weather_dir = tmp_path / "Weather"
    weather_dir.mkdir()
    with open(weather_dir / "open_weather_map.json", "w") as f:
        json.dump(TOOLBENCH_WEATHER_TOOL, f)

    finance_dir = tmp_path / "Finance"
    finance_dir.mkdir()
    with open(finance_dir / "exchange_rate.json", "w") as f:
        json.dump(TOOLBENCH_FINANCE_TOOL, f)
    with open(finance_dir / "messy_api.json", "w") as f:
        json.dump(TOOLBENCH_MESSY_TOOL, f)

    return tmp_path


@pytest.fixture
def flat_data_dir(tmp_path: Path) -> Path:
    """Create a flat directory with a legacy-style JSON list."""
    data_dir = tmp_path / "flat"
    data_dir.mkdir()
    tools = [
        {
            "tool_name": "flat_tool",
            "tool_description": "Flat dir tool.",
            "category_name": "General",
            "api_list": [
                {
                    "name": "action",
                    "description": "Do an action.",
                    "method": "POST",
                    "required_parameters": [
                        {"name": "input", "type": "STRING", "description": "Input"}
                    ],
                    "optional_parameters": [],
                }
            ],
        }
    ]
    with open(data_dir / "tools.json", "w") as f:
        json.dump(tools, f)
    return data_dir


class TestDirectoryStructureLoading:
    """Tests for loading from real ToolBench Category/tool.json tree."""

    def test_loads_all_tools(self, toolbench_tree: Path) -> None:
        registry = load_toolbench(toolbench_tree)
        assert len(registry.tools) == 3

    def test_tool_names(self, toolbench_tree: Path) -> None:
        registry = load_toolbench(toolbench_tree)
        names = {t.name for t in registry.tools}
        assert "Open Weather Map" in names
        assert "Exchange Rate API" in names
        assert "Messy API" in names

    def test_category_from_directory(self, toolbench_tree: Path) -> None:
        registry = load_toolbench(toolbench_tree)
        weather_tool = registry.get_tool("Open Weather Map")
        assert weather_tool is not None
        assert weather_tool.category == "Weather"

        finance_tool = registry.get_tool("Exchange Rate API")
        assert finance_tool is not None
        assert finance_tool.category == "Finance"

    def test_endpoint_count(self, toolbench_tree: Path) -> None:
        registry = load_toolbench(toolbench_tree)
        assert len(registry.all_endpoints) == 5


class TestParameterParsing:
    """Tests for parameter extraction and normalization."""

    def test_required_parameters(self, toolbench_tree: Path) -> None:
        registry = load_toolbench(toolbench_tree)
        ep = registry.get_endpoint("Open Weather Map.current weather data")
        assert ep is not None
        required = ep.required_parameters
        assert len(required) == 1
        assert required[0].name == "q"
        assert required[0].type == "STRING"
        assert required[0].required is True

    def test_optional_parameters(self, toolbench_tree: Path) -> None:
        registry = load_toolbench(toolbench_tree)
        ep = registry.get_endpoint("Open Weather Map.current weather data")
        assert ep is not None
        optional = ep.optional_parameters
        assert len(optional) == 1
        assert optional[0].name == "units"
        assert optional[0].default == "metric"

    def test_multiple_required_parameters(self, toolbench_tree: Path) -> None:
        registry = load_toolbench(toolbench_tree)
        ep = registry.get_endpoint("Exchange Rate API.convert")
        assert ep is not None
        required = ep.required_parameters
        assert len(required) == 3
        param_names = {p.name for p in required}
        assert param_names == {"from", "to", "amount"}

    def test_type_normalization(self, toolbench_tree: Path) -> None:
        registry = load_toolbench(toolbench_tree)
        ep = registry.get_endpoint("Exchange Rate API.convert")
        assert ep is not None
        amount_param = next(p for p in ep.parameters if p.name == "amount")
        assert amount_param.type == "NUMBER"

    def test_no_parameters(self, toolbench_tree: Path) -> None:
        registry = load_toolbench(toolbench_tree)
        ep = registry.get_endpoint("Exchange Rate API.list_currencies")
        assert ep is not None
        assert len(ep.parameters) == 0


class TestResponseFieldExtraction:
    """Tests for extracting response fields from body/schema."""

    def test_response_fields_from_body_dict(self, toolbench_tree: Path) -> None:
        registry = load_toolbench(toolbench_tree)
        ep = registry.get_endpoint("Open Weather Map.current weather data")
        assert ep is not None
        field_names = {rf.name for rf in ep.response_fields}
        assert "coord" in field_names
        assert "weather" in field_names
        assert "main" in field_names

    def test_response_fields_from_schema(self, toolbench_tree: Path) -> None:
        registry = load_toolbench(toolbench_tree)
        ep = registry.get_endpoint("Exchange Rate API.convert")
        assert ep is not None
        field_names = {rf.name for rf in ep.response_fields}
        assert "result" in field_names
        assert "rate" in field_names

    def test_schema_type_normalization(self, toolbench_tree: Path) -> None:
        registry = load_toolbench(toolbench_tree)
        ep = registry.get_endpoint("Exchange Rate API.convert")
        assert ep is not None
        result_field = next(rf for rf in ep.response_fields if rf.name == "result")
        assert result_field.type == "NUMBER"

    def test_fallback_when_no_body_or_schema(self, toolbench_tree: Path) -> None:
        registry = load_toolbench(toolbench_tree)
        ep = registry.get_endpoint("Exchange Rate API.list_currencies")
        assert ep is not None
        field_names = {rf.name for rf in ep.response_fields}
        assert "result" in field_names
        assert "status" in field_names

    def test_placeholder_body_gets_fallback(self, toolbench_tree: Path) -> None:
        registry = load_toolbench(toolbench_tree)
        ep = registry.get_endpoint("Messy API.placeholder_body")
        assert ep is not None
        field_names = {rf.name for rf in ep.response_fields}
        assert "result" in field_names
        assert "status" in field_names


class TestMessyDataHandling:
    """Tests for the various ToolBench data inconsistencies."""

    def test_null_score_is_handled(self, toolbench_tree: Path) -> None:
        registry = load_toolbench(toolbench_tree)
        tool = registry.get_tool("Exchange Rate API")
        assert tool is not None

    def test_empty_description_is_handled(self, toolbench_tree: Path) -> None:
        registry = load_toolbench(toolbench_tree)
        tool = registry.get_tool("Messy API")
        assert tool is not None
        assert tool.description == ""

    def test_statuscode_111_endpoint_is_loaded(self, toolbench_tree: Path) -> None:
        registry = load_toolbench(toolbench_tree)
        ep = registry.get_endpoint("Messy API.do_thing")
        assert ep is not None

    def test_empty_body_string_handled(self, toolbench_tree: Path) -> None:
        registry = load_toolbench(toolbench_tree)
        ep = registry.get_endpoint("Messy API.do_thing")
        assert ep is not None
        field_names = {rf.name for rf in ep.response_fields}
        assert "result" in field_names

    def test_malformed_json_is_skipped(self, tmp_path: Path) -> None:
        cat_dir = tmp_path / "BadCategory"
        cat_dir.mkdir()
        with open(cat_dir / "broken.json", "w") as f:
            f.write("{invalid json!!!}")
        registry = load_toolbench(tmp_path)
        assert len(registry.tools) == 0

    def test_tool_with_no_api_list_is_skipped(self, tmp_path: Path) -> None:
        cat_dir = tmp_path / "Empty"
        cat_dir.mkdir()
        with open(cat_dir / "no_apis.json", "w") as f:
            json.dump({"tool_name": "empty", "api_list": []}, f)
        registry = load_toolbench(tmp_path)
        assert len(registry.tools) == 0


class TestConceptInference:
    """Tests for concept tag derivation."""

    def test_weather_concept(self, toolbench_tree: Path) -> None:
        registry = load_toolbench(toolbench_tree)
        tool = registry.get_tool("Open Weather Map")
        assert tool is not None
        concept_names = [c.name for c in tool.concepts]
        assert "weather" in concept_names

    def test_finance_concept(self, toolbench_tree: Path) -> None:
        registry = load_toolbench(toolbench_tree)
        tool = registry.get_tool("Exchange Rate API")
        assert tool is not None
        concept_names = [c.name for c in tool.concepts]
        assert "finance" in concept_names


class TestFlatDirectoryFallback:
    """Tests for backward-compatible flat directory loading."""

    def test_flat_dir_loads(self, flat_data_dir: Path) -> None:
        registry = load_toolbench(flat_data_dir)
        assert len(registry.tools) == 1
        assert registry.tools[0].name == "flat_tool"

    def test_no_json_files_raises(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            load_toolbench(empty_dir)


class TestRegistrySerialization:
    """Tests for registry round-trip serialization."""

    def test_roundtrip(self, toolbench_tree: Path, tmp_path: Path) -> None:
        registry = load_toolbench(toolbench_tree)
        out_path = tmp_path / "registry.json"
        save_registry(registry, out_path)
        loaded = load_registry(out_path)
        assert len(loaded.tools) == len(registry.tools)
        assert len(loaded.all_endpoints) == len(registry.all_endpoints)
        for orig, loaded_ep in zip(registry.all_endpoints, loaded.all_endpoints):
            assert orig.id == loaded_ep.id
            assert orig.name == loaded_ep.name
            assert len(orig.parameters) == len(loaded_ep.parameters)


class TestRegistryModel:
    """Tests for ToolRegistry model methods."""

    def test_get_tool_returns_none_for_missing(self) -> None:
        registry = ToolRegistry()
        assert registry.get_tool("nonexistent") is None

    def test_get_endpoint_returns_none_for_missing(self) -> None:
        registry = ToolRegistry()
        assert registry.get_endpoint("nonexistent.ep") is None


class TestRealToolBenchData:
    """Test loading the actual ToolBench data bundled with the project."""

    @pytest.fixture
    def real_data_dir(self) -> Path:
        path = Path("data/toolbench/tools")
        if not path.exists() or not any(path.iterdir()):
            pytest.skip("Real ToolBench data not present")
        return path

    def test_loads_all_real_tools(self, real_data_dir: Path) -> None:
        registry = load_toolbench(real_data_dir)
        assert len(registry.tools) >= 40
        assert len(registry.all_endpoints) >= 100

    def test_all_tools_have_endpoints(self, real_data_dir: Path) -> None:
        registry = load_toolbench(real_data_dir)
        for tool in registry.tools:
            assert len(tool.endpoints) > 0, f"{tool.name} has no endpoints"

    def test_all_endpoints_have_ids(self, real_data_dir: Path) -> None:
        registry = load_toolbench(real_data_dir)
        for ep in registry.all_endpoints:
            assert ep.id, f"Endpoint {ep.name} has no id"
            assert "." in ep.id, f"Endpoint id {ep.id} missing dot separator"

    def test_categories_are_diverse(self, real_data_dir: Path) -> None:
        registry = load_toolbench(real_data_dir)
        categories = {t.category for t in registry.tools}
        assert len(categories) >= 5

    def test_roundtrip_with_real_data(self, real_data_dir: Path, tmp_path: Path) -> None:
        registry = load_toolbench(real_data_dir)
        out_path = tmp_path / "registry.json"
        save_registry(registry, out_path)
        loaded = load_registry(out_path)
        assert len(loaded.tools) == len(registry.tools)
        assert len(loaded.all_endpoints) == len(registry.all_endpoints)
