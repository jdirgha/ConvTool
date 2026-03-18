"""Unit tests for the offline execution engine.

Tests cover:
- Argument validation against endpoint schema
- Mock response generation consistent with response schema
- Session state chaining across multi-step tool calls
- Deterministic outputs (same inputs → same outputs)
- Error handling for unknown endpoints
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tooluse.execution.engine import ExecutionEngine, SessionState
from tooluse.registry.loader import load_toolbench


@pytest.fixture
def registry():
    path = Path("data/toolbench/tools")
    if not path.exists():
        pytest.skip("Real ToolBench data not present")
    return load_toolbench(path)


@pytest.fixture
def engine(registry):
    return ExecutionEngine(registry)


class TestArgumentValidation:
    """Validates arguments against the endpoint schema."""

    def test_valid_execution_with_required_params(self, engine: ExecutionEngine) -> None:
        session = SessionState()
        result = engine.execute(
            "Currency Conversion and Exchange Rates.Convert",
            {"from": "USD", "to": "EUR", "amount": "100"},
            session,
        )
        assert result.validation.valid
        assert len(result.validation.errors) == 0

    def test_missing_required_param_fails_validation(self, engine: ExecutionEngine) -> None:
        session = SessionState()
        result = engine.execute(
            "Currency Conversion and Exchange Rates.Convert",
            {"from": "USD"},
            session,
        )
        assert not result.validation.valid
        assert any("to" in err for err in result.validation.errors)

    def test_missing_all_required_params(self, engine: ExecutionEngine) -> None:
        session = SessionState()
        result = engine.execute(
            "Alpha Vantage.TIME_SERIES_DAILY",
            {},
            session,
        )
        assert not result.validation.valid
        assert len(result.validation.errors) >= 2

    def test_extra_params_are_accepted(self, engine: ExecutionEngine) -> None:
        session = SessionState()
        result = engine.execute(
            "Deezer.Radio",
            {"id": "123", "extra_field": "ignored"},
            session,
        )
        assert result.validation.valid


class TestTypeValidation:
    """Validates argument types match parameter schema types."""

    def test_correct_string_type_passes(self, engine: ExecutionEngine) -> None:
        session = SessionState()
        result = engine.execute(
            "Real-Time Finance Data.Stock Quote",
            {"symbol": "AAPL"},
            session,
        )
        assert result.validation.valid

    def test_number_where_string_expected_fails(self, engine: ExecutionEngine) -> None:
        session = SessionState()
        result = engine.execute(
            "Real-Time Finance Data.Stock Quote",
            {"symbol": 12345},
            session,
        )
        assert not result.validation.valid
        assert any("expects STRING" in e or "symbol" in e for e in result.validation.errors)

    def test_numeric_string_for_number_param_passes(self, engine: ExecutionEngine) -> None:
        """ToolBench has many NUMBER params passed as strings (e.g. "100")."""
        session = SessionState()
        result = engine.execute(
            "Deezer.Radio",
            {"id": "42"},
            session,
        )
        assert result.validation.valid

    def test_list_where_string_expected_fails(self, engine: ExecutionEngine) -> None:
        session = SessionState()
        result = engine.execute(
            "Real-Time Finance Data.Stock Quote",
            {"symbol": ["AAPL", "GOOGL"]},
            session,
        )
        assert not result.validation.valid
        assert any("ARRAY" in e or "expects" in e for e in result.validation.errors)


class TestMockResponseGeneration:
    """Mock response is consistent with response schema."""

    def test_response_has_schema_fields(self, engine: ExecutionEngine) -> None:
        session = SessionState()
        result = engine.execute(
            "Currency Conversion and Exchange Rates.Convert",
            {"from": "USD", "to": "EUR", "amount": "100"},
            session,
        )
        assert "date" in result.output
        assert "info" in result.output
        assert "query" in result.output
        assert "result" in result.output

    def test_response_fields_from_schema(self, engine: ExecutionEngine) -> None:
        session = SessionState()
        result = engine.execute(
            "Real-Time Finance Data.Stock Quote",
            {"symbol": "AAPL"},
            session,
        )
        assert "status" in result.output
        assert "request_id" in result.output
        assert "data" in result.output

    def test_response_for_endpoint_with_fallback_fields(self, engine: ExecutionEngine) -> None:
        session = SessionState()
        result = engine.execute(
            "Deezer.Radio",
            {"id": "123"},
            session,
        )
        assert "result" in result.output
        assert "status" in result.output

    def test_response_with_rich_schema(self, engine: ExecutionEngine) -> None:
        session = SessionState()
        result = engine.execute(
            "Currency Conversion and Exchange Rates.Historical Exchange Rates",
            {"date": "2024-01-15"},
            session,
        )
        assert "base" in result.output
        assert "date" in result.output
        assert "rates" in result.output
        assert "historical" in result.output


class TestSessionStateChaining:
    """Session state carries forward so later calls reference earlier outputs."""

    def test_session_records_outputs(self, engine: ExecutionEngine) -> None:
        session = SessionState()
        engine.execute(
            "Currency Conversion and Exchange Rates.Convert",
            {"from": "USD", "to": "EUR", "amount": "100"},
            session,
        )
        assert session.get("result") is not None
        assert session.get("date") is not None

    def test_session_enriches_missing_params(self, engine: ExecutionEngine) -> None:
        session = SessionState()

        engine.execute(
            "Currency Conversion and Exchange Rates.Recent Exchange Rates",
            {},
            session,
        )
        stored_date = session.get("date")
        assert stored_date is not None

        result = engine.execute(
            "Currency Conversion and Exchange Rates.Historical Exchange Rates",
            {},
            session,
        )
        assert result.arguments.get("date") == stored_date

    def test_multi_step_chaining(self, engine: ExecutionEngine) -> None:
        """Chain: search → get details using ID from first call."""
        session = SessionState()

        engine.execute(
            "Flightera Flight Data.airlineStatistics",
            {"ident": "LH"},
            session,
        )
        stored_ident = session.get("ident")

        result = engine.execute(
            "Flightera Flight Data.airlineAircrafts",
            {},
            session,
        )
        assert result.arguments.get("ident") is not None

    def test_session_output_history(self, engine: ExecutionEngine) -> None:
        session = SessionState()
        engine.execute("Deezer.Radio", {"id": "1"}, session)
        engine.execute("Deezer.Artist", {"id": "2"}, session)
        outputs = session.get_outputs()
        assert len(outputs) == 2
        assert outputs[0]["endpoint_id"] == "Deezer.Radio"
        assert outputs[1]["endpoint_id"] == "Deezer.Artist"


class TestDeterminism:
    """Same inputs always produce the same outputs."""

    def test_deterministic_outputs(self, engine: ExecutionEngine) -> None:
        s1, s2 = SessionState(), SessionState()
        r1 = engine.execute(
            "Real-Time Finance Data.Stock Quote",
            {"symbol": "AAPL"},
            s1,
        )
        r2 = engine.execute(
            "Real-Time Finance Data.Stock Quote",
            {"symbol": "AAPL"},
            s2,
        )
        assert r1.output == r2.output

    def test_different_args_produce_different_outputs(self, engine: ExecutionEngine) -> None:
        s1, s2 = SessionState(), SessionState()
        r1 = engine.execute(
            "Real-Time Finance Data.Stock Quote",
            {"symbol": "AAPL"},
            s1,
        )
        r2 = engine.execute(
            "Real-Time Finance Data.Stock Quote",
            {"symbol": "GOOGL"},
            s2,
        )
        assert r1.output != r2.output


class TestErrorHandling:
    """Graceful handling of unknown endpoints and edge cases."""

    def test_unknown_endpoint(self, engine: ExecutionEngine) -> None:
        session = SessionState()
        result = engine.execute("nonexistent.endpoint", {"x": 1}, session)
        assert not result.validation.valid
        assert "error" in result.output

    def test_empty_arguments(self, engine: ExecutionEngine) -> None:
        session = SessionState()
        result = engine.execute(
            "Love Calculator.getPercentage",
            {},
            session,
        )
        assert not result.validation.valid
        assert len(result.validation.errors) >= 1
