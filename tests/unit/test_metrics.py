"""Unit tests for the diversity metrics engine."""

from __future__ import annotations

from tooluse.metrics.diversity import compute_metrics


def _make_record(tool_names: list[str], pattern: str = "linear") -> dict:
    return {
        "tool_calls": [
            {"endpoint_id": f"{t}.some_endpoint"} for t in tool_names
        ],
        "metadata": {
            "pattern_type": pattern,
            "corpus_memory_enabled": True,
        },
    }


def test_empty_dataset() -> None:
    m = compute_metrics([])
    assert m.num_conversations == 0
    assert m.avg_jaccard_dissimilarity == 0.0


def test_single_conversation() -> None:
    records = [_make_record(["weather_api", "hotel_api"])]
    m = compute_metrics(records, total_tool_count=6)
    assert m.num_conversations == 1
    assert m.avg_jaccard_dissimilarity == 0.0


def test_identical_conversations_have_zero_dissimilarity() -> None:
    records = [
        _make_record(["weather_api", "hotel_api"]),
        _make_record(["weather_api", "hotel_api"]),
        _make_record(["weather_api", "hotel_api"]),
    ]
    m = compute_metrics(records)
    assert m.avg_jaccard_dissimilarity == 0.0


def test_disjoint_conversations_have_max_dissimilarity() -> None:
    records = [
        _make_record(["weather_api"]),
        _make_record(["hotel_api"]),
    ]
    m = compute_metrics(records)
    assert m.avg_jaccard_dissimilarity == 1.0


def test_partial_overlap() -> None:
    records = [
        _make_record(["weather_api", "hotel_api"]),
        _make_record(["hotel_api", "restaurant_api"]),
    ]
    m = compute_metrics(records)
    # Jaccard({w,h}, {h,r}) = 1 - 1/3 = 0.6667
    assert 0.6 < m.avg_jaccard_dissimilarity < 0.7


def test_pattern_entropy_uniform() -> None:
    records = [
        _make_record(["a"], "linear"),
        _make_record(["b"], "fan_out"),
        _make_record(["c"], "conditional"),
    ]
    m = compute_metrics(records)
    # Three equally likely outcomes -> entropy = log2(3) ≈ 1.585
    assert 1.5 < m.pattern_entropy < 1.6


def test_pattern_entropy_single_type() -> None:
    records = [
        _make_record(["a"], "linear"),
        _make_record(["b"], "linear"),
    ]
    m = compute_metrics(records)
    assert m.pattern_entropy == 0.0


def test_tool_coverage() -> None:
    records = [
        _make_record(["weather_api", "hotel_api"]),
        _make_record(["restaurant_api"]),
    ]
    m = compute_metrics(records, total_tool_count=6)
    assert m.tool_coverage == 0.5
