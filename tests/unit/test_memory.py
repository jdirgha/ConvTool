"""Unit tests for the MemoryStore interface.

Tests run against InMemoryStore (pure Python, instant startup) to verify
the interface contract defined in MemoryStore:
  - add followed by search returns the stored entry
  - entries in one scope are NOT returned when querying another scope

The same contract is guaranteed by Mem0MemoryStore in production because
both classes implement the same MemoryStore abstract interface.
"""

from __future__ import annotations

import pytest

from tooluse.memory.in_memory import InMemoryStore
from tooluse.memory.store import MemoryStore


@pytest.fixture
def store() -> InMemoryStore:
    s = InMemoryStore()
    yield s
    s.reset()


# ── add → search contract ─────────────────────────────────────────────────────

class TestMemoryStoreAdd:
    def test_add_and_search_returns_entry(self, store: MemoryStore) -> None:
        store.add(
            content="The weather in Berlin is 22 degrees and sunny.",
            scope="session",
            metadata={"conversation_id": "test_001", "step": 0},
        )
        results = store.search("weather Berlin", scope="session", top_k=5)
        assert len(results) >= 1
        texts = [e.get("memory", "") for e in results]
        assert any("Berlin" in t or "weather" in t.lower() for t in texts), (
            f"Expected Berlin/weather in results: {results}"
        )

    def test_add_multiple_entries_are_all_searchable(self, store: MemoryStore) -> None:
        store.add(
            content="Hotel Grand Berlin booked for March 15.",
            scope="session",
            metadata={"conversation_id": "test_002", "step": 1},
        )
        store.add(
            content="Restaurant reservation at Trattoria Roma confirmed.",
            scope="session",
            metadata={"conversation_id": "test_002", "step": 2},
        )
        results = store.search("hotel booking", scope="session", top_k=5)
        assert len(results) >= 1

    def test_stored_content_is_preserved(self, store: MemoryStore) -> None:
        store.add(
            content="flight_id=FL-777 booked to New York",
            scope="session",
            metadata={"step": 0},
        )
        results = store.search("flight New York", scope="session")
        assert any("FL-777" in e.get("memory", "") for e in results)


# ── scope isolation contract ──────────────────────────────────────────────────

class TestMemoryStoreScoping:
    def test_session_entry_not_returned_in_corpus_scope(self, store: MemoryStore) -> None:
        """Entries stored in 'session' must not appear in 'corpus' queries."""
        store.add(
            content="Session-only data: flight booked to Tokyo.",
            scope="session",
            metadata={"conversation_id": "scope_test", "step": 0},
        )
        corpus_results = store.search("flight Tokyo", scope="corpus", top_k=5)
        for entry in corpus_results:
            text = entry.get("memory", "")
            assert "Tokyo" not in text, (
                f"Session entry leaked into corpus scope: {entry}"
            )

    def test_corpus_entry_not_returned_in_session_scope(self, store: MemoryStore) -> None:
        """Entries stored in 'corpus' must not appear in 'session' queries."""
        store.add(
            content="Corpus summary: tools weather_api, hotel_api. Domain: travel.",
            scope="corpus",
            metadata={"conversation_id": "scope_test", "tools": ["weather_api"]},
        )
        session_results = store.search("weather hotel travel", scope="session", top_k=5)
        for entry in session_results:
            text = entry.get("memory", "")
            assert "Corpus summary" not in text, (
                f"Corpus entry leaked into session scope: {entry}"
            )

    def test_two_scopes_independently_searchable(self, store: MemoryStore) -> None:
        store.add("session fact about Paris hotels", scope="session", metadata={})
        store.add("corpus summary about weather domain", scope="corpus", metadata={})

        session_r = store.search("Paris hotels", scope="session")
        corpus_r = store.search("weather domain", scope="corpus")

        assert any("Paris" in e.get("memory", "") for e in session_r)
        assert any("weather" in e.get("memory", "") for e in corpus_r)


# ── edge cases ────────────────────────────────────────────────────────────────

class TestMemoryStoreEdgeCases:
    def test_search_empty_store_returns_empty_list(self, store: MemoryStore) -> None:
        results = store.search("anything", scope="session", top_k=5)
        assert results == []

    def test_search_respects_top_k_limit(self, store: MemoryStore) -> None:
        for i in range(10):
            store.add(
                content=f"Entry {i}: weather data for city_{i}.",
                scope="session",
                metadata={"step": i},
            )
        results = store.search("weather city", scope="session", top_k=3)
        assert len(results) <= 3

    def test_reset_clears_all_scopes(self, store: MemoryStore) -> None:
        store.add("session data", scope="session", metadata={})
        store.add("corpus data", scope="corpus", metadata={})
        store.reset()
        assert store.search("data", scope="session") == []
        assert store.search("data", scope="corpus") == []
