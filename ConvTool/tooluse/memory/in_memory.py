"""Dict-backed MemoryStore for offline use and testing."""

from __future__ import annotations

from typing import Any

from tooluse.memory.store import MemoryStore


class InMemoryStore(MemoryStore):
    """Pure-Python MemoryStore with keyword-based search and strict scope isolation."""

    def __init__(self) -> None:
        self._data: dict[str, list[dict[str, Any]]] = {}

    def add(self, content: str, scope: str, metadata: dict[str, Any]) -> None:
        self._data.setdefault(scope, []).append({"memory": content, **metadata})

    def search(
        self, query: str, scope: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        entries = self._data.get(scope, [])
        if not entries:
            return []

        query_tokens = set(query.lower().split())

        def _score(entry: dict[str, Any]) -> int:
            text = entry.get("memory", "").lower()
            return sum(1 for token in query_tokens if token in text)

        ranked = sorted(entries, key=_score, reverse=True)
        matches = [e for e in ranked if _score(e) > 0]
        return (matches or ranked)[:top_k]

    def reset(self) -> None:
        self._data.clear()
