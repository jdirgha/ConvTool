"""mem0-backed MemoryStore for production use."""

from __future__ import annotations

import logging
from typing import Any

from tooluse.memory.store import MemoryStore

logger = logging.getLogger(__name__)


class Mem0MemoryStore(MemoryStore):
    """Wraps mem0.Memory, using ``user_id`` as the scope namespace."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        # Lazy import so that tests and offline runs don't crash on mem0 load.
        from mem0 import Memory
        self._client = Memory(config) if config else Memory()
        logger.info("Mem0MemoryStore ready (in-process Qdrant).")

    def add(self, content: str, scope: str, metadata: dict[str, Any]) -> None:
        self._client.add(
            content,
            user_id=scope,
            metadata=metadata,
            infer=False,
        )
        logger.debug("Memory added [scope=%s]: %.80s", scope, content)

    def search(
        self, query: str, scope: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        try:
            raw = self._client.search(query, user_id=scope, limit=top_k)
            results = raw.get("results", []) if isinstance(raw, dict) else raw
            entries = [
                item for item in results
                if isinstance(item, dict) and item.get("memory")
            ]
            logger.debug(
                "Memory search [scope=%s, query=%.40s]: %d results",
                scope, query, len(entries),
            )
            return entries
        except Exception:
            logger.exception("Memory search failed [scope=%s]", scope)
            return []

    def reset(self) -> None:
        for scope in ("session", "corpus"):
            try:
                self._client.delete_all(user_id=scope)
            except Exception:
                logger.debug("Could not reset scope '%s'.", scope)
        logger.info("Mem0MemoryStore reset.")
