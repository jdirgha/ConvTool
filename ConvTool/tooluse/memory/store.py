"""Abstract MemoryStore interface.

All agents depend on this interface rather than mem0 directly, keeping the
codebase decoupled from the underlying memory backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MemoryStore(ABC):
    """Scoped memory store. Scope ("session" / "corpus") prevents cross-namespace leakage."""

    @abstractmethod
    def add(self, content: str, scope: str, metadata: dict[str, Any]) -> None: ...

    @abstractmethod
    def search(self, query: str, scope: str, top_k: int = 5) -> list[dict[str, Any]]: ...

    def reset(self) -> None:
        """Clear all stored memories."""
