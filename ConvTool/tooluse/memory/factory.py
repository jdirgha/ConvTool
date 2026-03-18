"""Factory for selecting the right MemoryStore backend at runtime."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def make_memory_store():
    """Return a MemoryStore for the current environment.

    Uses Mem0MemoryStore when an OPENAI_API_KEY or ANTHROPIC_API_KEY is present
    and MEM0_ENABLED is not "false". Falls back to InMemoryStore otherwise so
    the system works fully offline.
    """
    from tooluse.memory.in_memory import InMemoryStore
    from tooluse.memory.mem0_store import Mem0MemoryStore

    if os.environ.get("MEM0_ENABLED", "true").lower() == "false":
        logger.info("MEM0_ENABLED=false — using InMemoryStore.")
        return InMemoryStore()

    has_api_key = bool(
        os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    )
    if not has_api_key:
        logger.info("No API key found — using InMemoryStore (offline mode).")
        return InMemoryStore()

    logger.info("API key detected — using Mem0MemoryStore.")
    return Mem0MemoryStore()
