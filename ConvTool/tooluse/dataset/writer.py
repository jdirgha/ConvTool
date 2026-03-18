"""JSONL dataset writer and reader.

Each record contains the full conversation, tool calls, tool outputs,
and metadata required for training or evaluating tool-use agents.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from tooluse.agents.orchestrator import GeneratedConversation

logger = logging.getLogger(__name__)


class DatasetWriter:
    """Writes generated conversations to a JSONL file."""

    def __init__(self, output_path: Path) -> None:
        self._path = output_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._count = 0

    def write(self, conversation: GeneratedConversation) -> None:
        """Append a single conversation record to the dataset."""
        record = {
            "messages": conversation.messages,
            "tool_calls": conversation.tool_calls,
            "tool_outputs": conversation.tool_outputs,
            "metadata": conversation.metadata,
        }
        mode = "a" if self._count > 0 else "w"
        with open(self._path, mode, encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._count += 1

    @property
    def count(self) -> int:
        return self._count

    @property
    def path(self) -> Path:
        return self._path


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL dataset into a list of conversation records."""
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d records from %s", len(records), path)
    return records
