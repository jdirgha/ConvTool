"""End-to-end test: build registry, build graph, generate dataset, validate.

Runs fully offline — no external services required.
Uses ``make_memory_store()`` so the correct backend is selected
automatically: Mem0MemoryStore when an API key is present, InMemoryStore
otherwise.  This mirrors exactly what the CLI does.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.timeout(300)
class TestEndToEnd:
    """Full pipeline: build → generate → validate → metrics."""

    def test_full_pipeline_generates_valid_dataset(self, tmp_path: Path) -> None:
        from tooluse.registry.loader import load_toolbench, save_registry, load_registry
        from tooluse.graph.builder import GraphBuilder
        from tooluse.memory.factory import make_memory_store
        from tooluse.agents.orchestrator import ConversationOrchestrator
        from tooluse.dataset.writer import DatasetWriter, load_dataset
        from tooluse.agents.validator_agent import ValidatorAgent
        from tooluse.metrics.diversity import compute_metrics

        data_dir = Path("data/toolbench/tools")
        if not data_dir.exists():
            pytest.skip("Real ToolBench data not present")

        registry_path = tmp_path / "registry.json"
        dataset_path = tmp_path / "conversations.jsonl"
        seed = 42
        num_conversations = 50

        # --- Build phase ---
        registry = load_toolbench(data_dir)
        save_registry(registry, registry_path)

        assert len(registry.tools) > 0
        assert len(registry.all_endpoints) > 0

        with GraphBuilder() as graph:
            graph.ingest(registry)

            # Some endpoints may be skipped during ingestion (e.g. empty IDs,
            # duplicate names) — require at least 95 % coverage.
            graph_ep_count = len(graph.get_all_endpoint_ids())
            registry_ep_count = len(registry.all_endpoints)
            assert graph_ep_count >= registry_ep_count * 0.95, (
                f"Graph has {graph_ep_count} endpoints but registry has "
                f"{registry_ep_count}; too many were dropped during ingestion."
            )

            # --- Generate phase ---
            memory = make_memory_store()
            orchestrator = ConversationOrchestrator(
                registry=registry,
                graph=graph,
                memory=memory,
                corpus_memory_enabled=True,
            )
            writer = DatasetWriter(dataset_path)

            for i in range(num_conversations):
                conversation = orchestrator.generate(seed=seed, conversation_index=i)
                writer.write(conversation)

            assert writer.count == num_conversations

        # --- Validate phase ---
        records = load_dataset(dataset_path)
        assert len(records) == num_conversations

        validator = ValidatorAgent(registry)
        valid_count = sum(1 for r in records if validator.validate(r).valid)

        # At least 80% of conversations must be valid
        assert valid_count >= num_conversations * 0.8, (
            f"Only {valid_count}/{num_conversations} conversations passed validation"
        )

        # --- Metrics phase ---
        metrics_result = compute_metrics(records, total_tool_count=len(registry.tools))
        assert metrics_result.num_conversations == num_conversations
        assert metrics_result.avg_jaccard_dissimilarity >= 0.0
        assert metrics_result.pattern_entropy >= 0.0

        # --- Dataset format check ---
        for record in records:
            assert "messages" in record
            assert "tool_calls" in record
            assert "tool_outputs" in record
            assert "metadata" in record

            meta = record["metadata"]
            for field in (
                "seed", "tool_ids_used", "num_turns",
                "num_clarification_questions", "memory_grounding_rate",
                "corpus_memory_enabled",
            ):
                assert field in meta, f"Missing metadata field: {field}"

    def test_no_corpus_memory_run(self, tmp_path: Path) -> None:
        """Run A: corpus memory disabled — verify output is still valid."""
        from tooluse.registry.loader import load_toolbench
        from tooluse.graph.builder import GraphBuilder
        from tooluse.memory.factory import make_memory_store
        from tooluse.agents.orchestrator import ConversationOrchestrator
        from tooluse.dataset.writer import DatasetWriter, load_dataset

        data_dir = Path("data/toolbench/tools")
        if not data_dir.exists():
            pytest.skip("Real ToolBench data not present")

        dataset_path = tmp_path / "run_a.jsonl"
        seed = 42
        num_conversations = 10

        registry = load_toolbench(data_dir)

        with GraphBuilder() as graph:
            graph.ingest(registry)

            memory = make_memory_store()
            orchestrator = ConversationOrchestrator(
                registry=registry,
                graph=graph,
                memory=memory,
                corpus_memory_enabled=False,
            )
            writer = DatasetWriter(dataset_path)

            for i in range(num_conversations):
                writer.write(orchestrator.generate(seed=seed, conversation_index=i))

        records = load_dataset(dataset_path)
        assert len(records) == num_conversations
        for record in records:
            assert record["metadata"]["corpus_memory_enabled"] is False
