"""CLI interface for the tooluse pipeline.

Commands:
    build     — Load ToolBench data and build the tool registry.
    generate  — Generate synthetic conversations.
    validate  — Validate an existing dataset.
    metrics   — Compute diversity metrics on a dataset.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tooluse.cli")


@click.group()
def main() -> None:
    """Offline synthetic multi-agent tool-use conversation generator."""


@main.command()
@click.option("--data-dir", type=click.Path(exists=True), default="data/toolbench/tools",
              help="Directory containing ToolBench tool definitions (Category/tool.json).")
@click.option("--output-dir", type=click.Path(), default="output",
              help="Directory for output artifacts.")
def build(data_dir: str, output_dir: str) -> None:
    """Build the tool registry from ToolBench data."""
    from tooluse.registry.loader import load_toolbench, save_registry

    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    registry_path = out_path / "registry.json"

    logger.info("Loading ToolBench data from %s", data_path)
    registry = load_toolbench(data_path)
    save_registry(registry, registry_path)
    logger.info(
        "Registry built: %d tools, %d endpoints. Saved to %s",
        len(registry.tools), len(registry.all_endpoints), registry_path,
    )
    logger.info("Build complete.")


@main.command()
@click.option("--seed", type=int, default=42, help="Random seed for determinism.")
@click.option("--num-conversations", "-n", type=int, default=50,
              help="Number of conversations to generate.")
@click.option("--output", "-o", type=click.Path(), default="output/conversations.jsonl",
              help="Output JSONL file path.")
@click.option("--registry", type=click.Path(exists=True), default="output/registry.json",
              help="Path to the registry JSON file.")
@click.option("--no-corpus-memory", is_flag=True, default=False,
              help="Disable corpus memory for diversity comparison (Run A).")
def generate(
    seed: int,
    num_conversations: int,
    output: str,
    registry: str,
    no_corpus_memory: bool,
) -> None:
    """Generate synthetic multi-agent conversations."""
    from tooluse.registry.loader import load_registry as _load_registry
    from tooluse.graph.builder import GraphBuilder
    from tooluse.memory.factory import make_memory_store
    from tooluse.agents.orchestrator import ConversationOrchestrator
    from tooluse.dataset.writer import DatasetWriter

    corpus_memory_enabled = not no_corpus_memory
    logger.info(
        "Generating %d conversations (seed=%d, corpus_memory=%s)",
        num_conversations, seed, corpus_memory_enabled,
    )

    reg = _load_registry(Path(registry))
    memory = make_memory_store()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with GraphBuilder() as graph:
        graph.ingest(reg)

        orchestrator = ConversationOrchestrator(
            registry=reg,
            graph=graph,
            memory=memory,
            corpus_memory_enabled=corpus_memory_enabled,
        )
        writer = DatasetWriter(output_path)

        valid_count = 0
        for i in range(num_conversations):
            conversation = orchestrator.generate(seed=seed, conversation_index=i)
            writer.write(conversation)
            if conversation.validation.valid:
                valid_count += 1

            if (i + 1) % 10 == 0:
                logger.info("Progress: %d/%d generated (%d valid)",
                            i + 1, num_conversations, valid_count)

    logger.info(
        "Generation complete: %d/%d valid conversations written to %s",
        valid_count, num_conversations, output_path,
    )


@main.command()
@click.option("--dataset", "-d", type=click.Path(exists=True),
              default="output/conversations.jsonl", help="Dataset file to validate.")
@click.option("--registry", type=click.Path(exists=True), default="output/registry.json",
              help="Path to the registry JSON file.")
def validate(dataset: str, registry: str) -> None:
    """Validate an existing conversation dataset."""
    from tooluse.registry.loader import load_registry as _load_registry
    from tooluse.dataset.writer import load_dataset
    from tooluse.agents.validator_agent import ValidatorAgent

    reg = _load_registry(Path(registry))
    records = load_dataset(Path(dataset))
    validator = ValidatorAgent(reg)

    valid_count = 0
    total_errors: list[str] = []

    for i, record in enumerate(records):
        report = validator.validate(record)
        if report.valid:
            valid_count += 1
        else:
            for err in report.errors:
                total_errors.append(f"Record {i}: {err}")

    click.echo(f"\nValidation Results")
    click.echo(f"{'='*40}")
    click.echo(f"Total records:  {len(records)}")
    click.echo(f"Valid:          {valid_count}")
    click.echo(f"Invalid:        {len(records) - valid_count}")

    if total_errors:
        click.echo(f"\nErrors ({len(total_errors)}):")
        for err in total_errors[:20]:
            click.echo(f"  - {err}")
        if len(total_errors) > 20:
            click.echo(f"  ... and {len(total_errors) - 20} more.")


@main.command()
@click.option("--dataset", "-d", type=click.Path(exists=True),
              default="output/conversations.jsonl", help="Dataset file to analyze.")
@click.option("--registry", type=click.Path(exists=True), default="output/registry.json",
              help="Path to the registry JSON file.")
def metrics(dataset: str, registry: str) -> None:
    """Compute diversity metrics on a generated dataset."""
    from tooluse.registry.loader import load_registry as _load_registry
    from tooluse.dataset.writer import load_dataset
    from tooluse.metrics.diversity import compute_metrics

    reg = _load_registry(Path(registry))
    records = load_dataset(Path(dataset))
    total_tools = len(reg.tools)

    result = compute_metrics(records, total_tool_count=total_tools)

    click.echo(f"\nDiversity Metrics")
    click.echo(f"{'='*50}")
    click.echo(f"Conversations analyzed:     {result.num_conversations}")
    click.echo(f"Corpus memory enabled:      {result.corpus_memory_enabled}")
    click.echo(f"")
    click.echo(f"Avg Jaccard dissimilarity:  {result.avg_jaccard_dissimilarity:.4f}")
    click.echo(f"Pattern entropy:            {result.pattern_entropy:.4f}")
    click.echo(f"Tool coverage:              {result.tool_coverage:.4f}")
    click.echo(f"Avg tools per conversation: {result.avg_tools_per_conversation:.2f}")
    click.echo(f"")
    click.echo(f"Pattern distribution:")
    for pattern, freq in result.pattern_distribution.items():
        click.echo(f"  {pattern:20s} {freq:.4f}")


if __name__ == "__main__":
    main()
