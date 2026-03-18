# ConvTool — Offline Multi-Agent Tool-Use Conversation Generator

ConvTool generates synthetic multi-turn, multi-tool conversations from ToolBench API schemas. It is designed to produce training data for tool-use AI agents systems that need to select the right API, fill arguments correctly, and chain multiple calls to complete a user task.

The system runs entirely offline: no API keys, no Docker, no external services.

---

## How it works

```
ToolBench JSON files
  → Tool Registry           normalize schemas into typed dataclasses
    → Knowledge Graph       NetworkX DiGraph with 5 node types, 6 edge types
      → 2-Layer Sampler     ChainSampler (graph walk) + PatternSampler (topology)
        → 5-Agent System    Planner, UserProxy, Assistant, Validator, Orchestrator
          → Execution       schema validation, deterministic mock outputs, session state
            → Memory        mem0 session scope (in-conversation) + corpus scope (cross-run)
              → JSONL        structured dataset with messages, tool calls, and metadata
```

---

## Getting Started

**Prerequisites:** Python 3.10+ (no API keys, no Docker, no external services needed)

### 1. Clone and install

```bash
git clone https://github.com/jdirgha/ConvTool.git
cd ConvTool

python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

### 2. Build the tool registry

The repository ships with a curated subset of **43 ToolBench tools** across 6 domains (`Finance`, `Weather`, `Travel`, `Food`, `Location`, `Entertainment`) in `data/toolbench/tools/`.

```bash
tooluse build --data-dir data/toolbench/tools --output-dir output
```

### 3. Generate conversations

```bash
# Run A — corpus memory disabled (diversity baseline)
tooluse generate --seed 42 -n 55 -o output/run_a.jsonl --no-corpus-memory

# Run B — corpus memory enabled (default)
tooluse generate --seed 42 -n 55 -o output/run_b.jsonl
```

Generated datasets are saved to `output/run_a.jsonl` and `output/run_b.jsonl`.

### 4. Validate and compute metrics

```bash
tooluse validate -d output/run_a.jsonl
tooluse validate -d output/run_b.jsonl

tooluse metrics -d output/run_a.jsonl
tooluse metrics -d output/run_b.jsonl
```

### 5. Run tests

```bash
pytest
```

> **Note:** Pre-generated `run_a.jsonl` and `run_b.jsonl` are already included in `output/`, so you can skip to step 4 to inspect results immediately after cloning.

---

## Running on the Full ToolBench Dataset

The loader handles any number of tools and categories. It gracefully skips malformed entries without crashing (see [ToolBench Data Handling](#toolbench-data-handling) below).

### Step 1 — Download ToolBench

```bash
# 1) Download the full ToolBench repo (this gives you the folder layout)
cd /Users/dirghajivani/Desktop/Agentic
git clone https://github.com/OpenBMB/ToolBench.git
cd ToolBench

# 2) Download `data.zip` (contains ToolBench/data/toolenv/tools/<Category>/*.json)
#
# Recommended (tool-assisted) download:
# - Install gdown
pip install gdown

# - Download the exact `data.zip` from the official Google Drive folder:
#   https://drive.google.com/drive/folders/1TysbSWYpP8EioFu9xPJtpbJZMLLmwAmL
gdown --id 1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk -O data.zip

# Sanity-check before unzip (avoid the “HTML page downloaded instead of zip” issue)
file data.zip

# If it is not a real zip (often it reports “HTML document” and is ~1–2KB):
#   - delete the bad file and re-run the gdown command
# Unzip
unzip data.zip
```

### Step 2 — Build the registry

If you prefer manual download: open the same Google Drive folder link, download `data.zip` to `ToolBench/`, then run `unzip data.zip`.

```bash
tooluse build \
  --data-dir ./ToolBench/data/toolenv/tools \
  --output-dir output
```

### Step 3 — Generate as normal

```bash
tooluse generate --seed 42 -n 100 -o output/run_b.jsonl
```

No other changes are needed. The sampler, agents, and memory layer scale automatically with the registry size.

---

## ToolBench Data Handling

The loader (`registry/loader.py`) handles every inconsistency pattern found in the ToolBench dataset without crashing:

| Inconsistency | Handling |
|---|---|
| `required_parameters` / `optional_parameters` is `None` | Treated as empty list via `or []` |
| Parameter entry is not a dict | Silently skipped |
| Parameter `name` missing | Falls back to `param_name` then `"unknown"` |
| Parameter `type` empty or non-string | Defaults to `"STRING"` |
| `description` is not a string | Reset to empty string |
| `method` missing or not a valid HTTP verb | Defaults to `"GET"` |
| `body` is an empty string or `None` | Returns no response fields |
| `body` is the placeholder `{"key1":"value","key2":"value"}` | Detected and ignored |
| `schema` field missing or malformed | Falls through to `body` extraction |
| `enum` field is not a list of strings | Set to `None` |
| `default` is an empty string | Set to `None` |
| Entire JSON file is malformed or non-UTF-8 | Logged as warning, file skipped |
| `api_list` missing or empty | Tool skipped with warning |
| Tool `name` field missing | Tries `tool_name` → `name` → `title` → `"unknown_tool"` |
| Individual API entry raises an exception | Caught per-entry, logged, skipped |

Response field extraction follows a 3-level priority chain:
1. `schema.properties` — JSON Schema format, most structured
2. `body` — example response object, types inferred from values
3. Fallback: `[{result: STRING}, {status: STRING}]`

---

## CLI Reference

### `build`
```bash
tooluse build [OPTIONS]

  --data-dir PATH    Directory containing ToolBench JSONs.  Default: data/toolbench/tools
  --output-dir PATH  Directory for output artifacts.        Default: output
```

### `generate`
```bash
tooluse generate [OPTIONS]

  --seed INTEGER       Random seed for reproducibility.  Default: 42
  -n INTEGER           Number of conversations.          Default: 50
  -o PATH              Output JSONL file.                Default: output/conversations.jsonl
  --registry PATH      Path to registry JSON.            Default: output/registry.json
  --no-corpus-memory   Disable corpus memory (Run A mode).
```

### `validate`
```bash
tooluse validate [OPTIONS]

  -d PATH          Dataset file to validate.  Default: output/conversations.jsonl
  --registry PATH  Path to registry JSON.     Default: output/registry.json
```

### `metrics`
```bash
tooluse metrics [OPTIONS]

  -d PATH          Dataset file to analyse.   Default: output/conversations.jsonl
  --registry PATH  Path to registry JSON.     Default: output/registry.json
```

---

## Output Format

Each JSONL record contains:

```json
{
  "messages": [
    {"role": "user",      "content": "I need to check the GOOGL stock price. The symbol is GOOGL."},
    {"role": "assistant", "content": "Before I proceed, I need the interval. What should it be?"},
    {"role": "user",      "content": "It's 1day."},
    {"role": "assistant", "content": "I retrieved the GOOGL data from Twelve Data successfully."}
  ],
  "tool_calls": [
    {
      "endpoint_id": "Twelve Data.ADD",
      "arguments": {"symbol": "GOOGL", "interval": "1day", "outputsize": 30}
    }
  ],
  "tool_outputs": [
    {"result": "result_8710c86f", "status": "confirmed"}
  ],
  "metadata": {
    "seed": 42,
    "tool_ids_used": ["Twelve Data"],
    "num_turns": 4,
    "num_clarification_questions": 1,
    "memory_grounding_rate": 1.0,
    "corpus_memory_enabled": true,
    "pattern_type": "pipeline",
    "domain": "finance"
  }
}
```

---

## Project Structure

```
ConvTool/
├── ConvTool/
│   └── tooluse/
│       ├── cli.py                 ← CLI entry point
│       ├── registry/
│       │   ├── models.py          ← Tool, Endpoint, ParameterSchema, ResponseField dataclasses
│       │   └── loader.py          ← ToolBench parser and registry serialization
│       ├── graph/
│       │   ├── builder.py         ← NetworkX DiGraph: 5 node types, 6 edge types
│       │   ├── sampler.py         ← ChainSampler: weighted random walk (Layer 1)
│       │   └── patterns.py        ← PatternSampler: conversation topology (Layer 2)
│       ├── sampler/
│       │   └── base.py            ← ToolChain, PatternType, ToolChainStep dataclasses
│       ├── execution/
│       │   └── engine.py          ← Schema validation, mock output, session state
│       ├── agents/
│       │   ├── params.py          ← Context-aware parameter value generation
│       │   ├── sampler_agent.py   ← Proposes tool chains from the graph
│       │   ├── planner_agent.py   ← Scenario selection and step planning
│       │   ├── user_proxy.py      ← User-side messages and follow-ups
│       │   ├── assistant_agent.py ← Tool calls, clarifications, result summaries
│       │   ├── validator_agent.py ← Structural and chaining validation
│       │   └── orchestrator.py    ← Coordinates all agents end-to-end
│       ├── memory/
│       │   ├── store.py           ← MemoryStore ABC (add / search / reset)
│       │   ├── mem0_store.py      ← mem0-backed store (Memory())
│       │   ├── in_memory.py       ← Pure-Python fallback for offline / tests
│       │   └── factory.py         ← Selects backend based on environment
│       ├── dataset/
│       │   └── writer.py          ← JSONL writer and reader
│       └── metrics/
│           └── diversity.py       ← Jaccard dissimilarity, pattern entropy, tool coverage
├── tests/
│   ├── unit/
│   │   ├── test_registry.py       ← Parsing and type normalization
│   │   ├── test_execution.py      ← Validation and mock output
│   │   ├── test_memory.py         ← MemoryStore add/search and scope isolation
│   │   └── test_metrics.py        ← Diversity metric calculations
│   └── e2e/
│       └── test_pipeline.py       ← Full pipeline: build → generate → validate → metrics
├── data/
│   └── toolbench/tools/           ← Included 43-tool subset (6 domains, 517 endpoints)
├── output/                        ← Generated artifacts (registry.json, run_a.jsonl, run_b.jsonl)
├── DESIGN.md
├── README.md
└── pyproject.toml
```
