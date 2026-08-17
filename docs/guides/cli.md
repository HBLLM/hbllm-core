---
title: "CLI Reference — Complete Command-Line Manual"
description: "Comprehensive guide to all HBLLM command-line utilities: serve, train, data, info, nodes, plugin, agent, code, diagnostics, and DPO export."
---

# CLI Reference Manual

HBLLM provides an extensive command-line interface (`hbllm`) for managing the cognitive server, pre-training pipelines, plugin scaffolding, interactive agents, and system diagnostics.

---

## Command Overview

```bash
hbllm [command] [options]
```

| Command | Purpose |
|---|---|
| [`hbllm serve`](#hbllm-serve) | Start the FastAPI REST and MCP server |
| [`hbllm train`](#hbllm-train) | Start the causal language model pre-training loop |
| [`hbllm data`](#hbllm-data) | Run the data preparation and tokenization pipeline |
| [`hbllm info`](#hbllm-info) | Display active system architecture and memory tiers |
| [`hbllm nodes`](#hbllm-nodes) | List all 28+ loaded cognitive nodes |
| [`hbllm plugin`](#hbllm-plugin) | List installed plugins or scaffold a new custom plugin |
| [`hbllm agent`](#hbllm-agent) | Launch the autonomous developer terminal agent |
| [`hbllm code`](#hbllm-code) | Launch terminal-based pair programming assistant |
| [`hbllm.cli.diagnostics`](#diagnostics-cli) | System runtime health and native Rust kernel validator |
| [`hbllm.cli.export_dpo`](#dpo-export-cli) | Export preference data for Direct Preference Optimization |

---

## Core Commands

### `hbllm serve`

Starts the production-ready HTTP server with OpenAI-compatible chat endpoints and cognitive REST APIs.

```bash
hbllm serve [OPTIONS]
```

**Options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--host` | `str` | `0.0.0.0` | Network interface to bind |
| `--port` | `int` | `8000` | Port to listen on |
| `--workers` | `int` | `1` | Number of worker processes |
| `--model-size` | `str` | `125m` | Native preset (`125m`, `500m`, `1.5b`) or HuggingFace repo |

**Example:**
```bash
hbllm serve --host 127.0.0.1 --port 8080 --model-size 500m
```

---

### `hbllm train`

Launches the distributed or local autoregressive pre-training loop with AdamW optimizer and cosine learning rate schedules.

```bash
hbllm train [OPTIONS]
```

**Options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model-size` | `str` | `125m` | Model architecture size (`125m`, `500m`, `1.5b`) |
| `--work-dir` | `str` | `./workspace` | Directory containing tokenized `.bin` shards |
| `--wandb-project` | `str` | `None` | Optional Weights & Biases project name for logging |

**Example:**
```bash
hbllm train --model-size 500m --work-dir ./data/shards --wandb-project my-hbllm-run
```

---

### `hbllm data`

Runs end-to-end dataset extraction, MinHash deduplication, Rust BPE tokenization, and binary shard generation.

```bash
hbllm data [OPTIONS]
```

**Options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--dataset` | `str` | `fineweb` | Dataset name from HuggingFace Hub |
| `--samples` | `int` | `100000` | Number of raw documents to process |
| `--vocab-size` | `int` | `32768` | Target tokenizer vocabulary size |
| `--seq-len` | `int` | `2048` | Target sequence length for training shards |
| `--work-dir` | `str` | `./workspace` | Output directory for dataset artifacts |

---

---

### `hbllm info`

Display system architecture, active zones, parameter presets, and memory tiers:

```bash
hbllm info
```

---

### `hbllm nodes`

List all 28+ loaded cognitive nodes across Perception, Brain, Memory, and Action zones:

```bash
hbllm nodes
```

---

### `hbllm plugin`

Manage dynamic hot-swappable plugins:

```bash
# List all discovered plugins in ./plugins directory
hbllm plugin list

# Scaffold a new plugin boilerplate with message bus subscription decorators
hbllm plugin new github_syncer
```

---

### `hbllm agent`

Launch the autonomous interactive developer terminal agent with tool sandboxing:

```bash
hbllm agent
```

---

### `hbllm code`

Launch the terminal pair-programming assistant for rapid inline refactoring:

```bash
hbllm code
```

---

## Standalone Subsystem CLIs

### Diagnostics CLI

Verify runtime dependencies, SIMD CPU instruction set detection, and native Rust acceleration crates:

```bash
python -m hbllm.cli.diagnostics
```

**Sample Output:**
```
============================================================
              HBLLM RUNTIME DIAGNOSTICS
============================================================
[Python Environment]
  Python Version:     3.11.8
  uvloop Active:      YES (High-performance event loop)
  SQLite Version:     3.45.1 (WAL & mmap supported)

[Native Rust Acceleration Crates]
  hbllm_compute:          LOADED (AVX-512 / NEON SIMD enabled)
  hbllm_tokenizer:        LOADED (Fast BPE tokenizer)
  hbllm_data_tools:       LOADED (MinHash deduplication)
  hbllm_semantic_search:  LOADED (Native SIMD vector index)
  hbllm_knowledge_graph:  LOADED (Native Graph Traversal)
  hbllm_policy_eval:      LOADED (Native Policy Engine)
  hbllm_confidence:       LOADED (Native Epistemic Calibrator)
============================================================
Status: ALL SYSTEMS OPERATIONAL
```

---

### DPO Export CLI

Export positive and negative preference pairs recorded in `ValueMemory` to JSONL datasets ready for Direct Preference Optimization training:

```bash
python -m hbllm.cli.export_dpo --output ./data/dpo_pairs.jsonl --tenant-id default
```

---

### Standalone Daemon & MCP Server

```bash
# Run continuous cognitive background daemon (AutonomyCore heartbeat)
python -m hbllm.serving.daemon --provider openai/gpt-4o-mini

# Run Model Context Protocol (MCP) server for IDE tools integration
python -m hbllm.serving.mcp_server
```
