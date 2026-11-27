---
title: "CLI Reference — Complete Command-Line Manual"
description: "Comprehensive guide to the HBLLM Typer CLI (hbllm): kernel, mind, bench, plugin, diagnostics, info, serve, agent, code, and daemon."
---

# CLI Reference Manual

HBLLM provides a unified Typer command-line interface (`hbllm` / `python -m hbllm`) defined in `hbllm.cli.main` for booting and inspecting the HCIR kernel, stepping the active-inference mind, running frozen benchmark suites, managing plugins, and launching the serving API.

---

## Command Overview

```bash
hbllm [command] [options]
```

| Command Group / Command | Purpose |
|---|---|
| [`hbllm kernel`](#hbllm-kernel) | `boot`, `status`, and `replay` for the HCIR Cognitive OS kernel |
| [`hbllm hcir`](#hbllm-hcir) | `inspect`, `print`, `validate`, and `roundtrip` canonical `.hcir` artifacts |
| [`hbllm mind`](#hbllm-mind) | `run` active inference loop or `step` a single JSON observation |
| [`hbllm bench`](#hbllm-bench) | Run `dev`, `heldout`, `local-suite`, `sge`, `ablation`, and `profile` suites |
| [`hbllm plugin`](#hbllm-plugin) | `list` discovered plugins or scaffold a `new` plugin |
| [`hbllm diagnostics`](#hbllm-diagnostics) | System runtime health, `uvloop`, SQLite, and 13 Rust crates check |
| [`hbllm info`](#hbllm-info) | Print the 8 architectural invariants and 5-stage cortical loop |
| [`hbllm serve`](#hbllm-serve) | Start the FastAPI REST and MCP server (`hbllm.serving.api:app`) |
| [`hbllm agent` / `hbllm code`](#hbllm-agent-and-hbllm-code) | Developer terminal utilities |
| [`hbllm daemon`](#hbllm-daemon) | Run or manage the background daemon service (`run`, `install`, `uninstall`, `status`) |

---

## Pure-HCIR Core Commands

### `hbllm kernel`

Manage and inspect the verified HCIR kernel (`build_cognitive_system()`):

```bash
# Boot verified HCIR kernel for a world context
hbllm kernel boot --world-id default

# Inspect verified TransactionManager stages, 4 tiers & native Rust substrate health
hbllm kernel status

# Replay transactions from an event log
hbllm kernel replay --log path/to/events.log
```

### `hbllm hcir`

Inspect cognitive nodes and execution traces, validate `.hcir` files, and verify canonical round-trips:

```bash
# Inspect a SkillNode from the workspace in canonical .hcir with BLAKE3 digests
hbllm hcir inspect skill prior_skill_geodesic_field --digest

# Print an executable InstructionStream trace from a node or receipt JSON
hbllm hcir print trace prior_skill_collision_cone

# Validate syntax, Rule-0 compliance, and core_priors domain purity
hbllm hcir validate hbllm/hcir/seed/core_priors.hcir

# Verify canonical idempotence and semantic digest stability
hbllm hcir roundtrip hbllm/hcir/seed/core_priors.hcir
```

### `hbllm mind`

Run or step `AutonomousEpistemicEngine` over `EmbodiedObservation` inputs:

```bash
# Run active inference loop with a driver
hbllm mind run --driver mock --budget 150 --world-id default --seed 42

# Execute a single deterministic decide() step with full provenance explanation
hbllm mind step --obs tests/fixtures/obs_minimal.json --world-id default --explain
```

### `hbllm bench`

Execute frozen evaluation suites, ablations, and latency profilers:

```bash
# Frozen dev evaluation suite (perception: handcrafted | snn | learned)
hbllm bench dev --perception handcrafted --episodes 5 --shared-memory --cross-seed

# Evaluate ADR 006 2x2 fuzzy grounding & fuzzy reasoning axes
hbllm bench dev --fuzzy-grounding --fuzzy-reasoning

# Sealed Held-Out v2 suite (anti-tamper: requires clean HEAD with eval-* git tag)
hbllm bench heldout --perception handcrafted

# 10-game ARC local suite, SGE benchmark, and faculty ablation
hbllm bench local-suite --game ls20 --seed 0
hbllm bench sge
hbllm bench ablation

# Profile hot paths and 32x32 5-stage CorticalPipeline step latency (p95 <= 45ms gate)
hbllm bench profile
```

### `hbllm diagnostics`

Verify runtime dependencies, `uvloop`, SQLite support, and availability of the 13 Rust acceleration crates (`NativeAccelerationRegistry`):

```bash
hbllm diagnostics
# or
python -m hbllm.cli.diagnostics
```

### `hbllm info`

Print version, the 8 non-negotiable architectural invariants, and the 5-stage cortical loop:

```bash
hbllm info
```

---

## Runtime, Plugins & Developer Commands

### `hbllm plugin`

```bash
# List discovered plugins under plugins/
hbllm plugin list

# Scaffold a new plugin skeleton
hbllm plugin new my_adapter
```

### `hbllm serve`

Starts the FastAPI REST server (`uvicorn hbllm.serving.api:app` with `uvloop`; requires `pip install ".[server]"`):

```bash
hbllm serve --host 0.0.0.0 --port 8000 --workers 1
```

### `hbllm daemon`

```bash
hbllm daemon run        # Run in foreground
hbllm daemon status     # Check service status
hbllm daemon install    # Install OS service
hbllm daemon uninstall  # Remove OS service
```

### `hbllm agent` and `hbllm code`

```bash
hbllm agent
hbllm code "Summarize module structure"
```

### Legacy Standalone Utilities

The following standalone CLIs remain available for legacy workflows:
- `python -m hbllm.serving.mcp_server` — MCP stdio tool server
- `python -m hbllm.cli.export_dpo --output ./data/dpo_pairs.jsonl` — Legacy preference pair export
- `hbllm train` / `hbllm data` — Legacy pre-training and data pipeline entry points
