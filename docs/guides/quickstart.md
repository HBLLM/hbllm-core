---
title: "Quick Start Guide — HBLLM Core"
description: "Install and run HBLLM Core (Pure-HCIR Cognitive OS) in under 5 minutes on any hardware — no GPU required."
---

# Quick Start

Get HBLLM Core running in under 5 minutes with zero cloud or GPU requirements.

## Prerequisites

- **Python 3.11+**
- `pip` or `uv`
- **No GPU required** — the Pure-HCIR Cognitive OS runs deterministically on CPU.
- *(Optional)* Rust toolchain + `maturin` for native acceleration crates (`rust/*`).
- *(Optional)* `.[server]` extra for the FastAPI REST + MCP server.

## Installation

```bash
# Clone the repository
git clone https://github.com/hbllm/hbllm-core.git
cd HBLLM/core

# Install core Pure-HCIR runtime in editable mode
pip install -e .

# Optional extras:
# pip install -e ".[server]"   # FastAPI, Uvicorn, PyJWT, WebSockets
# pip install -e ".[dev]"      # Server + storage + pytest/ruff/mypy
```

*(Optional)* Build native Rust acceleration wheels:

```bash
pip install maturin
for crate in rust/*/; do
  [ -f "$crate/Cargo.toml" ] && maturin develop --manifest-path "$crate/Cargo.toml" --release
done
```

## Verify Installation

```bash
# Inspect runtime, Rust crates, and Cognitive OS status
hbllm diagnostics

# View architectural invariants & 5-stage cortical loop
hbllm info

# Check verified HCIR kernel, transaction stages, and 4 memory tiers
hbllm kernel status

# Run fast HCIR & architecture invariant tests
pytest tests/unit/hcir/ tests/unit/architecture/ -q
```

## Python SDK (Single Composition Root)

All runtime entry points boot through `build_cognitive_system()` (`hbllm.runtime.bootstrap`):

```python
from hbllm.hcir.world.percept import EmbodiedObservation
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

# 1. Boot verified Pure-HCIR Cognitive OS
system = build_cognitive_system(
    CognitiveSystemConfig(
        world_id="quickstart",
        memory_dir=None,  # In-memory workspace (omit for ~/.hbllm/worlds/<world_id>)
        enable_grounding=True,
        perception_mode="handcrafted",  # "handcrafted" | "snn" | "learned"
    )
)

# 2. LLM-free dialogue, prose ingestion, and calibrated query answering
system.process_dialogue("Photosynthesis converts sunlight into chemical energy.")
turn = system.process_dialogue("What does photosynthesis convert?")
print(turn.response_text)

# 3. Deterministic 5-stage cortical active-inference step
obs = EmbodiedObservation(
    lattice=[[0, 0, 0], [0, 1, 0], [0, 0, 2]],
    available_actions=[1, 2, 3, 4],
    context_id="quickstart",
    episode_id="ep_0",
)
action, meta = system.engine.step(obs)
print(f"Chosen action: {action}, Phase: {meta.get('phase')}")
```

## CLI Workflows

```bash
hbllm kernel boot --world-id my_world       # Boot verified HCIR kernel
hbllm mind step --obs tests/fixtures/obs_minimal.json --explain
hbllm bench dev                             # Run frozen dev benchmark suite
hbllm bench profile                         # Profile Rust/Python step latency
hbllm plugin list                           # List installed plugins in plugins/
hbllm serve --host 127.0.0.1 --port 8000    # Start FastAPI + MCP Server (requires .[server])
```

## Next Steps

- [Architecture Overview](../architecture/overview.md) — Pure-HCIR invariants and layered topology.
- [Cortical World Model](../architecture/cortical-world-model.md) — 5-stage epistemic loop & `TransitionKnowledgeBase`.
- [Language & Reading](../architecture/language-and-reading.md) — Multilingual parser/realizer & Track T4 comprehension.
- [CLI Reference Manual](cli.md) — Complete CLI flags and subcommands.
- [Deployment Guide](deployment.md) — Docker, production auth, and Kubernetes setup.
