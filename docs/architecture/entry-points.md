---
title: "System Entry Points & Bootstrapping Architecture"
description: "Authoritative guide to HBLLM's single composition root (build_cognitive_system), CognitiveSystemConfig, dialogue processing, and unified Typer CLI."
---

# System Entry Points & Bootstrapping Architecture

> **Document Status:** Authoritative Architectural Specification  
> **Subsystem:** `hbllm.runtime.bootstrap`, `hbllm.cli`, `hbllm.__main__`  
> **Invariants:** 1 (State Ownership), 4 (Constitutional Governance), 7 (Domain-Agnostic Core)

---

## 1. Overview

HBLLM enforces a **single unified composition root** across all runtime contexts. Whether invoked via the Typer CLI, benchmark harnesses, dialogue scripts, or integration tests, the cognitive system is initialized through `build_cognitive_system()`.

Duplicate kernel initializations, unverified `TransactionManager` instances, and ad-hoc wiring scripts outside `hbllm.runtime.bootstrap` are prohibited (enforced by `tests/unit/architecture/test_single_composition_root.py`).

---

## 2. The Single Composition Root (`hbllm.runtime.bootstrap`)

```python
from hbllm.runtime.bootstrap import (
    CognitiveSystem,
    CognitiveSystemConfig,
    build_cognitive_system,
)

config = CognitiveSystemConfig(
    world_id="primary",
    seed=42,
    exploration_budget=150,
    enable_logging=True,
    memory_dir=None,  # None -> in-memory; default -> ~/.hbllm/worlds/<world_id>
    enable_grounding=True,
    perception_mode="handcrafted",  # "handcrafted" | "snn" | "learned"
    enable_brain_llm=False,
    enable_autonomous_curiosity=False,
    max_events_per_store=2000,
)

system: CognitiveSystem = build_cognitive_system(config)
```

### 2.1 Initialization Lifecycle

1. **Workspace & Event Store Initialization:**
   - Resolves `memory_dir` and creates `SQLiteEventStore(..., max_events=cfg.max_events_per_store)` when persisted.
   - Instantiates `TieredWorkspace` (`Working`, `Brain`, `Persistent`, `Meta`) and unifies `services.workspace = tiered_workspace.working_state`.
2. **Constitutional Governance:**
   - Instantiates `CommitAuthorityRegistry`, `ConstitutionalVerifier`, and the 6-stage verification pipeline (`create_default_pipeline`).
3. **Kernel Services (`KernelServices.create_default`):**
   - Assembles `TransactionManager`, `CapabilityResolver`, `CapabilitySandboxManager`, `KernelEventBus`, `KernelInstructionScheduler`, `KernelExecutor`, `KernelTelemetry`, `WorldKernel`, and registers all governed kernel capabilities (`world.*`, `planning.*`, `routing.*`, `fs.*`, `code.python_ast`, `git.status`, `shell.exec`, `memory.*`, `knowledge.*`, `web.*`).
4. **Autonomous Epistemic Engine & Hydration:**
   - Instantiates `AutonomousEpistemicEngine` with shared `KernelServices` and `TieredWorkspace`, hydrating prior lifelong world nodes (`hydrate_from_workspace()`).
5. **Cortical Grounding Bridge (Optional):**
   - When `enable_grounding=True`, instantiates `CorticalGroundingBridge` sharing the exact `KernelServices` and attaches it to the engine.
6. **Reasoning, SNN Neuromodulation, Language & Executive Runtime:**
   - Wires `UnifiedReasoningRuntime` (14 default operators), `NeuromodulationEngine`, `LexiconAcquisitionLoop`, optional `EpistemicCuriosityController`, and `ExecutiveRuntime`.

### 2.2 `CognitiveSystem.process_dialogue()`

`system.process_dialogue(dialogue, speaker_id="user", language=None)` runs the Pure-HCIR multilingual language loop (`MultilingualLanguageRuntime` → `DiscourseSegmenter` → `ExpositoryProseProcessor` → optional `EpistemicCuriosityController` → governed `CommandDispatcher`), returning a calibrated `LanguageTurnResult`.

---

## 3. Unified Typer CLI (`hbllm.cli`)

Executable via `python -m hbllm [COMMAND] [ARGS]...` or `hbllm [COMMAND] [ARGS]...`:

| Command Group | Command | Description |
|---|---|---|
| **`kernel`** | `hbllm kernel boot [--world-id ...]` | Bootstraps the HCIR kernel and verifies `TransactionManager` & stages |
| | `hbllm kernel status` | Queries live verification status, 4 memory tiers, and native Rust substrate health |
| | `hbllm kernel replay [--log ...]` | Replays transactional event streams |
| **`mind`** | `hbllm mind run [--driver ... --budget ...]` | Runs the active-inference loop and persists state |
| | `hbllm mind step --obs <file.json> [--explain]` | Executes a deterministic single-cycle `step()` with optional provenance explanation |
| **`bench`** | `hbllm bench dev` | Runs frozen `benchmarks/suites/dev.yaml` evaluation suite |
| | `hbllm bench heldout` | Runs anti-tamper Held-Out v2 evaluation suite (`eval-*` git tag gated) |
| | `hbllm bench local-suite` | Runs the 10-game ARC local suite benchmark |
| | `hbllm bench ablation` | Measures cortical faculty ablation impact |
| | `hbllm bench sge` | Benchmarks Symbol Grounding Engine integration |
| | `hbllm bench profile` | Profiles substrate & full `32×32` 5-stage cognitive step latency |
| **`plugin`** | `hbllm plugin list` / `hbllm plugin new <name>` | Lists or scaffolds plugins under `plugins/` |
| **Top-level** | `hbllm diagnostics` | Full runtime, Rust SIMD, and Cognitive OS check |
| | `hbllm info` | Displays the 8 architectural invariants and 5-stage cortical loop |
| | `hbllm serve [--host ... --port ... --workers ...]` | Starts the FastAPI REST + MCP server (`uvicorn hbllm.serving.api:app`) |
| | `hbllm agent` / `hbllm code "<prompt>"` | Developer CLI utilities |
| | `hbllm daemon [run\|install\|uninstall\|status]` | Background daemon lifecycle management |

---

## 4. Architectural Layering Guarantees

AST-level dependency boundaries are enforced automatically in `tests/unit/architecture/`:
- `test_layering.py`: `hbllm.hcir` never imports `hbllm.brain`; core `hbllm` never imports `plugins`.
- `test_loc_budget.py`: Enforces **zero** LLM client/scaffold imports in `hbllm/hcir/**`, zero `__path__.append` shadow shims, and shipped internal module resolution.
- `test_no_direct_sqlite.py`: Direct `sqlite3.connect` / `aiosqlite.connect` calls are confined strictly to infrastructure persistence modules (`hbllm.persistence`, `hbllm.hcir.stores`, `hbllm.security`, `hbllm.network.durable_bus`).
