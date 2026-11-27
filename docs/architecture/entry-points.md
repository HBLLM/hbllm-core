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

1. **Workspace, Tenant Scope & Event Store Initialization:**
   - Resolves `memory_dir` and creates `SQLiteEventStore(..., max_events=cfg.max_events_per_store)` when persisted.
   - Instantiates `TieredWorkspace` (`Working`, `Brain`, `Persistent`, `Meta`) bound to `tenant_id` and unifies `services.workspace = tiered_workspace.working_state`.
2. **Constitutional Governance:**
   - Instantiates `CommitAuthorityRegistry`, `ConstitutionalVerifier`, and the 6-stage verification pipeline (`create_default_pipeline`, scoped via `Scope(tenant_id=cfg.tenant_id)`).
3. **Kernel Services (`KernelServices.create_default`) & Innate Seed Hydration:**
   - Assembles `TransactionManager`, `CapabilityResolver`, `CapabilitySandboxManager`, `KernelEventBus`, `KernelInstructionScheduler`, `KernelExecutor`, `KernelTelemetry`, `WorldKernel`, and registers hardware/sandboxed capabilities (`motor.*`, `sensor.*`, `math.*`, `world.*`, `fs.*`, `memory.*`, `web.*`).
   - When `load_core_priors=True`, `hydrate_core_priors()` (`hbllm.hcir.seed.loader`) commits the minimal Spelke/active-inference seed rules (`core_priors.hcir.yaml` + `constructions_{en,si,ta}.yaml`) marked with `SecurityLevel.SYSTEM` / `"core_prior"`.
4. **Autonomous Epistemic Engine & Hydration:**
   - Instantiates `AutonomousEpistemicEngine` with shared `KernelServices` and `TieredWorkspace`, hydrating prior lifelong world nodes (`hydrate_from_workspace()`).
5. **Cortical Grounding Bridge (Optional):**
   - When `enable_grounding=True`, instantiates `CorticalGroundingBridge` sharing the exact `KernelServices` and attaches it to the engine.
6. **Reasoning, SNN Neuromodulation, Language & Executive Runtime:**
   - Wires `UnifiedReasoningRuntime` (14 default operators), `NeuromodulationEngine`, `LexiconAcquisitionLoop`, optional `EpistemicCuriosityController`, and `ExecutiveRuntime` (with Wake–Sleep `LibraryLearner` offline consolidation).

### 2.2 `CognitiveSystem.process_dialogue()`

`system.process_dialogue(dialogue, speaker_id="user", language=None)` runs the Pure-HCIR multilingual language loop (`MultilingualLanguageRuntime` + `ConstructionNode` unification → `DiscourseSegmenter` → `ExpositoryProseProcessor` → 4-step unknown-instruction resolution / optional `EpistemicCuriosityController` → governed `CommandDispatcher`), returning a calibrated `LanguageTurnResult`.

### 2.3 Serving & Driver Entry Points

- **`CognitiveSystemRegistry` (`hbllm.serving.cognitive_registry`):** Lazily provisions one isolated `CognitiveSystem` per `tenant_id` via `build_cognitive_system()`, backing `/v1/chat`, `/v1/chat/stream`, `/v1/chat/completions`, `/v1/chat/ws`, `GET /v1/memory/{session_id}`, and `GET /v1/rules` Pure-HCIR first (`X-HBLLM-Engine: pure-hcir`) before optional external `serving/provider.py` fallback.
- **`CognitiveBlackbox` (`hbllm.drivers.cognitive_blackbox`):** Thin modality adapter wrapping per-source `CognitiveSystem` instances built via `build_cognitive_system()`, delegating `step()` / `adapt_to_observation()` / `reset()` directly into `AutonomousEpistemicEngine` with explicit `tenant_id` scoping.

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
| | `hbllm info` | Displays the 9 architectural invariants and 5-stage cortical loop |
| | `hbllm serve [--host ... --port ... --workers ...]` | Starts the FastAPI REST + MCP server (`uvicorn hbllm.serving.api:app`) |
| | `hbllm agent` / `hbllm code "<prompt>"` | Developer CLI utilities |
| | `hbllm daemon [run\|install\|uninstall\|status]` | Background daemon lifecycle management |

---

## 4. Architectural Layering Guarantees

AST-level dependency and Phase 3 homoiconic boundary checks are enforced automatically in `tests/unit/architecture/`:
- `test_layering.py`: `hbllm.hcir` never imports from `hbllm.serving`; core `hbllm` never imports `plugins/`.
- `test_m0_substrate_hygiene.py`: Enforces **zero** LLM client/scaffold imports in `hbllm/hcir/**`, **0** `__path__.append` shadow shims (`MAX_PATH_APPEND_SHIMS = 0`), shipped-wheel module resolution, and `0` workspace-absent commit fallbacks.
- `test_homoiconic_invariants.py`: Permanently bars hardcoded lexical/color keyword tables in `compiler.py` and `world/`, blocks unallowlisted `EXECUTE` capabilities (`SyscallAllowlist(strict=True)`), enforces executable `bytecode_stream`s on all `SkillNode` producers, verifies serving/driver Pure-HCIR routing, and verifies system-prior budget preservation.
- `test_single_composition_root.py` & `test_no_direct_sqlite.py`: Confirms runtime entry points compose through `build_cognitive_system()` and raw SQLite connections remain confined to `hbllm.persistence`, `hbllm.hcir.stores`, `hbllm.security`, and `hbllm.network.durable_bus`.
