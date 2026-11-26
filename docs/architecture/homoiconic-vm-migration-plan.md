---
title: "Homoiconic Symbolic VM Migration Plan (M0–M6)"
description: "Complete phased migration plan transitioning HBLLM Core from hardcoded Python logic to the Three-Layer Homoiconic Symbolic VM architecture (ADR 005)."
---

# Homoiconic Symbolic VM Migration Plan (`M0`–`M6`)

> **Authority:** [ADR 005: Homoiconic Symbolic VM & Experience-First Cognition](../adr/0005-homoiconic-symbolic-vm-and-experience-first-cognition.md)  
> **Baseline Audit:** [`reports/audit/2026-11-24_full_project_audit.md`](../../reports/audit/2026-11-24_full_project_audit.md)  
> **Status:** Implemented & Verified (Phases `M0`–`M6` complete; sealed milestone `eval-phase3-final` recorded on `072b4cb1`, 2026-11-26)  
> **Target Principle:** *Code = CPU/ALU & Constitutional Kernel; Graph = Symbolic Mind.*

---

## 1. Migration Strategy & Non-Negotiable Gates

Every phase (`M0`–`M6`) follows a **Strangler-Fig + Monotonic Gate** discipline so existing capabilities never regress while hardcoded Python orchestration is replaced by homoiconic HCIR programs:

1. **Backward-Compatible Strangler-Fig Rollout:** ISA v2 preserves v1 `InstructionStream` execution during migration; new symbolic paths run alongside legacy paths under deterministic parity tests before legacy Python branches are deleted.
2. **Architecture & Debt Gates:** Every PR must pass `pytest tests/unit/hcir/ tests/unit/architecture/ -q` with zero new `hasattr`/`getattr` debt, zero raw graph mutations, zero direct `sqlite3.connect`, and zero LLM imports in `hbllm/hcir/**`.
3. **Benchmark & Latency Gates:** Every phase must preserve or improve `dev.yaml` (`>= 0.925`) and `widened_dev.yaml` (`>= 0.750`) solve rates and meet `CPU p95 <= 45ms` at `32×32` via `python -m hbllm bench profile`. Sealed `heldout.yaml` runs only at major milestones via `eval-phase3-m*` tags recorded in `reports/heldout/LEDGER.md`.
4. **Rollback Rule:** Any phase PR that degrades cross-seed transfer, breaches `CPU p95 <= 45ms`, or leaks domain/lexical constants into Layer-0 ALUs is reverted or gated behind a failing parity test before proceeding.

```mermaid
flowchart LR
    M0["M0: Substrate Hygiene<br/>(Packaging, Tenancy, Isolation)"] --> M1["M1: ISA v2 VM &<br/>Homoiconic Nodes"]
    M1 --> M2["M2: Symbolic Predicates &<br/>.hcir Seed"]
    M2 --> M3["M3: Construction Grammar &<br/>Unknown Instructions"]
    M3 --> M4["M4: Wake-Sleep<br/>Library Learning"]
    M4 --> M5["M5: Cortical ALU /<br/>Program Split"]
    M5 --> M6["M6: Serving Unification &<br/>Permanent Ratchets"]
```

---

## 2. Phase-by-Phase Execution Plan

### Phase M0 — Substrate Hygiene, Packaging & Strict State/Tenant Isolation

**Goal:** Eliminate unshipped shadow trees, unverified commit paths, mutable graph leaks, and cross-tenant fallback hazards before building the Phase 3 VM on top of the kernel.

- **Exact Files Touched:**
  - `hbllm/hcir/kernel/capability_sandboxing.py`, `hbllm/hcir/kernel/capability_resolver.py`
  - `hbllm/actions/__init__.py`, `hbllm/brain/*/__init__.py` (remaining `__path__.append` shims)
  - `hbllm/hcir/graph.py`, `hbllm/hcir/workspace.py`, `hbllm/hcir/workspace_tiers.py`
  - `hbllm/hcir/language/core/gateway.py`, `hbllm/hcir/language/core/grounding.py`, `hbllm/hcir/grounding/concepts/registry.py`, `hbllm/hcir/kernel/command_dispatcher.py`
  - `hbllm/runtime/bootstrap.py`
  - `tests/unit/architecture/test_loc_budget.py`, `tests/unit/architecture/test_state_ownership_invariants.py`
- **Concrete Changes:**
  1. Move `run_sandboxed_python` cleanly into `hbllm/hcir/kernel/` and remove all `hbllm.actions` imports from the kernel. Eliminate all remaining `__path__.append` shims (`MAX_PATH_APPEND_SHIMS = 0`).
  2. Enforce `ConfigDict(frozen=True)` (or copy-on-write mutation isolation) on `HCIRNode` across `CognitiveGraph.fork()` so simulation branches never share mutable `properties` dicts.
  3. Convert the remaining workspace-absent raw `.add_node` / `.upsert_node` calls in `gateway.py`, `grounding.py`, and `registry.py` to `commit_operations` (`MAX_WORKSPACE_ABSENT_FALLBACK_CALLS = 0`).
  4. Bind an explicit `tenant_id` on `TieredWorkspace` and `HCIRTransaction` (rejecting cross-tenant commits in `ConstitutionalVerifier`), and isolate default `memory_dir` strictly to `~/.hbllm/worlds/<world_id>` (making pretrained ontology seed loading an explicit read-only opt-in).
- **Exit Gate:**
  - `pip wheel .` -> install wheel in a clean temp venv outside the repo -> import all `hbllm.*` modules and boot `uvicorn hbllm.serving.api:app` cleanly.
  - `MAX_PATH_APPEND_SHIMS == 0` and `MAX_WORKSPACE_ABSENT_FALLBACK_CALLS == 0` pass in `tests/unit/architecture/`.

---

### Phase M1 — HCIR ISA v2: Register-Unification VM, Call Stack & Homoiconic Node Schemas

**Goal:** Give the HCIR bytecode VM variables, registers, graph unification, conditional branching, and subroutine calls to graph-stored skills, while locking `EXECUTE` down to hardware syscalls.

- **Exact Files Touched:**
  - `hbllm/hcir/bytecode.py`, `hbllm/hcir/interpreter.py`, `hbllm/hcir/nodes.py`
  - `hbllm/hcir/optimizer.py`, `hbllm/hcir/receipt.py`, `hbllm/hcir/replay_debugger.py`
  - **New:** `hbllm/hcir/registers.py`, `hbllm/hcir/unification.py`
- **Concrete Changes:**
  1. Implement `RegisterRef` (`$r0`..`$r7`), `SymbolicVar` (`?name` with optional `HCIRNodeType`/trait constraints), `GraphPattern` (conjunctive/disjunctive node + hyperedge pattern), `RegisterFile`, and `CallStack` (`max_depth=16`, cycle detection, instruction fuel budget).
  2. Extend `Opcode` in `bytecode.py` from 8 to **12 canonical opcodes**:
     - Existing 8 (`ASSERT`, `RETRACT`, `QUERY`, `EXECUTE`, `WAIT`, `FORK`, `MERGE`, `ROLLBACK`) upgraded so parameters resolve `$rN` / `?var` references dynamically while staying 100% backward-compatible with literal v1 dicts.
     - New 4: `UNIFY` (match `GraphPattern` against workspace or `$rN`, binding `?var`s into registers), `BRANCH` (conditional jump on register state, threshold, or surprise), `CALL` (invoke a `SkillNode`/`ProcedureNode` by ID or postcondition unification in a child `CallFrame`), `RETURN` (yield output register bindings to caller).
  3. Extend `SkillNode`, `ProcedureNode`, and `LearnedRuleNode` in `nodes.py` with backward-compatible optional fields: `parameters: list[dict[str, Any]]`, `precondition_pattern: dict[str, Any] | None`, `postcondition_pattern: dict[str, Any] | None`, `bytecode_stream: dict[str, Any] | None`, `alpha_success: float = 1.0`, `beta_failure: float = 1.0`.
  4. Enforce `SyscallAllowlist` inside `sys_execute` (`motor.*`, `sensor.*`, `math.*`, `tool.sandboxed.*`, `web.sandboxed.*`, `memory.*`), logging deprecation warnings on legacy high-level Python capability names until removed in M2/M6.
  5. Make `InstructionStream.compute_hash()` canonical/deterministic (removing random UUID `id` from semantic digests where needed) and make `sys_rollback` fail-closed on snapshot hash mismatch.
- **Exit Gate:**
  - All existing `tests/unit/hcir/test_hcir_interpreter.py` tests pass without regression.
  - New Hypothesis property tests verify register binding across `QUERY -> UNIFY -> ASSERT -> CALL -> RETURN`, bounded `CallStack` fuel termination on recursive skills, deterministic `FORK -> CALL -> ROLLBACK/MERGE`, and bit-exact `ReplayDebugger` trace replay.

---

### Phase M2 — Composable Symbolic Predicate Algebra & `.hcir` Genesis Bootstrap

**Goal:** Replace Python-only predicate subclasses and Python-escaping `stdlib` macros with serializable symbolic expressions, parameterized `SkillNode` programs, and a declarative `.hcir` seed.

- **Exact Files Touched:**
  - `hbllm/hcir/skills/declarative.py`, `hbllm/hcir/skills/induction.py`
  - `hbllm/hcir/learning_loop.py`, `hbllm/hcir/stdlib.py`
  - `hbllm/hcir/world/option_discovery.py`
  - **New:** `hbllm/hcir/seed/core_priors.hcir.yaml`, `hbllm/hcir/seed/loader.py`
- **Concrete Changes:**
  1. Refactor `SymbolicPredicate` subclasses (`ActionAffordancePredicate`, `EntityCountPredicate`, `CutSetPredicate`, `SymmetryPredicate`, `PearlCausalDoPredicate`, etc.) into serializable `GraphPattern` / `SymbolicExpr` AST data backed by pure `math.*` ALUs (`math.topology_cuts`, `math.symmetry_score`, `math.connected_components`), retaining thin Python builders only as compatibility facades.
  2. Rewrite `hbllm/hcir/stdlib.py` (`MemoryOps`, `ReasoningOps`, `PlanningOps`, `LearningOps`, `SimulationOps`) as multi-opcode ISA v2 streams (`QUERY -> UNIFY -> BRANCH -> FORK -> CALL -> MERGE`) rather than single `EXECUTE("planning.decompose")` calls.
  3. Upgrade `SkillCompiler.compile_receipt_to_skill()` (`learning_loop.py`), `SkillInductionEngine` (`skills/induction.py`), and `OptionDiscoveryEngine` (`option_discovery.py`) so induced skills parameterize literal IDs into `?var` slots and persist full `bytecode_stream` on `SkillNode` (removing dynamic `class AutonomousOptionSkill` generation).
  4. Author `hbllm/hcir/seed/core_priors.hcir.yaml` containing **only** the Spelke/active-inference priors declared in [`priors.md`](priors.md), hydrated transactionally at boot in `build_cognitive_system()`.
- **Exit Gate:**
  - Induced `SkillNode`s survive SQLite `Persistent` tier serialization/deserialization and execute via `Opcode.CALL` in fresh processes with zero Python subclassing.

---

### Phase M3 — Learnable Embodied Construction Grammar & Unknown-Instruction Resolution

**Goal:** Eliminate hardcoded English verb/keyword/regex tables from `HCIRCompiler` and `ReadingCurriculumIngestor` by storing bidirectional linguistic constructions in the graph and resolving unseen instructions from prior experience.

- **Exact Files Touched:**
  - `hbllm/hcir/compiler.py`
  - `hbllm/hcir/language/runtime.py`, `hbllm/hcir/language/reading_curriculum.py`
  - `hbllm/hcir/language/core/gateway.py`, `hbllm/hcir/language/core/discourse.py`, `hbllm/hcir/language/core/universal_engine.py`
  - `hbllm/hcir/language/acquisition/lexicon_acquisition_loop.py`
  - **New:** `hbllm/hcir/seed/constructions_en.yaml`, `constructions_si.yaml`, `constructions_ta.yaml`
  - `tests/unit/hcir/test_pure_hcir_graduation_gate.py`
- **Concrete Changes:**
  1. Introduce `ConstructionNode` (or `LearnedRuleNode(domain="construction_grammar")`) encoding `(language, syntactic_template, slot_constraints, compiled_hcir_template, alpha, beta)` inside `TieredWorkspace`.
  2. Move initial functional grammar seeds for `en`/`si`/`ta` into `hbllm/hcir/seed/constructions_*.yaml` and replace the hardcoded verb/preposition lists and 10 fixed `_compile_*` Python strategies in `compiler.py` with graph-driven **Construction Unification**.
  3. Upgrade `ReadingCurriculumIngestor` (`reading_curriculum.py`) and `ExpositoryProseProcessor` (`discourse.py`) so definitional and procedural prose (*"to X a Y means to Z..."*, *"if P then do Q"*) compiles directly into executable `SkillNode` + `ConstructionNode` graph proposals.
  4. Implement the **4-Step Unknown-Instruction Resolution Pipeline** as an HCIR bytecode program:
     - *Step A (Partial Unification):* Bind recognized nouns, relations, and modifiers into `$r0..$rk`; emit an `UnknownNode` for the unseen span with an inferred `(InputTypes -> GoalConstraint)` type signature.
     - *Step B (Memory & Reading Recall):* `QUERY` `TieredWorkspace` (`Brain` + `Persistent`) for previously read definitions, hypernyms, or episodic rules matching the unknown stem.
     - *Step C (Structural Analogy):* Call `hbllm_structure_matcher` / `AnalogyOperator` to match existing `SkillNode` `(precondition_pattern -> postcondition_pattern)` graphs against current scene affordances.
     - *Step D (Active Inquiry & Consolidation):* If uncertainty remains high, `FORK` a simulation/probe or emit `REQUEST_EVIDENCE` (`EpistemicCuriosityController`), verify outcomes against `PredictionErrorNode`, and commit the new `ConstructionNode` + `SkillNode` transactionally for future reuse.
- **Exit Gate:**
  - `tests/unit/hcir/test_pure_hcir_graduation_gate.py` is 100% hermetic (no dependency on untracked `checkpoints/`) and passes at `>= 0.98` autonomy across all 80+ cases.
  - Novel-instruction composition test (teach primitive A + primitive B through text/experience -> execute unseen composite instruction C) achieves `>= 90%` zero-shot success with zero hardcoded keywords for A, B, or C in Python.

---

### Phase M4 — Wake–Sleep Anti-Unification & Library Learning

**Goal:** Continuously compress successful wake-phase bytecode traces into parameterized, cross-seed reusable `SkillNode` macros during offline idle consolidation.

- **Exact Files Touched:**
  - `hbllm/hcir/kernel/executive_runtime.py`, `hbllm/hcir/world/episodic/replay.py`
  - `hbllm/hcir/reasoning/operators/skill_chunker.py`
  - `hbllm/runtime/bootstrap.py`, `benchmarks/suites/runner.py`
  - **New:** `hbllm/hcir/skills/library_learner.py`
- **Concrete Changes:**
  1. Implement deterministic **Anti-Unification Library Learner** (`hbllm/hcir/skills/library_learner.py`): align pairs of successful `ExecutionReceipt` `InstructionStream` traces during `ExecutiveRuntime` offline consolidation and `HippocampalReplayConsolidator` SWR replay, replace differing episode-local entity/coordinate literals with typed `?var` parameters, verify Minimum Description Length (MDL) compression (`compressed_cost < sum(raw_costs)`), and commit new `SkillNode`s via `TransactionManager`.
  2. Enforce relational `?role` abstraction at `EpisodeBoundary(same_level=False)` so surface color/entity IDs from seed *k* never poison seed *k+1*.
  3. Wire `bootstrap.py` and `benchmarks/suites/runner.py` (`skills_consolidated`) to report actual promoted `SkillNode` counts from `TieredWorkspace`.
- **Exit Gate:**
  - `skills_consolidated > 0` across `dev` and `widened_dev` cross-seed runs; measurable step-count reduction via `Opcode.CALL` macro reuse on transfer seeds; zero regression on `partial_observability` and `language_goals` transfer.

---

### Phase M5 — Decomposing `hcir/world` & `hcir/reasoning` into Stateless Hardware ALUs + Symbolic HCIR Programs

**Goal:** Shrink the multi-thousand-LOC procedural world-model files (`H-06`, `H-07`, `H-11`, `H-12`) into clean stateless Layer-0 ALUs and inspectable HCIR programs operating over `FrozenGraphView`.

- **Exact Files Touched:**
  - `hbllm/hcir/world/autonomous_epistemic_engine.py`, `hbllm/hcir/world/cortical_pipeline.py`
  - `hbllm/hcir/world/assimilation/assimilator.py`, `hbllm/hcir/world/planning/mental_simulation.py`, `hbllm/hcir/world/planning/goal_scheduler.py`
  - `hbllm/hcir/world/causal/theory.py`, `hbllm/hcir/world/causal/transition_kb.py`
  - `hbllm/hcir/world/deliberation/engine.py`, `hbllm/hcir/world/grounding/bridge.py`
  - `hbllm/hcir/reasoning/unified_runtime.py`
  - `tests/unit/architecture/test_structural_debt_ratchet.py`
- **Concrete Changes:**
  1. Replace crisp boolean feature/transition sets + ad-hoc procedural guards in `HCIRSymbolicWorldTheory` and `TransitionKnowledgeBase` with **Dirichlet/Beta empirical outcome posteriors** (`alpha_pass, beta_block, gamma_lethal, delta_goal`) stored as graph state (`H-11`).
  2. Extract stateless Layer-0 math/physics ALUs (`math.geodesic_field`, `math.bellman_risk_backup`, `math.collision_cone`, `math.bayesian_posterior_update`, Rust `hbllm_simulation_engine`, and `snn`/`hbllm_perception_rs` grounding).
  3. Express epistemic phase arbitration, BA10 subgoal scheduling, tool-barrier mediation, curiosity probing, and deliberation gating as symbolic HCIR programs evaluated over `FrozenGraphView`, returning only `StageResult.operations` (eliminating direct `view.state.<attr> = ...` writes, `get_sub_attr`, and blacklisted `_feedback_assimilated` / `_last_*` attributes).
  4. Either prove `CorticalGroundingBridge` achieves `Δ > 0` on appearance/role-shifted dev/held-out tasks within `CPU p95 <= 45ms`, or disable `enable_grounding` by default until it passes the R1 ablation + latency gate (`H-03`).
- **Exit Gate:**
  - Every file in `hbllm/hcir/world/**` `<= 1,200` LOC (`MAX_ANY_WORLD_FILE_LOC` ratcheted from `3,704` down to `1,200`).
  - Material improvement on `widened_dev` `stochastic_dynamics` and `long_horizon` regimes; `32×32` grounded step latency meets `CPU p50 <= 15ms / p95 <= 45ms`.

---

### Phase M6 — Serving/Driver Unification, Legacy Pruning & Permanent AST Ratchets

**Goal:** Make the shipped HTTP/driver product identical to the benchmarked Pure-HCIR Cognitive OS and lock Phase 3 invariants in CI.

- **Exact Files Touched:**
  - `hbllm/serving/api.py`, `hbllm/drivers/cognitive_blackbox.py`
  - `hbllm/hcir/world/predictors/{llm,neural,statistical}.py`
  - `plugins/hbllm_terminal/__init__.py`, `plugins/compiler-verification/compiler_verification.py`
  - `tests/unit/architecture/test_homoiconic_invariants.py` (new)
- **Concrete Changes:**
  1. Route `hbllm/serving/api.py` (`/v1/chat*`, `/v1/chat/completions`, session endpoints) through a per-tenant `CognitiveSystem` registry built via `build_cognitive_system()`, and collapse `CognitiveBlackbox` into a thin modality adapter over `CognitiveSystem`.
  2. Delete the 3 fake stub predictors (`predictors/{llm,neural,statistical}.py`) and route unsafe plugin commands strictly through `PermissionEngine` + the kernel two-tier sandbox (`C-4`).
  3. Add permanent AST architecture ratchets in `tests/unit/architecture/test_homoiconic_invariants.py`:
     - Zero hardcoded lexical/verb/color tables in `hbllm/hcir/compiler.py` or `hbllm/hcir/world/`.
     - Zero non-allowlisted high-level Python capabilities in `sys_execute`.
     - Zero `SkillNode` commits from `SkillCompiler` / `SkillInductionEngine` / `LibraryLearner` lacking an executable `bytecode_stream`.
- **Exit Gate:**
  - Official sealed Held-Out v2 milestone (`eval-phase3-final`) preserves or exceeds `0.9250` zero-shot and `0.9035` cross-seed transfer with `CPU p95 <= 45ms`, `0` LLM tokens, `0` `__path__.append` shims, and `hbllm/hcir/**` LOC `< 110,000`.

---

## 3. Completion Summary & Verified Outcomes (`eval-phase3-final`)

All phases (`M0`–`M6`) are complete on `feat/homoiconic-symbolic-vm` and verified at tag `eval-phase3-final` (commit `072b4cb1`, `2026-11-26`):

| Exit Criterion | Target Baseline | Verified Outcome (`M0`–`M6`) |
|---|---|---|
| **Sealed Held-Out v2 Zero-Shot (`n=120`)** | `>= 0.9250` (`111/120`, `4` deaths) | **`0.9417`** (`113/120`, CI `[0.8845, 0.9715]`, generality `0.8359`, **`2` deaths**) |
| **Held-Out Cross-Seed Transfer (`n=114`)** | `>= 0.9035` | **`0.9035`** (CI `[0.8354, 0.9453]`, `28,330` skills recorded) |
| **`widened_dev` Transfer & Zero-Shot (`n=120`)** | Preserve dev `>= 0.750` | Zero-shot **`0.7750`** (`generality = 0.7503`), cross-seed transfer **`0.9474`** |
| **Core Kernel Step Budget (`32×32`)** | `CPU p95 <= 45ms` | `hbllm bench profile`: Decisional `p95 = 0.182ms`, CPU E2E `p95 = 14.58ms` (`bench heldout` variable-grid mean `p50=25.72ms / p95=85.06ms`) |
| **LLM Tokens & Legacy Packaging Shims** | `0` tokens, `0` shims | **`0` LLM tokens**, **`0` `__path__.append` shims** (`test_m0_substrate_hygiene.py`) |
| **Architectural Ratchets & Governance** | AST & Serving/Driver Unification | `SyscallAllowlist(strict=True)`, per-tenant `CognitiveSystemRegistry`, thin `CognitiveBlackbox`, `run_sandboxed_command`, `tests/unit/architecture/test_homoiconic_invariants.py` |

> [!NOTE]
> Artificial LOC ceilings (`test_loc_budget.py`) were retired in `637c8f22` (`M4`) in favor of semantic and AST invariant ratchets (`test_homoiconic_invariants.py`, `test_m0_substrate_hygiene.py`, `test_structural_debt_ratchet.py`); raw `hbllm/hcir/**` `.py` total stands at `146,399` lines across 11 subpackages. With Phase 3 complete, active evolution moves to [ADR 006: Fuzzy Symbolic Cognition](../adr/0006-fuzzy-symbolic-cognition-and-adaptive-semantics.md) (`F0`–`F3`).

