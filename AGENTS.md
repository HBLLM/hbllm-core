# HBLLM — Project Instructions & Cognitive Architecture Guide

> **Primary Directive for AI Agents:**  
> This file is the authoritative reference for working on the HBLLM codebase. Read and uphold all architectural invariants, neuro-symbolic design rules, directory conventions, and engineering standards defined herein.

> [!WARNING]
> **Current project phase: Phase 2 — Pure-HCIR Cognitive OS Consolidation (verified 2026-11-21, R1–R10 complete).**  
> R1–R8 established honest all-seed `dev` evaluation (`n=80`, `zero_shot=0.7791`, `solve_rate=0.8750`, CI `[0.7850, 0.9307]`, `cross_seed transfer_solve_rate=0.9342`, `exploration_deaths=3`, mean latency `p50=10.18ms / p95=28.12ms`, `reports/dev/dev_handcrafted_20261116_180441.json`), sealed held-out v2 with `eval-*` gate & `reports/heldout/LEDGER.md`, retired held-out v1, promoted `novelty.yaml` (`zero_shot=0.8365`, `solve_rate=0.9000`, `p50=11.09ms / p95=33.53ms`), wired grounding into runtime decisions, deleted 28 zero-effect/dead faculties + unwired dead hooks, enforced structural debt + `<= 300,000` LOC ratchets (`<= 150,000` LOC for `hbllm/hcir/**`), added risk-aware exploration, implemented bounded event-store compaction + per-tier/world node budgets + per-stage latency profiling (`p50 ~ 10-16ms / p95 <= 45ms` at `32×32`, Rust kernels `< 0.2ms`).
>
> **R9–R10 & Pure-HCIR Consolidation:**
> - **R9** lifted widened-regime performance across 6 regimes (`reports/dev/widened_dev_after_r9_20261117.json`: `zero_shot` solve rate `0.3167 → 0.5083`, `cross_seed transfer_solve_rate` `0.4123 → 0.7456`, deaths `54 → 33`), reaching `zero_shot=0.6083` (`73/120`, generality `0.6472`) on `eval-phase2-r9`.
> - **Official Sealed Held-Out v2 Milestone (`eval-phase2-r10` on `a26f09bd`, 2026-11-21, `n=120`, 20 seeds/task, `reports/heldout/LEDGER.md`):** `zero_shot` solve rate = **0.9250** (`111/120` solved, CI `[0.8636, 0.9600]`, generality = **0.8236**, 4 exploration deaths / `3.33%`); `cross_seed transfer_solve_rate` = **0.9035** (`114` transfer episodes, CI `[0.8354, 0.9453]`, `skills_consolidated=161`); `same_layout_replay` solve rate = **1.0000** (generality = **0.9021**). Per-family zero-shot: `continuous=1.00`, `symbolic=1.00`, `grid=0.95`, `stochastic_dynamic=0.85`, `partial_observability=0.80`.
> - **Pure-HCIR Core Consolidation:** Consolidated `hcir/reasoning` (14 default operators), `hcir/language` (A16/A17 multilingual `en`/`si`/`ta` + Track T4 `ReadingCurriculumIngestor`), `hcir/snn`, and `hcir/capabilities` (`ConceptExplainer`, `EpistemicCuriosityController`, `BeliefConflictResolver`, sandboxed `Tool`/`Web`/`Memory` executors) directly under `hbllm/hcir/`; modularized `hbllm/hcir/world/` into 17 subpackages with HCIR-backed `TransitionKnowledgeBase` as single source of truth; unified HTTP serving in `Dockerfile`. Production gaps & roadmap are tracked in `reports/audit/2026-11-18_full_project_audit.md`.

---

## 1. Project Vision & Core Invariants

HBLLM is a **human-brain-inspired, event-sourced Cognitive OS and neuro-symbolic AI architecture**. Intelligence is modeled not as a monolithic black-box transformer, but as specialized cognitive subsystems (nodes, cortices, and operators) executing over a formal state representation (**HCIR**).

### Non-Negotiable Invariants

1. **State Ownership Invariant:**
   > *"Nodes emit intent. HCIR owns state."*  
   No brain node, cortex module, or LLM directly mutates reality or memory stores. All modifications are proposed as transactional operations (`TransactionOperation`) committed atomically by the `TransactionManager`.

2. **Reasoning Invariant:**
   > *"Operators calculate. HCIR decides what becomes reality."*  
   Reasoning operators and planners receive immutable snapshots (`FrozenGraphView` / `CognitiveContext`) and return pure result structures (`CognitiveResult`).

3. **Zero LLM Dependency in Cognitive Core:**
   > The core cognitive runtime (`hbllm.hcir` — including `world`, `grounding`, `reasoning`, `language`, `snn`, `capabilities`, `kernel` — and `hbllm.brain.reasoning`) is **100% LLM-independent**.  
   Core constraint solving, formal logic, A*/MCTS planning, kinematic simulation, and spatial transforms must run deterministically without requiring external API calls (native Rust spatial/graph kernels run sub-millisecond `< 0.2ms`; the full 5-stage Python `AutonomousEpistemicEngine.step()` loop runs in low tens of milliseconds — `p50 ~ 10-15ms / p95 <= 45ms` at `32×32`, profiled by `hbllm bench profile`).  
   *(Permitted: deterministic, offline-trained feed-forward neural or spiking neural feature encoders conforming to the `FeatureExtractor` protocol, with zero runtime LLM or cloud dependencies).*

4. **Constitutional Governance:**
   > Every state mutation passes through `ConstitutionalVerifier` before commit. Violations (e.g., budget breach, policy failure, safety conflict) trigger compensating rollbacks.

5. **Multi-Tenant Memory Isolation:**
   > All episodic, semantic, procedural, and value memories are strictly partitioned by tenant ID across the tiered workspace.

6. **Target State Evolution Principle (Forward Architecture Directive):**
   > *"Never write temporary duct tape. Always upgrade toward the target architecture."*  
   Whenever modifying existing or legacy code, **do not add new procedural hacks, monkey-patches, or level-specific `if/else` branches**. Instead, refactor the code toward its intended canonical architecture: declarative HCIR grammar, formal transactions, or dynamic causal predicates. Every change must move the codebase closer to the future target state.

7. **General Intelligence Invariant (Task-Agnostic Core Directive):**
   > *"The core is general intelligence, never a single-task solver."*  
   The cognitive core (`hbllm.brain`, `hbllm.hcir`, `hbllm.hcir.world`) must always remain strictly task-agnostic and domain-agnostic. No special functions, ad-hoc solvers, or heuristics designed solely for individual tasks or specific benchmarks (such as dedicated ARC-AGI puzzle solvers, hardcoded game mechanics, or competition-only shortcuts) may be included in the core. All capabilities must emerge from universal neuro-symbolic principles, active inference, Pearl's causal calculus, spatial physics, and domain-general reasoning. Single-task adapters or benchmark harnesses belong strictly in external evaluation or plugin modules (`benchmarks/`, `plugins/`), never in `core/`.

8. **Internal Memory Governance Invariant (No Ad-Hoc Attribute Sprawl):**
   > *"Never attach ad-hoc tracking attributes or monkey-patched state to engine instances. Route all learned knowledge and cognitive state through formal internal memory stores."*  
   Whenever the cognitive system learns something (e.g. what is a goal, barrier, cargo, patroller, or when salience shifts), **do not invent new instance variables** (`engine._last_xxx = ...`, `engine.custom_flag = ...`). Semantic ontology must be stored in `HCIRSymbolicWorldTheory` (`engine.symbolic_theory`) via `CausalPredicate`. Transient working salience and phase triggers must be managed by `PrefrontalWorkingMemory` (`engine.working_memory`). Behavioral programs belong in procedural memory (`DeclarativeNeuroSymbolicSkill`), and long-term concepts belong in `TieredWorkspace` via `TransactionManager`.

---

## 2. The Cortical Epistemic Loop (`hbllm.hcir.world`)

All embodied and spatial problem solving follows a biological, 5-stage active inference cycle:

```
  ┌───────────────────────────────────────────────────────────────────┐
  │ 1. SENSORY PERCEPTION & COORDINATE LIFTING (PerceptionStage)     │
  │    (perception/, parietal_coordinate_transform.py)                │
  │    Transforms raw lattices/sensory data into typed SpatialEntity  │
  │    components and relative coordinate frames.                     │
  └─────────────────────────────────┬─────────────────────────────────┘
                                    │
  ┌─────────────────────────────────▼─────────────────────────────────┐
  │ 2. ASSIMILATION & SURPRISE INDUCTION (AssimilationStage)          │
  │    (assimilation/assimilator.py, surprise_engine.py)              │
  │    Binds motor diffs to actions; updates EpistemicPhase; detects  │
  │    high surprise to invalidate stale causal hypotheses.           │
  └─────────────────────────────────┬─────────────────────────────────┘
                                    │
  ┌─────────────────────────────────▼─────────────────────────────────┐
  │ 3. COUNTERFACTUAL SIMULATION & SUBGOAL SCHEDULING                 │
  │    (SimulationStage / planning/, frontopolar_subgoal)             │
  │    OFC mental simulation rollouts; dead-end pruning (corners/    │
  │    walls); BA10 recursive subgoal decomposition (keys, doors).    │
  └─────────────────────────────────┬─────────────────────────────────┘
                                    │
  ┌─────────────────────────────────▼─────────────────────────────────┐
  │ 4. PREFRONTAL DELIBERATION & SAFETY GATING (DeliberationStage)   │
  │    (deliberation/engine.py, anterior_cingulate_conflict.py)       │
  │    Evaluates candidate plans against looming collision cones,     │
  │    dynamic patroller trajectories, and Habenular action inhibition.│
  └─────────────────────────────────┬─────────────────────────────────┘
                                    │
  ┌─────────────────────────────────▼─────────────────────────────────┐
  │ 5. MOTOR DISPATCH & TRANSACTION EXECUTION (MotorStage)            │
  │    (motor/, KernelServices, TransactionManager)                   │
  │    Commits state mutation transactions; dispatches chosen motor   │
  │    action via CapabilityResolver.                                 │
  └───────────────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Domain-Agnostic Invariant:** Cortical faculties must never be coupled to a specific game, level, or simulation environment. Mechanisms must be based on universal physical principles (topology, collision cones, reachability, symmetries, causal credit assignment).

> [!NOTE]
> **Faculty Roster (latest R1 ablation on saturated dev: `reports/ablation/20261124_084702_r1_dev.json`; earlier R5 baseline: `reports/ablation/20261116_175717_r1_dev.json`):**
> - **Stage owners (core, always kept):** `perception`, `assimilator`, `deliberation_engine`, `planner`, `motor`, `episodic_cortex`, `working_memory`, `symbolic_theory`.
> - **Measured R1 `keep` (2026-11-24):** `body_schema`, `habenular_ior`, `remote_causal`, `object_planner`, `prediction_budget`, `budget_governor`.
> - **Marked `experimental` on saturated dev (pending `widened_dev` R1 ablation):** `subgoal_stack`, `hazard_tracker`, `acc_conflict_monitor`, `option_discovery`, `grounding_bridge` (`world_kernel` = `unused`, 0 calls).
> - **Removed in R5 & Rescoped LOC Budget:** 28 zero-effect/dead modules deleted and dead pipeline hooks unwired (`hbllm/**/*.py` reduced to `242,638` LOC in R5). Active codebase budget is **`<= 300,000` LOC** across `hbllm/**/*.py` (with `<= 150,000` LOC scoped to `hbllm/hcir/**`, tracked by `tests/unit/architecture/test_loc_budget.py`) so high-value non-LLM subsystems (`reasoning/`, `snn/`, `A16`/`A17` `language/`, multimodal sensory fusion, and governed tool capabilities) can be consolidated directly into `hcir/` without `__path__.append` shadow-tree shims. Every newly admitted cortical/grounding faculty must still pass R1 ablation (`Δ > 0`) and latency (`p95 <= 45ms` at `32×32`) gates.

---

## 3. Substrate (Code/Rust) vs. Cognitive Programs (HCIR Language)

HBLLM maintains a strict separation between low-level computational substrates and high-level cognitive programs:

| Tier | Role | Implementation Substrate | Examples |
| :--- | :--- | :--- | :--- |
| **Sensory-Motor Substrate** | Tight math, spatial physics, grid transforms, graph hashing | **Fast Python / Rust** | Ray projection, 2D topology cuts, BLAKE3 chunk hashing, flood fill |
| **Cognitive Core (ISA)** | Bytecode virtual machine & syscall dispatch | **Kernel Bytecode** (`hbllm.hcir.bytecode`) | 8 opcodes: `ASSERT`, `RETRACT`, `QUERY`, `EXECUTE`, `WAIT`, `FORK`, `MERGE`, `ROLLBACK` |
| **Cognitive Programs** | High-level planning, subgoals, skills, arbitration | **Declarative HCIR DSL** (`hbllm.hcir.skills.declarative`) | `DeclarativeNeuroSymbolicSkill`, `SymbolicPredicate`, `SubgoalSequence` |
| **World Theory** | Empirical causal knowledge & affordances | **Dynamic Predicates** (`cortex_causal.py`) | `HCIRSymbolicWorldTheory`, `CausalPredicate` (Pearl's *do(a)* calculus) |

### Role of LLMs in the Architecture
- **Do NOT** insert LLMs into low-level sensory-motor loops or grid step-by-step simulations (too slow, hallucination-prone).
- **DO** use LLMs as high-level **Meta-Compilers & Synthesizers** (`hbllm.hcir.compiler.HCIRCompiler`):
  - Parsing ambiguous human natural language into `SemanticAST`.
  - Compiling intent into declarative skill programs (`InstructionStream`).
  - Synthesizing new domain hypotheses stored in the `TieredWorkspace`.

---

## 4. Concrete Code Patterns vs. Anti-Patterns

### Pattern A: State Mutation

```python
# ❌ FORBIDDEN: Direct graph or memory mutation
workspace.graph.nodes["entity_10"] = new_node
engine.goals.append((r, c))

# ✅ CORRECT: Transactional proposal via TransactionManager
async with tx_mgr.begin_transaction() as tx:
    tx.add_operation(
        TransactionOperation(
            op=TransactionOp.INSERT_NODE,
            node=new_node,
        )
    )
    await tx.commit()

# ✅ ALSO CORRECT: subsystems holding only a workspace reference (adapters, tier
# promotion, world kernel) use the workspace's attached commit authority
workspace.commit_operations(
    [TransactionOperation(op=TransactionOp.UPSERT_NODE, node=new_node)], author="memory_adapter"
)

# ❌ FORBIDDEN: the raw primitives `workspace.add_node/upsert_node/remove_node/add_edge/
# remove_edge` are kernel-internal (used only by TransactionManager._apply_one).
```

Kernel guarantees you can rely on: commits are atomic (partial failure is reverted
in-flight), `MERGE`/`merge_branch` is a single verified diff transaction, `ROLLBACK`
restores the exact snapshot via `TransactionManager.rollback_to_version`, and every
commit/rejection is published on `KernelEventBus` as `transaction.committed|rejected`.

### Pattern B: Defining Skills & Behaviors

```python
# ❌ FORBIDDEN: Procedural spaghetti with hundreds of if/else statements
def solve_puzzle(grid, actions):
    if grid[2, 3] == 5 and actions == [1, 2]:
        return [1, 1, 3]
    ...


# ✅ CORRECT: Declarative Neuro-Symbolic Skill with composable predicates
class KeyDoorSkill(DeclarativeNeuroSymbolicSkill):
    signature = ActionAffordancePredicate(required={1, 2, 3, 4}) & CutSetPredicate(min_partitions=2)
    program = SubgoalSequence(
        SymbolicSubgoal(
            intent=SpatialActionIntent.NAVIGATE_TO,
            target_query={"role": "key"},
        ),
        SymbolicSubgoal(
            intent=SpatialActionIntent.INTERACT,
            target_query={"role": "barrier"},
        ),
    )
```

### Pattern C: Asynchronous Coordination

```python
# ❌ FORBIDDEN: Busy-waiting with polling loops
while not engine.is_settled:
    await asyncio.sleep(0.05)

# ✅ CORRECT: EventBus subscription or WAIT opcode
await services.event_bus.wait_for(
    event_type="perception.stabilized",
    timeout_ms=500,
)
```

### Pattern D: Actuator Features vs In-Place Action Mutations

```python
# ❌ FORBIDDEN: Treating in-place action diffs or avatar displacement as distal switches
# Action mutations (e.g. avatar stepping on void, paint actions) must NEVER pollute actuator_features
theory.actuator_features.add(hyp.premise_feature)

# ✅ CORRECT: Distinguish in-place spatial transforms from distal actuators
# Only verified remote switches/buttons belong in actuator_features; in-place mutations
# are recorded strictly as CausalPredicate state transforms.
if hyp.hypothesis_type == HypothesisType.REMOTE_MECHANISM:
    if not theory.is_avatar_feature(hyp.premise_feature) and not theory.is_walkable(
        hyp.premise_feature
    ):
        theory.actuator_features.add(hyp.premise_feature)
```

### Pattern E: Sandboxed Capability Execution & Subprocess Isolation

```python
# ❌ FORBIDDEN: Executing untrusted code in-process or blocking parent pipe writers
proc.start()
result = await _await_child(proc, parent_conn, timeout)
child_conn.close()  # Bug: writer open in parent leads to EOF hang

# ✅ CORRECT: Two-tier isolation with immediate parent write-end cleanup and responsive polling
proc.start()
try:
    child_conn.close()  # Close write-end immediately in parent so pipe EOF signals properly
except Exception:
    pass
result = await loop.run_in_executor(None, lambda: _await_child(proc, parent_conn, timeout))
```

---

## 5. Agent Anti-Patterns Blacklist (Strictly Forbidden)

1. **No Single-Task Solvers, Game-Specific Hacks, or Benchmark Solvers in Core:**  
   Never insert special functions, single-task routines, or dedicated game/puzzle solvers (such as ARC-AGI puzzle solvers, hardcoded game mechanics, level IDs, or bespoke competition heuristics) into `core/hbllm/`. The core must always implement general intelligence. If an environment or benchmark exposes a reasoning bottleneck, address it by strengthening universal neuro-symbolic faculties (e.g. topological manifold routing, Pearl's causal *do(a)* calculus, active inference dead-end pruning), never by adding specialized task solvers.
2. **No Direct SQLite / DB Queries:**  
   Never issue raw SQL queries directly against `working_memory.db`, `procedural_memory.db`, etc. Always route through `HCIRMemoryBackend`.
3. **No External LLM Client Imports in Core Reasoning:**  
   Never import `openai`, `anthropic`, `litellm`, `transformers`, `plugins.llm_scaffold`, or `hbllm.serving.provider` inside `core/hbllm/hcir/**` or `core/hbllm/brain/reasoning/` (enforced by `tests/unit/architecture/test_loc_budget.py`).
4. **No Bypassing `ConstitutionalVerifier`:**  
   Every state change must be verifiable and reversible. Never commit unvalidated state directly to the graph.
5. **No Monkey-Patching / Band-Aid Fixes:**  
   Always implement the intended target architecture (Rule 1.6).
6. **No Ad-Hoc Instance Variables or Monkey-Patched State on Engine:**  
   Never add ad-hoc tracking attributes (e.g. `_last_sim_trajectory_count`, `_last_sim_goal_count`, `_feedback_assimilated`) directly onto `AutonomousEpistemicEngine` or cortical classes to patch edge cases. Always use `PrefrontalWorkingMemory` or `HCIRSymbolicWorldTheory` (Rule 1.8).

---

## 6. Rust Substrate & Build Workflow (`rust/hcir_graph`)

HBLLM provides a chunked structural-sharing graph substrate written in native Rust with PyO3 bindings (`hbllm_hcir_graph`), with an automatic Python fallback:

- **Location:** `core/rust/hcir_graph/`
- **When to build:** After any changes to Rust source files or when installing native graph acceleration:
  ```bash
  cd core/rust/hcir_graph
  maturin develop --release
  ```
- **Parity Requirement:** The Python `NativeGraphAdapter` must remain functionally identical to the pure Python dictionary implementation for transparent fallback when native binaries are not compiled.
- **Substrate health:** `CognitiveGraph.native_health()` reports `{enabled, available, divergences, last_error}`; divergences raise `SubstrateDivergenceError` (Python is authoritative).
- **Fork/diff:** `graph.fork()` is copy-on-write with dirty-set tracking; `workspace.diff_branch()` is O(dirty) for forked branches. See `docs/architecture/graph_substrate.md`.

---

## 7. Codebase Architecture & Key Locations

```
HBLLM/
├── core/
│   ├── hbllm/
│   │   ├── hcir/                    # Pure-HCIR Cognitive OS Kernel & Subsystems
│   │   │   ├── bytecode.py          # 8 primitive opcodes + instruction streams
│   │   │   ├── compiler.py          # Semantic AST → HCIR Bytecode
│   │   │   ├── interpreter.py       # Virtual machine & SyscallDispatcher
│   │   │   ├── graph.py             # CognitiveGraph, HCIRNode, HCIREdge
│   │   │   ├── workspace_tiers.py   # TieredWorkspace (Working, Brain, Persistent, Meta)
│   │   │   ├── kernel/              # CognitiveKernel, TransactionManager, Services, CommandDispatcher, AutonomyTracker
│   │   │   ├── world/               # 5-stage CorticalPipeline & 17 domain subpackages
│   │   │   │   ├── autonomous_epistemic_engine.py # Core epistemic active inference
│   │   │   │   ├── cortical_pipeline.py           # Perception → Assimilation → Simulation → Deliberation → Motor
│   │   │   │   ├── causal/transition_kb.py        # HCIR-backed TransitionKnowledgeBase + theory.py
│   │   │   │   ├── assimilation/, deliberation/, episodic/, grounding/, hypothesis/, learning/
│   │   │   │   └── macro/, motor/, objects/, perception/, planning/, prediction/, predictors/, spatial/, state/, working_memory/
│   │   │   ├── grounding/           # Symbol Grounding Engine (features, tracking, concepts, committer)
│   │   │   ├── language/            # Non-LLM Multilingual Runtime (en/si/ta, discourse, Track T4 reading, acquisition)
│   │   │   ├── reasoning/           # UnifiedReasoningRuntime + 14 formal operators (0 LLM deps)
│   │   │   ├── snn/                 # LIF, STDP, STP, WTA, oscillations, neuromodulation, perception
│   │   │   ├── capabilities/        # ConceptExplainer, EpistemicCuriosity, BeliefConflictResolver, sandboxed tools/web/memory
│   │   │   ├── skills/declarative.py# Declarative Neuro-Symbolic Skill Grammar
│   │   │   ├── stores/              # SQLiteEventStore, LexicalOntologyStore, JSONGraphStore
│   │   │   ├── adapters/            # HCIRMemoryBackend, HCIRBusBridge, MemoryMigrationProxy
│   │   │   └── algorithms/          # Graph connectivity, paths, centrality, flow, spans
│   │   ├── runtime/bootstrap.py     # Canonical single composition root (build_cognitive_system)
│   │   ├── brain/                   # Compatibility shims (reasoning/language/snn) & legacy MessageBus nodes
│   │   ├── native/registry.py       # NativeAccelerationRegistry (13 Rust crates + Python fallbacks)
│   │   ├── network/                 # Distributed MessageBus, Federation, Resilience
│   │   ├── security/                # TenantGuard, AuditTrail, EncryptionVault, PIIRedactor, RBAC
│   │   └── serving/                 # FastAPI REST/Studio API, fail-closed JWTAuthMiddleware, MCP server
│   ├── rust/                        # 13 PyO3 native Rust crates (hcir_graph, simulation_engine, structure_matcher, ...)
│   ├── plugins/                     # External environment adapters & plugin bundles
│   ├── benchmarks/suites/           # Frozen dev, novelty, widened_dev, heldout, independent YAML suites
│   ├── docs/                        # Pure-HCIR Markdown architecture, guides & API documentation
│   ├── graphify-out/                # Persistent AST & semantic knowledge graph
│   └── tests/                       # Unit, architecture & integration test suites
└── ...
```

---

## 8. Knowledge Graph Navigation (`graphify`)

This project maintains a navigable knowledge graph in `graphify-out/`.

### Rules for Agents:
1. **Explore with graphify first:** For codebase or architectural questions, query the graph before doing broad greps:
   ```bash
   graphify query "<concept or question>"
   graphify explain "<path/to/file.py>::<Symbol>"
   graphify path "<SymbolA>" "<SymbolB>"
   ```
2. **Keep the graph in sync:** After making substantial modifications to code or adding new files in a session, run:
   ```bash
   graphify update .
   ```
   *(AST-only scan, fast with zero LLM API cost).*

---

## 9. Engineering Standards & Workflows

### Python Standards
- Python **3.11+** required.
- Always include `from __future__ import annotations` at the top of Python files.
- Enforce strict static typing (`typing.Any` used only where dynamic polymorphism is strictly required).
- Format code cleanly; adhere to Ruff and Pyright / MyPy conventions.
- Maintain existing docstrings and comments. Never delete explanatory cognitive science references without reason.

### Testing & Validation Protocols
Run relevant tests from `HBLLM/core/` before concluding any changes:
```bash
# Fast HCIR & architecture invariant suites
pytest tests/unit/hcir/ tests/unit/architecture/ -q

# Epistemic & Cortical world model tests
pytest tests/unit/ -k "epistemic or cortex or hcir" -q

# Specific component test
pytest tests/unit/ -k "<relevant_component>" -q

# Full system runtime diagnostics
python -m hbllm diagnostics

# Benchmark evaluation & latency profiling
python -m hbllm bench dev
python -m hbllm bench local-suite
python -m hbllm bench profile

# Kernel status and cognitive mind single step
python -m hbllm kernel status
python -m hbllm mind step --obs tests/fixtures/obs_minimal.json
```

### Ecosystem Context
- **Web App / Admin Control Plane:** Multi-tenant web platform built on Laravel with Filament.
- **Mobile Client:** Cross-platform mobile applications built in Flutter.
- **Integrations:** Sentra plugin adapters (e.g. AI2-THOR, Safety Gym, MineDojo, ROS2).

---

## 10. Quick Decision Checklist for Agents

Before completing any task, verify:
1. [ ] **Target Architecture:** Does this change upgrade the code toward its canonical target state, avoiding temporary monkey-patches?
2. [ ] **State Ownership:** Is state mutated via `TransactionManager` instead of direct assignment?
3. [ ] **Declarative over Procedural:** Are new skills or faculties expressed via declarative grammar (`DeclarativeNeuroSymbolicSkill` + `SymbolicPredicate`) rather than ad-hoc if-statements?
4. [ ] **LLM-Free Core:** Does reasoning remain completely independent of external LLM API calls?
5. [ ] **General Intelligence & Domain Agnostic:** Does the core remain strictly task-agnostic, free of single-task functions, game-specific hacks, or ARC-AGI-specific puzzle solvers?
6. [ ] **Unit Tests:** Have changes been validated with `pytest`?
7. [ ] **Internal Memory Governance:** Is state and learned ontology stored in `HCIRSymbolicWorldTheory` and `PrefrontalWorkingMemory` rather than ad-hoc class instance variables?
8. [ ] **Knowledge Graph:** Has `graphify update .` been executed after modifying code files?
