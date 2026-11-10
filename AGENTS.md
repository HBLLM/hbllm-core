# HBLLM — Project Instructions & Cognitive Architecture Guide

> **Primary Directive for AI Agents:**  
> This file is the authoritative reference for working on the HBLLM codebase. Read and uphold all architectural invariants, neuro-symbolic design rules, directory conventions, and engineering standards defined herein.

> [!WARNING]
> **Current project phase: Phase 1 — Scope Freeze, final gate (re-verified 2026-11-10).**  
> Phase 1 structural work is done and independently verified: LOC 249,249 (≤250k), 1687 tests pass / 1 fails, single composition root with unified tiered workspace, persistence on by default, event-driven runtime, closed-loop procedural harness in `benchmarks/envs/` whose oracle/random/stub baselines discriminate correctly.
>
> **The freeze is NOT lifted.** The harness produced the project's first honest generality measurement and it is poor: **GridNav 0/3, SymbolicRules 2/3** (oracle 3/3), and `reports/heldout/` is empty because held-out was never run. Read `reports/audit/2026-11-09_post_implementation_review.md` **§7b and §9** before any change. Remaining gate items: fix `benchmarks/suites/runner.py` to unpack `(action, meta)` from `engine.step()`; seed the random baseline; run `hbllm bench heldout` once and commit the JSON whatever the score; fix `test_tiered_workspace_transaction_mirroring_and_promotion`; eliminate the 9 bare `AutonomousEpistemicEngine(` constructions outside `bootstrap.py`; delete the placebo `benchmarks/suites/{grid,symbolic,continuous}/`.
>
> **Phase 2 definition (when unlocked):** the held-out score is the compass. A change is admissible only with before/after held-out JSON attached. Do not add modules or extend gap-analysis G-items; first diagnose why the engine fails procedurally generated navigation (`ReplayDebugger` on `GridNav seed 0`), then re-run faculty ablation on the real harness and delete what does not move the score. A benchmark that cannot fail is not a benchmark; a score of zero is a result, not a pass.

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
   > The core reasoning runtime (`hbllm.brain.reasoning`, `hbllm.hcir.world`) is **100% LLM-independent**.  
   Core constraint solving, formal logic, A*/MCTS planning, kinematic simulation, and spatial transforms must run deterministically with sub-millisecond execution without requiring external API calls.  
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
  │    (cortex_perception.py, parietal_coordinate_transform.py)       │
  │    Transforms raw lattices/sensory data into typed SpatialEntity  │
  │    components and relative coordinate frames.                     │
  └─────────────────────────────────┬─────────────────────────────────┘
                                    │
  ┌─────────────────────────────────▼─────────────────────────────────┐
  │ 2. ASSIMILATION & SURPRISE INDUCTION (AssimilationStage)          │
  │    (cortex_assimilator.py, surprise_engine.py)                    │
  │    Binds motor diffs to actions; updates EpistemicPhase; detects  │
  │    high surprise to invalidate stale causal hypotheses.           │
  └─────────────────────────────────┬─────────────────────────────────┘
                                    │
  ┌─────────────────────────────────▼─────────────────────────────────┐
  │ 3. COUNTERFACTUAL SIMULATION & SUBGOAL SCHEDULING                 │
  │    (SimulationStage / cortex_planner.py, frontopolar_subgoal)     │
  │    OFC mental simulation rollouts; dead-end pruning (corners/    │
  │    walls); BA10 recursive subgoal decomposition (keys, doors).    │
  └─────────────────────────────────┬─────────────────────────────────┘
                                    │
  ┌─────────────────────────────────▼─────────────────────────────────┐
  │ 4. PREFRONTAL DELIBERATION & SAFETY GATING (DeliberationStage)   │
  │    (cortex_deliberation.py, anterior_cingulate_conflict.py)       │
  │    Evaluates candidate plans against looming collision cones,     │
  │    dynamic patroller trajectories, and Habenular action inhibition.│
  └─────────────────────────────────┬─────────────────────────────────┘
                                    │
  ┌─────────────────────────────────▼─────────────────────────────────┐
  │ 5. MOTOR DISPATCH & TRANSACTION EXECUTION (MotorStage)            │
  │    (cortex_motor.py, KernelServices, TransactionManager)          │
  │    Commits state mutation transactions; dispatches chosen motor   │
  │    action via CapabilityResolver.                                 │
  └───────────────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Domain-Agnostic Invariant:** Cortical faculties must never be coupled to a specific game, level, or simulation environment. Mechanisms must be based on universal physical principles (topology, collision cones, reachability, symmetries, causal credit assignment).

> [!NOTE]
> **Faculty Roster (WP2 measured ablation, `reports/ablation/20261101_211519.json`):**
> - **Stage owners (core, always kept):** `perception`, `assimilator`, `deliberation_engine`, `planner`, `motor`, `episodic_cortex`, `working_memory`, `symbolic_theory`.
> - **Kept faculties (Δlevels > sd or load-bearing):** `acc_conflict_monitor`, `collision_cones`, `deformation_tracker`, `dorsal_kinetic_stream`, `hazard_tracker`, `metacognitive_agency_cortex`, `neuromodulatory_system`, `precision_controller`, `prediction_budget`, `predictive_coding_cortex`, `snn_predictor`, `surprise_engine`.
> - **Experimental/unused (ablated to `AblatedFaculty` nulls; classes importable via `hbllm.hcir.world.experimental`):** 52 faculties — see the report for the measured evidence and the decision rule.

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
   Never import `openai`, `anthropic`, `transformers`, or network clients inside `core/hbllm/brain/reasoning/` or `core/hbllm/hcir/world/`.
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
│   │   ├── hcir/                    # HCIR Cognitive OS Kernel
│   │   │   ├── bytecode.py          # 8 primitive opcodes + instruction streams
│   │   │   ├── compiler.py          # NL → Semantic AST → HCIR Bytecode
│   │   │   ├── interpreter.py       # Virtual machine & SyscallDispatcher
│   │   │   ├── graph.py             # CognitiveGraph, HCIRNode, HCIREdge
│   │   │   ├── workspace_tiers.py   # TieredWorkspace (Working, Brain, Persistent, Meta)
│   │   │   ├── kernel/              # CognitiveKernel, TransactionManager, Services
│   │   │   ├── skills/declarative.py# Declarative Neuro-Symbolic Skill Grammar
│   │   │   └── world/               # Cortical faculties & epistemic world models
│   │   │       ├── cortex_planner.py         # OFC / BA10 mental simulation planner
│   │   │       ├── cortex_deliberation.py     # dlPFC / vmPFC arbitration & gating
│   │   │       ├── cortex_causal.py           # Pearl's do(a) causal theory & affordances
│   │   │       ├── cortex_episodic.py         # Episodic memory consolidation
│   │   │       ├── cortex_assimilator.py      # Motor dynamic assimilation
│   │   │       └── autonomous_epistemic_engine.py # Core epistemic active inference
│   │   ├── brain/                   # Cognitive nodes, composites, & reasoning
│   │   │   ├── reasoning/           # 13 formal reasoning operators (LLM-free)
│   │   │   ├── wiring/              # Brain wiring & factory pipelines
│   │   │   └── self_model/          # Attention manager & metacognition
│   │   ├── network/                 # Distributed MessageBus, Node protocols
│   │   └── serving/                 # API server, studio, and pipeline
│   ├── rust/hcir_graph/             # Rust native chunked structural-sharing graph (PyO3)
│   └── tests/                       # Unit and integration test suites
├── docs/                            # Markdown architecture & API documentation
├── graphify-out/                    # Persistent AST & semantic knowledge graph
└── AGENTS.md                        # This instruction file
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
Run relevant tests before concluding any changes:
```bash
# Fast HCIR test suite
pytest core/tests/unit/hcir/ -q

# Epistemic & Cortical world model tests
pytest core/tests/unit/ -k "epistemic or cortex or hcir" -q

# Specific component test
pytest core/tests/unit/ -k "<relevant_component>" -q

# Full system runtime diagnostics
python -m hbllm diagnostics

# Benchmark evaluation harness
python -m hbllm bench dev
python -m hbllm bench local-suite

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
