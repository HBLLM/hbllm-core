# HBLLM — Project Instructions & Cognitive Architecture Guide

> **Primary Directive for AI Agents:**
> This file is the authoritative reference for working on the HBLLM codebase. Read and uphold all architectural invariants, neuro-symbolic design rules, directory conventions, and engineering standards defined herein.

> [!WARNING]
> **Active Phase: Phase 3 — Homoiconic Symbolic VM & Experience-First Cognition (`M0`–`M6`, starting 2026-11-25).**
> - **Target Architecture ([ADR 005](docs/adr/0005-homoiconic-symbolic-vm-and-experience-first-cognition.md) & [Migration Plan](docs/architecture/homoiconic-vm-migration-plan.md)):** *"Code = CPU/ALU & Constitutional Kernel; Graph = Symbolic Mind."* Python and Rust implement strictly the deterministic Layer-0 substrate (register-unification VM, sensory/SNN encoders, math/topology/Bellman/Bayesian ALUs, sandboxed I/O, constitutional transactions). All concepts, grammar constructions, causal laws, reasoning strategies, and skills live as **homoiconic symbolic programs in `TieredWorkspace`**, learned and composed from experience.
> - **Phase 2 Verified Baseline (`eval-phase2-r10` on `a26f09bd`, 2026-11-21, `n=120`, `reports/heldout/LEDGER.md`):** `zero_shot=0.9250` (`111/120`, CI `[0.8636, 0.9600]`, generality `0.8236`, 4 deaths / `3.33%`), `cross_seed transfer=0.9035` (`114` eps, `skills_consolidated=161`), `same_layout_replay=1.0000` (`0` LLM tokens). Pure-HCIR consolidated `reasoning/` (14 operators), `language/` (`en`/`si`/`ta` + Track T4), `snn/`, `capabilities/`, and 17 `world/` subpackages over HCIR-backed `TransitionKnowledgeBase`. Open gaps tracked in `reports/audit/2026-11-24_full_project_audit.md`.

---

## 1. Project Vision & Core Invariants

HBLLM is a **human-brain-inspired, event-sourced Cognitive OS and homoiconic neuro-symbolic AI architecture**. Intelligence is not a monolithic transformer nor a pile of hardcoded Python `if/else` heuristics — it is a formal register-unification VM executing **experience-learned symbolic programs** stored inside **HCIR**.

### Non-Negotiable Invariants

1. **State Ownership Invariant:**
   > *"Nodes emit intent. HCIR owns state."*
   No brain node, cortex stage, or LLM directly mutates reality or memory stores. All mutations are proposed as `TransactionOperation`s committed atomically by `TransactionManager` (or `workspace.commit_operations(...)`).

2. **Reasoning Invariant:**
   > *"Operators calculate. HCIR decides what becomes reality."*
   Reasoning operators and cortical stages receive immutable snapshots (`FrozenGraphView` / `CognitiveContext`) and return pure result structures (`CognitiveResult` / `StageResult.operations`).

3. **Zero LLM Dependency & Latency Budget in Cognitive Core:**
   > The core runtime (`hbllm.hcir` — `world`, `grounding`, `reasoning`, `language`, `snn`, `capabilities`, `kernel` — and `hbllm.brain.reasoning`) is **100% LLM-independent**.
   Constraint solving, unification, planning, and spatial transforms run deterministically with zero external LLM calls (Rust kernels `< 0.2ms` per op; target `32×32` step budget `CPU p95 <= 45ms`). Offline-trained deterministic `FeatureExtractor` encoders (`snn`, `learned`) are permitted.

4. **Constitutional Governance:**
   > Every state mutation passes through `ConstitutionalVerifier` before commit. Unverified fallback managers are forbidden in production; violations trigger atomic rollback.

5. **Multi-Tenant Memory Isolation:**
   > All episodic, semantic, procedural, construction, and value nodes are strictly partitioned by `tenant_id` across every `TieredWorkspace` tier.

6. **Target State Evolution Principle (Forward Architecture Directive):**
   > *"Never write temporary duct tape. Always upgrade toward the target architecture."*
   Never add procedural hacks, monkey-patches, or level-specific `if/else` branches. Move the codebase monotonically along [M0–M6](docs/architecture/homoiconic-vm-migration-plan.md) toward declarative HCIR programs, formal transactions, and dynamic causal posteriors.

7. **General Intelligence Invariant (Task-Agnostic Core Directive):**
   > *"The core is general intelligence, never a single-task solver."*
   `hbllm.hcir`, `hbllm.brain`, and `hbllm.runtime` must remain strictly domain-agnostic and task-agnostic. Never add dedicated puzzle/ARC-AGI solvers, game mechanics, or benchmark shortcuts inside `core/hbllm/`. Only the Spelke/active-inference innate priors cataloged in [`docs/architecture/priors.md`](docs/architecture/priors.md) are permitted in core; everything else must be learned in-environment. Single-task adapters belong strictly in `benchmarks/` or `plugins/`.

8. **Internal Memory Governance Invariant (No Ad-Hoc Attribute Sprawl):**
   > *"Never attach ad-hoc tracking attributes or monkey-patched state to engine instances."*
   Never invent instance flags (`engine._last_*`, `_feedback_assimilated`, `view.state.<attr> = ...`). Semantic ontology lives in `HCIRSymbolicWorldTheory` (`TransitionKnowledgeBase`), transient working salience in `PrefrontalWorkingMemory`, executable skills in `SkillNode`s, and long-term concepts/constructions in `TieredWorkspace`.

9. **Homoiconic Symbolic-First Invariant ("Code = CPU, Graph = Mind"):**
   > *"Python/Rust is the CPU and hardware ALU; all concepts, grammar, causal laws, and skills live as symbolic programs in HCIR."*
   - Layer-0 code (`hbllm/hcir/` runtime, Rust crates, SNN, math ALUs) must remain ontology-blind and language-blind.
   - Never add hardcoded English/multilingual verb/keyword lists, regex rule tables, Python-only `SymbolicPredicate` subclasses for domain mechanics, high-level Python "cheat" capabilities behind `Opcode.EXECUTE`, or metadata-only `SkillNode`s lacking executable `bytecode_stream`s.
   - Express linguistic patterns as `ConstructionNode`s, causal rules as probabilistic `CausalPredicate`/`LearnedRuleNode`s, and behaviors as parameterized ISA v2 `InstructionStream`s stored in `TieredWorkspace` and composed via `UNIFY` + `CALL`.

---

## 2. The Cortical Epistemic Loop (`hbllm.hcir.world`)

All embodied and spatial problem solving follows the 5-stage active inference cycle:

```text
  1. SENSORY PERCEPTION & COORDINATE LIFTING (PerceptionStage)
     └─► 2. ASSIMILATION & SURPRISE INDUCTION (AssimilationStage)
         └─► 3. COUNTERFACTUAL SIMULATION & SUBGOAL SCHEDULING (SimulationStage)
             └─► 4. PREFRONTAL DELIBERATION & SAFETY GATING (DeliberationStage)
                 └─► 5. MOTOR DISPATCH & TRANSACTION EXECUTION (MotorStage)
```

> [!IMPORTANT]
> **Domain-Agnostic & Faculty Admission Gate:** Cortical faculties must rely solely on universal physical/topological/causal principles. Every newly admitted cortical or grounding faculty must pass R1 ablation (`Δ > 0` on `widened_dev`) and latency (`CPU p95 <= 45ms` at `32×32`) gates.

> [!NOTE]
> **Faculty Roster (`2026-11-24` / `2026-11-25` R1 Ablations):**
> - **Stage owners (always kept):** `perception`, `assimilator`, `deliberation_engine`, `planner`, `motor`, `episodic_cortex`, `working_memory`, `symbolic_theory`.
> - **Measured `keep` (`dev` + `widened_dev`):** `body_schema`, `habenular_ior`, `remote_causal`, `object_planner`, `prediction_budget`, `budget_governor`, `acc_conflict_monitor` (`Δgen=+0.0527`), `hazard_tracker` (`Δgen=+0.0238`), `subgoal_stack` (`Δgen=+0.0044`).
> - **`experimental`:** `option_discovery` (`Δgen=0.0000`), `grounding_bridge` (`Δgen=0.0000`). (`world_kernel` = `unused`).

---

## 3. The Three-Layer Cognitive Computer (Hardware ALU vs. Symbolic Mind)

HBLLM enforces a strict boundary between fixed substrate hardware and learned homoiconic software:

| Layer | Role | Implementation Substrate | Examples |
| :--- | :--- | :--- | :--- |
| **Layer 0: Hardware & ALUs** | Tight math, sensory lifting, graph hashing, unification VM | **Rust / SNN / Python CPU** | `hcir_graph` (BLAKE3), `hbllm_structure_matcher`, `hbllm_simulation_engine`, SNN encoders, geodesic/Bellman/Dirichlet math, `ConstitutionalVerifier` |
| **Layer 0: ISA v2 VM** | Register-unification virtual machine & allowlisted syscalls | **HCIR VM** (`bytecode.py`, `interpreter.py`) | Registers `$r0..$r7`, `CallStack`, 12 opcodes: `ASSERT`, `RETRACT`, `QUERY`, `EXECUTE` (hardware-only), `WAIT`, `FORK`, `MERGE`, `ROLLBACK`, `UNIFY`, `BRANCH`, `CALL`, `RETURN` |
| **Layer 1: Innate Seed** | Minimal Spelke & active-inference priors at genesis | **Declarative `.hcir` Seed** (`hbllm/hcir/seed/`) | Object permanence, adjacency/containment, Pearl's *do(a)*, curiosity/active inquiry (~25–35 seed rules bounded by `priors.md`) |
| **Layer 2: Symbolic Mind** | Concepts, grammar, causal laws, skills, macro consolidation | **Homoiconic Graph Nodes** (`TieredWorkspace`) | `ConstructionNode` (language ↔ program), parameterized `SkillNode` (`bytecode_stream`), `LearnedRuleNode`, Wake–Sleep anti-unification |

### Role of LLMs in the Architecture
- **Never** call LLMs inside `hbllm/hcir/**` or real-time sensory-motor/reasoning loops.
- **Optionally** use external LLMs strictly outside core (`plugins/llm_scaffold/` or offline tooling) as high-level meta-compilers or optional shadow distillation aids when native construction unification is absent.

---

## 4. Concrete Code Patterns vs. Anti-Patterns

### Pattern A: State Mutation

```python
# ❌ FORBIDDEN: Direct graph, state, or engine mutation
workspace.graph.nodes["entity_10"] = new_node
view.state._feedback_assimilated = True

# ✅ CORRECT: Transactional commit via TransactionManager or commit_operations
workspace.commit_operations(
    [TransactionOperation(op=TransactionOp.UPSERT_NODE, node_id=new_node.id, node_data=new_node.model_dump())],
    author="memory_adapter",
)
```

Kernel guarantees: atomic commits (in-flight rollback on failure), verified `MERGE` diffs, snapshot `ROLLBACK`, and `KernelEventBus` notifications. Raw `add_node`/`upsert_node`/`remove_node`/`add_edge`/`remove_edge` are kernel-internal.

### Pattern B: Homoiconic Parameterized Skills vs. Hardcoded Python

```python
# ❌ FORBIDDEN: Procedural Python solver or metadata-only SkillNode
def solve_puzzle(grid, actions):
    if grid[2, 3] == 5:
        return [1, 1, 3]

SkillNode(skill_name="unlock", description="opens door")  # Discards executable bytecode!

# ✅ CORRECT: Parameterized homoiconic InstructionStream stored on SkillNode
stream = InstructionStream(
    author="wake_sleep_learner",
    instructions=[
        Instruction(opcode=Opcode.UNIFY, params={"pattern": {"role": "tool", "affords": "?target"}, "bind": "$r0"}),
        Instruction(opcode=Opcode.CALL, params={"goal_pattern": {"intent": "navigate_to", "target": "$r0"}}),
        Instruction(opcode=Opcode.EXECUTE, params={"capability": "motor.interact", "params": {"target": "$r0"}}),
        Instruction(opcode=Opcode.RETURN, params={"yield": ["$r0"]}),
    ],
)
```

### Pattern C: Asynchronous Coordination

```python
# ❌ FORBIDDEN: Busy-waiting polling loops
while not engine.is_settled:
    await asyncio.sleep(0.05)

# ✅ CORRECT: KernelEventBus subscription or WAIT opcode
await services.event_bus.wait_for(event_type="perception.stabilized", timeout_ms=500)
```

### Pattern D: Actuator Features vs. In-Place Action Mutations

```python
# ❌ FORBIDDEN: Polluting actuator_features with avatar steps or in-place mutations
theory.actuator_features.add(hyp.premise_feature)

# ✅ CORRECT: Only verified distal switches belong in actuator_features
if hyp.hypothesis_type == HypothesisType.REMOTE_MECHANISM:
    if not theory.is_avatar_feature(hyp.premise_feature) and not theory.is_walkable(hyp.premise_feature):
        theory.actuator_features.add(hyp.premise_feature)
```

### Pattern E: Sandboxed Subprocess Isolation

```python
# ❌ FORBIDDEN: Executing untrusted code in-process or keeping parent pipe write-end open
proc.start()
result = await _await_child(proc, parent_conn, timeout)
child_conn.close()

# ✅ CORRECT: Close write-end immediately in parent; await via executor + PermissionEngine
proc.start()
try:
    child_conn.close()
except Exception:
    pass
result = await loop.run_in_executor(None, lambda: _await_child(proc, parent_conn, timeout))
```

### Pattern F: Learned Constructions & Unknown Instructions vs. Hardcoded Keywords

```python
# ❌ FORBIDDEN: Hardcoding English verb/rule lists in compiler.py
for act_word in ("push", "move", "grab", "pull"):
    if text_lower.startswith(act_word): ...

# ✅ CORRECT: Graph Construction Unification + 4-Step Unknown Resolution
# 1. Partial UNIFY binds known slots into $r0..$rk; isolates unknown span as UnknownNode.
# 2. QUERY TieredWorkspace (reading definitions / hypernyms) + Analogy (hbllm_structure_matcher).
# 3. If unresolved, FORK simulation or emit REQUEST_EVIDENCE to ground outcome.
# 4. Commit learned ConstructionNode + SkillNode transactionally so future instructions reuse it.
```

---

## 5. Agent Anti-Patterns Blacklist (Strictly Forbidden)

1. **No Single-Task Solvers or Game Hacks in Core:** Never insert task-specific heuristics or ARC-AGI solvers into `core/hbllm/`.
2. **No Direct SQLite / DB Queries:** Never call `sqlite3.connect` / `aiosqlite.connect` outside `stores/`, `persistence/`, or `security/`. Route through `HCIRMemoryBackend` / `TieredWorkspace`.
3. **No External LLM Imports in Core:** Never import `openai`, `anthropic`, `litellm`, `transformers`, `plugins.llm_scaffold`, or `hbllm.serving.provider` inside `hbllm/hcir/**` or `hbllm/brain/reasoning/`.
4. **No Bypassing `ConstitutionalVerifier`:** Never commit unvalidated state or instantiate production `TransactionManager(..., verification_stages=[])`.
5. **No Monkey-Patching / Band-Aid Fixes:** Always upgrade toward [ADR 005](docs/adr/0005-homoiconic-symbolic-vm-and-experience-first-cognition.md) and the [Migration Plan](docs/architecture/homoiconic-vm-migration-plan.md).
6. **No Ad-Hoc Engine Attributes:** Never add `_last_*` or `_feedback_assimilated` fields on `AutonomousEpistemicEngine` or `view.state` (Rule 1.8).
7. **No Hardcoded Lexical Tables, High-Level `EXECUTE` Escapes, or Metadata-Only Skills:** Never add new Python verb/color keyword tables in `hcir/`, hide planning/reasoning logic behind `EXECUTE` syscalls, or save `SkillNode`s without executable `bytecode_stream`s (Rule 1.9).

---

## 6. Rust Substrate & Build Workflow (`rust/hcir_graph`)

Native PyO3 acceleration lives across 13 crates under `core/rust/` (including `hcir_graph`, `simulation_engine`, `structure_matcher`, `perception`, `compute_kernel`), each with an identical Python fallback:

```bash
cd core/rust/hcir_graph && maturin develop --release
```

- **Parity:** `NativeGraphAdapter` must match pure-Python semantics; divergences raise `SubstrateDivergenceError`.
- **Health & Fork/Diff:** `CognitiveGraph.native_health()` reports substrate status; `graph.fork()` is copy-on-write with `O(dirty)` `workspace.diff_branch()`. See `docs/architecture/graph_substrate.md`.

---

## 7. Codebase Architecture & Key Locations

```text
HBLLM/core/
├── AGENTS.md                        # Authoritative project instructions (< 24 KB rule limit)
├── .agents/
│   ├── rules/                       # Always-on agent rules (general_intelligence, homoiconic_symbolic_vm)
│   └── skills/                      # On-demand skills (hcir-symbolic-vm, embodied-construction-grammar, ...)
├── hbllm/
│   ├── hcir/                        # Pure-HCIR Cognitive OS (0 LLM imports)
│   │   ├── bytecode.py, interpreter.py, stdlib.py, compiler.py, nodes.py, graph.py
│   │   ├── workspace_tiers.py, learning_loop.py, kernel/, stores/, adapters/, algorithms/
│   │   ├── skills/                  # Declarative skill ASTs, induction & wake-sleep library learning
│   │   ├── world/                   # 5-stage CorticalPipeline & 17 subpackages + TransitionKB
│   │   ├── grounding/               # Symbol Grounding Engine (features, tracking, concepts)
│   │   ├── language/                # Non-LLM Multilingual Runtime (en/si/ta, Track T4 reading)
│   │   ├── reasoning/               # UnifiedReasoningRuntime + 14 operators
│   │   ├── snn/                     # LIF, STDP, STP, WTA, oscillations, perception
│   │   └── capabilities/            # Curiosity, BeliefConflictResolver, sandboxed executors
│   ├── runtime/bootstrap.py         # Single composition root (build_cognitive_system)
│   ├── brain/, native/, network/, security/, serving/
├── rust/, plugins/, benchmarks/suites/, graphify-out/, tests/
└── docs/
    ├── adr/0005-homoiconic-symbolic-vm-and-experience-first-cognition.md
    └── architecture/homoiconic-vm-migration-plan.md
```

---

## 8. Knowledge Graph Navigation (`graphify`)

1. **Query `graphify-out/` before broad greps:**
   ```bash
   graphify query "<concept or question>"
   graphify explain "<path/to/file.py>::<Symbol>"
   graphify path "<SymbolA>" "<SymbolB>"
   ```
2. **Keep in sync after substantial `.py` edits:** Run `graphify update .` (AST-only, zero LLM cost).

---

## 9. Engineering Standards, Skills & Workflows

- **Python 3.11+**, `from __future__ import annotations` at top of every `.py` file, strict static typing, Ruff/MyPy clean, preserve explanatory docstrings.
- **Workspace Skills:** Activate the relevant skill in `.agents/skills/` before modifying the ISA/VM (`hcir-symbolic-vm`), compiler/language (`embodied-construction-grammar`), world/ALUs (`cortical-alu-and-world-migration`), skill induction/replay (`wake-sleep-skill-consolidation`), or running final checks (`hcir-architecture-verification`).
- **Validation Commands (run from `HBLLM/core/`):**
  ```bash
  pytest tests/unit/hcir/ tests/unit/architecture/ -q
  pytest tests/unit/ -k "epistemic or cortex or hcir" -q
  python -m hbllm diagnostics
  python -m hbllm bench dev && python -m hbllm bench profile
  python -m hbllm kernel status
  ```
- **Ecosystem Context:** Multi-tenant admin web app (Laravel + Filament), cross-platform mobile (Flutter), Sentra adapters (`plugins/`).

---

## 10. Quick Decision Checklist for Agents

Before completing any task, verify:
1. [ ] **Code = CPU, Graph = Mind:** Is new cognition expressed as homoiconic HCIR programs/constructions/posteriors rather than hardcoded Python `if/else` or lexical tables?
2. [ ] **Migration Alignment:** Does the change move toward [ADR 005](docs/adr/0005-homoiconic-symbolic-vm-and-experience-first-cognition.md) and [`M0`–`M6`](docs/architecture/homoiconic-vm-migration-plan.md)?
3. [ ] **State Ownership & Tenancy:** Are all mutations committed via `TransactionManager` / `commit_operations` with proper `tenant_id` scoping?
4. [ ] **Homoiconic Skills:** Do learned/induced `SkillNode`s carry parameterized executable `bytecode_stream`s?
5. [ ] **LLM-Free & Domain-Agnostic Core:** Zero LLM imports, zero single-task hacks, only `priors.md` innate priors?
6. [ ] **Internal Memory Governance:** No ad-hoc `_last_*` or `view.state` attributes?
7. [ ] **Verified & Synced:** Relevant `pytest` + architecture tests passed, `AGENTS.md` `< 24 KB`, and `graphify update .` run after `.py` edits?
