# HBLLM Core — Deep Architecture Audit & Remediation Plan

**Date:** 2026-11-08  
**Scope:** `core/` (≈320k LOC Python; `hbllm/brain` ≈110k, `hbllm/hcir` ≈108k of which `hcir/world` ≈67k, `hcir/grounding` ≈7.2k)  
**Audience:** the agent that will execute the fixes. Every finding has file:line evidence, a target-state description (per AGENTS.md §1.6), and acceptance criteria. Work packages are ordered by dependency, not by ease.

**Baseline health (measured today):**
- `pytest tests/unit/hcir tests/unit/world -n 4` → **1476 passed, 2 skipped** (4m03s)
- `pytest tests/unit/hcir/grounding tests/integration/grounding tests/integration/world/test_cortical_grounding_integration.py tests/integration/world/test_sge_e2e_cognitive_loop.py` → **149 passed**
- `ruff check hbllm/hcir/grounding …` → clean; `mypy hbllm/hcir/grounding hbllm/hcir/world/cortical_grounding_bridge.py` → clean (52 files)
- Git tree clean at `03294a40`.

So the code *compiles, lints and passes its own tests*. The problems below are architectural: the system is **not yet a single general-intelligence loop**; it is four partially-overlapping cognitive stacks with no canonical entry point, and the newest one (grounding/SGE) is wired in a way that is both unsafe (bypasses governance) and inert (most of its outputs go nowhere).

---

## 0. Executive summary (read this first)

| # | Finding | Severity | WP |
|---|---------|----------|----|
| 1 | **Four disconnected "brains"**: LLM Brain (`hbllm.brain`+`serving`), Epistemic Engine (`hcir.world.AutonomousEpistemicEngine`), CognitiveBlackbox (`hbllm.drivers`), and Grounding/SGE (`hcir.grounding`). Each has its own concept learner, belief/evidence engine, causal model and world/entity tracker. No shared state, no single decision loop. | Critical | WP1, WP5 |
| 2 | **No canonical entry point.** `hbllm` CLI is the legacy transformer-training CLI (`data/train/serve/info/nodes`); `python -m hbllm.cli` runs *training*; nothing launches the HCIR kernel + epistemic engine. The real GI loop is only reachable through `plugins/arc_agi_adapter/my_agent.py`. | Critical | WP2 |
| 3 | **Epistemic engine runs without the kernel in every production path** (`AutonomousEpistemicEngine()` with `services=None` in ARC plugin and CognitiveBlackbox) → no `TieredWorkspace`, no `ConstitutionalVerifier`, no event log. Invariants 1 & 4 are not enforced where it matters. | Critical | WP1 |
| 4 | **Grounding bridge bypasses `ConstitutionalVerifier`**: when the engine has no TM it builds `TransactionManager(HCIRWorkspaceState())` with zero verification stages and commits into a private throwaway graph. | Critical | WP3 |
| 5 | **Grounding is mostly inert**: surprise never reaches `SurpriseEngine` (method `record_surprise` doesn't exist → silent `hasattr` no-op), induced causal schemas are never registered into the predictor, concept promotion is never triggered, `pending_epistemic_goals` is never consumed, `induce_relations()` is never called, velocity is never supplied so kinematic predictions are trivial. | High | WP3 |
| 6 | **Grounding is unbounded**: every step proposes O(N²) relation claims over *all* historical dyads and commits a fresh `BeliefNode` + 2 edges per accepted claim with new UUIDs → graph grows without bound; tracker trajectories, audit log, dyad histories never pruned. | High | WP3 |
| 7 | **Identity is positional, not perceptual**: tracklet ids are `trk_cortex_{color}_{idx}` (enumeration index), so any reordering re-binds tracklets to the wrong object; concept embeddings include absolute normalised position, so "concepts" are location clusters. | High | WP3 |
| 8 | **Domain hardcodes in core** (Invariant 7): `color / 10.0  # ARC uses 0-9 colour palette`; grid default 30×30; "level" vocabulary throughout AEE; `_map_predicate_to_edge_type` string table; `RelationLearner._seed_canonical_relations` hardcoded ON/IN/NEAR. | Medium | WP3, WP5 |
| 9 | **"Immutable context" is nominal** (Invariant 2): `CognitiveContext.signals["_engine"]` hands the live engine to every stage; `cortex_deliberation.py` writes `view.phase`, `view.mental_plan`, cooldowns directly (25 sites). Faculty journal (WP1) mitigates but does not enforce. | Medium | WP4 |
| 10 | **Ad-hoc engine attributes still appear** (Invariant 8): `self._pipeline_runs = getattr(self, "_pipeline_runs", 0)+1`, `self._last_decision_branch`, 5× `self._last_*` in AEE; `CorticalGroundingState` is a parallel ad-hoc working memory. | Medium | WP4 |
| 11 | **Dependency inversion**: `hcir/kernel/governance/governance_engine.py` imports `hbllm.brain.governance.*` and language lexicons → the kernel depends on the LLM-brain layer. | Medium | WP5 |
| 12 | **Direct SQLite in 48 modules** (Anti-pattern 2), 20+ inside `hbllm/brain/`. | Medium | WP6 |
| 13 | **Duplicate kernel wiring**: `serving/boot.py:210-267` builds a second `TransactionManager`/`KernelServices` *without* a verifier, independent of `BrainFactory._wire_hcir_cognitive_os`. HCIR `run_cycle()` is invoked *after* the LLM pipeline has already answered (`brain/core/factory.py:468-476`), errors swallowed at DEBUG. | Medium | WP1 |
| 14 | Hygiene: `hbllm/brain/concepts/` has no `__init__.py`; 25 scorecard/report JSON files, 3 `.db` files, 4 `.so` binaries and `knowledge_graph.json` (33 bytes) at repo root; `llama_venv/`, `venv/`, `.venv/` all present; no design doc for SGE though code cites "SGE Section 8/13". | Low | WP7 |
| 15 | **Total amnesia across runs**: persistent tier is `InMemoryEventStore()` everywhere; engine/theory/working memory have no save/load. | Critical (for brain-like operation) | WP8 |
| 16 | **No autonomous cognition**: loop is purely `step(obs)`-driven; no idle/offline consolidation rhythm, no clock. | High | WP9 |
| 17 | **No intrinsic value/goal arbitration** in the GI loop; homeostatic and GNW faculties ablated; LLM-side value machinery disconnected. | High | WP10 |
| 18 | **Language and world never bind**: `SEMANTIC_LLM_ADAPTER`, `bind_linguistic_alias`, `HCIRCompiler` have no runtime connection to the epistemic engine. | High | WP11 |
| 19 | Perception is 2-D-grid/avatar-shaped; no cross-context skill transfer; no latency budget for real-time embodiment. | Medium | WP12–14 |

---

## 0b. Phase 1 scope freeze (binding for the executing agent)

**Rule: until Phase 1 is complete, no new faculty, module, benchmark scenario or cognitive capability may be added to `hbllm/`.** The codebase must get *smaller and truer*, not larger. Phase 1 consists of exactly:

1. **WP0** — frozen held-out evaluation harness (so every later change is measurable)
2. **WP1** — single composition root + governance everywhere
3. **WP2** — entry points
4. **WP3** — grounding made safe, bounded and connected
5. **WP5-D** — the deletion list (duplicate stacks removed, LOC target met)
6. **WP15** — the learned-perception *decision* (not implementation): re-scope Invariant 3, land the SNN-backed `FeatureExtractor` behind a flag, run the comparison on the WP0 held-out set

Exit criteria for Phase 1: WP0 report shows dev/held-out numbers for the frozen agent; `rg "TransactionManager\(" hbllm` has one construction site; `hbllm mind step` works; grounding graph growth plateaus; `hbllm/` LOC ≤ 250k (from ≈320k); AGENTS.md Invariant 3 re-scoped; `reports/perception_comparison.md` exists with a decision. Only then do WP4, WP6–WP14 open.

---

## 1. System map as it actually exists

```
                     ┌──────────────── production paths ────────────────┐
                     │                                                   │
 hbllm serve ──► serving/api.py ─► BrainFactory.create() ─► hbllm.brain.* (LLM nodes: Router, Planner, …)
                                   │                          │  own: belief_store, causal_graph, concepts,
                                   │                          │       scene_graph, epistemics/*  (SQLite)
                                   │                          └─► pipeline.process() ──► answer
                                   │                                          └─► hcir_runtime.run_cycle()  (post-hoc, no candidates)
                                   ├─► _wire_hcir_cognitive_os(): KernelServices#1 (with ConstitutionalVerifier)
                                   └─► serving/boot.py:          KernelServices#2 (NO verifier)   ← duplicate
                                   └─► DriverManager.set_cognitive_engine(CognitiveBlackbox(workspace=hcir_ws))
                                                                      │
                                                                      └─► get_epistemic_engine(): AutonomousEpistemicEngine()  ← services=None
                                                                                                   .attach_grounding_bridge()   ← private TM, no verifier

 plugins/arc_agi_adapter/my_agent.py ─► AutonomousEpistemicEngine(instructions, seed)   ← services=None
                                        .attach_grounding_bridge()                        ← private TM, no verifier
                                        .step(EmbodiedObservation) ─► CorticalPipeline.run(ctx)
                                                                       ├ PerceptionStage ─► bridge.process_perception()
                                                                       ├ AssimilationStage ─► bridge.process_assimilation()
                                                                       ├ SimulationStage
                                                                       ├ DeliberationStage  (mutates engine via ctx.signals["_engine"])
                                                                       └ MotorStage ─► bridge.process_motor()
```

Observations:
- The **only** place the AGENTS.md §2 cortical loop runs is `AutonomousEpistemicEngine.decide()` (`hcir/world/autonomous_epistemic_engine.py:4972`). It is reached from exactly two callers outside tests: `plugins/arc_agi_adapter/my_agent.py:804` and `hbllm/drivers/cognitive_blackbox.py` (per source id).
- `ExecutiveRuntime.run_cycle()` (`hcir/kernel/executive_runtime.py:99`) is a *different* loop (`KernelExecutiveController`) that never touches the AEE.
- `hbllm.brain` has a *third* epistemic loop (`brain/epistemics/epistemic_loop.py`, `brain/control/executive_controller.py`).
- There is no `hbllm/__main__.py`; `python -m hbllm.cli` executes `hbllm.cli.train.main` (`hbllm/cli/__main__.py`), not the dispatcher.

---

## 2. Grounding module (`hbllm/hcir/grounding`) — detailed review

### 2.1 What is good (keep)
- Clean layering: `contracts/` (frozen dataclasses, typed refs `TrackletRef → EntityCandidateRef → GroundedEntityRef`), `bus/` (bounded ring buffers, sequence checks), `tracking/` (reversible merge/split with audit ops), `epistemics/` (calibration → dependency-group fusion → contradiction → decision), `committer/` (only component that touches the graph, via `TransactionManager.commit`). This is the right shape for Invariant 1.
- 100% LLM-free, ruff/mypy clean, 149 tests.

### 2.2 Integration defects (bridge: `hbllm/hcir/world/cortical_grounding_bridge.py`)

| Line(s) | Defect | Target state |
|---|---|---|
| 100-101 | `self.workspace = workspace or HCIRWorkspaceState(); self.tx_manager = transaction_manager or TransactionManager(self.workspace)` → unverified TM on a private graph. | Bridge must **require** `KernelServices` (or at minimum a TM that already carries the verification pipeline). Never construct a TM. If no services: raise, don't degrade. |
| 193 | `tracklet_id = TrackletRef(f"trk_cortex_{color}_{idx}")` | Tracklet identity must come from the perception stage's persistent object ids (`SpatialEntity` already has stable ids via `spatiotemporal_tracker.py`/`object_state_graph.py`); fall back to appearance+motion association (Hungarian/greedy on feature distance + predicted position), never enumeration index. |
| 209-227 | `relation_learner.propose_claims()` over *all* dyad histories each step, every ACCEPT committed via `commit_decision` → new `BeliefNode`+`SUPPORTS` edge+relation edge with fresh UUIDs per step. | Deterministic ids: `belief_id = hash(subject, predicate, object, polarity)`; `UPSERT` (not `ADD`) and only commit on **belief change** (new, polarity flip, confidence delta > ε, or retraction). Propose claims only for dyads touched in the current observation. |
| 259 | `if step_counter % 5 == 0: concept_learner.maintain()` | Fine, but also call `ConceptLifecycleManager.should_propose_promotion()` and `committer.commit_concept_promotion()` — currently **never called** anywhere outside tests. |
| 334-344 | `if hasattr(self.surprise_engine, "record_surprise")` — `SurpriseEngine` (`hcir/world/surprise_engine.py:58`) has no such method → permanently silent. | Add a real, typed API on `SurpriseEngine` (e.g. `ingest_external_prediction_error(score: float, source: str)`) and call it unconditionally. Remove `hasattr` guards; they hide integration failures. |
| 120-122 | `ForwardWorldPredictor()` created with no schemas; `SchemaInducer` is only used by `ActiveInferenceFeedback`. | Predictor must consume `schema_inducer` schemas (same instance or `predictor.register_schema` on `induce()`), **and** should delegate to the engine's existing `HCIRSymbolicWorldTheory`/`predictive_coding_cortex` instead of maintaining a second physics model. |
| 201-206, 444-450 | `entity_states` never carries `velocity`; `predict_next_state` therefore predicts "stays put" for everything → every moving entity is a "surprise". | Compute velocity from tracker trajectory (`spatial_trajectory[-1] - [-2]`) or take it from `dorsal_kinetic_stream`. |
| 331 | `self.state.pending_epistemic_goals.append(goal)` — never read. | Route `GoalProposal`s into the engine's goal system (`frontopolar_subgoal` / `BayesianGoalInductionEngine`) via `PrefrontalWorkingMemory`, or into the TieredWorkspace as `TaskFrame`s. |
| 372-377, 413-417 | Default target `"e_avatar"` fabricated with confidence 1.0 at (0,0); all tracked entities get confidence 0.95 hard-coded. | Use the engine's avatar identity (`ctx.avatar_pos`, `avatar_feature`) and the tracker's actual association confidence. |
| 472 | `report_execution(..., success=True)` before verification. | Success must come from the actuator/driver feedback (`DriverFeedback`, or grid diff ≠ ∅), not assumed. |
| 395 | Always `KinematicRenderer` regardless of action kind. | Select renderer from `ActuatorSchema.kind` (`EmbodiedObservation.actuators`). |
| 62-75 | `CorticalGroundingState` dataclass = ad-hoc working memory. | Fold into `PrefrontalWorkingMemory` (Invariant 8) — a `grounding` slot with the same retention policy machinery (`FIELD_RETENTION`, `apply_boundary_retention`). |

### 2.3 Internal defects in `hcir/grounding/*`

| File:line | Defect | Target state |
|---|---|---|
| `features/feature_extractor.py:135-140` | Embedding = `(row/grid_h, col/grid_w, color/10, log1p(size))`; comment `# ARC uses 0-9 colour palette`; `grid_h/grid_w` default 30. | Position must **not** be in the concept embedding (translation invariance). Use shape/topology features already available in core: Hu moments (`ShapeArchetypeSignature`, AEE:2389), bbox aspect, hole count, color *as categorical one-hot or hashed*, not `/10`. Scale normalisers must come from `ObservationEnvelope` (lattice shape), never defaults. |
| `features/feature_extractor.py:289` | Text embedding = bag-of-chars hash stub. | Either plug the existing `hbllm.perception`/`semantic` encoders behind the `FeatureExtractor` protocol, or mark the extractor experimental and exclude it from `FeatureExtractorRegistry.default()`. |
| `relations/relation_learner.py:58-95` | Seeded "canonical" relations with English alias tables; `_match_relation` is an if-ladder. | Keep `TopologicalInvariant` as the primitive (that's domain-general); aliases must be late-bound by `bind_linguistic_alias` from the language layer, not seeded. Replace the ladder with a declarative predicate table (`SymbolicPredicate` from `hcir/skills/declarative.py`). |
| `relations/relation_learner.py:190-219` | `induce_relations()` never invoked by any runtime path. | Call from `process_perception` after `observe_entity_states`; cap `_dyad_histories` (LRU by last timestamp). |
| `committer/committer.py:303-331` | `_map_predicate_to_edge_type` hard-coded string→enum table. | Carry the `HCIREdgeType` on the `LearnedRelation`/`TopologicalInvariant` itself (one source of truth). |
| `committer/committer.py:73-82, 89-98` | Every commit upserts `PhysicalEntityNode(id, entity_name=id)` with *no* properties → overwrites any richer node data already in the graph. | Upsert only when node is absent, or merge properties (`UPDATE_NODE` with `changes`), never blind overwrite. |
| `tracking/tracker.py` | `spatial_trajectory`, `_audit_log`, `_tracklets` unbounded; tracklets never deactivated when an entity vanishes; `auto_associate` returns existing binding forever (line 132-134) so a reused tracklet id can never re-associate. | Bounded deques; occlusion timeout → `is_active=False`; re-association when distance > threshold for k steps (emit `EntitySplitOp`). |
| `epistemics/evaluator.py:36` vs bridge `:97` | Default acceptance 0.85, bridge passes 0.65 with a single evidence group → nearly everything is ACCEPTed. | Make thresholds a `KernelServices`-provided policy (constitutional), and require ≥2 independent dependency groups *or* temporal persistence ≥ k frames for ACCEPT. |
| `actions/gateway.py:82` | Irreversibility check = `action_type.upper() in ("DELETE","DESTROY","PURGE","OVERWRITE_ALL")`. | Reversibility must be derived from the `CausalSchema`/`TransactionManager` undo-log availability, not a word list. |
| `actions/gateway.py:197-234` | Precondition/outcome mini-language parsed by `split("==")`. | Use `CausalPredicate` (`cortex_causal.py`) — the project already has a typed predicate; don't invent a second one. |
| `__init__.py` | Re-exports everything (90 symbols); `HCIRCommitter` importable by anyone. | Export contracts + bridge factory only; keep `committer` kernel-internal. |

### 2.4 Duplication the grounding module introduced (must be reconciled, not deleted blindly)

| Capability | `hcir/grounding` | Pre-existing in `hcir/world` | Pre-existing in `hbllm/brain` |
|---|---|---|---|
| Entity identity / tracking | `tracking/tracker.py` | `spatiotemporal_tracker.py`, `object_state_graph.py` | `brain/world/entity_registry.py`, `object_permanence.py` |
| Concept formation | `concepts/concept_learner.py` | `representation_expansion.py`, `rule_induction.py` | `brain/concepts/*` (6 files, no `__init__`), `brain/reasoning/concept_formation.py` |
| Causal schemas | `causal/schema.py` | `cortex_causal.py` (`HCIRSymbolicWorldTheory`, `CausalPredicate`) | `brain/causality/causal_graph.py`, `causal_model_builder.py` |
| Belief / evidence | `epistemics/*` | `cortex_episodic.py`, `relational_confidence_map` | `brain/epistemics/{belief_manager,calibration,contradiction_engine,evidence_evaluator}.py` |
| Prediction & surprise | `predictive/*` | `surprise_engine.py`, `predictive_coding_cortex`, `snn_predictor` | `brain/epistemics/prediction_tracker.py` |
| Spatial relations | `relations/*` | `spatial_containment.py`, `relational_graph_matcher.py` | `brain/world/spatial_ontology.py`, `scene_graph.py` |

**Decision required (recommendation):** make `hcir/grounding/contracts` + `bus` + `tracking` + `committer` the *canonical sensorimotor front-end* (they are the best-typed versions), make `hcir/world/cortex_causal.HCIRSymbolicWorldTheory` the *single* causal/predictive model, and make `hcir/world/surprise_engine` the single surprise source. Then delete/thin `grounding/predictive`, `grounding/causal/schema.py` (keep `events.py` detector), and route `grounding/epistemics` decisions into `HCIRSymbolicWorldTheory` predicates. `brain/*` duplicates become adapters over HCIR (see WP5).

---

## 3. Core invariant audit (`hbllm/hcir/world`, `hcir/kernel`)

### 3.1 Invariant 1 (State ownership) — PARTIAL
- ✅ No raw `workspace.add_node/upsert_node/...` calls outside the TM (grep clean). Deprecation shim fires in 6 old tests (`tests/unit/hcir/test_*counterfactual_planner.py`) — fix those tests.
- ❌ Engine state lives in Python attributes with **write-through + journal rollback** (`commit_faculty_journal`, AEE:5254). Rollback is by `repr()` comparison; `_MIRRORED` set is a hand-maintained list. Target: the journal should be generated from `PrefrontalWorkingMemory` field declarations, not a literal set.

### 3.2 Invariant 2 (Operators receive immutable snapshots) — VIOLATED BY DESIGN
- `decide()` puts `"_engine": self` in `ctx.signals` (AEE:5008). All five stages reach through it. `cortex_deliberation.py:1285-1985` performs 25 direct writes (`view.phase = …`, `view.mental_plan = deque(...)`, cooldown resets).
- Target: stages return `StageResult.operations` / `signals` only; `MotorStage`/engine applies them. Introduce typed `WorkingMemoryDelta` ops (`SET_PHASE`, `SET_PLAN`, `RESET_COOLDOWN`) and apply them in `commit_faculty_journal`. Remove `"_engine"` from signals; expose read-only `FrozenGraphView`/`CognitiveContext` fields instead.

### 3.3 Invariant 3 (LLM-free core) — HOLDS
- No `openai/anthropic/transformers/httpx/torch` imports in `brain/reasoning`, `hcir/world`, `hcir/grounding`, `hcir/kernel`.

### 3.4 Invariant 4 (Constitutional governance) — NOT ENFORCED IN PRODUCTION PATHS
- `ConstitutionalVerifier` is only attached in `BrainFactory._wire_hcir_cognitive_os` (factory.py:1213-1215). `serving/boot.py:227` creates a bare TM; `CorticalGroundingBridge` creates a bare TM; `AutonomousEpistemicEngine()` without services has `transaction_manager=None` so `CorticalPipeline` skips the batch commit entirely (`cortical_pipeline.py:147`).
- Target: a single `KernelServices.create_default()` is the **only** way to obtain a TM, and it always installs `create_default_pipeline(ConstitutionalVerifier)`. `TransactionManager.__init__` should refuse `verification_stages=None` outside tests (or log at WARNING, never DEBUG).

### 3.5 Invariant 7 (Task-agnostic core) — MOSTLY HOLDS, WITH LEAKS
- Leaks: `feature_extractor.py:138` ARC palette; `cortex_perception.py:151` and `percept.py:6/359`, `macro_actions.py:950`, `grid_operator.py:3`, `visual_symmetry.py:4` mention ARC in docstrings (acceptable as examples) but `grid_operator.py` is described as "common operator contract for ARC-AGI abstract visual rule induction" — review whether it belongs in `plugins/arc_agi_adapter`.
- AEE vocabulary: `current_level`, `level_learned_barriers`, `level_lethal_positions`, `level_failed_transitions`, `CrossGameKnowledgeTransfer`, `EpisodeBoundaryKind.LEVEL_COMPLETE`. These are *episode/context* concepts wearing game clothing. Rename toward `context_id` / `EpisodeBoundaryKind.GOAL_REACHED` and key stores by `context_id: str` not `level: int`.

### 3.6 Invariant 8 (No ad-hoc engine attributes) — MINOR VIOLATIONS
- AEE:4999 `self._pipeline_runs = getattr(self, "_pipeline_runs", 0) + 1`; AEE:5024 `self._last_decision_branch`; 5× `self._last_*` assignments; `global_workspace_ignition.py` 1×. Move to `PrefrontalWorkingMemory` / telemetry.
- `relational_confidence_map`, `evidence_provenance_log` (AEE:1013-1014) are bare dicts/lists on the engine — should be `HCIRSymbolicWorldTheory` predicates or `TieredWorkspace` nodes.

### 3.7 Kernel-layer dependency inversion
- `hcir/kernel/governance/governance_engine.py:17-18, 275, 287` imports `hbllm.brain.governance.*` and `hbllm.brain.language.{sinhala,tamil}.lexicon`. The kernel must not depend on the brain layer. Move `OwnerRuleStore`/`PolicyEngine` under `hcir/kernel/governance/policies/` (they are governance primitives) and make lexicon checks a pluggable `IVerificationStage` registered from the language layer.

### 3.8 Ablated faculties
- `hcir/world/` still contains **139 files / 65k LOC**; `experimental/` holds only a lazy registry. The 52 ablated faculties are physically still in the core package and importable. Move their modules under `hcir/world/experimental/` (matching the AGENTS.md note) so the core package size reflects the measured architecture, and so tests can gate on "core imports no experimental module".

---

## 4. Entry points & flow — current vs target

### Current
| Command | What it actually does |
|---|---|
| `hbllm` (pyproject script → `hbllm.cli:main` → `_cli_app.main`) | argparse: `data`, `train` (transformer pretraining), `serve` (uvicorn LLM API), `info`/`nodes` (prints a 28-node LLM-brain banner), `plugin`, `agent` (LLM developer REPL), `code`, `daemon`. **No kernel, no epistemic engine, no benchmark, no world/driver loop.** |
| `python -m hbllm.cli` | runs `hbllm.cli.train.main` (training!) — contradicts the script entry. |
| `python -m hbllm` | fails: no `hbllm/__main__.py`. |
| `python -m hbllm.cli.diagnostics` | hardware/Rust probes only; no cognitive subsystem status despite docstring. |
| `python core/plugins/arc_agi_adapter/run_local_suite.py` | the only way to exercise the full cortical loop end-to-end. |
| `python -m benchmarks.sge_integration_benchmark` | grounding micro-benchmarks (bridge in isolation). |

### Target (WP2)
```
hbllm                       # typer app (already a dependency) — single dispatcher
├── kernel   boot|status|replay          # KernelServices.create_default(), ExecutiveRuntime, event-log replay
├── mind     run --driver <name> [--obs-source …]   # AEE + grounding + drivers loop (the GI loop)
│            step --obs file.json        # one decide() on a serialized EmbodiedObservation
├── bench    local-suite|ablation|sge    # moves run_local_suite / benchmarks/* behind one verb
├── serve                                # existing API, but boot through the same kernel factory
├── agent / code / daemon                # existing LLM-agent verbs, unchanged
├── train / data                         # legacy transformer verbs → move under `hbllm llm …` or deprecate
└── diagnostics                          # + cognitive status (kernel online, verifier attached, faculties loaded, substrate health)
```
- `hbllm/__main__.py` → `from hbllm.cli import main; main()`. Fix `hbllm/cli/__main__.py` to call the dispatcher.
- Introduce **one composition root**: `hbllm/runtime/bootstrap.py::build_cognitive_system(config) -> CognitiveSystem` returning `{services, engine, bridge, driver_manager, brain?}`. `serving/boot.py`, `BrainFactory._wire_hcir_cognitive_os`, `CognitiveBlackbox.get_epistemic_engine`, `my_agent.py` and the CLI all call this and nothing else. Delete the duplicate wiring in `boot.py:210-267`.

### Canonical runtime flow (target)
```
Driver/Env ─► EmbodiedObservation ─► ObservationBus (grounding) ─► PerceptionStage (lifts SpatialEntity w/ stable ids)
          ─► Tracker (identity) ─► RelationLearner/EventDetector ─► EpistemicEvaluator ─► HCIRCommitter ─► TransactionManager(+Verifier) ─► TieredWorkspace
          ─► AssimilationStage (SurpriseEngine ← grounding prediction error) ─► SimulationStage (HCIRSymbolicWorldTheory as the single forward model)
          ─► DeliberationStage (pure; emits WorkingMemoryDelta ops) ─► MotorStage ─► ActionValidationGateway ─► CapabilityResolver ─► Driver.act()
          ─► DriverFeedback ─► gateway.report_execution(success=from feedback) ─► verify_outcome ─► credit assignment → HCIRSymbolicWorldTheory
LLM Brain (hbllm.brain) attaches only as: (a) HCIRCompiler meta-compiler producing InstructionStreams/skills, (b) a sensory channel (ModalitySource.SEMANTIC_LLM_ADAPTER) on the ObservationBus, (c) a renderer for DialogueRenderer output.
```

---

## 5. Work packages (ordered; each is independently mergeable)

### WP0 — Frozen held-out evaluation harness (before everything else)
**Why first.** The project's only external number is 10/10 on `plugins/arc_agi_adapter/run_local_suite.py`, and `git log` shows that suite being tuned against game by game (`b2df4cc7`, `ab151ed2`). `benchmarks/` contains only `ablation/faculty_ablation.py`, `sge_integration_benchmark.py` and 4 synthetic scenarios; `rg -i "held.?out|ood|unseen" benchmarks plugins/arc_agi_adapter` → 0. Nothing in the repo can currently distinguish "general" from "tuned". Every later WP must be judged against this harness, so it goes first.

**Design**
1. `benchmarks/suites/` with three environment families, each split `dev/` and `heldout/`:
   - `grid/` — the existing local ARC games + `plugins/{sokoban,babyai,crafter}_adapter` levels; dev = the current 10 games, **heldout = games/levels never run during development**.
   - `continuous/` — `benchmarks/scenarios/robotics_manipulation.py` style tasks and `plugins/{ai2thor,safety_gym}_adapter` where available.
   - `symbolic/text/` — `plugins/{alfworld,nethack}_adapter` or the `language_grounding.py` scenario.
2. `hbllm/benchmarks/protocol.py` (already a package) defines `EvaluationProtocol`: frozen agent build (`build_cognitive_system(config)` from WP1, config hash recorded), fixed seeds, per-task **steps-to-first-success**, **steps-to-stable-policy**, **success@budget**, wall-clock p50/p95 of `decide()`.
3. `hbllm bench dev` may be run freely. `hbllm bench heldout` writes `reports/heldout/<git_sha>_<date>.json` and **refuses to run twice on the same sha** (file exists → exit 1). This is the mechanism that keeps the held-out set held out.
4. Declared priors: `docs/architecture/priors.md` lists the innate knowledge the core may assume (objectness, 2-D/3-D topology, agency detection, counting, elementary physics). Anything not on the list must be learned in-environment; reviewers check new core code against it.
5. Generality score = mean normalised steps-to-first-success on `heldout/` across all families with the *same* frozen agent and *no* per-family modules; reported alongside dev score so the gap is visible.

**Acceptance:** `hbllm bench dev` and `hbllm bench heldout` run end-to-end on the current (pre-WP1) agent and produce the first baseline report; `tests/unit/benchmarks/test_protocol.py` covers the refuse-to-rerun rule and config-hash recording; CI runs `dev` only.

### WP1 — Single composition root & governance everywhere (prereq for all others)
1. Create `hbllm/runtime/bootstrap.py` with `build_cognitive_system(config: CognitiveSystemConfig) -> CognitiveSystem`. It must call `KernelServices.create_default(workspace, tiered_workspace=TieredWorkspace())` (verifier included), construct `AutonomousEpistemicEngine(services=services)`, and `engine.attach_grounding_bridge(CorticalGroundingBridge(services=services))`.
2. Make `AutonomousEpistemicEngine.__init__` **require** `services` (keep a `for_tests()` classmethod that builds default services, so 1476 tests keep passing with a one-line fixture change in `tests/conftest.py`).
3. Replace the wiring in `serving/boot.py:210-267`, `brain/core/factory.py:_wire_hcir_cognitive_os`, `drivers/cognitive_blackbox.py:651-657`, `plugins/arc_agi_adapter/my_agent.py:244-246` with calls to the bootstrap.
4. `TransactionManager.__init__`: if `verification_stages` is empty, log **WARNING** `"TransactionManager created without verification pipeline (author=…)"` and set `self.unverified = True`; `KernelServices.__post_init__` asserts `not tm.unverified`.
5. `brain/core/factory.py:468-476`: the HCIR cycle must run **before** `pipeline.process()` and feed `candidate_actions`; errors at WARNING not DEBUG.
- **Acceptance:** grep shows exactly one `TransactionManager(` construction site outside tests and `kernel/`; `pytest tests/unit/hcir tests/unit/world tests/integration/world -n 4` green; new test `tests/integration/runtime/test_bootstrap_governance.py` asserts a grounding commit with a policy-violating node is **rejected** by the verifier in the ARC-plugin configuration.

### WP2 — Entry points
1. Add `hbllm/__main__.py`; fix `hbllm/cli/__main__.py`; migrate `_cli_app.py` argparse to a `typer` app (`typer` already in deps) with the verb tree in §4.
2. `hbllm mind run --driver <name>`: uses `DriverManager` + bootstrap; `hbllm mind step --obs <json>` for deterministic single-step debugging; `hbllm bench local-suite` wraps `plugins/arc_agi_adapter/run_local_suite.py`; `hbllm bench sge` wraps `benchmarks/sge_integration_benchmark.py`.
3. `hbllm diagnostics`: add kernel/verifier/faculty/substrate (`graph.native_health()`) status.
4. Update `run_info` banner (it describes the 2025 LLM-node design, not the Cognitive OS). Update README "Quick start" and AGENTS.md §9 commands.
- **Acceptance:** `hbllm --help` lists the tree; `python -m hbllm mind step --obs tests/fixtures/obs_minimal.json` prints an action; `tests/unit/cli/test_entrypoints.py` covers each verb with `CliRunner`.

### WP3 — Make grounding safe, bounded and *connected*
(All items in §2.2 and §2.3. Highlights:)
1. Bridge takes `services: KernelServices` (no fallbacks); `CorticalGroundingState` → `PrefrontalWorkingMemory.grounding`.
2. Stable identity: consume `SpatialEntity.id` from perception; appearance+motion association; occlusion timeout; bounded buffers.
3. Deterministic belief ids + upsert-on-change; claims only for touched dyads; `induce_relations()` in the loop; `_dyad_histories` LRU.
4. Real `SurpriseEngine` API; predictor delegates to `HCIRSymbolicWorldTheory`; velocity from tracker; schemas flow inducer → theory.
5. Concept promotion wired (`should_propose_promotion` → `commit_concept_promotion`), `GoalProposal`s routed into goal induction.
6. Feature embedding translation-invariant; palette/grid-size from envelope; remove ARC comment.
7. `report_execution(success=…)` from `DriverFeedback` / observed diff; renderer chosen by `ActuatorSchema.kind`.
8. Thresholds from a `GroundingPolicy` dataclass owned by `KernelServices` (constitutional).
- **Acceptance:** new tests — (a) 200-step synthetic loop: `len(workspace.graph.nodes)` and edges plateau (assert bounded growth), (b) permuting entity order between frames does not change candidate ids, (c) a moving entity with constant velocity yields zero surprise alerts after warm-up, (d) `SurpriseEngine` receives grounding prediction error (no `hasattr`), (e) a concept with ≥N exemplars and quality ≥ threshold appears as `GroundedConceptNode` in the graph, (f) translation of the whole scene by (+5,+5) assigns the same concept ids. Existing 149 grounding tests stay green (update ones that encode old behaviour).

### WP4 — Enforce Invariant 2/8 inside the cortical loop
1. Remove `"_engine"` from `CognitiveContext.signals`; give stages the read-only fields they need (`CognitiveContext` already has 56 avatar/grid fields; extend as required).
2. Introduce `WorkingMemoryDelta` ops; convert the 25 writes in `cortex_deliberation.py` (and any in planner/motor) into returned ops applied by the engine in `commit_faculty_journal`.
3. Generate the journal's `_MIRRORED` set from `PrefrontalWorkingMemory` field metadata.
4. Move `_pipeline_runs`, `_last_decision_branch`, `_last_*`, `relational_confidence_map`, `evidence_provenance_log` into working memory / symbolic theory / telemetry.
- **Acceptance:** `rg '"_engine"' hbllm/hcir/world` → 0; `rg 'view\.[a-z_]+\s*=[^=]' hbllm/hcir/world/cortex_*.py` → 0; a test proves that a stage raising mid-way leaves `engine.working_memory` byte-identical (journal rollback) — currently only `repr` equality is checked.

### WP5 — Converge the four stacks on HCIR (general-intelligence refactor)
1. **Causal/predictive:** `HCIRSymbolicWorldTheory` is the single forward model. `grounding/causal/schema.py::CausalSchema` → thin adapter producing `CausalPredicate`s; `brain/causality/*` → adapter reading HCIR predicates (deprecate its SQLite store).
2. **Concepts:** `grounding/concepts` is canonical for *perceptual* concepts; `brain/concepts/*` (add `__init__.py` first, or delete if unused — only `experiment/cohorts.py` and one test import it) and `brain/reasoning/concept_formation.py` become consumers of `GroundedConceptNode`s via `TieredWorkspace` queries.
3. **Beliefs:** `grounding/epistemics` is canonical for sensory beliefs; `brain/epistemics/belief_manager` etc. operate on `BeliefNode`s in the workspace rather than their own SQLite.
4. **Entities/world:** `brain/world/{entity_registry,object_permanence,scene_graph,world_state}` → read models over the HCIR graph (`PhysicalEntityNode`, `SAME_AS`, `LOCATED_IN`).
5. **Governance inversion:** move `OwnerRuleStore`/`PolicyEngine` into `hcir/kernel/governance/policies/`; lexicon checks become a registered `IVerificationStage` from `hbllm.brain.language`.
6. **LLM role:** `hbllm.brain` nodes attach via (a) `HCIRCompiler` (NL → `InstructionStream`), (b) `ObservationBus` channel `ModalitySource.SEMANTIC_LLM_ADAPTER`, (c) `DialogueRenderer`. The brain's own `epistemic_loop`/`executive_controller` loops are retired in favour of `ExecutiveRuntime.run_cycle()` → `AutonomousEpistemicEngine.decide()`.
7. **Vocabulary de-gaming:** `level` → `context_id`-keyed stores; `EpisodeBoundaryKind.LEVEL_COMPLETE` → `GOAL_REACHED`; review `grid_operator.py` for relocation to `plugins/arc_agi_adapter`.
8. Move the 52 ablated faculty modules under `hcir/world/experimental/`.
- **Acceptance:** `rg "from hbllm\.brain" hbllm/hcir` → 0; architecture test `tests/unit/architecture/test_layering.py` (import-linter style) enforcing `hcir.kernel ↛ brain`, `hcir.world ↛ brain`, `hcir.grounding ↛ brain`, `core ↛ plugins`; local suite score (`hbllm bench local-suite`) ≥ current 10/10 (`reports/local_suite/`).

### WP5-D — Deletion list & LOC target (Phase 1; do not "adapt" what nobody imports)
WP5 above says "become adapters". Measured today, most duplicates have **no production importer outside `hbllm/brain`** — they are kept alive only by their own tests. For those, the target state is deletion, not adaptation. (`prod` = importers in `hbllm/` + `plugins/` excluding `hbllm/brain/**`; `tests` = test files importing them. Internal `hbllm/brain` importers must be checked and migrated as part of the deletion.)

| Module | prod | tests | Decision |
|---|---|---|---|
| `brain/concepts/*` (6 files, no `__init__.py`) | 1 (`experiment/cohorts.py`) | 2 | **Delete.** Perceptual concepts → `hcir/grounding/concepts`; move `cohorts.py` dependency to `GroundedConceptNode` queries. |
| `brain/world/scene_graph.py`, `object_permanence.py`, `entity_registry.py`, `spatial_ontology.py` | 0 | 1 each | **Delete.** Superseded by `hcir/grounding/tracking` + `relations` and `hcir/world/{spatiotemporal_tracker,object_state_graph,spatial_containment}`. |
| `brain/world/world_state.py` | 0 | 6 | **Delete** after migrating the 6 tests to `TieredWorkspace`/`PhysicalEntityNode` read models. |
| `brain/causality/*` (3 files, SQLite) | 0 | 5 | **Delete.** `HCIRSymbolicWorldTheory` is canonical. |
| `brain/epistemics/{belief_manager,evidence_evaluator,calibration}.py` | 0 | 5/1/4 | **Delete.** `hcir/grounding/epistemics` is canonical. |
| `brain/epistemics/contradiction_engine.py` | 1 | 4 | **Replace** the single importer with `BeliefRevisionEngine`, then delete. |
| `brain/epistemics/epistemic_loop.py`, `brain/control/executive_controller.py` | 0 | 5/1 | **Delete.** `ExecutiveRuntime.run_cycle → AutonomousEpistemicEngine.decide` is the one loop. |
| `brain/reasoning/concept_formation.py` | 0 | 3 | **Delete** (perceptual) or fold into `hcir/skills/declarative` if it does symbolic rule abstraction — read before deleting. |
| `hcir/grounding/predictive/*`, `hcir/grounding/causal/schema.py` | bridge only | yes | **Thin to adapters** over `HCIRSymbolicWorldTheory` / `SurpriseEngine` (WP3.4); keep `causal/events.py`. |
| 52 ablated faculties in `hcir/world/` | lazy | yes | **Move** to `hcir/world/experimental/` (WP5.8); any with zero test coverage → delete. |
| Root clutter: 25 scorecard JSONs, `*.db`, `*.so`, `knowledge_graph.json`, `venv/`, `llama_venv/` | — | — | **Move/ignore/delete** (WP7; confirm `venv` removal with owner). |
| `hbllm/_cli_app.py` `run_info`/`run_nodes` banners, `hbllm/cli/train.py` as `__main__` | — | — | **Rewrite** under WP2; `train`/`data` verbs move to `hbllm llm …` or are removed if the transformer stack is no longer a goal (owner decision). |

**LOC target:** `hbllm/` from ≈320k to **≤ 250k** by end of Phase 1 (measured with `fd -e py . hbllm | xargs cat | wc -l`), and `hcir/world/` from 65k to ≤ 40k excluding `experimental/`. Record before/after in `reports/audit/loc_<date>.txt`.

**Acceptance:** every row above has a commit; `tests/unit/architecture/test_no_duplicate_faculties.py` asserts exactly one importable implementation each for entity tracking, concept learning, belief revision, causal theory, surprise; LOC report committed.

### WP6 — Memory access discipline
- 48 modules call `sqlite3.connect`/`aiosqlite.connect` directly (list: `rg -l "sqlite3\.connect|aiosqlite\.connect" hbllm --glob '!hbllm/hcir/stores/**'`). Route each through `HCIRMemoryBackend` / `hcir/stores`. Prioritise the 20 in `hbllm/brain/` (epistemics, causality, self_model, learning, skills) because they hold *cognitive* state that should live in the TieredWorkspace.
- **Acceptance:** the grep returns only `hbllm/hcir/stores/**`, `hbllm/persistence/**`, `hbllm/security/**` (infrastructure), and a lint rule (ruff `banned-api` via `flake8-tidy-imports`) blocks regressions.

### WP7 — Hygiene & docs
- Move root scorecards/reports into `reports/`; `.gitignore` `*.db`, `*.so` at root, `knowledge_graph.json`; remove `venv/` and `llama_venv/` from the tree (confirm with owner first — destructive).
- Write `docs/architecture/grounding.md` (the "SGE" spec the code cites does not exist in-repo) and `docs/architecture/entry-points.md` (the §4 target).
- Fix the 6 tests using deprecated `ws.upsert_node`.
- Run `graphify update .` after each WP (AGENTS.md §8).

---

## 5b. Beyond plumbing — what is still missing for brain-like operation (WP8–WP11, WP12–14 generality)

WP1–WP7 produce *one governed cognitive system with a front door*. They do **not** make it behave like a brain. The following gaps were verified in code; they should be scheduled after WP1 (composition root) and WP3 (grounding connected), in this order.

### WP8 — Persistent memory across runs (currently total amnesia)
**Evidence**
- `TieredWorkspace.__init__` backs the *persistent* tier with `InMemoryEventStore()` when `persistent_store is None` (`hbllm/hcir/workspace_tiers.py:402-407`); no caller in `hbllm/` or `plugins/` passes a real store (`rg "TieredWorkspace\(" hbllm plugins` → only `TieredWorkspace()`).
- `AutonomousEpistemicEngine`, `PrefrontalWorkingMemory`, `HCIRSymbolicWorldTheory` have **no** `save/load/snapshot/restore` methods (grep clean). Every ARC game and every driver session starts from a blank brain.
- `hcir/stores/sqlite_event_store.py`, `cognitive_journal.py`, `replay_debugger.py`, `snapshot.py` exist but are not the source of truth for engine state.

**Target**
1. Bootstrap (WP1) constructs `TieredWorkspace(persistent_store=SqliteEventStore(path=cfg.memory_dir/"persistent.db"))`.
2. `HCIRSymbolicWorldTheory` predicates, `GroundedConceptNode`s, promoted `DeclarativeNeuroSymbolicSkill`s and `CausalPredicate`s live in the persistent tier (via `TransactionManager`), not in engine attributes. Working-memory fields with `RetentionPolicy.LIFELONG` are mirrored there on `commit_faculty_journal`.
3. `build_cognitive_system()` rehydrates: replay the persistent event log → `TieredWorkspace`; `engine.hydrate_from_workspace()` rebuilds `symbolic_theory` and lifelong working-memory slots.
4. Snapshot on `ExecutiveRuntime.stop()` and every `snapshot_interval` commits (already a parameter, unused by callers).

**Acceptance:** test — run 50 steps in env A, stop, build a new system against the same `memory_dir`, assert `symbolic_theory` predicates and concept ids are identical and `step_counter`-independent; `hbllm kernel status` shows event-log length and last snapshot.

### WP9 — Autonomous dual-rhythm runtime (currently it only thinks when poked)
**Evidence**
- The whole cortical loop is synchronous `engine.step(obs) → action`; no `while`/background task in AEE (`rg "async def|asyncio|while True"` → only `execute_subgoal_program_with_checkpoint`).
- `hippocampal_replay.py`, `insula_allostasis.py`, consolidation code in `cortex_episodic.py` exist but are invoked *inside* a step; there is no idle-time cognition and no clock.

**Target**
1. `ExecutiveRuntime` owns two rhythms: **online** — `decide()` triggered by `ObservationBus` arrival (observation is the interrupt, not the heartbeat); **offline** — when the perception channel is quiet for `> idle_ms`, run a consolidation cycle: `hippocampal_replay` of recent episodes, `ConceptLearner.maintain()`, schema pruning in `HCIRSymbolicWorldTheory`, concept promotion (WP3.5), snapshot (WP8).
2. Both rhythms commit through the same `TransactionManager`; offline cycles are tagged `author="offline_consolidation"` so replay/ablation can filter them.
3. `hbllm mind run` (WP2) starts the runtime and blocks; `hbllm mind step` still exists for deterministic debugging.

**Acceptance:** with a driver that emits no observations for 2 s, the event log shows ≥1 `offline_consolidation` transaction; concept count after idle ≤ before (merges happened); online latency is unaffected (p95 of `decide()` unchanged vs WP1 baseline).

### WP10 — Intrinsic value & goal arbitration (currently nothing is *wanted*)
**Evidence**
- Goals come only from the environment (`TerminalSignal`, `learned_goal_positions`) or from epistemic surprise (`GoalProposal` — which WP3 shows is never consumed). `insula_allostasis.py`, `global_workspace_ignition.py` are among the ablated faculties; `brain/emotion`, `brain/decision` value machinery lives in the LLM stack and never reaches `hcir.world`.
- `NeuromodulatorySystem` exists (kept faculty) but modulates search parameters only; it does not prioritise *which goal* to pursue.

**Target**
1. Define a small homeostatic state in `PrefrontalWorkingMemory` (compute/energy budget, safety margin, competence progress, social/instruction compliance) updated each cycle from `ComputeBudget`, `hazard_tracker`, prediction-error trend and `load_instructions`.
2. One arbitration point — a `GoalArbiter` stage between Assimilation and Simulation — receives `GoalProposal`s from three sources (epistemic/surprise, homeostatic, instructed via `HCIRCompiler`) and selects via expected free energy = extrinsic value + epistemic value − cost; result committed as the active `TaskFrame` in `TieredWorkspace.working`.
3. `NeuromodulatorySystem` levels become the weights of that arbitration (dopamine ↔ extrinsic, noradrenaline ↔ epistemic, serotonin ↔ cost/patience), so neuromodulation has a behavioural effect, not just a search-depth effect.

**Acceptance:** scenario test — same environment, two agents differing only in homeostatic budget: the low-budget one picks the nearer/safer goal; an instructed goal overrides curiosity until satisfied; `GoalProposal` queue never grows unbounded.

### WP11 — Language ↔ world binding (currently the two halves never meet)
**Evidence**
- `ModalitySource.SEMANTIC_LLM_ADAPTER` is referenced nowhere outside `hcir/grounding`; `RelationLearner.bind_linguistic_alias` and `tracker.add_alias` have no callers; `HCIRCompiler` is not connected to `AutonomousEpistemicEngine`.

**Target**
1. The LLM brain publishes utterances/NER/intent as `ObservationEnvelope(source_modality=SEMANTIC_LLM_ADAPTER)` on the same `ObservationBus`; `SemanticTextFeatureExtractor` (after WP3.2 replaces the char-hash stub with the existing `hbllm.perception`/`semantic` encoders) produces features that `ConceptLearner` can co-cluster with perceptual features → `GroundedConcept.linguistic_aliases`.
2. `HCIRCompiler` output (`InstructionStream` / `GoalProposal`) resolves noun phrases against `WorkingConceptStore` aliases → `GroundedEntityRef`s; unresolved → `UnresolvedRef` + `DecisionOutcome.REQUEST_EVIDENCE` (ask / look).
3. `DialogueRenderer` is the only outbound path for language, gated by `CommunicationValidationGateway`.

**Acceptance:** test — after N frames with a red block visible and the utterance "the red block", the concept bound to that entity has alias `red block`; "push the red block left" yields a `GoalProposal` whose target is that `GroundedEntityRef`; an utterance naming an unseen object yields `REQUEST_EVIDENCE`, not a hallucinated ref.

### WP12 — Dimension-generic perception (currently a 2-D integer lattice with an avatar)
**Evidence:** `EmbodiedObservation.lattice`, 56 avatar/grid fields on `CognitiveContext`, `TopologyClassifier.classify_2d_boxes`, predictor `(pos[0]+vel[0]*dt, pos[1]+vel[1]*dt)`, `SpatialEntity.coords` as `(r,c)` pairs. `RoboticsObservationAdapter`/`TextObservationAdapter` exist but downstream faculties cannot consume continuous 3-D poses.
**Target:** typed `ReferenceFrame(dim, discrete|continuous, units, origin)` on the envelope; kinematics/topology bodies made `tuple[float, ...]`-generic (signatures already are); avatar becomes an optional `self_entity_ref` rather than a required grid cell.
**Acceptance:** `benchmarks/scenarios/robotics_manipulation.py` runs through the *same* `CorticalPipeline` as the grid scenario with no 2-D branches hit (coverage assertion).

### WP13 — Cross-context symbolic learning (currently within-episode only)
**Evidence:** `CrossGameKnowledgeTransfer` is ablated; `skill_program_synthesizer.py` output is not persisted or reloaded as procedural memory; `DeclarativeNeuroSymbolicSkill` grammar exists but no skill survives a process restart.
**Target:** synthesized skills and `CausalPredicate`s promoted to the persistent tier (WP8) with a provenance `context_id`; on a new context, `SchemaInducer`/`mpfc_schema_engine` first tries transfer (predicate-signature match) before induction from scratch.
**Acceptance:** env A then env B sharing a mechanic (e.g. key→door): steps-to-first-success in B is lower with memory than with a fresh brain, measured in `hbllm bench local-suite --transfer`.

### WP14 — Latency budget for a real-time loop
**Evidence:** `hcir/world` is 65k LOC / ~90 faculties constructed per engine; AGENTS.md promises sub-millisecond core ops; Rust covers only the graph substrate (`rust/hcir_graph`).
**Target:** profile one `decide()` (`benchmarks/algorithms_profile.py` is a start); if p95 > 10 ms, move perception lifting (flood fill / connected components), A*, collision-cone sweep into `rust/` behind the existing Python-fallback pattern (`NativeGraphAdapter` parity rule applies).
**Acceptance:** `hbllm bench profile` reports p50/p95 of `decide()`; p95 ≤ 10 ms on the local suite grids; parity tests between Rust and Python paths.

### WP15 — The learned-perception decision (Phase 1; decision + flagged implementation, not a rewrite)
**Evidence.** Every feature the core reasons about is hand-designed: connected components (`cortex_perception.py`), Hu moments (`ShapeArchetypeSignature`), bbox topology (`TopologyClassifier`), `(row, col, color/10, log1p(size))` (`grounding/features/feature_extractor.py:135-140`). The 90-faculty sprawl is the downstream symptom — each new mechanic needed a new module. Meanwhile `hbllm/brain/snn/` (LIF, STDP `plasticity.py`, WTA, dendrites, `perception/`) is a real, deterministic-at-inference learning substrate used only for track prediction (`snn_predictor.py`). AGENTS.md Invariant 3 as written ("100% LLM-independent … deterministically") is read by the codebase as "no learned weights in core", which forecloses the only known route to representations the designer did not anticipate.

**This WP is a decision with a measurement, not a feature:**
1. **Re-scope Invariant 3 in AGENTS.md** to what it is actually protecting: *no external model/API calls; no online stochastic training inside the decision path; every inference deterministic and replayable given weights + inputs.* Offline-trained encoders whose weights are versioned artifacts in `TieredWorkspace.persistent` are permitted behind the `FeatureExtractor` protocol. (Text for AGENTS.md is in the companion edit.)
2. Implement `SNNPerceptualFeatureExtractor` (backed by `brain/snn/perception`) and, optionally, a small offline-trained contrastive encoder, both as `FeatureExtractor`s registered behind `--perception {handcrafted,snn,learned}` in the WP1 config. Training, if any, is offline on *unlabelled* frames from `dev/` only — never on `heldout/`.
3. Run WP0 `hbllm bench dev` and one `hbllm bench heldout` per variant with everything else frozen.
4. Write `reports/perception_comparison.md`: per-family steps-to-first-success, generality gap, latency. **Decision rule:** a learned extractor is adopted as default only if its held-out generality score is ≥ handcrafted *and* p95 latency stays within WP14 budget; otherwise handcrafted remains and the learned path stays available for research.

**Acceptance:** AGENTS.md Invariant 3 updated; the flag exists and all three variants pass the grounding test suite; `reports/perception_comparison.md` committed with the decision recorded; no `heldout/` task was used for any training (protocol hash check from WP0).

**Dependency order:** **Phase 1 = WP0 → WP1 → WP2 → WP3 → WP5-D → WP15.** Phase 2 = WP4 → WP5 (adapters for whatever survives WP5-D) → WP6 → WP7 → **WP8 → WP9 → WP10 → WP11** → WP12 → WP13 → WP14. WP0 gates everything (nothing is measurable without it); WP8 is the Phase 2 keystone (without persistence, WP9–WP13 have nothing to accumulate into).

---

## 6. Verification matrix for the executing agent

| After WP | Run |
|---|---|
| any | `ruff check hbllm tests && mypy hbllm/hcir` |
| 0 | `hbllm bench dev` (free) and `hbllm bench heldout` (once per sha) → `reports/heldout/<sha>_<date>.json`; `pytest tests/unit/benchmarks -q` |
| 5-D | `fd -e py . hbllm \| xargs cat \| wc -l` ≤ 250k; `pytest tests/unit/architecture/test_no_duplicate_faculties.py` |
| 15 | `hbllm bench heldout --perception {handcrafted,snn,learned}`; `reports/perception_comparison.md` present with decision |
| 1,3,4 | `pytest tests/unit/hcir tests/unit/world tests/integration/world tests/integration/grounding -n 4 -q` (baseline 1476+149 green) |
| 2 | `hbllm --help`, `python -m hbllm diagnostics`, `pytest tests/unit/cli -q` |
| 3 | `python -m benchmarks.sge_integration_benchmark` (compare to `reports/` baseline; graph size must plateau) |
| 5 | `hbllm bench local-suite` → compare to `reports/local_suite/golden_closed_loop.json`; `pytest tests/unit/architecture -q` |
| 8 | `pytest tests/integration/runtime/test_persistence_roundtrip.py`; `hbllm kernel status` shows event-log length + last snapshot |
| 9 | `pytest tests/integration/runtime/test_offline_consolidation.py`; p95 `decide()` unchanged vs WP1 baseline |
| 10 | `pytest tests/integration/world/test_goal_arbitration.py` |
| 11 | `pytest tests/integration/grounding/test_language_world_binding.py` |
| 12–14 | `benchmarks/scenarios/robotics_manipulation.py` through `CorticalPipeline`; `hbllm bench local-suite --transfer`; `hbllm bench profile` p95 ≤ 10 ms |
| all | `python -m hbllm.cli.diagnostics` shows kernel online + verifier attached |

---

## 7. Things I verified so the next agent does not need to
- Invariant 3 (LLM-free core) holds — no network/LLM imports in `hcir/*` or `brain/reasoning`.
- No raw graph mutation outside the TM (`workspace.add_node/upsert_node/...` grep clean in `hbllm/`).
- Grounding, bridge and pipeline are ruff + mypy clean.
- `SurpriseEngine` has **no** `record_surprise` method (so §2.2 row 334-344 is a confirmed dead path, not a hypothesis).
- `commit_concept_promotion`, `commit_entity_reidentification`, `reidentify_entity`, `merge_candidates`, `induce_relations`, `predictor.register_schema` have **zero** non-test call sites in `hbllm/` or `plugins/` (only `benchmarks/scenarios/adversarial_surprise.py` registers a schema manually).
- `AutonomousEpistemicEngine.step/decide` has exactly two production callers (ARC plugin, CognitiveBlackbox); `ExecutiveRuntime.run_cycle` has one (post-hoc in `Brain.process`).
- `python -m hbllm.cli` → training `main`, not the dispatcher; `hbllm/__main__.py` absent.
