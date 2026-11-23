---
title: "API Reference — Cortical World Model & Declarative Skills"
description: "API reference for AutonomousEpistemicEngine, CorticalPipeline, TransitionKnowledgeBase, HCIRSymbolicWorldTheory, PrefrontalWorkingMemory, CorticalGroundingBridge, and DeclarativeNeuroSymbolicSkill."
---

# Cortical World Model API (`hbllm.hcir.world`)

**Packages:** `hbllm.hcir.world`, `hbllm.hcir.skills.declarative`

---

## `AutonomousEpistemicEngine`

**Module:** `hbllm.hcir.world.autonomous_epistemic_engine.AutonomousEpistemicEngine`

The core active-inference and causal discovery engine driving the 5-stage `CorticalPipeline`. In production and benchmarks, instantiate via [`build_cognitive_system()`](../architecture/entry-points.md).

```python
from hbllm.hcir.world.percept import EmbodiedObservation, EpisodeBoundary, EpisodeBoundaryKind
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

system = build_cognitive_system(
    CognitiveSystemConfig(
        world_id="grid_demo",
        exploration_budget=150,
        enable_grounding=True,
        perception_mode="handcrafted",  # "handcrafted" | "snn" | "learned"
    )
)

obs = EmbodiedObservation(
    lattice=[[0, 0, 0], [0, 1, 0], [0, 0, 2]],
    available_actions=[1, 2, 3, 4],
    context_id="level_1",
    episode_id="ep_0",
)

action, decision_meta = system.engine.step(obs)
print(f"Chosen action={action}, meta={decision_meta}")

# Signal procedural boundary transitions cleanly
system.engine.on_boundary(
    EpisodeBoundary(kind=EpisodeBoundaryKind.NEW_LEVEL, context_id="level_2")
)
```

### Key Methods

| Method | Signature | Description |
|---|---|---|
| `step()` | `(obs: EmbodiedObservation, last_action: Any \| None = None) -> tuple[Any, dict[str, Any] \| None]` | Executes one full 5-stage cortical cycle and returns `(action, metadata)` |
| `on_boundary()` | `(boundary: EpisodeBoundary \| Any) -> None` | Applies formal retention policies at episode, level, or context boundaries |
| `reset_episode()` | `(retain_dynamics: bool = True, ...) -> None` | Resets episode-transient state while retaining context/lifelong theory |
| `explain_last_decision()` | `() -> str` | Returns human-readable causal/deliberation provenance for the last step |
| `persist_to_workspace()` | `(target: Any = None, author: str = 'epistemic_engine') -> list[TransactionOperation]` | Exports learned predicates, transitions, and rules into `TieredWorkspace` |
| `hydrate_from_workspace()` | `(source: Any = None) -> int` | Reconstructs `TransitionKnowledgeBase` and ontology from persistent tier nodes |

---

## `CorticalPipeline`

**Module:** `hbllm.hcir.world.cortical_pipeline.CorticalPipeline`

Executes the ordered 5-stage cognitive cycle (`PerceptionStage` → `AssimilationStage` → `SimulationStage` → `DeliberationStage` → `MotorStage`).

| Attribute / Method | Description |
|---|---|
| `run(ctx: CognitiveContext) -> PipelineResult` | Runs all 5 stages against an immutable `CognitiveContext` |
| `PipelineResult.final_action` | Selected motor action after deliberation & safety gating |
| `PipelineResult.stage_timings_ms` | Per-stage latency breakdown (`dict[str, float]`) |
| `PipelineResult.operations` | Proposed `TransactionOperation` batch |

---

## `TransitionKnowledgeBase` & `HCIRSymbolicWorldTheory`

**Modules:** `hbllm.hcir.world.causal.transition_kb`, `hbllm.hcir.world.causal.theory`

`TransitionKnowledgeBase` is the HCIR-backed single source of truth inside `HCIRSymbolicWorldTheory` for empirical feature, locus, and conditional transition beliefs.

```python
from hbllm.hcir.world.causal.transition_kb import TransitionKnowledgeBase, TransitionOutcome

kb = TransitionKnowledgeBase()

# Record an empirical traversal outcome for feature 3
kb.record_outcome(
    context_key=3,
    outcome=TransitionOutcome.TRAVERSABLE,
    scope="feature",
    evidence="step_displacement",
)

# Record a locus-specific lethal outcome
kb.record_outcome(
    context_key=(4, 5),
    outcome=TransitionOutcome.LETHAL,
    scope="locus",
    evidence="episode_termination",
)

eval_res = kb.evaluate_transition(feature_id=3, locus=(4, 5))
print(f"traversable={eval_res.traversable}, risk_cost={eval_res.risk_cost}")

# Export as HCIR BeliefNode / LearnedRuleNode transactions
ops = kb.to_transaction_operations()
```

### `TransitionKnowledgeBase` Methods

| Method | Description |
|---|---|
| `record_outcome(context_key, outcome, *, scope='feature', ...)` | Records a weighted `TransitionOutcome` (`TRAVERSABLE`, `BLOCKED`, `LETHAL`, etc.) |
| `retract_outcome(context_key, outcome, *, scope='feature')` | Retracts/decays stale beliefs upon counter-evidence |
| `record_causal_rule(trigger_key, effect_signature, *, precondition_tokens=(), ...)` | Records a conditional `CausalEffectRule` |
| `evaluate_transition(feature_id, *, locus=None, active_state_tokens=None, ...)` | Returns a `TransitionEdgeEvaluation` (`traversable`, `step_cost`, `risk_cost`, `goal_value`, `info_gain`) |
| `begin_context_epoch(epoch=None)` / `clear_context_beliefs()` / `clear_locus_beliefs()` | Manages cross-seed / context boundary invalidation |
| `to_transaction_operations()` / `hydrate_from_nodes()` | Round-trips beliefs and causal rules with `TieredWorkspace` |

---

## `PrefrontalWorkingMemory`

**Module:** `hbllm.hcir.world.working_memory.memory.PrefrontalWorkingMemory`

Governs transient working state, action inhibitions, held items, homeostatic energy, and domain procedural skill synthesis:

- `inhibit_action(action, duration=4)` / `decay_inhibited_actions()`
- `acquire_item(feature_id, role, step, position, properties=None)` / `expend_item(feature_id)` / `is_holding(feature_id)`
- `record_domain_episode(domain_id, success, mean_error=0.2, predicted_confidence=0.8)`
- `synthesize_skill_from_schema(schema, required_actions=None)` / `find_matching_skill(...)`
- `to_transaction_operations()` / `export_causal_predicates()`

---

## `CorticalGroundingBridge`

**Module:** `hbllm.hcir.world.grounding.bridge.CorticalGroundingBridge` (re-exported by `hbllm.hcir.world.cortical_grounding_bridge`)

Connects the Symbol Grounding Engine (`hbllm.hcir.grounding`) to `AutonomousEpistemicEngine` and `KernelServices`, enforcing transactional `HCIRCommitter` mutations across `process_perception()`, `process_assimilation()`, `process_motor()`, and `verify_post_action_outcome()`.

---

## `DeclarativeNeuroSymbolicSkill`

**Module:** `hbllm.hcir.skills.declarative`

Defines domain-agnostic behavior programs as composable `SymbolicPredicate` signatures paired with `SubgoalSequence` instructions:

```python
from hbllm.hcir.skills.declarative import (
    ActionAffordancePredicate,
    CutSetPredicate,
    DeclarativeNeuroSymbolicSkill,
    SpatialActionIntent,
    SubgoalSequence,
    SymbolicSubgoal,
)


class ReachCutSetSkill(DeclarativeNeuroSymbolicSkill):
    name = "reach_cutset_skill"
    description = "Acquire key and traverse topological cut barrier"
    signature = ActionAffordancePredicate(required={1, 2, 3, 4}) & CutSetPredicate(min_partitions=2)
    program = SubgoalSequence(
        SymbolicSubgoal(intent=SpatialActionIntent.NAVIGATE_TO, target_query={"role": "cargo"}),
        SymbolicSubgoal(intent=SpatialActionIntent.INTERACT, target_query={"role": "barrier"}),
    )
```

