---
name: cortical-alu-and-world-migration
description: >-
  Use this skill when modifying `hbllm/hcir/world/**`,
  `AutonomousEpistemicEngine`, `CorticalPipeline`, `TransitionKnowledgeBase`,
  `CorticalGroundingBridge`, or refactoring procedural cortical modules into
  stateless Layer-0 Rust/Python ALUs plus homoiconic HCIR programs (Phase M5).
---

# Skill: Cortical ALUs & World-Model Migration

## Scope & Authoritative References
- [ADR 005](../../../docs/adr/0005-homoiconic-symbolic-vm-and-experience-first-cognition.md)
- [Migration Plan — Phase M5](../../../docs/architecture/homoiconic-vm-migration-plan.md)
- [Innate Priors Declaration](../../../docs/architecture/priors.md)
- Key files: `hbllm/hcir/world/autonomous_epistemic_engine.py`, `hbllm/hcir/world/cortical_pipeline.py`, `hbllm/hcir/world/causal/{theory,transition_kb}.py`, `hbllm/hcir/world/assimilation/assimilator.py`, `hbllm/hcir/world/planning/{mental_simulation,goal_scheduler}.py`, `hbllm/hcir/world/deliberation/engine.py`.

## Design Rules & Execution Steps

1. **Separate Stateless Layer-0 ALUs from Symbolic Control Programs:**
   - Keep inner numerical/spatial loops inside pure, stateless Layer-0 ALUs (`math.geodesic_field`, `math.bellman_risk_backup`, `math.collision_cone`, `math.bayesian_posterior_update`, Rust `hbllm_simulation_engine`, `hbllm_structure_matcher`, and `snn`/`hbllm_perception_rs`).
   - Express epistemic phase arbitration, BA10 subgoal decomposition, tool-barrier mediation, and safety gating as symbolic HCIR programs over `FrozenGraphView`, emitting `StageResult.operations` rather than procedural `if/else` walls.

2. **Replace Crisp Boolean Ontology Sets with Dirichlet/Beta Posteriors:**
   - Represent empirical feature and `(pos, action)` transition outcomes in `TransitionKnowledgeBase` and `HCIRSymbolicWorldTheory` using Dirichlet/Beta counts (`alpha_pass`, `beta_block`, `gamma_lethal`, `delta_goal`) instead of brittle boolean sets and ad-hoc procedural guards.

3. **Eliminate Direct State Writes & Duck-Typing Debt:**
   - Never assign `view.state.<attr> = ...` or engine `_last_*` / `_feedback_assimilated` attributes.
   - Do not add `hasattr(...)`, 3-arg `getattr(...)`, or `get_sub_attr(...)` helpers; pass typed frozen views/contracts to cortical stages.
   - Keep every `hbllm/hcir/world/**` file trending toward `<= 1,200` LOC (ratcheted by `test_structural_debt_ratchet.py`).

4. **Enforce R1 Faculty & Latency Gates:**
   - Any new or modified cortical/grounding faculty must demonstrate positive ablation impact (`Δ > 0` on `widened_dev`) and stay within `CPU p95 <= 45ms` at `32×32`.

## Verification
Run from `HBLLM/core/`:
```bash
pytest tests/unit/architecture/test_structural_debt_ratchet.py tests/unit/architecture/test_state_ownership_invariants.py -q
pytest tests/unit/ -k "epistemic or cortex or transition_kb" -q
python -m hbllm bench dev && python -m hbllm bench profile
```

