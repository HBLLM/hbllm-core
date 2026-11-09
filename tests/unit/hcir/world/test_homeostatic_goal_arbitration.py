"""Homeostatic Intrinsic Motivation & Goal Arbitration test suite.

Validates:
1. Homeostatic state tracking (energy budget, safety margins, competence progress, social instructions).
2. Expected Free Energy (EFE) goal arbitration across extrinsic, epistemic, and cost dimensions.
3. Social instructions preempting open-ended curiosity until satisfied.
4. Low-energy state autonomously prioritizing high-safety, low-cost subgoals over exploration.
5. Neuromodulatory modulation of arbitration weights (Dopamine, Noradrenaline, Serotonin).
6. Automatic TaskFrame activation in TieredWorkspace for selected goals.
"""

from __future__ import annotations

import numpy as np
import pytest

from hbllm.hcir.workspace_tiers import TieredWorkspace
from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.goal_arbiter import CandidateGoal, GoalArbiter
from hbllm.hcir.world.neuromodulatory_system import NeuromodulatorySystem
from hbllm.hcir.world.prefrontal_working_memory import HomeostaticState


def test_homeostatic_state_initialization_and_operations() -> None:
    """Validate homeostatic state tracking, energy consumption, and predicate export."""
    state = HomeostaticState(
        energy_budget=100.0,
        safety_margin=1.0,
        competence_progress=0.0,
    )
    assert not state.is_critical_energy

    # Consume energy
    remaining = state.consume_energy(40.0)
    assert remaining == pytest.approx(60.0)
    assert state.fatigue == pytest.approx(20.0)
    assert not state.is_critical_energy

    # Further consumption into critical energy
    state.consume_energy(45.0)
    assert state.energy_budget == pytest.approx(15.0)
    assert state.is_critical_energy

    # Replenish energy
    state.replenish_energy(70.0)
    assert state.energy_budget == pytest.approx(85.0)
    assert not state.is_critical_energy

    # Social instruction tracking
    instruction = {"goal_id": "escort_unit", "target": (4, 4), "utility": 75.0}
    state.add_instruction(instruction)
    assert len(state.social_instructions) == 1
    assert state.social_instructions[0]["goal_id"] == "escort_unit"

    # Export to causal predicates and transaction operations
    preds = state.to_causal_predicates()
    assert len(preds) == 3
    ops = state.to_transaction_operations()
    assert len(ops) == 1
    assert ops[0].node_id == "homeostatic_state"

    state.clear_instructions()
    assert len(state.social_instructions) == 0


def test_social_instructions_preempt_curiosity() -> None:
    """Verify that an explicit social instruction preempts open-ended curiosity."""
    arbiter = GoalArbiter()
    homeostatic = HomeostaticState(energy_budget=100.0)
    neuromod = NeuromodulatorySystem()

    # Candidate 1: High curiosity exploration target (novel, unvisited, but uninstructed)
    curiosity_goal = CandidateGoal(
        goal_id="curiosity_frontier_alpha",
        target=(18, 18),
        goal_type="epistemic_exploration",
        extrinsic_utility=2.0,
        epistemic_utility=40.0,
        estimated_cost=3.0,
    )

    # Candidate 2: Confirmed standard extrinsic goal
    extrinsic_goal = CandidateGoal(
        goal_id="standard_confirmed_exit",
        target=(10, 10),
        goal_type="confirmed_extrinsic",
        extrinsic_utility=30.0,
        epistemic_utility=1.0,
        estimated_cost=5.0,
    )

    # Without social instruction, the confirmed extrinsic or high curiosity goal wins
    result_baseline = arbiter.arbitrate(
        candidates=[curiosity_goal, extrinsic_goal],
        homeostatic_state=homeostatic,
        neuromodulator_system=neuromod,
    )
    assert result_baseline is not None
    assert result_baseline.selected_goal.goal_id in (
        "curiosity_frontier_alpha",
        "standard_confirmed_exit",
    )

    # Candidate 3: User/Social Instruction received
    instructed_goal = CandidateGoal(
        goal_id="instructed_priority_objective",
        target=(2, 2),
        goal_type="social_instruction",
        extrinsic_utility=80.0,
        epistemic_utility=0.0,
        estimated_cost=2.0,
    )

    # With social instruction, the instructed goal MUST decisively win
    result_instructed = arbiter.arbitrate(
        candidates=[curiosity_goal, extrinsic_goal, instructed_goal],
        homeostatic_state=homeostatic,
        neuromodulator_system=neuromod,
    )
    assert result_instructed is not None
    assert result_instructed.selected_goal.goal_id == "instructed_priority_objective"
    assert (
        result_instructed.all_scores["instructed_priority_objective"]
        > result_instructed.all_scores["curiosity_frontier_alpha"]
    )


def test_critical_energy_state_prioritizes_low_cost_safety() -> None:
    """Verify that when energy is critically depleted, agent prioritizes low-cost safety over exploration."""
    arbiter = GoalArbiter()
    neuromod = NeuromodulatorySystem()

    # Distant high-epistemic exploration goal (high reward if full energy, but costly)
    distant_exploration = CandidateGoal(
        goal_id="distant_unexplored_ruins",
        target=(25, 25),
        goal_type="epistemic_exploration",
        extrinsic_utility=5.0,
        epistemic_utility=50.0,
        estimated_cost=22.0,
        hazard_risk=0.2,
    )

    # Immediate low-cost resting / safety waypoint
    near_safe_haven = CandidateGoal(
        goal_id="local_safe_waypoint",
        target=(1, 1),
        goal_type="safety_preservation",
        extrinsic_utility=20.0,
        epistemic_utility=0.0,
        estimated_cost=1.0,
        hazard_risk=0.0,
    )

    # 1. Full Energy state (energy_budget = 100.0):
    full_energy_state = HomeostaticState(energy_budget=100.0)
    result_full = arbiter.arbitrate(
        candidates=[distant_exploration, near_safe_haven],
        homeostatic_state=full_energy_state,
        neuromodulator_system=neuromod,
    )
    assert result_full is not None
    # At full energy, high information gain compensates for cost
    assert result_full.selected_goal.goal_id == "distant_unexplored_ruins"

    # 2. Critical Energy state (energy_budget = 10.0, is_critical_energy = True):
    depleted_energy_state = HomeostaticState(energy_budget=10.0)
    assert depleted_energy_state.is_critical_energy

    result_critical = arbiter.arbitrate(
        candidates=[distant_exploration, near_safe_haven],
        homeostatic_state=depleted_energy_state,
        neuromodulator_system=neuromod,
    )
    assert result_critical is not None
    # At critical energy, cost penalty quadruples and curiosity drops 90%, safety haven wins decisively
    assert result_critical.selected_goal.goal_id == "local_safe_waypoint"
    assert (
        result_critical.all_scores["local_safe_waypoint"]
        > result_critical.all_scores["distant_unexplored_ruins"]
    )


def test_neuromodulatory_modulation_of_arbitration_weights() -> None:
    """Verify that Dopamine, Noradrenaline, and Serotonin modulate arbitration parameters."""
    arbiter = GoalArbiter()
    homeostatic = HomeostaticState(energy_budget=100.0)

    # Baseline neuromodulators (DA=1.0, NE=1.0, 5HT=1.0)
    neuromod_base = NeuromodulatorySystem(baseline_da=1.0, baseline_ne=1.0, baseline_5ht=1.0)
    w_base = arbiter.compute_arbitration_weights(homeostatic, neuromod_base)
    assert w_base["w_extrinsic"] == pytest.approx(1.0, rel=0.1)
    assert w_base["w_epistemic"] == pytest.approx(1.0, rel=0.1)
    assert w_base["w_cost"] == pytest.approx(1.0, rel=0.1)

    # High Dopamine surges extrinsic weight
    neuromod_da = NeuromodulatorySystem(baseline_da=2.5, baseline_ne=1.0, baseline_5ht=1.0)
    w_da = arbiter.compute_arbitration_weights(homeostatic, neuromod_da)
    assert w_da["w_extrinsic"] > w_base["w_extrinsic"]
    assert w_da["w_extrinsic"] == pytest.approx(2.5, rel=0.1)

    # High Noradrenaline (surprise) surges epistemic weight
    neuromod_ne = NeuromodulatorySystem(baseline_da=1.0, baseline_ne=2.8, baseline_5ht=1.0)
    w_ne = arbiter.compute_arbitration_weights(homeostatic, neuromod_ne)
    assert w_ne["w_epistemic"] > w_base["w_epistemic"]
    assert w_ne["w_epistemic"] == pytest.approx(2.8, rel=0.1)

    # Low Serotonin (temporal stress / impatience) heavily amplifies cost penalty
    neuromod_low_5ht = NeuromodulatorySystem(baseline_da=1.0, baseline_ne=1.0, baseline_5ht=0.4)
    w_low_5ht = arbiter.compute_arbitration_weights(homeostatic, neuromod_low_5ht)
    assert w_low_5ht["w_cost"] > w_base["w_cost"]
    assert w_low_5ht["w_cost"] == pytest.approx(2.5, rel=0.1)


def test_goal_arbiter_task_frame_binding() -> None:
    """Verify that GoalArbiter creates and attaches an active TaskFrame in TieredWorkspace."""
    tiered = TieredWorkspace()
    arbiter = GoalArbiter()
    homeostatic = HomeostaticState(energy_budget=100.0)

    candidate = CandidateGoal(
        goal_id="task_frame_target_alpha",
        target=(5, 5),
        goal_type="confirmed_extrinsic",
        extrinsic_utility=50.0,
        estimated_cost=2.0,
    )

    result = arbiter.arbitrate(
        candidates=[candidate],
        homeostatic_state=homeostatic,
        tiered_workspace=tiered,
    )

    assert result is not None
    assert result.active_task_frame is not None
    frame = result.active_task_frame
    assert frame.goal_id == "task_frame_target_alpha"
    assert frame.is_active
    assert frame in tiered.active_task_frames

    # Re-arbitrating for the same goal reuses the active TaskFrame
    result_second = arbiter.arbitrate(
        candidates=[candidate],
        homeostatic_state=homeostatic,
        tiered_workspace=tiered,
    )
    assert result_second is not None
    assert result_second.active_task_frame.frame_id == frame.frame_id


def test_cortical_pipeline_goal_arbitration_integration() -> None:
    """Verify that AutonomousEpistemicEngine and CorticalPipeline run goal arbitration."""
    engine = AutonomousEpistemicEngine()
    engine.add_social_instruction(
        {"goal_id": "commanded_waypoint", "target": (4, 4), "utility": 90.0}
    )

    # Set up a synthetic grid
    grid = np.zeros((10, 10), dtype=np.int32)
    engine.avatar_pos = (1, 1)
    engine.prev_grid = grid

    # Extract candidate goals via engine
    candidates = engine.goal_arbiter.extract_candidate_goals(engine)
    assert len(candidates) >= 1
    assert any(c.goal_id == "commanded_waypoint" for c in candidates)

    # Arbitrate
    res = engine.goal_arbiter.arbitrate(candidates=candidates, engine=engine)
    assert res is not None
    assert res.selected_goal.goal_id == "commanded_waypoint"
    assert engine.active_arbitrated_goal is not None
    assert engine.active_arbitrated_goal.goal_id == "commanded_waypoint"
