"""Tests for slip-stable calibration, stochastic safety margins, and partition-gate priority."""

from __future__ import annotations

import numpy as np

from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.motor.curiosity import EpistemicCuriosityExplorer
from hbllm.hcir.world.motor_calibration import (
    ActionDynamicsModel,
    disambiguate_stochastic_motor_basis,
)
from hbllm.hcir.world.objects.risk import compute_stance_priority


def test_action_dynamics_modal_tie_preserves_direction_and_recovers_mode() -> None:
    model = ActionDynamicsModel(action_id=4, delta_r=0, delta_c=1, confidence=0.6, probes_tested=1)
    assert model.delta_counts == {(0, 1): 1}

    # Single orthogonal slip (1 vs 1 tie) preserves current modal (0, 1)
    model.update_from_trial((-1, 0), success=True)
    assert model.get_displacement() == (0, 1)
    assert model.slip_rate > 0.05

    # Subsequent true observations solidify (0, 1)
    model.update_from_trial((0, 1), success=True)
    assert model.get_displacement() == (0, 1)


def test_compute_stance_priority_prefers_dividing_wall_gate_with_inventory() -> None:
    eng = AutonomousEpistemicEngine(world_id="test_gate_prio")
    grid = np.zeros((10, 10), dtype=int)
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    grid[:, 5] = 1  # dividing wall
    grid[4, 5] = 3  # gate embedded in dividing wall
    grid[2, 2] = 9  # singleton unverified interior anomaly
    grid[1, 1] = 8  # avatar
    eng.state.avatar_pos = (1, 1)
    eng.symbolic_theory.induce_avatar(8, evidence="test")
    eng.symbolic_theory.induce_barrier(1, evidence="test")
    eng.working_memory.collected_inventory_features.add(2)
    eng.working_memory.pending_gate_check = False

    def _role_scores(_feat: int) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    gate_prio = compute_stance_priority(
        eng,
        grid,
        10,
        10,
        0,
        (1, 1),
        (4, 5),
        {(4, 5): {"entity_pos": (4, 5), "feature": 3}},
        set(),
        _role_scores,
    )
    anomaly_prio = compute_stance_priority(
        eng,
        grid,
        10,
        10,
        0,
        (1, 1),
        (2, 2),
        {(2, 2): {"entity_pos": (2, 2), "feature": 9}},
        set(),
        _role_scores,
    )
    assert gate_prio < anomaly_prio


def test_curiosity_prefers_step_away_from_unverified_anomaly_under_slip() -> None:
    eng = AutonomousEpistemicEngine(world_id="test_slip_curiosity")
    grid = np.zeros((7, 7), dtype=int)
    grid[3, 1] = 8  # avatar
    grid[2, 2] = 9  # unverified anomaly
    grid[3, 5] = 4  # distant exploration target
    eng.state.avatar_pos = (3, 1)
    eng.symbolic_theory.induce_avatar(8, evidence="test")
    eng.action_dynamics = {
        1: ActionDynamicsModel(action_id=1, delta_r=-1, delta_c=0, confidence=0.8, probes_tested=4),
        2: ActionDynamicsModel(action_id=2, delta_r=1, delta_c=0, confidence=0.8, probes_tested=4),
        4: ActionDynamicsModel(action_id=4, delta_r=0, delta_c=1, confidence=0.8, probes_tested=4),
    }
    eng.action_dynamics[4].delta_counts = {(0, 1): 3, (-1, 0): 1}  # slip_rate = 0.25

    # Both (3,2) [adj to anomaly (2,2)] and (4,1) [safe step] are distance 4 from target (3,5)?
    # Wait: from (3,1) -> (3,5) dist=4; (3,2) has dist 3, (4,1) has dist 5; penalty +55 outweighs diff=2
    chosen = EpistemicCuriosityExplorer._probe_active_inference(
        eng,
        grid,
        [(1, (-1, 0)), (2, (1, 0)), (4, (0, 1))],
        target_pos=(3, 5),
        bg=0,
        bg_is_impassable_void=False,
        H=7,
        W=7,
    )
    assert chosen != 4


def test_disambiguate_stochastic_motor_basis_resolves_tie() -> None:
    d1 = ActionDynamicsModel(action_id=1, delta_r=-1, delta_c=0, confidence=0.7, probes_tested=2)
    d2 = ActionDynamicsModel(action_id=2, delta_r=1, delta_c=0, confidence=0.8, probes_tested=4)
    d4 = ActionDynamicsModel(action_id=4, delta_r=-1, delta_c=0, confidence=0.7, probes_tested=5)
    d4.delta_counts = {(0, 1): 2, (-1, 0): 2, (1, 0): 1}

    changed = disambiguate_stochastic_motor_basis({1: d1, 2: d2, 4: d4})
    assert changed == [4]
    assert d4.get_displacement() == (0, 1)


def test_probed_dividing_wall_gate_contact_outranks_interior_anomaly() -> None:
    eng = AutonomousEpistemicEngine(world_id="test_probed_gate_prio")
    grid = np.zeros((10, 10), dtype=int)
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    grid[5, :] = 1  # horizontal dividing wall
    grid[5, 7] = 9  # probed dividing-wall gate
    grid[3, 9] = 12  # singleton unverified interior anomaly
    grid[4, 7] = 8  # avatar standing adjacent to probed gate
    eng.state.avatar_pos = (4, 7)
    eng.symbolic_theory.induce_avatar(8, evidence="test")
    eng.symbolic_theory.induce_barrier(1, evidence="test")
    eng.working_memory.collected_inventory_features.update({2, 3, 5})
    eng.symbolic_theory.mark_locus_explored((4, 7), reason="actuator_probed")

    def _role_scores(_feat: int) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    gate_contact_prio = compute_stance_priority(
        eng,
        grid,
        10,
        10,
        0,
        (4, 7),
        (5, 7),
        {},
        set(),
        _role_scores,
    )
    anomaly_prio = compute_stance_priority(
        eng,
        grid,
        10,
        10,
        0,
        (4, 7),
        (3, 9),
        {(3, 9): {"entity_pos": (3, 9), "feature": 12}},
        set(),
        _role_scores,
    )
    assert gate_contact_prio < anomaly_prio


def test_unlocked_dividing_wall_gate_outranks_unlocked_interior_anomaly() -> None:
    eng = AutonomousEpistemicEngine(world_id="test_unlocked_gate_prio")
    grid = np.zeros((10, 10), dtype=int)
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    grid[5, :] = 1
    grid[5, 7] = 9  # dividing-wall gate
    grid[3, 9] = 12  # interior singleton anomaly
    grid[4, 7] = 8
    eng.state.avatar_pos = (4, 7)
    eng.symbolic_theory.induce_avatar(8, evidence="test")
    eng.symbolic_theory.induce_barrier(1, evidence="test")
    eng.working_memory.collected_inventory_features.add(5)
    eng.working_memory.pending_gate_check = True
    unlocked = {(5, 7), (3, 9)}

    def _role_scores(_feat: int) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    gate_prio = compute_stance_priority(
        eng,
        grid,
        10,
        10,
        0,
        (4, 7),
        (5, 7),
        {(5, 7): {"entity_pos": (5, 7), "feature": 9}},
        unlocked,
        _role_scores,
    )
    anomaly_prio = compute_stance_priority(
        eng,
        grid,
        10,
        10,
        0,
        (4, 7),
        (3, 9),
        {(3, 9): {"entity_pos": (3, 9), "feature": 12}},
        unlocked,
        _role_scores,
    )
    assert gate_prio < anomaly_prio


def test_deliberation_clears_stale_mental_plan_on_discrete_slip() -> None:
    from hbllm.hcir.world.deliberation.engine import PrefrontalDeliberationEngine
    from hbllm.hcir.world.planning.mental_simulation import MentalSimulationStep

    eng = AutonomousEpistemicEngine(world_id="test_slip_replan")
    grid = np.zeros((7, 7), dtype=int)
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    grid[3, 3] = 8
    eng.state.avatar_pos = (3, 3)
    eng.symbolic_theory.induce_avatar(8, evidence="test")
    eng.action_dynamics = {
        1: ActionDynamicsModel(action_id=1, delta_r=-1, delta_c=0, confidence=0.9, probes_tested=5),
        2: ActionDynamicsModel(action_id=2, delta_r=1, delta_c=0, confidence=0.9, probes_tested=5),
        4: ActionDynamicsModel(action_id=4, delta_r=0, delta_c=1, confidence=0.9, probes_tested=5),
    }
    eng.working_memory.active_plan_step = MentalSimulationStep(
        action=2, predicted_avatar_pos=(4, 2), expected_mutation="move"
    )
    eng.mental_plan.append(
        MentalSimulationStep(action=4, predicted_avatar_pos=(4, 3), expected_mutation="move")
    )

    delib = PrefrontalDeliberationEngine()
    delib.deliberate(eng, grid, [1, 2, 4])
    assert not any(s.action == 4 and s.predicted_avatar_pos == (4, 3) for s in eng.mental_plan)
