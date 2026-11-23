from __future__ import annotations

import numpy as np

from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.motor_calibration import ActionDynamicsModel
from hbllm.hcir.world.percept import (
    EmbodiedObservation,
    EpisodeBoundary,
    EpisodeBoundaryKind,
    TerminalSignal,
)


def _eng():
    return AutonomousEpistemicEngine(enable_logging=False, seed=0)


def test_death_retains_motor_posteriors():
    e = _eng()
    e.record_action_effect(2, "translate")
    e.record_action_effect(1, "no_op")
    e.on_boundary(EpisodeBoundary(kind=EpisodeBoundaryKind.DEATH))
    assert e.motor_posterior(2) == {"translate": 1.0}
    assert e.expected_effect(1) == "no_op"


def test_level_complete_clears_level_local_state():
    e = _eng()
    e.inhibited_actions[2] = 8
    e.mental_plan.appendleft("plan_item")
    e.on_boundary(EpisodeBoundary(kind=EpisodeBoundaryKind.LEVEL_COMPLETE))
    assert not e.mental_plan
    # motor posteriors are context-local — retained
    e.record_action_effect(3, "translate")
    assert e.expected_effect(3) == "translate"


def test_boundary_kind_mapping():
    e = _eng()
    e.on_boundary(EpisodeBoundary(kind=EpisodeBoundaryKind.CONTEXT_RESET))
    assert e.step_counter == 0


def test_same_level_death_preserves_barriers_failed_transitions_exhausted_and_habenular_inhibitions():
    e = _eng()
    e.symbolic_theory.induce_barrier(5, evidence="wall")
    e.learned_barriers.add((1, 2))
    e.failed_transitions.add(((2, 2), 4))
    e.exhausted_candidate_goals.add((3, 3))
    e.object_planner.ledger.interaction_counts["obj_1"] = 4
    e.episodic_cortex.record_transition(step=1, avatar_pos=(2, 1), action=4)
    e.episodic_cortex.record_transition(step=2, avatar_pos=(2, 2), action=2)
    e.step_counter = 2

    e.on_boundary(EpisodeBoundary(kind=EpisodeBoundaryKind.DEATH))

    assert e.symbolic_theory.is_barrier(5)
    assert (1, 2) in e.learned_barriers
    assert ((2, 2), 4) in e.failed_transitions
    assert (3, 3) in e.exhausted_candidate_goals
    assert e.working_memory.habenular_ior.is_action_inhibited((2, 1), 4)
    assert len(e.working_memory.habenular_ior.active_inhibitions) > 0
    assert e.object_planner.ledger.interaction_counts["obj_1"] == 2
    assert len(e.working_memory.habenular_ior.episode_trace) == 0


def test_death_with_explicit_same_level_false_starts_new_context():
    e = _eng()
    initial_epoch = e.context_epoch
    e.learned_barriers.add((1, 2))

    boundary = EpisodeBoundary(kind=EpisodeBoundaryKind.DEATH, same_level=False)
    assert boundary.is_same_context is False

    e.on_boundary(boundary)
    assert e.context_epoch == initial_epoch + 1
    assert (1, 2) not in e.learned_barriers


def test_context_reset_with_retain_dynamics_false_clears_dynamics():
    e = _eng()
    e.action_dynamics[1] = ActionDynamicsModel(
        action_id=1, delta_r=-1, delta_c=0, confidence=1.0, probes_tested=3
    )
    e.symbolic_theory.induce_barrier(5, evidence="wall")

    e.on_boundary(EpisodeBoundary(kind=EpisodeBoundaryKind.CONTEXT_RESET, retain_dynamics=False))

    assert not e.action_dynamics
    assert not e.symbolic_theory.is_barrier(5)


def test_assimilate_feedback_records_pre_action_position_and_inhibits_root_on_death():
    e = _eng()
    e.avatar_feature = 1
    e.avatar_features = {1}
    e.bg_feature = 0
    e.symbolic_theory.induce_walkable(0, evidence="floor")
    e.symbolic_theory.induce_barrier(5, evidence="wall")
    for act_id, (dr, dc) in ((1, (-1, 0)), (2, (1, 0)), (3, (0, -1)), (4, (0, 1))):
        e.action_dynamics[act_id] = ActionDynamicsModel(
            action_id=act_id, delta_r=dr, delta_c=dc, confidence=1.0, probes_tested=3
        )
        e.tested_actions.add(act_id)

    base = np.zeros((7, 7), dtype=int)
    base[0, :] = 5
    base[-1, :] = 5
    base[:, 0] = 5
    base[:, -1] = 5
    base[2, 5] = 8

    s0 = (2, 1)
    g0 = base.copy()
    g0[s0] = 1
    a0, _ = e.step(EmbodiedObservation(lattice=g0, available_actions=[1, 2, 3, 4]))

    dr0, dc0 = e.action_dynamics[int(a0)].get_displacement()
    s1 = (s0[0] + dr0, s0[1] + dc0)
    assert s1 != s0
    g1 = base.copy()
    g1[s1] = 1
    a1, _ = e.step(EmbodiedObservation(lattice=g1, available_actions=[1, 2, 3, 4]))

    trace = list(e.working_memory.habenular_ior.episode_trace)
    assert len(trace) == 1
    assert trace[0].avatar_pos == s0
    assert trace[0].action_id == a0

    dr1, dc1 = e.action_dynamics[int(a1)].get_displacement()
    s2 = (max(1, min(5, s1[0] + dr1)), max(1, min(5, s1[1] + dc1)))
    g2 = base.copy()
    g2[s2] = 1
    e.step(
        EmbodiedObservation(
            lattice=g2,
            available_actions=[1, 2, 3, 4],
            terminal=TerminalSignal.FAILURE,
        )
    )

    assert e.working_memory.habenular_ior.is_action_inhibited(s0, a0)
    assert e.working_memory.habenular_ior.is_action_inhibited(s1, a1)
    assert len(e.working_memory.habenular_ior.episode_trace) == 0
    assert len(e.episodic_cortex.catastrophic_replays) == 1
