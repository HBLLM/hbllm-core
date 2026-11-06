from __future__ import annotations

from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.percept import EpisodeBoundary, EpisodeBoundaryKind


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
