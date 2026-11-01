from __future__ import annotations

import numpy as np

from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine


def _engine():
    return AutonomousEpistemicEngine(enable_logging=False, seed=0)


def _grid(av=(2, 2)):
    g = np.zeros((6, 6), dtype=int)
    g[av] = 5
    return g


def test_translate_learned_from_motion_sample():
    e = _engine()
    e.record_action_effect(2, "translate")
    assert e.expected_effect(2) == "translate"
    assert e.motor_posterior(2) == {"translate": 1.0}
    assert e.effect_entropy(2) == 0.0
    assert e.effect_entropy(9) == 1.0  # unprobed -> max uncertainty


def test_blocked_then_free_converges():
    e = _engine()
    e.record_action_effect(1, "no_op")  # first try blocked (0 diff)
    e.record_action_effect(1, "translate")  # free
    e.record_action_effect(1, "translate")
    e.record_action_effect(1, "translate")
    post = e.motor_posterior(1)
    assert post["translate"] == 0.75
    assert e.expected_effect(1) == "translate"


def test_state_change_non_moving_action():
    e = _engine()
    e.record_action_effect(5, "state_change")
    e.record_action_effect(5, "state_change")
    assert e.expected_effect(5) == "state_change"
    assert e.consecutive_null_count(5) == 0


def test_true_noop_after_three_samples_and_consec_cap():
    e = _engine()
    for _ in range(3):
        e.record_action_effect(7, "no_op")
    assert e.expected_effect(7) == "no_op"
    assert e.consecutive_null_count(7) == 3
    e.record_action_effect(7, "translate")  # resets consecutive nulls
    assert e.consecutive_null_count(7) == 0


def test_point_effect_and_entropy_mixture():
    e = _engine()
    e.record_action_effect(6, "point_effect")
    e.record_action_effect(6, "no_op")
    post = e.motor_posterior(6)
    assert post == {"point_effect": 0.5, "no_op": 0.5}
    assert 0.9 < e.effect_entropy(6) <= 1.0
