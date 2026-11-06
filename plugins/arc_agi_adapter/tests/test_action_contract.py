from __future__ import annotations

import numpy as np

from plugins.arc_agi_adapter.action_contract import ActionContract, EffectKind


def test_legal_id():
    r = ActionContract.validate(1, None, [1, 2, 3, 4], (8, 8))
    assert r.ok


def test_illegal_id():
    assert not ActionContract.validate(9, None, [1], (8, 8)).ok
    assert "illegal_action_id" in (ActionContract.validate(9, None, [1], (8, 8)).violation or "")


def test_unavailable():
    r = ActionContract.validate(5, None, [1, 2], (8, 8))
    assert not r.ok and "unavailable" in (r.violation or "")


def test_action6_needs_int_xy_in_bounds():
    base = (8, 8)
    assert not ActionContract.validate(6, None, [6], base).ok
    assert not ActionContract.validate(6, {"x": 1.5, "y": 2}, [6], base).ok
    assert not ActionContract.validate(6, {"x": 99, "y": 2}, [6], base).ok
    assert ActionContract.validate(6, {"x": 3, "y": 4}, [6], base).ok


def test_repair_nearest_legal():
    r = ActionContract.validate_or_repair(6, {"x": 99, "y": 1}, [1, 2, 3, 4], (8, 8))
    assert not r.ok and r.repaired_action_id == 4  # nearest legal discrete id


def test_reconcile_changed_and_rejected():
    a = np.zeros((4, 4), int)
    b = a.copy()
    b[1, 1] = 9
    e = ActionContract.reconcile(a, b, 4)
    assert e.kind == EffectKind.CHANGED and e.changed_cells == 1
    # blocked move: zero diff is UNCHANGED (real motor evidence), not rejected
    e2 = ActionContract.reconcile(a, a.copy(), 4)
    assert e2.kind == EffectKind.UNCHANGED
    e3 = ActionContract.reconcile(a, a.copy(), 7)
    assert e3.kind == EffectKind.UNCHANGED
    # env rejection: action not in available_actions at send time
    e4 = ActionContract.reconcile(a, b, 4, action_was_available=False)
    assert e4.kind == EffectKind.ENV_REJECTED
    # malformed frame (shape change) also counts as rejection
    e5 = ActionContract.reconcile(a, np.zeros((2, 2), int), 4)
    assert e5.kind == EffectKind.ENV_REJECTED
