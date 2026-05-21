"""Tests for Idempotency Tracking Engine."""

import pytest

from hbllm.brain.embodiment.idempotency import IdempotencyEngine


def test_idempotency_locking():
    engine = IdempotencyEngine()

    key1 = engine.generate_key("goal_1", "email.send", {"to": "alice"})

    # First execution should succeed
    assert engine.check_and_lock(key1) is True

    # Immediate retry should fail (duplicate prevention)
    assert engine.check_and_lock(key1) is False

    # Different parameters yield different key
    key2 = engine.generate_key("goal_1", "email.send", {"to": "bob"})
    assert engine.check_and_lock(key2) is True

    # Clear lock allows retry
    engine.clear_lock(key1)
    assert engine.check_and_lock(key1) is True
