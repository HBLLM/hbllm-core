"""Tests for Intervention and Reversibility Model."""

import asyncio

import pytest

from hbllm.brain.control.intervention import InterventionAPI, ReversibilityPolicy


@pytest.mark.asyncio
async def test_reversibility_rollback():
    class DummyCore:
        pass

    api = InterventionAPI(DummyCore())

    rollback_called = False

    async def mock_rollback():
        nonlocal rollback_called
        rollback_called = True
        return True

    policy = ReversibilityPolicy(action_name="file.move", rollback_handler=mock_rollback)
    api.register_reversibility(policy)

    # Attempt undo
    success = await api.attempt_undo("file.move")
    assert success is True
    assert rollback_called is True

    # Attempt undo for unregistered action
    success_fail = await api.attempt_undo("email.send")
    assert success_fail is False


@pytest.mark.asyncio
async def test_reversibility_compensation():
    class DummyCore:
        pass

    api = InterventionAPI(DummyCore())

    async def mock_rollback_fail():
        return False

    compensation_called = False

    async def mock_compensate():
        nonlocal compensation_called
        compensation_called = True
        return True

    policy = ReversibilityPolicy(
        action_name="db.drop",
        rollback_handler=mock_rollback_fail,
        compensation_handler=mock_compensate,
    )
    api.register_reversibility(policy)

    success = await api.attempt_undo("db.drop")
    assert success is True
    assert compensation_called is True
