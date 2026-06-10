"""
End-to-End Integration Test for Tool Recovery, Rollbacks, and Human Intervention in HBLLM Core.

Verifies:
1. ReversibilityPolicy rollback execution when a mutating tool action needs to be undone.
2. Graceful compensation handler fallback when a rollback fails.
3. InterventionAPI human control overrides (pause, resume, stop) correctly propagating to AutonomyCore.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from hbllm.brain.control.intervention import InterventionAPI, ReversibilityPolicy
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


@pytest.fixture
async def intervention_bus_system():
    """Boot InProcessBus and setup mock autonomy core and InterventionAPI."""
    bus = InProcessBus()
    await bus.start()

    # Create mock AutonomyCore supporting pause, resume, stop
    mock_core = MagicMock()
    mock_core.pause = MagicMock()
    mock_core.resume = MagicMock()
    mock_core.stop = MagicMock()

    api = InterventionAPI(mock_core)

    yield bus, api, mock_core

    await bus.stop()


@pytest.mark.asyncio
async def test_e2e_tool_rollback_success(intervention_bus_system) -> None:
    bus, api, mock_core = intervention_bus_system

    # State variables to track side effects
    db_record_created = True
    rollback_called = False

    async def mock_db_rollback() -> bool:
        nonlocal db_record_created, rollback_called
        db_record_created = False  # Reverse the creation
        rollback_called = True
        return True

    # 1. Register reversibility policy
    policy = ReversibilityPolicy(
        action_name="db.create_record",
        rollback_handler=mock_db_rollback,
    )
    api.register_reversibility(policy)

    # 2. Trigger rollback (simulating an undo request for a failed action)
    success = await api.attempt_undo("db.create_record")

    # 3. Verify that rollback succeeded and state was restored
    assert success is True
    assert db_record_created is False
    assert rollback_called is True


@pytest.mark.asyncio
async def test_e2e_tool_rollback_compensation_fallback(intervention_bus_system) -> None:
    bus, api, mock_core = intervention_bus_system

    rollback_called = False
    compensation_called = False

    async def mock_rollback_fail() -> bool:
        nonlocal rollback_called
        rollback_called = True
        return False  # Rollback fails (e.g. database connection lost)

    async def mock_compensate() -> bool:
        nonlocal compensation_called
        compensation_called = True  # Run compensation (e.g. queue offline sync job)
        return True

    # 1. Register policy with failing rollback and working compensation
    policy = ReversibilityPolicy(
        action_name="file.write_external",
        rollback_handler=mock_rollback_fail,
        compensation_handler=mock_compensate,
    )
    api.register_reversibility(policy)

    # 2. Trigger undo
    success = await api.attempt_undo("file.write_external")

    # 3. Verify that rollback failed, but compensation was executed to resolve the state
    assert success is True
    assert rollback_called is True
    assert compensation_called is True


@pytest.mark.asyncio
async def test_e2e_human_intervention_api(intervention_bus_system) -> None:
    bus, api, mock_core = intervention_bus_system

    # 1. Test Human Override: Pause
    api.pause()
    mock_core.pause.assert_called_once()

    # 2. Test Human Override: Resume
    api.resume()
    mock_core.resume.assert_called_once()

    # 3. Test Human Override: Stop
    api.stop(flush_queue=True)
    mock_core.stop.assert_called_once_with(flush_queue=True)
