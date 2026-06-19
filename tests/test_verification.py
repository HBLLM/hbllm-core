"""Tests for the Phase 4.1 Verification and Conflict Resolution Layer."""

from __future__ import annotations

import sqlite3
import time

import pytest

from hbllm.brain.autonomy.task_graph import (
    Goal,
    TaskGraphRuntime,
    TaskNode,
    TaskStatus,
    VerificationRule,
)
from hbllm.brain.world_state import WorldStateEngine
from hbllm.perception.reality_bus import EventOrigin, PerceptionEvent, PerceptionModality


@pytest.fixture
def temp_runtime(tmp_path):
    """Provides a temporary SQLite TaskGraphRuntime."""
    return TaskGraphRuntime(data_dir=tmp_path)


@pytest.fixture
def world_engine():
    """Provides a fresh WorldStateEngine."""
    return WorldStateEngine()


class TestVerificationAndConflictResolution:
    @pytest.mark.asyncio
    async def test_world_state_conflict_resolution(self, world_engine):
        """Test that WorldStateEngine correctly resolves conflicts using trust and recency."""
        now = time.time()

        # 1. App event (Medium trust)
        app_event = PerceptionEvent(
            entity_id="device_1",
            event_type="status",
            modality=PerceptionModality.APP,
            origin=EventOrigin.EXTERNAL,
            confidence=0.8,
            source_trust=0.8,
            payload={"state": "idle"},
            event_timestamp=now - 10.0,  # 10 seconds ago
        )
        await world_engine.handle_normalized_event(app_event)

        state = world_engine.get_entity_state("device_1")
        assert state.properties["state"] == "idle"

        # 2. System event (High trust) - Should override App
        sys_event = PerceptionEvent(
            entity_id="device_1",
            event_type="status",
            modality=PerceptionModality.SYSTEM,
            origin=EventOrigin.SYSTEM,
            confidence=0.9,
            source_trust=1.0,
            payload={"state": "active"},
            event_timestamp=now,
        )
        await world_engine.handle_normalized_event(sys_event)

        state = world_engine.get_entity_state("device_1")
        assert state.properties["state"] == "active"
        assert state.confidence > 0.8

        # 3. Inferred event (Low trust) - Should NOT override System
        inferred_event = PerceptionEvent(
            entity_id="device_1",
            event_type="status",
            modality=PerceptionModality.INFERRED,
            origin=EventOrigin.AUTONOMY,
            confidence=0.5,
            source_trust=0.2,
            payload={"state": "sleeping"},
            event_timestamp=now,
        )
        await world_engine.handle_normalized_event(inferred_event)

        state = world_engine.get_entity_state("device_1")
        # Still active, inferred event was too weak
        assert state.properties["state"] == "active"

    def test_hybrid_verification_success(self, temp_runtime, world_engine):
        """Test that event-driven verification marks task as completed."""
        goal = Goal(name="Test Goal")

        rule = VerificationRule(
            entity_id="test_entity",
            property_name="status",
            expected_value="done",
            min_match_score=0.7,
            time_window_s=10.0,
        )
        task = TaskNode(name="Test Task", verification_rule=rule)

        temp_runtime.create_goal(goal, [task])

        # Mark running then completed -> transitions to VERIFYING because it has a rule
        temp_runtime.mark_task_running(task.task_id)
        temp_runtime.complete_task(task.task_id)

        # Verify it is in VERIFYING state
        tasks = temp_runtime.get_tasks_for_goal(goal.goal_id)
        assert tasks[0].status == TaskStatus.VERIFYING

        # Provide the missing reality event
        event = PerceptionEvent(
            entity_id="test_entity",
            event_type="test",
            confidence=0.9,
            source_trust=1.0,
            payload={"status": "done"},
            event_timestamp=time.time(),
        )
        # We manually update for test
        entity_state = world_engine.get_entity_state("test_entity")
        if not entity_state:
            from hbllm.brain.world_state import EntityState

            entity_state = EntityState(entity_id="test_entity")
            world_engine._graph["test_entity"] = entity_state
        entity_state.update(event, now=time.time())

        # Run verification loop
        temp_runtime.verify_pending_tasks(world_engine)

        # Should now be COMPLETED
        tasks = temp_runtime.get_tasks_for_goal(goal.goal_id)
        assert tasks[0].status == TaskStatus.COMPLETED

    def test_hybrid_verification_timeout(self, temp_runtime, world_engine):
        """Test that timeout fallback transitions to CORRECTING then UNCERTAIN."""
        goal = Goal(name="Test Goal")

        rule = VerificationRule(
            entity_id="test_entity",
            property_name="status",
            expected_value="done",
            max_wait_time_s=1.0,  # 1 second timeout
        )
        task = TaskNode(name="Test Task", verification_rule=rule, max_correction_attempts=1)

        temp_runtime.create_goal(goal, [task])
        temp_runtime.mark_task_running(task.task_id)
        temp_runtime.complete_task(task.task_id)

        # Simulate time passing beyond max_wait_time_s
        with sqlite3.connect(temp_runtime.db_path) as conn:
            conn.execute(
                "UPDATE task_nodes SET verification_started_at = ? WHERE task_id = ?",
                (time.time() - 5.0, task.task_id),
            )

        # Run verification loop 1 -> Should hit timeout and go to CORRECTING (attempt 1)
        temp_runtime.verify_pending_tasks(world_engine)
        tasks = temp_runtime.get_tasks_for_goal(goal.goal_id)
        assert tasks[0].status == TaskStatus.CORRECTING
        assert tasks[0].correction_attempts == 1

        # We simulate the planner putting it back to VERIFYING (skipping RUNNING for brevity)
        with sqlite3.connect(temp_runtime.db_path) as conn:
            conn.execute(
                "UPDATE task_nodes SET status = ?, verification_started_at = ? WHERE task_id = ?",
                (TaskStatus.VERIFYING.value, time.time() - 5.0, task.task_id),
            )

        # Run verification loop 2 -> Should hit timeout and go to UNCERTAIN (exceeded max_correction)
        temp_runtime.verify_pending_tasks(world_engine)
        tasks = temp_runtime.get_tasks_for_goal(goal.goal_id)
        assert tasks[0].status == TaskStatus.UNCERTAIN
