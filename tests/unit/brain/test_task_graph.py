"""Tests for TaskGraphRuntime — persistent DAG-based goal execution."""

from __future__ import annotations

from typing import Any

import pytest

from hbllm.brain.autonomy.task_graph import (
    Goal,
    GoalStatus,
    TaskGraphRuntime,
    TaskNode,
    TaskPriority,
    TaskStatus,
)


@pytest.fixture
def runtime(tmp_path):
    """Create a TaskGraphRuntime with a temp directory."""
    return TaskGraphRuntime(data_dir=str(tmp_path))


def _make_goal(**kwargs) -> Goal:
    defaults: dict[str, Any] = {"name": "Test Goal", "status": GoalStatus.ACTIVE}
    defaults.update(kwargs)
    return Goal(**defaults)  # type: ignore[arg-type]


def _make_task(**kwargs) -> TaskNode:
    defaults: dict[str, Any] = {"name": "Test Task", "action_topic": "test.action"}
    defaults.update(kwargs)
    return TaskNode(**defaults)  # type: ignore[arg-type]


class TestGoalLifecycle:
    def test_create_goal(self, runtime):
        goal = _make_goal()
        task_a = _make_task(name="Step A")
        goal_id = runtime.create_goal(goal, [task_a])
        assert goal_id == goal.goal_id

        fetched = runtime.get_goal(goal_id)
        assert fetched is not None
        assert fetched.name == "Test Goal"

    def test_create_goal_with_multiple_tasks(self, runtime):
        goal = _make_goal()
        tasks = [_make_task(name=f"Step {i}") for i in range(5)]
        runtime.create_goal(goal, tasks)
        all_tasks = runtime.get_tasks_for_goal(goal.goal_id)
        assert len(all_tasks) == 5

    def test_activate_goal(self, runtime):
        goal = _make_goal(status=GoalStatus.PENDING)
        runtime.create_goal(goal, [_make_task()])
        assert runtime.activate_goal(goal.goal_id) is True
        assert runtime.get_goal(goal.goal_id).status == GoalStatus.ACTIVE

    def test_pause_and_resume(self, runtime):
        goal = _make_goal()
        runtime.create_goal(goal, [_make_task()])
        assert runtime.pause_goal(goal.goal_id) is True
        assert runtime.get_goal(goal.goal_id).status == GoalStatus.PAUSED

        assert runtime.resume_goal(goal.goal_id) is True
        assert runtime.get_goal(goal.goal_id).status == GoalStatus.ACTIVE

    def test_cancel_goal(self, runtime):
        goal = _make_goal()
        t1 = _make_task(name="A")
        t2 = _make_task(name="B")
        runtime.create_goal(goal, [t1, t2])
        runtime.cancel_goal(goal.goal_id)

        fetched = runtime.get_goal(goal.goal_id)
        assert fetched.status == GoalStatus.CANCELLED

        tasks = runtime.get_tasks_for_goal(goal.goal_id)
        for t in tasks:
            assert t.status == TaskStatus.SKIPPED

    def test_get_active_goals(self, runtime):
        g1 = _make_goal(name="Active 1")
        g2 = _make_goal(name="Active 2", status=GoalStatus.PENDING)
        runtime.create_goal(g1, [_make_task()])
        runtime.create_goal(g2, [_make_task()])

        active = runtime.get_active_goals()
        assert len(active) == 1
        assert active[0].name == "Active 1"

    def test_get_active_goals_by_tenant(self, runtime):
        g1 = _make_goal(tenant_id="tenant_a")
        g2 = _make_goal(tenant_id="tenant_b")
        runtime.create_goal(g1, [_make_task()])
        runtime.create_goal(g2, [_make_task()])

        filtered = runtime.get_active_goals(tenant_id="tenant_a")
        assert len(filtered) == 1

    def test_get_nonexistent_goal(self, runtime):
        assert runtime.get_goal("nonexistent") is None


class TestTaskDependencies:
    def test_root_tasks_become_ready(self, runtime):
        goal = _make_goal()
        t1 = _make_task(name="Root")
        runtime.create_goal(goal, [t1])

        ready = runtime.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].status == TaskStatus.READY

    def test_dependent_task_stays_pending(self, runtime):
        goal = _make_goal()
        t1 = _make_task(name="First")
        t2 = _make_task(name="Second", dependencies=[t1.task_id])
        runtime.create_goal(goal, [t1, t2])

        ready = runtime.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].task_id == t1.task_id

    def test_completing_dependency_unlocks_next(self, runtime):
        goal = _make_goal()
        t1 = _make_task(name="First")
        t2 = _make_task(name="Second", dependencies=[t1.task_id])
        runtime.create_goal(goal, [t1, t2])

        runtime.mark_task_running(t1.task_id)
        runtime.complete_task(t1.task_id, result={"data": "ok"})

        ready = runtime.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].task_id == t2.task_id

    def test_diamond_dependency(self, runtime):
        """A → B, A → C, B+C → D"""
        goal = _make_goal()
        a = _make_task(name="A")
        b = _make_task(name="B", dependencies=[a.task_id])
        c = _make_task(name="C", dependencies=[a.task_id])
        d = _make_task(name="D", dependencies=[b.task_id, c.task_id])
        runtime.create_goal(goal, [a, b, c, d])

        # Only A is ready
        ready = runtime.get_ready_tasks()
        assert len(ready) == 1

        # Complete A → B and C become ready
        runtime.mark_task_running(a.task_id)
        runtime.complete_task(a.task_id)
        ready = runtime.get_ready_tasks()
        assert len(ready) == 2

        # Complete B and C → D becomes ready
        runtime.mark_task_running(b.task_id)
        runtime.complete_task(b.task_id)
        runtime.mark_task_running(c.task_id)
        runtime.complete_task(c.task_id)

        ready = runtime.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].task_id == d.task_id

    def test_priority_ordering(self, runtime):
        goal = _make_goal()
        low = _make_task(name="Low", priority=TaskPriority.LOW)
        crit = _make_task(name="Critical", priority=TaskPriority.CRITICAL)
        runtime.create_goal(goal, [low, crit])

        ready = runtime.get_ready_tasks()
        assert ready[0].priority == TaskPriority.CRITICAL


class TestTaskExecution:
    def test_mark_running(self, runtime):
        goal = _make_goal()
        t = _make_task()
        runtime.create_goal(goal, [t])
        assert runtime.mark_task_running(t.task_id) is True

        tasks = runtime.get_tasks_for_goal(goal.goal_id)
        assert tasks[0].status == TaskStatus.RUNNING
        assert tasks[0].started_at > 0

    def test_complete_task(self, runtime):
        goal = _make_goal()
        t = _make_task()
        runtime.create_goal(goal, [t])
        runtime.mark_task_running(t.task_id)
        assert runtime.complete_task(t.task_id, {"output": "done"}) is True

        tasks = runtime.get_tasks_for_goal(goal.goal_id)
        assert tasks[0].status == TaskStatus.COMPLETED
        assert tasks[0].result == {"output": "done"}

    def test_goal_auto_completes(self, runtime):
        goal = _make_goal()
        t = _make_task()
        runtime.create_goal(goal, [t])
        runtime.mark_task_running(t.task_id)
        runtime.complete_task(t.task_id)

        fetched = runtime.get_goal(goal.goal_id)
        assert fetched.status == GoalStatus.COMPLETED

    def test_goal_fails_if_task_fails(self, runtime):
        goal = _make_goal()
        t = _make_task(max_retries=0)
        runtime.create_goal(goal, [t])
        runtime.mark_task_running(t.task_id)
        result = runtime.fail_task(t.task_id, error="crash")
        assert result == "failed"

        fetched = runtime.get_goal(goal.goal_id)
        assert fetched.status == GoalStatus.FAILED


class TestRetryAndFailure:
    def test_retry_on_failure(self, runtime):
        goal = _make_goal()
        t = _make_task(max_retries=2)
        runtime.create_goal(goal, [t])
        runtime.mark_task_running(t.task_id)
        result = runtime.fail_task(t.task_id, error="timeout")
        assert result == "retrying"

        ready = runtime.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].retry_count == 1

    def test_permanent_failure_after_retries(self, runtime):
        goal = _make_goal()
        t = _make_task(max_retries=1)
        runtime.create_goal(goal, [t])

        # First attempt + 1 retry = 2 tries total
        runtime.mark_task_running(t.task_id)
        runtime.fail_task(t.task_id, error="err1")  # retrying (count=1)
        runtime.mark_task_running(t.task_id)
        result = runtime.fail_task(t.task_id, error="err2")  # failed (count=2)
        assert result == "failed"

    def test_failure_blocks_dependents(self, runtime):
        goal = _make_goal()
        t1 = _make_task(name="A", max_retries=0)
        t2 = _make_task(name="B", dependencies=[t1.task_id])
        runtime.create_goal(goal, [t1, t2])

        runtime.mark_task_running(t1.task_id)
        runtime.fail_task(t1.task_id, error="crash")

        tasks = runtime.get_tasks_for_goal(goal.goal_id)
        blocked = [t for t in tasks if t.status == TaskStatus.BLOCKED]
        assert len(blocked) == 1
        assert blocked[0].task_id == t2.task_id


class TestRecovery:
    def test_recover_on_boot(self, runtime):
        goal = _make_goal()
        t = _make_task()
        runtime.create_goal(goal, [t])
        runtime.mark_task_running(t.task_id)

        # Simulate reboot — create new runtime against same DB
        runtime2 = TaskGraphRuntime(data_dir=str(runtime.data_dir))
        count = runtime2.recover_on_boot()
        assert count == 1

        ready = runtime2.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].task_id == t.task_id


class TestDynamicTasks:
    def test_add_task_to_goal(self, runtime):
        goal = _make_goal()
        t1 = _make_task(name="Initial")
        runtime.create_goal(goal, [t1])

        t2 = _make_task(name="Added later", dependencies=[t1.task_id])
        runtime.add_task_to_goal(goal.goal_id, t2)

        all_tasks = runtime.get_tasks_for_goal(goal.goal_id)
        assert len(all_tasks) == 2


class TestIntrospection:
    def test_goal_progress(self, runtime):
        goal = _make_goal()
        t1 = _make_task(name="A")
        t2 = _make_task(name="B", dependencies=[t1.task_id])
        runtime.create_goal(goal, [t1, t2])

        progress = runtime.get_goal_progress(goal.goal_id)
        assert progress["total"] == 2
        assert progress["completed"] == 0

        runtime.mark_task_running(t1.task_id)
        runtime.complete_task(t1.task_id)
        progress = runtime.get_goal_progress(goal.goal_id)
        assert progress["completed"] == 1
        assert progress["progress"] == 0.5

    def test_snapshot(self, runtime):
        goal = _make_goal()
        runtime.create_goal(goal, [_make_task()])
        snap = runtime.snapshot()
        assert "goals" in snap
        assert "tasks" in snap
        assert snap["max_concurrent"] == 5

    def test_empty_progress(self, runtime):
        progress = runtime.get_goal_progress("nonexistent")
        assert progress["total"] == 0
        assert progress["progress"] == 0.0
