"""Tests for the Goal DAG and Execution Journals."""

from __future__ import annotations

import pytest

from hbllm.brain.goal_manager import GoalManager, GoalPriority, GoalStatus


@pytest.fixture
def temp_goal_manager(tmp_path):
    """Provides a GoalManager using a temporary DB."""
    return GoalManager(data_dir=tmp_path)


def test_goal_creation_and_dependencies(temp_goal_manager):
    # Create parent goal
    goal_a = temp_goal_manager.create_goal(
        name="Build API client",
        description="Factual communication adapter",
        goal_type="optimization",
        priority=GoalPriority.HIGH,
    )

    # Create child goal that depends on parent goal
    goal_b = temp_goal_manager.create_goal(
        name="Integrate Client with planner",
        description="Connect adapter to MCTS generator",
        goal_type="optimization",
        priority=GoalPriority.MEDIUM,
        dependencies=[goal_a.goal_id],
    )

    assert goal_a.status == GoalStatus.PENDING

    # Reload goal_b from DB to verify state resolution
    active_goals = temp_goal_manager.get_active_goals()
    b_loaded = next(g for g in active_goals if g.goal_id == goal_b.goal_id)

    assert b_loaded.status == GoalStatus.BLOCKED
    assert "Blocked by dependencies" in b_loaded.block_reason
    assert b_loaded.dependencies == [goal_a.goal_id]


def test_dag_resolution_on_completion(temp_goal_manager):
    goal_a = temp_goal_manager.create_goal(name="Task A", description="Desc A")
    goal_b = temp_goal_manager.create_goal(
        name="Task B", description="Desc B", dependencies=[goal_a.goal_id]
    )

    # Verify B is blocked
    active = temp_goal_manager.get_active_goals()
    b_loaded = next(g for g in active if g.goal_id == goal_b.goal_id)
    assert b_loaded.status == GoalStatus.BLOCKED

    # Complete Goal A
    temp_goal_manager.complete_goal(goal_a.goal_id)

    # Verify B is now pending
    active = temp_goal_manager.get_active_goals()
    b_loaded = next(g for g in active if g.goal_id == goal_b.goal_id)
    assert b_loaded.status == GoalStatus.PENDING
    assert b_loaded.block_reason == ""


def test_execution_journal_updates(temp_goal_manager):
    goal = temp_goal_manager.create_goal(name="Journal Test", description="Desc")

    temp_goal_manager.update_execution_journal(
        goal_id=goal.goal_id,
        checkpoint="Step 2: Database schema check",
        completed_steps=["Step 1: Init db migration"],
        blocked_reason="",
        next_action="Run dry-run migration",
    )

    active = temp_goal_manager.get_active_goals()
    loaded = next(g for g in active if g.goal_id == goal.goal_id)

    assert loaded.execution_journal["checkpoint"] == "Step 2: Database schema check"
    assert loaded.execution_journal["completed_steps"] == ["Step 1: Init db migration"]
    assert loaded.execution_journal["next_action"] == "Run dry-run migration"
