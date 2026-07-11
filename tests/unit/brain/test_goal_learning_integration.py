"""Tests for Goal-Learning integration (Phase 5).

Verifies that learning becomes mission-driven:
    Goals → Attention → Learning → Beliefs → Concepts → Actions
"""

from __future__ import annotations

import pytest

from hbllm.brain.emotion.goal_manager import GoalManager, GoalPriority


@pytest.fixture
def gm(tmp_path):
    return GoalManager(data_dir=str(tmp_path))


class TestCreateLearningGoal:
    """Test create_learning_goal()."""

    def test_creates_learning_goal(self, gm):
        goal = gm.create_learning_goal(topic="SQL injection")
        assert goal.goal_type == "learning"
        assert "SQL injection" in goal.name
        assert goal.metadata["learning_topic"] == "SQL injection"
        assert goal.metadata["motivation"] == "system"

    def test_with_parent(self, gm):
        parent = gm.create_goal(
            name="Build secure API",
            description="Main goal",
            goal_type="learning",
            priority=GoalPriority.HIGH,
        )
        child = gm.create_learning_goal(
            topic="Authentication",
            parent_goal_id=parent.goal_id,
        )
        assert child.metadata["parent_goal_id"] == parent.goal_id
        assert parent.goal_id in child.dependencies

    def test_custom_priority(self, gm):
        goal = gm.create_learning_goal(
            topic="Rate limiting",
            priority=GoalPriority.CRITICAL,
        )
        assert goal.priority == GoalPriority.CRITICAL


class TestGenerateFromContradictions:
    """Test generate_from_contradictions()."""

    def test_generates_goals_from_contradictions(self, gm):
        contradictions = [
            {
                "concept": "caching",
                "claim_a": "Caching improves performance",
                "claim_b": "Caching causes stale data",
                "severity": 0.8,
                "contradiction_id": "ctr_abc",
            },
            {
                "concept": "ORM",
                "claim_a": "ORM is beneficial",
                "claim_b": "ORM hides complexity",
                "severity": 0.3,
                "contradiction_id": "ctr_def",
            },
        ]
        goals = gm.generate_from_contradictions(contradictions)
        assert len(goals) == 2
        assert goals[0].priority == GoalPriority.HIGH  # severity 0.8
        assert goals[1].priority == GoalPriority.LOW  # severity 0.3
        assert goals[0].metadata["contradiction_id"] == "ctr_abc"

    def test_with_parent(self, gm):
        parent = gm.create_goal(
            name="Resolve all contradictions",
            description="Meta goal",
            goal_type="maintenance",
        )
        goals = gm.generate_from_contradictions(
            [{"concept": "test", "claim_a": "A", "claim_b": "B", "severity": 0.5}],
            parent_goal_id=parent.goal_id,
        )
        assert len(goals) == 1
        assert parent.goal_id in goals[0].dependencies


class TestGenerateFromWeakAreas:
    """Test generate_from_weak_areas()."""

    def test_generates_goals(self, gm):
        weak_areas = [
            {"concept": "buffer overflow", "score": 0.2},
            {"concept": "race condition", "score": 0.6},
        ]
        goals = gm.generate_from_weak_areas(weak_areas)
        assert len(goals) == 2
        assert goals[0].priority == GoalPriority.HIGH  # score 0.2
        assert goals[1].priority == GoalPriority.LOW  # score 0.6


class TestGetLearningGoals:
    """Test get_learning_goals()."""

    def test_returns_only_learning(self, gm):
        gm.create_learning_goal(topic="test1")
        gm.create_goal(
            name="Not learning",
            description="Optimization goal",
            goal_type="optimization",
        )
        gm.create_learning_goal(topic="test2")

        learning = gm.get_learning_goals()
        assert len(learning) == 2
        assert all(g.goal_type == "learning" for g in learning)


class TestSubordinateTo:
    """Test goal hierarchy linking."""

    def test_link_child_to_parent(self, gm):
        parent = gm.create_goal(
            name="Parent",
            description="P",
            goal_type="learning",
        )
        child = gm.create_goal(
            name="Child",
            description="C",
            goal_type="learning",
        )

        gm.subordinate_to(child.goal_id, parent.goal_id)

        # Verify parent has sub-goal
        active = gm.get_active_goals()
        parent_goal = next(g for g in active if g.goal_id == parent.goal_id)
        assert child.goal_id in parent_goal.sub_goals

    def test_idempotent(self, gm):
        parent = gm.create_goal(name="P", description="P", goal_type="learning")
        child = gm.create_goal(name="C", description="C", goal_type="learning")

        gm.subordinate_to(child.goal_id, parent.goal_id)
        gm.subordinate_to(child.goal_id, parent.goal_id)

        active = gm.get_active_goals()
        parent_goal = next(g for g in active if g.goal_id == parent.goal_id)
        assert parent_goal.sub_goals.count(child.goal_id) == 1


class TestStats:
    """Test enhanced stats."""

    def test_stats_include_learning(self, gm):
        gm.create_learning_goal(topic="test")
        stats = gm.stats()
        assert stats["active_learning_goals"] == 1
        assert "total_goals" in stats


class TestLearningGoalParentId:
    """Test LearningGoal.parent_goal_id integration."""

    def test_parent_goal_id_serialized(self):
        from hbllm.brain.learning.autonomous_learner import LearningGoal

        goal = LearningGoal(topic="test", parent_goal_id="goal_123")
        d = goal.to_dict()
        assert d["parent_goal_id"] == "goal_123"
