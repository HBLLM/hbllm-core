"""Tests for Phase 8: Meta Workspace & Self-Improvement."""

from __future__ import annotations

from hbllm.hcir.graph import (
    GoalNode,
    NodeLifecycle,
    ObservationNode,
    SkillNode,
)
from hbllm.hcir.workspace_tiers import TieredWorkspace

# ═══════════════════════════════════════════════════════════════════════════
# Meta Workspace Population Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMetaWorkspacePopulation:
    """Verify populate_meta_stats populates the meta tier."""

    def setup_method(self) -> None:
        self.workspace = TieredWorkspace()

    def test_empty_workspace_stats(self) -> None:
        stats = self.workspace.populate_meta_stats()
        assert stats["persistent_node_count"] == 0
        assert stats["brain_node_count"] == 0
        assert stats["active_task_frames"] == 0
        assert stats["avg_skill_success_rate"] == 0.0
        assert stats["goal_completion_rate"] == 0.0

    def test_stats_with_skills(self) -> None:
        self.workspace.persistent.upsert_node(
            SkillNode(id="s1", skill_name="weather", success_rate=0.9),
            author="test",
        )
        self.workspace.persistent.upsert_node(
            SkillNode(id="s2", skill_name="calendar", success_rate=0.7),
            author="test",
        )

        stats = self.workspace.populate_meta_stats()
        assert stats["persistent_node_count"] == 2
        assert stats["avg_skill_success_rate"] == 0.8  # (0.9+0.7)/2

    def test_stats_with_goals(self) -> None:
        self.workspace.persistent.upsert_node(
            GoalNode(id="g1", description="Done", lifecycle=NodeLifecycle.ARCHIVED),
            author="test",
        )
        self.workspace.persistent.upsert_node(
            GoalNode(id="g2", description="Active", lifecycle=NodeLifecycle.ACTIVE),
            author="test",
        )
        self.workspace.persistent.upsert_node(
            GoalNode(id="g3", description="Also done", lifecycle=NodeLifecycle.ARCHIVED),
            author="test",
        )

        stats = self.workspace.populate_meta_stats()
        assert stats["goal_completion_rate"] == round(2 / 3, 3)

    def test_meta_observation_node_created(self) -> None:
        """populate_meta_stats should create an ObservationNode in meta tier."""
        self.workspace.populate_meta_stats()

        node = self.workspace.meta.get_node("meta_stats_latest")
        assert node is not None
        assert isinstance(node, ObservationNode)
        assert "meta" in node.tags
        assert "self_model" in node.tags

    def test_brain_tier_counted(self) -> None:
        self.workspace.brain.upsert_node(
            ObservationNode(id="obs1", sensor_source="test"),
            author="test",
        )

        stats = self.workspace.populate_meta_stats()
        assert stats["brain_node_count"] == 1

    def test_task_frame_counted(self) -> None:
        self.workspace.create_task_frame("frame1")
        self.workspace.create_task_frame("frame2")

        stats = self.workspace.populate_meta_stats()
        assert stats["active_task_frames"] == 2

    def test_type_counts(self) -> None:
        self.workspace.persistent.upsert_node(
            SkillNode(id="s1", skill_name="test"),
            author="test",
        )
        self.workspace.persistent.upsert_node(
            GoalNode(id="g1", description="test"),
            author="test",
        )

        stats = self.workspace.populate_meta_stats()
        type_counts = stats["type_counts"]
        assert type_counts.get("skill", 0) == 1
        assert type_counts.get("goal", 0) == 1

    def test_repeated_calls_update_meta(self) -> None:
        """Multiple calls should update rather than duplicate."""
        self.workspace.populate_meta_stats()
        self.workspace.persistent.upsert_node(
            SkillNode(id="s_new", skill_name="new_skill"),
            author="test",
        )
        stats = self.workspace.populate_meta_stats()

        # Should still have exactly one meta_stats node
        node = self.workspace.meta.get_node("meta_stats_latest")
        assert node is not None
        assert stats["persistent_node_count"] == 1
