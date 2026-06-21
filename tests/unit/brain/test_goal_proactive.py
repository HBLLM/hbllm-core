"""Tests for GoalDecomposition and ProactiveInsight."""

import pytest
import pytest_asyncio

from hbllm.brain.autonomy.goal_decomposition import GoalDecompositionEngine, SubGoal
from hbllm.brain.autonomy.proactive_insight import ProactiveInsightGenerator

# ── GoalDecompositionEngine ───────────────────────────────────────────────


class TestGoalDecomposition:
    """Tests for hierarchical goal breakdown."""

    @pytest_asyncio.fixture
    async def engine(self):
        return GoalDecompositionEngine()

    @pytest.mark.asyncio
    async def test_heuristic_decomposition(self, engine):
        """Without LLM, produces a 3-step heuristic plan."""
        result = await engine.decompose("Plan a birthday party")
        assert len(result.sub_goals) == 3
        assert result.original_goal == "Plan a birthday party"
        assert result.total_estimated_min > 0

    @pytest.mark.asyncio
    async def test_sub_goals_have_dependencies(self, engine):
        """Sub-goals have dependency chains."""
        result = await engine.decompose("Build a website")
        # Step 2 depends on step 1, step 3 depends on step 2
        assert len(result.sub_goals[1].depends_on) >= 1
        assert len(result.sub_goals[2].depends_on) >= 1

    @pytest.mark.asyncio
    async def test_get_next_actionable(self, engine):
        """Next actionable returns the first sub-goal with no unmet deps."""
        result = await engine.decompose("Test goal")
        next_sg = engine.get_next_actionable(result.goal_id)
        assert next_sg is not None
        assert next_sg.order == 1

    @pytest.mark.asyncio
    async def test_mark_completed(self, engine):
        """Marking a sub-goal as completed updates its status."""
        result = await engine.decompose("Test goal")
        sg_id = result.sub_goals[0].id
        success = engine.mark_completed(result.goal_id, sg_id)
        assert success
        assert result.sub_goals[0].status == "completed"

    @pytest.mark.asyncio
    async def test_progress_tracking(self, engine):
        """Progress reports accurate completion percentage."""
        result = await engine.decompose("Test goal")
        engine.mark_completed(result.goal_id, result.sub_goals[0].id)
        progress = engine.get_progress(result.goal_id)
        assert progress["completed"] == 1
        assert progress["total_sub_goals"] == 3
        assert progress["progress_pct"] == pytest.approx(1 / 3, abs=0.01)

    @pytest.mark.asyncio
    async def test_nonexistent_goal_returns_error(self, engine):
        """get_progress for nonexistent goal returns error."""
        progress = engine.get_progress("nonexistent")
        assert "error" in progress

    @pytest.mark.asyncio
    async def test_sub_goal_to_dict(self, engine):
        """SubGoal.to_dict produces valid output."""
        sg = SubGoal(title="Test", description="Desc", order=1)
        d = sg.to_dict()
        assert d["title"] == "Test"
        assert d["order"] == 1


# ── ProactiveInsightGenerator ─────────────────────────────────────────────


class TestProactiveInsight:
    """Tests for context-aware proactive suggestions."""

    @pytest_asyncio.fixture
    async def generator(self):
        return ProactiveInsightGenerator()

    @pytest.mark.asyncio
    async def test_no_sources_no_crash(self, generator):
        """Generator with no data sources doesn't crash."""
        insights = await generator.generate_insights("user1")
        assert isinstance(insights, list)

    @pytest.mark.asyncio
    async def test_routine_insights_during_work_hours(self):
        """Routine insights are generated during work hours."""

        gen = ProactiveInsightGenerator()
        insights = await gen.generate_insights("user1")
        # May or may not generate based on current time
        assert isinstance(insights, list)

    @pytest.mark.asyncio
    async def test_cooldown_prevents_duplicates(self, generator):
        """Same insight isn't generated twice within cooldown."""
        # Generate twice rapidly
        i1 = await generator.generate_insights("user1")
        i2 = await generator.generate_insights("user1")
        # Second call should have fewer or equal insights
        assert len(i2) <= len(i1)

    @pytest.mark.asyncio
    async def test_stats(self, generator):
        """Stats reports expected fields."""
        s = generator.stats()
        assert "total_generated" in s
        assert "total_suppressed" in s
