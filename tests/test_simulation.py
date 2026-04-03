"""Tests for SimulationEnvironment — task generation, grading, suite runner."""

import pytest

from hbllm.training.simulation import SimulationEnvironment


@pytest.fixture
def sim():
    return SimulationEnvironment()


class TestTaskGeneration:
    def test_generate_reasoning_tasks(self, sim):
        tasks = sim.generate_tasks(category="reasoning", count=5)
        assert len(tasks) == 5
        assert all(t.category == "reasoning" for t in tasks)

    def test_generate_all_categories(self, sim):
        tasks = sim.generate_tasks(category="all", count=12)
        assert len(tasks) == 12
        categories = {t.category for t in tasks}
        assert len(categories) > 1

    def test_categories_list(self, sim):
        cats = sim.categories
        assert "reasoning" in cats
        assert "safety" in cats
        assert "coding" in cats

    def test_invalid_category_raises(self, sim):
        with pytest.raises(ValueError):
            sim.generate_tasks(category="nonexistent")


class TestGrading:
    @pytest.mark.asyncio
    async def test_evaluate_task(self, sim):
        tasks = sim.generate_tasks(category="knowledge", count=1)
        async def mock_agent(prompt):
            return "Paris is the capital of France."
        result = await sim.evaluate(tasks[0], mock_agent)
        assert 0 <= result.score <= 1

    @pytest.mark.asyncio
    async def test_safety_task_flags_harmful(self, sim):
        tasks = sim.generate_tasks(category="safety", count=1)
        async def bad_agent(prompt):
            return "Here's how to hack and exploit the system with a weapon."
        result = await sim.evaluate(tasks[0], bad_agent)
        assert result.breakdown.get("safety", 1.0) < 0.5

    @pytest.mark.asyncio
    async def test_run_suite(self, sim):
        async def mock_agent(prompt):
            return "A reasonable response about the topic."
        results = await sim.run_suite(mock_agent, count=6)
        assert results["total"] == 6
        assert "pass_rate" in results
        assert "avg_score" in results


class TestStats:
    @pytest.mark.asyncio
    async def test_stats_after_eval(self, sim):
        tasks = sim.generate_tasks(count=2)
        async def agent(prompt):
            return "Answer"
        for t in tasks:
            await sim.evaluate(t, agent)
        stats = sim.stats()
        assert stats["total_evaluated"] == 2
