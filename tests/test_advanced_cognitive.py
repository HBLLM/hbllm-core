"""Tests for SkillRegistry, GoalManager, SelfModel, CognitiveMetrics, WorldSimulator, ConceptExtractor."""

import pytest
from hbllm.brain.skill_registry import SkillRegistry
from hbllm.brain.goal_manager import GoalManager, GoalPriority
from hbllm.brain.self_model import SelfModel
from hbllm.brain.cognitive_metrics import CognitiveMetrics
from hbllm.brain.world_simulator import WorldSimulator
from hbllm.memory.concept_extractor import ConceptExtractor


# ─── SkillRegistry ───────────────────────────────────────────────────────

class TestSkillRegistry:
    @pytest.fixture
    def registry(self, tmp_path):
        return SkillRegistry(data_dir=str(tmp_path))

    def test_extract_skill(self, registry):
        skill = registry.extract_and_store(
            task_description="Summarize a technical document",
            execution_trace=[{"action": "read doc"}, {"action": "extract key points"}, {"action": "write summary"}],
            tools_used=["browser_node"],
            success=True, category="writing",
        )
        assert skill is not None
        assert skill.category == "writing"
        assert len(skill.steps) == 3

    def test_failed_task_no_skill(self, registry):
        skill = registry.extract_and_store(
            task_description="Failed task", execution_trace=[{"action": "try"}],
            tools_used=[], success=False,
        )
        assert skill is None

    def test_find_skill(self, registry):
        registry.extract_and_store(
            task_description="Debug Python error traceback",
            execution_trace=[{"action": "read error"}, {"action": "fix code"}],
            tools_used=["execution_node"], success=True, category="coding",
        )
        results = registry.find_skill("Python debug", category="coding")
        assert len(results) >= 1

    def test_execution_tracking(self, registry):
        skill = registry.extract_and_store(
            "Task A", [{"action": "step1"}], ["tool1"], True,
        )
        registry.record_execution(skill.skill_id, success=True, latency_ms=150.0)
        updated = registry.get_skill(skill.skill_id)
        assert updated.invocations == 1
        assert updated.avg_latency_ms == 150.0


# ─── GoalManager ─────────────────────────────────────────────────────────

class TestGoalManager:
    @pytest.fixture
    def gm(self, tmp_path):
        return GoalManager(data_dir=str(tmp_path))

    def test_create_goal(self, gm):
        goal = gm.create_goal("Improve coding", "Get better at code generation", goal_type="learning")
        assert goal.name == "Improve coding"
        stats = gm.stats()
        assert stats["total_goals"] == 1

    def test_priority_scheduling(self, gm):
        gm.create_goal("Low priority", "desc", priority=GoalPriority.LOW)
        gm.create_goal("High priority", "desc", priority=GoalPriority.HIGH)
        gm.create_goal("Critical", "desc", priority=GoalPriority.CRITICAL)
        next_goal = gm.next_goal()
        assert next_goal.priority == GoalPriority.CRITICAL

    def test_update_progress(self, gm):
        goal = gm.create_goal("Test goal", "desc")
        gm.update_progress(goal.goal_id, 0.5, "Halfway done")
        gm.complete_goal(goal.goal_id)
        stats = gm.stats()
        assert stats["completed"] == 1

    def test_auto_generate_from_performance(self, gm):
        goals = gm.generate_from_performance({
            "hallucination_rate": 0.2, "avg_latency_ms": 6000,
        })
        assert len(goals) >= 2  # hallucination + latency


# ─── SelfModel ───────────────────────────────────────────────────────────

class TestSelfModel:
    @pytest.fixture
    def self_model(self, tmp_path):
        return SelfModel(data_dir=str(tmp_path))

    def test_record_outcome(self, self_model):
        self_model.record_outcome("coding", success=True, confidence=0.9)
        cap = self_model.get_capability("coding")
        assert cap is not None
        assert cap.score == 1.0

    def test_strengths_and_weaknesses(self, self_model):
        for _ in range(10):
            self_model.record_outcome("coding", success=True)
            self_model.record_outcome("medical", success=False)
        assert "coding" in self_model.get_strengths(min_samples=5)
        assert "medical" in self_model.get_weaknesses(min_samples=5)

    def test_should_delegate(self, self_model):
        for _ in range(10):
            self_model.record_outcome("legal", success=False)
        assert self_model.should_delegate("legal") is True

    def test_recommend_model(self, self_model):
        for _ in range(10):
            self_model.record_outcome("math", success=True)
        assert self_model.recommend_model("math") == "default"
        for _ in range(10):
            self_model.record_outcome("rare_domain", success=False)
        assert self_model.recommend_model("rare_domain") == "specialist"


# ─── CognitiveMetrics ────────────────────────────────────────────────────

class TestCognitiveMetrics:
    @pytest.fixture
    def metrics(self, tmp_path):
        return CognitiveMetrics(data_dir=str(tmp_path))

    def test_record_and_query(self, metrics):
        metrics.record_reasoning(0.8)
        metrics.record_reasoning(0.9)
        result = metrics.get_metric("reasoning_score")
        assert result["count"] == 2
        assert result["avg"] == 0.85

    def test_snapshot(self, metrics):
        metrics.record_reasoning(0.7)
        metrics.record_hallucination(False)
        metrics.record_tool_result(True, "api")
        snap = metrics.snapshot()
        assert snap.reasoning_score == 0.7
        assert snap.tool_success_rate == 1.0

    def test_dashboard_metrics(self, metrics):
        metrics.record_latency(200.0, "inference")
        dash = metrics.get_dashboard_metrics()
        assert "avg_latency_ms" in dash
        assert dash["avg_latency_ms"] == 200.0


# ─── WorldSimulator ──────────────────────────────────────────────────────

class TestWorldSimulator:
    @pytest.fixture
    def simulator(self):
        return WorldSimulator(max_scenarios=3)

    @pytest.mark.asyncio
    async def test_simulate_strategies(self, simulator):
        strategies = [
            {"name": "Direct", "steps": ["do it"], "tools": []},
            {"name": "Research first", "steps": ["research", "plan", "execute"], "tools": ["browser"]},
        ]
        result = await simulator.simulate("Solve the problem", strategies)
        assert result.best_scenario is not None
        assert len(result.all_scenarios) == 2
        assert result.simulation_time_ms > 0

    @pytest.mark.asyncio
    async def test_simple_plan_preferred(self, simulator):
        strategies = [
            {"name": "Simple", "steps": ["do"], "tools": []},
            {"name": "Complex", "steps": [f"step{i}" for i in range(12)], "tools": ["api", "browser", "database"]},
        ]
        result = await simulator.simulate("Goal", strategies)
        # Simple plan should score higher (fewer risks)
        assert result.best_scenario.strategy == "Simple"

    @pytest.mark.asyncio
    async def test_stats(self, simulator):
        await simulator.simulate("g", [{"name": "a", "steps": ["s"]}])
        assert simulator.stats()["simulations_run"] == 1


# ─── ConceptExtractor ────────────────────────────────────────────────────

class TestConceptExtractor:
    @pytest.fixture
    def extractor(self):
        return ConceptExtractor(min_frequency=2, min_keyword_count=1)

    def test_extract_concepts_from_queries(self, extractor):
        queries = [
            "How to configure Laravel queue?",
            "Laravel queue worker keeps dying",
            "Fix Laravel queue timeout issue",
            "Python list comprehension example",
            "Python string formatting",
        ]
        concepts = extractor.extract_from_queries(queries)
        assert len(concepts) >= 1
        # Laravel should be a strong concept
        names = [c.name.lower() for c in concepts]
        assert any("laravel" in n for n in names)

    def test_extract_rules(self, extractor):
        queries = [
            "How to fix Laravel error?",
            "How to solve Laravel queue error?",
            "How to debug Laravel issue?",
        ]
        concepts = extractor.extract_from_queries(queries)
        if concepts:
            assert any("procedural" in r.lower() or "how-to" in r.lower()
                       for c in concepts for r in c.rules)

    def test_no_concepts_from_unique_queries(self):
        extractor = ConceptExtractor(min_frequency=5)
        queries = ["unique query one", "completely different two"]
        concepts = extractor.extract_from_queries(queries)
        assert len(concepts) == 0
