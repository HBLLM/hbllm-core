"""
Tests for HBLLM Intelligence Feedback Loop.

Covers:
  - EvaluationNode — per-interaction scoring
  - SkillCompilerNode — pattern detection & skill extraction
  - ReflectionNode — periodic batch analysis
  - Factory wiring — all three nodes integrate into Brain
  - End-to-end feedback loop flow
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from hbllm.brain.cognitive_metrics import CognitiveMetrics
from hbllm.brain.evaluation_node import EvaluationNode
from hbllm.brain.goal_manager import GoalManager
from hbllm.brain.reflection_node import ReflectionInsight, ReflectionNode
from hbllm.brain.self_model import SelfModel
from hbllm.brain.skill_compiler_node import ActionPattern, SkillCompilerNode
from hbllm.brain.skill_registry import SkillRegistry
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType

pytestmark = pytest.mark.asyncio


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_experience_msg(
    text: str = "This is a well-formed response",
    intent: str = "answer",
    thought_type: str = "intuition",
    confidence: float = 0.7,
    correlation_id: str = "test_001",
) -> Message:
    return Message(
        type=MessageType.EVENT,
        source_node_id="decision",
        topic="system.experience",
        payload={
            "text": text,
            "intent": intent,
            "thought_type": thought_type,
            "confidence": confidence,
            "tools_used": [],
            "memory_hits": 2,
            "plan_steps": ["step1", "step2"],
        },
        correlation_id=correlation_id,
    )


def _make_evaluation_msg(
    overall_score: float = 0.75,
    flags: list[str] | None = None,
) -> Message:
    return Message(
        type=MessageType.EVENT,
        source_node_id="evaluation",
        topic="system.evaluation",
        payload={
            "correlation_id": "test_001",
            "timestamp": 1000.0,
            "task_success": overall_score,
            "plan_validity": 0.7,
            "tool_accuracy": 0.8,
            "memory_usage": 0.6,
            "confidence_error": 0.1,
            "overall_score": overall_score,
            "flags": flags or [],
        },
    )


def _make_reflection_msg(rules: list[dict] | None = None) -> Message:
    return Message(
        type=MessageType.EVENT,
        source_node_id="experience",
        topic="system.reflection",
        payload={
            "category": "coding",
            "rules": rules
            or [
                {
                    "condition": "user asks for Python code",
                    "action": "generate with type hints",
                    "confidence": 0.8,
                }
            ],
            "entities": ["Python", "typing"],
        },
    )


# ── EvaluationNode Tests ────────────────────────────────────────────────


class TestEvaluationNode:
    """Test EvaluationNode scoring and feedback loop."""

    @pytest.fixture
    async def bus(self):
        bus = InProcessBus()
        await bus.start()
        yield bus
        await bus.stop()

    @pytest.fixture
    async def eval_node(self, bus, tmp_path):
        metrics = CognitiveMetrics(data_dir=str(tmp_path))
        goal_mgr = GoalManager(data_dir=str(tmp_path))
        self_model = SelfModel(data_dir=str(tmp_path))

        node = EvaluationNode(
            node_id="test_eval",
            cognitive_metrics=metrics,
            goal_manager=goal_mgr,
            self_model=self_model,
            goal_trigger_interval=0.1,  # fast trigger for testing
        )
        await node.start(bus)
        yield node
        await node.stop()

    async def test_evaluation_scores_experience(self, eval_node: EvaluationNode, bus):
        """Experience message should produce an evaluation report."""
        msg = _make_experience_msg()
        await bus.publish("system.experience", msg)
        await asyncio.sleep(0.1)

        stats = eval_node.stats()
        assert stats["total_evaluated"] >= 1
        assert stats["avg_overall_score"] > 0

    async def test_task_success_scoring(self, eval_node: EvaluationNode):
        """Long, intent-aligned content should score higher than empty."""
        good = {
            "content": "Here is a detailed response with code ```python\\nprint()\\n``` and explanation.",
            "intent": "code",
            "confidence": 0.8,
        }
        bad = {"content": "Hmm", "intent": "answer", "confidence": 0.2}

        good_score = EvaluationNode._score_task_success(good)
        bad_score = EvaluationNode._score_task_success(bad)
        assert good_score > bad_score

    async def test_plan_validity_scoring(self, eval_node: EvaluationNode):
        """Well-formed plans should score higher."""
        good = {"plan_steps": ["Define schema", "Implement logic", "Write tests"]}
        empty = {"plan_steps": []}

        good_score = EvaluationNode._score_plan_validity(good)
        empty_score = EvaluationNode._score_plan_validity(empty)
        assert good_score >= empty_score

    async def test_confidence_error_detects_miscalibration(self, eval_node: EvaluationNode):
        """High confidence + hedging language = high error."""
        miscalibrated = {
            "confidence": 0.95,
            "content": "Maybe I think this is perhaps the answer, possibly correct.",
        }
        calibrated = {"confidence": 0.7, "content": "The answer is 42 based on the calculation."}

        misc_error = EvaluationNode._score_confidence_error(miscalibrated)
        cal_error = EvaluationNode._score_confidence_error(calibrated)
        assert misc_error > cal_error

    async def test_stats_structure(self, eval_node: EvaluationNode, bus):
        """Stats should contain all expected fields."""
        await bus.publish("system.experience", _make_experience_msg())
        await asyncio.sleep(0.1)

        stats = eval_node.stats()
        assert "total_evaluated" in stats
        assert "total_flagged" in stats
        assert "avg_overall_score" in stats
        assert "avg_task_success" in stats
        assert "flag_rate" in stats

    async def test_feedback_loop_triggers_goals(self, eval_node: EvaluationNode, bus):
        """After enough evaluations, GoalManager should receive triggers."""
        # Send multiple experiences to trigger goal generation
        for i in range(15):
            msg = _make_experience_msg(correlation_id=f"batch_{i}")
            await bus.publish("system.experience", msg)
            await asyncio.sleep(0.01)

        # Wait for goal trigger
        await asyncio.sleep(0.3)

        stats = eval_node.stats()
        assert stats["total_evaluated"] >= 10

    async def test_evaluation_publishes_event(self, bus, tmp_path):
        """EvaluationNode should publish system.evaluation messages."""
        received: list[Message] = []

        async def capture(msg: Message) -> None:
            received.append(msg)

        await bus.subscribe("system.evaluation", capture)

        node = EvaluationNode(node_id="test_pub")
        await node.start(bus)

        await bus.publish("system.experience", _make_experience_msg())
        await asyncio.sleep(0.1)

        await node.stop()
        assert len(received) >= 1
        assert "overall_score" in received[0].payload


# ── SkillCompilerNode Tests ──────────────────────────────────────────────


class TestSkillCompilerNode:
    """Test SkillCompilerNode pattern detection and compilation."""

    @pytest.fixture
    async def bus(self):
        bus = InProcessBus()
        await bus.start()
        yield bus
        await bus.stop()

    @pytest.fixture
    async def compiler(self, bus, tmp_path):
        registry = SkillRegistry(data_dir=str(tmp_path))
        node = SkillCompilerNode(
            node_id="test_compiler",
            skill_registry=registry,
            min_occurrences=3,
            min_success_rate=0.5,
        )
        await node.start(bus)
        yield node
        await node.stop()

    async def test_records_experience_actions(self, compiler: SkillCompilerNode, bus):
        """Experience events should be recorded in action history."""
        await bus.publish("system.experience", _make_experience_msg())
        await asyncio.sleep(0.1)

        stats = compiler.stats()
        assert stats["action_history_size"] >= 1

    async def test_detects_patterns_from_repeated_actions(self, compiler: SkillCompilerNode, bus):
        """Repeated action sequences should be detected as patterns."""
        for i in range(10):
            msg = _make_experience_msg(
                text=f"Python code response {i}",
                intent="code",
                thought_type="analytical",
                correlation_id=f"pattern_{i}",
            )
            await bus.publish("system.experience", msg)
            await asyncio.sleep(0.02)

        stats = compiler.stats()
        assert stats["patterns_detected"] >= 1

    async def test_compiles_skill_from_reflection_rules(self, compiler: SkillCompilerNode, bus):
        """High-confidence reflection rules should produce skills."""
        # Send enough rule occurrences to trigger compilation
        for i in range(5):
            msg = _make_reflection_msg(
                rules=[
                    {
                        "condition": "user asks for sorting",
                        "action": "use quicksort with pivot",
                        "confidence": 0.9,
                    }
                ]
            )
            await bus.publish("system.reflection", msg)
            await asyncio.sleep(0.02)

        stats = compiler.stats()
        assert stats["patterns_detected"] >= 1

    async def test_action_pattern_success_rate(self):
        """ActionPattern success_rate calculation should be correct."""
        pattern = ActionPattern(
            pattern_hash="abc",
            actions=["a", "b"],
            tools=[],
            category="test",
            occurrences=10,
            successes=7,
        )
        assert pattern.success_rate == 0.7

    async def test_stats_structure(self, compiler: SkillCompilerNode):
        """Stats should contain all expected fields."""
        stats = compiler.stats()
        assert "patterns_detected" in stats
        assert "skills_compiled" in stats
        assert "active_patterns" in stats
        assert "action_history_size" in stats
        assert "top_patterns" in stats


# ── ReflectionNode Tests ─────────────────────────────────────────────────


class TestReflectionNode:
    """Test ReflectionNode periodic analysis and insight generation."""

    @pytest.fixture
    async def bus(self):
        bus = InProcessBus()
        await bus.start()
        yield bus
        await bus.stop()

    @pytest.fixture
    async def reflection(self, bus, tmp_path):
        metrics = CognitiveMetrics(data_dir=str(tmp_path))
        goal_mgr = GoalManager(data_dir=str(tmp_path))
        self_model = SelfModel(data_dir=str(tmp_path))

        node = ReflectionNode(
            node_id="test_reflection",
            cognitive_metrics=metrics,
            goal_manager=goal_mgr,
            self_model=self_model,
            reflection_interval=9999.0,  # disable timer for testing
            min_evaluations=5,
        )
        await node.start(bus)
        yield node
        await node.stop()

    async def test_accumulates_evaluations(self, reflection: ReflectionNode, bus):
        """Evaluation events should be accumulated."""
        for i in range(10):
            await bus.publish("system.evaluation", _make_evaluation_msg(0.7))
            await asyncio.sleep(0.01)

        stats = reflection.stats()
        assert stats["evaluation_history_size"] >= 10

    async def test_manual_trigger_runs_reflection(self, reflection: ReflectionNode, bus):
        """Manual trigger should run a reflection session."""
        # Feed evaluations first
        for i in range(10):
            await bus.publish("system.evaluation", _make_evaluation_msg(0.6))
            await asyncio.sleep(0.01)

        # Trigger reflection
        trigger = Message(
            type=MessageType.QUERY,
            source_node_id="test",
            topic="reflection.trigger",
            payload={"deep": False},
        )
        await bus.publish("reflection.trigger", trigger)
        await asyncio.sleep(0.2)

        stats = reflection.stats()
        assert stats["total_sessions"] >= 1

    async def test_detects_performance_decline(self, reflection: ReflectionNode):
        """Declining scores should produce warning insights."""
        # Simulate declining evaluations
        evals = []
        for i in range(20):
            score = 0.8 - (i * 0.03)  # declining
            evals.append(
                {
                    "task_success": score,
                    "plan_validity": 0.7,
                    "tool_accuracy": 0.7,
                    "memory_usage": 0.6,
                }
            )

        insights = reflection._analyze_performance_trends(evals)
        # Should detect declining task_success
        warnings = [i for i in insights if i.severity == "warning"]
        assert len(warnings) >= 1

    async def test_detects_failure_patterns(self, reflection: ReflectionNode):
        """Recurring flags should produce failure pattern insights."""
        evals = [{"flags": ["low_task_success"], "overall_score": 0.3} for _ in range(10)]

        insights = reflection._detect_failure_patterns(evals)
        assert len(insights) >= 1
        assert insights[0].category == "failure_pattern"

    async def test_deep_analysis_detects_low_performance(self, reflection: ReflectionNode):
        """Deep analysis should flag critically low overall scores."""
        evals = [{"overall_score": 0.3, "confidence_error": 0.5} for _ in range(20)]

        insights = reflection._deep_strategy_analysis(evals)
        critical = [i for i in insights if i.severity == "critical"]
        assert len(critical) >= 1

    async def test_insight_to_dict(self):
        """ReflectionInsight serialization should work."""
        insight = ReflectionInsight(
            category="performance",
            severity="warning",
            description="Test insight",
            evidence={"metric": "test"},
            recommended_actions=["Fix it"],
        )
        d = insight.to_dict()
        assert d["category"] == "performance"
        assert d["severity"] == "warning"
        assert "Fix it" in d["recommended_actions"]

    async def test_reflection_publishes_session(self, bus, tmp_path):
        """Reflection session should be published on the bus."""
        received: list[Message] = []

        async def capture(msg: Message) -> None:
            received.append(msg)

        await bus.subscribe("system.reflection.session", capture)

        metrics = CognitiveMetrics(data_dir=str(tmp_path))
        node = ReflectionNode(
            node_id="test_pub",
            cognitive_metrics=metrics,
            reflection_interval=9999.0,
            min_evaluations=3,
        )
        await node.start(bus)

        # Feed evaluations
        for _ in range(5):
            await bus.publish("system.evaluation", _make_evaluation_msg(0.7))
            await asyncio.sleep(0.01)

        # Trigger
        trigger = Message(
            type=MessageType.QUERY,
            source_node_id="test",
            topic="reflection.trigger",
            payload={},
        )
        await bus.publish("reflection.trigger", trigger)
        await asyncio.sleep(0.2)

        await node.stop()
        assert len(received) >= 1
        assert "insight_count" in received[0].payload


# ── Factory Integration Tests ────────────────────────────────────────────


class TestV2FactoryIntegration:
    """Verify v2 nodes are wired into Brain via factory."""

    @pytest.fixture
    async def brain(self, tmp_path):
        from hbllm.brain.factory import BrainConfig, BrainFactory
        from hbllm.serving.provider import LLMProvider, LLMResponse

        class _Mock(LLMProvider):
            @property
            def name(self) -> str:
                return "mock"

            async def generate(
                self,
                messages: list[dict[str, str]],
                max_tokens: int = 1000,
                temperature: float = 0.7,
                top_p: float = 0.95,
                **kwargs: Any,
            ) -> LLMResponse:
                return LLMResponse(
                    content="Mock",
                    model="mock",
                    usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                )

            async def stream(
                self,
                messages: list[dict[str, str]],
                max_tokens: int = 1000,
                temperature: float = 0.7,
                **kwargs: Any,
            ) -> AsyncIterator[str]:
                yield "Mock"

        config = BrainConfig(
            inject_perception=False,
            inject_evaluation=True,
            inject_reflection=True,
            inject_skill_compiler=True,
            data_dir=str(tmp_path),
        )
        brain = await BrainFactory.create(provider=_Mock(), config=config)
        yield brain
        await brain.shutdown()

    async def test_evaluation_node_wired(self, brain):
        assert brain.evaluation_node is not None
        assert isinstance(brain.evaluation_node, EvaluationNode)

    async def test_reflection_node_wired(self, brain):
        assert brain.reflection_node is not None
        assert isinstance(brain.reflection_node, ReflectionNode)

    async def test_skill_compiler_wired(self, brain):
        assert brain.skill_compiler_node is not None
        assert isinstance(brain.skill_compiler_node, SkillCompilerNode)

    async def test_v2_nodes_in_node_list(self, brain):
        """v2 nodes should appear in the nodes list."""
        node_ids = [n.node_id for n in brain.nodes]
        assert "evaluation" in node_ids
        assert "reflection" in node_ids
        assert "skill_compiler" in node_ids

    async def test_disable_v2_nodes(self, tmp_path):
        """Disabling v2 flags should skip wiring."""
        from hbllm.brain.factory import BrainConfig, BrainFactory
        from hbllm.serving.provider import LLMProvider, LLMResponse

        class _Mock(LLMProvider):
            @property
            def name(self) -> str:
                return "mock"

            async def generate(
                self,
                messages: list[dict[str, str]],
                max_tokens: int = 1000,
                temperature: float = 0.7,
                top_p: float = 0.95,
                **kwargs: Any,
            ) -> LLMResponse:
                return LLMResponse(
                    content="Mock",
                    model="mock",
                    usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                )

            async def stream(
                self,
                messages: list[dict[str, str]],
                max_tokens: int = 1000,
                temperature: float = 0.7,
                **kwargs: Any,
            ) -> AsyncIterator[str]:
                yield "Mock"

        config = BrainConfig(
            inject_perception=False,
            inject_evaluation=False,
            inject_reflection=False,
            inject_skill_compiler=False,
            data_dir=str(tmp_path),
        )
        brain = await BrainFactory.create(provider=_Mock(), config=config)
        assert brain.evaluation_node is None
        assert brain.reflection_node is None
        assert brain.skill_compiler_node is None
        await brain.shutdown()


# ── Benchmark Task Dataset Tests ─────────────────────────────────────────


class TestBenchmarkTaskDatasets:
    """Verify benchmark task JSON files are valid and loadable."""

    @pytest.fixture
    def tasks_dir(self):
        from pathlib import Path

        return Path(__file__).parent.parent / "hbllm" / "benchmarks" / "tasks"

    def test_reasoning_tasks_loadable(self, tasks_dir):
        import json

        path = tasks_dir / "reasoning.json"
        assert path.exists(), f"Missing {path}"
        tasks = json.loads(path.read_text())
        assert len(tasks) >= 5
        for t in tasks:
            assert "task_id" in t
            assert "prompt" in t
            assert "category" in t

    def test_coding_tasks_loadable(self, tasks_dir):
        import json

        path = tasks_dir / "coding.json"
        assert path.exists()
        tasks = json.loads(path.read_text())
        assert len(tasks) >= 5

    def test_planning_tasks_loadable(self, tasks_dir):
        import json

        path = tasks_dir / "planning.json"
        assert path.exists()
        tasks = json.loads(path.read_text())
        assert len(tasks) >= 5

    def test_memory_tasks_loadable(self, tasks_dir):
        import json

        path = tasks_dir / "memory.json"
        assert path.exists()
        tasks = json.loads(path.read_text())
        assert len(tasks) >= 5

    def test_all_tasks_have_grading_rubric(self, tasks_dir):
        import json

        for fname in ["reasoning.json", "coding.json", "planning.json", "memory.json"]:
            tasks = json.loads((tasks_dir / fname).read_text())
            for t in tasks:
                assert "grading_rubric" in t, f"Missing rubric in {fname}/{t['task_id']}"
