"""Tests for EvaluationNode — intelligence feedback loop and micro-learning trigger."""

import asyncio

import pytest

from hbllm.brain.evaluation_node import EvaluationNode, EvaluationReport
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType

# ── Unit Tests: Scoring Engine ────────────────────────────────────────────────


class TestEvaluationScoring:
    """Test the 5-dimension scoring engine."""

    def setup_method(self):
        self.node = EvaluationNode(node_id="eval_test")

    def test_task_success_long_response(self):
        """Long, confident responses should score well."""
        ctx = {"content": " ".join(["word"] * 60), "confidence": 0.9, "intent": "answer"}
        score = self.node._score_task_success(ctx)
        assert score > 0.7

    def test_task_success_short_response(self):
        """Very short responses should score poorly."""
        ctx = {"content": "yes", "confidence": 0.5, "intent": "answer"}
        score = self.node._score_task_success(ctx)
        assert score < 0.5

    def test_task_success_code_with_blocks(self):
        """Code intent with code blocks should get a bonus."""
        ctx = {
            "content": "Here's the code:\n```python\nprint('hello')\n```",
            "intent": "code",
            "confidence": 0.7,
        }
        score = self.node._score_task_success(ctx)
        assert score > 0.6

    def test_plan_validity_well_scoped(self):
        """Plans with 2-5 clear steps should score well."""
        ctx = {"plan_steps": ["Step 1: Analyze data", "Step 2: Build model", "Step 3: Evaluate"]}
        score = self.node._score_plan_validity(ctx)
        assert score > 0.7

    def test_plan_validity_over_complex(self):
        """Plans with >10 steps should be penalized."""
        ctx = {"plan_steps": [f"Step {i}" for i in range(12)]}
        score = self.node._score_plan_validity(ctx)
        assert score <= 0.7

    def test_plan_validity_no_plan(self):
        """No plan is acceptable (not everything needs a plan)."""
        ctx = {"plan_steps": []}
        score = self.node._score_plan_validity(ctx)
        assert score == 0.6

    def test_tool_accuracy_all_success(self):
        """All successful tool calls should score 1.0."""
        ctx = {
            "tools_used": [{"name": "search", "success": True}, {"name": "calc", "success": True}]
        }
        score = self.node._score_tool_accuracy(ctx)
        assert score == 1.0

    def test_tool_accuracy_mixed(self):
        """Mixed results should be reflected."""
        ctx = {
            "tools_used": [{"name": "search", "success": True}, {"name": "calc", "success": False}]
        }
        score = self.node._score_tool_accuracy(ctx)
        assert score == 0.5

    def test_tool_accuracy_no_tools(self):
        """No tools used should return a reasonable default."""
        ctx = {"tools_used": []}
        score = self.node._score_tool_accuracy(ctx)
        assert score == 0.8

    def test_memory_usage_high_hits(self):
        """High memory hits should score well."""
        ctx = {"memory_hits": 5}
        score = self.node._score_memory_usage(ctx)
        assert score >= 0.9

    def test_confidence_error_well_calibrated(self):
        """Confident response without hedging = well calibrated."""
        ctx = {"confidence": 0.8, "content": "The answer is 42."}
        score = self.node._score_confidence_error(ctx)
        assert score < 0.2

    def test_confidence_error_miscalibrated(self):
        """High confidence with hedge words = miscalibrated."""
        ctx = {"confidence": 0.9, "content": "Maybe perhaps I think it might possibly be..."}
        score = self.node._score_confidence_error(ctx)
        assert score > 0.4


class TestEvaluationReport:
    """Test the EvaluationReport dataclass."""

    def test_to_dict(self):
        report = EvaluationReport(
            correlation_id="test-123",
            timestamp=1000.0,
            task_success=0.85,
            plan_validity=0.7,
            tool_accuracy=0.9,
            memory_usage=0.6,
            confidence_error=0.15,
            overall_score=0.78,
            flags=["low_task_success"],
        )
        d = report.to_dict()
        assert d["correlation_id"] == "test-123"
        assert d["overall_score"] == 0.78
        assert "low_task_success" in d["flags"]

    def test_composite_scoring(self):
        """Verify the composite weighted scoring formula."""
        node = EvaluationNode(node_id="eval_composite")
        ctx = {
            "content": " ".join(["word"] * 30),
            "confidence": 0.7,
            "intent": "answer",
            "plan_steps": ["Step 1", "Step 2"],
            "tools_used": [],
            "memory_hits": 3,
        }
        report = node._evaluate("test-456", ctx)

        # Overall score should be between 0 and 1
        assert 0.0 <= report.overall_score <= 1.0
        # A reasonable response should score above 0.5
        assert report.overall_score > 0.5


# ── Integration Tests: Bus Interaction ────────────────────────────────────────


@pytest.fixture
async def eval_env():
    bus = InProcessBus()
    await bus.start()

    node = EvaluationNode(node_id="eval_integration")
    await node.start(bus)

    yield bus, node

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_experience_triggers_evaluation(eval_env):
    """system.experience messages should trigger evaluation and publish results."""
    bus, node = eval_env

    evaluations = []
    await bus.subscribe("system.evaluation", lambda msg: evaluations.append(msg))

    # Simulate DecisionNode output
    experience_msg = Message(
        type=MessageType.EVENT,
        source_node_id="decision_node",
        topic="system.experience",
        payload={
            "text": "The capital of France is Paris, a beautiful city known for the Eiffel Tower.",
            "intent": "answer",
            "confidence": 0.9,
            "tools_used": [],
            "memory_hits": 2,
            "plan_steps": [],
        },
    )
    await bus.publish("system.experience", experience_msg)
    await asyncio.sleep(0.3)

    assert node._total_evaluated == 1
    assert len(evaluations) >= 1


@pytest.mark.asyncio
async def test_negative_feedback_triggers_micro_learn(eval_env):
    """Negative feedback should publish system.micro_learn event."""
    bus, node = eval_env

    micro_learn_events = []
    await bus.subscribe("system.micro_learn", lambda msg: micro_learn_events.append(msg))

    # First seed a pending context
    corr_id = "test-feedback-123"
    node._pending_contexts[corr_id] = {
        "content": "What is quantum computing?",
        "output": "I don't know",
        "confidence": 0.8,
        "timestamp": 1000.0,
    }

    # Send negative feedback WITH a correction
    feedback_msg = Message(
        type=MessageType.EVENT,
        source_node_id="user",
        topic="system.feedback",
        payload={
            "rating": -1,
            "correction": "Quantum computing uses qubits that can be in superposition.",
        },
        correlation_id=corr_id,
    )
    await bus.publish("system.feedback", feedback_msg)
    await asyncio.sleep(0.3)

    assert len(micro_learn_events) == 1
    payload = micro_learn_events[0].payload
    assert payload["query"] == "What is quantum computing?"
    assert payload["correction"] == "Quantum computing uses qubits that can be in superposition."


@pytest.mark.asyncio
async def test_stats_aggregation(eval_env):
    """Stats should aggregate evaluation history correctly."""
    _bus, node = eval_env

    # Manually add some evaluations
    for i in range(5):
        report = EvaluationReport(
            correlation_id=f"test-{i}",
            timestamp=1000.0 + i,
            task_success=0.8,
            plan_validity=0.7,
            tool_accuracy=0.9,
            memory_usage=0.6,
            confidence_error=0.1,
            overall_score=0.75,
        )
        node._evaluations.append(report)
        node._total_evaluated += 1

    stats = node.stats()
    assert stats["total_evaluated"] == 5
    assert stats["avg_overall_score"] == 0.75
    assert stats["window_size"] == 5
