"""
Tests for HBLLM v2 Phase 2: Resource Intelligence.

Covers:
  - AttentionManager — memory budgets, importance scoring, focus allocation
  - LoadManager — pressure levels, degradation policies, task management
  - ConfidenceEstimator v2 — calibration tracking, domain adjustments
  - Factory wiring — Phase 2 nodes integrate into Brain
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from hbllm.brain.attention_manager import AttentionManager, FocusAllocation, MemoryBudget
from hbllm.brain.confidence_estimator import ConfidenceEstimator
from hbllm.brain.load_manager import (
    DEFAULT_POLICIES,
    DegradationPolicy,
    LoadManager,
    SystemResources,
)
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType

pytestmark = pytest.mark.asyncio


# ── AttentionManager Tests ───────────────────────────────────────────────


class TestAttentionManager:
    """Test memory budgets, importance scoring, and focus allocation."""

    @pytest.fixture
    async def bus(self):
        bus = InProcessBus()
        await bus.start()
        yield bus
        await bus.stop()

    @pytest.fixture
    async def attn(self, bus):
        node = AttentionManager(node_id="test_attn", total_context_budget=4096)
        await node.start(bus)
        yield node
        await node.stop()

    async def test_memory_budgets_initialized(self, attn: AttentionManager):
        """Default memory budgets should be created."""
        for mem_type in ["episodic", "semantic", "procedural", "value"]:
            budget = attn.get_budget(mem_type)
            assert budget is not None
            assert budget.max_items > 0

    async def test_budget_utilization(self):
        """Utilization should track correctly."""
        b = MemoryBudget(memory_type="test", max_items=100, current_items=80)
        assert b.utilization == 0.8
        assert not b.needs_pruning

        b.current_items = 95
        assert b.needs_pruning

    async def test_should_accept_under_budget(self, attn: AttentionManager):
        """Under capacity, any memory should be accepted."""
        attn.update_item_count("episodic", 100)
        assert attn.should_accept("episodic", importance=0.1)

    async def test_should_reject_near_capacity_low_importance(self, attn: AttentionManager):
        """Near capacity, low-importance items should be rejected."""
        attn.update_item_count("episodic", 490)  # 98% of 500
        assert not attn.should_accept("episodic", importance=0.1)

    async def test_importance_scoring(self, attn: AttentionManager):
        """Importance scores should combine recency, frequency, relevance."""
        high = attn.score_importance("mem:1", recency=0.9, frequency=0.8, relevance=0.9)
        low = attn.score_importance("mem:2", recency=0.1, frequency=0.1, relevance=0.1)
        assert high > low

    async def test_decay_all_scores(self, attn: AttentionManager):
        """Decay should reduce all importance scores."""
        attn.score_importance("mem:1", recency=0.9, frequency=0.9, relevance=0.9)
        before = attn.get_importance("mem:1")
        attn.decay_all_scores()
        after = attn.get_importance("mem:1")
        assert after < before

    async def test_focus_allocation_creates_domain(self, attn: AttentionManager):
        """Allocating focus should create a new domain entry."""
        alloc = attn.allocate_focus("coding", priority=0.8)
        assert isinstance(alloc, FocusAllocation)
        assert alloc.priority == 0.8
        assert alloc.context_tokens > 0

    async def test_rebalance_proportional(self, attn: AttentionManager):
        """Rebalance should distribute tokens proportionally to priority."""
        attn.allocate_focus("coding", priority=0.8)
        attn.allocate_focus("writing", priority=0.2)
        result = attn.rebalance_focus()
        assert result["coding"] > result["writing"]

    async def test_stats_structure(self, attn: AttentionManager):
        """Stats should contain all expected fields."""
        stats = attn.stats()
        assert "total_context_budget" in stats
        assert "memory_budgets" in stats
        assert "focus_allocations" in stats
        assert "tracked_memories" in stats


# ── LoadManager Tests ────────────────────────────────────────────────────


class TestLoadManager:
    """Test pressure levels, degradation, and task management."""

    @pytest.fixture
    async def bus(self):
        bus = InProcessBus()
        await bus.start()
        yield bus
        await bus.stop()

    @pytest.fixture
    async def load(self, bus):
        node = LoadManager(
            node_id="test_load",
            monitor_interval=9999.0,  # disable timer for testing
        )
        await node.start(bus)
        yield node
        await node.stop()

    async def test_system_resources_pressure_levels(self):
        """Pressure levels should classify correctly."""
        assert SystemResources(cpu_percent=30, memory_percent=40).pressure_level == "normal"
        assert SystemResources(cpu_percent=65, memory_percent=50).pressure_level == "elevated"
        assert SystemResources(cpu_percent=80, memory_percent=70).pressure_level == "high"
        assert SystemResources(cpu_percent=95, memory_percent=92).pressure_level == "critical"

    async def test_default_policy_is_normal(self, load: LoadManager):
        """Initial policy should be 'normal'."""
        assert load.current_policy.level == "normal"
        assert load.get_max_context_tokens() == 4096

    async def test_degradation_policies_complete(self):
        """All four pressure levels should have policies."""
        for level in ["normal", "elevated", "high", "critical"]:
            assert level in DEFAULT_POLICIES
            policy = DEFAULT_POLICIES[level]
            assert isinstance(policy, DegradationPolicy)

    async def test_can_accept_task(self, load: LoadManager):
        """Under normal policy, should accept up to 8 tasks."""
        assert load.can_accept_task()

    async def test_task_tracking(self, load: LoadManager, bus):
        """Task started/completed messages should track active tasks."""
        start_msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="system.task.started",
            payload={"task_id": "task_1"},
        )
        await bus.publish("system.task.started", start_msg)
        await asyncio.sleep(0.05)

        stats = load.stats()
        assert stats["active_tasks"] == 1

        complete_msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="system.task.completed",
            payload={"task_id": "task_1"},
        )
        await bus.publish("system.task.completed", complete_msg)
        await asyncio.sleep(0.05)

        stats = load.stats()
        assert stats["active_tasks"] == 0

    async def test_queue_task(self, load: LoadManager):
        """Tasks should be queueable for later processing."""
        assert load.queue_task("task_a", priority=0.5)
        assert load.queue_task("task_b", priority=0.9)
        stats = load.stats()
        assert stats["queue_depth"] == 2

    async def test_dequeue_priority_order(self, load: LoadManager):
        """Dequeue should return highest priority first."""
        load.queue_task("low", priority=0.2)
        load.queue_task("high", priority=0.9)
        load.queue_task("mid", priority=0.5)

        task = load.dequeue_task()
        assert task is not None
        assert task["task_id"] == "high"

    async def test_model_preference_by_level(self):
        """Model preference should escalate down with pressure."""
        assert DEFAULT_POLICIES["normal"].model_preference == "large"
        assert DEFAULT_POLICIES["high"].model_preference == "small"
        assert DEFAULT_POLICIES["critical"].model_preference == "tiny"

    async def test_simulation_disabled_under_high(self):
        """Simulation should be disabled under high pressure."""
        assert DEFAULT_POLICIES["normal"].enable_simulation
        assert not DEFAULT_POLICIES["high"].enable_simulation

    async def test_stats_structure(self, load: LoadManager):
        """Stats should contain all expected fields."""
        stats = load.stats()
        assert "pressure_level" in stats
        assert "resources" in stats
        assert "active_policy" in stats
        assert "queue_depth" in stats


# ── ConfidenceEstimator v2 Tests ─────────────────────────────────────────


class TestConfidenceEstimatorV2:
    """Test v2 calibration tracking and uncertainty quantification."""

    def test_record_outcome_stores_history(self):
        """Recording outcomes should build calibration history."""
        ce = ConfidenceEstimator()
        ce.record_outcome(predicted=0.8, actual=0.6, domain="coding")
        ce.record_outcome(predicted=0.9, actual=0.3, domain="coding")

        stats = ce.calibration_stats()
        assert stats["total_feedback"] == 2
        assert stats["history_size"] == 2

    def test_calibration_error_computed(self):
        """ECE should reflect average prediction errors."""
        ce = ConfidenceEstimator()
        # Perfect predictions
        ce.record_outcome(predicted=0.8, actual=0.8, domain="math")
        assert ce.calibration_error("math") == 0.0

        # Bad predictions
        ce.record_outcome(predicted=0.9, actual=0.1, domain="writing")
        assert ce.calibration_error("writing") == pytest.approx(0.8, abs=0.01)

    def test_domain_adjustment_overconfident(self):
        """Consistently overconfident predictions should increase adjustment."""
        ce = ConfidenceEstimator()
        for _ in range(20):
            ce.record_outcome(predicted=0.9, actual=0.5, domain="coding")

        # coding domain should now have a positive adjustment (overconfident bias)
        stats = ce.calibration_stats()
        assert stats["domain_adjustments"]["coding"] > 0

    def test_calibrated_score_adjusts(self):
        """Calibrated score should differ from raw after feedback."""
        ce = ConfidenceEstimator()
        query = "Explain quantum entanglement"
        response = "Quantum entanglement is a phenomenon where particles become correlated."

        raw = ce.score(query, response)

        # Simulate overconfidence feedback
        for _ in range(20):
            ce.record_outcome(predicted=0.9, actual=0.4, domain="physics")

        calibrated = ce.calibrated_score(query, response, domain="physics")
        # Calibrated should be lower than raw (corrected for overconfidence)
        assert calibrated < raw

    def test_calibrated_score_bounded(self):
        """Calibrated scores should stay within [0.05, 0.95]."""
        ce = ConfidenceEstimator()
        # Extreme over-adjustment
        for _ in range(50):
            ce.record_outcome(predicted=1.0, actual=0.0, domain="extreme")

        score = ce.calibrated_score("test", "test response", domain="extreme")
        assert 0.05 <= score <= 0.95

    def test_calibration_stats_structure(self):
        """Stats should contain all expected fields."""
        ce = ConfidenceEstimator()
        ce.record_outcome(predicted=0.7, actual=0.6, domain="test")

        stats = ce.calibration_stats()
        assert "total_predictions" in stats
        assert "total_feedback" in stats
        assert "calibration_error" in stats
        assert "domain_adjustments" in stats
        assert "domain_errors" in stats

    def test_no_history_returns_empty_stats(self):
        """Empty estimator should return clean stats."""
        ce = ConfidenceEstimator()
        stats = ce.calibration_stats()
        assert stats["total_feedback"] == 0
        assert stats["calibration_error"] == 0.0


# ── LearnerNode Micro-Learning Tests ────────────────────────────────────


class TestLearnerNodeMicroLearning:
    """Test v2 micro-learning enhancements to LearnerNode."""

    @pytest.fixture
    async def bus(self):
        bus = InProcessBus()
        await bus.start()
        yield bus
        await bus.stop()

    @pytest.fixture
    async def learner(self, bus, tmp_path):
        from hbllm.brain.learner_node import LearnerNode

        node = LearnerNode(
            node_id="test_learner",
            enable_micro_learning=True,
            micro_learn_threshold=0.3,
            distillation_threshold=0.85,
        )
        node.queue_path = str(tmp_path / "dpo_queue.json")
        await node.start(bus)
        yield node
        await node.stop()

    async def test_low_score_queued_for_micro_learning(self, learner, bus):
        """Low-scoring evaluation should queue for micro-correction."""
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="eval",
            topic="system.evaluation",
            payload={
                "overall_score": 0.15,
                "query": "What is quantum computing?",
                "response": "I don't know",
                "dimensions": {"task_success": 0.1},
            },
        )
        await bus.publish("system.evaluation", msg)
        await asyncio.sleep(0.05)

        queue = learner.get_micro_learn_queue()
        assert len(queue) == 1
        assert queue[0]["query"] == "What is quantum computing?"
        assert queue[0]["score"] == 0.15

    async def test_high_score_banked_for_distillation(self, learner, bus):
        """High-scoring evaluation should be stored in distillation bank."""
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="eval",
            topic="system.evaluation",
            payload={
                "overall_score": 0.92,
                "query": "Explain neural networks",
                "response": "Neural networks are computational models inspired by biological neurons...",
            },
        )
        await bus.publish("system.evaluation", msg)
        await asyncio.sleep(0.05)

        bank = learner.get_distillation_bank()
        assert len(bank) == 1
        assert bank[0]["query"] == "Explain neural networks"

    async def test_mid_score_ignored(self, learner, bus):
        """Mid-range scores should not trigger either path."""
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="eval",
            topic="system.evaluation",
            payload={
                "overall_score": 0.55,
                "query": "What is Python?",
                "response": "Python is a programming language.",
            },
        )
        await bus.publish("system.evaluation", msg)
        await asyncio.sleep(0.05)

        assert len(learner.get_micro_learn_queue()) == 0
        assert len(learner.get_distillation_bank()) == 0

    async def test_micro_learn_without_model(self, learner):
        """micro_learn() should return False when no model is available."""
        result = await learner.micro_learn(
            query="test",
            bad_response="wrong",
            good_response="correct",
        )
        assert result is False

    async def test_clear_distillation_bank(self, learner, bus):
        """Clearing distillation bank should return count and empty it."""
        for i in range(3):
            msg = Message(
                type=MessageType.EVENT,
                source_node_id="eval",
                topic="system.evaluation",
                payload={
                    "overall_score": 0.95,
                    "query": f"Question {i}",
                    "response": f"Excellent answer {i} " * 20,
                },
            )
            await bus.publish("system.evaluation", msg)

        await asyncio.sleep(0.05)
        assert len(learner.get_distillation_bank()) == 3

        count = learner.clear_distillation_bank()
        assert count == 3
        assert len(learner.get_distillation_bank()) == 0

    async def test_micro_learning_stats(self, learner, bus):
        """Stats should report all micro-learning metrics."""
        # Send one low-score event
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="eval",
            topic="system.evaluation",
            payload={
                "overall_score": 0.1,
                "query": "Test question",
                "response": "Bad answer",
            },
        )
        await bus.publish("system.evaluation", msg)
        await asyncio.sleep(0.05)

        stats = learner.micro_learning_stats()
        assert stats["enabled"] is True
        assert stats["micro_queue_depth"] == 1
        assert stats["thresholds"]["micro_learn"] == 0.3
        assert stats["thresholds"]["distillation"] == 0.85

    async def test_disable_micro_learning(self, bus, tmp_path):
        """With micro_learning disabled, evaluation events should be ignored."""
        from hbllm.brain.learner_node import LearnerNode

        node = LearnerNode(
            node_id="disabled_learner",
            enable_micro_learning=False,
        )
        node.queue_path = str(tmp_path / "dpo_queue2.json")
        await node.start(bus)

        msg = Message(
            type=MessageType.EVENT,
            source_node_id="eval",
            topic="system.evaluation",
            payload={
                "overall_score": 0.05,
                "query": "Test",
                "response": "Bad",
            },
        )
        await bus.publish("system.evaluation", msg)
        await asyncio.sleep(0.05)

        # With micro-learning disabled, the handler is never subscribed
        assert len(node.get_micro_learn_queue()) == 0
        await node.stop()


# ── Factory Integration Tests ────────────────────────────────────────────


class TestPhase2FactoryIntegration:
    """Verify Phase 2 nodes are wired into Brain via factory."""

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
            inject_attention=True,
            inject_load_manager=True,
            data_dir=str(tmp_path),
        )
        brain = await BrainFactory.create(provider=_Mock(), config=config)
        yield brain
        await brain.shutdown()

    async def test_attention_manager_wired(self, brain):
        assert brain.attention_manager is not None
        assert isinstance(brain.attention_manager, AttentionManager)

    async def test_load_manager_wired(self, brain):
        assert brain.load_manager is not None
        assert isinstance(brain.load_manager, LoadManager)

    async def test_phase2_nodes_in_node_list(self, brain):
        node_ids = [n.node_id for n in brain.nodes]
        assert "attention" in node_ids
        assert "load_manager" in node_ids

    async def test_disable_phase2_nodes(self, tmp_path):
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
            inject_attention=False,
            inject_load_manager=False,
            data_dir=str(tmp_path),
        )
        brain = await BrainFactory.create(provider=_Mock(), config=config)
        assert brain.attention_manager is None
        assert brain.load_manager is None
        await brain.shutdown()
