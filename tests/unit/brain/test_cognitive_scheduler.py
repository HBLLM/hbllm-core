"""Tests for CognitivePriorityScheduler — the prefrontal attention system."""

from __future__ import annotations

import time

import pytest

from hbllm.brain.attention_manager import (
    AttentionManager,
    CognitivePriorityScheduler,
    CognitiveTask,
    CognitiveTaskType,
)
from hbllm.network.messages import Message, MessageType


class TestCognitiveTask:
    """Test CognitiveTask priority computation."""

    def test_weighted_sum_formula(self):
        """Priority uses weighted sum, not multiplication."""
        task = CognitiveTask(
            uncertainty=1.0,
            goal_relevance=1.0,
            contradiction_severity=1.0,
            novelty=1.0,
            expected_value=1.0,
        )
        score = task.compute_priority()
        # Sum of weights = 0.30 + 0.25 + 0.20 + 0.15 + 0.10 = 1.0
        # At time 0, age_boost ~= 1.0, so score ~= 1.0
        assert 0.95 <= score <= 1.05

    def test_zero_factor_doesnt_kill_priority(self):
        """A zero in any single factor should NOT make priority zero."""
        task = CognitiveTask(
            uncertainty=0.0,  # zero!
            goal_relevance=0.8,
            contradiction_severity=0.0,
            novelty=0.6,
            expected_value=0.5,
        )
        score = task.compute_priority()
        assert score > 0, "Zero factor must not kill priority (weighted sum, not product)"

    def test_age_boost_increases_priority(self):
        """Older tasks should have higher priority (anti-starvation)."""
        task = CognitiveTask(
            uncertainty=0.5,
            goal_relevance=0.5,
        )
        # Pretend task was created 30 minutes ago
        task.created_at = time.time() - 1800

        score_old = task.compute_priority()

        task_new = CognitiveTask(
            uncertainty=0.5,
            goal_relevance=0.5,
        )
        score_new = task_new.compute_priority()

        assert score_old > score_new, "Older tasks should have higher priority"

    def test_contradiction_severity_weight(self):
        """Contradictions with higher severity should rank higher."""
        high = CognitiveTask(
            task_type=CognitiveTaskType.CONTRADICTION,
            contradiction_severity=0.9,
        )
        low = CognitiveTask(
            task_type=CognitiveTaskType.CONTRADICTION,
            contradiction_severity=0.1,
        )
        assert high.compute_priority() > low.compute_priority()

    def test_to_dict(self):
        task = CognitiveTask(
            task_type=CognitiveTaskType.CURIOSITY,
            domain="security",
            description="Why do SQL injections work?",
        )
        task.compute_priority()
        d = task.to_dict()
        assert d["task_type"] == "curiosity"
        assert d["domain"] == "security"
        assert "priority_score" in d
        assert "age_s" in d


class TestCognitivePriorityScheduler:
    """Test the scheduler's task management."""

    def test_submit_and_retrieve(self):
        scheduler = CognitivePriorityScheduler()
        task = CognitiveTask(
            task_type=CognitiveTaskType.LEARNING,
            domain="python",
            uncertainty=0.8,
        )
        scheduler.submit(task)

        best = scheduler.next_task()
        assert best is not None
        assert best.task_id == task.task_id

    def test_highest_priority_wins(self):
        scheduler = CognitivePriorityScheduler()

        low = CognitiveTask(
            task_type=CognitiveTaskType.MAINTENANCE,
            uncertainty=0.1,
            goal_relevance=0.1,
        )
        high = CognitiveTask(
            task_type=CognitiveTaskType.CONTRADICTION,
            uncertainty=0.9,
            contradiction_severity=0.9,
            goal_relevance=0.8,
        )
        scheduler.submit(low)
        scheduler.submit(high)

        best = scheduler.next_task()
        assert best.task_id == high.task_id

    def test_filter_by_type(self):
        scheduler = CognitivePriorityScheduler()

        scheduler.submit(CognitiveTask(task_type=CognitiveTaskType.LEARNING))
        scheduler.submit(CognitiveTask(task_type=CognitiveTaskType.CURIOSITY))

        curiosity = scheduler.next_task(task_type=CognitiveTaskType.CURIOSITY)
        assert curiosity is not None
        assert curiosity.task_type == CognitiveTaskType.CURIOSITY

    def test_claiming_prevents_double_pick(self):
        scheduler = CognitivePriorityScheduler()
        scheduler.submit(CognitiveTask(task_type=CognitiveTaskType.LEARNING))

        first = scheduler.next_task(claimer="autonomous_learner")
        assert first is not None
        assert first.claimed_by == "autonomous_learner"

        # Second call should return None (task is claimed)
        second = scheduler.next_task()
        assert second is None

    def test_complete_task(self):
        scheduler = CognitivePriorityScheduler()
        task = scheduler.submit(CognitiveTask(task_type=CognitiveTaskType.LEARNING))

        scheduler.complete_task(task.task_id)
        assert scheduler.stats()["tasks_completed"] == 1

        # Completed task should not appear
        best = scheduler.next_task()
        assert best is None

    def test_get_pending(self):
        scheduler = CognitivePriorityScheduler()
        for i in range(5):
            scheduler.submit(CognitiveTask(
                task_type=CognitiveTaskType.LEARNING,
                uncertainty=i * 0.2,
            ))

        pending = scheduler.get_pending(limit=3)
        assert len(pending) == 3
        # Should be sorted by priority descending
        assert pending[0].priority_score >= pending[1].priority_score

    def test_prune_on_overflow(self):
        scheduler = CognitivePriorityScheduler(max_pending=5)
        for i in range(10):
            scheduler.submit(CognitiveTask(
                task_type=CognitiveTaskType.LEARNING,
                uncertainty=i * 0.1,
            ))

        assert len(scheduler._tasks) <= 5

    def test_stats(self):
        scheduler = CognitivePriorityScheduler()
        scheduler.submit(CognitiveTask(task_type=CognitiveTaskType.LEARNING))
        scheduler.submit(CognitiveTask(task_type=CognitiveTaskType.CURIOSITY))

        stats = scheduler.stats()
        assert stats["tasks_created"] == 2
        assert stats["pending"] == 2
        assert stats["claimed"] == 0

    def test_empty_scheduler_returns_none(self):
        scheduler = CognitivePriorityScheduler()
        assert scheduler.next_task() is None
        assert scheduler.get_pending() == []


class TestAttentionManagerIngestion:
    """Test cognitive task ingestion from learning events."""

    @pytest.mark.asyncio
    async def test_ingest_contradiction(self):
        manager = AttentionManager(node_id="test_attention")
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="autonomous_learner",
            topic="learning.contradiction.discovered",
            payload={
                "claim_a": "X causes Y",
                "claim_b": "X does not cause Y",
                "concept": "causality",
                "severity": 0.8,
            },
        )

        await manager._ingest_contradiction(msg)

        pending = manager.scheduler.get_pending()
        assert len(pending) == 1
        assert pending[0].task_type == CognitiveTaskType.CONTRADICTION
        assert pending[0].contradiction_severity == 0.8
        assert pending[0].domain == "causality"

    @pytest.mark.asyncio
    async def test_ingest_weak_area(self):
        manager = AttentionManager(node_id="test_attention")
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="autonomous_learner",
            topic="learning.weak_area",
            payload={
                "concept": "buffer overflows",
                "score": 0.2,
                "goal_topic": "security",
            },
        )

        await manager._ingest_weak_area(msg)

        pending = manager.scheduler.get_pending()
        assert len(pending) == 1
        assert pending[0].task_type == CognitiveTaskType.LEARNING
        assert pending[0].uncertainty == 0.8  # 1.0 - 0.2

    @pytest.mark.asyncio
    async def test_ingest_session_skips_low_model_count(self):
        """Sessions with < 2 models don't trigger concept formation."""
        manager = AttentionManager(node_id="test_attention")
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="autonomous_learner",
            topic="learning.session.complete",
            payload={
                "topic": "test",
                "causal_models_built": 1,
            },
        )

        await manager._ingest_session_complete(msg)
        assert len(manager.scheduler.get_pending()) == 0

    @pytest.mark.asyncio
    async def test_ingest_session_creates_concept_task(self):
        """Sessions with >= 2 models trigger concept formation."""
        manager = AttentionManager(node_id="test_attention")
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="autonomous_learner",
            topic="learning.session.complete",
            payload={
                "topic": "injection attacks",
                "causal_models_built": 3,
            },
        )

        await manager._ingest_session_complete(msg)

        pending = manager.scheduler.get_pending()
        assert len(pending) == 1
        assert pending[0].task_type == CognitiveTaskType.CONCEPT_FORMATION

    @pytest.mark.asyncio
    async def test_ingest_curiosity(self):
        manager = AttentionManager(node_id="test_attention")
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="learning_event_handler",
            topic="curiosity.investigate",
            payload={
                "question": "Why does XSS bypass CSP?",
                "domain": "security",
                "priority": "high",
            },
        )

        await manager._ingest_curiosity(msg)

        pending = manager.scheduler.get_pending()
        assert len(pending) == 1
        assert pending[0].task_type == CognitiveTaskType.CURIOSITY
        assert pending[0].goal_relevance == 0.8  # "high" mapped to 0.8

    @pytest.mark.asyncio
    async def test_stats_include_scheduler(self):
        manager = AttentionManager(node_id="test_attention")
        stats = manager.stats()
        assert "cognitive_scheduler" in stats
        assert "pending" in stats["cognitive_scheduler"]
