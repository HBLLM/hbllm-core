"""Integration tests for the self-learning pipeline.

Exercises the full Evaluation → Learner → DPO pipeline end-to-end
using the message bus, without requiring a model or tokenizer.
"""

import asyncio
import json
import os

import pytest

from hbllm.brain.evaluation_node import EvaluationNode
from hbllm.brain.learner_node import LearnerNode
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


@pytest.fixture
def clean_dpo_queue():
    """Ensure dpo_queue.json is clean before/after each test."""
    path = "workspace/reflection/dpo_queue.json"
    if os.path.exists(path):
        os.remove(path)
    yield path
    if os.path.exists(path):
        os.remove(path)


async def _setup_pipeline(
    *,
    micro_learn_threshold: float = 0.3,
    distillation_threshold: float = 0.85,
    mock_model: bool = False,
) -> tuple[InProcessBus, LearnerNode, EvaluationNode]:
    """Wire up a bus with LearnerNode + EvaluationNode subscribed."""
    bus = InProcessBus()
    await bus.start()

    from unittest.mock import MagicMock

    model = MagicMock() if mock_model else None
    tokenizer = MagicMock() if mock_model else None

    learner = LearnerNode(
        node_id="test_learner",
        model=model,
        tokenizer=tokenizer,
        enable_micro_learning=True,
        micro_learn_threshold=micro_learn_threshold,
        distillation_threshold=distillation_threshold,
    )
    await learner.start(bus)

    eval_node = EvaluationNode(
        node_id="test_evaluator",
    )
    await eval_node.start(bus)

    return bus, learner, eval_node


def _make_eval_event(
    query: str,
    response: str,
    score: float,
    *,
    correlation_id: str = "",
) -> Message:
    """Create a synthetic system.evaluation message."""
    return Message(
        type=MessageType.EVENT,
        source_node_id="test_evaluator",
        topic="system.evaluation",
        payload={
            "correlation_id": correlation_id or f"corr_{hash(query) % 10000}",
            "timestamp": 1000.0,
            "task_success": score,
            "plan_validity": score,
            "tool_accuracy": 0.8,
            "memory_usage": 0.5,
            "confidence_error": max(0.0, 1.0 - score),
            "overall_score": score,
            "query": query,
            "response": response,
            "flags": [],
            "dimensions": {"task_success": score},
        },
    )


# ── Test 1: Low-score → micro-learn queue ────────────────────────────────


class TestEvaluationToMicroLearnQueue:
    """Publishing a low-scoring evaluation should queue it in LearnerNode."""

    @pytest.mark.asyncio
    async def test_low_score_queued(self, clean_dpo_queue):
        bus, learner, eval_node = await _setup_pipeline()

        msg = _make_eval_event(
            query="What is the capital of France?",
            response="I'm not sure, maybe London?",
            score=0.15,
        )
        await bus.publish("system.evaluation", msg)
        await asyncio.sleep(0.05)

        queue = learner.get_micro_learn_queue()
        assert len(queue) == 1
        assert queue[0]["query"] == "What is the capital of France?"
        assert queue[0]["bad_response"] == "I'm not sure, maybe London?"
        assert queue[0]["score"] == 0.15

        await learner.stop()
        await eval_node.stop()
        await bus.stop()

    @pytest.mark.asyncio
    async def test_above_threshold_not_queued(self, clean_dpo_queue):
        """Score above micro_learn_threshold should NOT be queued."""
        bus, learner, eval_node = await _setup_pipeline()

        msg = _make_eval_event(
            query="What is 2+2?",
            response="4",
            score=0.5,
        )
        await bus.publish("system.evaluation", msg)
        await asyncio.sleep(0.05)

        queue = learner.get_micro_learn_queue()
        assert len(queue) == 0

        await learner.stop()
        await eval_node.stop()
        await bus.stop()


# ── Test 2: Retry correction triggers micro-learning ─────────────────────


class TestRetryCorrection:
    """Sending a high score for a previously low-scored query triggers micro-learn."""

    @pytest.mark.asyncio
    async def test_high_score_retry_triggers_micro_learn(self, clean_dpo_queue):
        bus, learner, eval_node = await _setup_pipeline(mock_model=True)

        # Step 1: Low-scoring evaluation → queues the bad response
        bad_msg = _make_eval_event(
            query="What is the capital of France?",
            response="Maybe London?",
            score=0.15,
        )
        await bus.publish("system.evaluation", bad_msg)
        await asyncio.sleep(0.05)

        assert len(learner.get_micro_learn_queue()) == 1

        # Step 2: High-scoring evaluation for the same query → triggers micro-learn
        good_msg = _make_eval_event(
            query="What is the capital of France?",
            response="The capital of France is Paris.",
            score=0.95,
        )
        await bus.publish("system.evaluation", good_msg)
        await asyncio.sleep(0.1)

        # The micro-learn queue should be drained (item matched & consumed)
        assert len(learner.get_micro_learn_queue()) == 0
        # Micro-learn step count should increment
        assert learner._micro_learn_steps >= 1

        await learner.stop()
        await eval_node.stop()
        await bus.stop()


# ── Test 3: Explicit feedback → EvaluationNode → micro_learn event ───────


class TestExplicitFeedbackToMicroLearn:
    """Negative user feedback should trigger micro_learn event via EvaluationNode."""

    @pytest.mark.asyncio
    async def test_negative_feedback_emits_micro_learn(self, clean_dpo_queue):
        bus, learner, eval_node = await _setup_pipeline(mock_model=True)

        micro_events: list[Message] = []

        async def _on_micro_events_191(msg):
            micro_events.append(msg)

        await bus.subscribe("system.micro_learn", _on_micro_events_191)

        # Seed a pending context so EvaluationNode has something to work with
        corr_id = "feedback_test_001"
        eval_node._pending_contexts[corr_id] = {
            "intent": "answer",
            "thought_type": "intuition",
            "content": "What is quantum computing?",
            "confidence": 0.8,
            "tools_used": [],
            "memory_hits": 0,
            "plan_steps": [],
            "output": "Quantum is about quarks.",
        }

        # Send negative feedback
        feedback_msg = Message(
            type=MessageType.FEEDBACK,
            source_node_id="user",
            topic="system.feedback",
            payload={
                "message_id": "msg_fb_001",
                "rating": -1,
                "prompt": "What is quantum computing?",
                "response": "Quantum is about quarks.",
            },
            correlation_id=corr_id,
        )
        await bus.publish("system.feedback", feedback_msg)
        await asyncio.sleep(0.1)

        # EvaluationNode should have published a micro_learn event
        assert len(micro_events) >= 1
        payload = micro_events[0].payload
        assert payload["query"] == "What is quantum computing?"
        assert payload["bad_response"] == "Quantum is about quarks."

        await learner.stop()
        await eval_node.stop()
        await bus.stop()


# ── Test 4: High-scoring → distillation bank ─────────────────────────────


class TestDistillationBank:
    """High-scoring evaluations (new queries) should be banked for distillation."""

    @pytest.mark.asyncio
    async def test_high_score_new_query_banked(self, clean_dpo_queue):
        bus, learner, eval_node = await _setup_pipeline()

        msg = _make_eval_event(
            query="Explain photosynthesis.",
            response="Photosynthesis is the process by which plants convert light...",
            score=0.92,
        )
        await bus.publish("system.evaluation", msg)
        await asyncio.sleep(0.05)

        bank = learner.get_distillation_bank()
        assert len(bank) == 1
        assert bank[0]["query"] == "Explain photosynthesis."
        assert learner._distillation_count == 1

        await learner.stop()
        await eval_node.stop()
        await bus.stop()

    @pytest.mark.asyncio
    async def test_medium_score_not_banked(self, clean_dpo_queue):
        """Score between thresholds should not be queued or banked."""
        bus, learner, eval_node = await _setup_pipeline()

        msg = _make_eval_event(
            query="Some question",
            response="Some answer",
            score=0.6,
        )
        await bus.publish("system.evaluation", msg)
        await asyncio.sleep(0.05)

        assert len(learner.get_distillation_bank()) == 0
        assert len(learner.get_micro_learn_queue()) == 0

        await learner.stop()
        await eval_node.stop()
        await bus.stop()


# ── Test 5: Sleep-triggered DPO → learning_update broadcast ──────────────


class TestSleepTriggeredDPO:
    """Sleep trigger with queued DPO pairs should broadcast learning_update."""

    @pytest.mark.asyncio
    async def test_sleep_trigger_broadcasts_update(self, clean_dpo_queue):
        bus, learner, eval_node = await _setup_pipeline()

        updates: list[Message] = []

        async def _on_updates_292(msg):
            updates.append(msg)

        await bus.subscribe("system.learning_update", _on_updates_292)

        # Manually populate the DPO queue on disk
        os.makedirs(os.path.dirname(learner.queue_path), exist_ok=True)
        with open(learner.queue_path, "w") as f:
            json.dump(
                [["What is AI?", "Good answer about AI", "Bad answer about AI"]],
                f,
            )

        # Fire the sleep DPO trigger
        sleep_msg = Message(
            type=MessageType.EVENT,
            source_node_id="test_sleep",
            topic="system.sleep.dpo_trigger",
            payload={},
        )
        await bus.publish("system.sleep.dpo_trigger", sleep_msg)
        await asyncio.sleep(0.5)

        # Should broadcast learning_update (even without model, it completes gracefully)
        assert len(updates) >= 1
        assert updates[0].payload["status"] == "weights_updated"

        await learner.stop()
        await eval_node.stop()
        await bus.stop()

    @pytest.mark.asyncio
    async def test_sleep_trigger_empty_queue(self, clean_dpo_queue):
        """Sleep trigger with no pending pairs should still broadcast."""
        bus, learner, eval_node = await _setup_pipeline()

        updates: list[Message] = []

        async def _on_updates_326(msg):
            updates.append(msg)

        await bus.subscribe("system.learning_update", _on_updates_326)

        sleep_msg = Message(
            type=MessageType.EVENT,
            source_node_id="test_sleep",
            topic="system.sleep.dpo_trigger",
            payload={},
        )
        await bus.publish("system.sleep.dpo_trigger", sleep_msg)
        await asyncio.sleep(0.2)

        assert len(updates) >= 1

        await learner.stop()
        await eval_node.stop()
        await bus.stop()


# ── Test 6: Stats integrity after events ─────────────────────────────────


class TestStatsIntegrity:
    """After a series of events, micro_learning_stats() should reflect accurate counts."""

    @pytest.mark.asyncio
    async def test_stats_after_mixed_events(self, clean_dpo_queue):
        bus, learner, eval_node = await _setup_pipeline()

        # 1. Send 3 low-scoring evaluations
        for i in range(3):
            msg = _make_eval_event(
                query=f"Bad query {i}",
                response=f"Bad response {i}",
                score=0.1,
            )
            await bus.publish("system.evaluation", msg)
        await asyncio.sleep(0.05)

        # 2. Send 2 high-scoring evaluations (new queries → distillation)
        for i in range(2):
            msg = _make_eval_event(
                query=f"Good query {i}",
                response=f"Excellent response {i}",
                score=0.95,
            )
            await bus.publish("system.evaluation", msg)
        await asyncio.sleep(0.05)

        # 3. Verify stats
        stats = learner.micro_learning_stats()
        assert stats["enabled"] is True
        assert stats["micro_queue_depth"] == 3
        assert stats["distillation_bank_size"] == 2
        assert stats["distillation_count"] == 2
        assert stats["thresholds"]["micro_learn"] == 0.3
        assert stats["thresholds"]["distillation"] == 0.85

        await learner.stop()
        await eval_node.stop()
        await bus.stop()

    @pytest.mark.asyncio
    async def test_stats_after_correction(self, clean_dpo_queue):
        """Stats should update correctly after a micro-learn correction."""
        bus, learner, eval_node = await _setup_pipeline(mock_model=True)

        # Low-score → queue
        bad_msg = _make_eval_event(
            query="Correction test query",
            response="Wrong answer",
            score=0.1,
        )
        await bus.publish("system.evaluation", bad_msg)
        await asyncio.sleep(0.05)
        assert learner.micro_learning_stats()["micro_queue_depth"] == 1

        # High-score same query → micro-learn fires, queue drains
        good_msg = _make_eval_event(
            query="Correction test query",
            response="Correct answer here",
            score=0.95,
        )
        await bus.publish("system.evaluation", good_msg)
        await asyncio.sleep(0.1)

        stats = learner.micro_learning_stats()
        assert stats["micro_queue_depth"] == 0
        assert stats["micro_learn_steps"] >= 1

        await learner.stop()
        await eval_node.stop()
        await bus.stop()
