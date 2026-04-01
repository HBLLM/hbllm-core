"""Tests for the RLHF feedback loop — API schema + LearnerNode integration."""

import pytest
import asyncio
import json
from pathlib import Path

from hbllm.serving.api import FeedbackRequest
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType, FeedbackPayload
from hbllm.brain.learner_node import LearnerNode


def test_feedback_request_schema():
    """Verify FeedbackRequest validates correctly."""
    req = FeedbackRequest(
        tenant_id="t1",
        message_id="msg_123",
        rating=1,
        prompt="What is AI?",
        response="AI is artificial intelligence.",
    )
    assert req.rating == 1
    assert req.message_id == "msg_123"


def test_feedback_request_rating_bounds():
    """Rating must be -1, 0, or 1."""
    with pytest.raises(Exception):
        FeedbackRequest(tenant_id="t1", message_id="m1", rating=5)
    with pytest.raises(Exception):
        FeedbackRequest(tenant_id="t1", message_id="m1", rating=-2)


@pytest.mark.asyncio
async def test_learner_node_receives_feedback(tmp_path):
    """Verify LearnerNode accumulates feedback and stitches DPO pairs."""
    bus = InProcessBus()
    await bus.start()
    
    # Use a temporary file for the queue to avoid interference
    q_dir = tmp_path / "reflection"
    q_dir.mkdir()
    q_path = q_dir / "dpo_queue.json"
    
    learner = LearnerNode(node_id="learner_test", batch_size=2)
    learner.queue_path = str(q_path)
    await learner.start(bus)
    
    assert len(learner.pending_pairs) == 0
    assert not q_path.exists()
    
    # Send a positive feedback message
    feedback_pos = Message(
        type=MessageType.FEEDBACK,
        source_node_id="api_server",
        tenant_id="t1",
        topic="system.feedback",
        payload={
            "message_id": "msg_001",
            "rating": 1,
            "prompt": "Hello",
            "response": "Hi there!",
        },
    )
    await bus.publish("system.feedback", feedback_pos)
    
    # Wait for processing with a short timeout (polling)
    for _ in range(50):
        if "Hello" in learner.pending_pairs:
            break
        await asyncio.sleep(0.01)
    
    assert "Hello" in learner.pending_pairs
    assert learner.pending_pairs["Hello"]["chosen"] == "Hi there!"
    assert learner.pending_pairs["Hello"]["rejected"] is None
    assert not q_path.exists()
    
    # Send a negative feedback message for the same prompt
    feedback_neg = Message(
        type=MessageType.FEEDBACK,
        source_node_id="api_server",
        tenant_id="t1",
        topic="system.feedback",
        payload={
            "message_id": "msg_002",
            "rating": -1,
            "prompt": "Hello",
            "response": "Crash",
        },
    )
    await bus.publish("system.feedback", feedback_neg)
    
    # Wait for stitching and persistence
    for _ in range(50):
        if q_path.exists():
            break
        await asyncio.sleep(0.01)
    
    # It should stitch them and move to the persistent queue
    assert "Hello" not in learner.pending_pairs
    assert q_path.exists()
    
    with open(q_path, "r") as f:
        queue = json.load(f)
    assert len(queue) == 1
    # Check content, converting result to list if it was a tuple
    assert list(queue[0]) == ["Hello", "Hi there!", "Crash"]
    
    await learner.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_learner_node_triggers_dpo_at_sleep(tmp_path):
    """LearnerNode should trigger DPO training only when sleep triggers."""
    q_dir = tmp_path / "reflection"
    q_dir.mkdir()
    q_path = q_dir / "dpo_queue.json"

    bus = InProcessBus()
    await bus.start()
    
    learner = LearnerNode(node_id="learner_test", batch_size=2, model=None, tokenizer=None)
    learner.queue_path = str(q_path)
    await learner.start(bus)
    
    # Create the queue file manually to simulate pending pairs
    initial_queue = [
        ["Q1", "Good 1", "Bad 1"],
        ["Q2", "Good 2", "Bad 2"]
    ]
    with open(q_path, "w") as f:
        json.dump(initial_queue, f)
    
    # Verify training has NOT started
    assert learner.training_task is None
    
    # Trigger Sleep
    sleep_trigger = Message(type=MessageType.EVENT, source_node_id="test", topic="system.sleep.dpo_trigger", payload={})
    await bus.publish("system.sleep.dpo_trigger", sleep_trigger)

    # Wait for DPO training trigger
    for _ in range(50):
        if learner.training_task is not None:
            break
        await asyncio.sleep(0.01)
    
    # Training task should have run
    assert learner.training_task is not None
    # Queue should be drained or draining (handle_sleep_trigger removes the file immediately)
    assert not q_path.exists()
    
    await learner.stop()
    await bus.stop()
