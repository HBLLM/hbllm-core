"""Tests for the RLHF feedback loop — API schema + LearnerNode integration."""

import pytest
import asyncio

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
async def test_learner_node_receives_feedback():
    """Verify LearnerNode accumulates feedback and stitches DPO pairs."""
    bus = InProcessBus()
    await bus.start()
    
    learner = LearnerNode(node_id="learner_test", batch_size=2)
    await learner.start(bus)
    
    assert len(learner.pending_pairs) == 0
    assert len(learner.paired_buffer) == 0
    
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
    await asyncio.sleep(0.1)
    
    assert "Hello" in learner.pending_pairs
    assert learner.pending_pairs["Hello"]["chosen"] == "Hi there!"
    assert learner.pending_pairs["Hello"]["rejected"] is None
    assert len(learner.paired_buffer) == 0
    
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
    await asyncio.sleep(0.1)
    
    # It should stitch them and move to paired_buffer
    assert "Hello" not in learner.pending_pairs
    assert len(learner.paired_buffer) == 1
    assert learner.paired_buffer[0] == ("Hello", "Hi there!", "Crash")
    
    await learner.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_learner_node_triggers_dpo_at_batch_size():
    """LearnerNode should trigger DPO training when batch_size pairs arrive."""
    bus = InProcessBus()
    await bus.start()
    
    learner = LearnerNode(node_id="learner_test", batch_size=2, model=None, tokenizer=None)
    await learner.start(bus)
    
    # Send batch_size pairs to trigger training
    for i in range(learner.batch_size):
        # Positive
        msg_pos = Message(
            type=MessageType.FEEDBACK,
            source_node_id="api_server",
            topic="system.feedback",
            payload={"message_id": f"msg_p_{i}", "rating": 1, "prompt": f"Q{i}", "response": f"Good {i}"},
        )
        await bus.publish("system.feedback", msg_pos)
        
        # Negative
        msg_neg = Message(
            type=MessageType.FEEDBACK,
            source_node_id="api_server",
            topic="system.feedback",
            payload={"message_id": f"msg_n_{i}", "rating": -1, "prompt": f"Q{i}", "response": f"Bad {i}"},
        )
        await bus.publish("system.feedback", msg_neg)
        await asyncio.sleep(0.1)
    
    # Wait for DPO training trigger (1s)
    await asyncio.sleep(0.5)
    
    # Buffer should be drained after training trigger
    assert len(learner.paired_buffer) == 0
    # Training task should have run
    assert learner.training_task is not None
    
    await learner.stop()
    await bus.stop()
