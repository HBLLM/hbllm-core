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
    """Verify LearnerNode accumulates feedback from the bus."""
    bus = InProcessBus()
    await bus.start()
    
    learner = LearnerNode(node_id="learner_test")
    await learner.start(bus)
    
    assert len(learner.feedback_buffer) == 0
    
    # Send a feedback message
    feedback_msg = Message(
        type=MessageType.FEEDBACK,
        source_node_id="api_server",
        tenant_id="t1",
        topic="system.feedback",
        payload={
            "message_id": "msg_001",
            "rating": 1,
            "prompt": "Hello",
            "response": "Hi there!",
            "comment": "Great response",
        },
    )
    await bus.publish("system.feedback", feedback_msg)
    await asyncio.sleep(0.2)
    
    assert len(learner.feedback_buffer) == 1
    assert learner.feedback_buffer[0].rating == 1
    assert learner.feedback_buffer[0].prompt == "Hello"
    
    await learner.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_learner_node_triggers_dpo_at_batch_size():
    """LearnerNode should trigger DPO training when batch_size feedbacks arrive."""
    bus = InProcessBus()
    await bus.start()
    
    learner = LearnerNode(node_id="learner_test")
    await learner.start(bus)
    
    # Send batch_size feedbacks to trigger training
    for i in range(learner.batch_size):
        msg = Message(
            type=MessageType.FEEDBACK,
            source_node_id="api_server",
            tenant_id="t1",
            topic="system.feedback",
            payload={
                "message_id": f"msg_{i}",
                "rating": 1 if i % 2 == 0 else -1,
                "prompt": f"Question {i}",
                "response": f"Answer {i}",
            },
        )
        await bus.publish("system.feedback", msg)
        await asyncio.sleep(0.1)
    
    # Wait for DPO training to complete (simulated 1s)
    await asyncio.sleep(2.0)
    
    # Buffer should be drained after training
    assert len(learner.feedback_buffer) == 0
    # Training task should have run
    assert learner.training_task is not None
    
    await learner.stop()
    await bus.stop()
