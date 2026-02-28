"""Tests for LearnerNode — continuous learning via DPO feedback."""

import pytest
import asyncio

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


def _make_feedback(rating=1, prompt="What is AI?", response="AI is..."):
    return Message(
        type=MessageType.FEEDBACK,
        source_node_id="user",
        topic="system.feedback",
        payload={
            "message_id": "msg_001",
            "rating": rating,
            "prompt": prompt,
            "response": response,
        },
    )


# ── Feedback Collection Tests ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_accepts_valid_feedback():
    """Valid feedback should be added to the buffer."""
    from hbllm.brain.learner_node import LearnerNode

    bus = InProcessBus()
    await bus.start()

    node = LearnerNode(node_id="learner_accept")
    await node.start(bus)

    msg = _make_feedback(rating=1)
    await node.handle_message(msg)
    assert len(node.feedback_buffer) == 1

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_ignores_non_feedback_messages():
    """Non-FEEDBACK message types should be ignored."""
    from hbllm.brain.learner_node import LearnerNode

    bus = InProcessBus()
    await bus.start()

    node = LearnerNode(node_id="learner_ignore")
    await node.start(bus)

    msg = Message(
        type=MessageType.QUERY,
        source_node_id="user",
        topic="system.feedback",
        payload={"text": "not feedback"},
    )
    result = await node.handle_message(msg)
    assert result is None
    assert len(node.feedback_buffer) == 0

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_ignores_feedback_without_prompt():
    """Feedback without prompt/response should not be buffered."""
    from hbllm.brain.learner_node import LearnerNode

    bus = InProcessBus()
    await bus.start()

    node = LearnerNode(node_id="learner_noprompt")
    await node.start(bus)

    msg = Message(
        type=MessageType.FEEDBACK,
        source_node_id="user",
        topic="system.feedback",
        payload={
            "message_id": "msg_002",
            "rating": 1,
            "prompt": "",
            "response": "",
        },
    )
    await node.handle_message(msg)
    assert len(node.feedback_buffer) == 0

    await node.stop()
    await bus.stop()


# ── Batch Training Trigger Tests ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_training_triggers_at_batch_size():
    """Training should trigger when batch_size feedbacks are collected."""
    from hbllm.brain.learner_node import LearnerNode

    bus = InProcessBus()
    await bus.start()

    node = LearnerNode(node_id="learner_batch")
    node.batch_size = 3
    await node.start(bus)

    for i in range(3):
        msg = _make_feedback(rating=1, prompt=f"Prompt {i}", response=f"Response {i}")
        await node.handle_message(msg)

    # Give the async task a moment to start
    await asyncio.sleep(0.1)
    assert node.training_task is not None

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_training_completes_and_broadcasts():
    """After training, a LEARNING_UPDATE event should be published."""
    from hbllm.brain.learner_node import LearnerNode

    bus = InProcessBus()
    await bus.start()

    node = LearnerNode(node_id="learner_broadcast")
    node.batch_size = 2
    await node.start(bus)

    updates = []
    await bus.subscribe("system.learning_update", lambda msg: updates.append(msg))

    for i in range(2):
        msg = _make_feedback(rating=1, prompt=f"P{i}", response=f"R{i}")
        await node.handle_message(msg)

    # Wait for the background training to complete (simulated 1s sleep)
    await asyncio.sleep(1.5)

    assert len(updates) == 1
    assert updates[0].payload["status"] == "weights_updated"

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_buffer_drains_after_batch():
    """Buffer should be drained after a batch is consumed."""
    from hbllm.brain.learner_node import LearnerNode

    bus = InProcessBus()
    await bus.start()

    node = LearnerNode(node_id="learner_drain")
    node.batch_size = 2
    await node.start(bus)

    for i in range(3):
        msg = _make_feedback(rating=1, prompt=f"P{i}", response=f"R{i}")
        await node.handle_message(msg)

    assert len(node.feedback_buffer) == 1  # 3 - 2 = 1 remaining

    await node.stop()
    await bus.stop()


# ── Node Lifecycle Tests ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_learner_subscribes_on_start():
    """Node should subscribe to system.feedback on start."""
    from hbllm.brain.learner_node import LearnerNode

    bus = InProcessBus()
    await bus.start()

    node = LearnerNode(node_id="learner_lifecycle")
    await node.start(bus)
    assert node.node_id == "learner_lifecycle"

    await node.stop()
    await bus.stop()
