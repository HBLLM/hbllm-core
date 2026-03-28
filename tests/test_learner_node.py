"""Tests for LearnerNode — continuous learning via DPO feedback during deep sleep."""

import os
import json
import pytest
import asyncio

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.brain.learner_node import LearnerNode

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

@pytest.fixture
def clean_queue():
    path = "workspace/reflection/dpo_queue.json"
    if os.path.exists(path):
        os.remove(path)
    yield
    if os.path.exists(path):
        os.remove(path)

# ── Feedback Collection Tests ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_accepts_valid_feedback_and_queues(clean_queue):
    """Valid feedback should be stitched and written to persistent JSON queue."""
    bus = InProcessBus()
    await bus.start()

    node = LearnerNode(node_id="learner_accept")
    await node.start(bus)

    msg_pos = _make_feedback(rating=1)
    await node.handle_message(msg_pos)
    assert "What is AI?" in node.pending_pairs
    
    msg_neg = _make_feedback(rating=-1, response="I don't know")
    await node.handle_message(msg_neg)
    
    # Pair should be completed, removed from pending, and written to disk
    assert "What is AI?" not in node.pending_pairs
    assert os.path.exists(node.queue_path)
    
    with open(node.queue_path) as f:
        queue = json.load(f)
        assert len(queue) == 1
        assert queue[0][0] == "What is AI?"

    await node.stop()
    await bus.stop()


# ── Sleep Trigger Tests (Overnight Learning) ─────────────────────────────────

@pytest.mark.asyncio
async def test_training_triggers_ONLY_on_sleep(clean_queue):
    """Training should not trigger until system.sleep.dpo_trigger is sent."""
    bus = InProcessBus()
    await bus.start()

    node = LearnerNode(node_id="learner_batch", model=None, tokenizer=None)
    node.batch_size = 3
    await node.start(bus)

    # Build 3 contrastive pairs (Hits batch_size exactly)
    for i in range(3):
        pos = _make_feedback(rating=1, prompt=f"Prompt {i}", response=f"Good {i}")
        await node.handle_message(pos)
        neg = _make_feedback(rating=-1, prompt=f"Prompt {i}", response=f"Bad {i}")
        await node.handle_message(neg)

    # 1. Ensure training did NOT start (Continuous learning requires SleepNode trigger)
    assert node.training_task is None

    # 2. Fire the SleepNode DPO trigger
    sleep_trigger = Message(
        type=MessageType.EVENT, source_node_id="test", topic="system.sleep.dpo_trigger", payload={}
    )
    await bus.publish("system.sleep.dpo_trigger", sleep_trigger)
    await asyncio.sleep(0.1)
    
    # 3. Training task should now be spawned
    assert node.training_task is not None

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_training_completes_and_broadcasts(clean_queue):
    """After SleepNode triggers training, a LEARNING_UPDATE event should be published."""
    bus = InProcessBus()
    await bus.start()

    node = LearnerNode(node_id="learner_broadcast", model=None, tokenizer=None)
    node.batch_size = 2
    await node.start(bus)

    updates = []
    await bus.subscribe("system.learning_update", lambda msg: updates.append(msg))

    # Add 1 pair
    pos = _make_feedback(rating=1, prompt=f"P", response=f"Good")
    await node.handle_message(pos)
    neg = _make_feedback(rating=-1, prompt=f"P", response=f"Bad")
    await node.handle_message(neg)

    # Trigger Sleep
    sleep_trigger = Message(type=MessageType.EVENT, source_node_id="test", topic="system.sleep.dpo_trigger", payload={})
    await bus.publish("system.sleep.dpo_trigger", sleep_trigger)

    # Wait for the background training to complete
    await asyncio.sleep(0.5)

    assert len(updates) == 1
    assert updates[0].payload["status"] == "weights_updated"

    await node.stop()
    await bus.stop()
