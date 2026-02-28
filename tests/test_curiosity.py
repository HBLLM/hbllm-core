"""Tests for the Curiosity Engine — knowledge gap detection and goal generation."""

import pytest
import asyncio

from hbllm.brain.curiosity_node import (
    UncertaintyEvent, LearningGoal, GoalQueue, CuriosityNode,
)
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


# ─── GoalQueue Tests ──────────────────────────────────────────────────────────

def test_goal_queue_add():
    q = GoalQueue()
    goal = q.add_or_update("math", "Improve math reasoning", 0.8, 3)
    assert goal.topic == "math"
    assert len(q.goals) == 1


def test_goal_queue_update_existing():
    q = GoalQueue()
    q.add_or_update("math", "v1", 0.5, 2)
    q.add_or_update("math", "v2", 0.9, 3)
    assert len(q.goals) == 1
    assert q.goals[0].priority == 0.9
    assert q.goals[0].source_events == 5  # 2 + 3


def test_goal_queue_priority_sort():
    q = GoalQueue()
    q.add_or_update("low", "low priority", 0.2, 1)
    q.add_or_update("high", "high priority", 0.9, 5)
    q.add_or_update("mid", "mid priority", 0.5, 3)
    
    assert q.goals[0].topic == "high"
    assert q.goals[-1].topic == "low"


def test_goal_queue_pop_top():
    q = GoalQueue()
    q.add_or_update("topic_a", "desc", 0.9, 1)
    q.add_or_update("topic_b", "desc", 0.3, 1)
    
    top = q.pop_top()
    assert top.topic == "topic_a"
    assert top.status == "dispatched"
    
    next_top = q.pop_top()
    assert next_top.topic == "topic_b"


def test_goal_queue_pop_empty():
    q = GoalQueue()
    assert q.pop_top() is None


def test_goal_queue_max_size():
    q = GoalQueue(max_size=3)
    for i in range(10):
        q.add_or_update(f"topic_{i}", "desc", i * 0.1, 1)
    assert len(q.goals) == 3
    assert q.goals[0].priority == 0.9  # Highest priority kept


def test_goal_queue_summary():
    q = GoalQueue()
    q.add_or_update("a", "desc", 0.5, 1)
    q.add_or_update("b", "desc", 0.3, 1)
    q.pop_top()  # dispatches "a"
    
    s = q.summary()
    assert s["total"] == 2
    assert s["pending"] == 1
    assert s["dispatched"] == 1


# ─── CuriosityNode Integration Tests ─────────────────────────────────────────

@pytest.fixture
async def curiosity_node():
    bus = InProcessBus()
    await bus.start()
    node = CuriosityNode(
        node_id="curiosity_test",
        uncertainty_threshold=2,
        goal_dispatch_interval=0.0,  # Immediate dispatch for testing
    )
    await node.start(bus)
    yield node
    await node.stop()
    await bus.stop()


async def test_node_tracks_negative_feedback(curiosity_node):
    bus = curiosity_node.bus
    
    msg = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        tenant_id="t1",
        topic="system.feedback",
        payload={"rating": -1, "topic": "math", "prompt": "What is 2+2?"},
    )
    await bus.publish("system.feedback", msg)
    await asyncio.sleep(0.1)
    
    assert len(curiosity_node.events) == 1
    assert curiosity_node.topic_counts["math"] == 1


async def test_node_generates_goal_at_threshold(curiosity_node):
    """After threshold events, a learning goal should be created."""
    bus = curiosity_node.bus
    
    for _ in range(3):
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            tenant_id="t1",
            topic="system.feedback",
            payload={"rating": -1, "topic": "physics"},
        )
        await bus.publish("system.feedback", msg)
        await asyncio.sleep(0.05)
    
    pending = curiosity_node.goal_queue.get_pending()
    # Goal should have been dispatched (interval=0), so check dispatched + pending
    total = len(curiosity_node.goal_queue.goals)
    assert total >= 1


async def test_node_query_stats(curiosity_node):
    """Query returns curiosity stats."""
    bus = curiosity_node.bus
    
    query = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="curiosity.query",
        payload={},
    )
    resp = await bus.request("curiosity.query", query, timeout=5.0)
    assert "event_count" in resp.payload
    assert "goal_queue" in resp.payload
    assert "top_gaps" in resp.payload
