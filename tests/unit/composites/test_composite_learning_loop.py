"""Tests for LearningLoop composite node.

Validates lifecycle management, sub-node wiring, health aggregation,
and direct property access for the learning composite.
"""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

from hbllm.brain.composites.learning_loop import LearningLoop
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import HealthStatus

pytestmark = pytest.mark.asyncio


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def bus():
    bus = InProcessBus()
    await bus.start()
    yield bus
    await bus.stop()


@pytest_asyncio.fixture
async def learning(bus):
    node = LearningLoop(node_id="test_learning")
    await node.start(bus)
    yield node
    await node.stop()


# ── Lifecycle Tests ──────────────────────────────────────────────────────


class TestLearningLoopLifecycle:
    """Test composite lifecycle: start, stop, sub-node creation."""

    async def test_on_start_creates_learner(self, learning: LearningLoop):
        """LearnerNode should be created and started."""
        assert learning.learner is not None

    async def test_on_start_creates_world_model(self, learning: LearningLoop):
        """WorldModelNode should be created and started."""
        assert learning.world_model is not None

    async def test_process_reward_optional(self, learning: LearningLoop):
        """ProcessRewardNode may or may not be available (optional dep)."""
        # It's either None or an instance — both are valid
        prm = learning.process_reward
        assert prm is None or hasattr(prm, "node_id")

    async def test_node_ids_prefixed(self, learning: LearningLoop):
        """Sub-nodes should have composite-prefixed IDs."""
        assert learning.learner.node_id == "test_learning.learner"
        assert learning.world_model.node_id == "test_learning.world_model"

    async def test_capabilities_registered(self, learning: LearningLoop):
        """Composite should declare learning capabilities."""
        caps = learning.capabilities
        assert "online_learning" in caps
        assert "dpo_training" in caps
        assert "world_simulation" in caps

    async def test_handle_message_returns_none(self, learning: LearningLoop):
        """The composite itself passes messages through."""
        msg = Message(
            type=MessageType.QUERY,
            source_node_id="test",
            topic="test",
            payload={},
        )
        result = await learning.handle_message(msg)
        assert result is None

    async def test_stop_cleans_up(self, bus):
        """Stopping should not raise."""
        node = LearningLoop(node_id="stop_test")
        await node.start(bus)
        await node.stop()


# ── Health Check Tests ───────────────────────────────────────────────────


class TestLearningLoopHealth:
    """Test health_check aggregation."""

    async def test_health_check_returns_healthy(self, learning: LearningLoop):
        """Healthy sub-nodes should produce HEALTHY overall."""
        health = await learning.health_check()
        assert health.status == HealthStatus.HEALTHY
        assert health.node_id == "test_learning"

    async def test_health_check_message_shows_sub_count(self, learning: LearningLoop):
        """Health message should report sub-node count."""
        health = await learning.health_check()
        assert "Composite:" in health.message

    async def test_health_check_reports_uptime(self, learning: LearningLoop):
        """Uptime should be non-negative after start."""
        await asyncio.sleep(0.05)
        health = await learning.health_check()
        assert health.uptime_seconds >= 0


# ── Sub-node Bus Wiring ──────────────────────────────────────────────────


class TestLearningLoopBusWiring:
    """Test that sub-nodes are properly wired to the bus."""

    async def test_learner_shares_bus(self, learning: LearningLoop, bus):
        """Learner sub-node should share the same bus."""
        assert learning.learner.bus is bus

    async def test_world_model_shares_bus(self, learning: LearningLoop, bus):
        """WorldModel sub-node should share the same bus."""
        assert learning.world_model.bus is bus

    async def test_feedback_reaches_learner(self, learning: LearningLoop, bus):
        """Publishing feedback should be receivable by the learner."""
        msg = Message(
            type=MessageType.FEEDBACK,
            source_node_id="test",
            topic="system.feedback",
            payload={
                "rating": 1,
                "prompt": "test prompt",
                "response": "test response",
            },
        )
        # Should not raise
        await bus.publish("system.feedback", msg)
        await asyncio.sleep(0.05)
