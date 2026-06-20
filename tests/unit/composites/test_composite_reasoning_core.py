"""Tests for ReasoningCore composite node.

Validates lifecycle management, sub-node wiring, health aggregation,
and direct property access for the reasoning pipeline composite.
"""

from __future__ import annotations

import asyncio

import pytest

from hbllm.brain.composites.reasoning_core import ReasoningCore
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import HealthStatus

pytestmark = pytest.mark.asyncio


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
async def bus():
    bus = InProcessBus()
    await bus.start()
    yield bus
    await bus.stop()


@pytest.fixture
async def reasoning(bus, tmp_path):
    node = ReasoningCore(node_id="test_reasoning", data_dir=str(tmp_path))
    await node.start(bus)
    yield node
    await node.stop()


# ── Lifecycle Tests ──────────────────────────────────────────────────────


class TestReasoningCoreLifecycle:
    """Test composite lifecycle: start, stop, sub-node creation."""

    async def test_on_start_creates_router(self, reasoning: ReasoningCore):
        """RouterNode should be created after start."""
        assert reasoning.router is not None

    async def test_on_start_creates_planner(self, reasoning: ReasoningCore):
        """PlannerNode should be created after start."""
        assert reasoning.planner is not None

    async def test_on_start_creates_critic(self, reasoning: ReasoningCore):
        """CriticNode should be created after start."""
        assert reasoning.critic is not None

    async def test_on_start_creates_decision(self, reasoning: ReasoningCore):
        """DecisionNode should be created after start."""
        assert reasoning.decision is not None

    async def test_on_start_creates_revision(self, reasoning: ReasoningCore):
        """RevisionNode should be created after start."""
        assert reasoning.revision is not None

    async def test_node_ids_prefixed(self, reasoning: ReasoningCore):
        """Sub-nodes should have composite-prefixed IDs."""
        assert reasoning.router.node_id == "test_reasoning.router"
        assert reasoning.planner.node_id == "test_reasoning.planner"
        assert reasoning.critic.node_id == "test_reasoning.critic"
        assert reasoning.decision.node_id == "test_reasoning.decision"

    async def test_capabilities_registered(self, reasoning: ReasoningCore):
        """Composite should declare reasoning capabilities."""
        caps = reasoning.capabilities
        assert "routing" in caps
        assert "planning" in caps
        assert "critic" in caps
        assert "decision" in caps
        assert "revision" in caps

    async def test_handle_message_returns_none(self, reasoning: ReasoningCore):
        """The composite itself passes messages through."""
        msg = Message(
            type=MessageType.QUERY,
            source_node_id="test",
            topic="test",
            payload={},
        )
        result = await reasoning.handle_message(msg)
        assert result is None

    async def test_stop_cleans_up(self, bus, tmp_path):
        """Stopping should not raise."""
        node = ReasoningCore(node_id="stop_test", data_dir=str(tmp_path))
        await node.start(bus)
        await node.stop()


# ── Health Check Tests ───────────────────────────────────────────────────


class TestReasoningCoreHealth:
    """Test health_check aggregation."""

    async def test_health_check_returns_healthy(self, reasoning: ReasoningCore):
        """Healthy sub-nodes should produce HEALTHY overall."""
        health = await reasoning.health_check()
        assert health.status == HealthStatus.HEALTHY
        assert health.node_id == "test_reasoning"

    async def test_health_check_message_shows_sub_count(self, reasoning: ReasoningCore):
        """Health message should report sub-node count."""
        health = await reasoning.health_check()
        assert "Composite: 4" in health.message

    async def test_health_check_reports_uptime(self, reasoning: ReasoningCore):
        """Uptime should be non-negative after start."""
        await asyncio.sleep(0.05)
        health = await reasoning.health_check()
        assert health.uptime_seconds >= 0


# ── Sub-node Bus Wiring ──────────────────────────────────────────────────


class TestReasoningCoreBusWiring:
    """Test that sub-nodes are properly wired to the bus."""

    async def test_all_node_subs_share_bus(self, reasoning: ReasoningCore, bus):
        """All Node-subclass sub-nodes should share the same bus."""
        assert reasoning.router.bus is bus
        assert reasoning.planner.bus is bus
        assert reasoning.critic.bus is bus
        assert reasoning.decision.bus is bus
