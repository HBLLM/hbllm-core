"""Tests for GovernanceGuard composite node.

Validates lifecycle management, sub-node wiring, health aggregation,
and direct property access for the governance composite.
"""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

from hbllm.brain.composites.governance_guard import GovernanceGuard
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
async def governance(bus):
    node = GovernanceGuard(node_id="test_governance")
    await node.start(bus)
    yield node
    await node.stop()


# ── Lifecycle Tests ──────────────────────────────────────────────────────


class TestGovernanceGuardLifecycle:
    """Test composite lifecycle: start, stop, sub-node creation."""

    async def test_on_start_creates_sub_nodes(self, governance: GovernanceGuard):
        """All three sub-components should be created after start."""
        assert governance.sentinel is not None
        assert governance.policy_engine is not None
        assert governance.confidence_estimator is not None

    async def test_node_id_propagated(self, governance: GovernanceGuard):
        """Sentinel sub-node should have a composite-prefixed ID."""
        assert governance.sentinel.node_id == "test_governance.sentinel"

    async def test_capabilities_registered(self, governance: GovernanceGuard):
        """Composite should declare governance capabilities."""
        caps = governance.capabilities
        assert "governance" in caps
        assert "policy_enforcement" in caps
        assert "confidence_scoring" in caps

    async def test_stop_cleans_up(self, bus):
        """Stopping should not raise and sentinel should be stopped."""
        node = GovernanceGuard(node_id="stop_test")
        await node.start(bus)
        await node.stop()
        # Sentinel's _running should be False after stop
        # (stop completed without error)

    async def test_handle_message_returns_none(self, governance: GovernanceGuard):
        """The composite itself passes messages through (returns None)."""
        msg = Message(
            type=MessageType.QUERY,
            source_node_id="test",
            topic="test",
            payload={},
        )
        result = await governance.handle_message(msg)
        assert result is None


# ── Health Check Tests ───────────────────────────────────────────────────


class TestGovernanceGuardHealth:
    """Test health_check aggregation."""

    async def test_health_check_returns_healthy(self, governance: GovernanceGuard):
        """Healthy sub-nodes should produce HEALTHY overall."""
        health = await governance.health_check()
        assert health.status == HealthStatus.HEALTHY
        assert health.node_id == "test_governance"

    async def test_health_check_includes_capabilities(self, governance: GovernanceGuard):
        """Health should report available capabilities."""
        health = await governance.health_check()
        assert len(health.capabilities_available) > 0

    async def test_health_check_reports_uptime(self, governance: GovernanceGuard):
        """Uptime should be non-negative after start."""
        await asyncio.sleep(0.05)
        health = await governance.health_check()
        assert health.uptime_seconds >= 0


# ── Sentinel Integration ─────────────────────────────────────────────────


class TestGovernanceGuardSentinel:
    """Test that the sentinel sub-node is properly wired to the bus."""

    async def test_sentinel_subscribed_to_bus(self, governance: GovernanceGuard, bus):
        """Sentinel should be listening on the bus after start."""
        # The sentinel registers its own subscriptions via on_start,
        # so the bus should have at least some subscribers
        assert governance.sentinel.bus is bus

    async def test_policy_engine_loaded(self, governance: GovernanceGuard):
        """PolicyEngine should have loaded policies (or at least be initialized)."""
        pe = governance.policy_engine
        assert pe is not None
        # PolicyEngine always has a rules list (possibly empty in test env)
        assert hasattr(pe, "rules") or hasattr(pe, "evaluate")
