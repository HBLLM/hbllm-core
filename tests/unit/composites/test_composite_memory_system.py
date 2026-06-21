"""Tests for MemorySystem composite node.

Validates lifecycle management, sub-node wiring, health aggregation,
proactive cache warming, and direct property access.
"""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

from hbllm.brain.composites.memory_system import MemorySystem
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
async def memory_system(bus):
    node = MemorySystem(node_id="test_memory")
    await node.start(bus)
    yield node
    await node.stop()


# ── Lifecycle Tests ──────────────────────────────────────────────────────


class TestMemorySystemLifecycle:
    """Test composite lifecycle: start, stop, sub-node creation."""

    async def test_on_start_creates_memory_node(self, memory_system: MemorySystem):
        """MemoryNode should be created after start."""
        assert memory_system.memory is not None

    async def test_on_start_creates_experience_node(self, memory_system: MemorySystem):
        """ExperienceNode should be created after start."""
        assert memory_system.experience is not None

    async def test_on_start_creates_sleep_node(self, memory_system: MemorySystem):
        """SleepCycleNode should be created after start."""
        assert memory_system.sleep is not None

    async def test_node_ids_prefixed(self, memory_system: MemorySystem):
        """Sub-nodes should have composite-prefixed IDs."""
        assert memory_system.memory.node_id == "test_memory.memory"
        assert memory_system.experience.node_id == "test_memory.experience"
        assert memory_system.sleep.node_id == "test_memory.sleep"

    async def test_capabilities_registered(self, memory_system: MemorySystem):
        """Composite should declare memory capabilities."""
        caps = memory_system.capabilities
        assert "memory" in caps
        assert "episodic_memory" in caps
        assert "sleep_cycle" in caps
        assert "salience_detection" in caps

    async def test_handle_message_returns_none(self, memory_system: MemorySystem):
        """The composite itself passes messages through."""
        msg = Message(
            type=MessageType.QUERY,
            source_node_id="test",
            topic="test",
            payload={},
        )
        result = await memory_system.handle_message(msg)
        assert result is None

    async def test_stop_cleans_up(self, bus):
        """Stopping should not raise."""
        node = MemorySystem(node_id="stop_test")
        await node.start(bus)
        await node.stop()


# ── Health Check Tests ───────────────────────────────────────────────────


class TestMemorySystemHealth:
    """Test health_check aggregation."""

    async def test_health_check_returns_healthy(self, memory_system: MemorySystem):
        """Healthy sub-nodes should produce HEALTHY overall."""
        health = await memory_system.health_check()
        assert health.status == HealthStatus.HEALTHY
        assert health.node_id == "test_memory"

    async def test_health_check_message_shows_sub_count(self, memory_system: MemorySystem):
        """Health message should report sub-node count."""
        health = await memory_system.health_check()
        assert "Composite: 3" in health.message

    async def test_health_check_reports_uptime(self, memory_system: MemorySystem):
        """Uptime should be non-negative after start."""
        await asyncio.sleep(0.05)
        health = await memory_system.health_check()
        assert health.uptime_seconds >= 0


# ── Sub-node Bus Wiring ──────────────────────────────────────────────────


class TestMemorySystemBusWiring:
    """Test that sub-nodes are properly wired to the bus."""

    async def test_all_sub_nodes_share_bus(self, memory_system: MemorySystem, bus):
        """All sub-nodes should share the composite's bus."""
        assert memory_system.memory.bus is bus
        assert memory_system.experience.bus is bus
        assert memory_system.sleep.bus is bus

    async def test_memory_store_reachable(self, memory_system: MemorySystem, bus):
        """Publishing to memory.store should not raise."""
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="memory.store",
            payload={"text": "test fact", "tenant_id": "default"},
        )
        await bus.publish("memory.store", msg)
        await asyncio.sleep(0.05)


# ── Cache Warming ────────────────────────────────────────────────────────


class TestMemorySystemWarmCache:
    """Test proactive cache warming fires without error."""

    async def test_warm_cache_runs_silently(self, memory_system: MemorySystem):
        """_warm_memory_cache should complete without raising."""
        # It was already kicked off in on_start; just wait a bit
        await asyncio.sleep(0.1)
        # No assertion needed — we just confirm no crash
