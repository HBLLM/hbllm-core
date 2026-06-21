"""Tests for SocialLayer composite node."""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

from hbllm.brain.composites.social_layer import SocialLayer
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import HealthStatus

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def bus():
    bus = InProcessBus()
    await bus.start()
    yield bus
    await bus.stop()


@pytest_asyncio.fixture
async def social(bus):
    node = SocialLayer(node_id="test_social")
    await node.start(bus)
    yield node
    await node.stop()


class TestSocialLayerLifecycle:
    async def test_on_start_creates_collective(self, social):
        assert social.collective is not None

    async def test_on_start_creates_identity(self, social):
        assert social.identity is not None

    async def test_node_ids_prefixed(self, social):
        assert social.collective.node_id == "test_social.collective"
        assert social.identity.node_id == "test_social.identity"

    async def test_capabilities_registered(self, social):
        caps = social.capabilities
        assert "collective_intelligence" in caps
        assert "identity" in caps
        assert "consensus_voting" in caps

    async def test_handle_message_returns_none(self, social):
        msg = Message(type=MessageType.QUERY, source_node_id="t", topic="t", payload={})
        assert await social.handle_message(msg) is None

    async def test_stop_cleans_up(self, bus):
        node = SocialLayer(node_id="stop_test")
        await node.start(bus)
        await node.stop()


class TestSocialLayerHealth:
    async def test_health_check_returns_healthy(self, social):
        health = await social.health_check()
        assert health.status == HealthStatus.HEALTHY

    async def test_health_check_message(self, social):
        health = await social.health_check()
        assert "Composite: 2" in health.message

    async def test_uptime(self, social):
        await asyncio.sleep(0.05)
        health = await social.health_check()
        assert health.uptime_seconds >= 0


class TestSocialLayerBusWiring:
    async def test_all_sub_nodes_share_bus(self, social, bus):
        assert social.collective.bus is bus
        assert social.identity.bus is bus

    async def test_collective_election_cap(self, social):
        """CollectiveNode should have a _max_concurrent_elections cap."""
        assert social.collective._max_concurrent_elections == 50
