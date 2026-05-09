"""Tests for ResourceManager composite node."""

from __future__ import annotations

import asyncio

import pytest

from hbllm.brain.composites.resource_manager import ResourceManager
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import HealthStatus

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def bus():
    bus = InProcessBus()
    await bus.start()
    yield bus
    await bus.stop()


@pytest.fixture
async def resource_mgr(bus, tmp_path):
    node = ResourceManager(node_id="test_resource", data_dir=str(tmp_path))
    await node.start(bus)
    yield node
    await node.stop()


class TestResourceManagerLifecycle:
    async def test_on_start_creates_all_sub_nodes(self, resource_mgr):
        assert resource_mgr.workspace is not None
        assert resource_mgr.attention is not None
        assert resource_mgr.load_manager is not None
        assert resource_mgr.scheduler is not None

    async def test_node_ids_prefixed(self, resource_mgr):
        assert resource_mgr.workspace.node_id == "test_resource.workspace"
        assert resource_mgr.attention.node_id == "test_resource.attention"
        assert resource_mgr.load_manager.node_id == "test_resource.load"
        assert resource_mgr.scheduler.node_id == "test_resource.scheduler"

    async def test_capabilities_registered(self, resource_mgr):
        caps = resource_mgr.capabilities
        assert "workspace" in caps
        assert "attention" in caps
        assert "scheduling" in caps

    async def test_handle_message_returns_none(self, resource_mgr):
        msg = Message(type=MessageType.QUERY, source_node_id="t", topic="t", payload={})
        assert await resource_mgr.handle_message(msg) is None

    async def test_stop_cleans_up(self, bus, tmp_path):
        node = ResourceManager(node_id="stop_test", data_dir=str(tmp_path))
        await node.start(bus)
        await node.stop()


class TestResourceManagerHealth:
    async def test_health_check_returns_healthy(self, resource_mgr):
        health = await resource_mgr.health_check()
        assert health.status == HealthStatus.HEALTHY

    async def test_health_check_message(self, resource_mgr):
        health = await resource_mgr.health_check()
        assert "Composite: 4" in health.message

    async def test_uptime(self, resource_mgr):
        await asyncio.sleep(0.05)
        health = await resource_mgr.health_check()
        assert health.uptime_seconds >= 0


class TestResourceManagerBusWiring:
    async def test_all_sub_nodes_share_bus(self, resource_mgr, bus):
        assert resource_mgr.workspace.bus is bus
        assert resource_mgr.attention.bus is bus
        assert resource_mgr.load_manager.bus is bus
        assert resource_mgr.scheduler.bus is bus

    async def test_scheduler_has_db(self, resource_mgr):
        assert resource_mgr.scheduler.db_path.exists()
