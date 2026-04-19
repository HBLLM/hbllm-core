"""Tests for GrpcBus distributed message bus."""

from __future__ import annotations

import pytest

from hbllm.network.messages import Message, MessageType


class TestGrpcBusImport:
    def test_module_imports(self):
        from hbllm.network import grpc_bus

        assert hasattr(grpc_bus, "GrpcBus")

    def test_raises_without_grpcio(self):
        from hbllm.network.grpc_bus import _HAS_GRPC, GrpcBus

        if not _HAS_GRPC:
            with pytest.raises(RuntimeError, match="grpcio"):
                GrpcBus()

    def test_has_message_bus_methods(self):
        from hbllm.network.grpc_bus import _HAS_GRPC

        if not _HAS_GRPC:
            pytest.skip("grpcio not installed")
        from hbllm.network.grpc_bus import GrpcBus

        bus = GrpcBus()
        for method in ("publish", "subscribe", "request", "start", "stop", "add_interceptor"):
            assert hasattr(bus, method)


class TestGrpcBusLocal:
    @pytest.mark.asyncio
    async def test_subscribe_and_dispatch(self):
        from hbllm.network.grpc_bus import _HAS_GRPC

        if not _HAS_GRPC:
            pytest.skip("grpcio not installed")
        from hbllm.network.grpc_bus import GrpcBus

        bus = GrpcBus()
        bus._running = True

        received = []

        async def handler(msg: Message) -> Message | None:
            received.append(msg)
            return None

        await bus.subscribe("test.topic", handler)
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="src",
            topic="test.topic",
            payload={"key": "val"},
        )
        await bus.publish("test.topic", msg)
        assert len(received) == 1
        assert received[0].payload["key"] == "val"

    @pytest.mark.asyncio
    async def test_wildcard_subscribe(self):
        from hbllm.network.grpc_bus import _HAS_GRPC

        if not _HAS_GRPC:
            pytest.skip("grpcio not installed")
        from hbllm.network.grpc_bus import GrpcBus

        bus = GrpcBus()
        bus._running = True

        received = []

        async def handler(msg: Message) -> Message | None:
            received.append(msg)
            return None

        await bus.subscribe("node.*", handler)
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="src",
            topic="node.health",
            payload={},
        )
        await bus.publish("node.health", msg)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_interceptor(self):
        from hbllm.network.grpc_bus import _HAS_GRPC

        if not _HAS_GRPC:
            pytest.skip("grpcio not installed")
        from hbllm.network.grpc_bus import GrpcBus

        bus = GrpcBus()
        bus._running = True

        async def block_all(msg: Message) -> Message | None:
            return None  # Block

        bus.add_interceptor(block_all)

        received = []

        async def handler(msg: Message) -> Message | None:
            received.append(msg)
            return None

        await bus.subscribe("test", handler)
        msg = Message(type=MessageType.EVENT, source_node_id="s", topic="test", payload={})
        await bus.publish("test", msg)
        assert len(received) == 0  # Blocked
