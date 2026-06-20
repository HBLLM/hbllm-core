"""Integration tests for Network subsystem — InProcessTransport, VectorClock."""

import asyncio
from typing import Any

import pytest

from hbllm.network.messages import Message, Priority, MessageType
from hbllm.network.transports.inprocess import InProcessTransport


# ── InProcessTransport Integration ───────────────────────────────────────────


class TestInProcessTransportIntegration:
    """Test message dispatch, subscriptions, and priority ordering."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_start_and_stop(self):
        transport = InProcessTransport(transport_id="test")
        await transport.start()
        assert transport._running is True
        await transport.stop()
        assert transport._running is False

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_publish_subscribe(self):
        transport = InProcessTransport(transport_id="test")
        await transport.start()

        received = []

        async def handler(msg: Message):
            received.append(msg)

        await transport.subscribe("test.topic", handler)

        msg = Message(
            type=MessageType.QUERY,
            source_node_id="test",
            topic="test.topic",
            payload={"data": "hello"},
        )
        await transport.send("test.topic", msg)

        # Wait for dispatch
        await asyncio.sleep(0.3)

        assert len(received) == 1
        assert received[0].payload["data"] == "hello"

        await transport.stop()

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_wildcard_subscription(self):
        transport = InProcessTransport(transport_id="test")
        await transport.start()

        received = []

        async def handler(msg: Message):
            received.append(msg)

        await transport.subscribe("test.*", handler)

        msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="test.subtopic",
            payload={},
        )
        await transport.send("test.subtopic", msg)
        await asyncio.sleep(0.3)

        assert len(received) == 1

        await transport.stop()

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_universal_wildcard(self):
        transport = InProcessTransport(transport_id="test")
        await transport.start()

        received = []

        async def handler(msg: Message):
            received.append(msg)

        await transport.subscribe("*", handler)

        msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="anything.goes",
            payload={},
        )
        await transport.send("anything.goes", msg)
        await asyncio.sleep(0.3)

        assert len(received) == 1

        await transport.stop()

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_unsubscribe(self):
        transport = InProcessTransport(transport_id="test")
        await transport.start()

        received = []

        async def handler(msg: Message):
            received.append(msg)

        sub = await transport.subscribe("test.topic", handler)
        await transport.unsubscribe(sub)

        msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="test.topic",
            payload={},
        )
        await transport.send("test.topic", msg)
        await asyncio.sleep(0.3)

        assert len(received) == 0

        await transport.stop()

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_tenant_isolation(self):
        transport = InProcessTransport(transport_id="test")
        await transport.start()

        received = []

        async def handler(msg: Message):
            received.append(msg)

        # Subscribe only for tenant_1
        await transport.subscribe("test.topic", handler, tenant_id="tenant_1")

        # Send message from tenant_2
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="test.topic",
            tenant_id="tenant_2",
            payload={},
        )
        await transport.send("test.topic", msg)
        await asyncio.sleep(0.3)

        # Should NOT be received — wrong tenant
        assert len(received) == 0

        # Send from tenant_1
        msg2 = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="test.topic",
            tenant_id="tenant_1",
            payload={},
        )
        await transport.send("test.topic", msg2)
        await asyncio.sleep(0.3)

        assert len(received) == 1

        await transport.stop()

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_priority_ordering(self):
        transport = InProcessTransport(transport_id="test", queue_size=100)
        # Don't start dispatch loop — we'll manually verify queue priority

        # Enqueue low, then high priority
        low = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="test",
            payload={"priority": "low"},
            priority=Priority.LOW,
        )
        high = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="test",
            payload={"priority": "high"},
            priority=Priority.HIGH,
        )

        transport._running = True  # Allow send without full start
        await transport.send("test", low)
        await transport.send("test", high)

        # High priority should come first from queue (lower priority_key value)
        first = await transport._queue.get()
        second = await transport._queue.get()

        # First should have lower priority_key (higher actual priority)
        assert first[0] <= second[0]

        transport._running = False

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_send_request_response(self):
        transport = InProcessTransport(transport_id="test")
        await transport.start()

        async def echo_handler(msg: Message):
            return Message(
                type=MessageType.RESPONSE,
                source_node_id="responder",
                topic="response.topic",
                payload={"echo": msg.payload.get("question")},
                correlation_id=msg.id,
            )

        await transport.subscribe("query.topic", echo_handler)

        request = Message(
            type=MessageType.QUERY,
            source_node_id="requester",
            topic="query.topic",
            payload={"question": "What is 1+1?"},
        )

        response = await transport.send_request("query.topic", request, timeout=5.0)
        assert response.payload["echo"] == "What is 1+1?"

        await transport.stop()

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_send_request_timeout(self):
        transport = InProcessTransport(transport_id="test")
        await transport.start()

        # No handler subscribed — request should timeout
        request = Message(
            type=MessageType.QUERY,
            source_node_id="requester",
            topic="no.handler",
            payload={},
        )

        with pytest.raises(TimeoutError):
            await transport.send_request("no.handler", request, timeout=0.5)

        await transport.stop()

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_multiple_subscribers(self):
        transport = InProcessTransport(transport_id="test")
        await transport.start()

        received_a = []
        received_b = []

        async def handler_a(msg: Message):
            received_a.append(msg)

        async def handler_b(msg: Message):
            received_b.append(msg)

        await transport.subscribe("broadcast", handler_a)
        await transport.subscribe("broadcast", handler_b)

        msg = Message(
            type=MessageType.EVENT,
            source_node_id="pub",
            topic="broadcast",
            payload={"data": "shared"},
        )
        await transport.send("broadcast", msg)
        await asyncio.sleep(0.3)

        assert len(received_a) == 1
        assert len(received_b) == 1

        await transport.stop()

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_has_subscribers(self):
        transport = InProcessTransport(transport_id="test")
        await transport.start()

        assert not transport.has_subscribers("test.topic")

        await transport.subscribe("test.topic", lambda m: None)
        assert transport.has_subscribers("test.topic")

        await transport.stop()

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_metrics_tracking(self):
        transport = InProcessTransport(transport_id="test")
        await transport.start()

        async def noop(msg: Message):
            pass

        await transport.subscribe("test", noop)

        msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="test",
            payload={},
        )
        await transport.send("test", msg)
        await asyncio.sleep(0.3)

        assert transport.metrics.messages_sent >= 1
        assert transport.metrics.messages_received >= 1

        await transport.stop()

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_dropped_when_stopped(self):
        transport = InProcessTransport(transport_id="test")
        # Not started — should drop

        msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="test",
            payload={},
        )
        await transport.send("test", msg)
        assert transport.metrics.messages_dropped >= 1
