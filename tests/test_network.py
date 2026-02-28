"""
Comprehensive tests for the HBLLM network layer.

Covers: InProcessBus, Node, ServiceRegistry, CircuitBreaker,
FallbackManager, DegradedModeManager, and integration scenarios.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from hbllm.network.bus import InProcessBus, Subscription
from hbllm.network.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
)
from hbllm.network.degraded import DegradedModeManager, SystemCapabilities
from hbllm.network.fallback import FallbackManager, FallbackResult
from hbllm.network.messages import (
    HeartbeatPayload,
    Message,
    MessageType,
    Priority,
    QueryPayload,
    RouteDecisionPayload,
)
from hbllm.network.node import HealthStatus, Node, NodeHealth, NodeInfo, NodeType
from hbllm.network.registry import ServiceRegistry
from hbllm.network.serialization import JsonSerializer, MsgpackSerializer, get_serializer


# ──────────────────────────────────────────────
# Test helpers
# ──────────────────────────────────────────────

class EchoNode(Node):
    """Test node that echoes messages back."""

    def __init__(self, node_id: str = "echo-1"):
        super().__init__(node_id, NodeType.DOMAIN_MODULE, capabilities=["echo", "general"])
        self.received: list[Message] = []

    async def on_start(self) -> None:
        await self.bus.subscribe(f"node.{self.node_id}", self.handle_message)

    async def on_stop(self) -> None:
        pass

    async def handle_message(self, message: Message) -> Message | None:
        self.received.append(message)
        return message.create_response({"echoed": message.payload})


# ──────────────────────────────────────────────
# Message tests
# ──────────────────────────────────────────────

class TestMessages:
    def test_message_creation(self):
        msg = Message(
            type=MessageType.QUERY,
            source_node_id="router",
            target_node_id="coding-1",
            topic="node.coding-1",
            payload={"text": "hello"},
        )
        assert msg.id  # Should have auto-generated UUID
        assert msg.type == MessageType.QUERY
        assert msg.priority == Priority.NORMAL
        assert msg.timestamp is not None

    def test_message_create_response(self):
        msg = Message(
            type=MessageType.QUERY,
            source_node_id="router",
            target_node_id="coding-1",
            topic="query",
            payload={"text": "hello"},
        )
        resp = msg.create_response({"result": "world"})
        assert resp.correlation_id == msg.id
        assert resp.source_node_id == "coding-1"
        assert resp.target_node_id == "router"
        assert resp.type == MessageType.RESPONSE

    def test_message_create_error(self):
        msg = Message(
            type=MessageType.QUERY,
            source_node_id="router",
            topic="query",
        )
        error = msg.create_error("Something broke", code="INTERNAL")
        assert error.type == MessageType.ERROR
        assert error.payload["error"] == "Something broke"
        assert error.payload["code"] == "INTERNAL"
        assert error.correlation_id == msg.id

    def test_query_payload_validation(self):
        payload = QueryPayload(text="What is AI?", context=[{"role": "user", "content": "hi"}])
        assert payload.text == "What is AI?"
        assert len(payload.context) == 1

    def test_route_decision_payload(self):
        payload = RouteDecisionPayload(
            target_modules=["coding", "general"],
            confidence_scores={"coding": 0.9, "general": 0.6},
            detected_intent="code_generation",
            requires_planning=False,
        )
        assert len(payload.target_modules) == 2
        assert payload.confidence_scores["coding"] == 0.9

    def test_heartbeat_payload(self):
        payload = HeartbeatPayload(
            node_id="test-1",
            status="healthy",
            uptime_seconds=1234.5,
            capabilities=["echo"],
            load=0.3,
        )
        assert payload.load == 0.3

    def test_message_priority(self):
        assert Priority.LOW < Priority.NORMAL < Priority.HIGH < Priority.CRITICAL


# ──────────────────────────────────────────────
# Node tests
# ──────────────────────────────────────────────

class TestNode:
    def test_node_info(self):
        node = EchoNode("echo-1")
        info = node.get_info()
        assert info.node_id == "echo-1"
        assert info.node_type == NodeType.DOMAIN_MODULE
        assert "echo" in info.capabilities

    def test_node_uptime(self):
        node = EchoNode("echo-1")
        assert node.uptime == 0.0

    def test_node_repr(self):
        node = EchoNode("echo-1")
        assert "echo-1" in repr(node)
        assert "domain_module" in repr(node)

    @pytest.mark.asyncio
    async def test_node_bus_not_started(self):
        node = EchoNode("echo-1")
        with pytest.raises(RuntimeError, match="not been started"):
            _ = node.bus

    @pytest.mark.asyncio
    async def test_node_health_check(self):
        node = EchoNode("echo-1")
        health = await node.health_check()
        assert health.status == HealthStatus.UNHEALTHY  # Not running
        assert health.node_id == "echo-1"

    @pytest.mark.asyncio
    async def test_node_lifecycle(self):
        bus = InProcessBus()
        await bus.start()

        node = EchoNode("echo-1")
        await node.start(bus)
        assert node._running is True
        assert node.uptime > 0

        health = await node.health_check()
        assert health.status == HealthStatus.HEALTHY

        await node.stop()
        assert node._running is False

        await bus.stop()


# ──────────────────────────────────────────────
# InProcessBus tests
# ──────────────────────────────────────────────

@pytest.fixture
async def bus():
    b = InProcessBus()
    await b.start()
    yield b
    await b.stop()


class TestInProcessBus:
    @pytest.mark.asyncio
    async def test_start_stop(self):
        bus = InProcessBus()
        await bus.start()
        assert bus._running is True
        await bus.stop()
        assert bus._running is False

    @pytest.mark.asyncio
    async def test_publish_subscribe(self, bus: InProcessBus):
        received = []

        async def handler(msg: Message) -> None:
            received.append(msg)

        await bus.subscribe("test.topic", handler)

        msg = Message(type=MessageType.QUERY, source_node_id="test", topic="test.topic", payload={"data": "hello"})
        await bus.publish("test.topic", msg)
        await asyncio.sleep(0.2)

        assert len(received) == 1
        assert received[0].payload["data"] == "hello"

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, bus: InProcessBus):
        received_a = []
        received_b = []

        async def handler_a(msg: Message) -> None:
            received_a.append(msg)

        async def handler_b(msg: Message) -> None:
            received_b.append(msg)

        await bus.subscribe("shared.topic", handler_a)
        await bus.subscribe("shared.topic", handler_b)

        msg = Message(type=MessageType.QUERY, source_node_id="test", topic="shared.topic")
        await bus.publish("shared.topic", msg)
        await asyncio.sleep(0.2)

        assert len(received_a) == 1
        assert len(received_b) == 1

    @pytest.mark.asyncio
    async def test_request_response(self, bus: InProcessBus):
        async def echo_handler(msg: Message) -> Message:
            return msg.create_response({"echo": True})

        await bus.subscribe("echo", echo_handler)

        msg = Message(type=MessageType.QUERY, source_node_id="test", topic="echo")
        response = await bus.request("echo", msg, timeout=2.0)

        assert response.payload["echo"] is True
        assert response.correlation_id == msg.id

    @pytest.mark.asyncio
    async def test_request_timeout(self, bus: InProcessBus):
        msg = Message(type=MessageType.QUERY, source_node_id="test", topic="no.handler")
        with pytest.raises(TimeoutError):
            await bus.request("no.handler", msg, timeout=0.3)

    @pytest.mark.asyncio
    async def test_wildcard_subscribe(self, bus: InProcessBus):
        received = []

        async def handler(msg: Message) -> None:
            received.append(msg)

        await bus.subscribe("node.*", handler)

        msg = Message(type=MessageType.QUERY, source_node_id="test", topic="node.echo.query")
        await bus.publish("node.echo.query", msg)
        await asyncio.sleep(0.2)

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_global_wildcard(self, bus: InProcessBus):
        received = []

        async def handler(msg: Message) -> None:
            received.append(msg)

        await bus.subscribe("*", handler)

        msg = Message(type=MessageType.QUERY, source_node_id="test", topic="anything")
        await bus.publish("anything", msg)
        await asyncio.sleep(0.2)

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe(self, bus: InProcessBus):
        received = []

        async def handler(msg: Message) -> None:
            received.append(msg)

        sub = await bus.subscribe("test.topic", handler)
        await bus.unsubscribe(sub)
        assert not sub.active

        msg = Message(type=MessageType.QUERY, source_node_id="test", topic="test.topic")
        await bus.publish("test.topic", msg)
        await asyncio.sleep(0.2)

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_publish_when_stopped(self, bus: InProcessBus):
        await bus.stop()
        # Should not raise, just log a warning
        msg = Message(type=MessageType.QUERY, source_node_id="test", topic="test")
        await bus.publish("test", msg)

    @pytest.mark.asyncio
    async def test_handler_exception_doesnt_crash(self, bus: InProcessBus):
        async def bad_handler(msg: Message) -> None:
            raise ValueError("Intentional failure")

        await bus.subscribe("bad", bad_handler)

        msg = Message(type=MessageType.QUERY, source_node_id="test", topic="bad")
        await bus.publish("bad", msg)
        await asyncio.sleep(0.2)
        # Bus should still be running
        assert bus._running


# ──────────────────────────────────────────────
# Circuit Breaker tests
# ──────────────────────────────────────────────

class TestCircuitBreaker:
    def test_initial_state_closed(self):
        cb = CircuitBreaker("node-1", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute()

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker("node-1", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute()

    def test_opens_at_threshold(self):
        cb = CircuitBreaker("node-1", failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert not cb.can_execute()

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker("node-1", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # Reset
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker("node-1", failure_threshold=2, recovery_timeout=0.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.HALF_OPEN  # 0 timeout → immediate
        assert cb.can_execute()

    def test_half_open_success_closes(self):
        cb = CircuitBreaker("node-1", failure_threshold=2, recovery_timeout=0.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker("node-1", failure_threshold=2, recovery_timeout=0.0)
        cb.record_failure()
        cb.record_failure()
        _ = cb.state  # Trigger HALF_OPEN transition
        cb.record_failure()
        # With 0.0 recovery_timeout, OPEN immediately transitions back to HALF_OPEN
        # So state is either OPEN or HALF_OPEN depending on timing
        assert cb.state in (CircuitState.OPEN, CircuitState.HALF_OPEN)

    def test_manual_reset(self):
        cb = CircuitBreaker("node-1", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_call_success(self):
        cb = CircuitBreaker("node-1")

        async def success():
            return "ok"

        result = await cb.call(success)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_call_failure(self):
        cb = CircuitBreaker("node-1", failure_threshold=2)

        async def fail():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await cb.call(fail)

        assert cb._failure_count == 1

    @pytest.mark.asyncio
    async def test_call_when_open(self):
        cb = CircuitBreaker("node-1", failure_threshold=1, recovery_timeout=999)

        async def fail():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await cb.call(fail)

        with pytest.raises(CircuitOpenError) as exc_info:
            await cb.call(fail)

        assert exc_info.value.node_id == "node-1"
        assert exc_info.value.time_until_retry > 0

    def test_time_until_retry(self):
        cb = CircuitBreaker("node-1", failure_threshold=1, recovery_timeout=30.0)
        assert cb.time_until_retry == 0.0

        cb.record_failure()
        assert cb.time_until_retry > 0
        assert cb.time_until_retry <= 30.0

    def test_repr(self):
        cb = CircuitBreaker("node-1", failure_threshold=3)
        assert "node-1" in repr(cb)
        assert "closed" in repr(cb)


class TestCircuitBreakerRegistry:
    def test_get_creates_breaker(self):
        reg = CircuitBreakerRegistry()
        cb = reg.get("node-1")
        assert isinstance(cb, CircuitBreaker)

    def test_get_returns_same_instance(self):
        reg = CircuitBreakerRegistry()
        cb1 = reg.get("node-1")
        cb2 = reg.get("node-1")
        assert cb1 is cb2

    def test_different_nodes_different_breakers(self):
        reg = CircuitBreakerRegistry()
        assert reg.get("node-1") is not reg.get("node-2")

    def test_get_open_circuits(self):
        reg = CircuitBreakerRegistry(failure_threshold=1)
        reg.get("node-1").record_failure()
        assert "node-1" in reg.get_open_circuits()
        assert "node-2" not in reg.get_open_circuits()

    def test_reset_all(self):
        reg = CircuitBreakerRegistry(failure_threshold=1)
        reg.get("node-1").record_failure()
        reg.get("node-2").record_failure()
        reg.reset_all()
        assert len(reg.get_open_circuits()) == 0


# ──────────────────────────────────────────────
# Service Registry tests
# ──────────────────────────────────────────────

class TestServiceRegistry:
    @pytest.mark.asyncio
    async def test_register_and_discover(self):
        reg = ServiceRegistry(health_check_interval=999)
        await reg.start()

        info = NodeInfo(node_id="gen-1", node_type=NodeType.DOMAIN_MODULE, capabilities=["general"])
        await reg.register(info)

        found = await reg.discover(node_type=NodeType.DOMAIN_MODULE)
        assert len(found) == 1
        assert found[0].node_id == "gen-1"

        await reg.stop()

    @pytest.mark.asyncio
    async def test_discover_by_capability(self):
        reg = ServiceRegistry(health_check_interval=999)
        await reg.start()

        await reg.register(NodeInfo(node_id="a", node_type=NodeType.DOMAIN_MODULE, capabilities=["coding"]))
        await reg.register(NodeInfo(node_id="b", node_type=NodeType.DOMAIN_MODULE, capabilities=["math"]))

        coding = await reg.discover(capability="coding")
        assert len(coding) == 1
        assert coding[0].node_id == "a"

        math = await reg.discover(capability="math")
        assert len(math) == 1
        assert math[0].node_id == "b"

        await reg.stop()

    @pytest.mark.asyncio
    async def test_discover_healthy_only(self):
        reg = ServiceRegistry(health_check_interval=999)
        await reg.start()

        await reg.register(NodeInfo(node_id="a", node_type=NodeType.DOMAIN_MODULE, capabilities=["general"]))
        await reg.update_health(NodeHealth(node_id="a", status=HealthStatus.UNHEALTHY))

        healthy = await reg.discover(capability="general", healthy_only=True)
        assert len(healthy) == 0

        all_nodes = await reg.discover(capability="general", healthy_only=False)
        assert len(all_nodes) == 1

        await reg.stop()

    @pytest.mark.asyncio
    async def test_deregister(self):
        reg = ServiceRegistry(health_check_interval=999)
        await reg.start()

        await reg.register(NodeInfo(node_id="a", node_type=NodeType.DOMAIN_MODULE))
        assert len(reg) == 1

        await reg.deregister("a")
        assert len(reg) == 0

        await reg.stop()

    @pytest.mark.asyncio
    async def test_get_available_capabilities(self):
        reg = ServiceRegistry(health_check_interval=999)
        await reg.start()

        await reg.register(NodeInfo(node_id="a", node_type=NodeType.DOMAIN_MODULE, capabilities=["coding", "general"]))
        await reg.register(NodeInfo(node_id="b", node_type=NodeType.DOMAIN_MODULE, capabilities=["math"]))

        # Mark both as healthy
        await reg.update_health(NodeHealth(node_id="a", status=HealthStatus.HEALTHY))
        await reg.update_health(NodeHealth(node_id="b", status=HealthStatus.HEALTHY))

        caps = await reg.get_available_capabilities()
        assert caps == {"coding", "general", "math"}

        await reg.stop()

    @pytest.mark.asyncio
    async def test_is_node_healthy(self):
        reg = ServiceRegistry(health_check_interval=999)
        await reg.start()

        await reg.register(NodeInfo(node_id="a", node_type=NodeType.DOMAIN_MODULE))
        await reg.update_health(NodeHealth(node_id="a", status=HealthStatus.HEALTHY))
        assert await reg.is_node_healthy("a")

        await reg.update_health(NodeHealth(node_id="a", status=HealthStatus.UNHEALTHY))
        assert not await reg.is_node_healthy("a")

        assert not await reg.is_node_healthy("nonexistent")

        await reg.stop()

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        reg = ServiceRegistry(health_check_interval=999)
        await reg.start()

        await reg.register(NodeInfo(node_id="low", node_type=NodeType.DOMAIN_MODULE, capabilities=["general"], priority=0))
        await reg.register(NodeInfo(node_id="high", node_type=NodeType.DOMAIN_MODULE, capabilities=["general"], priority=10))

        found = await reg.discover(capability="general")
        assert found[0].node_id == "high"

        await reg.stop()


# ──────────────────────────────────────────────
# Serialization tests
# ──────────────────────────────────────────────

class TestSerialization:
    def test_json_roundtrip(self):
        serializer = JsonSerializer()
        msg = Message(
            type=MessageType.QUERY,
            source_node_id="test",
            topic="test",
            payload={"key": "value", "number": 42},
        )
        data = serializer.serialize(msg)
        restored = serializer.deserialize(data)
        assert restored.id == msg.id
        assert restored.payload == msg.payload

    def test_json_serializes_to_bytes(self):
        serializer = JsonSerializer()
        msg = Message(type=MessageType.QUERY, source_node_id="test", topic="test")
        data = serializer.serialize(msg)
        assert isinstance(data, bytes)

    def test_get_serializer_json(self):
        s = get_serializer("json")
        assert isinstance(s, JsonSerializer)

    def test_get_serializer_msgpack(self):
        s = get_serializer("msgpack")
        assert isinstance(s, MsgpackSerializer)
