"""Resilience tests — failure injection, concurrent load, subscriber isolation."""

import pytest
import asyncio

from hbllm.network.bus import InProcessBus
from hbllm.network.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry, CircuitOpenError
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType


# ── Helper Nodes ─────────────────────────────────────────────────────────────

class FailingNode(Node):
    """Node that raises on every message after start."""

    def __init__(self, node_id: str):
        super().__init__(node_id, NodeType.DOMAIN_MODULE, capabilities=["fail"])
        self.handled = 0

    async def on_start(self):
        await self.bus.subscribe("fail.test", self.handle_message)

    async def on_stop(self):
        pass

    async def handle_message(self, message):
        self.handled += 1
        raise RuntimeError("Deliberate failure")


class CountingNode(Node):
    """Node that counts received messages."""

    def __init__(self, node_id: str, topic: str = "count.test"):
        super().__init__(node_id, NodeType.DOMAIN_MODULE, capabilities=["count"])
        self.topic = topic
        self.received: list[Message] = []

    async def on_start(self):
        await self.bus.subscribe(self.topic, self.handle_message)

    async def on_stop(self):
        pass

    async def handle_message(self, message):
        self.received.append(message)
        return None


# ── Bus Subscriber Isolation ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bad_subscriber_does_not_crash_bus():
    """One failing subscriber should not prevent others from receiving messages."""
    bus = InProcessBus()
    await bus.start()

    good_messages = []
    bad_calls = []

    async def good_handler(msg):
        good_messages.append(msg)

    async def bad_handler(msg):
        bad_calls.append(msg)
        raise ValueError("I always fail!")

    await bus.subscribe("test.topic", bad_handler)
    await bus.subscribe("test.topic", good_handler)

    msg = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        topic="test.topic",
        payload={"data": "hello"},
    )
    await bus.publish("test.topic", msg)
    await asyncio.sleep(0.1)

    # Good handler should still receive the message
    assert len(good_messages) >= 1

    await bus.stop()


# ── Node Restart After Failure ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_node_restart_after_failure():
    """Nodes can be stopped and restarted after a failure."""
    bus = InProcessBus()
    await bus.start()

    counter = CountingNode("counter_1")
    await counter.start(bus)

    msg = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        topic="count.test",
        payload={"seq": 1},
    )
    await bus.publish("count.test", msg)
    await asyncio.sleep(0.1)
    assert len(counter.received) == 1

    # Simulate failure → restart cycle
    await counter.stop()
    await counter.start(bus)

    await bus.publish("count.test", msg)
    await asyncio.sleep(0.1)
    assert len(counter.received) >= 2

    await counter.stop()
    await bus.stop()


# ── Circuit Breaker Under Load ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_circuit_breaker_opens_under_failures():
    """Circuit breaker opens after threshold failures, blocking further calls."""
    cb = CircuitBreaker("load_node", failure_threshold=3, recovery_timeout=1.0)

    async def failing():
        raise RuntimeError("fail")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            await cb.call(failing)

    # Now circuit should be open
    with pytest.raises(CircuitOpenError):
        await cb.call(failing)


@pytest.mark.asyncio
async def test_circuit_breaker_recovers():
    """Circuit breaker transitions back to CLOSED after successful test."""
    cb = CircuitBreaker("recover_node", failure_threshold=1, recovery_timeout=0.01)

    async def failing():
        raise RuntimeError("fail")

    async def ok():
        return "recovered"

    with pytest.raises(RuntimeError):
        await cb.call(failing)

    await asyncio.sleep(0.02)  # Wait for HALF_OPEN

    result = await cb.call(ok)
    assert result == "recovered"
    assert cb.can_execute()


# ── Concurrent Message Flood ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_concurrent_message_flood():
    """Bus handles 1000 concurrent messages without dropping any."""
    bus = InProcessBus()
    await bus.start()

    received = []

    async def handler(msg):
        received.append(msg)

    await bus.subscribe("flood.test", handler)

    tasks = []
    for i in range(1000):
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="flood",
            topic="flood.test",
            payload={"seq": i},
        )
        tasks.append(bus.publish("flood.test", msg))

    await asyncio.gather(*tasks)
    await asyncio.sleep(0.5)

    assert len(received) == 1000

    await bus.stop()


# ── Multiple Nodes Concurrent ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_multiple_nodes_concurrent():
    """Multiple nodes can operate on the same bus without interference."""
    bus = InProcessBus()
    await bus.start()

    n1 = CountingNode("n1", topic="multi.a")
    n2 = CountingNode("n2", topic="multi.b")
    await n1.start(bus)
    await n2.start(bus)

    for i in range(10):
        await bus.publish("multi.a", Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="multi.a",
            payload={"i": i},
        ))
        await bus.publish("multi.b", Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="multi.b",
            payload={"i": i},
        ))

    await asyncio.sleep(0.3)

    assert len(n1.received) == 10
    assert len(n2.received) == 10

    await n1.stop()
    await n2.stop()
    await bus.stop()


# ── Registry Failure Tracking ────────────────────────────────────────────────

def test_registry_tracks_multiple_nodes():
    """Registry correctly tracks failure states across many nodes."""
    reg = CircuitBreakerRegistry(failure_threshold=2)

    healthy = [f"healthy_{i}" for i in range(5)]
    unhealthy = [f"unhealthy_{i}" for i in range(3)]

    for n in healthy:
        reg.get(n)  # Creates in CLOSED state

    for n in unhealthy:
        cb = reg.get(n)
        cb.record_failure()
        cb.record_failure()

    open_circuits = reg.get_open_circuits()
    assert len(open_circuits) == 3
    for n in unhealthy:
        assert n in open_circuits


# ── Memory Under Concurrent Access ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_episodic_concurrent_writes():
    """EpisodicMemory handles concurrent writes without corruption."""
    import tempfile
    from pathlib import Path
    from hbllm.memory.episodic import EpisodicMemory

    with tempfile.TemporaryDirectory() as tmp:
        mem = EpisodicMemory(db_path=Path(tmp) / "concurrent.db")

        async def writer(session: str, count: int):
            for i in range(count):
                mem.store_turn(session, "user", f"msg_{i}")

        await asyncio.gather(
            writer("s1", 50),
            writer("s2", 50),
            writer("s3", 50),
        )

        assert mem.get_turn_count() == 150
        assert mem.get_session_count() == 3
