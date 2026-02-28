"""Tests for the BusMetrics observability instrumentation."""

import pytest
import asyncio

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.network.tracing import BusMetrics


@pytest.mark.asyncio
async def test_bus_metrics_publish_counter():
    """Verify publish counter increments on each message."""
    bus = InProcessBus()
    await bus.start()
    
    assert bus.metrics.messages_published == 0
    
    msg = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        topic="test.topic",
        payload={"data": "hello"},
    )
    await bus.publish("test.topic", msg)
    await asyncio.sleep(0.05)
    
    assert bus.metrics.messages_published == 1
    
    await bus.stop()


@pytest.mark.asyncio
async def test_bus_metrics_delivery_and_latency():
    """Verify delivery counter and latency recording when a handler processes a message."""
    bus = InProcessBus()
    await bus.start()
    
    received = []
    
    async def handler(msg: Message):
        received.append(msg)
        return None
    
    await bus.subscribe("test.topic", handler)
    
    msg = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        topic="test.topic",
        payload={"data": "hello"},
    )
    await bus.publish("test.topic", msg)
    await asyncio.sleep(0.15)
    
    assert len(received) == 1
    assert bus.metrics.messages_delivered >= 1
    assert bus.metrics.avg_latency_ms > 0
    
    await bus.stop()


@pytest.mark.asyncio
async def test_bus_metrics_error_counter():
    """Verify error counter increments when a handler raises."""
    bus = InProcessBus()
    await bus.start()
    
    async def bad_handler(msg: Message):
        raise ValueError("Intentional test error")
    
    await bus.subscribe("test.error", bad_handler)
    
    msg = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        topic="test.error",
        payload={},
    )
    await bus.publish("test.error", msg)
    await asyncio.sleep(0.15)
    
    assert bus.metrics.handler_errors >= 1
    
    await bus.stop()


@pytest.mark.asyncio
async def test_bus_metrics_subscribe_unsubscribe():
    """Verify subscription counter tracks active subscriptions."""
    bus = InProcessBus()
    await bus.start()
    
    assert bus.metrics.active_subscriptions == 0
    
    async def handler(msg: Message):
        return None
    
    sub = await bus.subscribe("test.topic", handler)
    assert bus.metrics.active_subscriptions == 1
    
    await bus.unsubscribe(sub)
    assert bus.metrics.active_subscriptions == 0
    
    await bus.stop()


@pytest.mark.asyncio 
async def test_bus_metrics_drop_counter():
    """Verify drop counter increments when bus is stopped."""
    bus = InProcessBus()
    # Don't start the bus
    
    msg = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        topic="test.topic",
        payload={},
    )
    await bus.publish("test.topic", msg)
    
    assert bus.metrics.messages_dropped == 1


def test_bus_metrics_snapshot():
    """Verify BusMetrics.snapshot() returns expected keys."""
    m = BusMetrics()
    m.record_publish("t")
    m.record_publish("t")
    m.record_delivery("t", 5.0)
    m.record_error("t")
    m.record_subscribe()
    
    snap = m.snapshot()
    assert snap["messages_published"] == 2
    assert snap["messages_delivered"] == 1
    assert snap["handler_errors"] == 1
    assert snap["active_subscriptions"] == 1
    assert snap["avg_latency_ms"] == 5.0
    assert "p99_latency_ms" in snap


def test_bus_metrics_p99_latency():
    """Verify p99 latency calculation."""
    m = BusMetrics()
    # Add 100 samples
    for i in range(100):
        m.record_delivery("t", float(i))
    
    # p99 should be around 99
    assert m.p99_latency_ms >= 98.0
