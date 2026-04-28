"""Tests for core CognitiveStream."""

import asyncio

import pytest

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.serving.streaming import CognitiveStream


@pytest.fixture
def bus():
    return InProcessBus()


@pytest.mark.asyncio
async def test_stream_lifecycle(bus):
    """Test basic stream start/stop."""
    await bus.start()
    stream = CognitiveStream(bus, correlation_id="test-123", timeout=2.0)
    await stream.start()
    assert len(stream._subscriptions) == 4
    await stream.stop()
    assert len(stream._subscriptions) == 0
    await bus.stop()


@pytest.mark.asyncio
async def test_stream_receives_chunks(bus):
    """Test that stream receives token chunks."""
    await bus.start()
    stream = CognitiveStream(bus, correlation_id="c1", timeout=2.0)
    await stream.start()

    # Publish a chunk
    await bus.publish(
        "sensory.stream.chunk",
        Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="sensory.stream.chunk",
            correlation_id="c1",
            payload={"text": "Hello"},
        ),
    )
    # Publish end sentinel
    await bus.publish(
        "sensory.stream.end",
        Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="sensory.stream.end",
            correlation_id="c1",
            payload={"text": " world"},
        ),
    )

    # Collect chunks
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0]["type"] == "token"
    assert chunks[0]["text"] == "Hello"
    assert chunks[1]["text"] == " world"

    await bus.stop()


@pytest.mark.asyncio
async def test_stream_ignores_other_correlations(bus):
    """Test stream ignores messages with different correlation IDs."""
    await bus.start()
    stream = CognitiveStream(bus, correlation_id="mine", timeout=1.0)
    await stream.start()

    # Different correlation id
    await bus.publish(
        "sensory.stream.chunk",
        Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="sensory.stream.chunk",
            correlation_id="other",
            payload={"text": "ignored"},
        ),
    )

    # End for mine
    await bus.publish(
        "sensory.stream.end",
        Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="sensory.stream.end",
            correlation_id="mine",
            payload={},
        ),
    )

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) == 0
    await bus.stop()


@pytest.mark.asyncio
async def test_stream_thought_type(bus):
    """Test stream captures internal thoughts."""
    await bus.start()
    stream = CognitiveStream(bus, correlation_id="t1", timeout=2.0)
    await stream.start()

    await bus.publish(
        "system.thought",
        Message(
            type=MessageType.EVENT,
            source_node_id="planner",
            topic="system.thought",
            correlation_id="t1",
            payload={"text": "thinking..."},
        ),
    )
    await bus.publish(
        "sensory.stream.end",
        Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="sensory.stream.end",
            correlation_id="t1",
            payload={},
        ),
    )

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert any(c["type"] == "thought" for c in chunks)
    await bus.stop()


@pytest.mark.asyncio
async def test_stream_timeout(bus):
    """Test stream handles timeout gracefully."""
    await bus.start()
    stream = CognitiveStream(bus, correlation_id="timeout", timeout=0.1)
    await stream.start()

    # Don't publish anything — should timeout
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) == 0
    await bus.stop()
