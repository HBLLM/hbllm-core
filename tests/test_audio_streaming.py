"""
Tests for AudioInputNode streaming support.

Tests hex chunk buffering, silence timeout flush, max buffer flush,
is_final flush, and error handling — all WITHOUT loading Whisper.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.perception.audio_in_node import (
    AudioInputNode,
)


@pytest.fixture
async def audio_node():
    """Set up AudioInputNode with bus (no model loaded)."""
    bus = InProcessBus()
    await bus.start()

    node = AudioInputNode(node_id="audio_test")
    await node.start(bus)

    yield node, bus

    await bus.stop()


@pytest.mark.asyncio
async def test_stream_buffer_initialization(audio_node):
    """Sending a stream chunk should create a session buffer."""
    node, bus = audio_node

    chunk_msg = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        session_id="sess_1",
        topic="sensory.audio.stream",
        payload={"chunk": "0102030405", "sample_rate": 16000},
    )
    await bus.publish("sensory.audio.stream", chunk_msg)
    await asyncio.sleep(0.3)

    assert "sess_1" in node._stream_buffers
    buf = node._stream_buffers["sess_1"]
    assert len(buf["chunks"]) == 1
    assert buf["chunks"][0] == bytes.fromhex("0102030405")
    assert buf["sample_rate"] == 16000


@pytest.mark.asyncio
async def test_stream_multiple_chunks(audio_node):
    """Multiple chunks should accumulate in the same buffer."""
    node, bus = audio_node

    for i in range(5):
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            session_id="sess_2",
            topic="sensory.audio.stream",
            payload={"chunk": f"0{i}" * 4},
        )
        await bus.publish("sensory.audio.stream", msg)
        await asyncio.sleep(0.05)

    await asyncio.sleep(0.3)

    assert "sess_2" in node._stream_buffers
    assert len(node._stream_buffers["sess_2"]["chunks"]) == 5


@pytest.mark.asyncio
async def test_stream_is_final_flushes(audio_node):
    """is_final=True should flush the buffer."""
    node, bus = audio_node

    # Mock _flush_stream_buffer to avoid loading Whisper
    node._flush_stream_buffer = AsyncMock()

    # Send chunk
    await bus.publish(
        "sensory.audio.stream",
        Message(
            type=MessageType.EVENT,
            source_node_id="test",
            session_id="sess_3",
            topic="sensory.audio.stream",
            payload={"chunk": "aabbccdd"},
        ),
    )
    await asyncio.sleep(0.3)

    # Send is_final
    await bus.publish(
        "sensory.audio.stream",
        Message(
            type=MessageType.EVENT,
            source_node_id="test",
            session_id="sess_3",
            topic="sensory.audio.stream",
            payload={"is_final": True},
        ),
    )
    await asyncio.sleep(0.3)

    node._flush_stream_buffer.assert_called_once()
    call_args = node._flush_stream_buffer.call_args
    assert call_args[0][0] == "sess_3"


@pytest.mark.asyncio
async def test_stream_invalid_hex_returns_error(audio_node):
    """Invalid hex data should return an error message."""
    node, bus = audio_node

    msg = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        session_id="sess_4",
        topic="sensory.audio.stream",
        payload={"chunk": "not_valid_hex"},
    )

    # Call handler directly to check return value
    result = await node.handle_stream(msg)

    assert result is not None
    assert result.type == MessageType.ERROR


@pytest.mark.asyncio
async def test_stream_empty_no_action(audio_node):
    """Empty chunk with no is_final should be ignored."""
    node, bus = audio_node

    msg = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        session_id="sess_5",
        topic="sensory.audio.stream",
        payload={"chunk": ""},
    )
    result = await node.handle_stream(msg)

    assert result is None
    assert "sess_5" not in node._stream_buffers


@pytest.mark.asyncio
async def test_stream_separate_sessions(audio_node):
    """Different sessions should have independent buffers."""
    node, bus = audio_node

    hex_data = {"s_a": "aabb", "s_b": "ccdd"}
    for sess in ["s_a", "s_b"]:
        await bus.publish(
            "sensory.audio.stream",
            Message(
                type=MessageType.EVENT,
                source_node_id="test",
                session_id=sess,
                topic="sensory.audio.stream",
                payload={"chunk": hex_data[sess]},
            ),
        )

    await asyncio.sleep(0.3)

    assert "s_a" in node._stream_buffers
    assert "s_b" in node._stream_buffers
    assert node._stream_buffers["s_a"]["chunks"][0] == bytes.fromhex("aabb")
    assert node._stream_buffers["s_b"]["chunks"][0] == bytes.fromhex("ccdd")
