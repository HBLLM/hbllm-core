"""
Tests for PhysicalPerceptionStream and CognitiveStream embodied integration.

Validates:
- Physical observation ingestion and bus publishing.
- Embodied thoughts and motor action events.
- Real-time ingestion by CognitiveStream with correlation isolation.
- Graceful termination via sentinel and unsubscribing.
"""

import asyncio

import pytest
import pytest_asyncio

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.perception.physical_perception_stream import PhysicalPerceptionStream
from hbllm.serving.streaming import CognitiveStream


@pytest_asyncio.fixture
async def bus():
    """Create and start an InProcessBus instance."""
    b = InProcessBus()
    await b.start()
    yield b
    await b.stop()


@pytest.mark.asyncio
async def test_embodied_cognitive_stream_flow(bus):
    """Verify CognitiveStream captures embodied observations, thoughts, and actions."""
    corr_id = "test-embodied-session-123"

    stream = CognitiveStream(bus, correlation_id=corr_id, timeout=5.0)
    await stream.start()

    perception_stream = PhysicalPerceptionStream(
        bus, domain="crafter", default_correlation_id=corr_id
    )

    collected_chunks = []

    async def consume_stream():
        async for chunk in stream:
            collected_chunks.append(chunk)

    consumer_task = asyncio.create_task(consume_stream())

    # Ingest mock observation
    mock_obs = {
        "player_pos": [5, 5],
        "inventory": {"wood": 2, "stone": 4},
        "health": 9,
    }
    await perception_stream.ingest_observation(mock_obs, step=1)

    # Ingest embodied deliberative thought
    await perception_stream.emit_thought(
        "Detected stone >= 4 and crafting table adjacent. Planning furnace placement."
    )

    # Ingest embodied motor action
    await perception_stream.emit_action(
        action="PLACE_FURNACE",
        payload={"target_pos": [5, 6], "status": "executed"},
    )

    # End stream with sensory sentinel
    end_msg = Message.model_construct(
        id="sentinel-end",
        type=MessageType.EVENT,
        source_node_id="test.emitter",
        topic="sensory.stream.end",
        payload={"text": "Goal accomplished."},
        correlation_id=corr_id,
    )
    await bus.publish("sensory.stream.end", end_msg)

    await asyncio.wait_for(consumer_task, timeout=5.0)

    # Verify all embodied events arrived in order
    assert len(collected_chunks) == 4

    obs_chunk = collected_chunks[0]
    assert obs_chunk["type"] == "embodied_observation"
    assert obs_chunk["domain"] == "crafter"
    assert obs_chunk["step"] == 1
    assert obs_chunk["data"]["player_pos"] == [5, 5]
    assert obs_chunk["data"]["inventory"]["stone"] == 4

    thought_chunk = collected_chunks[1]
    assert thought_chunk["type"] == "embodied_thought"
    assert "furnace placement" in thought_chunk["text"]

    action_chunk = collected_chunks[2]
    assert action_chunk["type"] == "embodied_action"
    assert action_chunk["action"] == "PLACE_FURNACE"
    assert action_chunk["payload"]["target_pos"] == [5, 6]

    final_chunk = collected_chunks[3]
    assert final_chunk["type"] == "token"
    assert final_chunk["text"] == "Goal accomplished."


@pytest.mark.asyncio
async def test_embodied_cognitive_stream_correlation_isolation(bus):
    """Verify events with a mismatched correlation_id are ignored by CognitiveStream."""
    corr_id_target = "target-session-456"
    corr_id_other = "other-session-789"

    stream = CognitiveStream(bus, correlation_id=corr_id_target, timeout=5.0)
    await stream.start()

    perception_stream_other = PhysicalPerceptionStream(
        bus, domain="overcooked", default_correlation_id=corr_id_other
    )
    perception_stream_target = PhysicalPerceptionStream(
        bus, domain="overcooked", default_correlation_id=corr_id_target
    )

    collected_chunks = []

    async def consume_stream():
        async for chunk in stream:
            collected_chunks.append(chunk)

    consumer_task = asyncio.create_task(consume_stream())

    # Emit from other session (should be filtered out)
    await perception_stream_other.ingest_observation({"pot": "empty"}, step=0)
    await perception_stream_other.emit_thought("Irrelevant thought")

    # Emit from target session
    await perception_stream_target.ingest_observation({"pot": "cooking"}, step=1)

    # Terminate target session
    end_msg = Message.model_construct(
        id="target-end",
        type=MessageType.EVENT,
        source_node_id="test.emitter",
        topic="sensory.output",
        payload={"text": "Done"},
        correlation_id=corr_id_target,
    )
    await bus.publish("sensory.output", end_msg)

    await asyncio.wait_for(consumer_task, timeout=5.0)

    # Only target events should be received
    assert len(collected_chunks) == 2
    assert collected_chunks[0]["type"] == "embodied_observation"
    assert collected_chunks[0]["data"]["pot"] == "cooking"
    assert collected_chunks[1]["type"] == "token"
    assert collected_chunks[1]["text"] == "Done"
