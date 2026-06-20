"""
Tests for Multimodal Vision Node with Rust perception engine integration.

Verifies loading, frame-caching change detection, embedding, and workspace thought publishing.
"""

from __future__ import annotations

import asyncio
import io

import pytest
from PIL import Image

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.perception.vision_node import VisionNode

try:
    import hbllm_perception_rs  # type: ignore  # noqa: F401

    HAS_RUST_ENGINE = True
except ImportError:
    HAS_RUST_ENGINE = False

skip_no_rust = pytest.mark.skipif(
    not HAS_RUST_ENGINE,
    reason="hbllm_perception_rs not installed (build with maturin)",
)


def make_test_image(color: tuple[int, int, int] = (100, 100, 100)) -> str:
    """Helper to generate a solid color test image in hex format."""
    img = Image.new("RGB", (64, 64), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue().hex()


@skip_no_rust
@pytest.mark.asyncio
async def test_vision_node_rust_caching_and_embedding() -> None:
    """Test VisionNode uses Rust perception engine, change detection caching, and extracts embeddings."""
    bus = InProcessBus()
    await bus.start()

    node = VisionNode(node_id="test_vision_node")
    node._bus = bus

    # Load model (mock mode inside Rust engine)
    node._load_model()
    assert node.rust_engine is not None
    assert node.change_detector is not None

    image_hex1 = make_test_image((255, 255, 255))
    image_hex2 = make_test_image((0, 0, 0))

    # Send first frame
    msg1 = Message(
        type=MessageType.QUERY,
        source_node_id="client",
        topic="vision.process",
        payload={"image_data": image_hex1, "entity_id": "ent1"},
        session_id="s1",
    )
    res1 = await node.handle_message(msg1)
    assert res1 is not None
    assert res1.error is False
    assert res1.payload["cached"] is False
    assert len(res1.payload["embedding"]) == 768

    # Send identical frame -> should return cached result
    msg2 = Message(
        type=MessageType.QUERY,
        source_node_id="client",
        topic="vision.process",
        payload={"image_data": image_hex1, "entity_id": "ent1"},
        session_id="s1",
    )
    res2 = await node.handle_message(msg2)
    assert res2 is not None
    assert res2.payload["cached"] is True
    assert res2.payload["text"] == res1.payload["text"]

    # Send different frame -> should run full inference
    msg3 = Message(
        type=MessageType.QUERY,
        source_node_id="client",
        topic="vision.process",
        payload={"image_data": image_hex2, "entity_id": "ent1"},
        session_id="s1",
    )
    res3 = await node.handle_message(msg3)
    assert res3 is not None
    assert res3.payload["cached"] is False

    await bus.stop()


@skip_no_rust
@pytest.mark.asyncio
async def test_vision_node_workspace_thought_projection() -> None:
    """Test handle_workspace_query publishes projected embeddings to blackboard."""
    bus = InProcessBus()
    await bus.start()

    node = VisionNode(node_id="test_vision_node")
    node._bus = bus
    node._load_model()

    published_thoughts = []

    async def capture_thought(msg: Message) -> None:
        published_thoughts.append(msg)

    await bus.subscribe("workspace.thought", capture_thought)

    image_hex = make_test_image((128, 128, 128))
    query_msg = Message(
        type=MessageType.QUERY,
        source_node_id="workspace",
        topic="module.evaluate",
        payload={"image_data": image_hex, "entity_id": "ent2"},
        session_id="s2",
    )

    await node.handle_workspace_query(query_msg)

    # Allow event processing time
    await asyncio.sleep(0.1)

    assert len(published_thoughts) == 1
    thought = published_thoughts[0]
    assert thought.topic == "workspace.thought"
    assert thought.payload["type"] == "vision_perception"
    assert len(thought.payload["embedding"]) == 4096  # Projected dimension
    assert thought.payload["embedding"][768:] == [0.0] * (4096 - 768)  # Fallback padded with zeros

    await bus.stop()
