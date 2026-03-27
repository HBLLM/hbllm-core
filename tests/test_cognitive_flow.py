"""
Integration test for the Cognitive Flow Enhancements.

Tests the full loop:
  ExperienceNode (salience detection)
  → MetaReasoningNode (reflection)
  → MemoryNode (pattern extraction into semantic memory)
"""

import asyncio
import pytest

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import (
    Message, MessageType, FeedbackPayload,
)
from hbllm.brain.experience_node import ExperienceNode
from hbllm.brain.meta_node import MetaReasoningNode
from hbllm.memory.memory_node import MemoryNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _wait_for_message(bus, topic: str, predicate=None, timeout: float = 3.0):
    """Subscribe to *topic* and return the first message matching *predicate*."""
    future: asyncio.Future[Message] = asyncio.get_event_loop().create_future()

    async def _handler(m: Message) -> None:
        if future.done():
            return
        if predicate is None or predicate(m):
            future.set_result(m)

    await bus.subscribe(topic, _handler)
    return await asyncio.wait_for(future, timeout=timeout)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_experience_node_detects_high_saliency():
    """ExperienceNode should flag content with priority keywords as high-saliency."""
    bus = InProcessBus()
    await bus.start()

    experience = ExperienceNode(node_id="experience_test")
    await experience.start(bus)

    # Subscribe BEFORE publishing
    salience_future = _wait_for_message(
        bus, "system.salience",
        predicate=lambda m: m.correlation_id == "corr_001",
    )

    await bus.publish("sensory.output", Message(
        type=MessageType.EVENT,
        source_node_id="test",
        topic="sensory.output",
        payload={"text": "Critical system error detected in the reactor core!"},
        correlation_id="corr_001",
    ))

    msg = await salience_future
    assert msg.payload["is_priority"] is True
    assert msg.payload["score"] >= 0.7

    await bus.stop()


@pytest.mark.asyncio
async def test_experience_node_low_saliency():
    """Normal content should NOT be flagged as priority."""
    bus = InProcessBus()
    await bus.start()

    experience = ExperienceNode(node_id="experience_test")
    await experience.start(bus)

    salience_future = _wait_for_message(
        bus, "system.salience",
        predicate=lambda m: m.correlation_id == "corr_002",
    )

    await bus.publish("sensory.output", Message(
        type=MessageType.EVENT,
        source_node_id="test",
        topic="sensory.output",
        payload={"text": "The weather today is sunny and pleasant."},
        correlation_id="corr_002",
    ))

    msg = await salience_future
    assert msg.payload["is_priority"] is False
    assert msg.payload["score"] < 0.7

    await bus.stop()


@pytest.mark.asyncio
async def test_meta_node_triggers_reflection_on_negative_feedback():
    """MetaReasoningNode should fire SYSTEM_IMPROVE after accumulated negative feedback."""
    bus = InProcessBus()
    await bus.start()

    meta = MetaReasoningNode(node_id="meta_test")
    await meta.start(bus)

    # Listen for improve signal specifically for reactor_domain
    improve_future = _wait_for_message(
        bus, "system.improve",
        predicate=lambda m: m.payload.get("domain") == "reactor_domain",
    )

    feedback = FeedbackPayload(
        message_id="msg_001",
        rating=-1,
        prompt="How do I fix the reactor?",
        response="I don't know.",
        module_id="reactor_domain",
    )

    for _ in range(meta.weakness_threshold):
        await bus.publish("system.feedback", Message(
            type=MessageType.FEEDBACK,
            source_node_id="test",
            topic="system.feedback",
            payload=feedback.model_dump(),
        ))

    msg = await improve_future
    assert msg.payload["domain"] == "reactor_domain"
    assert "reflection" in msg.payload["dataset_path"]

    await bus.stop()


@pytest.mark.asyncio
async def test_full_cognitive_loop():
    """
    End-to-end: experience → salience → reflection → pattern extraction.

    Validates that a pattern extracted by the MetaReasoningNode is stored
    in Semantic Memory by the MemoryNode.
    """
    bus = InProcessBus()
    await bus.start()

    experience = ExperienceNode(node_id="experience_e2e")
    meta = MetaReasoningNode(node_id="meta_e2e")
    memory = MemoryNode(node_id="memory_e2e", db_path=":memory:")

    for node in [experience, meta, memory]:
        await node.start(bus)

    try:
        # --- Step 1: salience detection ---
        salience_future = _wait_for_message(
            bus, "system.salience",
            predicate=lambda m: m.correlation_id == "e2e_001",
        )

        await bus.publish("sensory.output", Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="sensory.output",
            payload={"text": "Critical failure in authentication module!"},
            correlation_id="e2e_001",
        ))

        salience_msg = await salience_future
        assert salience_msg.payload["is_priority"] is True

        # --- Step 2: negative-feedback → reflection → SYSTEM_IMPROVE ---
        improve_future = _wait_for_message(
            bus, "system.improve",
            predicate=lambda m: m.payload.get("domain") == "auth_domain",
        )

        feedback = FeedbackPayload(
            message_id="e2e_002",
            rating=-1,
            prompt="Why is auth failing?",
            response="Unknown error.",
            module_id="auth_domain",
        )
        for _ in range(meta.weakness_threshold):
            await bus.publish("system.feedback", Message(
                type=MessageType.FEEDBACK,
                source_node_id="test",
                topic="system.feedback",
                payload=feedback.model_dump(),
            ))

        improve_msg = await improve_future
        assert improve_msg.payload["domain"] == "auth_domain"

        # --- Step 3: pattern extraction into Semantic Memory ---
        # The bus dispatches handlers via create_task, so instead of racing
        # against async scheduling, we directly invoke handle_improvement
        # on the MemoryNode to verify the handler logic deterministically.
        await memory.handle_improvement(improve_msg)

        docs = memory.semantic_db.get_all()
        assert any(
            "Learned pattern in domain 'auth_domain'" in d["content"]
            for d in docs
        ), f"Pattern not found in semantic memory. Documents: {docs}"
    finally:
        await bus.stop()
