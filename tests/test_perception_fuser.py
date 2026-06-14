"""Tests for PerceptionFuser — cross-modal temporal alignment."""

from __future__ import annotations

import asyncio
import time

import pytest

from hbllm.network.messages import Message, MessageType
from hbllm.perception.perception_fuser import (
    FusedContext,
    PerceptionEvent,
    PerceptionFuser,
)

# ── PerceptionEvent ──────────────────────────────────────────────────────


def test_perception_event_age():
    event = PerceptionEvent(
        modality="audio",
        content="user said hello",
        timestamp=time.time() - 2.0,
    )
    assert event.age_seconds() >= 1.5


# ── FusedContext ─────────────────────────────────────────────────────────


def test_fused_context_multimodal():
    """Context with multiple modalities should report is_multimodal."""
    ctx = FusedContext(
        events=[
            PerceptionEvent(modality="audio", content="speech"),
            PerceptionEvent(modality="visual", content="screen"),
        ],
        modalities={"audio", "visual"},
    )
    assert ctx.is_multimodal is True


def test_fused_context_single_modal():
    """Single modality should not be multimodal."""
    ctx = FusedContext(
        events=[PerceptionEvent(modality="audio", content="speech")],
        modalities={"audio"},
    )
    assert ctx.is_multimodal is False


def test_fused_context_to_dict():
    """to_dict should group events by modality."""
    ctx = FusedContext(
        events=[
            PerceptionEvent(modality="audio", content="said hello"),
            PerceptionEvent(modality="visual", content="VSCode active"),
        ],
        modalities={"audio", "visual"},
    )
    d = ctx.to_dict()
    assert "audio" in d
    assert "visual" in d
    assert d["is_multimodal"] is True
    assert d["event_count"] == 2


def test_fused_context_summary():
    """summary_text should combine all events."""
    ctx = FusedContext(
        events=[
            PerceptionEvent(modality="audio", content="said hello"),
            PerceptionEvent(modality="visual", content="VSCode"),
        ],
        modalities={"audio", "visual"},
    )
    summary = ctx.summary_text()
    assert "audio" in summary.lower()
    assert "visual" in summary.lower()


# ── PerceptionFuser ──────────────────────────────────────────────────────


def test_fuser_topic_mapping():
    """Topic-to-modality mapping should work correctly."""
    fuser = PerceptionFuser()
    assert fuser._topic_to_modality("sensory.audio.in") == "audio"
    assert fuser._topic_to_modality("perception.vision") == "visual"
    assert fuser._topic_to_modality("perception.filesystem.changes") == "system"
    assert fuser._topic_to_modality("unknown.topic") == "text"


def test_fuser_content_extraction():
    """Should extract content from common payload fields."""
    fuser = PerceptionFuser()
    assert fuser._extract_content({"text": "hello"}, "audio") == "hello"
    assert fuser._extract_content({"summary": "file changed"}, "system") == "file changed"
    assert fuser._extract_content({}, "audio") == ""


@pytest.mark.asyncio
async def test_fuser_single_event():
    """Single modality should not trigger fusion."""
    fuser = PerceptionFuser(window_seconds=5.0, min_modalities=2)

    msg = Message(
        type=MessageType.EVENT,
        source_node_id="audio_in",
        topic="sensory.audio.in",
        payload={"text": "user said hello"},
    )
    await fuser.on_perception_event(msg)

    assert fuser._total_events == 1
    assert fuser._total_fusions == 0


@pytest.mark.asyncio
async def test_fuser_multimodal_triggers_fusion():
    """Multiple modalities within window should trigger fusion."""
    published: list[Message] = []

    class FakeBus:
        async def publish(self, topic: str, msg: Message) -> None:
            published.append(msg)

    fuser = PerceptionFuser(
        window_seconds=5.0,
        min_modalities=2,
        bus=FakeBus(),
    )
    fuser._fusion_delay = 0.05  # Speed up test

    # Audio event
    await fuser.on_perception_event(
        Message(
            type=MessageType.EVENT,
            source_node_id="audio",
            topic="sensory.audio.in",
            payload={"text": "user said deploy"},
        )
    )

    # Visual event
    await fuser.on_perception_event(
        Message(
            type=MessageType.EVENT,
            source_node_id="vision",
            topic="perception.vision",
            payload={"text": "terminal showing git status"},
        )
    )

    # Wait for debounced fusion
    await asyncio.sleep(0.2)

    assert fuser._total_events == 2
    assert fuser._total_fusions == 1
    assert len(published) == 1
    assert published[0].topic == "perception.fused"
    assert published[0].payload["is_multimodal"] is True


@pytest.mark.asyncio
async def test_fuser_prune_old_events():
    """Events outside the window should be pruned."""
    fuser = PerceptionFuser(window_seconds=1.0)

    # Add an old event
    fuser._window.append(
        PerceptionEvent(
            modality="audio",
            content="old event",
            timestamp=time.time() - 5.0,  # 5 seconds ago
        )
    )

    fuser._prune_window()
    assert len(fuser._window) == 0


def test_fuser_snapshot():
    """Snapshot should return useful info."""
    fuser = PerceptionFuser()
    snap = fuser.snapshot()
    assert "window_size" in snap
    assert "total_events_processed" in snap
    assert "total_fusions" in snap


def test_fuser_modality_summary():
    """Should count events per modality."""
    fuser = PerceptionFuser()
    fuser._window.append(PerceptionEvent(modality="audio", content="a"))
    fuser._window.append(PerceptionEvent(modality="audio", content="b"))
    fuser._window.append(PerceptionEvent(modality="visual", content="c"))

    summary = fuser.get_modality_summary()
    assert summary["audio"] == 2
    assert summary["visual"] == 1
