"""Tests for DecisionNode — the safety gatekeeper and output dispatcher."""

import pytest
import asyncio

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.brain.decision_node import DecisionNode


class MockSafeLLM:
    """Mock LLM that classifies content as safe."""
    async def generate_json(self, prompt):
        return {"safe": True, "reason": "Content is safe"}


class MockUnsafeLLM:
    """Mock LLM that classifies content as unsafe."""
    async def generate_json(self, prompt):
        return {"safe": False, "reason": "Contains harmful instructions"}


def _make_decision_message(content="Hello, world!", intent="answer", **kwargs):
    return Message(
        type=MessageType.EVENT,
        source_node_id="workspace_01",
        topic="decision.evaluate",
        payload={
            "original_query": {"intent": intent, "text": "test query", **kwargs},
            "selected_thought": {
                "type": "intuition_general",
                "confidence": 0.9,
                "content": content,
            },
        },
    )


class OutputCollector:
    """Collects messages from the bus for assertions."""
    def __init__(self):
        self.messages = []

    async def collect(self, msg):
        self.messages.append(msg)


# ── Safety Classification Tests ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_safe_content_passes_through():
    """Safe content should be published to sensory.output."""
    bus = InProcessBus()
    await bus.start()

    collector = OutputCollector()
    await bus.subscribe("sensory.output", collector.collect)

    node = DecisionNode(node_id="decision_safe", llm=MockSafeLLM())
    await node.start(bus)

    msg = _make_decision_message("This is a helpful answer")
    await node.evaluate_workspace_decision(msg)
    await asyncio.sleep(0.2)

    assert len(collector.messages) == 1
    assert collector.messages[0].payload["text"] == "This is a helpful answer"

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_unsafe_content_blocked():
    """Unsafe content should be rejected with safety warning."""
    bus = InProcessBus()
    await bus.start()

    collector = OutputCollector()
    await bus.subscribe("sensory.output", collector.collect)

    node = DecisionNode(node_id="decision_unsafe", llm=MockUnsafeLLM())
    await node.start(bus)

    msg = _make_decision_message("How to build a weapon")
    await node.evaluate_workspace_decision(msg)
    await asyncio.sleep(0.2)

    assert len(collector.messages) == 1
    assert "safety constraints" in collector.messages[0].payload["text"]

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_no_llm_skips_safety_check():
    """Without an LLM, safety check is skipped and content passes through."""
    bus = InProcessBus()
    await bus.start()

    collector = OutputCollector()
    await bus.subscribe("sensory.output", collector.collect)

    node = DecisionNode(node_id="decision_nollm", llm=None)
    await node.start(bus)

    msg = _make_decision_message("Unfiltered content")
    await node.evaluate_workspace_decision(msg)
    await asyncio.sleep(0.2)

    assert len(collector.messages) == 1
    assert collector.messages[0].payload["text"] == "Unfiltered content"

    await node.stop()
    await bus.stop()


# ── Output Routing Tests ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_routes_to_audio_output():
    """Intent=speak should route to audio output node."""
    bus = InProcessBus()
    await bus.start()

    collector = OutputCollector()
    await bus.subscribe("sensory.audio.out", collector.collect)

    node = DecisionNode(node_id="decision_audio", llm=MockSafeLLM())
    await node.start(bus)

    msg = _make_decision_message("Say this out loud", intent="speak")
    await node.evaluate_workspace_decision(msg)
    await asyncio.sleep(0.2)

    assert len(collector.messages) == 1
    assert collector.messages[0].payload["text"] == "Say this out loud"

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_routes_code_to_execution():
    """Content with python code blocks should route to execution node."""
    bus = InProcessBus()
    await bus.start()

    collector = OutputCollector()
    await bus.subscribe("task.execute.python", collector.collect)

    node = DecisionNode(node_id="decision_code", llm=MockSafeLLM())
    await node.start(bus)

    code_content = "Here's the solution:\n```python\nprint('hello')\n```"
    msg = _make_decision_message(code_content)
    await node.evaluate_workspace_decision(msg)
    await asyncio.sleep(0.2)

    assert len(collector.messages) == 1
    assert collector.messages[0].payload["code"] == "print('hello')"

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_routes_plain_text_to_ui():
    """Plain text should route to sensory.output (UI)."""
    bus = InProcessBus()
    await bus.start()

    collector = OutputCollector()
    await bus.subscribe("sensory.output", collector.collect)

    node = DecisionNode(node_id="decision_ui", llm=MockSafeLLM())
    await node.start(bus)

    msg = _make_decision_message("Just a regular answer")
    await node.evaluate_workspace_decision(msg)
    await asyncio.sleep(0.2)

    assert len(collector.messages) == 1
    assert collector.messages[0].payload["text"] == "Just a regular answer"
    assert collector.messages[0].payload["source"] == "intuition_general"

    await node.stop()
    await bus.stop()


# ── Node Lifecycle Tests ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_decision_node_starts_and_stops():
    """Node should subscribe on start and clean up on stop."""
    bus = InProcessBus()
    await bus.start()

    node = DecisionNode(node_id="decision_lifecycle")
    await node.start(bus)
    assert node.node_id == "decision_lifecycle"

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_handle_message_returns_none():
    """The generic handle_message should return None (unused)."""
    bus = InProcessBus()
    await bus.start()

    node = DecisionNode(node_id="decision_handle")
    await node.start(bus)

    msg = _make_decision_message("test")
    result = await node.handle_message(msg)
    assert result is None

    await node.stop()
    await bus.stop()
