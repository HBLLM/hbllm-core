"""Tests for SpawnerNode — new event-based path (Execution OS).

Guarded for Python 3.10 (StrEnum in hbllm.network.messages requires 3.11+).
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# Guard: Skip entire module if hbllm.network can't import
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="hbllm.network.messages requires Python 3.11+ (StrEnum)",
)


def _import_spawner():
    """Lazy import to avoid collection-time ImportError."""
    from hbllm.brain.emotion.spawner_node import SpawnerNode
    from hbllm.network.messages import Message, MessageType

    return SpawnerNode, Message, MessageType


class FakeBus:
    """Minimal bus mock for testing."""

    def __init__(self) -> None:
        self.published: list[tuple[str, object]] = []
        self._handlers: dict[str, object] = {}

    async def subscribe(self, topic: str, handler: object) -> str:
        self._handlers[topic] = handler
        return f"sub_{topic}"

    async def publish(self, topic: str, message: object) -> None:
        self.published.append((topic, message))

    async def unsubscribe(self, sub_id: str) -> None:
        pass


@pytest.fixture()
def spawner_with_exec_os():
    """SpawnerNode in Execution OS mode with a fake bus."""
    SpawnerNode, _, _ = _import_spawner()
    model = MagicMock()
    tokenizer = MagicMock()
    node = SpawnerNode(
        node_id="spawner_test",
        model=model,
        tokenizer=tokenizer,
        use_execution_os=True,
    )
    bus = FakeBus()
    node._bus = bus  # type: ignore[assignment]
    return node, bus


@pytest.fixture()
def spawner_legacy():
    """SpawnerNode in legacy mode."""
    SpawnerNode, _, _ = _import_spawner()
    model = MagicMock()
    tokenizer = MagicMock()
    return SpawnerNode(
        node_id="spawner_legacy",
        model=model,
        tokenizer=tokenizer,
        use_execution_os=False,
    )


class TestSpawnerExecutionOS:
    @pytest.mark.asyncio()
    async def test_emit_skill_discovered(self, spawner_with_exec_os) -> None:
        """New path should emit skill.discovered event."""
        spawner, bus = spawner_with_exec_os

        await spawner._emit_skill_discovered(
            domain_name="medical",
            topic="medical",
            tenant_id=None,
            lora_rank=32,
        )

        topics = [t for t, _ in bus.published]
        assert "skill.discovered" in topics
        assert "system.domain_registered" in topics

    @pytest.mark.asyncio()
    async def test_skill_discovered_payload(self, spawner_with_exec_os) -> None:
        """skill.discovered event should contain domain metadata."""
        spawner, bus = spawner_with_exec_os

        await spawner._emit_skill_discovered(
            domain_name="coding_python",
            topic="coding.python",
            tenant_id="alice",
            lora_rank=16,
        )

        skill_event = next(msg for topic, msg in bus.published if topic == "skill.discovered")
        payload = skill_event.payload
        assert payload["domain"] == "coding_python"
        assert payload["topic"] == "coding.python"
        assert payload["tenant_id"] == "alice"
        assert payload["suggested_rank"] == 16
        assert "centroid_text" in payload

    @pytest.mark.asyncio()
    async def test_spawn_uses_event_path(self, spawner_with_exec_os) -> None:
        """_spawn_new_module should use event path when use_execution_os=True."""
        spawner, bus = spawner_with_exec_os

        await spawner._spawn_new_module("astronomy", tenant_id=None, lora_rank=8)

        topics = [t for t, _ in bus.published]
        assert "skill.discovered" in topics
        assert "system.spawn.complete" not in topics

    @pytest.mark.asyncio()
    async def test_zero_lora_imports_in_new_path(self, spawner_with_exec_os) -> None:
        """No LoRA/training modules should be imported in the new path."""
        spawner, bus = spawner_with_exec_os
        assert spawner.synthesizer is None

        await spawner._spawn_new_module("quantum_physics", tenant_id=None, lora_rank=8)

        for _, msg in bus.published:
            payload = msg.payload
            assert "lora_state_dict" not in payload
            assert "has_lora" not in payload


class TestSpawnerLegacy:
    def test_synthesizer_created(self, spawner_legacy) -> None:
        """Legacy path should create DataSynthesizer."""
        assert spawner_legacy.synthesizer is not None

    def test_execution_os_flag(self, spawner_legacy) -> None:
        assert spawner_legacy.use_execution_os is False
