"""
End-to-End Integration Test for Self-Expansion / SpawnerNode in HBLLM Core.

Verifies the complete self-expansion loop:
1. Classification of domain rank based on domain complexity.
2. SpawnerNode listening for spawn requests on the bus.
3. On-demand dynamic dataset synthesis (falling back to templates when model is None).
4. Spawning, starting, and registering a new DomainModuleNode on the live bus.
5. Announcement of completion via spawn complete and domain registered events.
"""

from __future__ import annotations

import asyncio

import pytest

from hbllm.brain.spawner_node import SpawnerNode, _classify_domain_rank
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


def test_lora_rank_classification() -> None:
    """Verify that domain ranks are classified correctly based on complexity."""
    assert _classify_domain_rank("coding.python") == 32
    assert _classify_domain_rank("medical.diagnosis") == 32
    assert _classify_domain_rank("cooking.recipes") == 16
    assert _classify_domain_rank("unknown_simple_domain") == 8


@pytest.fixture
async def spawner_system():
    """Boot InProcessBus and SpawnerNode with mock model/tokenizer."""
    bus = InProcessBus()
    await bus.start()

    # Model and tokenizer set to None triggers template-based data synthesis fallback
    spawner = SpawnerNode(
        node_id="spawner",
        model=None,
        tokenizer=None,
    )
    await spawner.start(bus)

    yield bus, spawner

    await spawner.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_e2e_self_expansion_loop(spawner_system) -> None:
    """Verify full self-expansion flow on the message bus."""
    bus, spawner = spawner_system

    # Track how many subscribers currently listen to module evaluations
    initial_subscribers = len(bus._subscriptions.get("module.evaluate", []))

    # Capture announcements
    spawn_complete_events: list[Message] = []
    domain_registered_events: list[Message] = []

    async def on_complete(msg: Message) -> None:
        spawn_complete_events.append(msg)

    async def on_registered(msg: Message) -> None:
        domain_registered_events.append(msg)

    await bus.subscribe("system.spawn.complete", on_complete)
    await bus.subscribe("system.domain_registered", on_registered)

    # 1. Publish a SPAWN_REQUEST message to trigger expansion
    spawn_msg = Message(
        type=MessageType.SPAWN_REQUEST,
        source_node_id="router",
        tenant_id="default",
        topic="system.spawn",
        payload={
            "topic": "coding.python",
            "trigger_query": "how do I define a decorator in python?",
            "confidence_score": 0.4,
        },
    )
    await bus.publish("system.spawn", spawn_msg)

    # 2. Wait for background task to complete (template generation + fake training + register)
    # The Spawner runs training in asyncio.to_thread, which resolves quickly when model is None
    for _ in range(40):
        if len(spawn_complete_events) > 0 and len(domain_registered_events) > 0:
            break
        await asyncio.sleep(0.1)

    # 3. Assert completion and registration messages
    assert len(spawn_complete_events) == 1, "Should have received system.spawn.complete event"
    assert len(domain_registered_events) == 1, "Should have received system.domain_registered event"

    complete_msg = spawn_complete_events[0]
    assert complete_msg.payload["domain"] == "default_coding.python"
    assert complete_msg.payload["status"] == "active"
    assert complete_msg.payload["has_lora"] is False  # template fallback has no real LoRA dict

    reg_msg = domain_registered_events[0]
    assert reg_msg.payload["domain"] == "default_coding.python"
    assert "default coding python" in reg_msg.payload["centroid_text"]

    # 4. Verify new DomainModuleNode is alive and registered on the bus
    final_subscribers = bus._subscriptions.get("module.evaluate", [])
    assert len(final_subscribers) == initial_subscribers + 1, (
        "New module should have subscribed to module.evaluate"
    )

    # Find the new subscriber and verify its properties
    new_sub = None
    for sub in final_subscribers:
        handler_self = getattr(sub.handler, "__self__", None)
        if handler_self and getattr(handler_self, "node_id", "") == "domain_default_coding_python":
            new_sub = handler_self
            break

    assert new_sub is not None, "Could not find registered DomainModuleNode on the bus"
    assert new_sub.domain_name == "default_coding.python"
    assert new_sub.node_id == "domain_default_coding_python"
