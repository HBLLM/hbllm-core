"""
End-to-End Integration Test for Continuous Autonomy and Curiosity Loop in HBLLM Core.

Verifies:
1. Negative feedback published on the bus triggers CuriosityNode knowledge gap detection.
2. CuriosityNode automatically generates and dispatches a learning goal.
3. SpawnerNode intercepts the autonomous spawn request and dynamically registers a new DomainModuleNode.
4. AutonomyCore tick loop reacts to the curiosity events, updating its cognitive state and telemetry.
"""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

from hbllm.brain.autonomy.loop import AutonomyCore
from hbllm.brain.autonomy.state_machine import CognitiveState
from hbllm.brain.emotion.curiosity_node import CuriosityNode
from hbllm.brain.emotion.spawner_node import SpawnerNode
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


@pytest_asyncio.fixture
async def autonomy_curiosity_system():
    """Boot InProcessBus, CuriosityNode, SpawnerNode, and AutonomyCore together."""
    bus = InProcessBus()
    await bus.start()

    # Configure CuriosityNode with immediate dispatch and low threshold
    curiosity = CuriosityNode(
        node_id="curiosity",
        uncertainty_threshold=1,
        goal_dispatch_interval=0.0,
    )
    await curiosity.start(bus)

    # Configure SpawnerNode with template fallback
    spawner = SpawnerNode(
        node_id="spawner",
        model=None,
        tokenizer=None,
    )
    await spawner.start(bus)

    # Configure AutonomyCore to monitor curiosity and system topics
    core = AutonomyCore(
        fast_path_topics=[
            "system.feedback",
            "curiosity.goal",
            "system.spawn",
            "system.spawn.complete",
        ]
    )
    await core.start(bus)

    yield bus, curiosity, spawner, core

    await core.stop()
    await spawner.stop()
    await curiosity.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_e2e_autonomy_and_curiosity_loop(autonomy_curiosity_system) -> None:
    bus, curiosity, spawner, core = autonomy_curiosity_system

    # Capture announcements
    spawn_complete_events: list[Message] = []
    curiosity_goal_events: list[Message] = []

    async def on_spawn_complete(msg: Message) -> None:
        spawn_complete_events.append(msg)

    async def on_curiosity_goal(msg: Message) -> None:
        curiosity_goal_events.append(msg)

    await bus.subscribe("system.spawn.complete", on_spawn_complete)
    await bus.subscribe("curiosity.goal", on_curiosity_goal)

    # 1. Publish negative user feedback rating to simulate a detected knowledge gap
    feedback_msg = Message(
        type=MessageType.EVENT,
        source_node_id="user_interface",
        tenant_id="default",
        topic="system.feedback",
        payload={
            "topic": "quantum_physics",
            "query": "How does quantum entanglement work?",
            "rating": -1,  # Negative feedback
        },
    )
    await bus.publish("system.feedback", feedback_msg)

    # 2. Wait for background processing to complete:
    # feedback -> CuriosityNode -> uncertainty event -> Goal -> system.spawn -> SpawnerNode -> DomainModuleNode
    for _ in range(40):
        if len(spawn_complete_events) > 0 and len(curiosity_goal_events) > 0:
            break
        await asyncio.sleep(0.1)

    # 3. Assert CuriosityNode successfully detected and dispatched the goal
    assert len(curiosity_goal_events) == 1, "Should have received curiosity.goal event"
    goal_payload = curiosity_goal_events[0].payload
    assert goal_payload["goal_topic"] == "quantum_physics"
    assert goal_payload["source_events"] == 1

    # 4. Assert SpawnerNode automatically synthesized data and registered the module
    assert len(spawn_complete_events) == 1, "Should have received system.spawn.complete event"
    assert spawn_complete_events[0].payload["domain"] == "default_quantum_physics"
    assert spawn_complete_events[0].payload["status"] == "active"

    # Verify new DomainModuleNode is registered on the bus
    subscribers = bus._subscriptions.get("module.evaluate", [])
    has_quantum_module = any(
        getattr(sub.handler, "__self__", None) is not None
        and getattr(sub.handler.__self__, "domain_name", "") == "default_quantum_physics"
        for sub in subscribers
    )
    assert has_quantum_module, "New quantum physics module should be active on the bus"

    # 5. Assert AutonomyCore tracked these events in its cognitive telemetry
    snapshot = core.snapshot()
    assert snapshot["fast_path_wakes"] > 0, "AutonomyCore should have woken on fast-path events"
    assert snapshot["ticks_completed"] > 0, "AutonomyCore tick loop should have completed ticks"
    assert snapshot["state_machine"]["state"] in (
        CognitiveState.OBSERVING.value,
        CognitiveState.INTERRUPTED.value,
        CognitiveState.IDLE.value,
    ), "AutonomyCore should be in a valid operating state"
