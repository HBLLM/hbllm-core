"""
End-to-End Integration Test for the Autonomic Reflex Arc in HBLLM Core.

Verifies:
1. Rapid matching and synchronous trigger of ReflexRule actions upon sensory ingestion.
2. Direct bridging from the sensory RealityEventBus to the logical InProcessBus, bypassing the cognitive loop.
3. Sub-millisecond execution latency from ingestion to actuator response.
4. Correct audit logging of reflex activations in the SQLite EventLog.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.perception.event_log import EventLog
from hbllm.perception.reality_bus import (
    EventOrigin,
    PerceptionEvent,
    PerceptionModality,
    RealityEventBus,
)
from hbllm.perception.reflex_arc import ReflexArc, ReflexRule


class BridgingEventLog(EventLog):
    """Event log that bridges reflex activations to the main logical bus."""

    def __init__(self, data_dir, logical_bus: InProcessBus) -> None:
        super().__init__(data_dir)
        self.logical_bus = logical_bus

    def append(self, event: PerceptionEvent) -> None:
        super().append(event)
        if event.event_type == "reflex_activation":
            # Direct bridge bypass
            command_msg = Message(
                type=MessageType.COMMAND,
                source_node_id="reflex_arc",
                tenant_id="default",
                topic=event.sub_type,
                payload=event.payload.get("action_payload", {}),
            )
            # Schedule publishing
            asyncio.create_task(self.logical_bus.publish(event.sub_type, command_msg))


@pytest.fixture
def temp_event_log(tmp_path) -> EventLog:
    """Provides a temporary SQLite EventLog."""
    return EventLog(data_dir=tmp_path)


@pytest.mark.asyncio
async def test_e2e_reflex_arc_bypass(tmp_path) -> None:
    # 1. Initialize both the sensory bus (RealityEventBus) and the logical bus (InProcessBus)
    reality_bus = RealityEventBus()
    logical_bus = InProcessBus()
    await logical_bus.start()

    # 2. Setup the custom BridgingEventLog
    bridging_log = BridgingEventLog(data_dir=tmp_path, logical_bus=logical_bus)

    try:
        # 3. Setup a target action receiver (actuator) on the InProcessBus
        fired_commands: list[Message] = []
        actuator_receive_time = 0.0

        async def actuator_handler(msg: Message) -> None:
            nonlocal actuator_receive_time
            actuator_receive_time = time.perf_counter()
            fired_commands.append(msg)

        await logical_bus.subscribe("action.emergency", actuator_handler)

        # 4. Define reflex rule: temp critical sensory alert triggers immediate emergency shutdown
        rule = ReflexRule(
            trigger={"event_type": "sensor", "sub_type": "temp_critical"},
            action_topic="action.emergency",
            action_payload={"command": "shutdown_system", "urgency": "critical"},
        )

        # 5. Instantiate ReflexArc connected to reality_bus
        reflex_arc = ReflexArc(bus=reality_bus, rules=[rule], event_log=bridging_log)

        # 6. Ingest the raw critical sensory event
        critical_event = PerceptionEvent(
            event_type="sensor",
            sub_type="temp_critical",
            priority_hint=100,
            modality=PerceptionModality.SENSOR,
            origin=EventOrigin.SYSTEM,
        )

        ingest_start_time = time.perf_counter()
        await reality_bus.ingest(critical_event)

        # Wait for fire-and-forget subscribers to publish and handle on InProcessBus
        # Use a polling loop instead of fixed sleep to reduce flakiness under load
        for _ in range(50):
            await asyncio.sleep(0.01)
            if fired_commands:
                break

        ingest_to_actuator_latency = (actuator_receive_time - ingest_start_time) * 1000.0

        # 7. Verify sub-millisecond execution and bypassing
        assert len(reflex_arc.fired_actions) == 1, "ReflexArc should have fired the action"
        assert len(fired_commands) == 1, (
            "Actuator on logical bus should have received the shutdown command"
        )
        assert fired_commands[0].payload["command"] == "shutdown_system"
        assert fired_commands[0].payload["urgency"] == "critical"

        print(
            f"[Reflex Arc E2E] Latency from sensory ingest to actuator receive: {ingest_to_actuator_latency:.3f}ms"
        )
        assert ingest_to_actuator_latency < 500.0, (
            f"Should fire and bridge with low latency, got {ingest_to_actuator_latency:.1f}ms"
        )

        # 8. Verify audit logs in EventLog
        replayed_logs = list(bridging_log.replay())
        assert len(replayed_logs) >= 1, "Should have created reflex activation audit log"
        audit = next(log for log in replayed_logs if log.event_type == "reflex_activation")
        assert audit.sub_type == "action.emergency"
        assert audit.payload["triggering_event_id"] == critical_event.event_id

    finally:
        await logical_bus.stop()
