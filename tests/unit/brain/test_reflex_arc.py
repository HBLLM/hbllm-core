"""
Tests for Autonomic Reflex Arc.

Verifies pattern-matching rules, sub-millisecond bypass, and EventLog integration.
"""

from __future__ import annotations

import pytest

from hbllm.perception.event_log import EventLog
from hbllm.perception.reality_bus import (
    PerceptionEvent,
    PerceptionModality,
    RealityEventBus,
)
from hbllm.perception.reflex_arc import ReflexArc, ReflexRule


@pytest.fixture
def temp_event_log(tmp_path) -> EventLog:
    """Provides a temporary SQLite EventLog."""
    return EventLog(data_dir=tmp_path)


@pytest.mark.asyncio
async def test_reflex_rule_matching() -> None:
    """Test ReflexRule trigger matching logic (exact and range comparisons)."""
    # 1. Exact match
    rule1 = ReflexRule(
        trigger={"event_type": "sensor", "sub_type": "smoke_detected"},
        action_topic="action.emergency",
        action_payload={"msg": "Fire!"},
    )

    event_match = PerceptionEvent(event_type="sensor", sub_type="smoke_detected")
    event_mismatch = PerceptionEvent(event_type="sensor", sub_type="temp_high")

    assert rule1.matches(event_match) is True
    assert rule1.matches(event_mismatch) is False

    # 2. Comparison match (__gte, __lte)
    rule2 = ReflexRule(
        trigger={"priority_hint__gte": 90, "modality": PerceptionModality.SENSOR},
        action_topic="action.emergency",
        action_payload={"msg": "High priority sensor event!"},
    )

    ev1 = PerceptionEvent(priority_hint=95, modality=PerceptionModality.SENSOR, event_type="motion")
    ev2 = PerceptionEvent(priority_hint=85, modality=PerceptionModality.SENSOR, event_type="motion")
    ev3 = PerceptionEvent(priority_hint=95, modality=PerceptionModality.APP, event_type="focus")

    assert rule2.matches(ev1) is True
    assert rule2.matches(ev2) is False
    assert rule2.matches(ev3) is False


@pytest.mark.asyncio
async def test_reflex_arc_fast_path_activation(temp_event_log) -> None:
    """Test ReflexArc synchronous fast-path triggering and EventLog audit trail."""
    bus = RealityEventBus()

    rule = ReflexRule(
        trigger={"event_type": "sensor", "sub_type": "smoke_detected"},
        action_topic="action.emergency",
        action_payload={"msg": "Fire!"},
    )

    reflex_arc = ReflexArc(bus=bus, rules=[rule], event_log=temp_event_log)

    # Ingest triggering event
    trigger_event = PerceptionEvent(
        event_type="sensor",
        sub_type="smoke_detected",
        priority_hint=100,
        modality=PerceptionModality.SENSOR,
    )

    # Synchronous pre-subscriber execution is triggered during ingest
    await bus.ingest(trigger_event)

    # 1. Action fired immediately
    assert len(reflex_arc.fired_actions) == 1
    topic, payload = reflex_arc.fired_actions[0]
    assert topic == "action.emergency"
    assert payload == {"msg": "Fire!"}

    # 2. Audit trail created in EventLog
    replayed = list(temp_event_log.replay())
    assert len(replayed) == 1
    audit_event = replayed[0]
    assert audit_event.event_type == "reflex_activation"
    assert audit_event.sub_type == "action.emergency"
    assert audit_event.payload["triggering_event_id"] == trigger_event.event_id
