"""Tests for the Perception Ingestion Layer (Phase 3.1)."""

from __future__ import annotations

import asyncio

import pytest

from hbllm.perception.event_log import EventLog
from hbllm.perception.normalizer import EventNormalizer
from hbllm.perception.reality_bus import (
    EventOrigin,
    PerceptionEvent,
    PerceptionModality,
    RealityEventBus,
)


@pytest.fixture
def temp_event_log(tmp_path):
    """Provides a temporary SQLite EventLog."""
    return EventLog(data_dir=tmp_path)


@pytest.fixture
def bus():
    """Provides a RealityEventBus."""
    return RealityEventBus()


@pytest.mark.asyncio
class TestRealityEventBus:
    async def test_ingest_assigns_clocks_and_timestamps(self, bus):
        event = PerceptionEvent(event_type="test", modality=PerceptionModality.APP)
        await bus.ingest(event)

        assert event.logical_clock == 1
        assert event.ingest_timestamp > 0.0
        assert event.event_timestamp <= event.ingest_timestamp

    async def test_subscribers_receive_events(self, bus):
        received = []

        def _sub(e: PerceptionEvent):
            received.append(e)

        bus.subscribe(_sub)
        await bus.ingest(PerceptionEvent(event_type="A"))
        await bus.ingest(PerceptionEvent(event_type="B"))

        # Wait briefly for fire-and-forget sync wrapper if it were fully async,
        # but our bus fires them synchronously if they aren't coroutines.
        await asyncio.sleep(0.01)

        assert len(received) == 2
        assert received[0].logical_clock == 1
        assert received[1].logical_clock == 2

    async def test_async_subscribers(self, bus):
        received = []

        async def _async_sub(e: PerceptionEvent):
            await asyncio.sleep(0.01)
            received.append(e)

        bus.subscribe(_async_sub)
        await bus.ingest(PerceptionEvent(event_type="A"))

        # Wait for the async task to complete
        await asyncio.sleep(0.05)

        assert len(received) == 1


@pytest.mark.asyncio
class TestEventNormalizer:
    async def test_deduplication(self):
        """Test that rapid identical events are coalesced."""
        normalizer = EventNormalizer()
        received = []
        normalizer.subscribe(received.append)

        # Same signature
        e1 = PerceptionEvent(entity_id="u1", event_type="type1", sub_type="sub1", modality=PerceptionModality.APP)
        e2 = PerceptionEvent(entity_id="u1", event_type="type1", sub_type="sub1", modality=PerceptionModality.APP)

        await normalizer.handle_raw_event(e1)
        await normalizer.handle_raw_event(e2)

        # Only one should make it through
        assert len(received) == 1

    async def test_modality_budget_enforcement(self):
        """Test that throttling works per modality."""
        normalizer = EventNormalizer()
        received = []
        normalizer.subscribe(received.append)

        # SENSOR budget is 60/min. We'll send 65 distinct events.
        for i in range(65):
            e = PerceptionEvent(
                entity_id=f"u{i}",
                event_type="motion",
                sub_type=f"cam_{i}",
                modality=PerceptionModality.SENSOR
            )
            await normalizer.handle_raw_event(e)

        assert len(received) == 60
        assert normalizer._modality_counts[PerceptionModality.SENSOR.value] == 60

    async def test_system_events_unlimited(self):
        """Test that SYSTEM modality events are not throttled."""
        normalizer = EventNormalizer()
        received = []
        normalizer.subscribe(received.append)

        for i in range(100):
            e = PerceptionEvent(
                entity_id=f"sys_{i}",
                event_type="os",
                sub_type=f"evt_{i}",
                modality=PerceptionModality.SYSTEM
            )
            await normalizer.handle_raw_event(e)

        assert len(received) == 100

    async def test_routing_to_event_log(self, temp_event_log):
        """Test that the normalizer automatically writes to the Truth Ledger."""
        normalizer = EventNormalizer(event_log=temp_event_log)

        e1 = PerceptionEvent(entity_id="test", event_type="t1", modality=PerceptionModality.APP)
        e1.logical_clock = 1

        await normalizer.handle_raw_event(e1)

        # Read from log
        events = list(temp_event_log.replay())
        assert len(events) == 1
        assert events[0].entity_id == "test"
        assert events[0].logical_clock == 1


class TestEventLog:
    def test_append_and_replay(self, temp_event_log):
        e1 = PerceptionEvent(event_type="A", logical_clock=1)
        e2 = PerceptionEvent(event_type="B", logical_clock=2)

        temp_event_log.append(e1)
        temp_event_log.append(e2)

        replayed = list(temp_event_log.replay())
        assert len(replayed) == 2
        assert replayed[0].event_type == "A"
        assert replayed[1].event_type == "B"

    def test_get_latest_clock(self, temp_event_log):
        assert temp_event_log.get_latest_clock() == 0

        temp_event_log.append(PerceptionEvent(event_type="A", logical_clock=42))
        assert temp_event_log.get_latest_clock() == 42

    def test_idempotent_append(self, temp_event_log):
        """Test that appending the same event ID fails silently (SQLite IGNORE)."""
        e1 = PerceptionEvent(event_id="same_id", event_type="A", logical_clock=1)
        temp_event_log.append(e1)

        e2 = PerceptionEvent(event_id="same_id", event_type="B", logical_clock=2)
        temp_event_log.append(e2)

        replayed = list(temp_event_log.replay())
        assert len(replayed) == 1
        assert replayed[0].event_type == "A"
