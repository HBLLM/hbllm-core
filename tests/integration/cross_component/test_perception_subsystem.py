"""Integration tests for Perception subsystem — RealityEventBus, EventNormalizer, EventLog."""

import asyncio

import pytest

from hbllm.perception.event_log import EventLog
from hbllm.perception.normalizer import EventNormalizer, cosine_similarity
from hbllm.perception.reality_bus import (
    EventOrigin,
    PerceptionEvent,
    PerceptionModality,
    RealityEventBus,
)

# ── PerceptionEvent Tests ────────────────────────────────────────────────────


class TestPerceptionEvent:
    """Test the event data model."""

    def test_default_creation(self):
        event = PerceptionEvent()
        assert event.event_id.startswith("evt_")
        assert event.entity_id == "unknown"
        assert event.modality == PerceptionModality.APP
        assert event.origin == EventOrigin.EXTERNAL
        assert event.confidence == 1.0

    def test_custom_creation(self):
        event = PerceptionEvent(
            entity_id="user_1",
            event_type="activity",
            sub_type="window_focus",
            modality=PerceptionModality.SYSTEM,
            payload={"app": "VSCode"},
        )
        assert event.entity_id == "user_1"
        assert event.event_type == "activity"
        assert event.payload["app"] == "VSCode"

    def test_serialization_round_trip(self):
        event = PerceptionEvent(
            entity_id="sensor_1",
            event_type="motion",
            modality=PerceptionModality.SENSOR,
            confidence=0.8,
            payload={"direction": "left"},
        )
        d = event.to_dict()
        restored = PerceptionEvent.from_dict(d)

        assert restored.entity_id == "sensor_1"
        assert restored.event_type == "motion"
        assert restored.modality == PerceptionModality.SENSOR
        assert restored.confidence == 0.8
        assert restored.payload["direction"] == "left"


# ── RealityEventBus Tests ───────────────────────────────────────────────────


class TestRealityEventBus:
    """Test the raw event ingestion bus."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_ingest_assigns_clock(self):
        bus = RealityEventBus()
        event = PerceptionEvent(entity_id="test", event_type="test")

        await bus.ingest(event)

        assert event.logical_clock == 1
        assert event.ingest_timestamp > 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_monotonic_clock(self):
        bus = RealityEventBus()

        events = []
        for i in range(5):
            e = PerceptionEvent(entity_id=f"test_{i}", event_type="test")
            await bus.ingest(e)
            events.append(e)

        clocks = [e.logical_clock for e in events]
        assert clocks == [1, 2, 3, 4, 5]
        assert bus.get_current_clock() == 5

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_subscribe_receives_events(self):
        bus = RealityEventBus()
        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe(handler)
        await bus.ingest(PerceptionEvent(entity_id="test", event_type="activity"))
        await asyncio.sleep(0.1)

        assert len(received) == 1
        assert received[0].entity_id == "test"

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_unsubscribe(self):
        bus = RealityEventBus()
        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe(handler)
        bus.unsubscribe(handler)

        await bus.ingest(PerceptionEvent(entity_id="test", event_type="test"))
        await asyncio.sleep(0.1)

        assert len(received) == 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_pre_subscriber(self):
        bus = RealityEventBus()
        pre_received = []

        def pre_handler(event):
            pre_received.append(event)

        bus.subscribe_pre(pre_handler)
        await bus.ingest(PerceptionEvent(entity_id="test", event_type="test"))

        assert len(pre_received) == 1

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_multiple_subscribers(self):
        bus = RealityEventBus()
        a_received = []
        b_received = []

        async def handler_a(event):
            a_received.append(event)

        async def handler_b(event):
            b_received.append(event)

        bus.subscribe(handler_a)
        bus.subscribe(handler_b)

        await bus.ingest(PerceptionEvent(entity_id="test", event_type="test"))
        await asyncio.sleep(0.1)

        assert len(a_received) == 1
        assert len(b_received) == 1


# ── EventLog Tests ───────────────────────────────────────────────────────────


class TestEventLogIntegration:
    """Test SQLite-backed event logging."""

    def test_append_and_replay(self, tmp_path):
        log = EventLog(data_dir=tmp_path / "events")

        event = PerceptionEvent(
            entity_id="user_1",
            event_type="activity",
            sub_type="click",
            modality=PerceptionModality.APP,
            logical_clock=1,
            payload={"button": "submit"},
        )
        log.append(event)

        events = list(log.replay(start_clock=0))
        assert len(events) == 1
        assert events[0].entity_id == "user_1"
        assert events[0].payload["button"] == "submit"

    def test_replay_ordering(self, tmp_path):
        log = EventLog(data_dir=tmp_path / "events")

        for i in range(5):
            event = PerceptionEvent(
                entity_id=f"entity_{i}",
                event_type="test",
                logical_clock=i + 1,
            )
            log.append(event)

        events = list(log.replay(start_clock=0))
        assert len(events) == 5
        clocks = [e.logical_clock for e in events]
        assert clocks == [1, 2, 3, 4, 5]

    def test_replay_from_clock(self, tmp_path):
        log = EventLog(data_dir=tmp_path / "events")

        for i in range(10):
            log.append(
                PerceptionEvent(
                    entity_id=f"e_{i}",
                    event_type="test",
                    logical_clock=i + 1,
                )
            )

        events = list(log.replay(start_clock=6))
        assert len(events) == 5
        assert events[0].logical_clock == 6

    def test_get_latest_clock(self, tmp_path):
        log = EventLog(data_dir=tmp_path / "events")
        assert log.get_latest_clock() == 0

        log.append(PerceptionEvent(event_type="test", logical_clock=42))
        assert log.get_latest_clock() == 42

    def test_duplicate_event_ignored(self, tmp_path):
        log = EventLog(data_dir=tmp_path / "events")
        event = PerceptionEvent(event_type="test", logical_clock=1)

        log.append(event)
        log.append(event)  # Duplicate — should be silently ignored

        events = list(log.replay())
        assert len(events) == 1

    def test_persistence_round_trip(self, tmp_path):
        db_dir = tmp_path / "events"

        log1 = EventLog(data_dir=db_dir)
        log1.append(PerceptionEvent(entity_id="persist_test", event_type="test", logical_clock=1))

        # Reload from same path
        log2 = EventLog(data_dir=db_dir)
        events = list(log2.replay())
        assert len(events) == 1
        assert events[0].entity_id == "persist_test"


# ── EventNormalizer Tests ────────────────────────────────────────────────────


class TestEventNormalizerIntegration:
    """Test event filtering, deduplication, and routing."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_passes_normal_event(self, tmp_path):
        log = EventLog(data_dir=tmp_path / "events")
        normalizer = EventNormalizer(event_log=log)

        received = []
        normalizer.subscribe(lambda e: received.append(e))

        event = PerceptionEvent(
            entity_id="user_1",
            event_type="activity",
            sub_type="click",
            modality=PerceptionModality.APP,
        )
        await normalizer.handle_raw_event(event)

        assert len(received) == 1
        # Also persisted to log
        assert len(list(log.replay())) == 1

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_deduplicates_rapid_events(self, tmp_path):
        normalizer = EventNormalizer()

        received = []
        normalizer.subscribe(lambda e: received.append(e))

        # Same signature sent rapidly
        for _ in range(5):
            event = PerceptionEvent(
                entity_id="button",
                event_type="click",
                sub_type="submit",
                modality=PerceptionModality.APP,
            )
            await normalizer.handle_raw_event(event)

        # Only first should pass (rest within dedup window)
        assert len(received) == 1

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_allows_different_signatures(self, tmp_path):
        normalizer = EventNormalizer()

        received = []
        normalizer.subscribe(lambda e: received.append(e))

        # Different entities — different signatures
        for i in range(3):
            event = PerceptionEvent(
                entity_id=f"entity_{i}",
                event_type="activity",
                sub_type="action",
                modality=PerceptionModality.SYSTEM,
            )
            await normalizer.handle_raw_event(event)

        assert len(received) == 3

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_budget_enforcement(self, tmp_path):
        normalizer = EventNormalizer()
        # Set very low budget for INFERRED modality
        normalizer.BUDGETS[PerceptionModality.INFERRED] = 2
        normalizer.DEDUP_WINDOWS[PerceptionModality.INFERRED] = 0  # No dedup window

        received = []
        normalizer.subscribe(lambda e: received.append(e))

        for i in range(5):
            event = PerceptionEvent(
                entity_id=f"inferred_{i}",
                event_type="inference",
                sub_type=f"type_{i}",
                modality=PerceptionModality.INFERRED,
            )
            await normalizer.handle_raw_event(event)

        assert len(received) <= 2

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_embedding_bypass_dedup(self, tmp_path):
        normalizer = EventNormalizer()

        received = []
        normalizer.subscribe(lambda e: received.append(e))

        # First event with embedding
        e1 = PerceptionEvent(
            entity_id="user",
            event_type="speech",
            sub_type="utterance",
            modality=PerceptionModality.APP,
            embedding=[1.0, 0.0, 0.0],
        )
        await normalizer.handle_raw_event(e1)

        # Second event with VERY different embedding (same signature)
        e2 = PerceptionEvent(
            entity_id="user",
            event_type="speech",
            sub_type="utterance",
            modality=PerceptionModality.APP,
            embedding=[0.0, 1.0, 0.0],  # Orthogonal — similarity ≈ 0
        )
        await normalizer.handle_raw_event(e2)

        # Both should pass — embeddings are semantically different
        assert len(received) == 2


# ── Cosine Similarity Tests ──────────────────────────────────────────────────


class TestCosineSimilarity:
    """Test the cosine similarity helper function."""

    def test_identical_vectors(self):
        assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_empty_vectors(self):
        assert cosine_similarity([], []) == 0.0

    def test_mismatched_lengths(self):
        assert cosine_similarity([1, 2], [1, 2, 3]) == 0.0

    def test_zero_vectors(self):
        assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0


# ── Full Perception Pipeline Tests ───────────────────────────────────────────


class TestPerceptionPipelineIntegration:
    """Test the full RealityEventBus → EventNormalizer → EventLog pipeline."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_full_pipeline(self, tmp_path):
        """Events flow from bus → normalizer → log."""
        log = EventLog(data_dir=tmp_path / "events")
        normalizer = EventNormalizer(event_log=log)

        downstream_received = []
        normalizer.subscribe(lambda e: downstream_received.append(e))

        bus = RealityEventBus()
        bus.subscribe(normalizer.handle_raw_event)

        # Ingest events through the bus
        await bus.ingest(
            PerceptionEvent(
                entity_id="user_1",
                event_type="activity",
                sub_type="keystroke",
                modality=PerceptionModality.APP,
                payload={"key": "enter"},
            )
        )

        await asyncio.sleep(0.2)

        # Event should reach downstream AND be persisted
        assert len(downstream_received) >= 1
        logged_events = list(log.replay())
        assert len(logged_events) >= 1
        assert logged_events[0].entity_id == "user_1"
