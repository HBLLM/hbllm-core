"""Integration tests for Brain subsystem — CognitiveAwareness, PatternDetector."""

import asyncio
import time
from typing import Any

import pytest

from hbllm.brain.awareness import (
    AwarenessSensor,
    CognitiveAwareness,
    CognitiveSnapshot,
    CognitiveTrigger,
    _ActivityWindow,
    _PatternDetector,
)
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


# ── ActivityWindow Tests ─────────────────────────────────────────────────────


class TestActivityWindow:
    """Test the sliding window for activity metrics."""

    def test_record_and_count(self):
        window = _ActivityWindow(window_seconds=60.0)
        window.record("query")
        window.record("query")
        window.record("response")

        assert window.count("query") == 2
        assert window.count("response") == 1
        assert window.count() == 3

    def test_time_based_expiry(self):
        window = _ActivityWindow(window_seconds=0.1)
        window.record("query")

        import time as t
        t.sleep(0.2)

        assert window.count("query") == 0

    def test_top_types(self):
        window = _ActivityWindow(window_seconds=60.0)
        for _ in range(5):
            window.record("query")
        for _ in range(3):
            window.record("response")
        window.record("error")

        top = window.top_types(n=2)
        assert top[0] == "query"
        assert top[1] == "response"

    def test_max_events(self):
        window = _ActivityWindow(window_seconds=60.0, max_events=10)
        for i in range(20):
            window.record(f"event_{i}")

        assert len(window._events) == 10


# ── PatternDetector Tests ────────────────────────────────────────────────────


class TestPatternDetector:
    """Test cognitive pattern detection logic."""

    def test_overload_detection(self):
        detector = _PatternDetector()
        snap = CognitiveSnapshot(
            timestamp=time.time(),
            cognitive_load=0.9,
        )
        triggers = detector.evaluate(snap)
        overload = [t for t in triggers if t.trigger_type == "overload"]
        assert len(overload) == 1
        assert overload[0].priority == "high"

    def test_error_burst_detection(self):
        detector = _PatternDetector()

        # Need 3 consecutive evaluations with high error rate
        for _ in range(3):
            snap = CognitiveSnapshot(
                timestamp=time.time(),
                error_rate=0.7,
                queries_last_minute=5,
            )
            triggers = detector.evaluate(snap)

        degradation = [t for t in triggers if t.trigger_type == "degradation"]
        assert len(degradation) >= 1

    def test_context_switch_detection(self):
        detector = _PatternDetector()

        # First set of topics
        snap1 = CognitiveSnapshot(
            timestamp=time.time(),
            active_topics=["python", "coding"],
        )
        detector.evaluate(snap1)

        # Completely different topics
        snap2 = CognitiveSnapshot(
            timestamp=time.time(),
            active_topics=["cooking", "recipes"],
        )
        triggers = detector.evaluate(snap2)

        switches = [t for t in triggers if t.trigger_type == "context_switch"]
        assert len(switches) == 1

    def test_no_switch_when_topics_overlap(self):
        detector = _PatternDetector()

        snap1 = CognitiveSnapshot(
            timestamp=time.time(),
            active_topics=["python", "coding"],
        )
        detector.evaluate(snap1)

        snap2 = CognitiveSnapshot(
            timestamp=time.time(),
            active_topics=["python", "testing"],  # Overlaps on "python"
        )
        triggers = detector.evaluate(snap2)

        switches = [t for t in triggers if t.trigger_type == "context_switch"]
        assert len(switches) == 0

    def test_milestone_detection(self):
        detector = _PatternDetector()

        snap = CognitiveSnapshot(
            timestamp=time.time(),
            queries_total=10,
        )
        triggers = detector.evaluate(snap)
        milestones = [t for t in triggers if t.trigger_type == "milestone"]
        assert len(milestones) == 1
        assert "10" in milestones[0].message

    def test_milestone_not_repeated(self):
        detector = _PatternDetector()

        snap1 = CognitiveSnapshot(timestamp=time.time(), queries_total=10)
        detector.evaluate(snap1)

        snap2 = CognitiveSnapshot(timestamp=time.time(), queries_total=15)
        triggers = detector.evaluate(snap2)

        milestones = [t for t in triggers if t.trigger_type == "milestone"]
        assert len(milestones) == 0  # Already reported 10, next is 50

    def test_quality_decline_detection(self):
        detector = _PatternDetector()

        # Build up good confidence history
        for _ in range(5):
            snap = CognitiveSnapshot(timestamp=time.time(), avg_confidence=0.9)
            detector.evaluate(snap)

        # Then declining confidence
        triggers = []
        for _ in range(5):
            snap = CognitiveSnapshot(timestamp=time.time(), avg_confidence=0.5)
            triggers.extend(detector.evaluate(snap))

        quality_alerts = [t for t in triggers if t.trigger_type == "quality_alert"]
        assert len(quality_alerts) >= 1


# ── CognitiveSnapshot Tests ──────────────────────────────────────────────────


class TestCognitiveSnapshot:
    """Test snapshot data model."""

    def test_default_values(self):
        snap = CognitiveSnapshot()
        assert snap.queries_total == 0
        assert snap.cognitive_load == 0.0
        assert snap.current_emotion == "neutral"

    def test_to_dict(self):
        snap = CognitiveSnapshot(
            timestamp=1234567890.0,
            queries_total=100,
            cognitive_load=0.75,
        )
        d = snap.to_dict()
        assert d["queries_total"] == 100
        assert d["cognitive_load"] == 0.75
        assert d["timestamp"] == 1234567890.0


class TestCognitiveTrigger:
    """Test trigger data model."""

    def test_trigger_creation(self):
        trigger = CognitiveTrigger(
            trigger_type="overload",
            message="Load at 90%",
            context={"load": 0.9},
            priority="high",
            timestamp=time.time(),
        )
        assert trigger.trigger_type == "overload"
        assert trigger.priority == "high"

    def test_trigger_to_dict(self):
        trigger = CognitiveTrigger(
            trigger_type="milestone",
            message="100 queries",
            timestamp=1234567890.0,
        )
        d = trigger.to_dict()
        assert d["trigger_type"] == "milestone"
        assert d["timestamp"] == 1234567890.0


# ── CognitiveAwareness Bus Integration Tests ─────────────────────────────────


class TestCognitiveAwarenessIntegration:
    """Test CognitiveAwareness node with real InProcessBus."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_start_and_stop(self):
        bus = InProcessBus()
        await bus.start()

        awareness = CognitiveAwareness(node_id="test_awareness", poll_interval=0.1)
        await awareness.start(bus)

        assert awareness._running is True

        await awareness.stop()
        assert awareness._running is False

        await bus.stop()

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_tracks_queries(self):
        bus = InProcessBus()
        await bus.start()

        awareness = CognitiveAwareness(node_id="test_awareness", poll_interval=60.0)
        await awareness.start(bus)

        # Simulate a query
        query_msg = Message(
            type=MessageType.QUERY,
            source_node_id="user",
            topic="sensory.input",
            payload={"query": "Hello"},
            session_id="session_1",
        )
        await bus.publish("sensory.input", query_msg)
        await asyncio.sleep(0.2)

        assert awareness._total_queries >= 1
        assert len(awareness._active_sessions) >= 1

        await awareness.stop()
        await bus.stop()

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_tracks_errors(self):
        bus = InProcessBus()
        await bus.start()

        awareness = CognitiveAwareness(node_id="test_awareness", poll_interval=60.0)
        await awareness.start(bus)

        error_msg = Message(
            type=MessageType.EVENT,
            source_node_id="system",
            topic="system.error",
            payload={"error": "Something failed"},
        )
        await bus.publish("system.error", error_msg)
        await asyncio.sleep(0.2)

        assert awareness._total_errors >= 1

        await awareness.stop()
        await bus.stop()

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_snapshot(self):
        awareness = CognitiveAwareness(node_id="test")
        awareness._total_queries = 42
        awareness._total_responses = 40

        snap = awareness.snapshot()
        assert snap.queries_total == 42
        assert snap.responses_total == 40
        assert snap.time_of_day in ("morning", "afternoon", "evening", "night")

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_sensor_registration(self):
        class MockSensor:
            name = "test_sensor"

            async def collect(self):
                return {"temperature": 72}

        awareness = CognitiveAwareness(node_id="test")
        sensor = MockSensor()
        awareness.register_sensor(sensor)

        assert "test_sensor" in awareness._sensors

        awareness.unregister_sensor("test_sensor")
        assert "test_sensor" not in awareness._sensors

    def test_stats(self):
        awareness = CognitiveAwareness(node_id="test")
        awareness._total_queries = 10
        awareness._total_errors = 2

        stats = awareness.stats()
        assert stats["total_queries"] == 10
        assert stats["total_errors"] == 2

    def test_get_pending_triggers(self):
        awareness = CognitiveAwareness(node_id="test")
        awareness._pending_triggers.append(
            CognitiveTrigger(trigger_type="test", message="Test trigger")
        )

        triggers = awareness.get_pending_triggers()
        assert len(triggers) == 1
        assert len(awareness._pending_triggers) == 0  # Cleared after get
