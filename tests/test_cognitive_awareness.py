"""Tests for core CognitiveAwareness engine."""

import asyncio
import time

import pytest

from hbllm.brain.awareness import (
    AwarenessSensor,
    CognitiveAwareness,
    CognitiveSnapshot,
    CognitiveTrigger,
    _ActivityWindow,
    _PatternDetector,
)


class TestActivityWindow:
    def test_record_and_count(self):
        w = _ActivityWindow(window_seconds=60.0)
        w.record("query")
        w.record("query")
        w.record("error")
        assert w.count("query") == 2
        assert w.count("error") == 1
        assert w.count() == 3

    def test_window_expiry(self):
        w = _ActivityWindow(window_seconds=1.0)
        w.record("old", timestamp=time.time() - 2.0)
        w.record("recent")
        assert w.count() == 1

    def test_top_types(self):
        w = _ActivityWindow(window_seconds=60.0)
        for _ in range(5):
            w.record("query")
        for _ in range(3):
            w.record("error")
        w.record("tool")
        tops = w.top_types(n=2)
        assert tops[0] == "query"
        assert tops[1] == "error"


class TestPatternDetector:
    def test_idle_return(self):
        d = _PatternDetector()
        # First: mark idle
        snap1 = CognitiveSnapshot(idle_seconds=200, queries_last_minute=0, timestamp=time.time())
        d.evaluate(snap1)
        # Second: activity resumes
        snap2 = CognitiveSnapshot(idle_seconds=5, queries_last_minute=3, timestamp=time.time())
        triggers2 = d.evaluate(snap2)
        assert any(t.trigger_type == "idle_return" for t in triggers2)

    def test_context_switch(self):
        d = _PatternDetector()
        snap1 = CognitiveSnapshot(
            active_topics=["coding", "python"], queries_last_minute=5, timestamp=time.time()
        )
        d.evaluate(snap1)

        snap2 = CognitiveSnapshot(
            active_topics=["music", "art"], queries_last_minute=3, timestamp=time.time()
        )
        triggers = d.evaluate(snap2)
        assert any(t.trigger_type == "context_switch" for t in triggers)

    def test_no_context_switch_on_overlap(self):
        d = _PatternDetector()
        snap1 = CognitiveSnapshot(
            active_topics=["coding", "python"], queries_last_minute=5, timestamp=time.time()
        )
        d.evaluate(snap1)

        snap2 = CognitiveSnapshot(
            active_topics=["coding", "rust"], queries_last_minute=3, timestamp=time.time()
        )
        triggers = d.evaluate(snap2)
        assert not any(t.trigger_type == "context_switch" for t in triggers)

    def test_error_burst(self):
        d = _PatternDetector()
        # Need 3 consecutive high-error snapshots
        for i in range(3):
            snap = CognitiveSnapshot(error_rate=0.7, queries_last_minute=5, timestamp=time.time())
            triggers = d.evaluate(snap)
        assert any(t.trigger_type == "degradation" for t in triggers)

    def test_overload(self):
        d = _PatternDetector()
        snap = CognitiveSnapshot(cognitive_load=0.9, timestamp=time.time())
        triggers = d.evaluate(snap)
        assert any(t.trigger_type == "overload" for t in triggers)

    def test_milestone(self):
        d = _PatternDetector()
        snap = CognitiveSnapshot(queries_total=50, timestamp=time.time())
        triggers = d.evaluate(snap)
        assert any(t.trigger_type == "milestone" for t in triggers)


class TestCognitiveSnapshot:
    def test_to_dict(self):
        snap = CognitiveSnapshot(timestamp=123.0, queries_total=10)
        d = snap.to_dict()
        assert d["timestamp"] == 123.0
        assert d["queries_total"] == 10


class TestCognitiveTrigger:
    def test_to_dict(self):
        t = CognitiveTrigger(
            trigger_type="idle_return",
            message="Back from idle",
            priority="low",
            timestamp=time.time(),
        )
        d = t.to_dict()
        assert d["trigger_type"] == "idle_return"


class TestSensorProtocol:
    def test_protocol_check(self):
        class MockSensor:
            name = "mock"

            async def collect(self):
                return {"test": True}

        sensor = MockSensor()
        assert isinstance(sensor, AwarenessSensor)

    def test_register_sensor(self):
        awareness = CognitiveAwareness.__new__(CognitiveAwareness)
        awareness._sensors = {}

        class TestSensor:
            name = "test_sensor"

            async def collect(self):
                return {"data": 42}

        awareness.register_sensor(TestSensor())
        assert "test_sensor" in awareness._sensors

    def test_unregister_sensor(self):
        awareness = CognitiveAwareness.__new__(CognitiveAwareness)
        awareness._sensors = {}

        class TestSensor:
            name = "test_sensor"

            async def collect(self):
                return {}

        awareness.register_sensor(TestSensor())
        awareness.unregister_sensor("test_sensor")
        assert "test_sensor" not in awareness._sensors
