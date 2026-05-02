"""Tests for TemporalEngine cognitive plugin."""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pytest_asyncio
from temporal_engine import (
    Deadline,
    TemporalEngine,
    TemporalEvent,
    parse_temporal_references,
)

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def bus():
    b = InProcessBus()
    await b.start()
    yield b
    await b.stop()


@pytest_asyncio.fixture
async def engine(bus):
    e = TemporalEngine(node_id="test_temporal")
    await e.start(bus)
    yield e
    await e.stop()


class TestTemporalEvent:
    def test_age_human_just_now(self):
        e = TemporalEvent(topic="test", summary="hello")
        assert e.age_human() == "just now"

    def test_age_human_minutes(self):
        e = TemporalEvent(topic="test", summary="hello", timestamp=time.time() - 120)
        assert "minute" in e.age_human()

    def test_to_dict(self):
        e = TemporalEvent(topic="test", summary="hello", domain="coding")
        d = e.to_dict()
        assert d["topic"] == "test"
        assert d["domain"] == "coding"


class TestDeadline:
    def test_not_overdue_when_future(self):
        d = Deadline(task_id="t1", description="test", due_at=time.time() + 3600)
        assert not d.is_overdue
        assert d.time_remaining > 0

    def test_overdue_when_past(self):
        d = Deadline(task_id="t1", description="test", due_at=time.time() - 10)
        assert d.is_overdue
        assert d.time_remaining_human() == "overdue"

    def test_completed_not_overdue(self):
        d = Deadline(task_id="t1", description="test", due_at=time.time() - 10, completed=True)
        assert not d.is_overdue


class TestTemporalReferences:
    def test_parses_yesterday(self):
        refs = parse_temporal_references("What did we discuss yesterday?")
        assert any(kw == "yesterday" for kw, _ in refs)

    def test_parses_recently(self):
        refs = parse_temporal_references("I recently asked about Python")
        assert any(kw == "recently" for kw, _ in refs)

    def test_no_match_returns_empty(self):
        refs = parse_temporal_references("What is Python?")
        assert refs == []


class TestTemporalEngine:
    async def test_records_events(self, engine, bus):
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="system.experience",
            payload={"text": "Testing response", "query": "What is Python?"},
        )
        await bus.publish("system.experience", msg)
        await asyncio.sleep(0.05)
        assert engine.stats()["total_events"] == 1

    async def test_add_deadline(self, engine):
        d = engine.add_deadline("task_1", "Finish report", due_in_seconds=3600)
        assert d.task_id == "task_1"
        assert not d.is_overdue
        assert engine.stats()["total_deadlines"] == 1

    async def test_overdue_deadlines(self, engine):
        engine.add_deadline("task_1", "Past due", due_in_seconds=-10)
        overdue = engine.get_overdue_deadlines()
        assert len(overdue) == 1

    async def test_upcoming_deadlines(self, engine):
        engine.add_deadline("task_1", "Due soon", due_in_seconds=1800)
        engine.add_deadline("task_2", "Due later", due_in_seconds=7200)
        upcoming = engine.get_upcoming_deadlines(within_seconds=3600)
        assert len(upcoming) == 1

    async def test_find_events_about(self, engine, bus):
        for topic in ["Python basics", "JavaScript frameworks", "Python advanced"]:
            msg = Message(
                type=MessageType.EVENT,
                source_node_id="test",
                topic="system.experience",
                payload={"query": topic, "text": f"Response about {topic}"},
            )
            await bus.publish("system.experience", msg)
            await asyncio.sleep(0.02)
        results = engine.find_events_about("Python")
        assert len(results) == 2

    async def test_task_completion_marks_deadline(self, engine, bus):
        engine.add_deadline("task_x", "Test task", due_in_seconds=3600)
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="system.task.completed",
            payload={"task_id": "task_x"},
        )
        await bus.publish("system.task.completed", msg)
        await asyncio.sleep(0.05)
        assert engine._deadlines["task_x"].completed

    async def test_stats_structure(self, engine):
        stats = engine.stats()
        assert "total_events" in stats
        assert "total_deadlines" in stats
        assert "overdue_deadlines" in stats
        assert "recent_events" in stats
