"""Tests for TemporalEngine SQLite persistence."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import time

import pytest


# Import TemporalEngine from the plugin directory (dashes in path).
def _import_temporal():
    engine_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "hbllm",
        "plugins",
        "temporal-reasoning",
        "temporal_engine.py",
    )
    spec = importlib.util.spec_from_file_location("temporal_engine", engine_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["temporal_engine"] = mod
    spec.loader.exec_module(mod)
    return mod


_te = _import_temporal()
TemporalEngine = _te.TemporalEngine
TemporalEvent = _te.TemporalEvent
parse_temporal_references = _te.parse_temporal_references


# ── TemporalEvent ────────────────────────────────────────────────────────


def test_temporal_event_age():
    """Age should increase over time."""
    event = TemporalEvent(topic="test", summary="hello", timestamp=time.time() - 60)
    assert event.age_seconds() >= 59
    assert "minute" in event.age_human()


def test_temporal_event_just_now():
    """Recent event should say 'just now'."""
    event = TemporalEvent(topic="test", summary="hello")
    assert event.age_human() == "just now"


def test_temporal_event_to_dict():
    """to_dict should include all fields."""
    event = TemporalEvent(topic="test", summary="hello", domain="coding")
    d = event.to_dict()
    assert d["topic"] == "test"
    assert d["domain"] == "coding"
    assert "age" in d


# ── parse_temporal_references ────────────────────────────────────────────


def test_parse_yesterday():
    refs = parse_temporal_references("What did we discuss yesterday?")
    assert any(kw == "yesterday" for kw, _ in refs)


def test_parse_last_week():
    refs = parse_temporal_references("Show me what happened last week")
    assert any(kw == "last week" for kw, _ in refs)


def test_parse_no_reference():
    refs = parse_temporal_references("How does Python work?")
    assert len(refs) == 0


def test_parse_multiple_references():
    refs = parse_temporal_references("Earlier today, and also yesterday, we talked")
    assert len(refs) >= 2


# ── TemporalEngine (in-memory only) ─────────────────────────────────────


def test_engine_in_memory():
    """TemporalEngine without data_dir should work in memory only."""
    engine = TemporalEngine(data_dir=None)
    assert engine._db_path is None
    assert len(engine._events) == 0


def test_engine_add_event_in_memory():
    """Events should be stored in the deque."""
    engine = TemporalEngine(data_dir=None)
    event = TemporalEvent(topic="test", summary="hello")
    engine._events.append(event)
    assert len(engine._events) == 1


def test_engine_find_events_about():
    """find_events_about should search in-memory cache."""
    engine = TemporalEngine(data_dir=None)
    engine._events.append(TemporalEvent(topic="conv", summary="Python decorators"))
    engine._events.append(TemporalEvent(topic="conv", summary="JavaScript async"))
    engine._events.append(TemporalEvent(topic="conv", summary="Python generators"))

    matches = engine.find_events_about("Python")
    assert len(matches) == 2


# ── TemporalEngine (SQLite persistence) ──────────────────────────────────


def test_engine_sqlite_init():
    """SQLite DB should be created on init."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = TemporalEngine(data_dir=tmpdir)
        assert engine._db_path is not None
        assert engine._db_path.exists()


def test_engine_sqlite_persist_and_load():
    """Events should survive restart."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # First instance — add events
        engine1 = TemporalEngine(data_dir=tmpdir)
        event1 = TemporalEvent(topic="conv", summary="First conversation")
        event2 = TemporalEvent(topic="conv", summary="Second conversation")
        engine1._events.append(event1)
        engine1._persist_event(event1)
        engine1._events.append(event2)
        engine1._persist_event(event2)

        # Second instance — should load from DB
        engine2 = TemporalEngine(data_dir=tmpdir)
        assert len(engine2._events) == 2


def test_engine_sqlite_find_events_about():
    """SQLite search should find matching events."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = TemporalEngine(data_dir=tmpdir)
        for topic in ["Python OOP", "JavaScript closures", "Python async"]:
            e = TemporalEvent(topic="conv", summary=topic)
            engine._events.append(e)
            engine._persist_event(e)

        matches = engine.find_events_about("Python")
        assert len(matches) == 2


def test_engine_sqlite_find_between():
    """Time-range query should work."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = TemporalEngine(data_dir=tmpdir)
        now = time.time()

        for i, offset in enumerate([-3600, -1800, -900, -300, -60]):
            e = TemporalEvent(
                topic="conv",
                summary=f"Event {i}",
                timestamp=now + offset,
            )
            engine._events.append(e)
            engine._persist_event(e)

        matches = engine.find_events_between(now - 1800, now)
        assert len(matches) == 4  # -1800, -900, -300, -60


def test_engine_sqlite_detect_patterns():
    """Pattern detection should find frequently occurring domains."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = TemporalEngine(data_dir=tmpdir)

        for _ in range(5):
            e = TemporalEvent(topic="conv", summary="code stuff", domain="coding")
            engine._events.append(e)
            engine._persist_event(e)
        for _ in range(2):
            e = TemporalEvent(topic="conv", summary="chat", domain="general")
            engine._events.append(e)
            engine._persist_event(e)

        patterns = engine.detect_patterns(window_days=7)
        assert len(patterns) >= 1
        assert patterns[0]["domain"] == "coding"
        assert patterns[0]["count"] == 5


# ── Deadlines ────────────────────────────────────────────────────────────


def test_deadline_add():
    engine = TemporalEngine(data_dir=None)
    dl = engine.add_deadline("task_1", "Deploy feature", due_in_seconds=3600)
    assert dl.task_id == "task_1"
    assert not dl.is_overdue


def test_deadline_overdue():
    engine = TemporalEngine(data_dir=None)
    dl = engine.add_deadline("task_1", "Overdue task", due_in_seconds=-1)
    assert dl.is_overdue


def test_deadline_upcoming():
    engine = TemporalEngine(data_dir=None)
    engine.add_deadline("t1", "Soon", due_in_seconds=60)
    engine.add_deadline("t2", "Later", due_in_seconds=7200)
    upcoming = engine.get_upcoming_deadlines(within_seconds=3600)
    assert len(upcoming) == 1
    assert upcoming[0].task_id == "t1"


def test_stats():
    """Stats should include persistence info."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = TemporalEngine(data_dir=tmpdir)
        s = engine.stats()
        assert "total_events_cached" in s
        assert "total_events_persisted" in s
        assert "persistent_storage" in s
        assert s["persistent_storage"] is not None
