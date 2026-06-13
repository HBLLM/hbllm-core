"""Tests for autonomy environment watcher adapters."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest

from hbllm.brain.autonomy.watchers.calendar_watcher import (
    CalendarEvent,
    CalendarWatcher,
    _parse_ics_datetime,
    _parse_ics_file,
)
from hbllm.brain.autonomy.watchers.filesystem_watcher import FilesystemWatcher
from hbllm.brain.autonomy.watchers.idle_detector import IdleDetector
from hbllm.brain.autonomy.watchers.system_health_watcher import (
    HealthThresholds,
    SystemHealthWatcher,
)
from hbllm.network.messages import Message, MessageType

# ── FilesystemWatcher ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_filesystem_watcher_initialization():
    """First check should initialize the baseline, not report changes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a file before first scan
        Path(tmpdir, "existing.txt").write_text("hello")

        watcher = FilesystemWatcher(
            watch_dirs=[tmpdir],
            min_change_interval=0,
        )
        result = await watcher.check()
        assert result is None  # First scan = baseline, no report
        assert watcher._initialized is True
        assert watcher._snapshot  # Should have tracked the file


@pytest.mark.asyncio
async def test_filesystem_watcher_detects_new_file():
    """New files should be detected after initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        watcher = FilesystemWatcher(
            watch_dirs=[tmpdir],
            min_change_interval=0,
        )
        await watcher.check()  # Initialize

        # Create a new file
        Path(tmpdir, "new_file.txt").write_text("created")

        result = await watcher.check()
        assert result is not None
        assert len(result) >= 1
        payload = result[0].payload
        assert payload["total_changes"] >= 1
        assert any(c["path"].endswith("new_file.txt") for c in payload.get("created", []))


@pytest.mark.asyncio
async def test_filesystem_watcher_detects_modification():
    """Modified files should be detected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir, "file.txt")
        filepath.write_text("original")

        watcher = FilesystemWatcher(
            watch_dirs=[tmpdir],
            min_change_interval=0,
        )
        await watcher.check()  # Initialize

        # Wait a bit and modify
        time.sleep(0.05)
        filepath.write_text("modified content")
        # Force mtime change
        os.utime(filepath, (time.time() + 1, time.time() + 1))

        result = await watcher.check()
        assert result is not None
        payload = result[0].payload
        assert payload.get("modified") or payload.get("total_changes", 0) > 0


@pytest.mark.asyncio
async def test_filesystem_watcher_ignores_pycache():
    """Files in __pycache__ should be ignored."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir, "__pycache__")
        cache_dir.mkdir()
        Path(cache_dir, "cached.pyc").write_text("bytecode")

        watcher = FilesystemWatcher(
            watch_dirs=[tmpdir],
            min_change_interval=0,
        )
        await watcher.check()  # Initialize
        assert "__pycache__" not in str(watcher._snapshot)


@pytest.mark.asyncio
async def test_filesystem_watcher_no_watch_dirs():
    """Should return None when no dirs configured."""
    watcher = FilesystemWatcher(watch_dirs=[], min_change_interval=0)
    result = await watcher.check()
    assert result is None


@pytest.mark.asyncio
async def test_filesystem_watcher_snapshot():
    """Snapshot should return useful info."""
    watcher = FilesystemWatcher(watch_dirs=["/tmp"])
    snap = watcher.snapshot()
    assert "watch_dirs" in snap
    assert "tracked_files" in snap


# ── SystemHealthWatcher ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_system_health_watcher_normal():
    """Under normal conditions, should return None (no alerts)."""
    watcher = SystemHealthWatcher(
        thresholds=HealthThresholds(
            disk_free_min_gb=0.001,  # Very low threshold = no alert
            load_avg_max=999.0,  # Very high threshold = no alert
            check_interval=0,
        ),
    )
    result = await watcher.check()
    assert result is None


@pytest.mark.asyncio
async def test_system_health_watcher_disk_alert():
    """Disk alert should trigger when threshold is impossibly high."""
    watcher = SystemHealthWatcher(
        thresholds=HealthThresholds(
            disk_free_min_gb=99999.0,  # Higher than any disk
            check_interval=0,
        ),
    )
    result = await watcher.check()
    assert result is not None
    assert len(result) >= 1
    assert result[0].payload["type"] == "disk_low"


@pytest.mark.asyncio
async def test_system_health_watcher_cooldown():
    """Same alert should not fire twice within cooldown period."""
    watcher = SystemHealthWatcher(
        thresholds=HealthThresholds(
            disk_free_min_gb=99999.0,
            check_interval=0,
        ),
    )
    result1 = await watcher.check()
    assert result1 is not None

    result2 = await watcher.check()
    assert result2 is None  # On cooldown


@pytest.mark.asyncio
async def test_system_health_watcher_status():
    """get_current_status should return platform and disk info."""
    watcher = SystemHealthWatcher()
    status = watcher.get_current_status()
    assert "platform" in status


# ── IdleDetector ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_idle_detector_normal_activity():
    """Should not trigger idle when activity is recent."""
    detector = IdleDetector(idle_threshold_s=300)
    result = await detector.check()
    assert result is None
    assert detector.is_idle is False


@pytest.mark.asyncio
async def test_idle_detector_goes_idle():
    """Should emit idle event when threshold exceeded."""
    detector = IdleDetector(idle_threshold_s=0.01)  # 10ms threshold
    detector._last_activity = time.monotonic() - 1.0  # 1 second ago

    result = await detector.check()
    assert result is not None
    assert len(result) >= 1
    assert result[0].topic == "system.user_idle"
    assert result[0].payload["level"] == "idle"
    assert detector._idle_event_emitted is True


@pytest.mark.asyncio
async def test_idle_detector_deep_idle():
    """Should emit deep idle event after longer threshold."""
    detector = IdleDetector(idle_threshold_s=0.01, deep_idle_threshold_s=0.02)
    detector._last_activity = time.monotonic() - 1.0  # 1 second ago

    result = await detector.check()
    assert result is not None
    # Should get both idle and deep_idle
    levels = [m.payload["level"] for m in result]
    assert "idle" in levels
    assert "deep_idle" in levels


@pytest.mark.asyncio
async def test_idle_detector_no_double_emit():
    """Should not emit idle event twice."""
    detector = IdleDetector(idle_threshold_s=0.01)
    detector._last_activity = time.monotonic() - 1.0

    result1 = await detector.check()
    assert result1 is not None

    result2 = await detector.check()
    assert result2 is None  # Already emitted


@pytest.mark.asyncio
async def test_idle_detector_user_return():
    """on_user_activity should reset idle state."""
    detector = IdleDetector(idle_threshold_s=0.01)
    detector._last_activity = time.monotonic() - 1.0
    await detector.check()  # Go idle

    msg = Message(type=MessageType.QUERY, source_node_id="test", topic="user.input", payload={})
    await detector.on_user_activity(msg)

    assert detector.is_idle is False
    assert detector._idle_event_emitted is False


@pytest.mark.asyncio
async def test_idle_detector_snapshot():
    """Snapshot should return useful info."""
    detector = IdleDetector()
    snap = detector.snapshot()
    assert "is_idle" in snap
    assert "idle_duration_s" in snap


# ── CalendarWatcher ──────────────────────────────────────────────────────


def test_parse_ics_datetime_utc():
    """Parse UTC datetime."""
    dt = _parse_ics_datetime("20260615T140000Z")
    assert dt is not None
    assert dt.year == 2026
    assert dt.month == 6
    assert dt.hour == 14


def test_parse_ics_datetime_local():
    """Parse local datetime."""
    dt = _parse_ics_datetime("20260615T140000")
    assert dt is not None


def test_parse_ics_datetime_date_only():
    """Parse date-only value."""
    dt = _parse_ics_datetime("20260615")
    assert dt is not None
    assert dt.day == 15


def test_parse_ics_datetime_invalid():
    """Invalid value should return None."""
    assert _parse_ics_datetime("invalid") is None


def test_parse_ics_file():
    """Parse a minimal .ics file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ics", delete=False) as f:
        f.write(
            "BEGIN:VCALENDAR\n"
            "BEGIN:VEVENT\n"
            "UID:test-123\n"
            "SUMMARY:Team Standup\n"
            "DTSTART:20260615T090000Z\n"
            "DTEND:20260615T093000Z\n"
            "LOCATION:Zoom\n"
            "END:VEVENT\n"
            "END:VCALENDAR\n"
        )
        f.flush()
        events = _parse_ics_file(Path(f.name))

    os.unlink(f.name)
    assert len(events) == 1
    assert events[0].summary == "Team Standup"
    assert events[0].uid == "test-123"
    assert events[0].location == "Zoom"


@pytest.mark.asyncio
async def test_calendar_watcher_no_dir():
    """Should return None when no calendar dir configured."""
    watcher = CalendarWatcher(calendar_dir=None)
    result = await watcher.check()
    assert result is None


@pytest.mark.asyncio
async def test_calendar_watcher_empty_dir():
    """Should return None for empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        watcher = CalendarWatcher(calendar_dir=tmpdir, check_interval=0)
        result = await watcher.check()
        assert result is None


@pytest.mark.asyncio
async def test_calendar_watcher_snapshot():
    """Snapshot should return useful info."""
    watcher = CalendarWatcher(calendar_dir="/tmp")
    snap = watcher.snapshot()
    assert "calendar_dir" in snap
    assert "lookahead_minutes" in snap


def test_calendar_event_starts_in():
    """starts_in_seconds should calculate correctly."""
    from datetime import datetime, timezone

    future = datetime.now(timezone.utc) + __import__("datetime").timedelta(minutes=10)
    event = CalendarEvent(summary="Test", start=future)
    assert 500 < event.starts_in_seconds < 700  # ~600s
