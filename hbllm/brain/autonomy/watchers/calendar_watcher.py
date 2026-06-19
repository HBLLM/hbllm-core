"""Calendar Watcher — monitors .ics files for upcoming events.

Reads iCalendar files from a configurable directory and emits events
for upcoming calendar entries. Uses stdlib only (no icalendar dependency).
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hbllm.network.messages import Message, MessageType

logger = logging.getLogger(__name__)


@dataclass
class CalendarEvent:
    """A parsed calendar event."""

    uid: str = ""
    summary: str = ""
    start: datetime | None = None
    end: datetime | None = None
    location: str = ""
    description: str = ""

    @property
    def starts_in_seconds(self) -> float:
        """Seconds until this event starts (negative if past)."""
        if self.start is None:
            return float("inf")
        delta = self.start - datetime.now(timezone.utc)
        return delta.total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "summary": self.summary,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "location": self.location,
            "starts_in_minutes": round(self.starts_in_seconds / 60, 1),
        }


def _parse_ics_datetime(value: str) -> datetime | None:
    """Parse a basic iCalendar DTSTART/DTEND value."""
    value = value.strip()
    # Remove property parameters like VALUE=DATE, TZID=...
    if ":" in value:
        value = value.split(":")[-1]

    try:
        if value.endswith("Z"):
            return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        elif "T" in value:
            return datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        else:
            return datetime.strptime(value, "%Y%m%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _parse_ics_file(filepath: Path) -> list[CalendarEvent]:
    """Parse a .ics file and extract VEVENT blocks."""
    events: list[CalendarEvent] = []
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return events

    # Split into VEVENT blocks
    vevent_pattern = re.compile(r"BEGIN:VEVENT(.*?)END:VEVENT", re.DOTALL)
    for match in vevent_pattern.finditer(content):
        block = match.group(1)
        event = CalendarEvent()

        for line in block.split("\n"):
            line = line.strip()
            if line.startswith("SUMMARY"):
                event.summary = line.split(":", 1)[-1].strip()
            elif line.startswith("DTSTART"):
                event.start = _parse_ics_datetime(line.split(":", 1)[-1] if ":" in line else "")
            elif line.startswith("DTEND"):
                event.end = _parse_ics_datetime(line.split(":", 1)[-1] if ":" in line else "")
            elif line.startswith("LOCATION"):
                event.location = line.split(":", 1)[-1].strip()
            elif line.startswith("UID"):
                event.uid = line.split(":", 1)[-1].strip()
            elif line.startswith("DESCRIPTION"):
                event.description = line.split(":", 1)[-1].strip()[:200]

        if event.summary and event.start:
            events.append(event)

    return events


class CalendarWatcher:
    """Proactive handler that monitors .ics files for upcoming events.

    Scans a directory for .ics files and emits events when calendar
    entries are approaching.

    Usage::

        watcher = CalendarWatcher(calendar_dir="~/.calendars")
        autonomy_core.add_proactive_handler("calendar", watcher.check)
    """

    def __init__(
        self,
        calendar_dir: str | Path | None = None,
        lookahead_minutes: float = 15.0,
        check_interval: float = 60.0,
    ) -> None:
        self.calendar_dir = Path(calendar_dir).expanduser() if calendar_dir else None
        self.lookahead_minutes = lookahead_minutes
        self.check_interval = check_interval

        self._last_check: float = 0.0
        self._notified_uids: set[str] = set()  # Don't re-notify same event

    async def check(self) -> list[Message] | None:
        """Proactive handler callback — check for upcoming calendar events."""
        now = time.monotonic()
        if now - self._last_check < self.check_interval:
            return None
        self._last_check = now

        if self.calendar_dir is None or not self.calendar_dir.exists():
            return None

        # Scan for .ics files
        all_events: list[CalendarEvent] = []
        try:
            for entry in os.scandir(self.calendar_dir):
                if entry.name.endswith(".ics") and entry.is_file():
                    all_events.extend(_parse_ics_file(Path(entry.path)))
        except (PermissionError, OSError) as e:
            logger.debug("[CalendarWatcher] Cannot scan %s: %s", self.calendar_dir, e)
            return None

        if not all_events:
            return None

        # Filter for upcoming events within lookahead window
        lookahead_s = self.lookahead_minutes * 60
        upcoming = [
            e
            for e in all_events
            if 0 < e.starts_in_seconds <= lookahead_s and e.uid not in self._notified_uids
        ]

        if not upcoming:
            return None

        messages: list[Message] = []
        for event in upcoming:
            self._notified_uids.add(event.uid)
            minutes_until = round(event.starts_in_seconds / 60, 1)

            urgency = 0.7 if minutes_until <= 5 else 0.4

            messages.append(
                Message(
                    type=MessageType.EVENT,
                    source_node_id="autonomy.watcher.calendar",
                    topic="perception.calendar.upcoming",
                    payload={
                        **event.to_dict(),
                        "_urgency": urgency,
                        "_goal_alignment": 0.3,
                    },
                )
            )

            logger.info(
                "[CalendarWatcher] Upcoming: '%s' in %.1f minutes",
                event.summary,
                minutes_until,
            )

        # Clean up old notifications (events more than 1 hour past)
        current_uids = {e.uid for e in all_events if e.starts_in_seconds > -3600}
        self._notified_uids &= current_uids

        return messages if messages else None

    def snapshot(self) -> dict[str, Any]:
        """Introspection snapshot."""
        return {
            "calendar_dir": str(self.calendar_dir) if self.calendar_dir else None,
            "lookahead_minutes": self.lookahead_minutes,
            "notified_events": len(self._notified_uids),
        }
