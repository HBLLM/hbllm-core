"""Routine Reflexes — schedule and habit awareness rules.

4 reflexes for routine/schedule monitoring:
    1. calendar_conflict    — overlapping events detected
    2. meeting_reminder     — upcoming meeting in 5/15 minutes
    3. commute_alert        — unusual commute time considerations
    4. bedtime_wind_down    — approaching typical bedtime

All reflexes are deterministic (Tier 1) — zero LLM cost.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from hbllm.brain.autonomy.attention import AttentionEvent
from hbllm.brain.autonomy.reflexes import make_push_message
from hbllm.network.messages import Message

logger = logging.getLogger(__name__)

ReflexRule = Callable[[AttentionEvent], Message | None]


def _calendar_conflict(event: AttentionEvent) -> Message | None:
    """Overlapping calendar events detected."""
    if event.source.startswith("calendar.") or event.source.startswith("autonomy.watcher.calendar"):
        conflicts = event.payload.get("conflicts", [])
        if conflicts and len(conflicts) >= 2:
            event_names = [c.get("summary", "Event") for c in conflicts[:3]]
            time_str = event.payload.get("conflict_time", "")
            return make_push_message(
                title="📅 Calendar Conflict",
                body=f"Overlapping events: {', '.join(event_names)}"
                + (f" at {time_str}" if time_str else ""),
                priority="high",
                category="reminder",
            )
    return None


def _meeting_reminder(event: AttentionEvent) -> Message | None:
    """Upcoming meeting reminder (5 or 15 minutes before)."""
    if event.source.startswith("calendar.") or event.source.startswith("autonomy.watcher.calendar"):
        minutes_until = event.payload.get("minutes_until")
        if minutes_until is not None and minutes_until in (5, 15, 10, 1):
            summary = event.payload.get("summary", "Meeting")
            location = event.payload.get("location", "")
            location_text = f" at {location}" if location else ""
            return make_push_message(
                title=f"📅 {summary} in {minutes_until} min",
                body=f"Your meeting starts in {minutes_until} minutes{location_text}.",
                priority="high" if minutes_until <= 5 else "info",
                category="reminder",
            )
    return None


def _commute_alert(event: AttentionEvent) -> Message | None:
    """Time-based commute awareness (mornings and evenings)."""
    if event.source.startswith("autonomy.watcher.") or event.source == "internal.routine":
        commute_type = event.payload.get("commute_type")  # "morning" or "evening"
        if commute_type:
            conditions = event.payload.get("conditions", "")
            eta_minutes = event.payload.get("eta_minutes")
            eta_text = f" Estimated travel: {eta_minutes} min." if eta_minutes else ""
            return make_push_message(
                title=f"🚗 {commute_type.title()} Commute",
                body=f"Time to prepare for your {commute_type} commute.{eta_text}"
                + (f" {conditions}" if conditions else ""),
                priority="suggestion",
                category="reminder",
            )
    return None


def _bedtime_wind_down(event: AttentionEvent) -> Message | None:
    """Approaching typical bedtime — suggest winding down."""
    if event.source.startswith("autonomy.watcher.") or event.source == "internal.routine":
        bedtime_approaching = event.payload.get("bedtime_approaching", False)
        if bedtime_approaching:
            minutes_until = event.payload.get("minutes_until_bedtime", 30)
            return make_push_message(
                title="🌙 Wind Down",
                body=f"Your usual bedtime is in about {minutes_until} minutes. "
                f"Consider wrapping up for the night.",
                priority="suggestion",
                category="habit",
            )
    return None


def get_routine_reflexes() -> dict[str, ReflexRule]:
    """Return all routine reflexes."""
    return {
        "calendar_conflict": _calendar_conflict,
        "meeting_reminder": _meeting_reminder,
        "commute_alert": _commute_alert,
        "bedtime_wind_down": _bedtime_wind_down,
    }
