"""
Temporal Reasoning Plugin — Time-aware context for conversations.

Tracks conversation timestamps, recognizes temporal references ("yesterday",
"last week"), and provides deadline tracking for scheduled tasks.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.plugin.sdk import HBLLMPlugin, subscribe

logger = logging.getLogger(__name__)


# ── Data Structures ───────────────────────────────────────────────────────────


@dataclass
class TemporalEvent:
    """A timestamped event in conversation history."""

    topic: str
    summary: str
    timestamp: float = field(default_factory=time.time)
    domain: str = "general"
    correlation_id: str = ""

    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def age_human(self) -> str:
        """Human-readable age string."""
        age = self.age_seconds()
        if age < 60:
            return "just now"
        elif age < 3600:
            mins = int(age / 60)
            return f"{mins} minute{'s' if mins != 1 else ''} ago"
        elif age < 86400:
            hours = int(age / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        else:
            days = int(age / 86400)
            return f"{days} day{'s' if days != 1 else ''} ago"

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "summary": self.summary,
            "timestamp": self.timestamp,
            "age": self.age_human(),
            "domain": self.domain,
            "correlation_id": self.correlation_id,
        }


@dataclass
class Deadline:
    """A tracked deadline or scheduled task."""

    task_id: str
    description: str
    due_at: float
    created_at: float = field(default_factory=time.time)
    completed: bool = False

    @property
    def is_overdue(self) -> bool:
        return not self.completed and time.time() > self.due_at

    @property
    def time_remaining(self) -> float:
        return max(0.0, self.due_at - time.time())

    def time_remaining_human(self) -> str:
        remaining = self.time_remaining
        if remaining <= 0:
            return "overdue"
        elif remaining < 3600:
            return f"{int(remaining / 60)} minutes"
        elif remaining < 86400:
            return f"{remaining / 3600:.1f} hours"
        else:
            return f"{remaining / 86400:.1f} days"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "due_at": self.due_at,
            "time_remaining": self.time_remaining_human(),
            "is_overdue": self.is_overdue,
            "completed": self.completed,
        }


# ── Temporal Reference Parser ─────────────────────────────────────────────────

# Keywords that indicate temporal references in user queries
_TEMPORAL_KEYWORDS = {
    "yesterday": timedelta(days=1),
    "last week": timedelta(weeks=1),
    "last month": timedelta(days=30),
    "earlier today": timedelta(hours=6),
    "earlier": timedelta(hours=2),
    "recently": timedelta(hours=24),
    "before": timedelta(hours=1),
    "previously": timedelta(hours=24),
    "last time": timedelta(days=7),
    "a while ago": timedelta(days=3),
}


def parse_temporal_references(text: str) -> list[tuple[str, timedelta]]:
    """Extract temporal references from text."""
    text_lower = text.lower()
    found = []
    for keyword, delta in _TEMPORAL_KEYWORDS.items():
        if keyword in text_lower:
            found.append((keyword, delta))
    return found


# ── Temporal Engine Plugin ────────────────────────────────────────────────────


class TemporalEngine(HBLLMPlugin):
    """Adds time-aware context to conversations."""

    def __init__(
        self,
        node_id: str = "temporal_engine",
        history_size: int = 500,
        max_deadlines: int = 50,
    ) -> None:
        super().__init__(
            node_id=node_id,
            capabilities=["temporal_awareness", "deadline_tracking", "time_references"],
        )
        self._events: deque[TemporalEvent] = deque(maxlen=history_size)
        self._deadlines: dict[str, Deadline] = {}
        self._max_deadlines = max_deadlines

    @subscribe("system.experience")
    async def on_experience(self, message: Message) -> None:
        """Record conversation events with timestamps."""
        text = message.payload.get("text", "")
        query = message.payload.get("query", "")
        intent = message.payload.get("intent", "general")

        # Store the event
        event = TemporalEvent(
            topic="conversation",
            summary=query[:200] if query else text[:200],
            domain=intent,
            correlation_id=message.correlation_id or "",
        )
        self._events.append(event)

        # Check for temporal references in the query
        if query:
            refs = parse_temporal_references(query)
            if refs:
                context = self._find_relevant_events(refs)
                if context:
                    await self._publish_context(context, message.correlation_id)

    @subscribe("system.task.started")
    async def on_task_started(self, message: Message) -> None:
        """Track task start times."""
        task_id = message.payload.get("task_id", "")
        if task_id:
            self._events.append(
                TemporalEvent(
                    topic="task_started",
                    summary=f"Task started: {task_id}",
                    correlation_id=task_id,
                )
            )

    @subscribe("system.task.completed")
    async def on_task_completed(self, message: Message) -> None:
        """Track task completions and update deadlines."""
        task_id = message.payload.get("task_id", "")
        if task_id and task_id in self._deadlines:
            self._deadlines[task_id].completed = True

    def add_deadline(self, task_id: str, description: str, due_in_seconds: float) -> Deadline:
        """Add a tracked deadline."""
        if len(self._deadlines) >= self._max_deadlines:
            # Remove oldest completed deadlines
            completed = [k for k, v in self._deadlines.items() if v.completed]
            for k in completed[:10]:
                del self._deadlines[k]

        deadline = Deadline(
            task_id=task_id,
            description=description,
            due_at=time.time() + due_in_seconds,
        )
        self._deadlines[task_id] = deadline
        return deadline

    def get_overdue_deadlines(self) -> list[Deadline]:
        """Get all overdue, incomplete deadlines."""
        return [d for d in self._deadlines.values() if d.is_overdue]

    def get_upcoming_deadlines(self, within_seconds: float = 3600) -> list[Deadline]:
        """Get deadlines due within the given timeframe."""
        now = time.time()
        return [
            d
            for d in self._deadlines.values()
            if not d.completed and now < d.due_at <= now + within_seconds
        ]

    def find_events_about(self, query: str, max_results: int = 5) -> list[TemporalEvent]:
        """Find past events matching a query string."""
        query_lower = query.lower()
        matches = []
        for event in reversed(self._events):
            if query_lower in event.summary.lower() or query_lower in event.domain.lower():
                matches.append(event)
                if len(matches) >= max_results:
                    break
        return matches

    def _find_relevant_events(
        self, temporal_refs: list[tuple[str, timedelta]]
    ) -> list[dict[str, Any]]:
        """Find events matching temporal references."""
        context = []
        now = time.time()

        for keyword, delta in temporal_refs:
            target_time = now - delta.total_seconds()
            # Find events within 20% of the target timeframe
            window = delta.total_seconds() * 0.2

            for event in self._events:
                if abs(event.timestamp - target_time) < window:
                    context.append(
                        {
                            "reference": keyword,
                            "event": event.to_dict(),
                        }
                    )

        return context

    async def _publish_context(
        self, context: list[dict[str, Any]], correlation_id: str | None
    ) -> None:
        """Publish temporal context for the current query."""
        if self.bus:
            await self.bus.publish(
                "temporal.context",
                Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    topic="temporal.context",
                    payload={
                        "temporal_references": context,
                        "count": len(context),
                    },
                    correlation_id=correlation_id,
                ),
            )

    def stats(self) -> dict[str, Any]:
        """Return temporal statistics."""
        overdue = self.get_overdue_deadlines()
        upcoming = self.get_upcoming_deadlines(within_seconds=3600)
        return {
            "total_events": len(self._events),
            "total_deadlines": len(self._deadlines),
            "overdue_deadlines": len(overdue),
            "upcoming_deadlines": [d.to_dict() for d in upcoming],
            "recent_events": [e.to_dict() for e in list(self._events)[-5:]],
        }
