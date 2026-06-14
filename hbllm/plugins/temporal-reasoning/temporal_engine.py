"""
Temporal Reasoning Plugin — Time-aware context for conversations.

Tracks conversation timestamps, recognizes temporal references ("yesterday",
"last week"), and provides deadline tracking for scheduled tasks.

Now with SQLite persistence — temporal context survives restarts.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
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
    """Adds time-aware context to conversations.

    Uses SQLite for persistent event storage with an in-memory hot cache
    for fast recent event access.
    """

    def __init__(
        self,
        node_id: str = "temporal_engine",
        history_size: int = 500,
        max_deadlines: int = 50,
        data_dir: str | Path | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            capabilities=["temporal_awareness", "deadline_tracking", "time_references"],
        )
        # In-memory hot cache for fast access to recent events
        self._events: deque[TemporalEvent] = deque(maxlen=history_size)
        self._deadlines: dict[str, Deadline] = {}
        self._max_deadlines = max_deadlines

        # SQLite persistence
        self._db_path: Path | None = None
        if data_dir is not None:
            data_path = Path(data_dir)
            data_path.mkdir(parents=True, exist_ok=True)
            self._db_path = data_path / "temporal_events.db"
            self._init_db()
            self._load_recent_events(history_size)

    def _init_db(self) -> None:
        """Create the SQLite schema."""
        if self._db_path is None:
            return
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS temporal_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    domain TEXT DEFAULT 'general',
                    correlation_id TEXT DEFAULT ''
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_temporal_ts ON temporal_events(timestamp)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_temporal_domain ON temporal_events(domain)"
            )

    def _load_recent_events(self, limit: int) -> None:
        """Load recent events from SQLite into the hot cache."""
        if self._db_path is None:
            return
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM temporal_events ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                for row in reversed(rows):
                    self._events.append(
                        TemporalEvent(
                            topic=row["topic"],
                            summary=row["summary"],
                            timestamp=row["timestamp"],
                            domain=row["domain"],
                            correlation_id=row["correlation_id"],
                        )
                    )
            logger.info(
                "[TemporalEngine] Loaded %d events from persistent storage", len(self._events)
            )
        except Exception as e:
            logger.warning("[TemporalEngine] Failed to load events: %s", e)

    def _persist_event(self, event: TemporalEvent) -> None:
        """Persist an event to SQLite."""
        if self._db_path is None:
            return
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """INSERT INTO temporal_events
                       (topic, summary, timestamp, domain, correlation_id)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        event.topic,
                        event.summary,
                        event.timestamp,
                        event.domain,
                        event.correlation_id,
                    ),
                )
        except Exception as e:
            logger.debug("[TemporalEngine] Failed to persist event: %s", e)

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
        self._persist_event(event)

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
            event = TemporalEvent(
                topic="task_started",
                summary=f"Task started: {task_id}",
                correlation_id=task_id,
            )
            self._events.append(event)
            self._persist_event(event)

    @subscribe("system.task.completed")
    async def on_task_completed(self, message: Message) -> None:
        """Track task completions and update deadlines."""
        task_id = message.payload.get("task_id", "")
        if task_id and task_id in self._deadlines:
            self._deadlines[task_id].completed = True

    def add_deadline(self, task_id: str, description: str, due_in_seconds: float) -> Deadline:
        """Add a tracked deadline."""
        if len(self._deadlines) >= self._max_deadlines:
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
        """Find past events matching a query string.

        Searches SQLite if available, otherwise falls back to in-memory cache.
        """
        if self._db_path is not None:
            return self._find_events_about_db(query, max_results)

        query_lower = query.lower()
        matches = []
        for event in reversed(self._events):
            if query_lower in event.summary.lower() or query_lower in event.domain.lower():
                matches.append(event)
                if len(matches) >= max_results:
                    break
        return matches

    def _find_events_about_db(self, query: str, max_results: int) -> list[TemporalEvent]:
        """Search SQLite for events matching a query."""
        if self._db_path is None:
            return []
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """SELECT * FROM temporal_events
                       WHERE summary LIKE ? OR domain LIKE ?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (f"%{query}%", f"%{query}%", max_results),
                ).fetchall()
                return [
                    TemporalEvent(
                        topic=r["topic"],
                        summary=r["summary"],
                        timestamp=r["timestamp"],
                        domain=r["domain"],
                        correlation_id=r["correlation_id"],
                    )
                    for r in rows
                ]
        except Exception:
            return []

    def find_events_between(
        self, start: float, end: float, max_results: int = 50
    ) -> list[TemporalEvent]:
        """Find events within a time range.

        Args:
            start: Start timestamp (unix epoch).
            end: End timestamp (unix epoch).
            max_results: Maximum events to return.
        """
        if self._db_path is not None:
            try:
                with sqlite3.connect(self._db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute(
                        """SELECT * FROM temporal_events
                           WHERE timestamp BETWEEN ? AND ?
                           ORDER BY timestamp DESC LIMIT ?""",
                        (start, end, max_results),
                    ).fetchall()
                    return [
                        TemporalEvent(
                            topic=r["topic"],
                            summary=r["summary"],
                            timestamp=r["timestamp"],
                            domain=r["domain"],
                            correlation_id=r["correlation_id"],
                        )
                        for r in rows
                    ]
            except Exception:
                pass

        # Fallback to in-memory
        return [e for e in self._events if start <= e.timestamp <= end][:max_results]

    def detect_patterns(self, window_days: int = 7) -> list[dict[str, Any]]:
        """Detect recurring patterns in recent events.

        Looks for domains/topics that appear frequently within the window.

        Returns:
            List of detected patterns with domain, count, and frequency.
        """
        cutoff = time.time() - (window_days * 86400)
        domain_counts: dict[str, int] = {}

        if self._db_path is not None:
            try:
                with sqlite3.connect(self._db_path) as conn:
                    rows = conn.execute(
                        """SELECT domain, COUNT(*) as cnt
                           FROM temporal_events
                           WHERE timestamp > ?
                           GROUP BY domain
                           ORDER BY cnt DESC LIMIT 20""",
                        (cutoff,),
                    ).fetchall()
                    domain_counts = {r[0]: r[1] for r in rows}
            except Exception:
                pass

        if not domain_counts:
            for event in self._events:
                if event.timestamp > cutoff:
                    domain_counts[event.domain] = domain_counts.get(event.domain, 0) + 1

        patterns = []
        for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
            if count >= 3:  # Minimum threshold for a pattern
                patterns.append(
                    {
                        "domain": domain,
                        "count": count,
                        "frequency_per_day": round(count / max(window_days, 1), 2),
                        "window_days": window_days,
                    }
                )

        return patterns

    def _find_relevant_events(
        self, temporal_refs: list[tuple[str, timedelta]]
    ) -> list[dict[str, Any]]:
        """Find events matching temporal references."""
        context = []
        now = time.time()

        for keyword, delta in temporal_refs:
            target_time = now - delta.total_seconds()
            window = delta.total_seconds() * 0.2
            start = target_time - window
            end = target_time + window

            matching = self.find_events_between(start, end, max_results=5)
            for event in matching:
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

        total_persisted = 0
        if self._db_path is not None:
            try:
                with sqlite3.connect(self._db_path) as conn:
                    total_persisted = conn.execute(
                        "SELECT COUNT(*) FROM temporal_events"
                    ).fetchone()[0]
            except Exception:
                pass

        return {
            "total_events_cached": len(self._events),
            "total_events_persisted": total_persisted,
            "total_deadlines": len(self._deadlines),
            "overdue_deadlines": len(overdue),
            "upcoming_deadlines": [d.to_dict() for d in upcoming],
            "recent_events": [e.to_dict() for e in list(self._events)[-5:]],
            "persistent_storage": str(self._db_path) if self._db_path else None,
        }
