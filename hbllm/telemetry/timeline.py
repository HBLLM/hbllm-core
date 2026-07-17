"""
Cognitive Timeline — high-resolution event timeline recorder.

Records a chronological sequence of ``TimelineEntry`` objects across all
cognitive subsystems.  Each entry carries full causal provenance metadata,
enabling post-hoc reconstruction of the exact event sequence for any turn.

Usage::

    from hbllm.telemetry.timeline import CognitiveTimeline, TimelineEntry

    timeline = CognitiveTimeline(max_entries=10000)
    timeline.record("perception.audio_in", "user_spoke", {"text": "Hello"})
    timeline.record("brain.snn.comprehension", "saliency_scored", {"score": 0.9})

    # Query recent entries
    recent = timeline.query(last_n=10)

    # Export for replay
    entries = timeline.export_range(start_time, end_time)
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from hbllm.brain.core.provenance import ProvenanceMetadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimelineEntry:
    """A single entry in the cognitive timeline.

    Attributes:
        entry_id: Globally unique identifier.
        subsystem: The subsystem that produced this entry.
        event_name: Human-readable name of the event.
        data: Structured event data.
        provenance: Causal provenance metadata.
        timestamp: When this entry was recorded.
    """

    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    subsystem: str = ""
    event_name: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    provenance: ProvenanceMetadata | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "entry_id": self.entry_id,
            "subsystem": self.subsystem,
            "event_name": self.event_name,
            "data": self.data,
            "timestamp": self.timestamp,
        }
        if self.provenance is not None:
            d["provenance"] = self.provenance.to_dict()
        return d


class CognitiveTimeline:
    """High-resolution chronological event timeline.

    Thread-safe ring buffer of ``TimelineEntry`` objects with
    subsystem filtering, time-range queries, and export capabilities.

    Args:
        max_entries: Maximum number of entries to retain.
    """

    def __init__(self, max_entries: int = 10000) -> None:
        self._entries: deque[TimelineEntry] = deque(maxlen=max_entries)
        self._max_entries = max_entries
        self._total_recorded = 0

    def record(
        self,
        subsystem: str,
        event_name: str,
        data: dict[str, Any] | None = None,
        provenance: ProvenanceMetadata | None = None,
    ) -> TimelineEntry:
        """Record a new timeline entry.

        Args:
            subsystem: The subsystem producing this event.
            event_name: Human-readable event name.
            data: Structured event data.
            provenance: Optional causal provenance.

        Returns:
            The recorded TimelineEntry.
        """
        entry = TimelineEntry(
            subsystem=subsystem,
            event_name=event_name,
            data=data or {},
            provenance=provenance,
        )
        self._entries.append(entry)
        self._total_recorded += 1
        return entry

    def query(
        self,
        last_n: int = 10,
        subsystem: str | None = None,
    ) -> list[TimelineEntry]:
        """Query recent timeline entries.

        Args:
            last_n: Number of recent entries to return.
            subsystem: Optional filter by subsystem name.

        Returns:
            List of matching entries, most recent last.
        """
        entries = list(self._entries)
        if subsystem:
            entries = [e for e in entries if e.subsystem == subsystem]
        return entries[-last_n:]

    def export_range(
        self,
        start_time: float,
        end_time: float,
        subsystem: str | None = None,
    ) -> list[dict[str, Any]]:
        """Export timeline entries within a time range.

        Args:
            start_time: Start of the time range (POSIX timestamp).
            end_time: End of the time range (POSIX timestamp).
            subsystem: Optional filter by subsystem.

        Returns:
            List of serialized entries.
        """
        results: list[dict[str, Any]] = []
        for entry in self._entries:
            if start_time <= entry.timestamp <= end_time:
                if subsystem and entry.subsystem != subsystem:
                    continue
                results.append(entry.to_dict())
        return results

    def clear(self) -> None:
        """Clear all timeline entries."""
        self._entries.clear()

    def stats(self) -> dict[str, Any]:
        """Timeline statistics."""
        return {
            "total_recorded": self._total_recorded,
            "current_size": len(self._entries),
            "max_entries": self._max_entries,
        }
