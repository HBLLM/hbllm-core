"""
Cognitive Trace — End-to-end observability for the cognitive pipeline.

Every cognitive event carries a ``trace_id`` that follows it through
saliency, competition, reasoning, planning, simulation, memory, and
back. This enables visualization of cognition and debugging of
multi-subsystem interactions.

Architecture::

    User says "fix the auth bug"
        ↓ trace_id = "t_abc123"
    TraceEvent(component="queue", action="enqueue")
    TraceEvent(component="saliency", action="scored", metadata={"score": 0.92})
    TraceEvent(component="competition", action="selected")
    TraceEvent(component="workspace", action="broadcast")
    TraceEvent(component="reasoner", action="evidence_gathered")
    TraceEvent(component="planner", action="candidates_generated")
    TraceEvent(component="simulation", action="approved", metadata={"score": 0.85})
    TraceEvent(component="decision", action="committed")
    TraceEvent(component="memory", action="stored")
        ↓
    CognitiveTrace — single object reconstructs the entire thought

Usage::

    from hbllm.brain.trace import CognitiveTrace, TraceCollector

    collector = TraceCollector()
    trace = collector.start_trace()
    trace.record("saliency", "scored", {"score": 0.92})
    trace.record("competition", "selected")
    ...
    collector.finish_trace(trace)
    # Later: collector.get_trace(trace_id) → full timeline
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# TraceEvent — single breadcrumb in a cognitive trace
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TraceEvent:
    """A single timestamped breadcrumb in a cognitive trace.

    Attributes:
        timestamp: When this event occurred (epoch seconds).
        component: Which subsystem generated this event
            (e.g., "saliency", "competition", "planner").
        action: What happened (e.g., "scored", "selected", "approved").
        metadata: Arbitrary key-value pairs for debugging context.
    """

    timestamp: float
    component: str
    action: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "component": self.component,
            "action": self.action,
            "metadata": self.metadata,
        }


# ═══════════════════════════════════════════════════════════════════════════
# CognitiveTrace — complete trace of a single cognitive event
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CognitiveTrace:
    """Complete trace of a single cognitive event through the pipeline.

    Collects ``TraceEvent`` breadcrumbs as the event flows through
    saliency → competition → workspace → reasoner → planner →
    simulation → decision → memory.

    Attributes:
        trace_id: Unique identifier for this trace.
        started_at: When the trace began.
        finished_at: When the trace completed (None if still active).
        events: Ordered list of trace events.
        source: What triggered this trace (e.g., "user_input", "goal_update").
        tenant_id: Multi-tenant isolation key.
    """

    trace_id: str = field(default_factory=lambda: f"t_{uuid.uuid4().hex[:12]}")
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    events: list[TraceEvent] = field(default_factory=list)
    source: str = ""
    tenant_id: str = "default"

    def record(
        self,
        component: str,
        action: str,
        metadata: dict[str, Any] | None = None,
    ) -> TraceEvent:
        """Record a trace event.

        Args:
            component: Subsystem name (e.g., "saliency", "planner").
            action: What happened (e.g., "scored", "approved").
            metadata: Optional debugging context.

        Returns:
            The created TraceEvent.
        """
        event = TraceEvent(
            timestamp=time.time(),
            component=component,
            action=action,
            metadata=metadata or {},
        )
        self.events.append(event)
        return event

    def finish(self) -> None:
        """Mark this trace as complete."""
        self.finished_at = time.time()

    @property
    def is_active(self) -> bool:
        """Whether this trace is still collecting events."""
        return self.finished_at is None

    @property
    def duration(self) -> float:
        """Total trace duration in seconds."""
        end = self.finished_at or time.time()
        return end - self.started_at

    @property
    def component_path(self) -> list[str]:
        """Ordered list of components this trace passed through."""
        return [e.component for e in self.events]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "source": self.source,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration": round(self.duration, 4),
            "event_count": len(self.events),
            "path": self.component_path,
            "events": [e.to_dict() for e in self.events],
        }


# ═══════════════════════════════════════════════════════════════════════════
# TraceCollector — manages active and completed traces
# ═══════════════════════════════════════════════════════════════════════════


class TraceCollector:
    """Manages cognitive traces across the system.

    Central collection point for all traces. Supports trace lookup,
    retention policies, and statistics.

    Args:
        max_retained: Maximum number of completed traces to retain.
    """

    def __init__(self, max_retained: int = 1000) -> None:
        self._max_retained = max_retained
        self._active_traces: dict[str, CognitiveTrace] = {}
        self._completed_traces: list[CognitiveTrace] = []
        self._total_traces: int = 0

    def start_trace(
        self,
        source: str = "",
        tenant_id: str = "default",
    ) -> CognitiveTrace:
        """Start a new cognitive trace.

        Args:
            source: What triggered this trace.
            tenant_id: Multi-tenant isolation key.

        Returns:
            A new active CognitiveTrace.
        """
        trace = CognitiveTrace(source=source, tenant_id=tenant_id)
        self._active_traces[trace.trace_id] = trace
        self._total_traces += 1
        logger.debug("Trace started: %s (source=%s)", trace.trace_id, source)
        return trace

    def finish_trace(self, trace: CognitiveTrace) -> None:
        """Finish an active trace and move it to completed storage.

        Args:
            trace: The trace to finish.
        """
        trace.finish()
        self._active_traces.pop(trace.trace_id, None)
        self._completed_traces.append(trace)

        # Retention policy: drop oldest traces if over limit
        if len(self._completed_traces) > self._max_retained:
            self._completed_traces = self._completed_traces[-self._max_retained :]

        logger.debug(
            "Trace finished: %s (duration=%.3fs, events=%d)",
            trace.trace_id,
            trace.duration,
            len(trace.events),
        )

    def get_trace(self, trace_id: str) -> CognitiveTrace | None:
        """Look up a trace by ID (active or completed).

        Args:
            trace_id: The trace identifier.

        Returns:
            The trace, or None if not found.
        """
        if trace_id in self._active_traces:
            return self._active_traces[trace_id]
        for trace in reversed(self._completed_traces):
            if trace.trace_id == trace_id:
                return trace
        return None

    def get_recent(self, count: int = 10) -> list[CognitiveTrace]:
        """Get the most recent completed traces.

        Args:
            count: Number of traces to return.

        Returns:
            List of recent traces (newest first).
        """
        return list(reversed(self._completed_traces[-count:]))

    def stats(self) -> dict[str, Any]:
        """Collector statistics."""
        avg_duration = 0.0
        if self._completed_traces:
            avg_duration = sum(t.duration for t in self._completed_traces) / len(
                self._completed_traces
            )
        return {
            "total_traces": self._total_traces,
            "active": len(self._active_traces),
            "completed": len(self._completed_traces),
            "avg_duration": round(avg_duration, 4),
        }
