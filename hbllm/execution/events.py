"""
Execution Events — domain events for the ExecutionBus.

These are domain events (not runtime callbacks). They enable:
  - Audit history
  - Replay
  - Swarm subscription
  - Observability
  - Debugging
  - Distributed execution

Persisted in the same journal philosophy as HCIR cognitive events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ExecutionEvent(str, Enum):
    """Domain events emitted by the ExecutionBus."""

    SUBMITTED = "execution.submitted"
    STARTED = "execution.started"
    PROGRESS = "execution.progress"
    COMPLETED = "execution.completed"
    FAILED = "execution.failed"
    CANCELLED = "execution.cancelled"
    RETRIED = "execution.retried"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ExecutionEventData:
    """
    A single execution domain event.

    Immutable and timestamped, suitable for journaling.
    """

    event: ExecutionEvent
    plan_id: str
    trace_id: str | None = None
    timestamp: str = field(default_factory=_now_iso)
    data: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def submitted(plan_id: str, trace_id: str | None = None) -> ExecutionEventData:
        return ExecutionEventData(
            event=ExecutionEvent.SUBMITTED,
            plan_id=plan_id,
            trace_id=trace_id,
        )

    @staticmethod
    def started(plan_id: str, trace_id: str | None = None) -> ExecutionEventData:
        return ExecutionEventData(
            event=ExecutionEvent.STARTED,
            plan_id=plan_id,
            trace_id=trace_id,
        )

    @staticmethod
    def completed(
        plan_id: str,
        trace_id: str | None = None,
        **data: Any,
    ) -> ExecutionEventData:
        return ExecutionEventData(
            event=ExecutionEvent.COMPLETED,
            plan_id=plan_id,
            trace_id=trace_id,
            data=data,
        )

    @staticmethod
    def failed(
        plan_id: str,
        error: str,
        trace_id: str | None = None,
    ) -> ExecutionEventData:
        return ExecutionEventData(
            event=ExecutionEvent.FAILED,
            plan_id=plan_id,
            trace_id=trace_id,
            data={"error": error},
        )

    @staticmethod
    def cancelled(plan_id: str, trace_id: str | None = None) -> ExecutionEventData:
        return ExecutionEventData(
            event=ExecutionEvent.CANCELLED,
            plan_id=plan_id,
            trace_id=trace_id,
        )

    @staticmethod
    def retried(
        plan_id: str,
        retry_count: int,
        trace_id: str | None = None,
    ) -> ExecutionEventData:
        return ExecutionEventData(
            event=ExecutionEvent.RETRIED,
            plan_id=plan_id,
            trace_id=trace_id,
            data={"retry_count": retry_count},
        )

    @staticmethod
    def progress(
        plan_id: str,
        tokens_generated: int = 0,
        trace_id: str | None = None,
    ) -> ExecutionEventData:
        return ExecutionEventData(
            event=ExecutionEvent.PROGRESS,
            plan_id=plan_id,
            trace_id=trace_id,
            data={"tokens_generated": tokens_generated},
        )
