"""
Execution State — mutable runtime state during execution.

ExecutionPlan is frozen (what we intend to do).
ExecutionState is mutable (what's happening right now).

Same pattern as HCIR: immutable events + mutable projections.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hbllm.execution.plan import ExecutionPlan


class ExecutionStatus(str, Enum):
    """Lifecycle status of an execution."""

    PENDING = "pending"
    RUNNING = "running"
    STREAMING = "streaming"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class ExecutionState:
    """
    Mutable state during execution.

    The plan is frozen (intent). This state tracks what's
    actually happening at runtime: streaming progress,
    modifier state, caches, telemetry, errors.
    """

    plan: ExecutionPlan
    status: ExecutionStatus = ExecutionStatus.PENDING

    # ── Streaming ─────────────────────────────────────────────
    tokens_generated: int = 0
    chunks_emitted: int = 0

    # ── Modifier State ────────────────────────────────────────
    active_modifiers: list[str] = field(default_factory=list)

    # ── Cache ─────────────────────────────────────────────────
    cache_hit: bool = False

    # ── Timing ────────────────────────────────────────────────
    start_time: float | None = None
    end_time: float | None = None
    provider_latency_ms: float = 0.0
    modifier_latency_ms: float = 0.0

    # ── Error Tracking ────────────────────────────────────────
    retries: int = 0
    last_error: str | None = None
    error_history: list[dict[str, Any]] = field(default_factory=list)

    def mark_started(self) -> None:
        """Mark execution as started."""
        self.status = ExecutionStatus.RUNNING
        self.start_time = time.monotonic()

    def mark_completed(self) -> None:
        """Mark execution as completed."""
        self.status = ExecutionStatus.COMPLETED
        self.end_time = time.monotonic()

    def mark_failed(self, error: str) -> None:
        """Mark execution as failed."""
        self.status = ExecutionStatus.FAILED
        self.end_time = time.monotonic()
        self.last_error = error
        self.error_history.append(
            {
                "error": error,
                "retry": self.retries,
                "timestamp": time.monotonic(),
            }
        )

    def mark_cancelled(self) -> None:
        """Mark execution as cancelled."""
        self.status = ExecutionStatus.CANCELLED
        self.end_time = time.monotonic()

    def mark_retrying(self) -> None:
        """Mark execution as retrying."""
        self.status = ExecutionStatus.RETRYING
        self.retries += 1

    @property
    def elapsed_ms(self) -> float:
        """Total elapsed time in milliseconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.monotonic()
        return (end - self.start_time) * 1000.0
