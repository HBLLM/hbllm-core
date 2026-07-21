"""
Kernel Event Bus — typed internal event system for the HCIR kernel.

Provides an in-process publish/subscribe bus for kernel-level events.
All kernel subsystems (scheduler, transaction manager, verifiers)
publish events here, enabling:

    - Monitoring and observability
    - Debugging and tracing
    - Training data collection
    - Analytics and telemetry

Bus events are different from graph events:
    - Graph events (``IEventStore``) are persisted, append-only state changes.
    - Bus events (``KernelEventBus``) are ephemeral notifications.

Usage::

    bus = KernelEventBus()
    bus.subscribe("transaction.committed", lambda e: print(e))
    bus.publish(KernelEvent(
        event_type="transaction.committed",
        source="transaction_manager",
        data={"tx_id": "tx_001"},
    ))
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Kernel Event Types
# ═══════════════════════════════════════════════════════════════════════════


class KernelEventType(StrEnum):
    """Standard kernel event types."""

    # Transaction lifecycle
    TRANSACTION_PROPOSED = "transaction.proposed"
    TRANSACTION_COMMITTED = "transaction.committed"
    TRANSACTION_REJECTED = "transaction.rejected"

    # Capability lifecycle
    CAPABILITY_REGISTERED = "capability.registered"
    CAPABILITY_BOUND = "capability.bound"
    CAPABILITY_FAILED = "capability.failed"

    # Instruction execution
    INSTRUCTION_STARTED = "instruction.started"
    INSTRUCTION_COMPLETED = "instruction.completed"
    INSTRUCTION_FAILED = "instruction.failed"

    # Scheduler
    TASK_SCHEDULED = "scheduler.task_scheduled"
    TASK_COMPLETED = "scheduler.task_completed"

    # Budget
    BUDGET_WARNING = "budget.warning"
    BUDGET_EXCEEDED = "budget.exceeded"

    # Simulation
    SIMULATION_FORKED = "simulation.forked"
    SIMULATION_MERGED = "simulation.merged"
    SIMULATION_ROLLED_BACK = "simulation.rolled_back"

    # Security
    SECURITY_VIOLATION = "security.violation"
    SCOPE_VIOLATION = "security.scope_violation"

    # Compilation
    COMPILATION_STARTED = "compiler.started"
    COMPILATION_COMPLETED = "compiler.completed"


# ═══════════════════════════════════════════════════════════════════════════
# Kernel Event
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class KernelEvent:
    """An ephemeral kernel event for internal observability."""

    event_type: str
    source: str  # Which kernel component emitted this
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        return "failed" in self.event_type or "violation" in self.event_type or "exceeded" in self.event_type


# ═══════════════════════════════════════════════════════════════════════════
# Event Handler type
# ═══════════════════════════════════════════════════════════════════════════

KernelEventHandler = Callable[[KernelEvent], None]


# ═══════════════════════════════════════════════════════════════════════════
# Kernel Event Bus
# ═══════════════════════════════════════════════════════════════════════════


class KernelEventBus:
    """In-process pub/sub bus for kernel-level events.

    Supports:
        - Topic-based subscription (exact match)
        - Wildcard subscription ("*" receives all events)
        - Prefix subscription ("transaction.*" matches "transaction.committed")
        - Event history (configurable ring buffer)
    """

    def __init__(self, history_size: int = 1000) -> None:
        self._handlers: dict[str, list[KernelEventHandler]] = defaultdict(list)
        self._history: list[KernelEvent] = []
        self._history_size = history_size
        self._event_count = 0

    def subscribe(self, topic: str, handler: KernelEventHandler) -> None:
        """Subscribe a handler to a topic.

        Topics:
            "transaction.committed"  — exact match
            "transaction.*"         — prefix match
            "*"                     — all events
        """
        self._handlers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: KernelEventHandler) -> bool:
        """Remove a handler from a topic."""
        handlers = self._handlers.get(topic, [])
        try:
            handlers.remove(handler)
            return True
        except ValueError:
            return False

    def publish(self, event: KernelEvent) -> None:
        """Publish an event to all matching subscribers."""
        self._event_count += 1

        # Store in history
        self._history.append(event)
        if len(self._history) > self._history_size:
            self._history = self._history[-self._history_size:]

        # Dispatch to handlers
        matched = set()

        # 1. Exact match
        for handler in self._handlers.get(event.event_type, []):
            if id(handler) not in matched:
                matched.add(id(handler))
                self._safe_call(handler, event)

        # 2. Prefix match (e.g., "transaction.*")
        for topic, handlers in self._handlers.items():
            if topic.endswith(".*"):
                prefix = topic[:-2]
                if event.event_type.startswith(prefix + "."):
                    for handler in handlers:
                        if id(handler) not in matched:
                            matched.add(id(handler))
                            self._safe_call(handler, event)

        # 3. Wildcard
        for handler in self._handlers.get("*", []):
            if id(handler) not in matched:
                matched.add(id(handler))
                self._safe_call(handler, event)

    @staticmethod
    def _safe_call(handler: KernelEventHandler, event: KernelEvent) -> None:
        """Call a handler, catching and logging any exceptions."""
        try:
            handler(event)
        except Exception as exc:
            logger.error(
                "Kernel event handler error for %s: %s",
                event.event_type, exc,
            )

    def get_history(
        self,
        event_type: str | None = None,
        last_n: int = 50,
    ) -> list[KernelEvent]:
        """Retrieve recent events from the ring buffer."""
        events = self._history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-last_n:]

    @property
    def event_count(self) -> int:
        return self._event_count

    def stats(self) -> dict[str, Any]:
        return {
            "total_events": self._event_count,
            "history_size": len(self._history),
            "subscriptions": {
                topic: len(handlers)
                for topic, handlers in self._handlers.items()
            },
        }
