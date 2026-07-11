"""
Event Queue — priority queue with cognitive semantics.

Implements ``IEventQueue``.  Events are stored in a priority heap
ordered by ``effective_priority`` (descending) and drained in batches
by the ``ExecutiveController``.

Features:
    - Async-safe via ``asyncio.Lock``
    - Priority-ordered draining (highest first)
    - Capacity limit with automatic overflow logging
    - Tenant isolation (events tagged, not filtered — filtering is
      the controller's responsibility)
    - Queue statistics for observability

Usage::

    from hbllm.brain.control.event_queue import CognitiveEventQueue

    queue = CognitiveEventQueue(max_size=1000)
    await queue.submit(event)
    batch = await queue.drain(max_batch=10)
"""

from __future__ import annotations

import asyncio
import heapq
import logging
from dataclasses import dataclass, field
from typing import Any

from hbllm.brain.core.cognitive_event import CognitiveEvent
from hbllm.brain.core.cognitive_interfaces import IEventQueue

logger = logging.getLogger(__name__)


# ── Internal heap entry (negative priority for max-heap via heapq) ──


@dataclass(order=True)
class _HeapEntry:
    """Min-heap entry. We negate priority so highest-priority comes first."""

    sort_key: float  # -effective_priority for max-heap behavior
    insertion_order: int = field(compare=True)  # Tie-break by FIFO
    event: CognitiveEvent = field(compare=False)


# ── CognitiveEventQueue ─────────────────────────────────────────────────


class CognitiveEventQueue(IEventQueue):
    """Priority queue for cognitive events.

    Thread-safe (via asyncio.Lock) and priority-ordered.

    Args:
        max_size: Maximum queue capacity.  Events submitted when full
            are dropped with a warning.  Default 1000.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._heap: list[_HeapEntry] = []
        self._lock = asyncio.Lock()
        self._counter = 0
        self._max_size = max_size
        self._total_submitted = 0
        self._total_dropped = 0

        logger.debug("CognitiveEventQueue created (max_size=%d)", max_size)

    async def submit(self, event: Any) -> None:
        """Submit a cognitive event to the queue.

        If the queue is at capacity, the event is dropped and a
        warning is logged.

        Args:
            event: A ``CognitiveEvent`` instance.
        """
        if not isinstance(event, CognitiveEvent):
            raise TypeError(f"Expected CognitiveEvent, got {type(event).__name__}")

        async with self._lock:
            if len(self._heap) >= self._max_size:
                self._total_dropped += 1
                logger.warning(
                    "EventQueue overflow: dropping %r (queue=%d/%d, dropped=%d)",
                    event,
                    len(self._heap),
                    self._max_size,
                    self._total_dropped,
                )
                return

            entry = _HeapEntry(
                sort_key=-event.effective_priority,
                insertion_order=self._counter,
                event=event,
            )
            heapq.heappush(self._heap, entry)
            self._counter += 1
            self._total_submitted += 1

    async def drain(self, max_batch: int = 10) -> list[Any]:
        """Drain up to ``max_batch`` events, highest priority first.

        Args:
            max_batch: Maximum events to return.

        Returns:
            List of ``CognitiveEvent``, ordered by priority (desc).
        """
        async with self._lock:
            batch: list[CognitiveEvent] = []
            for _ in range(min(max_batch, len(self._heap))):
                entry = heapq.heappop(self._heap)
                batch.append(entry.event)
            return batch

    async def size(self) -> int:
        """Return the current number of events in the queue."""
        async with self._lock:
            return len(self._heap)

    async def peek(self, n: int = 1) -> list[CognitiveEvent]:
        """Peek at the top N events without removing them.

        Args:
            n: Number of events to peek at.

        Returns:
            Up to N events, priority-ordered. Does not remove.
        """
        async with self._lock:
            # nsmallest because we negated priority
            entries = heapq.nsmallest(n, self._heap)
            return [e.event for e in entries]

    async def clear(self) -> int:
        """Remove all events from the queue.

        Returns:
            Number of events cleared.
        """
        async with self._lock:
            count = len(self._heap)
            self._heap.clear()
            return count

    def stats(self) -> dict[str, Any]:
        """Queue statistics for observability."""
        return {
            "current_size": len(self._heap),
            "max_size": self._max_size,
            "total_submitted": self._total_submitted,
            "total_dropped": self._total_dropped,
        }
