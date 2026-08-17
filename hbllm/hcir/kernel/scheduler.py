"""
Cognitive Scheduler — attention-driven instruction dispatch.

Attention is the direct input to the scheduler.
No hidden heuristics — priority comes from HCIR runtime state.

    Attention → Priority Queue → Scheduler → Execution

The scheduler manages lightweight Process/Thread abstractions
to track concurrent cognitive execution streams.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from heapq import heappop, heappush

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Process & Thread Abstractions
# ═══════════════════════════════════════════════════════════════════════════


class ProcessState(StrEnum):
    """Lifecycle states of a cognitive process."""

    READY = "ready"
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CognitiveThread:
    """A lightweight execution thread within a cognitive process.

    Holds an instruction stream reference and execution cursor.
    """

    thread_id: str = field(default_factory=lambda: f"thr_{uuid.uuid4().hex[:8]}")
    instruction_stream_ref: str = ""  # Reference to bytecode stream
    program_counter: int = 0
    state: ProcessState = ProcessState.READY
    created_at: float = field(default_factory=time.time)


@dataclass
class CognitiveProcess:
    """A cognitive process groups related threads.

    Maps to a conversation, a goal resolution, or a simulation run.
    """

    process_id: str = field(default_factory=lambda: f"proc_{uuid.uuid4().hex[:8]}")
    conversation_id: str = ""
    tenant_id: str = "default"
    threads: list[CognitiveThread] = field(default_factory=list)
    state: ProcessState = ProcessState.READY
    priority: float = 0.5  # From attention.salience
    created_at: float = field(default_factory=time.time)

    def add_thread(self, thread: CognitiveThread) -> None:
        self.threads.append(thread)

    @property
    def active_threads(self) -> list[CognitiveThread]:
        return [t for t in self.threads if t.state in (ProcessState.READY, ProcessState.RUNNING)]


# ═══════════════════════════════════════════════════════════════════════════
# Scheduler Entry (Priority Queue Item)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(order=True)
class SchedulerEntry:
    """A prioritized entry in the scheduler's dispatch queue.

    Lower ``sort_key`` = higher priority (dispatched first).
    """

    sort_key: float  # -(salience * activation) for max-heap behavior
    process_id: str = field(compare=False)
    thread_id: str = field(compare=False)
    enqueued_at: float = field(compare=False, default_factory=time.time)


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive Scheduler
# ═══════════════════════════════════════════════════════════════════════════


class CognitiveBudget:
    """Token budget for scheduling constraints.

    Tracks estimated and actual token consumption.  The scheduler
    refuses to dispatch processes that would exceed the budget.
    """

    def __init__(self, total_tokens: int = 100_000) -> None:
        self.total_tokens = total_tokens
        self.consumed_tokens: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.total_tokens - self.consumed_tokens)

    @property
    def utilization(self) -> float:
        if self.total_tokens == 0:
            return 1.0
        return self.consumed_tokens / self.total_tokens

    def can_afford(self, cost: int) -> bool:
        return self.consumed_tokens + cost <= self.total_tokens

    def consume(self, cost: int) -> bool:
        """Consume tokens. Returns False if budget exceeded."""
        if not self.can_afford(cost):
            return False
        self.consumed_tokens += cost
        return True

    def reset(self) -> None:
        self.consumed_tokens = 0


class KernelInstructionScheduler:
    """Attention-driven instruction scheduler with budget awareness.

    Dispatches cognitive instruction streams based on attention
    salience, resource availability, process priority, and
    token budget constraints.

    Usage::

        budget = CognitiveBudget(total_tokens=50000)
        scheduler = KernelInstructionScheduler(budget=budget)
        proc = CognitiveProcess(conversation_id="conv_123")
        thread = CognitiveThread(instruction_stream_ref="stream_42")
        proc.add_thread(thread)
        scheduler.register_process(proc)
        scheduler.enqueue(proc.process_id, thread.thread_id, salience=0.9)

        entry = scheduler.dispatch()  # Returns highest priority entry
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        budget: CognitiveBudget | None = None,
    ) -> None:
        self._processes: dict[str, CognitiveProcess] = {}
        self._queue: list[SchedulerEntry] = []  # Min-heap
        self._max_concurrent = max_concurrent
        self._running_count = 0
        self._budget = budget

    def register_process(self, process: CognitiveProcess) -> None:
        """Register a cognitive process."""
        self._processes[process.process_id] = process

    def unregister_process(self, process_id: str) -> CognitiveProcess | None:
        """Remove a process from the scheduler."""
        return self._processes.pop(process_id, None)

    def get_process(self, process_id: str) -> CognitiveProcess | None:
        return self._processes.get(process_id)

    def enqueue(
        self,
        process_id: str,
        thread_id: str,
        salience: float = 0.5,
        activation: float = 0.5,
    ) -> None:
        """Enqueue a thread for dispatch.

        Priority is derived from attention parameters:
        higher salience × activation = dispatched sooner.
        """
        # Negate for min-heap (highest priority first)
        sort_key = -(salience * activation)
        entry = SchedulerEntry(
            sort_key=sort_key,
            process_id=process_id,
            thread_id=thread_id,
        )
        heappush(self._queue, entry)

    def dispatch(self) -> SchedulerEntry | None:
        """Dispatch the highest-priority entry.

        Returns None if the queue is empty or max concurrency reached.
        """
        if self._running_count >= self._max_concurrent:
            return None
        if not self._queue:
            return None

        entry = heappop(self._queue)
        self._running_count += 1

        # Update process/thread state
        proc = self._processes.get(entry.process_id)
        if proc:
            proc.state = ProcessState.RUNNING
            for thread in proc.threads:
                if thread.thread_id == entry.thread_id:
                    thread.state = ProcessState.RUNNING
                    break

        return entry

    def complete(self, process_id: str, thread_id: str) -> None:
        """Mark a thread as completed."""
        self._running_count = max(0, self._running_count - 1)
        proc = self._processes.get(process_id)
        if proc:
            for thread in proc.threads:
                if thread.thread_id == thread_id:
                    thread.state = ProcessState.COMPLETED
                    break
            # If all threads completed, mark process completed
            if not proc.active_threads:
                proc.state = ProcessState.COMPLETED

    def fail(self, process_id: str, thread_id: str) -> None:
        """Mark a thread as failed."""
        self._running_count = max(0, self._running_count - 1)
        proc = self._processes.get(process_id)
        if proc:
            for thread in proc.threads:
                if thread.thread_id == thread_id:
                    thread.state = ProcessState.FAILED
                    break

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    @property
    def running_count(self) -> int:
        return self._running_count

    @property
    def process_count(self) -> int:
        return len(self._processes)

    @property
    def budget(self) -> CognitiveBudget | None:
        return self._budget

    def enqueue_with_cost(
        self,
        process_id: str,
        thread_id: str,
        salience: float = 0.5,
        activation: float = 0.5,
        estimated_cost: int = 0,
    ) -> bool:
        """Enqueue a thread with budget checking.

        Returns False if the budget would be exceeded.
        """
        if self._budget is not None and estimated_cost > 0:
            if not self._budget.can_afford(estimated_cost):
                logger.warning(
                    "Budget exceeded: cannot enqueue thread %s (cost=%d, remaining=%d)",
                    thread_id,
                    estimated_cost,
                    self._budget.remaining,
                )
                return False
            self._budget.consume(estimated_cost)

        self.enqueue(process_id, thread_id, salience, activation)
        return True
