"""
Budget-Aware Cognitive Scheduler — ADR 002 §6.

Arbitrates execution priorities via 5 strict task priority classes
combined with resource budget tracking (CPU, RAM, VRAM, attention).

Priority classes (highest to lowest):
    1. USER_INTERACTIVE — Real-time STT/TTS turn management
    2. SAFETY_CRITICAL — Cognitive Firewall & Restraint Engine checks
    3. LATENCY_SENSITIVE — Active tool execution & LLM inference
    4. BACKGROUND — Vector embeddings & knowledge graph building
    5. MAINTENANCE — Sleep cycle compaction, memory pruning

Design invariants (ADR 002):
    - Higher priority tasks always preempt lower priority queues.
    - Resource budgets prevent starvation of lower-priority tasks.
    - Tasks exceeding their budget are downgraded or deferred.
    - Scheduler statistics are exposed for DigitalTwin telemetry.

Usage::

    from hbllm.brain.control.cognitive_scheduler import (
        CognitiveScheduler, TaskPriority, ScheduledTask, ResourceBudget,
    )

    scheduler = CognitiveScheduler()
    task_id = scheduler.submit(ScheduledTask(
        name="process_audio",
        priority=TaskPriority.USER_INTERACTIVE,
        coroutine=handle_audio(),
        budget=ResourceBudget(cpu_shares=0.3),
    ))
    executed = await scheduler.run_next()
"""

from __future__ import annotations

import heapq
import logging
import time
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

logger = logging.getLogger(__name__)


class TaskPriority(IntEnum):
    """Cognitive task priority levels (lower value = higher priority)."""

    USER_INTERACTIVE = 0
    SAFETY_CRITICAL = 1
    LATENCY_SENSITIVE = 2
    BACKGROUND = 3
    MAINTENANCE = 4


class TaskStatus(str):
    """Task lifecycle status constants."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEFERRED = "deferred"


@dataclass(frozen=True)
class ResourceBudget:
    """Resource allocation contract for a scheduled task.

    Each value is a fraction [0.0, 1.0] representing the maximum share
    of that resource the task may consume.

    Attributes:
        cpu_shares: Maximum CPU utilization fraction.
        ram_mb: Maximum RAM allocation in megabytes.
        vram_mb: Maximum VRAM allocation in megabytes.
        attention_units: Maximum attention units (abstract budget).
        network_kbps: Maximum network bandwidth in KB/s.
    """

    cpu_shares: float = 0.0
    ram_mb: float = 0.0
    vram_mb: float = 0.0
    attention_units: float = 0.0
    network_kbps: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "cpu_shares": self.cpu_shares,
            "ram_mb": self.ram_mb,
            "vram_mb": self.vram_mb,
            "attention_units": self.attention_units,
            "network_kbps": self.network_kbps,
        }


# Default budgets per priority tier
DEFAULT_BUDGETS: dict[TaskPriority, ResourceBudget] = {
    TaskPriority.USER_INTERACTIVE: ResourceBudget(cpu_shares=0.4, ram_mb=512, attention_units=1.0),
    TaskPriority.SAFETY_CRITICAL: ResourceBudget(cpu_shares=0.3, ram_mb=256, attention_units=0.8),
    TaskPriority.LATENCY_SENSITIVE: ResourceBudget(cpu_shares=0.2, ram_mb=256, attention_units=0.5),
    TaskPriority.BACKGROUND: ResourceBudget(cpu_shares=0.1, ram_mb=128, attention_units=0.2),
    TaskPriority.MAINTENANCE: ResourceBudget(cpu_shares=0.05, ram_mb=64, attention_units=0.1),
}


@dataclass(order=True)
class ScheduledTask:
    """A task submitted to the CognitiveScheduler.

    Attributes:
        task_id: Globally unique identifier.
        name: Human-readable task name.
        priority: Execution priority level.
        budget: Resource budget contract.
        status: Current lifecycle status.
        created_at: Submission timestamp.
        started_at: Execution start timestamp.
        completed_at: Completion timestamp.
        result: Task result (populated after completion).
        error: Error message (populated on failure).
    """

    # order=True sorts by (priority, created_at) for the heap
    priority: TaskPriority = field(default=TaskPriority.BACKGROUND)
    created_at: float = field(default_factory=time.time)
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex, compare=False)
    name: str = field(default="", compare=False)
    budget: ResourceBudget = field(default_factory=ResourceBudget, compare=False)
    status: str = field(default=TaskStatus.PENDING, compare=False)
    started_at: float = field(default=0.0, compare=False)
    completed_at: float = field(default=0.0, compare=False)
    result: Any = field(default=None, compare=False, repr=False)
    error: str = field(default="", compare=False)

    # The coroutine to execute — not serializable, excluded from comparison
    _coro: Coroutine[Any, Any, Any] | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "priority": self.priority.name,
            "status": self.status,
            "budget": self.budget.to_dict(),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


class CognitiveScheduler:
    """Budget-aware priority scheduler for cognitive tasks.

    Maintains 5 priority-stratified queues and enforces resource
    budget contracts per task tier.

    Args:
        max_concurrent: Maximum number of concurrently running tasks.
        global_budget: Total system-wide resource budget.
    """

    def __init__(
        self,
        max_concurrent: int = 5,
        global_budget: ResourceBudget | None = None,
    ) -> None:
        self._max_concurrent = max_concurrent
        self._global_budget = global_budget or ResourceBudget(
            cpu_shares=1.0,
            ram_mb=2048,
            vram_mb=0,
            attention_units=1.0,
        )

        # Priority heap (min-heap by priority, then creation time)
        self._queue: list[ScheduledTask] = []
        self._running: dict[str, ScheduledTask] = {}
        self._completed: list[ScheduledTask] = []

        # Resource tracking
        self._allocated = ResourceBudget()

        # Telemetry
        self._total_submitted = 0
        self._total_completed = 0
        self._total_failed = 0
        self._total_deferred = 0

        logger.info(
            "CognitiveScheduler initialized (max_concurrent=%d)",
            max_concurrent,
        )

    # ── Submission ───────────────────────────────────────────────────

    def submit(
        self,
        name: str,
        priority: TaskPriority,
        coro: Coroutine[Any, Any, Any],
        budget: ResourceBudget | None = None,
    ) -> str:
        """Submit a task to the scheduler.

        Args:
            name: Human-readable task name.
            priority: Execution priority level.
            coro: Async coroutine to execute.
            budget: Resource budget (defaults to tier default).

        Returns:
            task_id of the submitted task.
        """
        task = ScheduledTask(
            priority=priority,
            name=name,
            budget=budget or DEFAULT_BUDGETS.get(priority, ResourceBudget()),
            _coro=coro,
        )
        heapq.heappush(self._queue, task)
        self._total_submitted += 1
        logger.debug(
            "Task submitted: %s (priority=%s, id=%s)",
            name,
            priority.name,
            task.task_id[:8],
        )
        return task.task_id

    # ── Execution ────────────────────────────────────────────────────

    async def run_next(self) -> ScheduledTask | None:
        """Execute the highest-priority pending task.

        Checks resource budgets before starting. If the budget would
        be exceeded, the task is deferred.

        Returns:
            The completed/failed task, or None if queue is empty.
        """
        if len(self._running) >= self._max_concurrent:
            return None

        if not self._queue:
            return None

        task = heapq.heappop(self._queue)

        # Check resource budget
        if not self._can_allocate(task.budget):
            task.status = TaskStatus.DEFERRED
            self._total_deferred += 1
            # Re-add with slightly delayed timestamp
            task = ScheduledTask(
                priority=task.priority,
                created_at=time.time(),
                task_id=task.task_id,
                name=task.name,
                budget=task.budget,
                status=TaskStatus.PENDING,
                _coro=task._coro,
            )
            heapq.heappush(self._queue, task)
            logger.debug("Task deferred (budget exceeded): %s", task.name)
            return None

        # Execute
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        self._running[task.task_id] = task
        self._allocate(task.budget)

        try:
            if task._coro is not None:
                task.result = await task._coro
            task.status = TaskStatus.COMPLETED
            self._total_completed += 1
        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.error = str(exc)
            self._total_failed += 1
            logger.error("Task failed: %s — %s", task.name, exc)
        finally:
            task.completed_at = time.time()
            self._running.pop(task.task_id, None)
            self._deallocate(task.budget)
            self._completed.append(task)

        return task

    async def run_all_pending(self) -> list[ScheduledTask]:
        """Execute all pending tasks in priority order.

        Returns:
            List of completed/failed tasks.
        """
        results: list[ScheduledTask] = []
        while self._queue:
            result = await self.run_next()
            if result is None:
                break
            results.append(result)
        return results

    # ── Resource budget tracking ─────────────────────────────────────

    def _can_allocate(self, budget: ResourceBudget) -> bool:
        """Check if allocating this budget would exceed global limits."""
        return (
            self._allocated.cpu_shares + budget.cpu_shares <= self._global_budget.cpu_shares
            and self._allocated.ram_mb + budget.ram_mb <= self._global_budget.ram_mb
            and self._allocated.attention_units + budget.attention_units
            <= self._global_budget.attention_units
        )

    def _allocate(self, budget: ResourceBudget) -> None:
        """Add budget to the currently allocated resources."""
        self._allocated = ResourceBudget(
            cpu_shares=self._allocated.cpu_shares + budget.cpu_shares,
            ram_mb=self._allocated.ram_mb + budget.ram_mb,
            vram_mb=self._allocated.vram_mb + budget.vram_mb,
            attention_units=self._allocated.attention_units + budget.attention_units,
            network_kbps=self._allocated.network_kbps + budget.network_kbps,
        )

    def _deallocate(self, budget: ResourceBudget) -> None:
        """Remove budget from currently allocated resources."""
        self._allocated = ResourceBudget(
            cpu_shares=max(0.0, self._allocated.cpu_shares - budget.cpu_shares),
            ram_mb=max(0.0, self._allocated.ram_mb - budget.ram_mb),
            vram_mb=max(0.0, self._allocated.vram_mb - budget.vram_mb),
            attention_units=max(0.0, self._allocated.attention_units - budget.attention_units),
            network_kbps=max(0.0, self._allocated.network_kbps - budget.network_kbps),
        )

    # ── Query ────────────────────────────────────────────────────────

    def pending_count(self) -> int:
        """Number of tasks in the pending queue."""
        return len(self._queue)

    def running_count(self) -> int:
        """Number of currently running tasks."""
        return len(self._running)

    def cancel(self, task_id: str) -> bool:
        """Cancel a pending task by ID. Returns True if found."""
        for i, task in enumerate(self._queue):
            if task.task_id == task_id:
                self._queue.pop(i)
                heapq.heapify(self._queue)
                return True
        return False

    # ── Telemetry ────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Scheduler statistics for DigitalTwin telemetry."""
        return {
            "total_submitted": self._total_submitted,
            "total_completed": self._total_completed,
            "total_failed": self._total_failed,
            "total_deferred": self._total_deferred,
            "pending_count": len(self._queue),
            "running_count": len(self._running),
            "allocated_resources": self._allocated.to_dict(),
            "global_budget": self._global_budget.to_dict(),
            "queue_by_priority": self._queue_breakdown(),
        }

    def _queue_breakdown(self) -> dict[str, int]:
        """Count pending tasks by priority level."""
        counts: dict[str, int] = {p.name: 0 for p in TaskPriority}
        for task in self._queue:
            counts[task.priority.name] += 1
        return counts
