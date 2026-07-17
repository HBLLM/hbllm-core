"""
Cognitive Scheduler — Central resource arbitration for background tasks.

The Cognitive Scheduler is the "OS scheduler" for HBLLM. It ensures that
background cognitive work (memory consolidation, goal planning, plugin
watchers, proactive reasoning) doesn't starve the interactive path.

Architecture::

    Interactive Path (user messages)
        ↓
    ConversationBus  →  Executive  (PRIORITY: HIGH)

    Background Path (autonomous work)
        ↓
    CognitiveScheduler  →  TaskQueue  →  Worker Pool  (PRIORITY: LOW)
        ↓
    Arbiter decides: run now / defer / drop

Resources managed:
    - LLM inference slots (bounded by concurrent call limit)
    - Memory I/O (reads/writes to SQLite/vector DBs)
    - CPU-bound work (SNN, code execution)
    - Network I/O (API calls, web research)

Priority levels::

    INTERACTIVE  → User is waiting for a response (immediate)
    BACKGROUND   → Autonomous tasks (can wait 5-30s)
    MAINTENANCE  → Consolidation, cleanup, compaction (can wait minutes)
    IDLE         → Curiosity, exploration (only when fully idle)

Usage::

    from hbllm.serving.cognitive_scheduler import CognitiveScheduler

    scheduler = CognitiveScheduler(max_concurrent_llm=3)
    await scheduler.start()

    # Submit interactive work (bypasses queue)
    result = await scheduler.submit_interactive(coro)

    # Submit background work
    task_id = await scheduler.submit_background(coro, name="consolidation")

    # Submit maintenance work
    task_id = await scheduler.schedule_recurring(
        coro_factory, interval_s=300, name="memory-compaction"
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Priority & Task State
# ═══════════════════════════════════════════════════════════════════════════


class TaskPriority(IntEnum):
    """Scheduling priority (higher value = higher priority)."""

    IDLE = 0  # Curiosity, exploration
    MAINTENANCE = 1  # Consolidation, compaction
    BACKGROUND = 2  # Autonomous reasoning, plugin work
    INTERACTIVE = 3  # User-facing responses


class TaskState:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEFERRED = "deferred"


# ═══════════════════════════════════════════════════════════════════════════
# Scheduled Task
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ScheduledTask:
    """A task managed by the CognitiveScheduler."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = ""
    priority: TaskPriority = TaskPriority.BACKGROUND
    state: str = TaskState.PENDING
    coro: Any = None  # The coroutine to execute
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    timeout_s: float = 120.0

    @property
    def elapsed_ms(self) -> float:
        if self.started_at == 0:
            return 0.0
        end = self.completed_at or time.time()
        return (end - self.started_at) * 1000


# ═══════════════════════════════════════════════════════════════════════════
# Resource Slots
# ═══════════════════════════════════════════════════════════════════════════


class ResourceSlots:
    """Bounded semaphore pool for different resource types."""

    def __init__(
        self,
        max_llm: int = 3,
        max_memory_io: int = 5,
        max_cpu: int = 2,
        max_network: int = 5,
    ) -> None:
        self.llm = asyncio.Semaphore(max_llm)
        self.memory_io = asyncio.Semaphore(max_memory_io)
        self.cpu = asyncio.Semaphore(max_cpu)
        self.network = asyncio.Semaphore(max_network)

        self._counts = {
            "llm": max_llm,
            "memory_io": max_memory_io,
            "cpu": max_cpu,
            "network": max_network,
        }

    def stats(self) -> dict[str, Any]:
        return {k: v for k, v in self._counts.items()}


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive Scheduler
# ═══════════════════════════════════════════════════════════════════════════


class CognitiveScheduler:
    """Central resource arbitrator for background cognitive tasks.

    Ensures background work (consolidation, planning, proactive
    reasoning, plugin watchers) doesn't starve the interactive path.
    """

    def __init__(
        self,
        max_concurrent_llm: int = 3,
        max_concurrent_background: int = 5,
        max_queue_size: int = 100,
    ) -> None:
        self._resources = ResourceSlots(max_llm=max_concurrent_llm)
        self._max_concurrent_bg = max_concurrent_background
        self._max_queue = max_queue_size

        # Priority queue (sorted by priority descending, then creation time)
        self._queue: asyncio.PriorityQueue[tuple[int, float, ScheduledTask]] = (
            asyncio.PriorityQueue(maxsize=max_queue_size)
        )

        # Active tasks
        self._active: dict[str, ScheduledTask] = {}
        self._completed: list[ScheduledTask] = []  # Ring buffer of last 50

        # Background workers
        self._workers: list[asyncio.Task[None]] = []
        self._recurring: dict[str, asyncio.Task[None]] = {}

        self._started = False
        self._interactive_count = 0
        self._background_count = 0

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self, num_workers: int = 3) -> None:
        """Start the scheduler with background worker pool."""
        if self._started:
            return

        for i in range(num_workers):
            worker = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self._workers.append(worker)

        self._started = True
        logger.info(
            "CognitiveScheduler started (%d workers, %d max LLM slots)",
            num_workers,
            self._resources._counts["llm"],
        )

    async def stop(self) -> None:
        """Stop all workers and cancel pending tasks."""
        self._started = False

        # Cancel recurring tasks
        for name, task in self._recurring.items():
            task.cancel()
        self._recurring.clear()

        # Cancel workers
        for w in self._workers:
            w.cancel()
        self._workers.clear()

        # Cancel active tasks
        for task_id, task in self._active.items():
            task.state = TaskState.CANCELLED
        self._active.clear()

        logger.info("CognitiveScheduler stopped")

    # ── Submit Methods ───────────────────────────────────────────────────

    async def submit_interactive(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Submit work on the interactive (user-facing) path.

        Bypasses the queue entirely — runs immediately with resource
        acquisition. Interactive work always gets priority.

        Args:
            coro: Coroutine to execute.

        Returns:
            Result of the coroutine.
        """
        self._interactive_count += 1
        async with self._resources.llm:
            return await coro

    async def submit_background(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        name: str = "",
        priority: TaskPriority = TaskPriority.BACKGROUND,
        timeout_s: float = 120.0,
    ) -> str:
        """Submit background work to the scheduler queue.

        Args:
            coro: Coroutine to execute when resources are available.
            name: Human-readable task name.
            priority: Scheduling priority.
            timeout_s: Maximum execution time.

        Returns:
            Task ID for tracking.
        """
        task = ScheduledTask(
            name=name or f"bg-{self._background_count}",
            priority=priority,
            coro=coro,
            timeout_s=timeout_s,
        )

        # Priority queue sorts ascending — negate priority for descending
        sort_key = (-int(priority), task.created_at)
        try:
            self._queue.put_nowait((sort_key[0], sort_key[1], task))
        except asyncio.QueueFull:
            logger.warning(
                "Scheduler queue full — dropping task '%s'", task.name
            )
            task.state = TaskState.CANCELLED
            return task.id

        self._background_count += 1
        logger.debug("Queued background task '%s' (priority=%s)", task.name, priority.name)
        return task.id

    async def submit_maintenance(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        name: str = "",
        timeout_s: float = 300.0,
    ) -> str:
        """Submit maintenance work (lowest interactive priority)."""
        return await self.submit_background(
            coro,
            name=name,
            priority=TaskPriority.MAINTENANCE,
            timeout_s=timeout_s,
        )

    def schedule_recurring(
        self,
        coro_factory: Callable[[], Coroutine[Any, Any, Any]],
        interval_s: float,
        *,
        name: str = "recurring",
        priority: TaskPriority = TaskPriority.MAINTENANCE,
    ) -> str:
        """Schedule a recurring task.

        Args:
            coro_factory: Callable that creates a new coroutine each interval.
            interval_s: Seconds between invocations.
            name: Task name.
            priority: Priority for each invocation.

        Returns:
            Recurring task ID (use to cancel).
        """
        task_id = str(uuid.uuid4())[:12]

        async def _loop() -> None:
            while self._started:
                try:
                    await asyncio.sleep(interval_s)
                    if self._started:
                        coro = coro_factory()
                        await self.submit_background(
                            coro, name=f"{name}-tick", priority=priority
                        )
                except asyncio.CancelledError:
                    return
                except Exception:
                    logger.exception("Error in recurring task '%s'", name)
                    await asyncio.sleep(5.0)

        self._recurring[task_id] = asyncio.create_task(_loop())
        logger.info(
            "Scheduled recurring task '%s' every %.0fs (id=%s)",
            name,
            interval_s,
            task_id,
        )
        return task_id

    def cancel_recurring(self, task_id: str) -> bool:
        """Cancel a recurring task."""
        task = self._recurring.pop(task_id, None)
        if task:
            task.cancel()
            return True
        return False

    # ── Worker Loop ──────────────────────────────────────────────────────

    async def _worker_loop(self, worker_name: str) -> None:
        """Background worker that processes the task queue."""
        while self._started:
            try:
                # Block until a task is available
                neg_priority, created_at, task = await asyncio.wait_for(
                    self._queue.get(), timeout=5.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                return

            if task.state == TaskState.CANCELLED:
                continue

            task.state = TaskState.RUNNING
            task.started_at = time.time()
            self._active[task.id] = task

            try:
                async with self._resources.llm:
                    task.result = await asyncio.wait_for(
                        task.coro, timeout=task.timeout_s
                    )
                task.state = TaskState.COMPLETED
            except asyncio.TimeoutError:
                task.state = TaskState.FAILED
                task.error = f"Timed out after {task.timeout_s}s"
                logger.warning("Task '%s' timed out", task.name)
            except asyncio.CancelledError:
                task.state = TaskState.CANCELLED
                return
            except Exception as e:
                task.state = TaskState.FAILED
                task.error = str(e)
                logger.exception("Task '%s' failed", task.name)
            finally:
                task.completed_at = time.time()
                self._active.pop(task.id, None)
                self._completed.append(task)
                # Keep only last 50
                if len(self._completed) > 50:
                    self._completed = self._completed[-50:]

    # ── Introspection ────────────────────────────────────────────────────

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    def stats(self) -> dict[str, Any]:
        """Scheduler statistics."""
        completed = [t for t in self._completed if t.state == TaskState.COMPLETED]
        failed = [t for t in self._completed if t.state == TaskState.FAILED]
        return {
            "active_tasks": len(self._active),
            "queued_tasks": self._queue.qsize(),
            "recurring_tasks": len(self._recurring),
            "interactive_served": self._interactive_count,
            "background_submitted": self._background_count,
            "completed": len(completed),
            "failed": len(failed),
            "avg_latency_ms": (
                sum(t.elapsed_ms for t in completed) / len(completed)
                if completed
                else 0.0
            ),
            "resources": self._resources.stats(),
        }

    def get_active_tasks(self) -> list[dict[str, Any]]:
        """Get details of currently running tasks."""
        return [
            {
                "id": t.id,
                "name": t.name,
                "priority": t.priority.name,
                "state": t.state,
                "elapsed_ms": t.elapsed_ms,
            }
            for t in self._active.values()
        ]
