"""
Cognitive Load Manager — system resource monitoring and graceful degradation.

Monitors CPU, memory, and task queue pressure. When the system is overloaded,
it degrades gracefully by:
  - Reducing context window size
  - Disabling expensive nodes (simulation, deep reflection)
  - Queuing non-critical requests
  - Switching to smaller/faster model configurations

This ensures HBLLM remains responsive under resource constraints,
which is critical for the edge/CPU deployment target.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


@dataclass
class SystemResources:
    """Snapshot of current system resource state."""

    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    active_tasks: int = 0
    queue_depth: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def pressure_level(self) -> str:
        """Classify resource pressure: normal, elevated, high, critical."""
        max_pressure = max(self.cpu_percent, self.memory_percent)
        if max_pressure > 90:
            return "critical"
        if max_pressure > 75:
            return "high"
        if max_pressure > 60:
            return "elevated"
        return "normal"


@dataclass
class DegradationPolicy:
    """Policy for graceful degradation at each pressure level."""

    level: str
    max_context_tokens: int
    enable_simulation: bool
    enable_deep_reflection: bool
    enable_planner: bool
    max_concurrent_tasks: int
    model_preference: str  # "large", "small", "tiny"

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "max_context_tokens": self.max_context_tokens,
            "enable_simulation": self.enable_simulation,
            "enable_deep_reflection": self.enable_deep_reflection,
            "enable_planner": self.enable_planner,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "model_preference": self.model_preference,
        }


# Default degradation policies
DEFAULT_POLICIES: dict[str, DegradationPolicy] = {
    "normal": DegradationPolicy(
        level="normal",
        max_context_tokens=4096,
        enable_simulation=True,
        enable_deep_reflection=True,
        enable_planner=True,
        max_concurrent_tasks=8,
        model_preference="large",
    ),
    "elevated": DegradationPolicy(
        level="elevated",
        max_context_tokens=2048,
        enable_simulation=True,
        enable_deep_reflection=True,
        enable_planner=True,
        max_concurrent_tasks=4,
        model_preference="large",
    ),
    "high": DegradationPolicy(
        level="high",
        max_context_tokens=1024,
        enable_simulation=False,
        enable_deep_reflection=False,
        enable_planner=True,
        max_concurrent_tasks=2,
        model_preference="small",
    ),
    "critical": DegradationPolicy(
        level="critical",
        max_context_tokens=512,
        enable_simulation=False,
        enable_deep_reflection=False,
        enable_planner=False,
        max_concurrent_tasks=1,
        model_preference="tiny",
    ),
}


class LoadManager(Node):
    """
    Manages cognitive load and system resource pressure.

    Subscribes to:
        system.task.started — tracks active tasks
        system.task.completed — releases task slots
        load.query — returns current status

    Publishes:
        system.load.pressure_change — when pressure level changes
        system.load.policy_update — when degradation policy changes
    """

    def __init__(
        self,
        node_id: str,
        monitor_interval: float = 30.0,
        policies: dict[str, DegradationPolicy] | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.META,
            capabilities=["load_management", "resource_monitoring"],
        )
        self.monitor_interval = monitor_interval
        self.policies = policies or DEFAULT_POLICIES

        self._resources = SystemResources()
        self._current_policy = self.policies["normal"]
        self._active_tasks: set[str] = set()
        self._task_queue: list[dict[str, Any]] = []
        self._max_queue = 50

        # Monitoring
        self._monitor_task: asyncio.Task[None] | None = None
        self._running = False
        self._pressure_changes = 0
        self._tasks_queued = 0
        self._tasks_processed = 0

    async def on_start(self) -> None:
        logger.info("Starting LoadManager (interval=%.0fs)", self.monitor_interval)
        await self.bus.subscribe("system.task.started", self._handle_task_start)
        await self.bus.subscribe("system.task.completed", self._handle_task_complete)
        await self.bus.subscribe("load.query", self._handle_query)

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def on_stop(self) -> None:
        self._running = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info(
            "Stopping LoadManager — pressure_changes=%d tasks_queued=%d",
            self._pressure_changes,
            self._tasks_queued,
        )

    async def handle_message(self, message: Message) -> Message | None:
        return None

    # ── Resource Monitoring ──────────────────────────────────────────

    async def _monitor_loop(self) -> None:
        """Periodically check system resources."""
        while self._running:
            await asyncio.sleep(self.monitor_interval)
            if not self._running:
                break
            try:
                self._resources = self._sample_resources()
                await self._evaluate_pressure()
            except Exception as e:
                logger.debug("[LoadManager] Monitor error: %s", e)

    def _sample_resources(self) -> SystemResources:
        """Sample current system resources."""
        cpu = 0.0
        mem = 0.0
        disk = 0.0

        try:
            import psutil  # type: ignore[import-untyped]

            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory().percent
            disk = psutil.disk_usage("/").percent
        except ImportError:
            # psutil not available — use conservative defaults
            cpu = 30.0
            mem = 50.0
            disk = 50.0
        except Exception as e:
            logger.debug("Resource sampling error: %s", e)

        return SystemResources(
            cpu_percent=cpu,
            memory_percent=mem,
            disk_percent=disk,
            active_tasks=len(self._active_tasks),
            queue_depth=len(self._task_queue),
        )

    async def _evaluate_pressure(self) -> None:
        """Evaluate pressure level and update policy if needed."""
        new_level = self._resources.pressure_level
        current_level = self._current_policy.level

        if new_level != current_level:
            self._current_policy = self.policies[new_level]
            self._pressure_changes += 1

            logger.info(
                "[LoadManager] Pressure: %s → %s (CPU=%.0f%% MEM=%.0f%%)",
                current_level,
                new_level,
                self._resources.cpu_percent,
                self._resources.memory_percent,
            )

            # Publish pressure change
            await self.bus.publish(
                "system.load.pressure_change",
                Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    topic="system.load.pressure_change",
                    payload={
                        "old_level": current_level,
                        "new_level": new_level,
                        "resources": {
                            "cpu": self._resources.cpu_percent,
                            "memory": self._resources.memory_percent,
                        },
                    },
                ),
            )

            # Publish policy update
            await self.bus.publish(
                "system.load.policy_update",
                Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    topic="system.load.policy_update",
                    payload=self._current_policy.to_dict(),
                ),
            )

    # ── Task Management ──────────────────────────────────────────────

    def can_accept_task(self) -> bool:
        """Check if a new task can be accepted under current policy."""
        return len(self._active_tasks) < self._current_policy.max_concurrent_tasks

    def queue_task(self, task_id: str, priority: float = 0.5) -> bool:
        """Queue a task for later processing."""
        if len(self._task_queue) >= self._max_queue:
            return False

        self._task_queue.append({
            "task_id": task_id,
            "priority": priority,
            "queued_at": time.time(),
        })
        self._task_queue.sort(key=lambda t: t["priority"], reverse=True)
        self._tasks_queued += 1
        return True

    def dequeue_task(self) -> dict[str, Any] | None:
        """Get the next task from the queue."""
        if not self._task_queue:
            return None
        if not self.can_accept_task():
            return None
        return self._task_queue.pop(0)

    async def _handle_task_start(self, message: Message) -> None:
        """Track task start."""
        task_id = message.payload.get("task_id", message.id)
        self._active_tasks.add(task_id)
        self._tasks_processed += 1

    async def _handle_task_complete(self, message: Message) -> None:
        """Track task completion and maybe dequeue next."""
        task_id = message.payload.get("task_id", message.id)
        self._active_tasks.discard(task_id)

        # Try to process queued tasks
        next_task = self.dequeue_task()
        if next_task:
            logger.info(
                "[LoadManager] Dequeuing task %s (queue=%d)",
                next_task["task_id"],
                len(self._task_queue),
            )

    async def _handle_query(self, message: Message) -> Message | None:
        """Return load manager status."""
        return message.create_response(self.stats())

    # ── Public API ───────────────────────────────────────────────────

    @property
    def current_policy(self) -> DegradationPolicy:
        """Get the active degradation policy."""
        return self._current_policy

    @property
    def pressure_level(self) -> str:
        """Get current pressure level."""
        return self._resources.pressure_level

    def get_max_context_tokens(self) -> int:
        """Get the max context tokens allowed under current pressure."""
        return self._current_policy.max_context_tokens

    def is_simulation_enabled(self) -> bool:
        """Check if simulation is enabled under current load."""
        return self._current_policy.enable_simulation

    def get_model_preference(self) -> str:
        """Get recommended model size under current pressure."""
        return self._current_policy.model_preference

    # ── Stats ────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "pressure_level": self._current_policy.level,
            "resources": {
                "cpu_percent": self._resources.cpu_percent,
                "memory_percent": self._resources.memory_percent,
                "disk_percent": self._resources.disk_percent,
            },
            "active_policy": self._current_policy.to_dict(),
            "active_tasks": len(self._active_tasks),
            "queue_depth": len(self._task_queue),
            "total_pressure_changes": self._pressure_changes,
            "total_tasks_queued": self._tasks_queued,
            "total_tasks_processed": self._tasks_processed,
        }
