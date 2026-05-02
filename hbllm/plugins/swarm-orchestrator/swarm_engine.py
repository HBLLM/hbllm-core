"""
Swarm Orchestrator Plugin — Multi-agent task decomposition and parallel execution.

Decomposes complex tasks into subtasks, dispatches them to independent brain
instances (workers), and aggregates results into a unified response.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.plugin.sdk import HBLLMPlugin, subscribe

logger = logging.getLogger(__name__)


# ── Data Structures ───────────────────────────────────────────────────────────


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SwarmTask:
    """A single subtask in a swarm execution."""

    task_id: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    error: str = ""
    worker_id: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    priority: float = 0.5

    @property
    def duration_ms(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "status": self.status.value,
            "result": self.result[:500] if self.result else "",
            "error": self.error,
            "worker_id": self.worker_id,
            "duration_ms": round(self.duration_ms, 1),
            "priority": self.priority,
        }


@dataclass
class SwarmExecution:
    """A complete swarm execution session."""

    execution_id: str
    original_task: str
    tasks: list[SwarmTask] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    aggregated_result: str = ""

    @property
    def progress(self) -> float:
        if not self.tasks:
            return 0.0
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        return completed / len(self.tasks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "original_task": self.original_task[:200],
            "status": self.status.value,
            "progress": round(self.progress, 2),
            "total_tasks": len(self.tasks),
            "completed_tasks": sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED),
            "failed_tasks": sum(1 for t in self.tasks if t.status == TaskStatus.FAILED),
            "tasks": [t.to_dict() for t in self.tasks],
        }


# ── Task Decomposer ──────────────────────────────────────────────────────────


class TaskDecomposer:
    """Decomposes complex tasks into parallelizable subtasks."""

    @staticmethod
    def decompose(task: str, max_subtasks: int = 5) -> list[SwarmTask]:
        """
        Decompose a task description into subtasks.

        Uses heuristic splitting. For LLM-powered decomposition, call
        ``decompose_with_llm()`` instead.
        """
        subtasks = []

        # Split by explicit step markers
        lines = task.strip().split("\n")
        step_lines = [
            l.strip()
            for l in lines
            if l.strip()
            and any(
                l.strip().lower().startswith(prefix)
                for prefix in ("1.", "2.", "3.", "4.", "5.", "step ", "- ", "* ")
            )
        ]

        if len(step_lines) >= 2:
            for i, line in enumerate(step_lines[:max_subtasks]):
                # Strip numbered prefix
                clean = line.lstrip("0123456789.-)*] ").strip()
                if clean.lower().startswith("step "):
                    clean = clean[5:].strip(": ")
                subtasks.append(
                    SwarmTask(
                        task_id=f"sub_{i + 1}",
                        description=clean,
                        priority=1.0 - (i * 0.1),
                    )
                )
        else:
            # Single task — break by semicolons or "and then"
            parts = task.replace(" and then ", ";").replace(" then ", ";").split(";")
            for i, part in enumerate(parts[:max_subtasks]):
                part = part.strip()
                if part:
                    subtasks.append(
                        SwarmTask(
                            task_id=f"sub_{i + 1}",
                            description=part,
                            priority=1.0 - (i * 0.1),
                        )
                    )

        # If still a single task, just wrap it
        if not subtasks:
            subtasks.append(
                SwarmTask(
                    task_id="sub_1",
                    description=task[:500],
                    priority=1.0,
                )
            )

        return subtasks

    @staticmethod
    def identify_dependencies(tasks: list[SwarmTask]) -> list[SwarmTask]:
        """
        Identify dependencies between tasks based on references.

        Simple heuristic: if task N references "result" or "output" of a
        previous step, mark it as dependent.
        """
        dependency_words = {"result", "output", "above", "previous", "from step"}

        for i, task in enumerate(tasks):
            desc_lower = task.description.lower()
            for j in range(i):
                ref = tasks[j].task_id
                if ref in desc_lower or any(w in desc_lower for w in dependency_words):
                    task.dependencies.append(tasks[j].task_id)
                    break  # Only first dependency for simplicity

        return tasks


# ── Swarm Engine Plugin ───────────────────────────────────────────────────────


class SwarmEngine(HBLLMPlugin):
    """Coordinates task decomposition and parallel execution across workers."""

    def __init__(
        self,
        node_id: str = "swarm_engine",
        max_workers: int = 4,
        task_timeout: float = 60.0,
    ) -> None:
        super().__init__(
            node_id=node_id,
            capabilities=["task_decomposition", "parallel_execution", "result_aggregation"],
        )
        self.max_workers = max_workers
        self.task_timeout = task_timeout
        self._executions: dict[str, SwarmExecution] = {}
        self._worker_fn: Any = None  # Async callable for executing subtasks
        self.decomposer = TaskDecomposer()

    def set_worker(self, worker_fn: Any) -> None:
        """Set the async worker function for executing subtasks.

        Args:
            worker_fn: Async callable(description: str) -> str
        """
        self._worker_fn = worker_fn

    @subscribe("swarm.request")
    async def on_swarm_request(self, message: Message) -> None:
        """Handle incoming swarm execution requests."""
        task = message.payload.get("task", "")
        if not task:
            return

        execution = await self.execute(task)

        if self.bus:
            await self.bus.publish(
                "swarm.complete",
                Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    topic="swarm.complete",
                    payload=execution.to_dict(),
                    correlation_id=message.correlation_id,
                ),
            )

    async def execute(self, task: str) -> SwarmExecution:
        """Decompose and execute a complex task using the swarm."""
        execution_id = str(uuid.uuid4())[:8]
        subtasks = self.decomposer.decompose(task)
        subtasks = self.decomposer.identify_dependencies(subtasks)

        execution = SwarmExecution(
            execution_id=execution_id,
            original_task=task,
            tasks=subtasks,
            status=TaskStatus.RUNNING,
        )
        self._executions[execution_id] = execution

        # Execute tasks respecting dependencies
        await self._execute_tasks(execution)

        # Aggregate results
        execution.aggregated_result = self._aggregate_results(execution)
        execution.status = (
            TaskStatus.COMPLETED
            if all(t.status == TaskStatus.COMPLETED for t in execution.tasks)
            else TaskStatus.FAILED
        )
        execution.completed_at = time.time()

        return execution

    async def _execute_tasks(self, execution: SwarmExecution) -> None:
        """Execute tasks with dependency resolution and parallelism."""
        completed_ids: set[str] = set()
        semaphore = asyncio.Semaphore(self.max_workers)

        async def run_task(task: SwarmTask) -> None:
            # Wait for dependencies
            while any(dep not in completed_ids for dep in task.dependencies):
                await asyncio.sleep(0.05)

            async with semaphore:
                task.status = TaskStatus.RUNNING
                task.started_at = time.time()
                task.worker_id = f"worker_{id(task) % 1000}"

                try:
                    if self._worker_fn:
                        task.result = await asyncio.wait_for(
                            self._worker_fn(task.description),
                            timeout=self.task_timeout,
                        )
                    else:
                        # No worker — just mark as completed with placeholder
                        task.result = (
                            f"[Task '{task.description}' completed (no worker configured)]"
                        )

                    task.status = TaskStatus.COMPLETED
                except asyncio.TimeoutError:
                    task.status = TaskStatus.FAILED
                    task.error = f"Task timed out after {self.task_timeout}s"
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                finally:
                    task.completed_at = time.time()
                    completed_ids.add(task.task_id)

        # Launch all tasks (they'll self-manage dependencies)
        await asyncio.gather(
            *(run_task(t) for t in execution.tasks),
            return_exceptions=True,
        )

    @staticmethod
    def _aggregate_results(execution: SwarmExecution) -> str:
        """Aggregate subtask results into a unified response."""
        parts = []
        for task in execution.tasks:
            status = "✅" if task.status == TaskStatus.COMPLETED else "❌"
            parts.append(f"{status} {task.description}")
            if task.result:
                parts.append(f"   Result: {task.result[:300]}")
            elif task.error:
                parts.append(f"   Error: {task.error}")
        return "\n".join(parts)

    def get_execution(self, execution_id: str) -> SwarmExecution | None:
        return self._executions.get(execution_id)

    def stats(self) -> dict[str, Any]:
        """Return swarm statistics."""
        total = len(self._executions)
        completed = sum(1 for e in self._executions.values() if e.status == TaskStatus.COMPLETED)
        return {
            "total_executions": total,
            "completed": completed,
            "failed": total - completed,
            "max_workers": self.max_workers,
            "active_executions": sum(
                1 for e in self._executions.values() if e.status == TaskStatus.RUNNING
            ),
        }
