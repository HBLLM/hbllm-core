"""
Cognition Router — distributes cognitive tasks across specialized brain instances.

Enables parallel reasoning by routing different task types to
specialized cognitive clusters based on domain expertise.

Architecture:
- Brain A → reasoning (math, logic, planning)
- Brain B → research (information retrieval, summarization)
- Brain C → creative (writing, brainstorming)
- Brain D → verification (fact-checking, code review)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CognitionWorker:
    """A registered cognitive worker with specialization."""

    worker_id: str
    specializations: list[str]  # domains this worker excels at
    capacity: int = 10  # max concurrent tasks
    current_load: int = 0
    performance: dict[str, float] = field(default_factory=dict)  # domain → score
    is_healthy: bool = True
    last_heartbeat: float = field(default_factory=time.time)


@dataclass
class CognitionTask:
    """A cognitive task to be routed."""

    task_id: str
    domain: str  # reasoning | research | creative | verification | general
    priority: int = 5  # 1 = highest, 10 = lowest
    payload: dict[str, Any] = field(default_factory=dict)
    assigned_worker: str | None = None
    created_at: float = field(default_factory=time.time)


class CognitionRouter:
    """
    Routes cognitive tasks to the best available worker.

    Routing strategies:
    1. Domain-match: route to worker specialized in the domain
    2. Load-balanced: distribute evenly when multiple workers match
    3. Performance-based: prefer workers with higher domain scores
    4. Fallback: route to any available worker if no specialist found

    Integration:
    - Uses SelfModel scores for worker performance
    - Uses CognitiveMetrics for load monitoring
    - Routes via network bus for distributed execution
    """

    def __init__(self) -> None:
        self._workers: dict[str, CognitionWorker] = {}
        self._task_history: list[CognitionTask] = []
        self._domain_map: dict[str, str] = {}  # domain → preferred worker_id

    # ─── Worker Management ───────────────────────────────────────────

    def register_worker(
        self,
        worker_id: str,
        specializations: list[str],
        capacity: int = 10,
    ) -> CognitionWorker:
        """Register a cognitive worker."""
        worker = CognitionWorker(
            worker_id=worker_id,
            specializations=specializations,
            capacity=capacity,
        )
        self._workers[worker_id] = worker
        for domain in specializations:
            self._domain_map[domain] = worker_id
        logger.info("Registered worker %s: %s", worker_id, specializations)
        return worker

    def deregister_worker(self, worker_id: str) -> None:
        worker = self._workers.pop(worker_id, None)
        if worker:
            for domain, wid in list(self._domain_map.items()):
                if wid == worker_id:
                    del self._domain_map[domain]

    def heartbeat(self, worker_id: str) -> None:
        if worker_id in self._workers:
            self._workers[worker_id].last_heartbeat = time.time()
            self._workers[worker_id].is_healthy = True

    # ─── Routing ─────────────────────────────────────────────────────

    def route(self, task: CognitionTask) -> CognitionWorker | None:
        """
        Route a task to the best available worker.

        Priority:
        1. Domain specialist with capacity
        2. Highest-performing available worker
        3. Least-loaded available worker
        4. None if all workers at capacity
        """
        self._task_history.append(task)
        available = [
            w for w in self._workers.values() if w.is_healthy and w.current_load < w.capacity
        ]

        if not available:
            logger.warning("No available workers for task %s", task.task_id)
            return None

        # Strategy 1: Domain specialist
        specialist = self._find_specialist(task.domain, available)
        if specialist:
            self._assign(task, specialist)
            return specialist

        # Strategy 2: Best performer for domain
        best = self._find_best_performer(task.domain, available)
        if best:
            self._assign(task, best)
            return best

        # Strategy 3: Least loaded
        least_loaded = min(available, key=lambda w: w.current_load)
        self._assign(task, least_loaded)
        return least_loaded

    def release(self, worker_id: str, task_id: str, success: bool = True) -> None:
        """Release a worker after task completion."""
        worker = self._workers.get(worker_id)
        if worker:
            worker.current_load = max(0, worker.current_load - 1)

    def _find_specialist(
        self, domain: str, available: list[CognitionWorker]
    ) -> CognitionWorker | None:
        specialists = [w for w in available if domain in w.specializations]
        if not specialists:
            return None
        # Prefer least loaded specialist
        return min(specialists, key=lambda w: w.current_load)

    def _find_best_performer(
        self, domain: str, available: list[CognitionWorker]
    ) -> CognitionWorker | None:
        scored = [(w, w.performance.get(domain, 0.5)) for w in available]
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored and scored[0][1] > 0.5:
            return scored[0][0]
        return None

    def _assign(self, task: CognitionTask, worker: CognitionWorker) -> None:
        task.assigned_worker = worker.worker_id
        worker.current_load += 1
        logger.debug("Assigned task %s → worker %s", task.task_id, worker.worker_id)

    # ─── Parallel Execution ──────────────────────────────────────────

    def split_and_route(
        self,
        subtasks: list[CognitionTask],
    ) -> dict[str, list[CognitionTask]]:
        """
        Route multiple subtasks for parallel execution.

        Returns: {worker_id: [tasks]} mapping
        """
        assignments: dict[str, list[CognitionTask]] = {}
        for task in subtasks:
            worker = self.route(task)
            if worker:
                assignments.setdefault(worker.worker_id, []).append(task)
        return assignments

    # ─── Stats ───────────────────────────────────────────────────────

    def get_cluster_status(self) -> dict[str, Any]:
        workers = []
        for w in self._workers.values():
            workers.append(
                {
                    "id": w.worker_id,
                    "specializations": w.specializations,
                    "load": f"{w.current_load}/{w.capacity}",
                    "healthy": w.is_healthy,
                }
            )
        return {
            "total_workers": len(self._workers),
            "total_tasks_routed": len(self._task_history),
            "workers": workers,
        }
