"""Task Graph Runtime — persistent, resumable goal execution.

Manages long-running cognitive goals as directed acyclic graphs (DAGs)
backed by SQLite. Each goal decomposes into steps that can be paused,
resumed, retried, or delegated — surviving reboots and device switches.

Concepts
────────
- **Goal**: A top-level objective (e.g. "prepare weekly report")
- **TaskNode**: A single step within a goal's DAG
- **Edge**: A dependency between two TaskNodes
- **Execution**: The runtime walks the DAG, executing ready nodes

Persistence
───────────
All state is stored in ``task_graphs.db`` (separate from scheduler.db).
On boot, incomplete goals are automatically resumed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Enumerations ─────────────────────────────────────────────────────────────


class GoalStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"  # All dependencies met
    RUNNING = "running"
    VERIFYING = "verifying"  # Awaiting reality verification
    CORRECTING = "correcting"  # Execution failed verification, adjusting plan
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"  # Dependency failed
    UNCERTAIN = "uncertain"  # Verification deadlock / ambiguity


class TaskPriority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


# ── Data Models ──────────────────────────────────────────────────────────────


@dataclass
class VerificationRule:
    """Probabilistic rule to verify task success against WorldStateEngine."""

    entity_id: str
    property_name: str
    expected_value: Any
    min_match_score: float = 0.8  # 0.0 to 1.0 (fuzzy match / confidence)
    max_wait_time_s: float = 60.0  # Timeout before escalating to UNCERTAIN
    time_window_s: float = 5.0  # Acceptable reality lag window (s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "property_name": self.property_name,
            "expected_value": self.expected_value,
            "min_match_score": self.min_match_score,
            "max_wait_time_s": self.max_wait_time_s,
            "time_window_s": self.time_window_s,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> VerificationRule | None:
        if not d:
            return None
        return cls(
            entity_id=d["entity_id"],
            property_name=d["property_name"],
            expected_value=d["expected_value"],
            min_match_score=d.get("min_match_score", 0.8),
            max_wait_time_s=d.get("max_wait_time_s", 60.0),
            time_window_s=d.get("time_window_s", 5.0),
        )


@dataclass
class TaskNode:
    """A single executable step within a goal's DAG."""

    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    goal_id: str = ""
    name: str = ""
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    action_topic: str = ""  # Bus topic to publish when executing
    action_payload: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    correction_attempts: int = 0
    max_correction_attempts: int = 2
    timeout_s: float = 300.0  # 5 min default
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    dependencies: list[str] = field(default_factory=list)  # task_ids
    verification_rule: VerificationRule | None = None
    verification_started_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "action_topic": self.action_topic,
            "action_payload": self.action_payload,
            "result": self.result,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "correction_attempts": self.correction_attempts,
            "max_correction_attempts": self.max_correction_attempts,
            "timeout_s": self.timeout_s,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "dependencies": self.dependencies,
            "verification_rule": self.verification_rule.to_dict()
            if self.verification_rule
            else None,
            "verification_started_at": self.verification_started_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskNode:
        return cls(
            task_id=d["task_id"],
            goal_id=d["goal_id"],
            name=d.get("name", ""),
            description=d.get("description", ""),
            status=TaskStatus(d.get("status", "pending")),
            priority=TaskPriority(d.get("priority", "normal")),
            action_topic=d.get("action_topic", ""),
            action_payload=d.get("action_payload", {}),
            result=d.get("result", {}),
            retry_count=d.get("retry_count", 0),
            max_retries=d.get("max_retries", 3),
            correction_attempts=d.get("correction_attempts", 0),
            max_correction_attempts=d.get("max_correction_attempts", 2),
            timeout_s=d.get("timeout_s", 300.0),
            created_at=d.get("created_at", 0.0),
            started_at=d.get("started_at", 0.0),
            completed_at=d.get("completed_at", 0.0),
            dependencies=d.get("dependencies", []),
            verification_rule=VerificationRule.from_dict(d.get("verification_rule")),
            verification_started_at=d.get("verification_started_at", 0.0),
        )


@dataclass
class Goal:
    """A top-level objective composed of TaskNodes in a DAG."""

    goal_id: str = field(default_factory=lambda: f"goal_{uuid.uuid4().hex[:12]}")
    tenant_id: str = "default"
    name: str = ""
    description: str = ""
    status: GoalStatus = GoalStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "tenant_id": self.tenant_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }


# ── Task Graph Runtime ───────────────────────────────────────────────────────


class TaskGraphRuntime:
    """Persistent DAG-based goal execution engine.

    Manages goals and their task DAGs with SQLite persistence.
    Integrates with AutonomyCore as a proactive handler.

    Usage::

        runtime = TaskGraphRuntime(data_dir="/path/to/data")
        goal = Goal(name="Weekly Report")
        task_a = TaskNode(name="Gather data", action_topic="data.gather")
        task_b = TaskNode(name="Analyze", dependencies=[task_a.task_id])
        runtime.create_goal(goal, [task_a, task_b])
        ready = runtime.get_ready_tasks()
    """

    def __init__(self, data_dir: str | Path, max_concurrent: int = 5) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "task_graphs.db"
        self._max_concurrent = max_concurrent
        self._running_tasks: set[str] = set()
        self._init_db()

    # ── Schema ────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS goals (
                    goal_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    name TEXT,
                    description TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    priority TEXT NOT NULL DEFAULT 'normal',
                    created_at REAL,
                    started_at REAL DEFAULT 0,
                    completed_at REAL DEFAULT 0,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_nodes (
                    task_id TEXT PRIMARY KEY,
                    goal_id TEXT NOT NULL,
                    name TEXT,
                    description TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    priority TEXT NOT NULL DEFAULT 'normal',
                    action_topic TEXT,
                    action_payload TEXT DEFAULT '{}',
                    result TEXT DEFAULT '{}',
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    correction_attempts INTEGER DEFAULT 0,
                    max_correction_attempts INTEGER DEFAULT 2,
                    timeout_s REAL DEFAULT 300.0,
                    created_at REAL,
                    started_at REAL DEFAULT 0,
                    completed_at REAL DEFAULT 0,
                    dependencies TEXT DEFAULT '[]',
                    verification_rule TEXT,
                    verification_started_at REAL DEFAULT 0,
                    FOREIGN KEY (goal_id) REFERENCES goals(goal_id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_task_goal ON task_nodes(goal_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_task_status ON task_nodes(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_goal_status ON goals(status)")

    # ── Goal Management ───────────────────────────────────────────────

    def create_goal(self, goal: Goal, tasks: list[TaskNode]) -> str:
        """Create a new goal with its task DAG. Returns the goal_id."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO goals
                   (goal_id, tenant_id, name, description, status, priority,
                    created_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    goal.goal_id,
                    goal.tenant_id,
                    goal.name,
                    goal.description,
                    goal.status.value,
                    goal.priority.value,
                    goal.created_at,
                    json.dumps(goal.metadata),
                ),
            )
            for task in tasks:
                task.goal_id = goal.goal_id
                self._insert_task(conn, task)

        # Mark root tasks (no dependencies) as READY
        self._update_ready_tasks(goal.goal_id)

        logger.info(
            "Created goal '%s' (%s) with %d tasks",
            goal.name,
            goal.goal_id,
            len(tasks),
        )
        return goal.goal_id

    def activate_goal(self, goal_id: str) -> bool:
        """Transition a goal from PENDING to ACTIVE."""
        return self._update_goal_status(goal_id, GoalStatus.PENDING, GoalStatus.ACTIVE)

    def pause_goal(self, goal_id: str) -> bool:
        """Pause an active goal. Running tasks will finish but no new ones start."""
        return self._update_goal_status(goal_id, GoalStatus.ACTIVE, GoalStatus.PAUSED)

    def resume_goal(self, goal_id: str) -> bool:
        """Resume a paused goal."""
        if self._update_goal_status(goal_id, GoalStatus.PAUSED, GoalStatus.ACTIVE):
            self._update_ready_tasks(goal_id)
            return True
        return False

    def cancel_goal(self, goal_id: str) -> bool:
        """Cancel a goal and all its pending/ready tasks."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE goals SET status = ? WHERE goal_id = ?",
                (GoalStatus.CANCELLED.value, goal_id),
            )
            conn.execute(
                """UPDATE task_nodes SET status = ?
                   WHERE goal_id = ? AND status IN (?, ?, ?)""",
                (
                    TaskStatus.SKIPPED.value,
                    goal_id,
                    TaskStatus.PENDING.value,
                    TaskStatus.READY.value,
                    TaskStatus.RUNNING.value,
                ),
            )
        return True

    def get_goal(self, goal_id: str) -> Goal | None:
        """Retrieve a goal by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM goals WHERE goal_id = ?", (goal_id,)).fetchone()
            if not row:
                return None
            return Goal(
                goal_id=row["goal_id"],
                tenant_id=row["tenant_id"],
                name=row["name"] or "",
                description=row["description"] or "",
                status=GoalStatus(row["status"]),
                priority=TaskPriority(row["priority"]),
                created_at=row["created_at"] or 0.0,
                started_at=row["started_at"] or 0.0,
                completed_at=row["completed_at"] or 0.0,
                metadata=json.loads(row["metadata"] or "{}"),
            )

    def get_active_goals(self, tenant_id: str | None = None) -> list[Goal]:
        """Retrieve all active goals, optionally filtered by tenant."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if tenant_id:
                rows = conn.execute(
                    "SELECT * FROM goals WHERE status = ? AND tenant_id = ?",
                    (GoalStatus.ACTIVE.value, tenant_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM goals WHERE status = ?",
                    (GoalStatus.ACTIVE.value,),
                ).fetchall()
            return [
                Goal(
                    goal_id=r["goal_id"],
                    tenant_id=r["tenant_id"],
                    name=r["name"] or "",
                    description=r["description"] or "",
                    status=GoalStatus(r["status"]),
                    priority=TaskPriority(r["priority"]),
                    created_at=r["created_at"] or 0.0,
                )
                for r in rows
            ]

    # ── Task Management ───────────────────────────────────────────────

    def get_ready_tasks(self, limit: int = 10) -> list[TaskNode]:
        """Get tasks that are READY to execute (all deps completed)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT t.* FROM task_nodes t
                   JOIN goals g ON t.goal_id = g.goal_id
                   WHERE t.status = ? AND g.status = ?
                   ORDER BY
                     CASE t.priority
                       WHEN 'critical' THEN 0
                       WHEN 'high' THEN 1
                       WHEN 'normal' THEN 2
                       WHEN 'low' THEN 3
                     END,
                     t.created_at ASC
                   LIMIT ?""",
                (TaskStatus.READY.value, GoalStatus.ACTIVE.value, limit),
            ).fetchall()
            return [self._row_to_task(r) for r in rows]

    def get_tasks_for_goal(self, goal_id: str) -> list[TaskNode]:
        """Get all tasks for a given goal."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM task_nodes WHERE goal_id = ? ORDER BY created_at",
                (goal_id,),
            ).fetchall()
            return [self._row_to_task(r) for r in rows]

    def mark_task_running(self, task_id: str) -> bool:
        """Mark a task as RUNNING."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE task_nodes SET status = ?, started_at = ? WHERE task_id = ? AND status = ?",
                (TaskStatus.RUNNING.value, time.time(), task_id, TaskStatus.READY.value),
            )
            if cursor.rowcount > 0:
                self._running_tasks.add(task_id)
                return True
            return False

    def complete_task(self, task_id: str, result: dict[str, Any] | None = None) -> bool:
        """Mark a task as COMPLETED (or VERIFYING) and cascade readiness."""
        now = time.time()
        goal_id: str | None = None
        has_rule = False
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT verification_rule, goal_id FROM task_nodes WHERE task_id = ?", (task_id,)
            ).fetchone()
            if not row:
                return False

            has_rule = bool(json.loads(row["verification_rule"] or "null"))
            goal_id = row["goal_id"]
            target_status = TaskStatus.VERIFYING.value if has_rule else TaskStatus.COMPLETED.value

            cursor = conn.execute(
                """UPDATE task_nodes SET status = ?, completed_at = ?, result = ?, verification_started_at = ?
                   WHERE task_id = ? AND status = ?""",
                (
                    target_status,
                    now if not has_rule else 0.0,
                    json.dumps(result or {}),
                    now if has_rule else 0.0,
                    task_id,
                    TaskStatus.RUNNING.value,
                ),
            )
            if cursor.rowcount == 0:
                return False
            self._running_tasks.discard(task_id)

        # Cascade OUTSIDE the transaction so the COMPLETED status is visible
        if not has_rule and goal_id:
            self._update_ready_tasks(goal_id)
            self._check_goal_completion(goal_id)
        return True

    def verify_pending_tasks(self, world_state: Any) -> None:
        """Evaluate tasks in VERIFYING state against the WorldStateEngine."""
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM task_nodes WHERE status = ?", (TaskStatus.VERIFYING.value,)
            ).fetchall()

            for row in rows:
                task = self._row_to_task(row)
                if not task.verification_rule:
                    continue

                rule = task.verification_rule
                entity = world_state.get_entity_state(rule.entity_id)

                # 1. Event-driven check
                if (
                    entity
                    and entity.last_updated >= task.verification_started_at - rule.time_window_s
                ):
                    val = entity.properties.get(rule.property_name)
                    # Simple heuristic match score for PoC:
                    match_score = 0.0
                    if val == rule.expected_value:
                        match_score = entity.confidence
                    elif (
                        isinstance(val, str)
                        and isinstance(rule.expected_value, str)
                        and rule.expected_value.lower() in val.lower()
                    ):
                        match_score = entity.confidence * 0.8

                    if match_score >= rule.min_match_score:
                        # Success
                        conn.execute(
                            "UPDATE task_nodes SET status = ?, completed_at = ? WHERE task_id = ?",
                            (TaskStatus.COMPLETED.value, now, task.task_id),
                        )
                        self._update_ready_tasks(task.goal_id)
                        continue

                # 2. Timeout fallback
                age = now - task.verification_started_at
                if age > rule.max_wait_time_s:
                    task.correction_attempts += 1
                    if task.correction_attempts <= task.max_correction_attempts:
                        conn.execute(
                            "UPDATE task_nodes SET status = ?, correction_attempts = ? WHERE task_id = ?",
                            (TaskStatus.CORRECTING.value, task.correction_attempts, task.task_id),
                        )
                        logger.info(
                            "Task %s verification timed out. Transitioning to CORRECTING (%d/%d)",
                            task.task_id,
                            task.correction_attempts,
                            task.max_correction_attempts,
                        )
                    else:
                        conn.execute(
                            "UPDATE task_nodes SET status = ? WHERE task_id = ?",
                            (TaskStatus.UNCERTAIN.value, task.task_id),
                        )
                        logger.warning(
                            "Task %s verification permanently failed. Marking UNCERTAIN.",
                            task.task_id,
                        )

    def fail_task(self, task_id: str, error: str = "") -> str:
        """Handle a task failure. Returns 'retrying', 'failed', or 'blocked'.

        If retries remain, the task is re-queued as READY.
        If not, the task fails and downstream dependents are BLOCKED.
        """
        self._running_tasks.discard(task_id)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM task_nodes WHERE task_id = ?", (task_id,)).fetchone()
            if not row:
                return "failed"

            task = self._row_to_task(row)
            task.retry_count += 1

            if task.retry_count <= task.max_retries:
                conn.execute(
                    "UPDATE task_nodes SET status = ?, retry_count = ? WHERE task_id = ?",
                    (TaskStatus.READY.value, task.retry_count, task_id),
                )
                logger.info(
                    "Task %s retrying (%d/%d)",
                    task_id,
                    task.retry_count,
                    task.max_retries,
                )
                return "retrying"

            # Permanent failure
            conn.execute(
                """UPDATE task_nodes SET status = ?, result = ?
                   WHERE task_id = ?""",
                (
                    TaskStatus.FAILED.value,
                    json.dumps({"error": error}),
                    task_id,
                ),
            )
            # Block downstream dependents
            self._block_dependents(conn, task.goal_id, task_id)

        logger.warning("Task %s permanently failed: %s", task_id, error)
        goal_id = self._get_task_goal_id(task_id)
        if goal_id:
            self._check_goal_completion(goal_id)
        return "failed"

    def add_task_to_goal(self, goal_id: str, task: TaskNode) -> str:
        """Dynamically add a new task to an existing goal."""
        task.goal_id = goal_id
        with sqlite3.connect(self.db_path) as conn:
            self._insert_task(conn, task)
        self._update_ready_tasks(goal_id)
        return task.task_id

    # ── Recovery ──────────────────────────────────────────────────────

    def recover_on_boot(self) -> int:
        """Recover interrupted tasks after a restart.

        Tasks left in RUNNING state are reset to READY for re-execution.
        Returns the count of recovered tasks.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE task_nodes SET status = ?, started_at = 0 WHERE status = ?",
                (TaskStatus.READY.value, TaskStatus.RUNNING.value),
            )
            count = cursor.rowcount
        if count > 0:
            logger.info("Recovered %d interrupted tasks on boot", count)
        return count

    # ── Introspection ─────────────────────────────────────────────────

    def get_goal_progress(self, goal_id: str) -> dict[str, Any]:
        """Get completion progress for a goal."""
        tasks = self.get_tasks_for_goal(goal_id)
        total = len(tasks)
        if total == 0:
            return {"total": 0, "completed": 0, "progress": 0.0}

        by_status: dict[str, int] = {}
        for t in tasks:
            by_status[t.status.value] = by_status.get(t.status.value, 0) + 1

        completed = by_status.get("completed", 0)
        return {
            "total": total,
            "completed": completed,
            "failed": by_status.get("failed", 0),
            "running": by_status.get("running", 0),
            "ready": by_status.get("ready", 0),
            "pending": by_status.get("pending", 0),
            "blocked": by_status.get("blocked", 0),
            "progress": round(completed / total, 3),
        }

    def snapshot(self) -> dict[str, Any]:
        """Telemetry snapshot."""
        with sqlite3.connect(self.db_path) as conn:
            goal_counts = {}
            for row in conn.execute(
                "SELECT status, COUNT(*) FROM goals GROUP BY status"
            ).fetchall():
                goal_counts[row[0]] = row[1]

            task_counts = {}
            for row in conn.execute(
                "SELECT status, COUNT(*) FROM task_nodes GROUP BY status"
            ).fetchall():
                task_counts[row[0]] = row[1]

        return {
            "goals": goal_counts,
            "tasks": task_counts,
            "running_tasks": len(self._running_tasks),
            "max_concurrent": self._max_concurrent,
        }

    # ── Internal Helpers ──────────────────────────────────────────────

    def _insert_task(self, conn: sqlite3.Connection, task: TaskNode) -> None:
        conn.execute(
            """INSERT INTO task_nodes
               (task_id, goal_id, name, description, status, priority,
                action_topic, action_payload, result, retry_count,
                max_retries, correction_attempts, max_correction_attempts,
                timeout_s, created_at, dependencies,
                verification_rule, verification_started_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task.task_id,
                task.goal_id,
                task.name,
                task.description,
                task.status.value,
                task.priority.value,
                task.action_topic,
                json.dumps(task.action_payload),
                json.dumps(task.result),
                task.retry_count,
                task.max_retries,
                task.correction_attempts,
                task.max_correction_attempts,
                task.timeout_s,
                task.created_at,
                json.dumps(task.dependencies),
                json.dumps(task.verification_rule.to_dict()) if task.verification_rule else None,
                task.verification_started_at,
            ),
        )

    def _row_to_task(self, row: sqlite3.Row) -> TaskNode:
        return TaskNode(
            task_id=row["task_id"],
            goal_id=row["goal_id"],
            name=row["name"] or "",
            description=row["description"] or "",
            status=TaskStatus(row["status"]),
            priority=TaskPriority(row["priority"]),
            action_topic=row["action_topic"] or "",
            action_payload=json.loads(row["action_payload"] or "{}"),
            result=json.loads(row["result"] or "{}"),
            retry_count=row["retry_count"] if row["retry_count"] is not None else 0,
            max_retries=row["max_retries"] if row["max_retries"] is not None else 3,
            correction_attempts=row["correction_attempts"]
            if "correction_attempts" in row.keys() and row["correction_attempts"] is not None
            else 0,
            max_correction_attempts=row["max_correction_attempts"]
            if "max_correction_attempts" in row.keys()
            and row["max_correction_attempts"] is not None
            else 2,
            timeout_s=row["timeout_s"] if row["timeout_s"] is not None else 300.0,
            created_at=row["created_at"] if row["created_at"] is not None else 0.0,
            started_at=row["started_at"] if row["started_at"] is not None else 0.0,
            completed_at=row["completed_at"] if row["completed_at"] is not None else 0.0,
            dependencies=json.loads(row["dependencies"] or "[]"),
            verification_rule=VerificationRule.from_dict(
                json.loads(row["verification_rule"] or "null")
            )
            if "verification_rule" in row.keys()
            else None,
            verification_started_at=row["verification_started_at"]
            if "verification_started_at" in row.keys()
            and row["verification_started_at"] is not None
            else 0.0,
        )

    def _update_goal_status(
        self, goal_id: str, from_status: GoalStatus, to_status: GoalStatus
    ) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            now = time.time()
            extra = ""
            params: list[Any] = [to_status.value]
            if to_status == GoalStatus.ACTIVE:
                extra = ", started_at = ?"
                params.append(now)
            elif to_status == GoalStatus.COMPLETED:
                extra = ", completed_at = ?"
                params.append(now)
            params.extend([goal_id, from_status.value])
            cursor = conn.execute(
                f"UPDATE goals SET status = ?{extra} WHERE goal_id = ? AND status = ?",
                params,
            )
            return cursor.rowcount > 0

    def _update_ready_tasks(self, goal_id: str) -> None:
        """Mark PENDING tasks as READY if all their dependencies are completed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            pending = conn.execute(
                "SELECT * FROM task_nodes WHERE goal_id = ? AND status = ?",
                (goal_id, TaskStatus.PENDING.value),
            ).fetchall()

            for row in pending:
                deps = json.loads(row["dependencies"] or "[]")
                if not deps:
                    conn.execute(
                        "UPDATE task_nodes SET status = ? WHERE task_id = ?",
                        (TaskStatus.READY.value, row["task_id"]),
                    )
                    continue

                # Check if all dependencies are completed
                placeholders = ",".join("?" for _ in deps)
                completed = conn.execute(
                    f"""SELECT COUNT(*) FROM task_nodes
                        WHERE task_id IN ({placeholders}) AND status = ?""",
                    [*deps, TaskStatus.COMPLETED.value],
                ).fetchone()[0]

                if completed == len(deps):
                    conn.execute(
                        "UPDATE task_nodes SET status = ? WHERE task_id = ?",
                        (TaskStatus.READY.value, row["task_id"]),
                    )

    def _check_goal_completion(self, goal_id: str) -> None:
        """Check if all tasks in a goal are done; if so, complete the goal."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM task_nodes WHERE goal_id = ?",
                (goal_id,),
            ).fetchone()[0]
            # BLOCKED is also terminal — a blocked task will never run
            terminal = conn.execute(
                """SELECT COUNT(*) FROM task_nodes WHERE goal_id = ?
                   AND status IN (?, ?, ?, ?)""",
                (
                    goal_id,
                    TaskStatus.COMPLETED.value,
                    TaskStatus.FAILED.value,
                    TaskStatus.SKIPPED.value,
                    TaskStatus.BLOCKED.value,
                ),
            ).fetchone()[0]

            if total > 0 and terminal == total:
                failed = conn.execute(
                    """SELECT COUNT(*) FROM task_nodes WHERE goal_id = ?
                       AND status IN (?, ?)""",
                    (goal_id, TaskStatus.FAILED.value, TaskStatus.BLOCKED.value),
                ).fetchone()[0]

                new_status = GoalStatus.FAILED if failed > 0 else GoalStatus.COMPLETED
                conn.execute(
                    "UPDATE goals SET status = ?, completed_at = ? WHERE goal_id = ? AND status != ?",
                    (new_status.value, time.time(), goal_id, new_status.value),
                )
                logger.info(
                    "Goal %s → %s (%d/%d tasks completed)",
                    goal_id,
                    new_status.value,
                    total - failed,
                    total,
                )

    def _block_dependents(
        self, conn: sqlite3.Connection, goal_id: str, failed_task_id: str
    ) -> None:
        """Block all tasks that depend on the failed task."""
        conn.row_factory = sqlite3.Row
        all_tasks = conn.execute(
            "SELECT * FROM task_nodes WHERE goal_id = ? AND status IN (?, ?)",
            (goal_id, TaskStatus.PENDING.value, TaskStatus.READY.value),
        ).fetchall()

        for row in all_tasks:
            deps = json.loads(row["dependencies"] or "[]")
            if failed_task_id in deps:
                conn.execute(
                    "UPDATE task_nodes SET status = ? WHERE task_id = ?",
                    (TaskStatus.BLOCKED.value, row["task_id"]),
                )

    def _get_task_goal_id(self, task_id: str) -> str | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT goal_id FROM task_nodes WHERE task_id = ?", (task_id,)
            ).fetchone()
            return row[0] if row else None
