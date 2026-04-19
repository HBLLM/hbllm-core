"""
Goal Manager — maintains persistent internal goals for autonomous self-improvement.

Goals are prioritized, scheduled, and executed during idle time.
The system continuously works toward improving itself without explicit user requests.

Goal Types:
- Learning: improve weak capability areas
- Exploration: discover new knowledge domains
- Optimization: reduce latency, improve accuracy
- Maintenance: consolidate memory, prune stale data
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum


class StrEnum(str, Enum):
    pass


from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class GoalStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class GoalPriority(StrEnum):
    CRITICAL = "critical"  # must do
    HIGH = "high"  # should do soon
    MEDIUM = "medium"  # do when idle
    LOW = "low"  # nice to have
    BACKGROUND = "background"  # continuous


@dataclass
class Goal:
    """A persistent internal goal."""

    goal_id: str
    name: str
    description: str
    goal_type: str  # learning | exploration | optimization | maintenance
    priority: GoalPriority
    status: GoalStatus = GoalStatus.PENDING
    progress: float = 0.0  # 0.0 to 1.0
    success_criteria: str = ""
    sub_goals: list[str] = field(default_factory=list)
    actions_taken: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    deadline: float | None = None


class GoalManager:
    """
    Manages persistent goals for autonomous self-improvement.

    Architecture:
    1. Goals are created from performance gaps or curiosity signals
    2. PriorityScheduler selects the next goal to work on
    3. Goals execute during idle time or low-traffic periods
    4. Progress is tracked and goals adapt to new information
    5. Completed goals feed back into capability improvements
    """

    def __init__(self, data_dir: str = "data"):
        self._db_path = Path(data_dir) / "goals.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS goals (
                    goal_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    goal_type TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    progress REAL DEFAULT 0.0,
                    success_criteria TEXT DEFAULT '',
                    sub_goals TEXT DEFAULT '[]',
                    actions_taken TEXT DEFAULT '[]',
                    metadata TEXT DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    deadline REAL
                )
            """)

    # ─── Goal CRUD ───────────────────────────────────────────────────

    def create_goal(
        self,
        name: str,
        description: str,
        goal_type: str = "learning",
        priority: GoalPriority = GoalPriority.MEDIUM,
        success_criteria: str = "",
        deadline: float | None = None,
    ) -> Goal:
        """Create a new goal."""
        goal_id = f"goal_{int(time.time())}_{hash(name) % 10000}"
        goal = Goal(
            goal_id=goal_id,
            name=name,
            description=description,
            goal_type=goal_type,
            priority=priority,
            success_criteria=success_criteria,
            deadline=deadline,
        )
        self._save(goal)
        logger.info("Created goal: %s [%s] priority=%s", name, goal_type, priority.value)
        return goal

    def _save(self, goal: Goal) -> None:
        now = time.time()
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO goals
                (goal_id, name, description, goal_type, priority, status,
                 progress, success_criteria, sub_goals, actions_taken,
                 metadata, created_at, updated_at, deadline)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    goal.goal_id,
                    goal.name,
                    goal.description,
                    goal.goal_type,
                    goal.priority.value,
                    goal.status.value,
                    goal.progress,
                    goal.success_criteria,
                    json.dumps(goal.sub_goals),
                    json.dumps(goal.actions_taken),
                    json.dumps(goal.metadata),
                    goal.created_at,
                    now,
                    goal.deadline,
                ),
            )

    def update_progress(self, goal_id: str, progress: float, action: str = "") -> None:
        """Update goal progress and optionally record an action."""
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute(
                "SELECT actions_taken FROM goals WHERE goal_id = ?", (goal_id,)
            ).fetchone()
            if not row:
                return

            actions = json.loads(row[0])
            if action:
                actions.append(f"[{time.strftime('%Y-%m-%d %H:%M')}] {action}")

            status = GoalStatus.COMPLETED.value if progress >= 1.0 else GoalStatus.ACTIVE.value

            conn.execute(
                "UPDATE goals SET progress = ?, status = ?, actions_taken = ?, updated_at = ? "
                "WHERE goal_id = ?",
                (min(1.0, progress), status, json.dumps(actions), time.time(), goal_id),
            )

    def complete_goal(self, goal_id: str) -> None:
        self.update_progress(goal_id, 1.0, "Goal completed")

    def fail_goal(self, goal_id: str, reason: str = "") -> None:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "UPDATE goals SET status = ?, updated_at = ? WHERE goal_id = ?",
                (GoalStatus.FAILED.value, time.time(), goal_id),
            )

    # ─── Scheduling ──────────────────────────────────────────────────

    def next_goal(self) -> Goal | None:
        """Get the highest-priority pending/active goal."""
        priority_order = {
            GoalPriority.CRITICAL.value: 0,
            GoalPriority.HIGH.value: 1,
            GoalPriority.MEDIUM.value: 2,
            GoalPriority.LOW.value: 3,
            GoalPriority.BACKGROUND.value: 4,
        }
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT * FROM goals WHERE status IN ('pending', 'active') ORDER BY updated_at ASC",
            ).fetchall()

        if not rows:
            return None

        goals = [self._row_to_goal(r) for r in rows]
        goals.sort(key=lambda g: (priority_order.get(g.priority.value, 5), -g.created_at))
        return goals[0]

    async def execute_goal(self, goal: Goal) -> None:
        """
        Execute a goal asynchronously.
        This bridges the goal persistence with the actual execution engine.
        """
        if goal.status.value in (GoalStatus.COMPLETED.value, GoalStatus.FAILED.value):
            return

        logger.info("Executing goal: %s (type: %s)", goal.name, goal.goal_type)

        try:
            # execution stub that advances progress
            # In a full system, this dispatches to PlannerNode/ExecutionNode
            progress = goal.progress + 0.25
            action = f"Executed auto-step: progressed to {progress * 100:.0f}%"

            self.update_progress(goal.goal_id, min(1.0, progress), action)
            logger.info("Goal progress updated: %s -> %.2f", goal.name, progress)
        except Exception as e:
            logger.error("Failed to execute goal %s: %s", goal.name, e)
            self.fail_goal(goal.goal_id, str(e))

    def get_active_goals(self) -> list[Goal]:
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT * FROM goals WHERE status IN ('pending', 'active') ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_goal(r) for r in rows]

    # ─── Auto-Goal Generation ────────────────────────────────────────

    def generate_from_performance(self, metrics: dict[str, float]) -> list[Goal]:
        """Auto-generate goals from performance gaps."""
        goals = []

        if metrics.get("hallucination_rate", 0) > 0.1:
            goals.append(
                self.create_goal(
                    name="Reduce hallucination rate",
                    description="Hallucination rate is above 10%. Improve factual accuracy.",
                    goal_type="optimization",
                    priority=GoalPriority.HIGH,
                    success_criteria="hallucination_rate < 0.05",
                )
            )

        if metrics.get("avg_latency_ms", 0) > 5000:
            goals.append(
                self.create_goal(
                    name="Optimize response latency",
                    description="Average latency exceeds 5s. Optimize inference pipeline.",
                    goal_type="optimization",
                    priority=GoalPriority.MEDIUM,
                    success_criteria="avg_latency_ms < 2000",
                )
            )

        if metrics.get("tool_success_rate", 1.0) < 0.8:
            goals.append(
                self.create_goal(
                    name="Improve tool usage accuracy",
                    description="Tool success rate below 80%. Learn better tool selection.",
                    goal_type="learning",
                    priority=GoalPriority.MEDIUM,
                    success_criteria="tool_success_rate > 0.9",
                )
            )

        if metrics.get("memory_utilization", 0) < 0.3:
            goals.append(
                self.create_goal(
                    name="Improve memory utilization",
                    description="Memory is underutilized. Store more contextual knowledge.",
                    goal_type="exploration",
                    priority=GoalPriority.LOW,
                    success_criteria="memory_utilization > 0.6",
                )
            )

        return goals

    # ─── Helpers ──────────────────────────────────────────────────────

    def _row_to_goal(self, row: tuple[Any, ...]) -> Goal:
        return Goal(
            goal_id=row[0],
            name=row[1],
            description=row[2],
            goal_type=row[3],
            priority=GoalPriority(row[4]),
            status=GoalStatus(row[5]),
            progress=row[6],
            success_criteria=row[7],
            sub_goals=json.loads(row[8]),
            actions_taken=json.loads(row[9]),
            metadata=json.loads(row[10]),
            created_at=row[11],
            deadline=row[13],
        )

    def stats(self) -> dict[str, Any]:
        with sqlite3.connect(str(self._db_path)) as conn:
            total = conn.execute("SELECT COUNT(*) FROM goals").fetchone()[0]
            active = conn.execute(
                "SELECT COUNT(*) FROM goals WHERE status IN ('pending','active')"
            ).fetchone()[0]
            completed = conn.execute(
                "SELECT COUNT(*) FROM goals WHERE status = 'completed'"
            ).fetchone()[0]
        return {"total_goals": total, "active": active, "completed": completed}
