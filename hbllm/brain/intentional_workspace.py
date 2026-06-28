"""Intentional Workspace — stores active agenda including goals, opportunities, and threats."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from hbllm.brain.autonomy.task_graph import Goal, GoalStatus, TaskPriority

logger = logging.getLogger(__name__)


class IntentionalWorkspace:
    """Manages the active agenda of the cognitive system.

    Maintains current, deferred, and interrupted goals, as well as curiosity
    leads, opportunities, and threats.
    """

    def __init__(self, data_dir: str = "data") -> None:
        self._db_path = Path(data_dir) / "intentional_workspace.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS goals (
                    goal_id TEXT PRIMARY KEY,
                    tenant_id TEXT,
                    name TEXT,
                    description TEXT,
                    status TEXT,
                    priority TEXT,
                    created_at REAL,
                    started_at REAL,
                    completed_at REAL,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS curiosity_goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal_description TEXT UNIQUE,
                    created_at REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS opportunities (
                    id TEXT PRIMARY KEY,
                    description TEXT,
                    metadata TEXT,
                    created_at REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS threats (
                    id TEXT PRIMARY KEY,
                    description TEXT,
                    severity REAL,
                    metadata TEXT,
                    created_at REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pending_reflections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT,
                    details TEXT,
                    created_at REAL
                )
            """)

    # ─── Goal Agenda Management ──────────────────────────────────────

    def add_goal(self, goal: Goal) -> None:
        """Add a new goal to the intentional workspace agenda."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO goals
                (goal_id, tenant_id, name, description, status, priority,
                 created_at, started_at, completed_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    goal.goal_id,
                    goal.tenant_id,
                    goal.name,
                    goal.description,
                    goal.status.value,
                    goal.priority.value,
                    goal.created_at,
                    goal.started_at,
                    goal.completed_at,
                    json.dumps(goal.metadata),
                ),
            )

    def get_goals_by_status(self, status: GoalStatus) -> list[Goal]:
        """Fetch all goals of a given status."""
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute("SELECT * FROM goals WHERE status = ?", (status.value,)).fetchall()

        goals = []
        for r in rows:
            goal = Goal(
                goal_id=r[0],
                tenant_id=r[1],
                name=r[2],
                description=r[3],
                status=GoalStatus(r[4]),
                priority=TaskPriority(r[5]),
                created_at=r[6],
                started_at=r[7],
                completed_at=r[8],
                metadata=json.loads(r[9] or "{}"),
            )
            goals.append(goal)
        return goals

    def update_goal_status(self, goal_id: str, new_status: GoalStatus) -> None:
        """Update the status of a goal (e.g. moving active to deferred/interrupted)."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "UPDATE goals SET status = ? WHERE goal_id = ?",
                (new_status.value, goal_id),
            )

    def get_active_goals(self) -> list[Goal]:
        return self.get_goals_by_status(GoalStatus.ACTIVE)

    def get_deferred_goals(self) -> list[Goal]:
        return self.get_goals_by_status(GoalStatus.PAUSED)

    # ─── Curiosity agenda ──────────────────────────────────────────

    def add_curiosity_goal(self, goal_description: str) -> None:
        """Add a curiosity task for idle reflection/exploration."""
        with sqlite3.connect(str(self._db_path)) as conn:
            try:
                conn.execute(
                    "INSERT INTO curiosity_goals (goal_description, created_at) VALUES (?, ?)",
                    (goal_description, time.time()),
                )
            except sqlite3.IntegrityError:
                pass  # Ignore duplicates

    def get_curiosity_goals(self) -> list[str]:
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT goal_description FROM curiosity_goals ORDER BY created_at ASC"
            ).fetchall()
        return [r[0] for r in rows]

    def remove_curiosity_goal(self, goal_description: str) -> None:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "DELETE FROM curiosity_goals WHERE goal_description = ?", (goal_description,)
            )

    # ─── Opportunities & Threats ─────────────────────────────────────

    def add_opportunity(
        self, opp_id: str, description: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Log a newly detected opportunity."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO opportunities (id, description, metadata, created_at) VALUES (?, ?, ?, ?)",
                (opp_id, description, json.dumps(metadata or {}), time.time()),
            )

    def get_opportunities(self) -> list[dict[str, Any]]:
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT id, description, metadata, created_at FROM opportunities"
            ).fetchall()
        return [
            {
                "id": r[0],
                "description": r[1],
                "metadata": json.loads(r[2] or "{}"),
                "created_at": r[3],
            }
            for r in rows
        ]

    def add_threat(
        self,
        threat_id: str,
        description: str,
        severity: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a newly detected system/security threat."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO threats (id, description, severity, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                (threat_id, description, severity, json.dumps(metadata or {}), time.time()),
            )

    def get_threats(self) -> list[dict[str, Any]]:
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT id, description, severity, metadata, created_at FROM threats ORDER BY severity DESC"
            ).fetchall()
        return [
            {
                "id": r[0],
                "description": r[1],
                "severity": r[2],
                "metadata": json.loads(r[3] or "{}"),
                "created_at": r[4],
            }
            for r in rows
        ]

    # ─── Pending Reflections ─────────────────────────────────────────

    def add_pending_reflection(self, topic: str, details: str = "") -> None:
        """Queue a topic for later reflection during idle cycles."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT INTO pending_reflections (topic, details, created_at) VALUES (?, ?, ?)",
                (topic, details, time.time()),
            )

    def get_pending_reflections(self) -> list[dict[str, Any]]:
        """Retrieve all pending reflection topics, oldest first."""
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT id, topic, details, created_at FROM pending_reflections ORDER BY created_at ASC"
            ).fetchall()
        return [{"id": r[0], "topic": r[1], "details": r[2], "created_at": r[3]} for r in rows]

    def remove_pending_reflection(self, reflection_id: int) -> None:
        """Remove a reflection after it has been processed."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("DELETE FROM pending_reflections WHERE id = ?", (reflection_id,))
