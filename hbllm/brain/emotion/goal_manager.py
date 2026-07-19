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
from enum import StrEnum
from pathlib import Path
from typing import Any

from hbllm.security import TenantSQLiteRepository

logger = logging.getLogger(__name__)


class GoalStatus(StrEnum):
    PENDING = "pending"
    BLOCKED = "blocked"
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    FAILED = "abandoned"


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

    dependencies: list[str] = field(default_factory=list)
    block_reason: str = ""
    execution_journal: dict[str, Any] = field(default_factory=dict)


class GoalManager(TenantSQLiteRepository):
    """
    Manages persistent goals for autonomous self-improvement.

    Architecture:
    1. Goals are created from performance gaps or curiosity signals
    2. PriorityScheduler selects the next goal to work on
    3. Goals execute during idle time or low-traffic periods
    4. Progress is tracked and goals adapt to new information
    5. Completed goals feed back into capability improvements
    """

    def __init__(self, data_dir: str = "data", bus: Any = None):
        db_path = Path(data_dir) / "goals.db"
        super().__init__(db_path)
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._bus = bus  # MessageBus for dispatching to execution nodes
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            # Check if goals table exists
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='goals'")
            table_exists = cur.fetchone() is not None

            if not table_exists:
                conn.execute("BEGIN TRANSACTION")
                try:
                    conn.execute("""
                        CREATE TABLE goals (
                            goal_id TEXT PRIMARY KEY,
                            tenant_id TEXT DEFAULT '__legacy__',
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
                            deadline REAL,
                            dependencies TEXT DEFAULT '[]',
                            block_reason TEXT DEFAULT '',
                            execution_journal TEXT DEFAULT '{}'
                        )
                    """)
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_goals_tenant ON goals(tenant_id, status)"
                    )
                    conn.execute("PRAGMA user_version = 2")
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
            else:
                # Table exists. Check if tenant_id column exists.
                cur = conn.execute("PRAGMA table_info(goals)")
                columns = [row[1] for row in cur.fetchall()]
                if "tenant_id" not in columns:
                    conn.execute("BEGIN TRANSACTION")
                    try:
                        conn.execute(
                            "ALTER TABLE goals ADD COLUMN tenant_id TEXT DEFAULT '__legacy__'"
                        )
                        conn.execute(
                            "CREATE INDEX IF NOT EXISTS idx_goals_tenant ON goals(tenant_id, status)"
                        )
                        conn.execute("PRAGMA user_version = 2")
                        conn.commit()
                    except Exception:
                        conn.rollback()
                        raise
                else:
                    # Column exists, just ensure user_version is set to 2
                    conn.execute("PRAGMA user_version = 2")

    # ─── Goal CRUD ───────────────────────────────────────────────────

    def create_goal(
        self,
        name: str,
        description: str,
        goal_type: str = "learning",
        priority: GoalPriority = GoalPriority.MEDIUM,
        success_criteria: str = "",
        deadline: float | None = None,
        dependencies: list[str] | None = None,
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
            dependencies=dependencies or [],
        )
        self._save(goal)
        logger.info("Created goal: %s [%s] priority=%s", name, goal_type, priority.value)
        self._resolve_dag_states()
        return goal

    def _save(self, goal: Goal) -> None:
        now = time.time()
        tenant_id = self.current_tenant() or "__legacy__"
        with sqlite3.connect(self._db_path) as conn:
            self.execute_tenant(
                conn,
                """
                INSERT OR REPLACE INTO goals
                (goal_id, tenant_id, name, description, goal_type, priority, status,
                 progress, success_criteria, sub_goals, actions_taken,
                 metadata, created_at, updated_at, deadline, dependencies, block_reason, execution_journal)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    goal.goal_id,
                    tenant_id,
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
                    json.dumps(goal.dependencies),
                    goal.block_reason,
                    json.dumps(goal.execution_journal),
                ),
                required_capability="goal_maintenance",
            )

    def update_progress(self, goal_id: str, progress: float, action: str = "") -> None:
        """Update goal progress and optionally record an action."""
        tenant_id = self.current_tenant()
        with sqlite3.connect(self._db_path) as conn:
            if tenant_id:
                row = self.execute_tenant(
                    conn,
                    "SELECT actions_taken FROM goals WHERE goal_id = ? AND tenant_id = ?",
                    (goal_id, tenant_id),
                ).fetchone()
            else:
                row = self.execute_tenant(
                    conn,
                    "SELECT actions_taken FROM goals WHERE goal_id = ?",
                    (goal_id,),
                    required_capability="goal_maintenance",
                ).fetchone()
            if not row:
                return

            actions = json.loads(row[0] or "[]")
            if action:
                actions.append(f"[{time.strftime('%Y-%m-%d %H:%M')}] {action}")

            status = GoalStatus.COMPLETED.value if progress >= 1.0 else GoalStatus.ACTIVE.value

            if tenant_id:
                self.execute_tenant(
                    conn,
                    "UPDATE goals SET progress = ?, status = ?, actions_taken = ?, updated_at = ? "
                    "WHERE goal_id = ? AND tenant_id = ?",
                    (
                        min(1.0, progress),
                        status,
                        json.dumps(actions),
                        time.time(),
                        goal_id,
                        tenant_id,
                    ),
                )
            else:
                self.execute_tenant(
                    conn,
                    "UPDATE goals SET progress = ?, status = ?, actions_taken = ?, updated_at = ? "
                    "WHERE goal_id = ?",
                    (min(1.0, progress), status, json.dumps(actions), time.time(), goal_id),
                    required_capability="goal_maintenance",
                )
        self._resolve_dag_states()

    def update_goal_status(self, goal_id: str, status: GoalStatus, block_reason: str = "") -> None:
        """Update goal status and block reason."""
        tenant_id = self.current_tenant()
        with sqlite3.connect(self._db_path) as conn:
            if tenant_id:
                self.execute_tenant(
                    conn,
                    "UPDATE goals SET status = ?, block_reason = ?, updated_at = ? WHERE goal_id = ? AND tenant_id = ?",
                    (status.value, block_reason, time.time(), goal_id, tenant_id),
                )
            else:
                self.execute_tenant(
                    conn,
                    "UPDATE goals SET status = ?, block_reason = ?, updated_at = ? WHERE goal_id = ?",
                    (status.value, block_reason, time.time(), goal_id),
                    required_capability="goal_maintenance",
                )

    def update_execution_journal(
        self,
        goal_id: str,
        checkpoint: str,
        completed_steps: list[str],
        blocked_reason: str = "",
        next_action: str = "",
    ) -> None:
        """Update the structured execution journal for a goal."""
        journal = {
            "goal_id": goal_id,
            "checkpoint": checkpoint,
            "completed_steps": completed_steps,
            "blocked_reason": blocked_reason,
            "next_action": next_action,
        }
        tenant_id = self.current_tenant()
        with sqlite3.connect(self._db_path) as conn:
            if tenant_id:
                self.execute_tenant(
                    conn,
                    "UPDATE goals SET execution_journal = ?, updated_at = ? WHERE goal_id = ? AND tenant_id = ?",
                    (json.dumps(journal), time.time(), goal_id, tenant_id),
                )
            else:
                self.execute_tenant(
                    conn,
                    "UPDATE goals SET execution_journal = ?, updated_at = ? WHERE goal_id = ?",
                    (json.dumps(journal), time.time(), goal_id),
                    required_capability="goal_maintenance",
                )

    def complete_goal(self, goal_id: str) -> None:
        self.update_progress(goal_id, 1.0, "Goal completed")

    def fail_goal(self, goal_id: str, reason: str = "") -> None:
        tenant_id = self.current_tenant()
        with sqlite3.connect(self._db_path) as conn:
            if tenant_id:
                self.execute_tenant(
                    conn,
                    "UPDATE goals SET status = ?, block_reason = ?, updated_at = ? WHERE goal_id = ? AND tenant_id = ?",
                    (GoalStatus.ABANDONED.value, reason, time.time(), goal_id, tenant_id),
                )
            else:
                self.execute_tenant(
                    conn,
                    "UPDATE goals SET status = ?, block_reason = ?, updated_at = ? WHERE goal_id = ?",
                    (GoalStatus.ABANDONED.value, reason, time.time(), goal_id),
                    required_capability="goal_maintenance",
                )
        self._resolve_dag_states()

    def _resolve_dag_states(self) -> None:
        """Resolve blocked/pending states based on goal dependency DAG."""
        tenant_id = self.current_tenant()
        with sqlite3.connect(self._db_path) as conn:
            if tenant_id:
                rows = self.execute_tenant(
                    conn,
                    "SELECT goal_id, status, dependencies FROM goals WHERE tenant_id = ?",
                    (tenant_id,),
                ).fetchall()
            else:
                rows = self.execute_tenant(
                    conn,
                    "SELECT goal_id, status, dependencies FROM goals",
                    required_capability="goal_maintenance",
                ).fetchall()

        status_map = {r[0]: r[1] for r in rows}
        dep_map = {r[0]: json.loads(r[2] or "[]") for r in rows}

        for goal_id, current_status in status_map.items():
            if current_status in ("pending", "blocked"):
                deps = dep_map.get(goal_id, [])
                uncompleted_deps = [d for d in deps if status_map.get(d) != "completed"]

                if uncompleted_deps:
                    if current_status != "blocked":
                        self.update_goal_status(
                            goal_id,
                            GoalStatus.BLOCKED,
                            block_reason=f"Blocked by dependencies: {', '.join(uncompleted_deps)}",
                        )
                else:
                    if current_status == "blocked":
                        self.update_goal_status(goal_id, GoalStatus.PENDING, block_reason="")

    # ─── Scheduling ──────────────────────────────────────────────────

    def next_goal(self) -> Goal | None:
        """Get the highest-priority pending/active goal."""
        self._resolve_dag_states()
        priority_order = {
            GoalPriority.CRITICAL.value: 0,
            GoalPriority.HIGH.value: 1,
            GoalPriority.MEDIUM.value: 2,
            GoalPriority.LOW.value: 3,
            GoalPriority.BACKGROUND.value: 4,
        }
        tenant_id = self.current_tenant()
        with sqlite3.connect(self._db_path) as conn:
            if tenant_id:
                rows = self.execute_tenant(
                    conn,
                    "SELECT * FROM goals WHERE status IN ('pending', 'active') AND tenant_id = ? ORDER BY updated_at ASC",
                    (tenant_id,),
                ).fetchall()
            else:
                rows = self.execute_tenant(
                    conn,
                    "SELECT * FROM goals WHERE status IN ('pending', 'active') ORDER BY updated_at ASC",
                    required_capability="goal_maintenance",
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
        if goal.status.value in (GoalStatus.COMPLETED.value, GoalStatus.ABANDONED.value):
            return

        # Double check if we are blocked before starting
        self._resolve_dag_states()
        # Reload status from db
        reloaded = self._get_goal_by_id(goal.goal_id)
        if not reloaded or reloaded.status.value != GoalStatus.PENDING.value:
            return

        self.update_goal_status(goal.goal_id, GoalStatus.ACTIVE)

        logger.info("Executing goal: %s (type: %s)", goal.name, goal.goal_type)

        try:
            # Dispatch to appropriate execution engine based on goal type
            if goal.goal_type == "learning":
                await self._execute_learning_goal(goal)
            elif goal.goal_type == "exploration":
                await self._execute_exploration_goal(goal)
            elif goal.goal_type == "optimization":
                await self._execute_optimization_goal(goal)
            elif goal.goal_type == "maintenance":
                await self._execute_maintenance_goal(goal)
            else:
                # Generic execution via bus if available
                await self._execute_generic_goal(goal)
        except Exception as e:
            logger.error("Failed to execute goal %s: %s", goal.name, e)
            self.fail_goal(goal.goal_id, str(e))

    async def _execute_learning_goal(self, goal: Goal) -> None:
        """Execute a learning goal by dispatching to AutonomousLearner."""
        if not self._bus:
            logger.warning("No bus available for goal execution, using stub")
            self._stub_execution(goal)
            return

        # Extract learning topic from metadata
        learning_topic = goal.metadata.get("learning_topic", goal.name)

        # Dispatch to AutonomousLearner via bus
        from hbllm.network.messages import Message, MessageType

        msg = Message(
            type=MessageType.QUERY,
            source_node_id="goal_manager",
            topic="learning.execute",
            tenant_id="system",
            payload={
                "goal_id": goal.goal_id,
                "topic": learning_topic,
                "motivation": goal.metadata.get("motivation", "system"),
            },
        )

        try:
            response = await self._bus.request(msg, timeout=60.0)
            if response and response.payload.get("success"):
                progress = response.payload.get("progress", 0.5)
                action = response.payload.get("action", "Learning step completed")
                self.update_progress(goal.goal_id, min(1.0, progress), action)
            else:
                raise RuntimeError(
                    f"Learning execution failed: {response.payload if response else 'No response'}"
                )
        except Exception as e:
            logger.error("Learning goal execution failed: %s", e)
            raise

    async def _execute_exploration_goal(self, goal: Goal) -> None:
        """Execute an exploration goal by dispatching to CuriosityNode."""
        if not self._bus:
            logger.warning("No bus available for goal execution, using stub")
            self._stub_execution(goal)
            return

        from hbllm.network.messages import Message, MessageType

        msg = Message(
            type=MessageType.QUERY,
            source_node_id="goal_manager",
            topic="curiosity.explore",
            tenant_id="system",
            payload={
                "goal_id": goal.goal_id,
                "topic": goal.name,
                "description": goal.description,
            },
        )

        try:
            response = await self._bus.request(msg, timeout=60.0)
            if response and response.payload.get("success"):
                progress = response.payload.get("progress", 0.5)
                action = response.payload.get("action", "Exploration step completed")
                self.update_progress(goal.goal_id, min(1.0, progress), action)
            else:
                raise RuntimeError(
                    f"Exploration execution failed: {response.payload if response else 'No response'}"
                )
        except Exception as e:
            logger.error("Exploration goal execution failed: %s", e)
            raise

    async def _execute_optimization_goal(self, goal: Goal) -> None:
        """Execute an optimization goal by dispatching to appropriate optimizer."""
        if not self._bus:
            logger.warning("No bus available for goal execution, using stub")
            self._stub_execution(goal)
            return

        from hbllm.network.messages import Message, MessageType

        msg = Message(
            type=MessageType.QUERY,
            source_node_id="goal_manager",
            topic="optimization.execute",
            tenant_id="system",
            payload={
                "goal_id": goal.goal_id,
                "target": goal.name,
                "criteria": goal.success_criteria,
            },
        )

        try:
            response = await self._bus.request(msg, timeout=60.0)
            if response and response.payload.get("success"):
                progress = response.payload.get("progress", 0.5)
                action = response.payload.get("action", "Optimization step completed")
                self.update_progress(goal.goal_id, min(1.0, progress), action)
            else:
                raise RuntimeError(
                    f"Optimization execution failed: {response.payload if response else 'No response'}"
                )
        except Exception as e:
            logger.error("Optimization goal execution failed: %s", e)
            raise

    async def _execute_maintenance_goal(self, goal: Goal) -> None:
        """Execute a maintenance goal (memory consolidation, pruning, etc.)."""
        if not self._bus:
            logger.warning("No bus available for goal execution, using stub")
            self._stub_execution(goal)
            return

        from hbllm.network.messages import Message, MessageType

        msg = Message(
            type=MessageType.QUERY,
            source_node_id="goal_manager",
            topic="maintenance.execute",
            tenant_id="system",
            payload={
                "goal_id": goal.goal_id,
                "task": goal.name,
            },
        )

        try:
            response = await self._bus.request(msg, timeout=60.0)
            if response and response.payload.get("success"):
                progress = response.payload.get("progress", 0.5)
                action = response.payload.get("action", "Maintenance step completed")
                self.update_progress(goal.goal_id, min(1.0, progress), action)
            else:
                raise RuntimeError(
                    f"Maintenance execution failed: {response.payload if response else 'No response'}"
                )
        except Exception as e:
            logger.error("Maintenance goal execution failed: %s", e)
            raise

    async def _execute_generic_goal(self, goal: Goal) -> None:
        """Execute a generic goal via PlannerNode/ExecutionNode."""
        if not self._bus:
            logger.warning("No bus available for goal execution, using stub")
            self._stub_execution(goal)
            return

        from hbllm.network.messages import Message, MessageType

        msg = Message(
            type=MessageType.QUERY,
            source_node_id="goal_manager",
            topic="planner.execute",
            tenant_id="system",
            payload={
                "goal_id": goal.goal_id,
                "goal": goal.name,
                "description": goal.description,
                "criteria": goal.success_criteria,
            },
        )

        try:
            response = await self._bus.request(msg, timeout=120.0)
            if response and response.payload.get("success"):
                progress = response.payload.get("progress", 0.5)
                action = response.payload.get("action", "Goal step completed")
                self.update_progress(goal.goal_id, min(1.0, progress), action)
            else:
                raise RuntimeError(
                    f"Generic goal execution failed: {response.payload if response else 'No response'}"
                )
        except Exception as e:
            logger.error("Generic goal execution failed: %s", e)
            raise

    def _stub_execution(self, goal: Goal) -> None:
        """Fallback stub execution when bus is not available."""
        progress = goal.progress + 0.25
        action = f"Executed auto-step: progressed to {progress * 100:.0f}%"
        self.update_progress(goal.goal_id, min(1.0, progress), action)
        logger.info("Goal progress updated (stub): %s -> %.2f", goal.name, progress)

    def get_active_goals(self) -> list[Goal]:
        self._resolve_dag_states()
        tenant_id = self.current_tenant()
        with sqlite3.connect(self._db_path) as conn:
            if tenant_id:
                rows = self.execute_tenant(
                    conn,
                    "SELECT * FROM goals WHERE status IN ('pending', 'active', 'blocked') AND tenant_id = ? ORDER BY created_at DESC",
                    (tenant_id,),
                ).fetchall()
            else:
                rows = self.execute_tenant(
                    conn,
                    "SELECT * FROM goals WHERE status IN ('pending', 'active', 'blocked') ORDER BY created_at DESC",
                    required_capability="goal_maintenance",
                ).fetchall()
        return [self._row_to_goal(r) for r in rows]

    # ─── Auto-Goal Generation ────────────────────────────────────────

    def generate_from_performance(self, metrics: dict[str, float]) -> list[Goal]:
        """Auto-generate goals from performance gaps."""
        goals = []

        if metrics.get("hallucination_rate", 0) > 0.1:
            goals.append(
                self.create_goal(
                    name="Calibrate Hallucination Detectors",
                    description="Run semantic consistency sweeps to tune detector sensitivity thresholds.",
                    goal_type="optimization",
                    priority=GoalPriority.HIGH,
                    success_criteria="Hallucination rate drops below 5%",
                )
            )

        if metrics.get("context_leakage_events", 0) > 0:
            goals.append(
                self.create_goal(
                    name="Audit Tenant Boundaries",
                    description="Run automated cross-tenant retrieval queries to verify isolation integrity.",
                    goal_type="security",
                    priority=GoalPriority.CRITICAL,
                    success_criteria="Zero boundary leakage events for 48 consecutive hours",
                )
            )

        if metrics.get("avg_latency_ms", 0) > 2500:
            goals.append(
                self.create_goal(
                    name="Optimize Memory Retrieval Cache",
                    description="Prune old episodic context windows and run sqlite VACUUM index maintenance.",
                    goal_type="maintenance",
                    priority=GoalPriority.MEDIUM,
                    success_criteria="Average latency drops below 1500ms",
                )
            )

        if metrics.get("concept_drift_score", 0) > 0.4:
            goals.append(
                self.create_goal(
                    name="Reflect and Formulate Abstract Concepts",
                    description="Consolidate recent episodic memories into high-level concept abstractions.",
                    goal_type="learning",
                    priority=GoalPriority.HIGH,
                    success_criteria="Average concept representation error drops below threshold",
                )
            )

        if goals:
            logger.info(
                "Generated %d learning goals from weak areas",
                len(goals),
            )
        return goals

    # ─── Learning Goal Integration ────────────────────────────────────

    def create_learning_goal(
        self,
        topic: str,
        motivation: str = "system",
        parent_goal_id: str | None = None,
        priority: GoalPriority = GoalPriority.MEDIUM,
    ) -> Goal:
        """Create a goal that dispatches to AutonomousLearner."""
        goal = self.create_goal(
            name=f"Learn: {topic}",
            description=f"Autonomously learn about '{topic}' ({motivation})",
            goal_type="learning",
            priority=priority,
            success_criteria=f"confidence >= 0.8 for '{topic}'",
            dependencies=[parent_goal_id] if parent_goal_id else None,
        )

        goal.metadata["learning_topic"] = topic
        goal.metadata["motivation"] = motivation
        if parent_goal_id:
            goal.metadata["parent_goal_id"] = parent_goal_id
        self._save(goal)

        return goal

    def generate_from_contradictions(
        self,
        contradictions: list[dict[str, Any]],
        parent_goal_id: str | None = None,
    ) -> list[Goal]:
        """Auto-generate learning goals from unresolved contradictions."""
        goals = []
        for ctr in contradictions:
            concept = ctr.get("concept", "unknown")
            claim_a = ctr.get("claim_a", "?")[:60]
            claim_b = ctr.get("claim_b", "?")[:60]
            severity = ctr.get("severity", 0.5)

            if severity >= 0.7:
                priority = GoalPriority.HIGH
            elif severity >= 0.4:
                priority = GoalPriority.MEDIUM
            else:
                priority = GoalPriority.LOW

            goal = self.create_learning_goal(
                topic=f"Resolve contradiction in '{concept}': '{claim_a}' vs '{claim_b}'",
                motivation="contradiction_resolution",
                parent_goal_id=parent_goal_id,
                priority=priority,
            )
            goal.metadata["contradiction_id"] = ctr.get("contradiction_id", "")
            goal.metadata["severity"] = severity
            self._save(goal)
            goals.append(goal)

        if goals:
            logger.info(
                "Generated %d learning goals from contradictions",
                len(goals),
            )
        return goals

    def generate_from_weak_areas(
        self,
        weak_areas: list[dict[str, Any]],
        parent_goal_id: str | None = None,
    ) -> list[Goal]:
        """Auto-generate learning goals from MetaLearner weak areas."""
        goals = []
        for area in weak_areas:
            concept = area.get("concept", "unknown")
            score = area.get("score", 0.5)

            if score < 0.3:
                priority = GoalPriority.HIGH
            elif score < 0.5:
                priority = GoalPriority.MEDIUM
            else:
                priority = GoalPriority.LOW

            goal = self.create_learning_goal(
                topic=f"Strengthen understanding of '{concept}'",
                motivation="weak_area_improvement",
                parent_goal_id=parent_goal_id,
                priority=priority,
            )
            goal.metadata["weak_area_score"] = score
            self._save(goal)
            goals.append(goal)

        if goals:
            logger.info(
                "Generated %d learning goals from weak areas",
                len(goals),
            )
        return goals

    def get_learning_goals(self) -> list[Goal]:
        """Get all active learning-type goals."""
        tenant_id = self.current_tenant()
        with sqlite3.connect(self._db_path) as conn:
            if tenant_id:
                rows = self.execute_tenant(
                    conn,
                    "SELECT * FROM goals WHERE goal_type = 'learning' "
                    "AND status IN ('pending', 'active', 'blocked') AND tenant_id = ? "
                    "ORDER BY created_at DESC",
                    (tenant_id,),
                ).fetchall()
            else:
                rows = self.execute_tenant(
                    conn,
                    "SELECT * FROM goals WHERE goal_type = 'learning' "
                    "AND status IN ('pending', 'active', 'blocked') "
                    "ORDER BY created_at DESC",
                    required_capability="goal_maintenance",
                ).fetchall()
        return [self._row_to_goal(r) for r in rows]

    def subordinate_to(self, child_goal_id: str, parent_goal_id: str) -> None:
        """Link a child goal to a parent goal."""
        tenant_id = self.current_tenant()
        with sqlite3.connect(self._db_path) as conn:
            if tenant_id:
                row = self.execute_tenant(
                    conn,
                    "SELECT dependencies FROM goals WHERE goal_id = ? AND tenant_id = ?",
                    (child_goal_id, tenant_id),
                ).fetchone()
            else:
                row = self.execute_tenant(
                    conn,
                    "SELECT dependencies FROM goals WHERE goal_id = ?",
                    (child_goal_id,),
                    required_capability="goal_maintenance",
                ).fetchone()
            if row:
                deps = json.loads(row[0] or "[]")
                if parent_goal_id not in deps:
                    deps.append(parent_goal_id)
                    if tenant_id:
                        self.execute_tenant(
                            conn,
                            "UPDATE goals SET dependencies = ? WHERE goal_id = ? AND tenant_id = ?",
                            (json.dumps(deps), child_goal_id, tenant_id),
                        )
                    else:
                        self.execute_tenant(
                            conn,
                            "UPDATE goals SET dependencies = ? WHERE goal_id = ?",
                            (json.dumps(deps), child_goal_id),
                            required_capability="goal_maintenance",
                        )

            if tenant_id:
                parent_row = self.execute_tenant(
                    conn,
                    "SELECT sub_goals FROM goals WHERE goal_id = ? AND tenant_id = ?",
                    (parent_goal_id, tenant_id),
                ).fetchone()
            else:
                parent_row = self.execute_tenant(
                    conn,
                    "SELECT sub_goals FROM goals WHERE goal_id = ?",
                    (parent_goal_id,),
                    required_capability="goal_maintenance",
                ).fetchone()
            if parent_row:
                subs = json.loads(parent_row[0] or "[]")
                if child_goal_id not in subs:
                    subs.append(child_goal_id)
                    if tenant_id:
                        self.execute_tenant(
                            conn,
                            "UPDATE goals SET sub_goals = ? WHERE goal_id = ? AND tenant_id = ?",
                            (json.dumps(subs), parent_goal_id, tenant_id),
                        )
                    else:
                        self.execute_tenant(
                            conn,
                            "UPDATE goals SET sub_goals = ? WHERE goal_id = ?",
                            (json.dumps(subs), parent_goal_id),
                            required_capability="goal_maintenance",
                        )

    # ─── Helpers ──────────────────────────────────────────────────────

    def _row_to_goal(self, row: tuple[Any, ...]) -> Goal:
        if len(row) > 17:  # v2 schema has 18 columns
            return Goal(
                goal_id=row[0],
                name=row[2],
                description=row[3],
                goal_type=row[4],
                priority=GoalPriority(row[5]),
                status=GoalStatus(row[6]),
                progress=row[7],
                success_criteria=row[8],
                sub_goals=json.loads(row[9] or "[]"),
                actions_taken=json.loads(row[10] or "[]"),
                metadata=json.loads(row[11] or "{}"),
                created_at=row[12],
                deadline=row[14],
                dependencies=json.loads(row[15] or "[]"),
                block_reason=row[16],
                execution_journal=json.loads(row[17] or "{}"),
            )
        else:
            # v1 legacy fallback
            return Goal(
                goal_id=row[0],
                name=row[1],
                description=row[2],
                goal_type=row[3],
                priority=GoalPriority(row[4]),
                status=GoalStatus(row[5]),
                progress=row[6],
                success_criteria=row[7],
                sub_goals=json.loads(row[8] or "[]"),
                actions_taken=json.loads(row[9] or "[]"),
                metadata=json.loads(row[10] or "{}"),
                created_at=row[11],
                deadline=row[13],
                dependencies=json.loads(row[14] or "[]") if len(row) > 14 else [],
                block_reason=row[15] if len(row) > 15 else "",
                execution_journal=json.loads(row[16] or "{}") if len(row) > 16 else {},
            )

    def stats(self) -> dict[str, Any]:
        tenant_id = self.current_tenant()
        with sqlite3.connect(self._db_path) as conn:
            if tenant_id:
                total = self.execute_tenant(
                    conn, "SELECT COUNT(*) FROM goals WHERE tenant_id = ?", (tenant_id,)
                ).fetchone()[0]
                active = self.execute_tenant(
                    conn,
                    "SELECT COUNT(*) FROM goals WHERE status IN ('pending','active','blocked') AND tenant_id = ?",
                    (tenant_id,),
                ).fetchone()[0]
                completed = self.execute_tenant(
                    conn,
                    "SELECT COUNT(*) FROM goals WHERE status = 'completed' AND tenant_id = ?",
                    (tenant_id,),
                ).fetchone()[0]
                learning = self.execute_tenant(
                    conn,
                    "SELECT COUNT(*) FROM goals WHERE goal_type = 'learning' "
                    "AND status IN ('pending','active','blocked') AND tenant_id = ?",
                    (tenant_id,),
                ).fetchone()[0]
            else:
                total = self.execute_tenant(
                    conn, "SELECT COUNT(*) FROM goals", required_capability="goal_maintenance"
                ).fetchone()[0]
                active = self.execute_tenant(
                    conn,
                    "SELECT COUNT(*) FROM goals WHERE status IN ('pending','active','blocked')",
                    required_capability="goal_maintenance",
                ).fetchone()[0]
                completed = self.execute_tenant(
                    conn,
                    "SELECT COUNT(*) FROM goals WHERE status = 'completed'",
                    required_capability="goal_maintenance",
                ).fetchone()[0]
                learning = self.execute_tenant(
                    conn,
                    "SELECT COUNT(*) FROM goals WHERE goal_type = 'learning' "
                    "AND status IN ('pending','active','blocked')",
                    required_capability="goal_maintenance",
                ).fetchone()[0]
        return {
            "total_goals": total,
            "active": active,
            "completed": completed,
            "active_learning_goals": learning,
        }
