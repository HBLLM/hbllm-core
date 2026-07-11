"""
Goal Memory — hierarchical goal graph with motive tracking.

Goes beyond "what am I doing?" to "**why** am I doing this?"

Every goal has:
    - A motive (why it was created)
    - A parent goal (goal decomposition hierarchy)
    - Subgoals (what needs to happen to achieve this)
    - Dependencies (what must be done first)
    - Supporting values (which user values this serves)
    - Constraints (boundaries that must not be violated)

The ``GoalMemory`` implements ``IGoalProvider`` from M1's cognitive
interfaces, allowing the ExecutiveController to bias saliency
scoring toward goal-relevant events.

Goal lifecycle::

    ACTIVE → COMPLETED (success)
           → FAILED (couldn't achieve)
           → ABANDONED (no longer relevant)
           → PAUSED (blocked by dependency)

Usage::

    from hbllm.memory.goal_memory import GoalMemory, GoalCube

    goals = GoalMemory()
    goal_id = await goals.add_goal(GoalCube(
        id="goal_001",
        description="Help user debug authentication issue",
        motive="User explicitly asked for help",
        priority=0.9,
    ))

    # Decompose into subgoals
    sub_id = await goals.add_goal(GoalCube(
        id="goal_002",
        description="Reproduce the auth error",
        parent_goal_id="goal_001",
        motive="Need to understand the failure mode",
    ))

    # Track progress
    await goals.update_progress("goal_002", 1.0)
    lineage = await goals.get_goal_lineage("goal_002")
    # → [goal_002, goal_001]  (traces from action to motive)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hbllm.brain.core.cognitive_interfaces import IGoalProvider

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Goal Status
# ═══════════════════════════════════════════════════════════════════════════


class GoalStatus(StrEnum):
    """Lifecycle states for goals."""

    ACTIVE = "active"
    PAUSED = "paused"  # Blocked by dependency
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"  # No longer relevant


# ═══════════════════════════════════════════════════════════════════════════
# GoalCube
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class GoalCube:
    """A goal in the goal graph.

    Attributes:
        id: Unique goal identifier.
        description: What this goal aims to achieve.
        status: Current lifecycle state.
        priority: Priority [0.0, 1.0].
        progress: Completion progress [0.0, 1.0].
        deadline: Optional epoch deadline.
        failure_count: How many times this goal failed.
        tenant_id: Multi-tenant isolation.
        created_at: When the goal was created.

        motive: Why this goal was created.
        parent_goal_id: Parent in the goal hierarchy.
        subgoal_ids: Children in the goal hierarchy.
        dependencies: Goal IDs that must complete first.
        supporting_values: Which user values this serves.
        constraints: Boundaries that must not be violated.
    """

    id: str
    description: str
    status: GoalStatus = GoalStatus.ACTIVE
    priority: float = 0.5
    progress: float = 0.0
    deadline: float | None = None
    failure_count: int = 0
    tenant_id: str = "default"
    created_at: float = field(default_factory=time.time)

    # Goal Graph: why + what
    motive: str = ""
    parent_goal_id: str | None = None
    subgoal_ids: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    supporting_values: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.status == GoalStatus.ACTIVE

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            GoalStatus.COMPLETED,
            GoalStatus.FAILED,
            GoalStatus.ABANDONED,
        )

    @property
    def is_overdue(self) -> bool:
        if self.deadline is None:
            return False
        return time.time() > self.deadline

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "priority": round(self.priority, 3),
            "progress": round(self.progress, 3),
            "deadline": self.deadline,
            "failure_count": self.failure_count,
            "tenant_id": self.tenant_id,
            "motive": self.motive,
            "parent_goal_id": self.parent_goal_id,
            "subgoal_ids": self.subgoal_ids,
            "dependencies": self.dependencies,
            "supporting_values": self.supporting_values,
            "constraints": self.constraints,
        }


# ═══════════════════════════════════════════════════════════════════════════
# GoalMemory — implements IGoalProvider
# ═══════════════════════════════════════════════════════════════════════════


class GoalMemory(IGoalProvider):
    """Hierarchical goal graph with motive tracking.

    Implements ``IGoalProvider`` for the ExecutiveController.
    """

    def __init__(self) -> None:
        self._goals: dict[str, GoalCube] = {}

    # ── IGoalProvider interface ──────────────────────────────────────

    async def get_active_goals(self, tenant_id: str) -> list[Any]:
        """Return all active goals for a tenant.

        Args:
            tenant_id: Tenant to filter by.

        Returns:
            Active GoalCube objects, sorted by priority (desc).
        """
        active = [g for g in self._goals.values() if g.is_active and g.tenant_id == tenant_id]
        active.sort(key=lambda g: g.priority, reverse=True)
        return active

    async def get_urgent_goals(self, horizon: float = 3600.0) -> list[Any]:
        """Return goals with deadlines within the horizon.

        Args:
            horizon: Time window in seconds.

        Returns:
            Urgent GoalCube objects, sorted by deadline.
        """
        now = time.time()
        urgent = [
            g
            for g in self._goals.values()
            if g.is_active and g.deadline is not None and g.deadline <= now + horizon
        ]
        urgent.sort(key=lambda g: g.deadline or float("inf"))
        return urgent

    # ── Goal management ──────────────────────────────────────────────

    async def add_goal(self, goal: GoalCube) -> str:
        """Add a new goal to the graph.

        If the goal has a parent, it's also added to the parent's
        subgoal_ids list.

        Args:
            goal: The goal to add.

        Returns:
            The goal's ID.
        """
        self._goals[goal.id] = goal

        # Link to parent if specified
        if goal.parent_goal_id and goal.parent_goal_id in self._goals:
            parent = self._goals[goal.parent_goal_id]
            if goal.id not in parent.subgoal_ids:
                parent.subgoal_ids.append(goal.id)

        logger.debug(
            "Goal added: %s (priority=%.2f, parent=%s)",
            goal.id,
            goal.priority,
            goal.parent_goal_id,
        )
        return goal.id

    async def get_goal(self, goal_id: str) -> GoalCube | None:
        """Get a goal by ID."""
        return self._goals.get(goal_id)

    async def update_progress(self, goal_id: str, progress: float) -> None:
        """Update goal progress.

        If progress reaches 1.0, the goal is marked COMPLETED.
        Also checks if parent goal should be updated.

        Args:
            goal_id: The goal to update.
            progress: New progress [0.0, 1.0].
        """
        goal = self._goals.get(goal_id)
        if not goal:
            return

        goal.progress = max(0.0, min(1.0, progress))

        if goal.progress >= 1.0 and goal.status == GoalStatus.ACTIVE:
            goal.status = GoalStatus.COMPLETED
            logger.info("Goal completed: %s", goal.id)

            # Update parent progress
            if goal.parent_goal_id:
                await self._update_parent_progress(goal.parent_goal_id)

    async def fail_goal(self, goal_id: str, reason: str = "") -> None:
        """Mark a goal as failed.

        Args:
            goal_id: The goal to fail.
            reason: Why the goal failed.
        """
        goal = self._goals.get(goal_id)
        if not goal:
            return

        goal.status = GoalStatus.FAILED
        goal.failure_count += 1
        logger.info("Goal failed: %s (reason: %s)", goal.id, reason)

    async def abandon_goal(self, goal_id: str, reason: str = "") -> None:
        """Mark a goal as abandoned.

        Also abandons all subgoals recursively.

        Args:
            goal_id: The goal to abandon.
            reason: Why the goal was abandoned.
        """
        goal = self._goals.get(goal_id)
        if not goal:
            return

        goal.status = GoalStatus.ABANDONED

        # Recursively abandon subgoals
        for sub_id in goal.subgoal_ids:
            await self.abandon_goal(sub_id, reason=f"Parent {goal_id} abandoned")

        logger.info("Goal abandoned: %s (reason: %s)", goal.id, reason)

    async def get_blocked_goals(self) -> list[GoalCube]:
        """Get goals whose dependencies are not yet met.

        Returns:
            Goals that are active but blocked by uncompleted dependencies.
        """
        blocked: list[GoalCube] = []
        for goal in self._goals.values():
            if not goal.is_active:
                continue
            for dep_id in goal.dependencies:
                dep = self._goals.get(dep_id)
                if dep and not dep.is_terminal:
                    blocked.append(goal)
                    break
        return blocked

    async def get_stale_goals(self, max_age: float = 3600.0) -> list[GoalCube]:
        """Get active goals that haven't progressed recently.

        Used by REM sleep to identify goals that should be abandoned.

        Args:
            max_age: Maximum age in seconds before a goal is "stale".

        Returns:
            Stale active goals.
        """
        now = time.time()
        stale: list[GoalCube] = []
        for goal in self._goals.values():
            if goal.is_active and (now - goal.created_at) > max_age:
                if goal.progress < 0.1:  # Barely started
                    stale.append(goal)
        return stale

    async def get_goal_lineage(self, goal_id: str) -> list[GoalCube]:
        """Trace the goal hierarchy: action → subgoal → goal → motive.

        Args:
            goal_id: Starting goal.

        Returns:
            List from current goal up to root, inclusive.
        """
        lineage: list[GoalCube] = []
        current_id: str | None = goal_id
        visited: set[str] = set()

        while current_id and current_id not in visited:
            visited.add(current_id)
            goal = self._goals.get(current_id)
            if not goal:
                break
            lineage.append(goal)
            current_id = goal.parent_goal_id

        return lineage

    # ── Internal helpers ─────────────────────────────────────────────

    async def _update_parent_progress(self, parent_id: str) -> None:
        """Recalculate parent progress from subgoals."""
        parent = self._goals.get(parent_id)
        if not parent or not parent.subgoal_ids:
            return

        total = 0.0
        count = 0
        for sub_id in parent.subgoal_ids:
            sub = self._goals.get(sub_id)
            if sub:
                total += sub.progress
                count += 1

        if count > 0:
            parent.progress = total / count
            if parent.progress >= 1.0:
                parent.status = GoalStatus.COMPLETED

    def stats(self) -> dict[str, Any]:
        """Goal memory statistics."""
        active = sum(1 for g in self._goals.values() if g.is_active)
        completed = sum(1 for g in self._goals.values() if g.status == GoalStatus.COMPLETED)
        failed = sum(1 for g in self._goals.values() if g.status == GoalStatus.FAILED)
        return {
            "total_goals": len(self._goals),
            "active": active,
            "completed": completed,
            "failed": failed,
            "abandoned": sum(1 for g in self._goals.values() if g.status == GoalStatus.ABANDONED),
        }
