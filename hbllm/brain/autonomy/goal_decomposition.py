"""Goal Decomposition Engine — hierarchical goal breakdown.

Automatically decomposes high-level goals into actionable sub-goals:
    "Plan a birthday party" →
        1. Choose a venue
        2. Create guest list
        3. Plan menu
        4. Send invitations
        5. Arrange decorations

Architecture:
    1. LLM-powered decomposition (using existing provider)
    2. Dependency analysis between sub-goals
    3. Progress tracking with automatic status updates
    4. Re-decomposition if a sub-goal fails

Integrates with existing GoalManager to extend goal capabilities
with automatic planning.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SubGoal:
    """A decomposed sub-goal within a parent goal."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent_goal_id: str = ""
    title: str = ""
    description: str = ""
    order: int = 0  # Execution order
    status: str = "pending"  # pending, active, completed, failed, blocked
    depends_on: list[str] = field(default_factory=list)  # IDs of prerequisite sub-goals
    estimated_duration_min: float = 0.0
    actual_duration_min: float = 0.0
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_goal_id": self.parent_goal_id,
            "title": self.title,
            "description": self.description,
            "order": self.order,
            "status": self.status,
            "depends_on": self.depends_on,
            "estimated_duration_min": self.estimated_duration_min,
            "created_at": self.created_at,
        }


@dataclass
class DecompositionResult:
    """Result of goal decomposition."""

    goal_id: str
    original_goal: str
    sub_goals: list[SubGoal] = field(default_factory=list)
    total_estimated_min: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class GoalDecompositionEngine:
    """Decomposes high-level goals into actionable sub-goals.

    Usage::

        engine = GoalDecompositionEngine(provider=llm_provider)
        result = await engine.decompose("Plan a birthday party")
        for sg in result.sub_goals:
            print(f"  {sg.order}. {sg.title}")
    """

    DECOMPOSITION_PROMPT = """You are a goal planning assistant. Break down the following goal into actionable sub-goals.

Goal: {goal}

Rules:
1. Create 3-8 specific, actionable sub-goals
2. Order them by logical dependency
3. Estimate duration in minutes for each
4. Identify which sub-goals depend on others

Respond in JSON format:
{{
    "sub_goals": [
        {{
            "title": "Short action title",
            "description": "Detailed description of what to do",
            "order": 1,
            "estimated_duration_min": 15,
            "depends_on_orders": []
        }}
    ],
    "critical_path_summary": "Brief summary of the critical path"
}}"""

    def __init__(
        self,
        provider: Any | None = None,
    ) -> None:
        self.provider = provider
        self._decompositions: dict[str, DecompositionResult] = {}
        self._max_cached_decompositions = 100
        self._total_decomposed = 0

    async def decompose(
        self,
        goal_description: str,
        goal_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> DecompositionResult:
        """Decompose a goal into sub-goals.

        If an LLM provider is available, uses it for intelligent decomposition.
        Otherwise, creates a simple 3-step plan.
        """
        gid = goal_id or uuid.uuid4().hex[:12]
        self._total_decomposed += 1

        if self.provider:
            result = await self._decompose_with_llm(gid, goal_description, context)
        else:
            result = self._decompose_heuristic(gid, goal_description)

        self._decompositions[gid] = result

        # Evict oldest entries if cache exceeds limit
        if len(self._decompositions) > self._max_cached_decompositions:
            oldest_keys = list(self._decompositions.keys())[
                : len(self._decompositions) - self._max_cached_decompositions
            ]
            for key in oldest_keys:
                del self._decompositions[key]

        return result

    async def _decompose_with_llm(
        self,
        goal_id: str,
        goal_description: str,
        context: dict[str, Any] | None = None,
    ) -> DecompositionResult:
        """Use LLM to decompose a goal."""
        prompt = self.DECOMPOSITION_PROMPT.format(goal=goal_description)

        try:
            # Call the LLM provider
            response = await self.provider.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
            )

            # Parse JSON response
            content = response.get("content", "") if isinstance(response, dict) else str(response)

            # Try to extract JSON from response
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(content[json_start:json_end])
            else:
                logger.warning("Failed to parse LLM decomposition response, using heuristic")
                return self._decompose_heuristic(goal_id, goal_description)

            # Build sub-goals
            sub_goals: list[SubGoal] = []
            order_to_id: dict[int, str] = {}

            for sg_data in data.get("sub_goals", []):
                sg = SubGoal(
                    parent_goal_id=goal_id,
                    title=sg_data.get("title", ""),
                    description=sg_data.get("description", ""),
                    order=sg_data.get("order", len(sub_goals) + 1),
                    estimated_duration_min=sg_data.get("estimated_duration_min", 0),
                )
                order_to_id[sg.order] = sg.id
                sub_goals.append(sg)

            # Resolve dependencies (order-based → id-based)
            for i, sg_data in enumerate(data.get("sub_goals", [])):
                deps = sg_data.get("depends_on_orders", [])
                if i < len(sub_goals):
                    sub_goals[i].depends_on = [order_to_id[d] for d in deps if d in order_to_id]

            total_est = sum(sg.estimated_duration_min for sg in sub_goals)

            return DecompositionResult(
                goal_id=goal_id,
                original_goal=goal_description,
                sub_goals=sub_goals,
                total_estimated_min=total_est,
                metadata={"source": "llm", "context": context},
            )

        except Exception as e:
            logger.warning("LLM decomposition failed: %s, using heuristic", e)
            return self._decompose_heuristic(goal_id, goal_description)

    def _decompose_heuristic(
        self,
        goal_id: str,
        goal_description: str,
    ) -> DecompositionResult:
        """Simple heuristic decomposition (no LLM required)."""
        sub_goals = [
            SubGoal(
                parent_goal_id=goal_id,
                title="Research and gather requirements",
                description=f"Understand what's needed to achieve: {goal_description}",
                order=1,
                estimated_duration_min=15,
            ),
            SubGoal(
                parent_goal_id=goal_id,
                title="Plan approach and resources",
                description="Identify the best approach and required resources.",
                order=2,
                estimated_duration_min=10,
                depends_on=[],
            ),
            SubGoal(
                parent_goal_id=goal_id,
                title="Execute the plan",
                description="Carry out the planned steps.",
                order=3,
                estimated_duration_min=30,
            ),
        ]
        # Set dependencies
        sub_goals[1].depends_on = [sub_goals[0].id]
        sub_goals[2].depends_on = [sub_goals[1].id]

        return DecompositionResult(
            goal_id=goal_id,
            original_goal=goal_description,
            sub_goals=sub_goals,
            total_estimated_min=55,
            metadata={"source": "heuristic"},
        )

    def get_next_actionable(self, goal_id: str) -> SubGoal | None:
        """Get the next sub-goal that can be worked on.

        A sub-goal is actionable if:
        1. It's in 'pending' status
        2. All its dependencies are 'completed'
        """
        result = self._decompositions.get(goal_id)
        if not result:
            return None

        completed_ids = {sg.id for sg in result.sub_goals if sg.status == "completed"}

        for sg in sorted(result.sub_goals, key=lambda s: s.order):
            if sg.status == "pending":
                deps_met = all(d in completed_ids for d in sg.depends_on)
                if deps_met:
                    return sg

        return None

    def mark_completed(self, goal_id: str, sub_goal_id: str) -> bool:
        """Mark a sub-goal as completed."""
        result = self._decompositions.get(goal_id)
        if not result:
            return False

        for sg in result.sub_goals:
            if sg.id == sub_goal_id:
                sg.status = "completed"
                sg.completed_at = time.time()
                if sg.started_at:
                    sg.actual_duration_min = (sg.completed_at - sg.started_at) / 60
                return True
        return False

    def get_progress(self, goal_id: str) -> dict[str, Any]:
        """Get progress summary for a goal."""
        result = self._decompositions.get(goal_id)
        if not result:
            return {"error": "Goal not found"}

        total = len(result.sub_goals)
        completed = sum(1 for sg in result.sub_goals if sg.status == "completed")
        active = sum(1 for sg in result.sub_goals if sg.status == "active")
        failed = sum(1 for sg in result.sub_goals if sg.status == "failed")

        return {
            "goal_id": goal_id,
            "original_goal": result.original_goal,
            "total_sub_goals": total,
            "completed": completed,
            "active": active,
            "failed": failed,
            "progress_pct": completed / total if total > 0 else 0,
            "sub_goals": [sg.to_dict() for sg in result.sub_goals],
        }

    def stats(self) -> dict[str, Any]:
        """Engine statistics."""
        return {
            "total_decomposed": self._total_decomposed,
            "active_decompositions": len(self._decompositions),
        }
