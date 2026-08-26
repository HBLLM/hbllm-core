"""Intentional Workspace — HCIR-native active agenda including goals, opportunities, and threats."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from hbllm.brain.autonomy.task_graph import Goal, GoalStatus, TaskPriority
from hbllm.hcir.graph import (
    ConstraintNode,
    GoalLifecycle,
    GoalNode,
    HypothesisNode,
    NodeLifecycle,
    UnknownNode,
)
from hbllm.hcir.types import Provenance, Scope
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.workspace_tiers import TieredWorkspace

logger = logging.getLogger(__name__)


class IntentionalWorkspace:
    """Manages the active agenda of the cognitive system via HCIR graph nodes.

    Maintains current, deferred, and interrupted goals, as well as curiosity
    leads, opportunities, and threats directly within the HCIR WorkspaceState.
    """

    def __init__(
        self,
        data_dir: str = "data",
        workspace: TieredWorkspace | HCIRWorkspaceState | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        if workspace is None:
            self._tiered = TieredWorkspace()
            self._ws = self._tiered.brain
        elif isinstance(workspace, TieredWorkspace):
            self._tiered = workspace
            self._ws = workspace.brain
        else:
            self._tiered = None
            self._ws = workspace

        self._goals_meta: dict[str, dict[str, Any]] = {}
        self._curiosity_goals: list[tuple[str, float]] = []
        self._opportunities: dict[str, dict[str, Any]] = {}
        self._threats: dict[str, dict[str, Any]] = {}
        self._pending_reflections: list[dict[str, Any]] = []
        self._reflection_counter = 0

    @property
    def hcir_workspace(self) -> HCIRWorkspaceState:
        """Get the underlying active HCIR workspace state."""
        return self._ws

    # ─── Goal Agenda Management ──────────────────────────────────────

    def add_goal(self, goal: Goal) -> None:
        """Add a new goal to the intentional workspace agenda as an HCIR GoalNode."""
        hcir_lifecycle = NodeLifecycle.ACTIVE
        goal_lifecycle = GoalLifecycle.EXECUTING
        if goal.status in (GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.CANCELLED):
            hcir_lifecycle = NodeLifecycle.ARCHIVED
            goal_lifecycle = GoalLifecycle.COMPLETED
        elif goal.status == GoalStatus.PAUSED:
            hcir_lifecycle = NodeLifecycle.ARCHIVED
            goal_lifecycle = GoalLifecycle.BLOCKED
        elif goal.status == GoalStatus.PENDING:
            hcir_lifecycle = NodeLifecycle.CREATED
            goal_lifecycle = GoalLifecycle.CREATED

        self._goals_meta[goal.goal_id] = {
            "name": goal.name,
            "description": goal.description,
            "status": goal.status,
            "priority": goal.priority,
            "tenant_id": goal.tenant_id,
            "created_at": goal.created_at,
            "started_at": goal.started_at,
            "completed_at": goal.completed_at,
            "metadata": goal.metadata,
        }

        goal_node = GoalNode(
            id=goal.goal_id,
            description=goal.description or goal.name,
            priority=1.0 if goal.priority == TaskPriority.HIGH else 0.5,
            lifecycle=hcir_lifecycle,
            goal_lifecycle=goal_lifecycle,
            provenance=Provenance(
                created_by=goal.tenant_id or "intentional_workspace",
                created_at=goal.created_at,
            ),
            scope=Scope(tenant_id=goal.tenant_id or "default"),
            tags=["goal", f"status:{goal.status.value}", f"priority:{goal.priority.value}"],
        )
        self._ws.upsert_node(goal_node)

    def get_goals_by_status(self, status: GoalStatus) -> list[Goal]:
        """Fetch all goals of a given status from the HCIR workspace."""
        goals: list[Goal] = []
        for goal_id, meta in list(self._goals_meta.items()):
            if meta["status"] == status:
                node = self._ws.get_node(goal_id)
                node_desc = node.description if node and node.description else ""
                goal = Goal(
                    goal_id=goal_id,
                    tenant_id=str(meta.get("tenant_id") or "default"),
                    name=str(meta.get("name") or node_desc),
                    description=str(meta.get("description") or node_desc),
                    status=status,
                    priority=meta.get("priority", TaskPriority.NORMAL),
                    created_at=float(meta.get("created_at") or time.time()),
                    started_at=float(meta.get("started_at") or 0.0),
                    completed_at=float(meta.get("completed_at") or 0.0),
                    metadata=meta.get("metadata", {}),
                )
                goals.append(goal)
        return goals

    def update_goal_status(self, goal_id: str, new_status: GoalStatus) -> None:
        """Update the status of a goal in the HCIR workspace."""
        if goal_id in self._goals_meta:
            self._goals_meta[goal_id]["status"] = new_status
            if new_status == GoalStatus.COMPLETED:
                self._goals_meta[goal_id]["completed_at"] = time.time()
            elif new_status == GoalStatus.ACTIVE and not self._goals_meta[goal_id].get(
                "started_at"
            ):
                self._goals_meta[goal_id]["started_at"] = time.time()

        node = self._ws.get_node(goal_id)
        if node and isinstance(node, GoalNode):
            node.tags = [t for t in node.tags if not t.startswith("status:")]
            node.tags.append(f"status:{new_status.value}")
            if new_status in (GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.CANCELLED):
                node.lifecycle = NodeLifecycle.ARCHIVED
                node.goal_lifecycle = GoalLifecycle.COMPLETED
                node.resolved = True
            elif new_status == GoalStatus.PAUSED:
                node.lifecycle = NodeLifecycle.ARCHIVED
                node.goal_lifecycle = GoalLifecycle.BLOCKED
            elif new_status == GoalStatus.ACTIVE:
                node.lifecycle = NodeLifecycle.ACTIVE
                node.goal_lifecycle = GoalLifecycle.EXECUTING
            self._ws.upsert_node(node)

    def get_active_goals(self) -> list[Goal]:
        return self.get_goals_by_status(GoalStatus.ACTIVE)

    def get_deferred_goals(self) -> list[Goal]:
        return self.get_goals_by_status(GoalStatus.PAUSED)

    # ─── Curiosity agenda ──────────────────────────────────────────

    def add_curiosity_goal(self, goal_description: str) -> None:
        """Add a curiosity task for idle reflection/exploration."""
        for desc, _ in self._curiosity_goals:
            if desc == goal_description:
                return
        self._curiosity_goals.append((goal_description, time.time()))

        node = UnknownNode(
            id=f"curiosity_{abs(hash(goal_description))}",
            question=goal_description,
            tags=["curiosity", "exploration"],
            provenance=Provenance(created_by="intentional_workspace"),
        )
        self._ws.upsert_node(node)

    def get_curiosity_goals(self) -> list[str]:
        return [desc for desc, _ in sorted(self._curiosity_goals, key=lambda x: x[1])]

    def remove_curiosity_goal(self, goal_description: str) -> None:
        self._curiosity_goals = [(d, t) for d, t in self._curiosity_goals if d != goal_description]
        node_id = f"curiosity_{abs(hash(goal_description))}"
        self._ws.remove_node(node_id)

    # ─── Opportunities & Threats ─────────────────────────────────────

    def add_opportunity(
        self, opp_id: str, description: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Log a newly detected opportunity as an HCIR HypothesisNode."""
        self._opportunities[opp_id] = {
            "id": opp_id,
            "description": description,
            "metadata": metadata or {},
            "created_at": time.time(),
        }
        node = HypothesisNode(
            id=opp_id,
            statement=description,
            tags=["opportunity"],
            provenance=Provenance(created_by="intentional_workspace"),
        )
        self._ws.upsert_node(node)

    def get_opportunities(self) -> list[dict[str, Any]]:
        return list(self._opportunities.values())

    def add_threat(
        self,
        threat_id: str,
        description: str,
        severity: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a newly detected system/security threat as an HCIR ConstraintNode."""
        self._threats[threat_id] = {
            "id": threat_id,
            "description": description,
            "severity": severity,
            "metadata": metadata or {},
            "created_at": time.time(),
        }
        node = ConstraintNode(
            id=threat_id,
            constraint_type="security_threat",
            target=description,
            tags=["threat", f"severity:{severity}"],
            provenance=Provenance(created_by="intentional_workspace"),
        )
        self._ws.upsert_node(node)

    def get_threats(self) -> list[dict[str, Any]]:
        return sorted(list(self._threats.values()), key=lambda x: x["severity"], reverse=True)

    # ─── Pending Reflections ─────────────────────────────────────────

    def add_pending_reflection(self, topic: str, details: str = "") -> None:
        """Queue a topic for later reflection during idle cycles."""
        self._reflection_counter += 1
        ref_id = self._reflection_counter
        self._pending_reflections.append(
            {
                "id": ref_id,
                "topic": topic,
                "details": details,
                "created_at": time.time(),
            }
        )

    def get_pending_reflections(self) -> list[dict[str, Any]]:
        """Retrieve all pending reflection topics, oldest first."""
        return list(sorted(self._pending_reflections, key=lambda x: x["created_at"]))

    def remove_pending_reflection(self, reflection_id: int) -> None:
        """Remove a reflection after it has been processed."""
        self._pending_reflections = [
            r for r in self._pending_reflections if r["id"] != reflection_id
        ]
