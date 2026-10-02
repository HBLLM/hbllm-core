"""
Tiered Workspace — 4-tier cognitive state hierarchy with task frames.

Separates transient reasoning from persistent knowledge:

    Working Workspace   →  Task frames — ephemeral scratch per goal
    Brain Workspace     →  Session-level cognitive state
    Persistent Workspace →  Long-term memory graph (forever)
    Meta Workspace      →  Cognitive performance & self-model

Each tier has different lifetime, eviction, and promotion semantics.
Task frames within the working tier preserve scratch reasoning across
conversational turns without flushing per-query.

Usage::

    tiered = TieredWorkspace()
    frame = tiered.create_task_frame("goal_123")
    frame.workspace.add_node(GoalNode(...))
    tiered.close_task_frame(frame.frame_id, reason="completed")
    tiered.promote("node_xyz", WorkspaceTier.WORKING, WorkspaceTier.PERSISTENT)
"""

from __future__ import annotations

import logging
import time
import uuid
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from hbllm.hcir.graph import (
    GoalNode,
    HCIRNode,
    HCIRNodeType,
    NodeLifecycle,
    SkillNode,
)
from hbllm.hcir.query import GraphQuery, QueryResult
from hbllm.hcir.snapshot import Snapshot
from hbllm.hcir.stores import IEventStore, InMemoryEventStore
from hbllm.hcir.types import BranchMode
from hbllm.hcir.workspace import HCIRWorkspaceState

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Workspace Tier
# ═══════════════════════════════════════════════════════════════════════════


class WorkspaceTier(StrEnum):
    """The four tiers of the cognitive workspace hierarchy."""

    WORKING = "working"  # Task frames — ephemeral scratch per goal
    BRAIN = "brain"  # Session-level cognitive state
    PERSISTENT = "persistent"  # Long-term memory graph (forever)
    META = "meta"  # Cognitive performance & self-model


# ═══════════════════════════════════════════════════════════════════════════
# Task Frame
# ═══════════════════════════════════════════════════════════════════════════

TaskFrameStatus = Literal["active", "completed", "abandoned", "evicted"]


class TaskFrame:
    """Scoped reasoning context within a conversation.

    A frame ends when:
        - The goal completes
        - The user changes topics
        - The planner abandons it
        - Memory pressure requires eviction

    Each frame has its own workspace for scratch reasoning that
    persists across multiple conversational turns.
    """

    def __init__(
        self,
        goal_id: str,
        frame_id: str | None = None,
    ) -> None:
        self.frame_id: str = frame_id or f"frame_{uuid.uuid4().hex[:8]}"
        self.goal_id: str = goal_id
        self.workspace: HCIRWorkspaceState = HCIRWorkspaceState(
            branch_mode=BranchMode.LIVE,
            branch_name=f"frame_{self.frame_id}",
        )
        self.created_at: float = time.time()
        self.closed_at: float | None = None
        self.status: TaskFrameStatus = "active"

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    def close(self, reason: str = "completed") -> None:
        """Close this task frame."""
        if reason in ("completed", "abandoned", "evicted"):
            self.status = reason  # type: ignore[assignment]
        else:
            self.status = "completed"
        self.closed_at = time.time()
        logger.info(
            "TaskFrame %s closed: reason=%s goal=%s age=%.1fs",
            self.frame_id,
            reason,
            self.goal_id,
            self.age_seconds,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Interruption Checkpoint & Goal Stack
# ═══════════════════════════════════════════════════════════════════════════


class InterruptionCheckpoint(BaseModel):
    """Snapshot of a suspended task frame when preempted by an urgent interrupt or higher-priority goal."""

    checkpoint_id: str = Field(default_factory=lambda: f"ckpt_{uuid.uuid4().hex[:8]}")
    parent_goal_id: str
    parent_frame_id: str
    interrupt_goal_id: str
    interrupt_frame_id: str
    target_conditions: list[str] = Field(default_factory=list)
    unsatisfied_conditions: list[str] = Field(default_factory=list)
    in_flight_action: str | None = None
    step_index: int = 0
    timestamp: float = Field(default_factory=time.time)
    context_data: dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Working Workspace — manages multiple task frames
# ═══════════════════════════════════════════════════════════════════════════


class WorkingWorkspace:
    """Working tier: manages a collection of active task frames and hierarchical goal interruptions.

    Each frame scopes scratch reasoning to a specific goal.
    Frames persist across conversational turns and are closed when
    the goal completes, topics change, or eviction is needed.
    """

    def __init__(self, max_frames: int = 10) -> None:
        self._frames: dict[str, TaskFrame] = {}
        self._max_frames = max_frames
        self._interruption_stack: list[InterruptionCheckpoint] = []

    @property
    def interruption_stack(self) -> list[InterruptionCheckpoint]:
        """Return the current stack of active interruption checkpoints."""
        return list(self._interruption_stack)

    @property
    def has_active_interruptions(self) -> bool:
        """Check whether there are active interrupted goal checkpoints pending resumption."""
        return len(self._interruption_stack) > 0

    def push_interruption(
        self,
        parent_frame_id: str,
        interrupt_goal_id: str,
        target_conditions: list[str] | None = None,
        unsatisfied_conditions: list[str] | None = None,
        in_flight_action: str | None = None,
        step_index: int = 0,
        context_data: dict[str, Any] | None = None,
    ) -> tuple[TaskFrame, InterruptionCheckpoint]:
        """Suspend an active parent frame, spawn an interrupt frame, and push checkpoint to stack."""
        parent_frame = self._frames.get(parent_frame_id)
        parent_goal = parent_frame.goal_id if parent_frame else "unknown_goal"

        interrupt_frame = self.create_frame(interrupt_goal_id)

        checkpoint = InterruptionCheckpoint(
            parent_goal_id=parent_goal,
            parent_frame_id=parent_frame_id,
            interrupt_goal_id=interrupt_goal_id,
            interrupt_frame_id=interrupt_frame.frame_id,
            target_conditions=target_conditions or [],
            unsatisfied_conditions=unsatisfied_conditions or [],
            in_flight_action=in_flight_action,
            step_index=step_index,
            context_data=context_data or {},
        )
        self._interruption_stack.append(checkpoint)
        logger.info(
            "Pushed interruption checkpoint %s: parent_goal=%s -> interrupt_goal=%s (stack depth=%d)",
            checkpoint.checkpoint_id,
            parent_goal,
            interrupt_goal_id,
            len(self._interruption_stack),
        )
        return interrupt_frame, checkpoint

    def pop_interruption(
        self,
        interrupt_frame_id: str | None = None,
        reason: str = "completed",
    ) -> InterruptionCheckpoint | None:
        """Close current interrupt frame and restore parent checkpoint from stack."""
        if not self._interruption_stack:
            return None

        if interrupt_frame_id is not None:
            idx = next(
                (
                    i
                    for i, ckpt in enumerate(reversed(self._interruption_stack))
                    if ckpt.interrupt_frame_id == interrupt_frame_id
                ),
                None,
            )
            if idx is not None:
                actual_idx = len(self._interruption_stack) - 1 - idx
                checkpoint = self._interruption_stack.pop(actual_idx)
            else:
                checkpoint = self._interruption_stack.pop()
        else:
            checkpoint = self._interruption_stack.pop()

        self.close_frame(checkpoint.interrupt_frame_id, reason=reason)
        logger.info(
            "Popped interruption checkpoint %s: resumed parent_goal=%s (remaining depth=%d)",
            checkpoint.checkpoint_id,
            checkpoint.parent_goal_id,
            len(self._interruption_stack),
        )
        return checkpoint

    def get_active_checkpoint(self, goal_id: str | None = None) -> InterruptionCheckpoint | None:
        """Get the topmost active checkpoint, optionally matching goal_id."""
        if not self._interruption_stack:
            return None
        if goal_id is None:
            return self._interruption_stack[-1]
        for ckpt in reversed(self._interruption_stack):
            if ckpt.parent_goal_id == goal_id or ckpt.interrupt_goal_id == goal_id:
                return ckpt
        return None

    @property
    def active_frames(self) -> list[TaskFrame]:
        """Return all active task frames."""
        return [f for f in self._frames.values() if f.is_active]

    @property
    def all_frames(self) -> list[TaskFrame]:
        """Return all frames (active and closed)."""
        return list(self._frames.values())

    def create_frame(self, goal_id: str) -> TaskFrame:
        """Create a new task frame for a goal.

        If the maximum number of frames is exceeded, the oldest
        inactive frame is evicted.
        """
        # Evict oldest inactive frames if needed
        while len(self._frames) >= self._max_frames:
            evicted = self._evict_oldest_inactive()
            if not evicted:
                # No inactive frames to evict — evict oldest active
                evicted = self._evict_oldest_active()
                if not evicted:
                    break  # Can't evict anything

        frame = TaskFrame(goal_id=goal_id)
        self._frames[frame.frame_id] = frame
        logger.info(
            "Created task frame %s for goal %s (total=%d)",
            frame.frame_id,
            goal_id,
            len(self._frames),
        )
        return frame

    def get_frame(self, frame_id: str) -> TaskFrame | None:
        """Get a task frame by ID."""
        return self._frames.get(frame_id)

    def get_frame_by_goal(self, goal_id: str) -> TaskFrame | None:
        """Get the active task frame for a goal."""
        for frame in self._frames.values():
            if frame.goal_id == goal_id and frame.is_active:
                return frame
        return None

    def close_frame(self, frame_id: str, reason: str = "completed") -> bool:
        """Close a task frame."""
        frame = self._frames.get(frame_id)
        if frame is None:
            return False
        frame.close(reason)
        return True

    def get_node_across_frames(self, node_id: str) -> HCIRNode | None:
        """Search for a node across all active task frames."""
        for frame in self.active_frames:
            node = frame.workspace.get_node(node_id)
            if node is not None:
                return node
        return None

    def cleanup_closed(self, max_age_seconds: float = 3600) -> int:
        """Remove closed frames older than max_age_seconds."""
        now = time.time()
        to_remove = [
            fid
            for fid, frame in self._frames.items()
            if not frame.is_active and frame.closed_at and (now - frame.closed_at) > max_age_seconds
        ]
        for fid in to_remove:
            del self._frames[fid]
        return len(to_remove)

    def _evict_oldest_inactive(self) -> bool:
        """Evict the oldest inactive frame. Returns True if evicted."""
        inactive = [f for f in self._frames.values() if not f.is_active]
        if not inactive:
            return False
        oldest = min(inactive, key=lambda f: f.created_at)
        oldest.close("evicted")
        del self._frames[oldest.frame_id]
        return True

    def _evict_oldest_active(self) -> bool:
        """Evict the oldest active frame. Returns True if evicted."""
        active = self.active_frames
        if not active:
            return False
        oldest = min(active, key=lambda f: f.created_at)
        oldest.close("evicted")
        del self._frames[oldest.frame_id]
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Tiered Workspace
# ═══════════════════════════════════════════════════════════════════════════


class TieredWorkspace:
    """4-tier cognitive workspace hierarchy.

    Manages the lifecycle and promotion of cognitive state across
    four tiers:

        Working (task frames) → Brain (session) → Persistent (forever) → Meta (self-model)

    Promotion rules:
        - Skill succeeds 3x in working → promote to persistent
        - Belief confirmed by 2+ sources → promote to persistent
        - Episode with reward > 0.7 → promote to persistent
        - Performance metric → always goes to meta workspace

    Usage::

        tiered = TieredWorkspace()
        frame = tiered.create_task_frame("goal_abc")
        frame.workspace.add_node(GoalNode(description="Plan dinner"))
        tiered.promote("node_xyz", WorkspaceTier.WORKING, WorkspaceTier.PERSISTENT)
    """

    def __init__(
        self,
        persistent_store: IEventStore | None = None,
        snapshot_interval: int = 1000,
    ) -> None:
        self.working = WorkingWorkspace()
        self.brain = HCIRWorkspaceState(
            branch_mode=BranchMode.LIVE,
            branch_name="brain",
        )
        self.persistent = HCIRWorkspaceState(
            event_store=persistent_store or InMemoryEventStore(),
            branch_mode=BranchMode.LIVE,
            branch_name="persistent",
        )
        self.meta = HCIRWorkspaceState(
            branch_mode=BranchMode.LIVE,
            branch_name="meta",
        )

        self._snapshot_interval = snapshot_interval
        self._commits_since_snapshot: int = 0

    # ── Task Frame & Interruption Management ─────────────────────────

    def create_task_frame(self, goal_id: str) -> TaskFrame:
        """Create a new task frame in the working tier."""
        return self.working.create_frame(goal_id)

    def close_task_frame(self, frame_id: str, reason: str = "completed") -> bool:
        """Close a task frame in the working tier."""
        return self.working.close_frame(frame_id, reason)

    def push_interruption(
        self,
        parent_frame_id: str,
        interrupt_goal_id: str,
        target_conditions: list[str] | None = None,
        unsatisfied_conditions: list[str] | None = None,
        in_flight_action: str | None = None,
        step_index: int = 0,
        context_data: dict[str, Any] | None = None,
    ) -> tuple[TaskFrame, InterruptionCheckpoint]:
        """Suspend an active parent frame and push an interrupt checkpoint."""
        return self.working.push_interruption(
            parent_frame_id=parent_frame_id,
            interrupt_goal_id=interrupt_goal_id,
            target_conditions=target_conditions,
            unsatisfied_conditions=unsatisfied_conditions,
            in_flight_action=in_flight_action,
            step_index=step_index,
            context_data=context_data,
        )

    def pop_interruption(
        self,
        interrupt_frame_id: str | None = None,
        reason: str = "completed",
    ) -> InterruptionCheckpoint | None:
        """Close interrupt frame and restore parent checkpoint from stack."""
        return self.working.pop_interruption(interrupt_frame_id=interrupt_frame_id, reason=reason)

    def get_active_checkpoint(self, goal_id: str | None = None) -> InterruptionCheckpoint | None:
        """Get the topmost active checkpoint, optionally matching goal_id."""
        return self.working.get_active_checkpoint(goal_id=goal_id)

    @property
    def interruption_stack(self) -> list[InterruptionCheckpoint]:
        """Return the current stack of active interruption checkpoints."""
        return self.working.interruption_stack

    @property
    def has_active_interruptions(self) -> bool:
        """Check whether there are active interrupted goal checkpoints pending resumption."""
        return self.working.has_active_interruptions

    # ── Tier Access ──────────────────────────────────────────────────

    def get_tier(self, tier: WorkspaceTier) -> HCIRWorkspaceState | None:
        """Get the workspace for a specific tier.

        For the working tier, returns None (use task frames instead).
        """
        if tier == WorkspaceTier.BRAIN:
            return self.brain
        elif tier == WorkspaceTier.PERSISTENT:
            return self.persistent
        elif tier == WorkspaceTier.META:
            return self.meta
        return None

    # ── Node Promotion / Demotion ────────────────────────────────────

    def promote(
        self,
        node_id: str,
        from_tier: WorkspaceTier,
        to_tier: WorkspaceTier,
    ) -> bool:
        """Promote a node from one tier to a higher tier.

        The node is copied (not moved) — it remains in the source tier.

        Args:
            node_id: The HCIR node ID to promote.
            from_tier: Source tier.
            to_tier: Destination tier.

        Returns:
            True if the node was promoted successfully.
        """
        source_node = self._find_node_in_tier(node_id, from_tier)
        if source_node is None:
            logger.warning("promote: node %s not found in %s", node_id, from_tier)
            return False

        dest_ws = self.get_tier(to_tier)
        if dest_ws is None:
            logger.warning("promote: invalid destination tier %s", to_tier)
            return False

        # Copy the node to the destination tier
        promoted = source_node.model_copy(deep=True)
        dest_ws.upsert_node(promoted, author=f"promotion:{from_tier}→{to_tier}")

        logger.info(
            "Promoted node %s from %s to %s",
            node_id,
            from_tier,
            to_tier,
        )
        return True

    def demote(
        self,
        node_id: str,
        from_tier: WorkspaceTier,
        to_tier: WorkspaceTier,
    ) -> bool:
        """Demote a node from one tier to a lower tier.

        The node is moved — removed from source, added to destination.
        """
        source_ws = self.get_tier(from_tier)
        if source_ws is None:
            return False

        node = source_ws.get_node(node_id)
        if node is None:
            return False

        dest_ws = self.get_tier(to_tier)
        if dest_ws is None:
            return False

        # Move: add to destination, remove from source
        dest_ws.upsert_node(node.model_copy(deep=True), author=f"demotion:{from_tier}→{to_tier}")
        source_ws.remove_node(node_id, author=f"demotion:{from_tier}→{to_tier}")

        logger.info("Demoted node %s from %s to %s", node_id, from_tier, to_tier)
        return True

    # ── Cross-Tier Queries ───────────────────────────────────────────

    def query_across_tiers(
        self,
        query: GraphQuery,
        tiers: list[WorkspaceTier] | None = None,
    ) -> QueryResult:
        """Execute a query across multiple workspace tiers.

        Results are merged with persistent results first,
        then brain, then working frames.

        Args:
            query: The graph query to execute.
            tiers: Which tiers to search (default: all).

        Returns:
            Combined QueryResult from all specified tiers.
        """
        search_tiers = tiers or [
            WorkspaceTier.PERSISTENT,
            WorkspaceTier.BRAIN,
            WorkspaceTier.WORKING,
            WorkspaceTier.META,
        ]

        all_nodes: list[HCIRNode] = []
        seen_ids: set[str] = set()

        for tier in search_tiers:
            if tier == WorkspaceTier.WORKING:
                # Search all active task frames
                for frame in self.working.active_frames:
                    result = frame.workspace.query(query)
                    for node in result.nodes:
                        if node.id not in seen_ids:
                            all_nodes.append(node)
                            seen_ids.add(node.id)
            else:
                ws = self.get_tier(tier)
                if ws is not None:
                    result = ws.query(query)
                    for node in result.nodes:
                        if node.id not in seen_ids:
                            all_nodes.append(node)
                            seen_ids.add(node.id)

        return QueryResult(nodes=all_nodes, total_matches=len(all_nodes))

    # ── Lifecycle ────────────────────────────────────────────────────

    def archive_brain(self) -> int:
        """Archive brain workspace contents to persistent tier.

        Called at end of session.  Promotes all brain nodes to
        persistent, then clears the brain workspace.

        Returns:
            Number of nodes archived.
        """
        count = 0
        for node in self.brain.graph.all_nodes():
            self.persistent.upsert_node(
                node.model_copy(deep=True),
                author="archive:brain→persistent",
            )
            count += 1

        # Copy edges too
        for edge in self.brain.graph.all_edges():
            try:
                self.persistent.add_edge(
                    edge.model_copy(deep=True),
                    author="archive:brain→persistent",
                )
            except Exception:
                pass  # Edge may reference nodes already present

        logger.info("Archived %d brain nodes to persistent workspace", count)
        return count

    def snapshot(self, tier: WorkspaceTier) -> Snapshot | None:
        """Create a graph snapshot for a specific tier.

        Useful for periodic snapshotting to avoid replaying
        millions of events on startup.
        """
        ws = self.get_tier(tier)
        if ws is None:
            return None
        return ws.create_snapshot(branch=tier.value)

    def notify_commit(self) -> None:
        """Notify the workspace that a transaction was committed.

        Triggers periodic snapshotting of the persistent tier.
        """
        self._commits_since_snapshot += 1
        if self._commits_since_snapshot >= self._snapshot_interval:
            self.snapshot(WorkspaceTier.PERSISTENT)
            self._commits_since_snapshot = 0
            logger.info(
                "Auto-snapshot persistent workspace (every %d commits)",
                self._snapshot_interval,
            )

    def populate_meta_stats(self) -> dict[str, int | float | dict[str, int]]:
        """Populate the Meta workspace with cognitive performance statistics.

        Reads node counts, skill success rates, goal completion rates,
        and tier utilization.  Stores the results as an ObservationNode
        in the meta workspace for the MetaReasoningNode to consume.

        Returns:
            Dict of computed statistics.
        """
        from hbllm.hcir.graph import ObservationNode

        # Count nodes per tier
        persistent_query = GraphQuery(limit=9999)
        persistent_result = self.persistent.query(persistent_query)
        brain_result = self.brain.query(persistent_query)

        # Count by type in persistent
        type_counts: dict[str, int] = {}
        for node in persistent_result.nodes:
            t = node.node_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        # Compute skill success rates
        skill_query = GraphQuery(node_type=HCIRNodeType.SKILL, limit=9999)
        skills = self.persistent.query(skill_query)
        avg_success = 0.0
        if skills.total_matches > 0:
            total = sum(n.success_rate for n in skills.nodes if isinstance(n, SkillNode))
            avg_success = total / skills.total_matches

        # Compute goal completion
        goal_query = GraphQuery(node_type=HCIRNodeType.GOAL, limit=9999)
        goals = self.persistent.query(goal_query)
        completed_goals = sum(
            1
            for g in goals.nodes
            if isinstance(g, GoalNode) and g.lifecycle == NodeLifecycle.ARCHIVED
        )
        goal_completion_rate = (
            completed_goals / goals.total_matches if goals.total_matches > 0 else 0.0
        )

        stats: dict[str, int | float | dict[str, int]] = {
            "persistent_node_count": persistent_result.total_matches,
            "brain_node_count": brain_result.total_matches,
            "active_task_frames": len(self.working.all_frames),
            "type_counts": type_counts,
            "avg_skill_success_rate": round(avg_success, 3),
            "goal_completion_rate": round(goal_completion_rate, 3),
            "commits_since_snapshot": self._commits_since_snapshot,
        }

        # Store as an observation node in the meta tier
        meta_node = ObservationNode(
            id="meta_stats_latest",
            sensor_source="tiered_workspace",
            tags=["meta", "self_model", "performance_stats"],
        )
        self.meta.upsert_node(meta_node, author="tiered_workspace")

        logger.debug("Meta workspace stats updated: %s", stats)
        return stats

    # ── Internal ─────────────────────────────────────────────────────

    def _find_node_in_tier(
        self,
        node_id: str,
        tier: WorkspaceTier,
    ) -> HCIRNode | None:
        """Find a node in a specific tier."""
        if tier == WorkspaceTier.WORKING:
            return self.working.get_node_across_frames(node_id)
        ws = self.get_tier(tier)
        if ws is None:
            return None
        return ws.get_node(node_id)
