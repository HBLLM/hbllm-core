"""
HCIR Workspace — the active cognitive state container.

The workspace owns the graph, runtime context, snapshot bookmarks,
resource budgets, and simulation branches.  It does NOT own persistence
(delegated to stores) or transaction lifecycle (delegated to the kernel).

    Workspace
      ├── Graph              (single CognitiveGraph instance)
      ├── Runtime            (attention, active goals, thread states)
      ├── Snapshots          (SnapshotManager bookmarks)
      ├── Resources          (hard + soft budget tracking)
      └── Branches           (simulation forks)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from hbllm.hcir.graph import (
    CognitiveGraph,
    GoalNode,
    HCIREdge,
    HCIRNode,
    HCIRNodeType,
    NodeLifecycle,
)
from hbllm.hcir.query import GraphQuery, InMemoryQueryEngine, QueryResult
from hbllm.hcir.snapshot import Snapshot, SnapshotManager
from hbllm.hcir.stores import IEventStore, InMemoryEventStore
from hbllm.hcir.types import Attention, BranchMode

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Resource Budgets
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ResourceBudget:
    """Tracks a single resource allocation."""

    name: str
    allocated: float = 0.0
    consumed: float = 0.0
    limit: float = float("inf")
    is_hard: bool = True  # Hard = physical; Soft = cognitive

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit - self.consumed)

    @property
    def is_exceeded(self) -> bool:
        return self.consumed > self.limit

    def consume(self, amount: float) -> bool:
        """Consume resources.  Returns False if budget exceeded."""
        if self.consumed + amount > self.limit and self.is_hard:
            return False
        self.consumed += amount
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Workspace State
# ═══════════════════════════════════════════════════════════════════════════


class HCIRWorkspaceState:
    """The active cognitive state container.

    Central blackboard for the HCIR Cognitive OS.  All reasoning nodes
    interact with the workspace through the kernel's transaction manager,
    never directly.

    Usage::

        workspace = HCIRWorkspaceState()
        workspace.add_node(GoalNode(description="Build solar dehydrator"))
        results = workspace.query(GraphQuery(node_type=HCIRNodeType.GOAL))
    """

    def __init__(
        self,
        event_store: IEventStore | None = None,
        branch_mode: BranchMode = BranchMode.LIVE,
    ) -> None:
        self._graph = CognitiveGraph()
        self._event_store = event_store or InMemoryEventStore()
        self._snapshot_manager = SnapshotManager(self._event_store)
        self._query_engine = InMemoryQueryEngine(self._graph)
        self._branch_mode = branch_mode

        # Runtime context
        self._global_attention = Attention()
        self._resources: dict[str, ResourceBudget] = {}

        # Simulation branches (branch_name → cloned workspace)
        self._branches: dict[str, HCIRWorkspaceState] = {}

        # Bootstrap: create initial snapshot
        self._snapshot_manager.create_snapshot(self._graph)

    # ── Properties ───────────────────────────────────────────────────

    @property
    def branch_mode(self) -> BranchMode:
        return self._branch_mode

    @property
    def graph(self) -> CognitiveGraph:
        return self._graph

    @property
    def snapshot_manager(self) -> SnapshotManager:
        return self._snapshot_manager

    @property
    def event_store(self) -> IEventStore:
        return self._event_store

    @property
    def current_version(self) -> int:
        return self._snapshot_manager.current_version

    @property
    def attention(self) -> Attention:
        return self._global_attention

    @attention.setter
    def attention(self, value: Attention) -> None:
        self._global_attention = value

    # ── Graph Operations (recorded in event log) ─────────────────────

    def add_node(self, node: HCIRNode, author: str = "system") -> None:
        """Add a node and record the event."""
        self._graph.add_node(node)
        self._snapshot_manager.record_node_added(node, author)

    def upsert_node(self, node: HCIRNode, author: str = "system") -> None:
        """Add or replace a node and record the event."""
        existing = self._graph.get_node(node.id)
        self._graph.upsert_node(node)
        if existing:
            self._snapshot_manager.record_node_modified(node.id, {"replaced": True}, author)
        else:
            self._snapshot_manager.record_node_added(node, author)

    def remove_node(self, node_id: str, author: str = "system") -> HCIRNode | None:
        """Remove a node and record the event."""
        node = self._graph.remove_node(node_id)
        if node is not None:
            self._snapshot_manager.record_node_removed(node_id, author)
        return node

    def add_edge(self, edge: HCIREdge, author: str = "system") -> None:
        """Add an edge and record the event."""
        self._graph.add_edge(edge)
        self._snapshot_manager.record_edge_added(edge, author)

    def remove_edge(self, edge_id: str, author: str = "system") -> HCIREdge | None:
        """Remove an edge and record the event."""
        edge = self._graph.remove_edge(edge_id)
        if edge is not None:
            self._snapshot_manager.record_edge_removed(edge_id, author)
        return edge

    def get_node(self, node_id: str) -> HCIRNode | None:
        return self._graph.get_node(node_id)

    def get_edge(self, edge_id: str) -> HCIREdge | None:
        return self._graph.get_edge(edge_id)

    # ── Query API ────────────────────────────────────────────────────

    def query(self, q: GraphQuery) -> QueryResult:
        """Execute a declarative graph query."""
        return self._query_engine.query_nodes(q)

    # ── Resource Management ──────────────────────────────────────────

    def register_resource(self, budget: ResourceBudget) -> None:
        """Register a resource budget."""
        self._resources[budget.name] = budget

    def get_resource(self, name: str) -> ResourceBudget | None:
        return self._resources.get(name)

    def consume_resource(self, name: str, amount: float) -> bool:
        """Consume from a resource budget.  Returns False if exceeded."""
        budget = self._resources.get(name)
        if budget is None:
            return True  # Unknown resources are unconstrained
        return budget.consume(amount)

    # ── Snapshot & Branching ─────────────────────────────────────────

    def create_snapshot(self, branch: str = "main") -> Snapshot:
        """Create a version snapshot at the current state."""
        return self._snapshot_manager.create_snapshot(self._graph, branch)

    def fork(
        self,
        branch_name: str,
        mode: BranchMode = BranchMode.SIMULATION,
    ) -> HCIRWorkspaceState:
        """Fork the workspace into an isolated branch (simulation, replay, training).

        Returns a deep-copied workspace that can be mutated
        independently.  Changes are NOT reflected in the parent
        until explicitly merged.
        """
        if branch_name in self._branches:
            raise ValueError(f"Branch '{branch_name}' already exists")

        # Deep copy the graph and create a new workspace with specified branch mode
        forked = HCIRWorkspaceState(branch_mode=mode)
        for node in self._graph.all_nodes():
            forked._graph.add_node(node.model_copy(deep=True))
        for edge in self._graph.all_edges():
            forked._graph.add_edge(edge.model_copy(deep=True))
        forked._global_attention = self._global_attention.model_copy(deep=True)
        forked._resources = {
            k: ResourceBudget(
                name=v.name,
                allocated=v.allocated,
                consumed=v.consumed,
                limit=v.limit,
                is_hard=v.is_hard,
            )
            for k, v in self._resources.items()
        }

        self._branches[branch_name] = forked
        logger.info(
            "Forked workspace branch '%s' (nodes=%d, edges=%d)",
            branch_name,
            forked._graph.node_count,
            forked._graph.edge_count,
        )
        return forked

    def get_branch(self, branch_name: str) -> HCIRWorkspaceState | None:
        """Get a forked branch workspace."""
        return self._branches.get(branch_name)

    def fork_branch(self, branch_name: str) -> HCIRWorkspaceState:
        """Convenience alias for fork()."""
        return self.fork(branch_name)

    def merge_branch(self, branch_name: str) -> bool:
        """Merge a forked branch into the main workspace state."""
        branch_ws = self._branches.pop(branch_name, None)
        if branch_ws is None:
            return False
        # Copy newly created nodes/edges from branch to main workspace
        for node in branch_ws._graph.all_nodes():
            self.upsert_node(node)
        for edge in branch_ws._graph.all_edges():
            self.add_edge(edge)
        return True

    def drop_branch(self, branch_name: str) -> bool:
        """Discard a simulation branch without merging."""
        return self._branches.pop(branch_name, None) is not None

    @property
    def branch_names(self) -> list[str]:
        return list(self._branches.keys())

    # ── Convenience Views ────────────────────────────────────────────

    def active_goals(self) -> list[GoalNode]:
        """Return all goal nodes in non-terminal lifecycle states."""
        goals = self._graph.nodes_by_type(HCIRNodeType.GOAL)
        return [
            g
            for g in goals
            if isinstance(g, GoalNode)
            and g.lifecycle not in (NodeLifecycle.ARCHIVED, NodeLifecycle.FORGOTTEN)
        ]
