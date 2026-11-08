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
from typing import Any

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
from hbllm.hcir.tier_types import WorkspaceTier
from hbllm.hcir.transactions import HCIRTransaction, TransactionOp, TransactionOperation
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
        workspace.commit_operations(
            [TransactionOperation(op=TransactionOp.UPSERT_NODE, node=GoalNode(description="Build solar dehydrator"))]
        )
        results = workspace.query(GraphQuery(node_type=HCIRNodeType.GOAL))
    """

    def __init__(
        self,
        event_store: IEventStore | None = None,
        branch_mode: BranchMode = BranchMode.LIVE,
        branch_name: str = "main",
        graph: CognitiveGraph | None = None,
        tier: WorkspaceTier = WorkspaceTier.WORKING,
    ) -> None:
        self._tier = tier
        self._graph = graph if graph is not None else CognitiveGraph()
        self._event_store = event_store or InMemoryEventStore()
        self._snapshot_manager = SnapshotManager(self._event_store)
        self._query_engine = InMemoryQueryEngine(self._graph)
        self._branch_mode = branch_mode
        self._branch_name = branch_name

        # Runtime context
        self._global_attention = Attention()
        self._resources: dict[str, ResourceBudget] = {}

        # Simulation branches (branch_name → cloned workspace)
        self._branches: dict[str, HCIRWorkspaceState] = {}

        # Single commit authority bound to this workspace (attached by the kernel)
        self._transaction_manager: Any | None = None

        # Bootstrap: create initial snapshot
        self._snapshot_manager.create_snapshot(self._graph)

    # ── Properties ───────────────────────────────────────────────────

    @property
    def branch_name(self) -> str:
        return self._branch_name

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

    # ── Commit Authority ─────────────────────────────────────────────

    def attach_transaction_manager(self, manager: Any, replace: bool = False) -> None:
        """Bind the kernel ``TransactionManager`` that governs this workspace.

        The first manager attached wins unless ``replace`` is set, so that the
        governed kernel manager (with its verification pipeline) remains the
        single commit authority even if ad-hoc managers are created later.
        """
        if self._transaction_manager is None or replace:
            self._transaction_manager = manager

    @property
    def transaction_manager(self) -> Any:
        """The commit authority for this workspace.

        Lazily creates a bare kernel manager when none has been attached so that
        every write still flows through the atomic, event-sourced commit path.
        """
        if self._transaction_manager is None:
            from hbllm.hcir.kernel.transaction_manager import TransactionManager

            self._transaction_manager = TransactionManager(self)
        return self._transaction_manager

    @property
    def tier(self) -> WorkspaceTier:
        """The workspace's tier (read-only; set at construction)."""
        return self._tier

    def commit_operations(
        self,
        operations: list[TransactionOperation],
        author: str = "system",
        metadata: dict[str, Any] | None = None,
        grant: Any = None,
    ) -> HCIRTransaction:
        """Propose and commit a batch of operations through the commit authority.

        This is the canonical write path for subsystems that hold a workspace
        reference but no ``KernelServices`` (adapters, tier promotion, world kernel).
        ``grant`` is a :class:`CommitGrant` issued by the kernel's
        CommitAuthorityRegistry for gated tiers (WP6.4).
        """
        meta = dict(metadata or {})
        if grant is not None:
            meta["commit_grant"] = grant.grant_id
        tx = HCIRTransaction(author=author, operations=list(operations), metadata=meta)
        return self.transaction_manager.commit(tx)

    # ── Graph Operations (kernel-internal; recorded in event log) ────
    # These are the raw primitives used exclusively by TransactionManager.
    # External callers must use ``commit_operations`` or a transaction.

    def _add_node(self, node: HCIRNode, author: str = "system") -> None:
        """Add a node and record the event (kernel-internal)."""
        self._graph.add_node(node)
        self._snapshot_manager.record_node_added(node, author)

    def _upsert_node(self, node: HCIRNode, author: str = "system") -> None:
        """Add or replace a node and record the event (kernel-internal)."""
        existing = self._graph.get_node(node.id)
        self._graph.upsert_node(node)
        if existing:
            self._snapshot_manager.record_node_modified(
                node.id, {"replaced": True}, author, node=node
            )
        else:
            self._snapshot_manager.record_node_added(node, author)

    def _remove_node(self, node_id: str, author: str = "system") -> HCIRNode | None:
        """Remove a node and record the event (kernel-internal)."""
        node = self._graph.remove_node(node_id)
        if node is not None:
            self._snapshot_manager.record_node_removed(node_id, author)
        return node

    def _add_edge(self, edge: HCIREdge, author: str = "system") -> None:
        """Add an edge and record the event (kernel-internal)."""
        self._graph.add_edge(edge)
        self._snapshot_manager.record_edge_added(edge, author)

    def _upsert_edge(self, edge: HCIREdge, author: str = "system") -> None:
        """Add or update an edge and record the event (kernel-internal)."""
        if self._graph.has_edge(edge.id):
            self._graph.remove_edge(edge.id)
            self._snapshot_manager.record_edge_removed(edge.id, author)
        self._graph.add_edge(edge)
        self._snapshot_manager.record_edge_added(edge, author)

    def _remove_edge(self, edge_id: str, author: str = "system") -> HCIREdge | None:
        """Remove an edge and record the event (kernel-internal)."""
        edge = self._graph.remove_edge(edge_id)
        if edge is not None:
            self._snapshot_manager.record_edge_removed(edge_id, author)
        return edge

    def add_node(self, node: HCIRNode, author: str = "system") -> None:
        """Deprecated: Use commit_operations or TransactionManager."""
        import warnings

        warnings.warn(
            "Direct workspace.add_node is deprecated; routing through commit_operations. Removal: next minor release.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.commit_operations(
            [TransactionOperation(op=TransactionOp.ADD_NODE, node_data=node.model_dump())],
            author=author,
        )

    def upsert_node(self, node: HCIRNode, author: str = "system") -> None:
        """Deprecated: Use commit_operations or TransactionManager."""
        import warnings

        warnings.warn(
            "Direct workspace.upsert_node is deprecated; routing through commit_operations. Removal: next minor release.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.commit_operations(
            [TransactionOperation(op=TransactionOp.UPSERT_NODE, node_data=node.model_dump())],
            author=author,
        )

    def remove_node(self, node_id: str, author: str = "system") -> HCIRNode | None:
        """Deprecated: Use commit_operations or TransactionManager."""
        import warnings

        warnings.warn(
            "Direct workspace.remove_node is deprecated; routing through commit_operations. Removal: next minor release.",
            DeprecationWarning,
            stacklevel=2,
        )
        existing = self.get_node(node_id)
        self.commit_operations(
            [TransactionOperation(op=TransactionOp.REMOVE_NODE, node_id=node_id)],
            author=author,
        )
        return existing

    def add_edge(self, edge: HCIREdge, author: str = "system") -> None:
        """Deprecated: Use commit_operations or TransactionManager."""
        import warnings

        warnings.warn(
            "Direct workspace.add_edge is deprecated; routing through commit_operations. Removal: next minor release.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.commit_operations(
            [TransactionOperation(op=TransactionOp.ADD_EDGE, edge_data=edge.model_dump())],
            author=author,
        )

    def remove_edge(self, edge_id: str, author: str = "system") -> HCIREdge | None:
        """Deprecated: Use commit_operations or TransactionManager."""
        import warnings

        warnings.warn(
            "Direct workspace.remove_edge is deprecated; routing through commit_operations. Removal: next minor release.",
            DeprecationWarning,
            stacklevel=2,
        )
        existing = self.get_edge(edge_id)
        self.commit_operations(
            [TransactionOperation(op=TransactionOp.REMOVE_EDGE, edge_id=edge_id)],
            author=author,
        )
        return existing

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

        # Structural sharing copy-on-write fork (WP4: < 1 ms @ 10k nodes)
        forked = HCIRWorkspaceState(branch_mode=mode, branch_name=branch_name, tier=self._tier)
        forked._graph = self._graph.fork()
        forked._graph._fork_base_version = self._snapshot_manager.current_version
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

    def resume_from_event(self, sequence_or_id: int | str) -> HCIRWorkspaceState:
        """Roll back or replay the event log to the specified point-in-time and produce
        an active workspace ready for new transactions (WP7).
        """
        all_events = self._event_store.get_events()
        target_seq = int(sequence_or_id) if str(sequence_or_id).isdigit() else None

        target_idx = -1
        for idx, ev in enumerate(all_events):
            if (target_seq is not None and ev.sequence == target_seq) or getattr(
                ev, "id", None
            ) == sequence_or_id:
                target_idx = idx
                break

        if target_idx == -1:
            raise ValueError(f"Event point '{sequence_or_id}' not found in event store")

        from hbllm.hcir.stores import EventType

        resumed = HCIRWorkspaceState(
            branch_mode=self._branch_mode,
            branch_name=f"{self._branch_name}_resumed_{sequence_or_id}",
        )
        for ev in all_events[: target_idx + 1]:
            d = ev.data or {}
            if ev.event_type == EventType.NODE_ADDED:
                node_payload = d.get("node") or d.get("node_data")
                if node_payload is not None:
                    resumed._graph.add_node(
                        node_payload
                        if isinstance(node_payload, HCIRNode)
                        else HCIRNode.model_validate(node_payload)
                    )
            elif ev.event_type == EventType.NODE_MODIFIED:
                node_payload = d.get("node") or d.get("node_data")
                if node_payload is not None:
                    resumed._graph.upsert_node(
                        node_payload
                        if isinstance(node_payload, HCIRNode)
                        else HCIRNode.model_validate(node_payload)
                    )
                elif "node_id" in d and "changes" in d:
                    if resumed._graph.has_node(d["node_id"]):
                        curr = resumed._graph.get_node(d["node_id"])
                        for k, v in d["changes"].items():
                            if k == "properties" and isinstance(v, dict):
                                curr.properties.update(v)
                            elif hasattr(curr, k):
                                setattr(curr, k, v)
            elif ev.event_type == EventType.NODE_REMOVED and "node_id" in d:
                resumed._graph.remove_node(d["node_id"])
            elif ev.event_type == EventType.EDGE_ADDED:
                edge_payload = d.get("edge") or d.get("edge_data")
                if edge_payload is not None:
                    resumed._graph.add_edge(
                        edge_payload
                        if isinstance(edge_payload, HCIREdge)
                        else HCIREdge.model_validate(edge_payload)
                    )
            elif ev.event_type == EventType.EDGE_REMOVED and "edge_id" in d:
                resumed._graph.remove_edge(d["edge_id"])

        return resumed

    def diff_branch(self, branch_name: str) -> list[TransactionOperation]:
        """Compute the transactional diff that would bring this workspace to the
        branch's state (adds, upserts, removals of nodes and edges).

        Pure: does not mutate anything. Operation order is safe to apply:
        edge removals → node removals → node adds/upserts → edge adds.
        """
        branch_ws = self._branches.get(branch_name)
        if branch_ws is None:
            raise ValueError(f"Branch '{branch_name}' not found")

        main_g, br_g = self._graph, branch_ws._graph
        # diff_branch is read-only: use the underlying dicts directly.
        main_nodes = main_g._nodes
        br_nodes = br_g._nodes
        main_edges = main_g._edges
        br_edges = br_g._edges

        # Fast path (WP4.4): a graph forked from this workspace tracks the ids it
        # mutated; only dirty ids can differ, so the rest of the comparison is
        # skipped. It is only sound while the parent is unchanged since the fork —
        # if this workspace committed afterwards, parent-side changes live outside
        # the branch's dirty sets, so fall back to the full O(N) comparison.
        dirty_nodes: set[str] | None = None
        dirty_edges: set[str] | None = None
        if (
            getattr(br_g, "_fork_parent", None) is main_g
            and getattr(br_g, "_fork_base_version", None) == self._snapshot_manager.current_version
        ):
            dirty_nodes = getattr(br_g, "_dirty_nodes", None)
            dirty_edges = getattr(br_g, "_dirty_edges", None)

        if dirty_nodes is None:
            removed_nodes = [nid for nid in main_nodes if nid not in br_nodes]
        else:
            removed_nodes = [
                nid for nid in main_nodes if nid in dirty_nodes and nid not in br_nodes
            ]
        if dirty_edges is None:
            changed_edges = {
                eid
                for eid, e in br_edges.items()
                if eid in main_edges and main_edges[eid].model_dump() != e.model_dump()
            }
        else:
            changed_edges = {
                eid
                for eid in dirty_edges
                if eid in main_edges
                and eid in br_edges
                and main_edges[eid].model_dump() != br_edges[eid].model_dump()
            }
        ops: list[TransactionOperation] = []
        for eid in main_edges:
            if dirty_edges is not None and eid not in dirty_edges:
                continue
            if eid not in br_edges or eid in changed_edges:
                edge = main_edges[eid]
                if any(n in removed_nodes for n in [*edge.sources, *edge.targets]):
                    continue  # cascaded by the node removal
                ops.append(TransactionOperation(op=TransactionOp.REMOVE_EDGE, edge_id=eid))
        for nid in removed_nodes:
            ops.append(TransactionOperation(op=TransactionOp.REMOVE_NODE, node_id=nid))
        for nid, node in br_nodes.items():
            if dirty_nodes is not None and nid not in dirty_nodes:
                continue
            existing = main_nodes.get(nid)
            if existing is None:
                ops.append(
                    TransactionOperation(
                        op=TransactionOp.ADD_NODE, node_id=nid, node_data=node.model_dump()
                    )
                )
            elif existing.model_dump() != node.model_dump():
                ops.append(
                    TransactionOperation(
                        op=TransactionOp.UPSERT_NODE, node_id=nid, node_data=node.model_dump()
                    )
                )
        for eid, edge in br_edges.items():
            if dirty_edges is not None and eid not in dirty_edges:
                continue
            if eid not in main_edges or eid in changed_edges:
                ops.append(
                    TransactionOperation(
                        op=TransactionOp.ADD_EDGE, edge_id=eid, edge_data=edge.model_dump()
                    )
                )
        return ops

    def merge_branch(
        self,
        branch_name: str,
        transaction_manager: Any | None = None,
        author: str = "merge",
    ) -> bool:
        """Merge a forked branch into this workspace as a single atomic transaction.

        The diff is committed through ``transaction_manager`` (default: the commit
        authority attached to this workspace) so that it passes the constitutional
        verification pipeline.

        Returns True only if the merge transaction was committed; the branch is
        dropped only on success.
        """
        if branch_name not in self._branches:
            return False
        ops = self.diff_branch(branch_name)
        transaction_manager = transaction_manager or self.transaction_manager

        if ops:
            tx = HCIRTransaction(
                author=author,
                operations=ops,
                metadata={"merge_branch": branch_name},
            )
            result = transaction_manager.commit(tx)
            if not result.is_committed:
                logger.warning("merge_branch('%s') rejected: %s", branch_name, result.annotations)
                return False
        self._branches.pop(branch_name, None)
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
