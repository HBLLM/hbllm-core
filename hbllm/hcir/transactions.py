"""
HCIR Transactions — atomic graph mutation proposals.

No node writes directly to the workspace.  Instead, nodes submit
``HCIRTransaction`` proposals that pass through a staged verification
pipeline before being committed by the ``TransactionManager``.

Transaction lifecycle::

    Proposed → Validated → Committed
                        ↘ Rejected

Kernel Invariant #1: Transactions are immutable after proposal.
Kernel Invariant #2: Only the TransactionManager commits state.
"""

from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hbllm.network.clocks import VectorClock

from pydantic import BaseModel, Field

from hbllm.hcir.types import Provenance, Timestamp

# ═══════════════════════════════════════════════════════════════════════════
# Transaction Operations
# ═══════════════════════════════════════════════════════════════════════════


class TransactionOp(StrEnum):
    """Allowed mutation operations within a transaction."""

    ADD_NODE = "add_node"
    MODIFY_NODE = "modify_node"
    REMOVE_NODE = "remove_node"
    UPSERT_NODE = "upsert_node"
    ADD_EDGE = "add_edge"
    REMOVE_EDGE = "remove_edge"


class TransactionOperation(BaseModel):
    """A single graph mutation operation within a transaction.

    For ``add_node`` / ``upsert_node``: ``node_data`` contains the
    full serialized node.
    For ``modify_node``: ``node_id`` + ``changes`` dict.
    For ``remove_node``: only ``node_id``.
    For ``add_edge``: ``edge_data`` contains the full serialized edge.
    For ``remove_edge``: only ``edge_id``.
    """

    op: TransactionOp
    node_id: str | None = None
    node_data: dict[str, Any] | None = None
    edge_id: str | None = None
    edge_data: dict[str, Any] | None = None
    changes: dict[str, Any] | None = None


# ═══════════════════════════════════════════════════════════════════════════
# Transaction Status & Annotations
# ═══════════════════════════════════════════════════════════════════════════


class TransactionStatus(StrEnum):
    """Lifecycle states of a transaction."""

    PROPOSED = "proposed"
    VALIDATED = "validated"
    COMMITTED = "committed"
    REJECTED = "rejected"


class TransactionAnnotation(BaseModel):
    """A non-structural metadata addition to a transaction.

    Added by verification pipeline stages (e.g., Critic warnings).
    """

    author: str
    assertion: str
    severity: str = "info"  # "info", "warning", "error"
    timestamp: Timestamp = Field(default_factory=time.time)


# ═══════════════════════════════════════════════════════════════════════════
# HCIRTransaction
# ═══════════════════════════════════════════════════════════════════════════


class HCIRTransaction(BaseModel):
    """An atomic, immutable graph mutation proposal.

    After creation (status=proposed), the transaction is frozen.
    Only the TransactionManager may change the status field
    through the verification pipeline.

    Kernel Invariant #1: immutable after proposal.
    Kernel Invariant #3: every committed transaction has provenance.
    """

    id: str = Field(default_factory=lambda: f"tx_{uuid.uuid4().hex[:12]}")
    author: str  # Node ID that proposed this transaction
    parent_snapshot_hash: str = ""
    timestamp: Timestamp = Field(default_factory=time.time)
    operations: list[TransactionOperation] = Field(default_factory=list)
    status: TransactionStatus = TransactionStatus.PROPOSED
    approvals: list[str] = Field(default_factory=list)  # Node IDs that approved
    annotations: list[TransactionAnnotation] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)

    @property
    def is_committed(self) -> bool:
        return self.status == TransactionStatus.COMMITTED

    @property
    def is_rejected(self) -> bool:
        return self.status == TransactionStatus.REJECTED

    @property
    def operation_count(self) -> int:
        return len(self.operations)


# ═══════════════════════════════════════════════════════════════════════════
# HCIRDelta — lightweight incremental update
# ═══════════════════════════════════════════════════════════════════════════


class HCIRDelta(BaseModel):
    """A lightweight incremental graph update with CRDT merge semantics.

    Used as the return type from cognitive node execution.
    Simpler than a full transaction — no lifecycle or approvals.
    The kernel wraps deltas into transactions for the commit pipeline.

    CRDT semantics:
        Each delta carries a ``VectorClock`` snapshot from its origin device.
        When two deltas are concurrent (neither causally before the other),
        ``merge()`` combines them deterministically:

        - **Add/Remove operations**: union (both applied).
        - **Modify same node**: Last-Writer-Wins using VectorClock comparison,
          with device_id as a tiebreaker for truly simultaneous writes.
        - **Annotations**: union of both annotation lists.

        This guarantees convergence: any two peers applying the same set of
        deltas in any order reach the same final state.
    """

    add_nodes: list[dict[str, Any]] = Field(default_factory=list)
    modify_nodes: list[dict[str, Any]] = Field(default_factory=list)
    remove_node_ids: list[str] = Field(default_factory=list)
    add_edges: list[dict[str, Any]] = Field(default_factory=list)
    remove_edge_ids: list[str] = Field(default_factory=list)
    annotations: list[TransactionAnnotation] = Field(default_factory=list)

    # ── CRDT fields ──────────────────────────────────────────────────────
    origin_device: str = "local"
    vector_clock: dict[str, int] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)

    def get_vector_clock(self) -> VectorClock:
        """Reconstruct a ``VectorClock`` instance from the serialized counters."""
        from hbllm.network.clocks import VectorClock

        return VectorClock(self.origin_device, dict(self.vector_clock))

    def set_vector_clock(self, clock: VectorClock) -> None:
        """Snapshot a ``VectorClock`` into this delta's serialized form."""
        self.vector_clock = clock.to_dict()
        self.origin_device = clock.node_id

    def causal_relation(self, other: HCIRDelta) -> str:
        """Compare causal ordering with another delta.

        Returns:
            ``"before"``, ``"after"``, ``"concurrent"``, or ``"equal"``.
        """
        return self.get_vector_clock().compare(other.get_vector_clock())

    def merge(self, other: HCIRDelta) -> HCIRDelta:
        """CRDT merge: combine two deltas into one that converges deterministically.

        Merge strategy:
          - ``add_nodes`` / ``add_edges``: union (deduplicated by node/edge ID).
          - ``remove_node_ids`` / ``remove_edge_ids``: union.
          - ``modify_nodes``: if both modify the same node, Last-Writer-Wins
            by VectorClock comparison (device_id tiebreaker).
          - ``annotations``: union.
          - ``vector_clock``: element-wise max (standard VC merge).

        Returns:
            A new ``HCIRDelta`` containing the merged result.
        """

        relation = self.causal_relation(other)

        # ── Trivial cases: one strictly before the other ─────────────
        if relation == "before":
            # other is strictly newer — its state supersedes ours
            return other.model_copy(deep=True)
        if relation == "after" or relation == "equal":
            # we are strictly newer or identical
            return self.model_copy(deep=True)

        # ── Concurrent: must merge ───────────────────────────────────

        # 1. Union of add_nodes (deduplicate by ID)
        seen_node_ids: set[str] = set()
        merged_add_nodes: list[dict[str, Any]] = []
        for node in self.add_nodes + other.add_nodes:
            nid = node.get("id", "")
            if nid not in seen_node_ids:
                seen_node_ids.add(nid)
                merged_add_nodes.append(node)

        # 2. Union of add_edges (deduplicate by ID)
        seen_edge_ids: set[str] = set()
        merged_add_edges: list[dict[str, Any]] = []
        for edge in self.add_edges + other.add_edges:
            eid = edge.get("id", "")
            if eid not in seen_edge_ids:
                seen_edge_ids.add(eid)
                merged_add_edges.append(edge)

        # 3. Union of removals
        merged_remove_nodes = sorted(set(self.remove_node_ids) | set(other.remove_node_ids))
        merged_remove_edges = sorted(set(self.remove_edge_ids) | set(other.remove_edge_ids))

        # 4. Modify nodes — LWW for conflicts, union for non-overlapping
        self_mods: dict[str, dict[str, Any]] = {}
        for mod in self.modify_nodes:
            mid = mod.get("id") or mod.get("node_id", "")
            self_mods[mid] = mod

        other_mods: dict[str, dict[str, Any]] = {}
        for mod in other.modify_nodes:
            mid = mod.get("id") or mod.get("node_id", "")
            other_mods[mid] = mod

        merged_modify_nodes: list[dict[str, Any]] = []
        all_mod_ids = set(self_mods.keys()) | set(other_mods.keys())
        for mid in sorted(all_mod_ids):
            if mid in self_mods and mid not in other_mods:
                merged_modify_nodes.append(self_mods[mid])
            elif mid not in self_mods and mid in other_mods:
                merged_modify_nodes.append(other_mods[mid])
            else:
                # Conflict — LWW by timestamp, device_id as tiebreaker
                if self.timestamp > other.timestamp:
                    merged_modify_nodes.append(self_mods[mid])
                elif other.timestamp > self.timestamp:
                    merged_modify_nodes.append(other_mods[mid])
                else:
                    # Exact same timestamp — deterministic tiebreak by device_id
                    winner = (
                        self_mods[mid]
                        if self.origin_device >= other.origin_device
                        else other_mods[mid]
                    )
                    merged_modify_nodes.append(winner)

        # 5. Merge annotations (union)
        merged_annotations = list(self.annotations) + list(other.annotations)

        # 6. Merge vector clocks (element-wise max)
        merged_clock = self.get_vector_clock()
        merged_clock.update(other.get_vector_clock())

        return HCIRDelta(
            add_nodes=merged_add_nodes,
            modify_nodes=merged_modify_nodes,
            remove_node_ids=merged_remove_nodes,
            add_edges=merged_add_edges,
            remove_edge_ids=merged_remove_edges,
            annotations=merged_annotations,
            origin_device=merged_clock.node_id,
            vector_clock=merged_clock.to_dict(),
            timestamp=max(self.timestamp, other.timestamp),
        )

    def to_operations(self) -> list[TransactionOperation]:
        """Convert this delta into a list of transaction operations."""
        ops: list[TransactionOperation] = []
        for node_data in self.add_nodes:
            ops.append(
                TransactionOperation(
                    op=TransactionOp.ADD_NODE,
                    node_id=node_data.get("id"),
                    node_data=node_data,
                )
            )
        for mod in self.modify_nodes:
            ops.append(
                TransactionOperation(
                    op=TransactionOp.MODIFY_NODE,
                    node_id=mod.get("id") or mod.get("node_id"),
                    changes=mod.get("changes", mod),
                )
            )
        for node_id in self.remove_node_ids:
            ops.append(
                TransactionOperation(
                    op=TransactionOp.REMOVE_NODE,
                    node_id=node_id,
                )
            )
        for edge_data in self.add_edges:
            ops.append(
                TransactionOperation(
                    op=TransactionOp.ADD_EDGE,
                    edge_id=edge_data.get("id"),
                    edge_data=edge_data,
                )
            )
        for edge_id in self.remove_edge_ids:
            ops.append(
                TransactionOperation(
                    op=TransactionOp.REMOVE_EDGE,
                    edge_id=edge_id,
                )
            )
        return ops
