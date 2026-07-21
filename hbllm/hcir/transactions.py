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
from typing import Any

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
    """A lightweight incremental graph update.

    Used as the return type from cognitive node execution.
    Simpler than a full transaction — no lifecycle or approvals.
    The kernel wraps deltas into transactions for the commit pipeline.
    """

    add_nodes: list[dict[str, Any]] = Field(default_factory=list)
    modify_nodes: list[dict[str, Any]] = Field(default_factory=list)
    remove_node_ids: list[str] = Field(default_factory=list)
    add_edges: list[dict[str, Any]] = Field(default_factory=list)
    remove_edge_ids: list[str] = Field(default_factory=list)
    annotations: list[TransactionAnnotation] = Field(default_factory=list)

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
