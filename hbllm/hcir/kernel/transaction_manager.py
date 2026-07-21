"""
Transaction Manager — the single commit authority for HCIR.

Enforces Kernel Invariants:
    #1: Transactions are immutable after proposal.
    #2: Only the TransactionManager commits state changes.
    #3: Every committed transaction has provenance.

Transaction lifecycle::

    Proposal → [Policy Check] → [Validation] → [Resource Check] → Commit
                                                                 ↘ Reject

The manager also maintains a committed transaction log for
audit, reflection, and skill induction.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol, runtime_checkable

from hbllm.hcir.graph import (
    HCIREdge,
    HCIRNode,
    HCIRNodeType,
    NODE_TYPE_REGISTRY,
)
from hbllm.hcir.stores import EventType
from hbllm.hcir.transactions import (
    HCIRDelta,
    HCIRTransaction,
    TransactionAnnotation,
    TransactionOp,
    TransactionOperation,
    TransactionStatus,
)
from hbllm.hcir.validation import GraphValidator, ValidationSeverity
from hbllm.hcir.workspace import HCIRWorkspaceState

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Verification Stage Interface
# ═══════════════════════════════════════════════════════════════════════════


@runtime_checkable
class IVerificationStage(Protocol):
    """A single stage in the transaction verification pipeline.

    Returns True if the transaction passes this stage.
    May append annotations to the transaction.
    """

    def verify(
        self,
        transaction: HCIRTransaction,
        workspace: HCIRWorkspaceState,
    ) -> bool: ...


# ═══════════════════════════════════════════════════════════════════════════
# Transaction Manager
# ═══════════════════════════════════════════════════════════════════════════


class TransactionManager:
    """The single commit authority for HCIR state mutations.

    No cognitive node may modify the workspace graph directly.
    All changes must be proposed as transactions and committed
    through this manager.

    Usage::

        manager = TransactionManager(workspace)
        tx = HCIRTransaction(author="PlannerNode", operations=[...])
        result = manager.commit(tx)
        if result.is_committed:
            print("Success!")
    """

    def __init__(
        self,
        workspace: HCIRWorkspaceState,
        verification_stages: list[IVerificationStage] | None = None,
        validator: GraphValidator | None = None,
    ) -> None:
        self._workspace = workspace
        self._verification_stages = verification_stages or []
        self._validator = validator or GraphValidator()
        self._committed_log: list[HCIRTransaction] = []
        self._rejected_log: list[HCIRTransaction] = []

    @property
    def committed_count(self) -> int:
        return len(self._committed_log)

    @property
    def rejected_count(self) -> int:
        return len(self._rejected_log)

    def add_verification_stage(self, stage: IVerificationStage) -> None:
        """Add a verification stage to the pipeline."""
        self._verification_stages.append(stage)

    def commit(self, transaction: HCIRTransaction) -> HCIRTransaction:
        """Process a transaction through the verification pipeline and commit.

        Returns the transaction with updated status.
        """
        # Invariant #1: don't re-process committed/rejected transactions
        if transaction.status in (TransactionStatus.COMMITTED, TransactionStatus.REJECTED):
            logger.warning("Transaction %s already %s", transaction.id, transaction.status)
            return transaction

        # Run verification pipeline
        for stage in self._verification_stages:
            if not stage.verify(transaction, self._workspace):
                transaction.status = TransactionStatus.REJECTED
                self._rejected_log.append(transaction)
                self._workspace.snapshot_manager.record_kernel_event(
                    EventType.TRANSACTION_REJECTED,
                    {"tx_id": transaction.id, "author": transaction.author},
                )
                logger.info("Transaction %s rejected by %s",
                            transaction.id, type(stage).__name__)
                return transaction

        transaction.status = TransactionStatus.VALIDATED

        # Apply operations to the workspace
        try:
            self._apply_operations(transaction)
        except Exception as exc:
            transaction.status = TransactionStatus.REJECTED
            transaction.annotations.append(TransactionAnnotation(
                author="TransactionManager",
                assertion=f"Apply failed: {exc}",
                severity="error",
            ))
            self._rejected_log.append(transaction)
            logger.error("Transaction %s apply failed: %s", transaction.id, exc)
            return transaction

        # Commit
        transaction.status = TransactionStatus.COMMITTED
        self._committed_log.append(transaction)

        # Create a new snapshot
        self._workspace.create_snapshot()

        # Emit kernel event
        self._workspace.snapshot_manager.record_kernel_event(
            EventType.TRANSACTION_COMMITTED,
            {
                "tx_id": transaction.id,
                "author": transaction.author,
                "op_count": transaction.operation_count,
            },
        )

        logger.debug("Transaction %s committed (%d ops)",
                      transaction.id, transaction.operation_count)
        return transaction

    def commit_delta(self, delta: HCIRDelta, author: str) -> HCIRTransaction:
        """Wrap an HCIRDelta in a transaction and commit it.

        Convenience method for cognitive nodes that return deltas
        rather than full transactions.
        """
        tx = HCIRTransaction(
            author=author,
            operations=delta.to_operations(),
            annotations=list(delta.annotations),
        )
        return self.commit(tx)

    def get_committed_transactions(
        self, limit: int = 100, author: str | None = None
    ) -> list[HCIRTransaction]:
        """Retrieve committed transaction history for skill induction."""
        txs = self._committed_log
        if author:
            txs = [t for t in txs if t.author == author]
        return txs[-limit:]

    # ── Internal ─────────────────────────────────────────────────────

    def _apply_operations(self, transaction: HCIRTransaction) -> None:
        """Apply transaction operations to the workspace graph."""
        for op in transaction.operations:
            if op.op == TransactionOp.ADD_NODE:
                node = self._deserialize_node(op.node_data or {})
                self._workspace.add_node(node, author=transaction.author)

            elif op.op == TransactionOp.UPSERT_NODE:
                node = self._deserialize_node(op.node_data or {})
                self._workspace.upsert_node(node, author=transaction.author)

            elif op.op == TransactionOp.MODIFY_NODE:
                if op.node_id and op.changes:
                    existing = self._workspace.get_node(op.node_id)
                    if existing is None:
                        raise ValueError(f"Cannot modify non-existent node: {op.node_id}")
                    # Create updated copy with changes applied
                    updated_data = existing.model_dump()
                    updated_data.update(op.changes)
                    updated = type(existing).model_validate(updated_data)
                    self._workspace.upsert_node(updated, author=transaction.author)

            elif op.op == TransactionOp.REMOVE_NODE:
                if op.node_id:
                    self._workspace.remove_node(op.node_id, author=transaction.author)

            elif op.op == TransactionOp.ADD_EDGE:
                edge = HCIREdge.model_validate(op.edge_data or {})
                self._workspace.add_edge(edge, author=transaction.author)

            elif op.op == TransactionOp.REMOVE_EDGE:
                if op.edge_id:
                    self._workspace.remove_edge(op.edge_id, author=transaction.author)

    @staticmethod
    def _deserialize_node(data: dict[str, Any]) -> HCIRNode:
        """Deserialize a node dict into the correct typed subclass."""
        node_type_str = data.get("node_type", "")
        try:
            node_type = HCIRNodeType(node_type_str)
        except ValueError:
            raise ValueError(f"Unknown node type: {node_type_str}")

        cls = NODE_TYPE_REGISTRY.get(node_type)
        if cls is None:
            raise ValueError(f"No registered class for node type: {node_type}")

        return cls.model_validate(data)
