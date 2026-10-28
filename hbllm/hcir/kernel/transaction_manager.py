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
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Protocol, runtime_checkable

from hbllm.hcir.graph import (
    NODE_TYPE_REGISTRY,
    HCIREdge,
    HCIRNode,
    HCIRNodeType,
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
from hbllm.hcir.validation import GraphValidator
from hbllm.hcir.workspace import HCIRWorkspaceState

logger = logging.getLogger(__name__)

# Canonical alias for committed transaction results
TransactionResult = HCIRTransaction


# ═══════════════════════════════════════════════════════════════════════════
# Transaction Builder
# ═══════════════════════════════════════════════════════════════════════════


class TransactionBuilder:
    """Asynchronous builder pattern for constructing and committing transactions."""

    def __init__(self, manager: TransactionManager, author: str = "system") -> None:
        self._manager = manager
        self.author = author
        self.operations: list[TransactionOperation] = []
        self.annotations: list[TransactionAnnotation] = []
        self.metadata: dict[str, Any] = {}
        self._committed_tx: HCIRTransaction | None = None

    def add_operation(
        self,
        op: TransactionOperation | None = None,
        *,
        operation: TransactionOp | str | None = None,
        node: Any = None,
        node_id: str | None = None,
        node_data: dict[str, Any] | None = None,
        edge: Any = None,
        edge_id: str | None = None,
        edge_data: dict[str, Any] | None = None,
        changes: dict[str, Any] | None = None,
    ) -> TransactionBuilder:
        """Add an operation to the pending transaction."""
        if op is not None:
            self.operations.append(op)
            return self

        op_enum = TransactionOp(operation) if operation else TransactionOp.ADD_NODE
        nd = node_data or (node.model_dump() if hasattr(node, "model_dump") else None)
        nid = node_id or (getattr(node, "id", None) if node else None)
        ed = edge_data or (edge.model_dump() if hasattr(edge, "model_dump") else None)
        eid = edge_id or (getattr(edge, "id", None) if edge else None)

        self.operations.append(
            TransactionOperation(
                op=op_enum,
                node_id=nid,
                node_data=nd,
                edge_id=eid,
                edge_data=ed,
                changes=changes,
            )
        )
        return self

    async def commit(self, context: Any = None) -> HCIRTransaction:
        """Commit the built transaction."""
        tx = HCIRTransaction(
            author=self.author,
            operations=list(self.operations),
            annotations=list(self.annotations),
            metadata=dict(self.metadata),
        )
        self._committed_tx = self._manager.commit(tx, context=context)
        return self._committed_tx

    @property
    def transaction(self) -> HCIRTransaction | None:
        return self._committed_tx


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
        self._undo_logs: dict[str, list[dict[str, Any]]] = {}

    @property
    def committed_count(self) -> int:
        return len(self._committed_log)

    @property
    def committed_log(self) -> list[HCIRTransaction]:
        return list(self._committed_log)

    @property
    def rejected_log(self) -> list[HCIRTransaction]:
        return list(self._rejected_log)

    @property
    def rejected_count(self) -> int:
        return len(self._rejected_log)

    def add_verification_stage(self, stage: IVerificationStage) -> None:
        """Add a verification stage to the pipeline."""
        self._verification_stages.append(stage)

    @asynccontextmanager
    async def begin_transaction(
        self, author: str = "system", auto_commit: bool = False
    ) -> AsyncIterator[TransactionBuilder]:
        """Asynchronous context manager for building and committing transactions."""
        builder = TransactionBuilder(self, author=author)
        try:
            yield builder
        finally:
            if auto_commit and builder.transaction is None and builder.operations:
                await builder.commit()

    def commit(
        self,
        transaction: HCIRTransaction,
        context: Any = None,
    ) -> HCIRTransaction:
        """Process a transaction through the verification pipeline and commit.

        Returns the transaction with updated status.
        """
        # Invariant #1: don't re-process committed/rejected transactions
        if transaction.status in (TransactionStatus.COMMITTED, TransactionStatus.REJECTED):
            logger.warning("Transaction %s already %s", transaction.id, transaction.status)
            return transaction

        if context is not None:
            if hasattr(context, "authority_chain"):
                auth_chain = getattr(context, "authority_chain")
                transaction.annotations.append(
                    TransactionAnnotation(
                        author="CognitiveAuthorityChain",
                        assertion=f"Authority: {auth_chain}",
                        severity="info",
                    )
                )
            if hasattr(context, "execution_context"):
                exec_ctx = getattr(context, "execution_context")
                if hasattr(exec_ctx, "tenant_scope"):
                    transaction.annotations.append(
                        TransactionAnnotation(
                            author="IdentityBridge",
                            assertion=f"Enforced scope: {exec_ctx.tenant_scope}",
                            severity="info",
                        )
                    )
            else:
                transaction.annotations.append(
                    TransactionAnnotation(
                        author="IdentityBridge",
                        assertion=f"Enforced context: {getattr(context, 'metadata', context)}",
                        severity="info",
                    )
                )

        # Run verification pipeline
        for stage in self._verification_stages:
            if not stage.verify(transaction, self._workspace):
                transaction.status = TransactionStatus.REJECTED
                self._rejected_log.append(transaction)
                self._workspace.snapshot_manager.record_kernel_event(
                    EventType.TRANSACTION_REJECTED,
                    {"tx_id": transaction.id, "author": transaction.author},
                )
                logger.info("Transaction %s rejected by %s", transaction.id, type(stage).__name__)
                return transaction

        transaction.status = TransactionStatus.VALIDATED

        # Apply operations to the workspace
        try:
            self._apply_operations(transaction)
        except Exception as exc:
            transaction.status = TransactionStatus.REJECTED
            transaction.annotations.append(
                TransactionAnnotation(
                    author="TransactionManager",
                    assertion=f"Apply failed: {exc}",
                    severity="error",
                )
            )
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

        logger.debug(
            "Transaction %s committed (%d ops)", transaction.id, transaction.operation_count
        )
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

    def compensate(
        self,
        original_tx_id: str,
        reason: str = "downstream_failure",
    ) -> HCIRTransaction | None:
        """Generate and commit a compensating transaction that reverses
        the effects of the original.

        Used when downstream execution fails (hardware, robotics,
        distributed execution) and the graph state must be rolled back
        to maintain consistency.

        Args:
            original_tx_id: ID of the committed transaction to reverse.
            reason: Human-readable reason for compensation.

        Returns:
            The committed compensating transaction, or ``None`` if the
            original transaction was not found.
        """
        # Find the original committed transaction
        original = self._find_committed(original_tx_id)
        if original is None:
            logger.warning("compensate: transaction %s not found", original_tx_id)
            return None

        # Generate inverse operations
        reverse_ops = self._invert_operations(original.operations)
        if not reverse_ops:
            logger.info("compensate: no invertible operations in %s", original_tx_id)
            return None

        from hbllm.hcir.types import Provenance

        compensating_tx = HCIRTransaction(
            author="transaction_manager",
            operations=reverse_ops,
            provenance=Provenance(
                created_by="transaction_manager",
                reason=f"Compensating tx {original_tx_id}: {reason}",
            ),
            annotations=[
                TransactionAnnotation(
                    author="TransactionManager",
                    assertion=f"Compensating transaction for {original_tx_id}: {reason}",
                    severity="warning",
                )
            ],
        )
        result = self.commit(compensating_tx)

        if result.is_committed:
            # Record the compensation event
            self._workspace.snapshot_manager.record_kernel_event(
                EventType.TRANSACTION_COMPENSATED,
                {
                    "original_tx_id": original_tx_id,
                    "compensating_tx_id": result.id,
                    "reason": reason,
                },
            )
            logger.info(
                "Compensated transaction %s → %s: %s",
                original_tx_id,
                result.id,
                reason,
            )

        return result

    def rollback(
        self,
        original_tx_id: str,
        reason: str = "acid_rollback",
    ) -> HCIRTransaction | None:
        """Full ACID rollback restoring all pre-mutation state (before-images).

        Unlike forward compensation which adheres to high-level domain semantics,
        rollback uses exact captured undo logs to restore modified and deleted
        nodes and edges.
        """
        original = self._find_committed(original_tx_id)
        if original is None:
            logger.warning("rollback: transaction %s not found", original_tx_id)
            return None

        undo_records = self._undo_logs.get(original_tx_id, [])
        if not undo_records:
            # Fall back to compensate if no undo log captured
            return self.compensate(original_tx_id, reason=reason)

        inverse: list[TransactionOperation] = []
        for rec in reversed(undo_records):
            op = rec["op"]
            prev_node = rec.get("previous_node")
            prev_edge = rec.get("previous_edge")
            nid = rec.get("node_id")
            eid = rec.get("edge_id")

            if op in (TransactionOp.ADD_NODE, TransactionOp.UPSERT_NODE):
                if prev_node:
                    inverse.append(
                        TransactionOperation(
                            op=TransactionOp.UPSERT_NODE,
                            node_id=nid,
                            node_data=prev_node,
                        )
                    )
                elif nid:
                    inverse.append(
                        TransactionOperation(
                            op=TransactionOp.REMOVE_NODE,
                            node_id=nid,
                        )
                    )
            elif op == TransactionOp.MODIFY_NODE:
                if prev_node:
                    inverse.append(
                        TransactionOperation(
                            op=TransactionOp.UPSERT_NODE,
                            node_id=nid,
                            node_data=prev_node,
                        )
                    )
            elif op == TransactionOp.REMOVE_NODE:
                if prev_node:
                    inverse.append(
                        TransactionOperation(
                            op=TransactionOp.ADD_NODE,
                            node_id=nid,
                            node_data=prev_node,
                        )
                    )
            elif op == TransactionOp.ADD_EDGE:
                if prev_edge:
                    inverse.append(
                        TransactionOperation(
                            op=TransactionOp.ADD_EDGE,
                            edge_id=eid,
                            edge_data=prev_edge,
                        )
                    )
                elif eid:
                    inverse.append(
                        TransactionOperation(
                            op=TransactionOp.REMOVE_EDGE,
                            edge_id=eid,
                        )
                    )
            elif op == TransactionOp.REMOVE_EDGE:
                if prev_edge:
                    inverse.append(
                        TransactionOperation(
                            op=TransactionOp.ADD_EDGE,
                            edge_id=eid,
                            edge_data=prev_edge,
                        )
                    )

        if not inverse:
            return None

        from hbllm.hcir.types import Provenance

        rollback_tx = HCIRTransaction(
            author="transaction_manager",
            operations=inverse,
            provenance=Provenance(
                created_by="transaction_manager",
                reason=f"Rollback tx {original_tx_id}: {reason}",
            ),
            annotations=[
                TransactionAnnotation(
                    author="TransactionManager",
                    assertion=f"Rollback transaction for {original_tx_id}: {reason}",
                    severity="warning",
                )
            ],
        )
        return self.commit(rollback_tx)

    def _find_committed(self, tx_id: str) -> HCIRTransaction | None:
        """Find a committed transaction by ID."""
        for tx in self._committed_log:
            if tx.id == tx_id:
                return tx
        return None

    def _invert_operations(
        self,
        operations: list[TransactionOperation],
    ) -> list[TransactionOperation]:
        """Generate inverse operations for a list of transaction operations.

        Inversion rules:
            ADD_NODE / UPSERT_NODE → REMOVE_NODE
            REMOVE_NODE → (no inverse — data is lost in forward compensation)
            ADD_EDGE → REMOVE_EDGE
            REMOVE_EDGE → (no inverse — data is lost in forward compensation)
            MODIFY_NODE → (no inverse — original values unknown without rollback log)
        """
        inverse: list[TransactionOperation] = []
        for op in reversed(operations):
            if op.op in (TransactionOp.ADD_NODE, TransactionOp.UPSERT_NODE):
                node_id = (op.node_data or {}).get("id") or op.node_id
                if node_id:
                    inverse.append(
                        TransactionOperation(
                            op=TransactionOp.REMOVE_NODE,
                            node_id=node_id,
                        )
                    )
            elif op.op == TransactionOp.ADD_EDGE:
                edge_id = (op.edge_data or {}).get("id") or op.edge_id
                if edge_id:
                    inverse.append(
                        TransactionOperation(
                            op=TransactionOp.REMOVE_EDGE,
                            edge_id=edge_id,
                        )
                    )
        return inverse

    def _apply_operations(self, transaction: HCIRTransaction) -> None:
        """Apply transaction operations to the workspace graph with undo logging."""
        undo_entries: list[dict[str, Any]] = []

        for op in transaction.operations:
            if op.op in (TransactionOp.ADD_NODE, TransactionOp.UPSERT_NODE):
                raw_data = dict(op.node_data or {})
                if "id" not in raw_data and op.node_id:
                    raw_data["id"] = op.node_id
                target_id = raw_data.get("id") or op.node_id
                existing = self._workspace.get_node(target_id) if target_id else None
                undo_entries.append(
                    {
                        "op": op.op,
                        "node_id": target_id,
                        "previous_node": existing.model_dump() if existing else None,
                    }
                )
                node = self._deserialize_node(raw_data)
                if op.op == TransactionOp.ADD_NODE:
                    self._workspace.add_node(node, author=transaction.author)
                else:
                    self._workspace.upsert_node(node, author=transaction.author)

            elif op.op == TransactionOp.MODIFY_NODE:
                changes = op.changes or op.node_data
                if op.node_id and changes:
                    existing = self._workspace.get_node(op.node_id)
                    if existing is None:
                        raise ValueError(f"Cannot modify non-existent node: {op.node_id}")
                    undo_entries.append(
                        {
                            "op": op.op,
                            "node_id": op.node_id,
                            "previous_node": existing.model_dump(),
                        }
                    )
                    # Create updated copy with changes applied
                    updated_data = existing.model_dump()
                    updated_data.update(changes)
                    updated = type(existing).model_validate(updated_data)
                    self._workspace.upsert_node(updated, author=transaction.author)

            elif op.op == TransactionOp.REMOVE_NODE:
                if op.node_id:
                    existing = self._workspace.get_node(op.node_id)
                    undo_entries.append(
                        {
                            "op": op.op,
                            "node_id": op.node_id,
                            "previous_node": existing.model_dump() if existing else None,
                        }
                    )
                    self._workspace.remove_node(op.node_id, author=transaction.author)

            elif op.op == TransactionOp.ADD_EDGE:
                raw_edge = dict(op.edge_data or {})
                edge_id = raw_edge.get("id") or op.edge_id
                existing_edge = (
                    self._workspace.graph.edges.get(edge_id)
                    if edge_id
                    and hasattr(self._workspace.graph, "edges")
                    and isinstance(self._workspace.graph.edges, dict)
                    else None
                )
                undo_entries.append(
                    {
                        "op": op.op,
                        "edge_id": edge_id,
                        "previous_edge": existing_edge.model_dump() if existing_edge else None,
                    }
                )
                edge = HCIREdge.model_validate(op.edge_data or {})
                self._workspace.add_edge(edge, author=transaction.author)

            elif op.op == TransactionOp.REMOVE_EDGE:
                if op.edge_id:
                    existing_edge = (
                        self._workspace.graph.edges.get(op.edge_id)
                        if hasattr(self._workspace.graph, "edges")
                        and isinstance(self._workspace.graph.edges, dict)
                        else None
                    )
                    undo_entries.append(
                        {
                            "op": op.op,
                            "edge_id": op.edge_id,
                            "previous_edge": existing_edge.model_dump() if existing_edge else None,
                        }
                    )
                    self._workspace.remove_edge(op.edge_id, author=transaction.author)

        self._undo_logs[transaction.id] = undo_entries

    @staticmethod
    def _deserialize_node(data: dict[str, Any]) -> HCIRNode:
        """Deserialize a node dict into the correct typed subclass."""
        node_type_str = data.get("node_type", "")
        if not node_type_str:
            from hbllm.hcir.graph import ConceptNode

            nid = data.get("id") or "node_untyped"
            known_fields = {"id", "category", "lifecycle", "tags", "confidence", "properties"}
            props = {k: v for k, v in data.items() if k not in known_fields}
            if "properties" in data and isinstance(data["properties"], dict):
                props.update(data["properties"])
            return ConceptNode(
                id=nid,
                label=str(data.get("role") or data.get("variable") or nid),
                definition=str(props),
                domain="cognitive_state",
                properties=props,
            )

        from hbllm.hcir.kernel.verification import NODE_TYPE_ALIASES

        if node_type_str in NODE_TYPE_ALIASES:
            node_type = NODE_TYPE_ALIASES[node_type_str]
        else:
            try:
                node_type = HCIRNodeType(node_type_str)
            except ValueError:
                raise ValueError(f"Unknown node type: {node_type_str}")

        cls = NODE_TYPE_REGISTRY.get(node_type)
        if cls is None:
            raise ValueError(f"No registered class for node type: {node_type}")

        return cls.model_validate(data)
