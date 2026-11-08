"""
Transaction Manager — the single commit authority for HCIR.

Enforces Kernel Invariants:
    #1: Transactions are immutable after proposal.
    #2: Only the TransactionManager commits state changes.
    #3: Every committed transaction has provenance.
    #4: Commits are atomic — a transaction either applies in full or leaves
        the workspace untouched (in-flight undo on partial failure).

Transaction lifecycle::

    Proposal → [Policy Check] → [Validation] → [Resource Check] → Commit
                                                                 ↘ Reject

The manager also maintains a bounded committed transaction log for
audit, reflection, skill induction, and version rollback.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from hbllm.hcir.graph import (
    NODE_TYPE_REGISTRY,
    HCIREdge,
    HCIRNode,
    HCIRNodeType,
)
from hbllm.hcir.kernel.event_bus import KernelEvent, KernelEventType
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

if TYPE_CHECKING:
    from hbllm.hcir.kernel.event_bus import KernelEventBus

logger = logging.getLogger(__name__)

# Canonical alias for committed transaction results
TransactionResult = HCIRTransaction

KERNEL_AUTHOR = "transaction_manager"


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

    def add_operations(self, ops: list[TransactionOperation]) -> TransactionBuilder:
        """Add a batch of pre-built operations."""
        self.operations.extend(ops)
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
        event_bus: KernelEventBus | None = None,
        max_log_size: int = 10_000,
    ) -> None:
        self._workspace = workspace
        self._verification_stages = verification_stages or []
        if not self._verification_stages:
            logger.warning("TransactionManager created without verification pipeline")
            self.unverified = True
        else:
            self.unverified = False
        self._validator = validator or GraphValidator()
        self._event_bus = event_bus
        self._max_log_size = max(1, max_log_size)
        self._committed_log: deque[HCIRTransaction] = deque(maxlen=self._max_log_size)
        self._rejected_log: deque[HCIRTransaction] = deque(maxlen=self._max_log_size)
        self._undo_logs: dict[str, list[dict[str, Any]]] = {}
        self._tx_versions: dict[str, int] = {}
        self._lock = threading.RLock()
        attach = getattr(workspace, "attach_transaction_manager", None)
        if callable(attach):
            attach(self)

    @property
    def workspace(self) -> HCIRWorkspaceState:
        return self._workspace

    @property
    def event_bus(self) -> KernelEventBus | None:
        return self._event_bus

    @event_bus.setter
    def event_bus(self, bus: KernelEventBus | None) -> None:
        self._event_bus = bus

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

    # ── Commit path ──────────────────────────────────────────────────

    def commit(
        self,
        transaction: HCIRTransaction,
        context: Any = None,
    ) -> HCIRTransaction:
        """Process a transaction through the verification pipeline and commit.

        Returns the transaction with updated status.
        """
        with self._lock:
            return self._commit(transaction, context, self._verification_stages)

    def _commit(
        self,
        transaction: HCIRTransaction,
        context: Any,
        stages: list[IVerificationStage],
    ) -> HCIRTransaction:
        # Invariant #1: don't re-process committed/rejected transactions
        if transaction.status in (TransactionStatus.COMMITTED, TransactionStatus.REJECTED):
            logger.warning("Transaction %s already %s", transaction.id, transaction.status)
            return transaction

        self._annotate_context(transaction, context)

        if not self._verify(transaction, stages):
            return transaction

        transaction.status = TransactionStatus.VALIDATED

        # Apply operations atomically: on any failure, revert the partial effects.
        try:
            undo_entries = self._apply_operations(transaction)
        except Exception as exc:
            transaction.status = TransactionStatus.REJECTED
            transaction.annotations.append(
                TransactionAnnotation(
                    author="TransactionManager",
                    assertion=f"Apply failed (atomically reverted): {exc}",
                    severity="error",
                )
            )
            self._rejected_log.append(transaction)
            self._publish(
                KernelEventType.TRANSACTION_REJECTED,
                {"tx_id": transaction.id, "author": transaction.author, "reason": str(exc)},
            )
            logger.error("Transaction %s apply failed, reverted: %s", transaction.id, exc)
            return transaction

        transaction.status = TransactionStatus.COMMITTED
        self._record_committed(transaction, undo_entries)

        snap = self._workspace.create_snapshot()
        self._tx_versions[transaction.id] = snap.version

        self._workspace.snapshot_manager.record_kernel_event(
            EventType.TRANSACTION_COMMITTED,
            {
                "tx_id": transaction.id,
                "author": transaction.author,
                "op_count": transaction.operation_count,
            },
        )
        self._publish(
            KernelEventType.TRANSACTION_COMMITTED,
            {
                "tx_id": transaction.id,
                "author": transaction.author,
                "op_count": transaction.operation_count,
                "version": snap.version,
            },
        )
        logger.debug(
            "Transaction %s committed (%d ops)", transaction.id, transaction.operation_count
        )
        return transaction

    def _verify(self, transaction: HCIRTransaction, stages: list[IVerificationStage]) -> bool:
        for stage in stages:
            if not stage.verify(transaction, self._workspace):
                transaction.status = TransactionStatus.REJECTED
                self._rejected_log.append(transaction)
                self._workspace.snapshot_manager.record_kernel_event(
                    EventType.TRANSACTION_REJECTED,
                    {"tx_id": transaction.id, "author": transaction.author},
                )
                self._publish(
                    KernelEventType.TRANSACTION_REJECTED,
                    {
                        "tx_id": transaction.id,
                        "author": transaction.author,
                        "stage": type(stage).__name__,
                    },
                )
                logger.info("Transaction %s rejected by %s", transaction.id, type(stage).__name__)
                return False
        return True

    @staticmethod
    def _annotate_context(transaction: HCIRTransaction, context: Any) -> None:
        if context is None:
            return
        if hasattr(context, "authority_chain"):
            transaction.annotations.append(
                TransactionAnnotation(
                    author="CognitiveAuthorityChain",
                    assertion=f"Authority: {context.authority_chain}",
                    severity="info",
                )
            )
        if hasattr(context, "execution_context"):
            exec_ctx = context.execution_context
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

    def _record_committed(
        self, transaction: HCIRTransaction, undo_entries: list[dict[str, Any]]
    ) -> None:
        if len(self._committed_log) == self._committed_log.maxlen:
            evicted = self._committed_log[0]
            self._undo_logs.pop(evicted.id, None)
            self._tx_versions.pop(evicted.id, None)
        self._committed_log.append(transaction)
        self._undo_logs[transaction.id] = undo_entries

    def _publish(self, event_type: KernelEventType, data: dict[str, Any]) -> None:
        if self._event_bus is None:
            return
        self._event_bus.publish(
            KernelEvent(event_type=event_type.value, source=KERNEL_AUTHOR, data=data)
        )

    def commit_with_verification(
        self,
        transaction: HCIRTransaction,
        verifiers: list[IVerificationStage] | None = None,
        context: Any = None,
    ) -> tuple[bool, HCIRTransaction, HCIRTransaction | None]:
        """Two-phase constitutional commit with automated compensating rollback on failure.

        Phase 1 (Prepare & Verify):
            Evaluates transaction against the default pipeline plus any extra
            ``verifiers`` exactly once. If any check fails, the transaction is
            marked REJECTED and no state is mutated.

        Phase 2 (Atomic Commit & Compensating Rollback):
            Attempts atomic state mutation. If an exception escapes, any partial
            effects are rolled back and a compensating transaction is committed.

        Returns:
            (is_successful, committed_or_rejected_tx, compensating_tx_if_any)
        """
        all_stages = list(self._verification_stages)
        if verifiers:
            all_stages.extend(verifiers)

        with self._lock:
            try:
                res = self._commit(transaction, context, all_stages)
                return res.is_committed, res, None
            except Exception as exc:
                logger.error(
                    "Phase 2 commit failed for tx %s: %s; initiating compensating rollback",
                    transaction.id,
                    exc,
                )
                comp_tx = self.compensate(transaction.id, reason=f"phase_2_failure: {exc}")
                return False, transaction, comp_tx

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
        txs: list[HCIRTransaction] = list(self._committed_log)
        if author:
            txs = [t for t in txs if t.author == author]
        return txs[-limit:]

    def version_of(self, tx_id: str) -> int | None:
        """Workspace snapshot version produced by a committed transaction."""
        return self._tx_versions.get(tx_id)

    # ── Compensation & Rollback ──────────────────────────────────────

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
        original = self._find_committed(original_tx_id)
        if original is None:
            logger.warning("compensate: transaction %s not found", original_tx_id)
            return None

        reverse_ops = self._invert_operations(original.operations)
        if not reverse_ops:
            logger.info("compensate: no invertible operations in %s", original_tx_id)
            return None

        result = self.commit(
            self._kernel_transaction(
                reverse_ops,
                reason=f"Compensating tx {original_tx_id}: {reason}",
                assertion=f"Compensating transaction for {original_tx_id}: {reason}",
            )
        )

        if result.is_committed:
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

        inverse = self._inverse_from_undo(undo_records)
        if not inverse:
            return None

        return self.commit(
            self._kernel_transaction(
                inverse,
                reason=f"Rollback tx {original_tx_id}: {reason}",
                assertion=f"Rollback transaction for {original_tx_id}: {reason}",
            )
        )

    def rollback_to_version(
        self,
        target_version: int,
        reason: str = "isa_rollback",
    ) -> list[HCIRTransaction]:
        """Restore the workspace to the state it had at snapshot ``target_version``.

        Every transaction committed after the target version (including prior
        rollback/compensation transactions) is reverted in reverse order using
        its exact undo log, so the result is the precise before-image.

        Returns the list of committed rollback transactions (newest first).
        """
        with self._lock:
            to_revert = [
                tx
                for tx in reversed(list(self._committed_log))
                if self._tx_versions.get(tx.id, 0) > target_version
            ]
            reverted: list[HCIRTransaction] = []
            for tx in to_revert:
                rb = self.rollback(tx.id, reason=reason)
                if rb is not None and rb.is_committed:
                    reverted.append(rb)
            self._workspace.snapshot_manager.record_kernel_event(
                EventType.ROLLBACK,
                {
                    "target_version": target_version,
                    "transactions_reverted": len(to_revert),
                    "reason": reason,
                },
            )
            return reverted

    @staticmethod
    def _kernel_transaction(
        ops: list[TransactionOperation], *, reason: str, assertion: str
    ) -> HCIRTransaction:
        from hbllm.hcir.types import Provenance

        return HCIRTransaction(
            author=KERNEL_AUTHOR,
            operations=ops,
            provenance=Provenance(created_by=KERNEL_AUTHOR, reason=reason),
            annotations=[
                TransactionAnnotation(
                    author="TransactionManager", assertion=assertion, severity="warning"
                )
            ],
        )

    def _find_committed(self, tx_id: str) -> HCIRTransaction | None:
        """Find a committed transaction by ID."""
        for tx in self._committed_log:
            if tx.id == tx_id:
                return tx
        return None

    @staticmethod
    def _inverse_from_undo(undo_records: list[dict[str, Any]]) -> list[TransactionOperation]:
        """Build exact inverse operations from captured before-images (newest first)."""
        inverse: list[TransactionOperation] = []
        for rec in reversed(undo_records):
            op = rec["op"]
            prev_node = rec.get("previous_node")
            prev_edge = rec.get("previous_edge")
            nid = rec.get("node_id")
            eid = rec.get("edge_id")

            if op in (TransactionOp.ADD_NODE, TransactionOp.UPSERT_NODE, TransactionOp.MODIFY_NODE):
                if prev_node:
                    inverse.append(
                        TransactionOperation(
                            op=TransactionOp.UPSERT_NODE, node_id=nid, node_data=prev_node
                        )
                    )
                elif nid:
                    inverse.append(TransactionOperation(op=TransactionOp.REMOVE_NODE, node_id=nid))
            elif op == TransactionOp.REMOVE_NODE:
                if prev_node:
                    inverse.append(
                        TransactionOperation(
                            op=TransactionOp.UPSERT_NODE, node_id=nid, node_data=prev_node
                        )
                    )
                    for edge_data in rec.get("cascaded_edges", []):
                        inverse.append(
                            TransactionOperation(
                                op=TransactionOp.ADD_EDGE,
                                edge_id=edge_data.get("id"),
                                edge_data=edge_data,
                            )
                        )
            elif op == TransactionOp.ADD_EDGE:
                if eid:
                    inverse.append(TransactionOperation(op=TransactionOp.REMOVE_EDGE, edge_id=eid))
            elif op == TransactionOp.UPSERT_EDGE:
                if prev_edge:
                    inverse.append(
                        TransactionOperation(
                            op=TransactionOp.UPSERT_EDGE, edge_id=eid, edge_data=prev_edge
                        )
                    )
                elif eid:
                    inverse.append(TransactionOperation(op=TransactionOp.REMOVE_EDGE, edge_id=eid))
            elif op == TransactionOp.REMOVE_EDGE:
                if prev_edge:
                    inverse.append(
                        TransactionOperation(
                            op=TransactionOp.ADD_EDGE, edge_id=eid, edge_data=prev_edge
                        )
                    )
        return inverse

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

    # ── Raw application (kernel-internal) ────────────────────────────

    def _apply_operations(self, transaction: HCIRTransaction) -> list[dict[str, Any]]:
        """Apply transaction operations to the workspace graph with undo logging.

        Atomic: if any operation raises, every operation already applied is
        reverted from its captured before-image before the exception propagates.
        """
        undo_entries: list[dict[str, Any]] = []
        try:
            for op in transaction.operations:
                self._apply_one(op, transaction.author, undo_entries)
        except Exception:
            if undo_entries:
                self._revert_in_flight(undo_entries)
            raise
        return undo_entries

    def _revert_in_flight(self, undo_entries: list[dict[str, Any]]) -> None:
        inverse = self._inverse_from_undo(undo_entries)
        scratch: list[dict[str, Any]] = []
        for op in inverse:
            try:
                self._apply_one(op, KERNEL_AUTHOR, scratch)
            except Exception as exc:  # pragma: no cover - defensive
                logger.critical("In-flight revert failed for %s: %s", op.op, exc)

    def _edge_snapshot(self, edge_id: str | None) -> dict[str, Any] | None:
        if not edge_id:
            return None
        edge = self._workspace.get_edge(edge_id)
        return edge.model_dump() if edge is not None else None

    def _apply_one(
        self, op: TransactionOperation, author: str, undo_entries: list[dict[str, Any]]
    ) -> None:
        ws = self._workspace
        if op.op in (TransactionOp.ADD_NODE, TransactionOp.UPSERT_NODE):
            raw_data = dict(op.node_data or {})
            if "id" not in raw_data and op.node_id:
                raw_data["id"] = op.node_id
            target_id = raw_data.get("id") or op.node_id
            existing = ws.get_node(target_id) if target_id else None
            node = self._deserialize_node(raw_data)
            if op.op == TransactionOp.ADD_NODE:
                ws._add_node(node, author=author)
            else:
                ws._upsert_node(node, author=author)
            undo_entries.append(
                {
                    "op": op.op,
                    "node_id": target_id,
                    "previous_node": existing.model_dump() if existing else None,
                }
            )

        elif op.op == TransactionOp.MODIFY_NODE:
            changes = op.changes or op.node_data
            if op.node_id and changes:
                existing = ws.get_node(op.node_id)
                if existing is None:
                    raise ValueError(f"Cannot modify non-existent node: {op.node_id}")
                updated_data = existing.model_dump()
                updated_data.update(changes)
                updated = type(existing).model_validate(updated_data)
                ws._upsert_node(updated, author=author)
                undo_entries.append(
                    {
                        "op": op.op,
                        "node_id": op.node_id,
                        "previous_node": existing.model_dump(),
                    }
                )

        elif op.op == TransactionOp.REMOVE_NODE:
            if op.node_id:
                existing = ws.get_node(op.node_id)
                cascaded: list[dict[str, Any]] = []
                if existing is not None:
                    touched = {
                        e.id: e
                        for e in [*ws.graph.edges_from(op.node_id), *ws.graph.edges_to(op.node_id)]
                    }
                    cascaded = [e.model_dump() for e in touched.values()]
                ws._remove_node(op.node_id, author=author)
                undo_entries.append(
                    {
                        "op": op.op,
                        "node_id": op.node_id,
                        "previous_node": existing.model_dump() if existing else None,
                        "cascaded_edges": cascaded,
                    }
                )

        elif op.op == TransactionOp.ADD_EDGE:
            raw_edge = dict(op.edge_data or {})
            if "id" not in raw_edge and op.edge_id:
                raw_edge["id"] = op.edge_id
            edge = HCIREdge.model_validate(raw_edge)
            ws._add_edge(edge, author=author)
            undo_entries.append({"op": op.op, "edge_id": edge.id, "previous_edge": None})

        elif op.op == TransactionOp.UPSERT_EDGE:
            raw_edge = dict(op.edge_data or {})
            if "id" not in raw_edge and op.edge_id:
                raw_edge["id"] = op.edge_id
            edge = HCIREdge.model_validate(raw_edge)
            prev_edge = self._edge_snapshot(edge.id)
            ws._upsert_edge(edge, author=author)
            undo_entries.append({"op": op.op, "edge_id": edge.id, "previous_edge": prev_edge})

        elif op.op == TransactionOp.REMOVE_EDGE:
            if op.edge_id:
                prev_edge = self._edge_snapshot(op.edge_id)
                ws._remove_edge(op.edge_id, author=author)
                undo_entries.append(
                    {"op": op.op, "edge_id": op.edge_id, "previous_edge": prev_edge}
                )

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
