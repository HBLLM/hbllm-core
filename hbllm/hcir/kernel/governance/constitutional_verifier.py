"""
Constitutional Verifier — governance enforcement as IVerificationStage.

Evaluates every transaction against the constitution BEFORE commit.
This is enforcement, not observation.

    Transaction → ConstitutionalVerifier → Approved? → Commit
                                        → Blocked  → Reject

Checks:
    1. Safety: no operations violate safety constraints
    2. Scope isolation: tenant boundaries are respected
    3. Budget: resource limits are not exceeded
    4. Integrity: graph invariants are maintained

Usage::

    verifier = ConstitutionalVerifier()
    tx_manager.add_verification_stage(verifier)
    # Now every transaction is constitutionally verified before commit
"""

from __future__ import annotations

import logging

from hbllm.hcir.transactions import (
    HCIRTransaction,
    TransactionAnnotation,
    TransactionOp,
)
from hbllm.hcir.workspace import HCIRWorkspaceState

logger = logging.getLogger(__name__)


class ConstitutionalVerifier:
    """Evaluates every transaction against the constitution before commit.

    Implements ``IVerificationStage`` for the ``TransactionManager``
    pipeline.  This is the governance gate — transactions that fail
    any constitutional check are rejected and never committed.

    Configurable rules:
        - ``max_operations_per_tx``: Maximum operations in a single transaction
        - ``forbidden_node_types``: Node types that cannot be created/modified
        - ``require_provenance``: Require non-empty provenance on every transaction
        - ``enforce_scope_isolation``: Check tenant scope on every node

    Usage::

        verifier = ConstitutionalVerifier(
            max_operations_per_tx=100,
            require_provenance=True,
        )
        tx_manager.add_verification_stage(verifier)
    """

    def __init__(
        self,
        max_operations_per_tx: int = 500,
        forbidden_node_types: list[str] | None = None,
        require_provenance: bool = True,
        enforce_scope_isolation: bool = True,
    ) -> None:
        self._max_operations = max_operations_per_tx
        self._forbidden_types = set(forbidden_node_types or [])
        self._require_provenance = require_provenance
        self._enforce_scope = enforce_scope_isolation
        self._evaluations: int = 0
        self._approvals: int = 0
        self._rejections: int = 0

    @property
    def evaluations(self) -> int:
        return self._evaluations

    @property
    def approvals(self) -> int:
        return self._approvals

    @property
    def rejections(self) -> int:
        return self._rejections

    @property
    def approval_rate(self) -> float:
        if self._evaluations == 0:
            return 1.0
        return self._approvals / self._evaluations

    def verify(
        self,
        transaction: HCIRTransaction,
        workspace: HCIRWorkspaceState,
    ) -> bool:
        """Evaluate a transaction against the constitution.

        Returns True if the transaction passes all checks.
        Appends rejection annotations on failure.
        """
        self._evaluations += 1

        # Check 1: Operation count limit
        if not self._check_operation_count(transaction):
            self._rejections += 1
            return False

        # Check 2: Provenance requirement
        if not self._check_provenance(transaction):
            self._rejections += 1
            return False

        # Check 3: Forbidden node types
        if not self._check_forbidden_types(transaction):
            self._rejections += 1
            return False

        # Check 4: Scope isolation
        if not self._check_scope_isolation(transaction, workspace):
            self._rejections += 1
            return False

        # Check 5: Safety (no destructive operations on protected nodes)
        if not self._check_safety(transaction, workspace):
            self._rejections += 1
            return False

        self._approvals += 1
        transaction.annotations.append(
            TransactionAnnotation(
                author="ConstitutionalVerifier",
                assertion="Transaction approved by constitutional review",
                severity="info",
            )
        )
        return True

    # ── Individual Checks ────────────────────────────────────────────

    def _check_operation_count(self, transaction: HCIRTransaction) -> bool:
        """Reject transactions with too many operations (DoS protection)."""
        if transaction.operation_count > self._max_operations:
            transaction.annotations.append(
                TransactionAnnotation(
                    author="ConstitutionalVerifier",
                    assertion=(
                        f"Operation count {transaction.operation_count} exceeds "
                        f"limit {self._max_operations}"
                    ),
                    severity="error",
                )
            )
            logger.warning(
                "Constitutional rejection: tx %s has %d ops (limit=%d)",
                transaction.id,
                transaction.operation_count,
                self._max_operations,
            )
            return False
        return True

    def _check_provenance(self, transaction: HCIRTransaction) -> bool:
        """Require non-empty provenance on every transaction."""
        if not self._require_provenance:
            return True

        if not transaction.author:
            transaction.annotations.append(
                TransactionAnnotation(
                    author="ConstitutionalVerifier",
                    assertion="Transaction author is empty — provenance required",
                    severity="error",
                )
            )
            return False
        return True

    def _check_forbidden_types(self, transaction: HCIRTransaction) -> bool:
        """Reject transactions that create/modify forbidden node types."""
        if not self._forbidden_types:
            return True

        for op in transaction.operations:
            if op.op in (TransactionOp.ADD_NODE, TransactionOp.UPSERT_NODE):
                node_type = (op.node_data or {}).get("node_type", "")
                if node_type in self._forbidden_types:
                    transaction.annotations.append(
                        TransactionAnnotation(
                            author="ConstitutionalVerifier",
                            assertion=f"Forbidden node type: {node_type}",
                            severity="error",
                        )
                    )
                    return False
        return True

    def _check_scope_isolation(
        self,
        transaction: HCIRTransaction,
        workspace: HCIRWorkspaceState,
    ) -> bool:
        """Verify tenant scope isolation for all operations."""
        if not self._enforce_scope:
            return True

        for op in transaction.operations:
            if op.op in (TransactionOp.MODIFY_NODE, TransactionOp.REMOVE_NODE):
                if op.node_id:
                    existing = workspace.get_node(op.node_id)
                    if existing is not None:
                        # Check that the transaction's provenance scope
                        # matches the existing node's scope
                        tx_tenant = transaction.provenance.session_id or ""
                        node_tenant = existing.scope.tenant_id

                        # Allow system-level transactions to modify any scope
                        if tx_tenant and node_tenant != "default" and tx_tenant != node_tenant:
                            transaction.annotations.append(
                                TransactionAnnotation(
                                    author="ConstitutionalVerifier",
                                    assertion=(
                                        f"Scope isolation violation: tx session={tx_tenant} "
                                        f"vs node tenant={node_tenant}"
                                    ),
                                    severity="error",
                                )
                            )
                            return False
        return True

    def _check_safety(
        self,
        transaction: HCIRTransaction,
        workspace: HCIRWorkspaceState,
    ) -> bool:
        """Check for unsafe operations (removing active goals, etc.)."""
        for op in transaction.operations:
            if op.op == TransactionOp.REMOVE_NODE and op.node_id:
                node = workspace.get_node(op.node_id)
                if node is not None:
                    # Prevent removing nodes in active lifecycle
                    from hbllm.hcir.graph import GoalNode, NodeLifecycle

                    if isinstance(node, GoalNode) and node.lifecycle == NodeLifecycle.ACTIVE:
                        transaction.annotations.append(
                            TransactionAnnotation(
                                author="ConstitutionalVerifier",
                                assertion=(
                                    f"Safety violation: cannot remove active goal {op.node_id}"
                                ),
                                severity="error",
                            )
                        )
                        return False
        return True
