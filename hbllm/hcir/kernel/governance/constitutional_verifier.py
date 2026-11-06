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
from typing import Any

from hbllm.hcir.transactions import (
    HCIRTransaction,
    TransactionAnnotation,
    TransactionOp,
)
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.workspace_tiers import WorkspaceTier

logger = logging.getLogger(__name__)


class ConstitutionalViolationError(RuntimeError):
    """Raised when an operation or transaction violates constitutional constraints."""


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
        authority_registry: Any | None = None,
    ) -> None:
        self._max_operations = max_operations_per_tx
        self._forbidden_types = set(forbidden_node_types or [])
        self._require_provenance = require_provenance
        self._enforce_scope = enforce_scope_isolation
        self.authority_registry = authority_registry
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

        # Check 6: Per-tier commit authority (WP6)
        if not self._check_tier_commit_authority(transaction, workspace):
            self._rejections += 1
            return False

        # Check 6b: Persistent skills are removed only via supersession
        if not self._check_persistent_supersede(transaction, workspace):
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

        tx_tenant = self._transaction_tenant(transaction)
        if not tx_tenant:
            # System-level / unscoped transactions may operate across scopes
            return True

        for op in transaction.operations:
            node_tenant: str | None = None
            if op.op in (TransactionOp.MODIFY_NODE, TransactionOp.REMOVE_NODE):
                if op.node_id:
                    existing = workspace.get_node(op.node_id)
                    if existing is not None:
                        node_tenant = existing.scope.tenant_id
            elif op.op in (TransactionOp.ADD_NODE, TransactionOp.UPSERT_NODE):
                scope = (op.node_data or {}).get("scope")
                if isinstance(scope, dict):
                    node_tenant = scope.get("tenant_id")
                else:
                    node_tenant = getattr(scope, "tenant_id", None)
                target_id = (op.node_data or {}).get("id") or op.node_id
                existing = workspace.get_node(target_id) if target_id else None
                if existing is not None and existing.scope.tenant_id != "default":
                    # Overwriting an existing node is governed by the node's current owner
                    node_tenant = existing.scope.tenant_id

            if node_tenant and node_tenant != "default" and tx_tenant != node_tenant:
                transaction.annotations.append(
                    TransactionAnnotation(
                        author="ConstitutionalVerifier",
                        assertion=(
                            f"Scope isolation violation: tx tenant={tx_tenant} "
                            f"vs node tenant={node_tenant}"
                        ),
                        severity="error",
                    )
                )
                return False
        return True

    @staticmethod
    def _transaction_tenant(transaction: HCIRTransaction) -> str | None:
        tx_metadata = getattr(transaction, "metadata", None)
        tx_tenant = tx_metadata.get("tenant_id") if isinstance(tx_metadata, dict) else None
        if not tx_tenant and getattr(transaction, "provenance", None):
            tx_tenant = (
                getattr(getattr(transaction.provenance, "scope", None), "tenant_id", None)
                or transaction.provenance.session_id
                or None
            )
        return tx_tenant

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

    def _check_tier_commit_authority(
        self,
        transaction: HCIRTransaction,
        workspace: HCIRWorkspaceState,
    ) -> bool:
        """Enforce per-tier commit authority via kernel-issued grants (WP6.4).

        WORKING is always open. Higher tiers require a ``commit_grant``
        metadata entry naming a live grant issued by the kernel's
        CommitAuthorityRegistry — self-declared author strings are ignored.
        Without a configured registry (standalone workspaces) all tiers
        remain open for backward compatibility.
        """
        if self.authority_registry is None:
            return True
        target_tier = getattr(workspace, "tier", None)
        if target_tier is None or target_tier == WorkspaceTier.WORKING:
            return True
        grant_id = (
            transaction.metadata.get("commit_grant")
            if isinstance(transaction.metadata, dict)
            else None
        )
        if self.authority_registry.authorizes(grant_id, target_tier):
            return True
        transaction.annotations.append(
            TransactionAnnotation(
                author="ConstitutionalVerifier",
                assertion=(
                    f"Commit authority violation: no valid grant for tier "
                    f"'{target_tier}' (author '{transaction.author}')"
                ),
                severity="critical",
            )
        )
        return False

    def _check_persistent_supersede(
        self,
        transaction: HCIRTransaction,
        workspace: HCIRWorkspaceState,
    ) -> bool:
        """PERSISTENT: removing a SkillNode requires metadata['supersedes']."""
        if getattr(workspace, "tier", None) != WorkspaceTier.PERSISTENT:
            return True
        from hbllm.hcir.graph import SkillNode

        supersedes = (
            transaction.metadata.get("supersedes")
            if isinstance(transaction.metadata, dict)
            else None
        )
        for op in transaction.operations:
            if op.op == TransactionOp.REMOVE_NODE and op.node_id:
                node = workspace.get_node(op.node_id)
                if isinstance(node, SkillNode) and not supersedes:
                    transaction.annotations.append(
                        TransactionAnnotation(
                            author="ConstitutionalVerifier",
                            assertion=(
                                f"Persistent skill {op.node_id} may only be removed "
                                "via supersedes (replacement node id in metadata)"
                            ),
                            severity="critical",
                        )
                    )
                    return False
        return True


# Backward and architectural compatibility alias
ConstitutionalTransactionVerifier = ConstitutionalVerifier
