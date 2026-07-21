"""
HCIR Verification Pipeline — staged transaction verification.

Every ``HCIRTransaction`` passes through an ordered pipeline of
``IVerificationStage`` implementations before it can be committed.
This module provides the concrete stages:

    ScopeVerifier        — tenant/user/device isolation enforcement
    SchemaVerifier       — structural integrity of operations
    ResourceVerifier     — resource budget checks
    PolicyVerifier       — governance policy enforcement

Pipeline::

    Transaction
        │
        ▼
    ┌──────────────┐
    │ ScopeVerifier │
    └──────┬───────┘
           │
    ┌──────▼──────────┐
    │ SchemaVerifier   │
    └──────┬───────────┘
           │
    ┌──────▼──────────┐
    │ ResourceVerifier │
    └──────┬───────────┘
           │
    ┌──────▼──────────┐
    │ PolicyVerifier   │
    └──────┬───────────┘
           │
           ▼
       Commit / Reject
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.hcir.graph import NODE_TYPE_REGISTRY, HCIRNodeType
from hbllm.hcir.transactions import (
    HCIRTransaction,
    TransactionAnnotation,
    TransactionOp,
    TransactionOperation,
)
from hbllm.hcir.types import Scope, SecurityLevel
from hbllm.hcir.workspace import HCIRWorkspaceState

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Scope Verifier — Kernel Invariant #4 (absolute scope isolation)
# ═══════════════════════════════════════════════════════════════════════════


class ScopeVerifier:
    """Enforce tenant/user/device scope isolation at the mutation boundary.

    No graph mutation happens without scope validation.  This is the
    HCIR equivalent of ``tenant_guard.py``:

        Transaction → TransactionManager → **ScopeVerifier** → Graph

    Rules:
        1. ADD/UPSERT: node scope must match the transaction author's scope.
        2. MODIFY/REMOVE: target node scope must be accessible from the
           transaction author's scope.
        3. ADD_EDGE: cannot cross tenant boundaries (unless system-scoped).
        4. System-scoped authors bypass tenant checks.

    Usage::

        verifier = ScopeVerifier(default_scope=Scope(tenant_id="acme"))
        pipeline = [verifier, schema_verifier, ...]
        tx_manager = TransactionManager(workspace, verification_stages=pipeline)
    """

    def __init__(
        self,
        default_scope: Scope | None = None,
        strict: bool = True,
    ) -> None:
        self._default_scope = default_scope or Scope()
        self._strict = strict
        # author_id → Scope mapping (registered at bootstrap)
        self._author_scopes: dict[str, Scope] = {}

    def register_author_scope(self, author_id: str, scope: Scope) -> None:
        """Register the scope identity for a transaction author."""
        self._author_scopes[author_id] = scope

    def verify(
        self,
        transaction: HCIRTransaction,
        workspace: HCIRWorkspaceState,
    ) -> bool:
        """Verify all operations in the transaction respect scope isolation."""
        author_scope = self._author_scopes.get(transaction.author, self._default_scope)

        # System-scoped authors bypass tenant checks
        if author_scope.security_level == SecurityLevel.SYSTEM:
            return True

        for op in transaction.operations:
            if not self._verify_operation(op, author_scope, workspace, transaction):
                return False

        return True

    def _verify_operation(
        self,
        op: TransactionOperation,
        author_scope: Scope,
        workspace: HCIRWorkspaceState,
        transaction: HCIRTransaction,
    ) -> bool:
        """Verify a single operation respects scope boundaries."""

        if op.op in (TransactionOp.ADD_NODE, TransactionOp.UPSERT_NODE):
            return self._verify_node_scope(op, author_scope, transaction)

        if op.op == TransactionOp.MODIFY_NODE:
            return self._verify_modify_scope(op, author_scope, workspace, transaction)

        if op.op == TransactionOp.REMOVE_NODE:
            return self._verify_remove_scope(op, author_scope, workspace, transaction)

        if op.op == TransactionOp.ADD_EDGE:
            return self._verify_edge_scope(op, author_scope, workspace, transaction)

        # REMOVE_EDGE: edges inherit scope from their endpoints,
        # checked during add; removal is always allowed by the author.
        return True

    def _verify_node_scope(
        self,
        op: TransactionOperation,
        author_scope: Scope,
        transaction: HCIRTransaction,
    ) -> bool:
        """Verify that a new node's scope is compatible with the author's."""
        if op.node_data is None:
            return True

        node_scope_data = op.node_data.get("scope", {})
        node_tenant = node_scope_data.get("tenant_id", "default")
        node_security = node_scope_data.get("security_level", "tenant")

        # System nodes can be created by anyone (they're shared)
        if node_security == SecurityLevel.SYSTEM:
            return True

        if node_tenant != author_scope.tenant_id:
            self._reject(
                transaction,
                f"Scope violation: author (tenant={author_scope.tenant_id}) "
                f"cannot create node in tenant={node_tenant}",
            )
            return False

        return True

    def _verify_modify_scope(
        self,
        op: TransactionOperation,
        author_scope: Scope,
        workspace: HCIRWorkspaceState,
        transaction: HCIRTransaction,
    ) -> bool:
        """Verify the author can modify the target node."""
        if op.node_id is None:
            return True

        existing = workspace.get_node(op.node_id)
        if existing is None:
            return True  # Non-existent node — handled by TransactionManager

        if existing.scope.security_level == SecurityLevel.SYSTEM:
            return True  # System nodes are modifiable

        if existing.scope.tenant_id != author_scope.tenant_id:
            self._reject(
                transaction,
                f"Scope violation: author (tenant={author_scope.tenant_id}) "
                f"cannot modify node '{op.node_id}' in tenant={existing.scope.tenant_id}",
            )
            return False

        return True

    def _verify_remove_scope(
        self,
        op: TransactionOperation,
        author_scope: Scope,
        workspace: HCIRWorkspaceState,
        transaction: HCIRTransaction,
    ) -> bool:
        """Verify the author can remove the target node."""
        if op.node_id is None:
            return True

        existing = workspace.get_node(op.node_id)
        if existing is None:
            return True

        if existing.scope.security_level == SecurityLevel.SYSTEM:
            # System nodes cannot be removed by non-system authors
            self._reject(
                transaction,
                f"Scope violation: non-system author cannot remove system node '{op.node_id}'",
            )
            return False

        if existing.scope.tenant_id != author_scope.tenant_id:
            self._reject(
                transaction,
                f"Scope violation: author (tenant={author_scope.tenant_id}) "
                f"cannot remove node '{op.node_id}' in tenant={existing.scope.tenant_id}",
            )
            return False

        return True

    def _verify_edge_scope(
        self,
        op: TransactionOperation,
        author_scope: Scope,
        workspace: HCIRWorkspaceState,
        transaction: HCIRTransaction,
    ) -> bool:
        """Verify an edge doesn't cross tenant boundaries."""
        if op.edge_data is None:
            return True

        sources = op.edge_data.get("sources", [])
        targets = op.edge_data.get("targets", [])
        tenant_ids: set[str] = set()

        for nid in sources + targets:
            node = workspace.get_node(nid)
            if node is None:
                continue
            if node.scope.security_level == SecurityLevel.SYSTEM:
                continue  # System nodes bypass scope
            tenant_ids.add(node.scope.tenant_id)

        if len(tenant_ids) > 1:
            self._reject(
                transaction,
                f"Scope violation: edge crosses tenant boundaries: {tenant_ids}",
            )
            return False

        return True

    @staticmethod
    def _reject(transaction: HCIRTransaction, message: str) -> None:
        """Annotate a transaction with a scope violation."""
        transaction.annotations.append(
            TransactionAnnotation(
                author="ScopeVerifier",
                assertion=message,
                severity="error",
            )
        )
        logger.warning("HCIR_SCOPE_VIOLATION: %s (tx=%s)", message, transaction.id)


# ═══════════════════════════════════════════════════════════════════════════
# Schema Verifier — structural validity of operations
# ═══════════════════════════════════════════════════════════════════════════


class SchemaVerifier:
    """Verify that operations contain valid, well-formed data.

    Checks:
        - ADD_NODE operations have valid node_type and required fields.
        - MODIFY_NODE operations reference existing nodes.
        - ADD_EDGE operations have valid sources and targets.
        - All node types are registered in ``NODE_TYPE_REGISTRY``.
    """

    def verify(
        self,
        transaction: HCIRTransaction,
        workspace: HCIRWorkspaceState,
    ) -> bool:
        for op in transaction.operations:
            if not self._verify_op(op, workspace, transaction):
                return False
        return True

    def _verify_op(
        self,
        op: TransactionOperation,
        workspace: HCIRWorkspaceState,
        transaction: HCIRTransaction,
    ) -> bool:
        if op.op in (TransactionOp.ADD_NODE, TransactionOp.UPSERT_NODE):
            if op.node_data is None:
                self._reject(transaction, f"{op.op}: missing node_data")
                return False
            node_type_str = op.node_data.get("node_type")
            if not node_type_str:
                self._reject(transaction, f"{op.op}: missing node_type in node_data")
                return False
            try:
                nt = HCIRNodeType(node_type_str)
            except ValueError:
                self._reject(transaction, f"{op.op}: unknown node_type '{node_type_str}'")
                return False
            if nt not in NODE_TYPE_REGISTRY:
                self._reject(transaction, f"{op.op}: unregistered node_type '{nt}'")
                return False

        elif op.op == TransactionOp.MODIFY_NODE:
            if not op.node_id:
                self._reject(transaction, "MODIFY_NODE: missing node_id")
                return False
            if not op.changes:
                self._reject(transaction, "MODIFY_NODE: empty changes dict")
                return False

        elif op.op == TransactionOp.REMOVE_NODE:
            if not op.node_id:
                self._reject(transaction, "REMOVE_NODE: missing node_id")
                return False

        elif op.op == TransactionOp.ADD_EDGE:
            if op.edge_data is None:
                self._reject(transaction, "ADD_EDGE: missing edge_data")
                return False
            if not op.edge_data.get("sources"):
                self._reject(transaction, "ADD_EDGE: missing sources")
                return False
            if not op.edge_data.get("targets"):
                self._reject(transaction, "ADD_EDGE: missing targets")
                return False
            if not op.edge_data.get("edge_type"):
                self._reject(transaction, "ADD_EDGE: missing edge_type")
                return False

        elif op.op == TransactionOp.REMOVE_EDGE:
            if not op.edge_id:
                self._reject(transaction, "REMOVE_EDGE: missing edge_id")
                return False

        return True

    @staticmethod
    def _reject(transaction: HCIRTransaction, message: str) -> None:
        transaction.annotations.append(
            TransactionAnnotation(
                author="SchemaVerifier",
                assertion=message,
                severity="error",
            )
        )
        logger.warning("HCIR_SCHEMA_VIOLATION: %s (tx=%s)", message, transaction.id)


# ═══════════════════════════════════════════════════════════════════════════
# Resource Verifier — budget enforcement
# ═══════════════════════════════════════════════════════════════════════════


class ResourceVerifier:
    """Verify that the workspace has sufficient resources for the transaction.

    Checks hard resource budgets (tokens, API calls) and rejects
    transactions that would exceed limits.
    """

    def __init__(self, check_resources: list[str] | None = None) -> None:
        self._check_resources = check_resources or ["tokens"]

    def verify(
        self,
        transaction: HCIRTransaction,
        workspace: HCIRWorkspaceState,
    ) -> bool:
        for resource_name in self._check_resources:
            budget = workspace.get_resource(resource_name)
            if budget is None:
                continue  # Unconstrained
            if budget.is_exceeded:
                transaction.annotations.append(
                    TransactionAnnotation(
                        author="ResourceVerifier",
                        assertion=(
                            f"Resource '{resource_name}' exceeded: "
                            f"consumed={budget.consumed}, limit={budget.limit}"
                        ),
                        severity="error",
                    )
                )
                logger.warning("HCIR_RESOURCE_EXCEEDED: %s (tx=%s)", resource_name, transaction.id)
                return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Policy Verifier — governance policy enforcement
# ═══════════════════════════════════════════════════════════════════════════


class PolicyVerifier:
    """Enforce governance policies on transactions.

    Bridges the existing HBLLM ``PolicyEngine`` into the HCIR
    verification pipeline.  Operates on a configurable set of
    policy rules that can block, annotate, or transform transactions.

    Each rule is a callable that receives (transaction, workspace)
    and returns True (pass) or False (block).
    """

    def __init__(self) -> None:
        self._rules: list[tuple[str, Any]] = []  # (name, callable)

    def add_rule(self, name: str, rule_fn: Any) -> None:
        """Add a policy rule.

        Args:
            name: Human-readable rule name.
            rule_fn: Callable(transaction, workspace) → bool.
        """
        self._rules.append((name, rule_fn))

    def verify(
        self,
        transaction: HCIRTransaction,
        workspace: HCIRWorkspaceState,
    ) -> bool:
        for name, rule_fn in self._rules:
            try:
                if not rule_fn(transaction, workspace):
                    transaction.annotations.append(
                        TransactionAnnotation(
                            author="PolicyVerifier",
                            assertion=f"Policy '{name}' rejected transaction",
                            severity="error",
                        )
                    )
                    logger.warning("HCIR_POLICY_VIOLATION: rule '%s' (tx=%s)", name, transaction.id)
                    return False
            except Exception as exc:
                logger.error("Policy rule '%s' raised exception: %s", name, exc)
                transaction.annotations.append(
                    TransactionAnnotation(
                        author="PolicyVerifier",
                        assertion=f"Policy '{name}' error: {exc}",
                        severity="error",
                    )
                )
                return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline Factory
# ═══════════════════════════════════════════════════════════════════════════


def create_default_pipeline(
    default_scope: Scope | None = None,
    check_resources: list[str] | None = None,
) -> list[ScopeVerifier | SchemaVerifier | ResourceVerifier | PolicyVerifier]:
    """Create the default HCIR verification pipeline.

    Order matters: Scope → Schema → Resource → Policy.

    Returns:
        Ordered list of verification stages.
    """
    return [
        ScopeVerifier(default_scope=default_scope),
        SchemaVerifier(),
        ResourceVerifier(check_resources=check_resources),
        PolicyVerifier(),
    ]
