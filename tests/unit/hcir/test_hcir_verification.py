"""Unit tests for HCIR Verification Pipeline."""

import pytest

from hbllm.hcir.graph import (
    BeliefNode,
    GoalNode,
    HCIREdge,
    HCIREdgeType,
    HCIRNodeType,
)
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.kernel.verification import (
    PolicyVerifier,
    ResourceVerifier,
    SchemaVerifier,
    ScopeVerifier,
    create_default_pipeline,
)
from hbllm.hcir.transactions import (
    HCIRTransaction,
    TransactionOp,
    TransactionOperation,
    TransactionStatus,
)
from hbllm.hcir.types import Scope, SecurityLevel
from hbllm.hcir.workspace import HCIRWorkspaceState, ResourceBudget


# ═══════════════════════════════════════════════════════════════════════════
# ScopeVerifier Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestScopeVerifier:
    def _make_system(self, author_tenant: str = "acme") -> tuple:
        ws = HCIRWorkspaceState()
        scope = Scope(tenant_id=author_tenant)
        verifier = ScopeVerifier(default_scope=scope)
        return ws, verifier

    def test_same_tenant_allowed(self):
        ws, verifier = self._make_system("acme")
        node = GoalNode(id="g1", description="test", scope=Scope(tenant_id="acme"))
        tx = HCIRTransaction(
            author="planner",
            operations=[TransactionOperation(
                op=TransactionOp.ADD_NODE, node_id="g1", node_data=node.model_dump(),
            )],
        )
        assert verifier.verify(tx, ws) is True

    def test_cross_tenant_add_rejected(self):
        ws, verifier = self._make_system("acme")
        node = GoalNode(id="g1", description="test", scope=Scope(tenant_id="evil_corp"))
        tx = HCIRTransaction(
            author="planner",
            operations=[TransactionOperation(
                op=TransactionOp.ADD_NODE, node_id="g1", node_data=node.model_dump(),
            )],
        )
        assert verifier.verify(tx, ws) is False

    def test_cross_tenant_modify_rejected(self):
        ws, verifier = self._make_system("acme")
        # Add a node owned by different tenant
        ws.add_node(GoalNode(id="g1", description="x", scope=Scope(tenant_id="other")))
        tx = HCIRTransaction(
            author="planner",
            operations=[TransactionOperation(
                op=TransactionOp.MODIFY_NODE, node_id="g1",
                changes={"description": "hacked"},
            )],
        )
        assert verifier.verify(tx, ws) is False

    def test_cross_tenant_remove_rejected(self):
        ws, verifier = self._make_system("acme")
        ws.add_node(GoalNode(id="g1", description="x", scope=Scope(tenant_id="other")))
        tx = HCIRTransaction(
            author="planner",
            operations=[TransactionOperation(op=TransactionOp.REMOVE_NODE, node_id="g1")],
        )
        assert verifier.verify(tx, ws) is False

    def test_system_scoped_author_bypasses(self):
        ws, verifier = self._make_system("acme")
        verifier.register_author_scope(
            "kernel", Scope(security_level=SecurityLevel.SYSTEM)
        )
        node = GoalNode(id="g1", description="test", scope=Scope(tenant_id="any_tenant"))
        tx = HCIRTransaction(
            author="kernel",
            operations=[TransactionOperation(
                op=TransactionOp.ADD_NODE, node_id="g1", node_data=node.model_dump(),
            )],
        )
        assert verifier.verify(tx, ws) is True

    def test_system_node_not_removable_by_tenant(self):
        ws, verifier = self._make_system("acme")
        ws.add_node(GoalNode(
            id="g1", description="system node",
            scope=Scope(security_level=SecurityLevel.SYSTEM),
        ))
        tx = HCIRTransaction(
            author="planner",
            operations=[TransactionOperation(op=TransactionOp.REMOVE_NODE, node_id="g1")],
        )
        assert verifier.verify(tx, ws) is False

    def test_cross_tenant_edge_rejected(self):
        ws, verifier = self._make_system("acme")
        ws.add_node(GoalNode(id="g1", description="a", scope=Scope(tenant_id="acme")))
        ws.add_node(GoalNode(id="g2", description="b", scope=Scope(tenant_id="other")))
        edge = HCIREdge(id="e1", edge_type=HCIREdgeType.SUPPORTS, sources=["g1"], targets=["g2"])
        tx = HCIRTransaction(
            author="planner",
            operations=[TransactionOperation(
                op=TransactionOp.ADD_EDGE, edge_id="e1", edge_data=edge.model_dump(),
            )],
        )
        assert verifier.verify(tx, ws) is False


# ═══════════════════════════════════════════════════════════════════════════
# SchemaVerifier Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSchemaVerifier:
    def test_valid_add_node(self):
        ws = HCIRWorkspaceState()
        verifier = SchemaVerifier()
        node = GoalNode(id="g1", description="test")
        tx = HCIRTransaction(
            author="test",
            operations=[TransactionOperation(
                op=TransactionOp.ADD_NODE, node_id="g1", node_data=node.model_dump(),
            )],
        )
        assert verifier.verify(tx, ws) is True

    def test_missing_node_data_rejected(self):
        ws = HCIRWorkspaceState()
        verifier = SchemaVerifier()
        tx = HCIRTransaction(
            author="test",
            operations=[TransactionOperation(op=TransactionOp.ADD_NODE, node_id="g1")],
        )
        assert verifier.verify(tx, ws) is False

    def test_missing_node_type_rejected(self):
        ws = HCIRWorkspaceState()
        verifier = SchemaVerifier()
        tx = HCIRTransaction(
            author="test",
            operations=[TransactionOperation(
                op=TransactionOp.ADD_NODE, node_id="g1", node_data={"id": "g1"},
            )],
        )
        assert verifier.verify(tx, ws) is False

    def test_unknown_node_type_rejected(self):
        ws = HCIRWorkspaceState()
        verifier = SchemaVerifier()
        tx = HCIRTransaction(
            author="test",
            operations=[TransactionOperation(
                op=TransactionOp.ADD_NODE, node_id="g1",
                node_data={"id": "g1", "node_type": "nonexistent_type"},
            )],
        )
        assert verifier.verify(tx, ws) is False

    def test_modify_missing_node_id_rejected(self):
        ws = HCIRWorkspaceState()
        verifier = SchemaVerifier()
        tx = HCIRTransaction(
            author="test",
            operations=[TransactionOperation(
                op=TransactionOp.MODIFY_NODE, changes={"x": 1},
            )],
        )
        assert verifier.verify(tx, ws) is False

    def test_add_edge_missing_sources_rejected(self):
        ws = HCIRWorkspaceState()
        verifier = SchemaVerifier()
        tx = HCIRTransaction(
            author="test",
            operations=[TransactionOperation(
                op=TransactionOp.ADD_EDGE, edge_id="e1",
                edge_data={"id": "e1", "edge_type": "supports", "targets": ["g1"]},
            )],
        )
        assert verifier.verify(tx, ws) is False


# ═══════════════════════════════════════════════════════════════════════════
# ResourceVerifier Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestResourceVerifier:
    def test_within_budget_passes(self):
        ws = HCIRWorkspaceState()
        ws.register_resource(ResourceBudget(name="tokens", limit=1000.0))
        verifier = ResourceVerifier(check_resources=["tokens"])
        tx = HCIRTransaction(author="test")
        assert verifier.verify(tx, ws) is True

    def test_exceeded_budget_rejected(self):
        ws = HCIRWorkspaceState()
        budget = ResourceBudget(name="tokens", limit=100.0)
        budget.consumed = 150.0  # Over budget
        ws.register_resource(budget)
        verifier = ResourceVerifier(check_resources=["tokens"])
        tx = HCIRTransaction(author="test")
        assert verifier.verify(tx, ws) is False

    def test_unknown_resource_passes(self):
        ws = HCIRWorkspaceState()
        verifier = ResourceVerifier(check_resources=["gpu_memory"])
        tx = HCIRTransaction(author="test")
        assert verifier.verify(tx, ws) is True


# ═══════════════════════════════════════════════════════════════════════════
# PolicyVerifier Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPolicyVerifier:
    def test_no_rules_passes(self):
        ws = HCIRWorkspaceState()
        verifier = PolicyVerifier()
        tx = HCIRTransaction(author="test")
        assert verifier.verify(tx, ws) is True

    def test_passing_rule(self):
        ws = HCIRWorkspaceState()
        verifier = PolicyVerifier()
        verifier.add_rule("always_pass", lambda tx, ws: True)
        tx = HCIRTransaction(author="test")
        assert verifier.verify(tx, ws) is True

    def test_blocking_rule(self):
        ws = HCIRWorkspaceState()
        verifier = PolicyVerifier()
        verifier.add_rule("always_block", lambda tx, ws: False)
        tx = HCIRTransaction(author="test")
        assert verifier.verify(tx, ws) is False

    def test_exception_in_rule_rejects(self):
        ws = HCIRWorkspaceState()
        verifier = PolicyVerifier()

        def bad_rule(tx, ws):
            raise RuntimeError("policy engine crash")

        verifier.add_rule("bad_rule", bad_rule)
        tx = HCIRTransaction(author="test")
        assert verifier.verify(tx, ws) is False


# ═══════════════════════════════════════════════════════════════════════════
# Full Pipeline Integration
# ═══════════════════════════════════════════════════════════════════════════


class TestFullPipeline:
    def test_pipeline_with_transaction_manager(self):
        """Test the full pipeline wired into TransactionManager."""
        ws = HCIRWorkspaceState()
        pipeline = create_default_pipeline(
            default_scope=Scope(tenant_id="acme"),
        )
        mgr = TransactionManager(ws, verification_stages=pipeline)

        # Valid transaction: same tenant
        node = GoalNode(id="g1", description="test", scope=Scope(tenant_id="acme"))
        tx = HCIRTransaction(
            author="planner",
            operations=[TransactionOperation(
                op=TransactionOp.ADD_NODE, node_id="g1", node_data=node.model_dump(),
            )],
        )
        result = mgr.commit(tx)
        assert result.is_committed
        assert ws.get_node("g1") is not None

    def test_pipeline_rejects_cross_tenant(self):
        """ScopeVerifier blocks cross-tenant write via pipeline."""
        ws = HCIRWorkspaceState()
        pipeline = create_default_pipeline(
            default_scope=Scope(tenant_id="acme"),
        )
        mgr = TransactionManager(ws, verification_stages=pipeline)

        node = GoalNode(id="g1", description="test", scope=Scope(tenant_id="evil_corp"))
        tx = HCIRTransaction(
            author="planner",
            operations=[TransactionOperation(
                op=TransactionOp.ADD_NODE, node_id="g1", node_data=node.model_dump(),
            )],
        )
        result = mgr.commit(tx)
        assert result.is_rejected
        assert ws.get_node("g1") is None

    def test_pipeline_rejects_bad_schema(self):
        """SchemaVerifier blocks malformed transaction."""
        ws = HCIRWorkspaceState()
        pipeline = create_default_pipeline(
            default_scope=Scope(tenant_id="acme"),
        )
        mgr = TransactionManager(ws, verification_stages=pipeline)

        tx = HCIRTransaction(
            author="planner",
            operations=[TransactionOperation(
                op=TransactionOp.ADD_NODE, node_id="g1",
                node_data={"id": "g1"},  # Missing node_type
            )],
        )
        result = mgr.commit(tx)
        assert result.is_rejected
