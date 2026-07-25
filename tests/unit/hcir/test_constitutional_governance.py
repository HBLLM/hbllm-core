"""Tests for Phase 4: Constitutional Governance + Compensating Transactions."""

from __future__ import annotations

from hbllm.hcir.graph import GoalNode, NodeLifecycle
from hbllm.hcir.kernel.governance.constitutional_verifier import ConstitutionalVerifier
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.transactions import (
    HCIRTransaction,
    TransactionOp,
    TransactionOperation,
)
from hbllm.hcir.workspace import HCIRWorkspaceState

# ═══════════════════════════════════════════════════════════════════════════
# Constitutional Verifier Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestConstitutionalVerifier:
    """Verify the ConstitutionalVerifier as IVerificationStage."""

    def setup_method(self) -> None:
        self.workspace = HCIRWorkspaceState()
        self.verifier = ConstitutionalVerifier()

    def _make_tx(self, ops: list[TransactionOperation], author: str = "test") -> HCIRTransaction:
        return HCIRTransaction(author=author, operations=ops)

    def test_approve_valid_transaction(self) -> None:
        tx = self._make_tx(
            [
                TransactionOperation(
                    op=TransactionOp.ADD_NODE,
                    node_data={"id": "g1", "node_type": "goal", "category": "planning"},
                )
            ]
        )
        assert self.verifier.verify(tx, self.workspace) is True
        assert self.verifier.approvals == 1

    def test_reject_too_many_operations(self) -> None:
        verifier = ConstitutionalVerifier(max_operations_per_tx=5)
        ops = [
            TransactionOperation(
                op=TransactionOp.ADD_NODE,
                node_data={"id": f"n{i}", "node_type": "observation", "category": "perception"},
            )
            for i in range(10)
        ]
        tx = self._make_tx(ops)
        assert verifier.verify(tx, self.workspace) is False
        assert verifier.rejections == 1

    def test_reject_empty_author(self) -> None:
        tx = HCIRTransaction(author="", operations=[])
        assert self.verifier.verify(tx, self.workspace) is False

    def test_reject_forbidden_node_type(self) -> None:
        verifier = ConstitutionalVerifier(forbidden_node_types=["world_variable"])
        tx = self._make_tx(
            [
                TransactionOperation(
                    op=TransactionOp.ADD_NODE,
                    node_data={"id": "wv1", "node_type": "world_variable"},
                )
            ]
        )
        assert verifier.verify(tx, self.workspace) is False

    def test_allow_non_forbidden_node_type(self) -> None:
        verifier = ConstitutionalVerifier(forbidden_node_types=["world_variable"])
        tx = self._make_tx(
            [
                TransactionOperation(
                    op=TransactionOp.ADD_NODE,
                    node_data={"id": "g1", "node_type": "goal", "category": "planning"},
                )
            ]
        )
        assert verifier.verify(tx, self.workspace) is True

    def test_reject_remove_active_goal(self) -> None:
        """Safety check: cannot remove an active goal node."""
        goal = GoalNode(id="g_active", description="Active goal", lifecycle=NodeLifecycle.ACTIVE)
        self.workspace.add_node(goal)

        tx = self._make_tx([TransactionOperation(op=TransactionOp.REMOVE_NODE, node_id="g_active")])
        assert self.verifier.verify(tx, self.workspace) is False

    def test_allow_remove_archived_goal(self) -> None:
        goal = GoalNode(id="g_archived", description="Done", lifecycle=NodeLifecycle.ARCHIVED)
        self.workspace.add_node(goal)

        tx = self._make_tx(
            [TransactionOperation(op=TransactionOp.REMOVE_NODE, node_id="g_archived")]
        )
        assert self.verifier.verify(tx, self.workspace) is True

    def test_approval_rate(self) -> None:
        verifier = ConstitutionalVerifier()
        # 2 approvals
        for _ in range(2):
            tx = self._make_tx([])
            verifier.verify(tx, self.workspace)
        # 1 rejection
        verifier.verify(HCIRTransaction(author="", operations=[]), self.workspace)

        assert verifier.evaluations == 3
        assert verifier.approvals == 2
        assert verifier.rejections == 1
        assert abs(verifier.approval_rate - 2 / 3) < 0.01

    def test_verifier_integrates_with_transaction_manager(self) -> None:
        """End-to-end: verifier in the pipeline blocks bad transactions."""
        verifier = ConstitutionalVerifier(max_operations_per_tx=2)
        manager = TransactionManager(self.workspace, verification_stages=[verifier])

        # Valid transaction — should commit
        tx_ok = HCIRTransaction(
            author="test",
            operations=[
                TransactionOperation(
                    op=TransactionOp.ADD_NODE,
                    node_data={
                        "id": "g1",
                        "node_type": "goal",
                        "category": "planning",
                        "description": "Test",
                    },
                )
            ],
        )
        result_ok = manager.commit(tx_ok)
        assert result_ok.is_committed

        # Invalid transaction — too many ops
        ops = [
            TransactionOperation(
                op=TransactionOp.ADD_NODE,
                node_data={
                    "id": f"obs{i}",
                    "node_type": "observation",
                    "category": "perception",
                },
            )
            for i in range(5)
        ]
        tx_bad = HCIRTransaction(author="test", operations=ops)
        result_bad = manager.commit(tx_bad)
        assert result_bad.is_rejected


# ═══════════════════════════════════════════════════════════════════════════
# Compensating Transaction Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCompensatingTransactions:
    """Verify TransactionManager.compensate() generates inverse ops."""

    def setup_method(self) -> None:
        self.workspace = HCIRWorkspaceState()
        self.manager = TransactionManager(self.workspace)

    def test_compensate_add_node(self) -> None:
        """Compensating an ADD_NODE should produce a REMOVE_NODE."""
        tx = HCIRTransaction(
            author="test",
            operations=[
                TransactionOperation(
                    op=TransactionOp.ADD_NODE,
                    node_data={
                        "id": "g_to_remove",
                        "node_type": "goal",
                        "category": "planning",
                        "description": "Will be compensated",
                    },
                )
            ],
        )
        committed = self.manager.commit(tx)
        assert committed.is_committed
        assert self.workspace.get_node("g_to_remove") is not None

        # Compensate
        compensated = self.manager.compensate(committed.id, "test_reason")
        assert compensated is not None
        assert compensated.is_committed

        # Node should be removed
        assert self.workspace.get_node("g_to_remove") is None

    def test_compensate_upsert_node(self) -> None:
        """Compensating an UPSERT_NODE should also REMOVE_NODE."""
        tx = HCIRTransaction(
            author="test",
            operations=[
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_data={
                        "id": "obs_1",
                        "node_type": "observation",
                        "category": "perception",
                        "sensor_source": "test",
                    },
                )
            ],
        )
        committed = self.manager.commit(tx)
        assert committed.is_committed

        compensated = self.manager.compensate(committed.id, "rollback")
        assert compensated is not None
        assert self.workspace.get_node("obs_1") is None

    def test_compensate_nonexistent_tx(self) -> None:
        result = self.manager.compensate("tx_nonexistent", "reason")
        assert result is None

    def test_compensate_multiple_operations(self) -> None:
        """Compensate a multi-op transaction — all nodes removed."""
        tx = HCIRTransaction(
            author="test",
            operations=[
                TransactionOperation(
                    op=TransactionOp.ADD_NODE,
                    node_data={
                        "id": "g1",
                        "node_type": "goal",
                        "category": "planning",
                        "description": "Goal 1",
                    },
                ),
                TransactionOperation(
                    op=TransactionOp.ADD_NODE,
                    node_data={
                        "id": "g2",
                        "node_type": "goal",
                        "category": "planning",
                        "description": "Goal 2",
                    },
                ),
            ],
        )
        committed = self.manager.commit(tx)
        assert committed.is_committed
        assert self.workspace.get_node("g1") is not None
        assert self.workspace.get_node("g2") is not None

        compensated = self.manager.compensate(committed.id, "batch_rollback")
        assert compensated is not None
        assert compensated.is_committed
        assert self.workspace.get_node("g1") is None
        assert self.workspace.get_node("g2") is None

    def test_compensate_remove_node_is_not_invertible(self) -> None:
        """REMOVE_NODE operations cannot be inverted (data is lost)."""
        # First, add a node
        tx1 = HCIRTransaction(
            author="test",
            operations=[
                TransactionOperation(
                    op=TransactionOp.ADD_NODE,
                    node_data={
                        "id": "g_gone",
                        "node_type": "goal",
                        "category": "planning",
                        "description": "Gone",
                    },
                )
            ],
        )
        self.manager.commit(tx1)

        # Then remove it
        tx2 = HCIRTransaction(
            author="test",
            operations=[TransactionOperation(op=TransactionOp.REMOVE_NODE, node_id="g_gone")],
        )
        committed = self.manager.commit(tx2)
        assert committed.is_committed

        # Compensating a REMOVE should return None (no invertible ops)
        result = self.manager.compensate(committed.id, "try_recover")
        assert result is None
