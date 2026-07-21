"""Unit tests for HCIR Transactions and TransactionManager."""

from hbllm.hcir.graph import (
    BeliefNode,
    GoalNode,
    HCIREdge,
    HCIREdgeType,
)
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.transactions import (
    HCIRDelta,
    HCIRTransaction,
    TransactionOp,
    TransactionOperation,
    TransactionStatus,
)
from hbllm.hcir.workspace import HCIRWorkspaceState

# ═══════════════════════════════════════════════════════════════════════════
# Transaction Model Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHCIRTransaction:
    def test_default_status(self):
        tx = HCIRTransaction(author="test")
        assert tx.status == TransactionStatus.PROPOSED
        assert tx.is_committed is False
        assert tx.is_rejected is False

    def test_operation_count(self):
        tx = HCIRTransaction(
            author="test",
            operations=[
                TransactionOperation(op=TransactionOp.ADD_NODE, node_data={"id": "g1"}),
                TransactionOperation(op=TransactionOp.ADD_NODE, node_data={"id": "g2"}),
            ],
        )
        assert tx.operation_count == 2

    def test_auto_id(self):
        tx1 = HCIRTransaction(author="a")
        tx2 = HCIRTransaction(author="b")
        assert tx1.id != tx2.id


class TestHCIRDelta:
    def test_to_operations(self):
        delta = HCIRDelta(
            add_nodes=[{"id": "g1", "node_type": "goal"}],
            remove_node_ids=["old1"],
            add_edges=[{"id": "e1", "edge_type": "supports"}],
            remove_edge_ids=["old_e1"],
        )
        ops = delta.to_operations()
        assert len(ops) == 4
        assert ops[0].op == TransactionOp.ADD_NODE
        assert ops[1].op == TransactionOp.REMOVE_NODE
        assert ops[2].op == TransactionOp.ADD_EDGE
        assert ops[3].op == TransactionOp.REMOVE_EDGE


# ═══════════════════════════════════════════════════════════════════════════
# TransactionManager Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTransactionManager:
    def _make_system(self):
        ws = HCIRWorkspaceState()
        mgr = TransactionManager(ws)
        return ws, mgr

    def test_commit_add_node(self):
        ws, mgr = self._make_system()
        node = GoalNode(id="g1", description="Test goal")
        tx = HCIRTransaction(
            author="planner",
            operations=[
                TransactionOperation(
                    op=TransactionOp.ADD_NODE,
                    node_id="g1",
                    node_data=node.model_dump(),
                ),
            ],
        )
        result = mgr.commit(tx)
        assert result.is_committed
        assert ws.get_node("g1") is not None
        assert mgr.committed_count == 1

    def test_commit_remove_node(self):
        ws, mgr = self._make_system()
        ws.add_node(GoalNode(id="g1", description="to remove"))
        tx = HCIRTransaction(
            author="planner",
            operations=[
                TransactionOperation(op=TransactionOp.REMOVE_NODE, node_id="g1"),
            ],
        )
        result = mgr.commit(tx)
        assert result.is_committed
        assert ws.get_node("g1") is None

    def test_commit_modify_node(self):
        ws, mgr = self._make_system()
        ws.add_node(GoalNode(id="g1", description="v1", priority=0.3))
        tx = HCIRTransaction(
            author="planner",
            operations=[
                TransactionOperation(
                    op=TransactionOp.MODIFY_NODE,
                    node_id="g1",
                    changes={"description": "v2", "priority": 0.9},
                ),
            ],
        )
        result = mgr.commit(tx)
        assert result.is_committed
        node = ws.get_node("g1")
        assert isinstance(node, GoalNode)
        assert node.description == "v2"
        assert node.priority == 0.9

    def test_commit_upsert_node(self):
        ws, mgr = self._make_system()
        node = BeliefNode(id="b1", claim="Initial belief")
        tx = HCIRTransaction(
            author="reasoner",
            operations=[
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id="b1",
                    node_data=node.model_dump(),
                ),
            ],
        )
        result = mgr.commit(tx)
        assert result.is_committed
        assert ws.get_node("b1") is not None

    def test_commit_add_edge(self):
        ws, mgr = self._make_system()
        ws.add_node(GoalNode(id="g1", description="a"))
        ws.add_node(BeliefNode(id="b1", claim="b"))
        edge = HCIREdge(id="e1", edge_type=HCIREdgeType.SUPPORTS, sources=["b1"], targets=["g1"])
        tx = HCIRTransaction(
            author="planner",
            operations=[
                TransactionOperation(
                    op=TransactionOp.ADD_EDGE,
                    edge_id="e1",
                    edge_data=edge.model_dump(),
                ),
            ],
        )
        result = mgr.commit(tx)
        assert result.is_committed
        assert ws.get_edge("e1") is not None

    def test_commit_delta(self):
        ws, mgr = self._make_system()
        delta = HCIRDelta(
            add_nodes=[GoalNode(id="g1", description="delta goal").model_dump()],
        )
        result = mgr.commit_delta(delta, author="test_node")
        assert result.is_committed
        assert ws.get_node("g1") is not None

    def test_reject_already_committed(self):
        ws, mgr = self._make_system()
        tx = HCIRTransaction(author="test", status=TransactionStatus.COMMITTED)
        result = mgr.commit(tx)
        assert result.status == TransactionStatus.COMMITTED  # No-op

    def test_modify_nonexistent_node_rejects(self):
        ws, mgr = self._make_system()
        tx = HCIRTransaction(
            author="planner",
            operations=[
                TransactionOperation(
                    op=TransactionOp.MODIFY_NODE,
                    node_id="nonexistent",
                    changes={"description": "new"},
                ),
            ],
        )
        result = mgr.commit(tx)
        assert result.is_rejected

    def test_committed_transaction_log(self):
        ws, mgr = self._make_system()
        for i in range(3):
            tx = HCIRTransaction(
                author="planner",
                operations=[
                    TransactionOperation(
                        op=TransactionOp.ADD_NODE,
                        node_id=f"g{i}",
                        node_data=GoalNode(id=f"g{i}", description=f"goal_{i}").model_dump(),
                    ),
                ],
            )
            mgr.commit(tx)
        history = mgr.get_committed_transactions(limit=10)
        assert len(history) == 3
