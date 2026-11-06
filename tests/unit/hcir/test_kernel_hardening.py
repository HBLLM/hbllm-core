"""Kernel hardening tests — atomicity, lossless event sourcing, ISA MERGE/ROLLBACK,
verifier single-pass, scope isolation on ADD_NODE, and event bus publication."""

from __future__ import annotations

import pytest

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.graph import BeliefNode, GoalNode, HCIREdge, HCIREdgeType
from hbllm.hcir.interpreter import HCIRInterpreter
from hbllm.hcir.kernel.capability_resolver import CapabilityResolver
from hbllm.hcir.kernel.event_bus import KernelEventBus
from hbllm.hcir.kernel.governance.constitutional_verifier import ConstitutionalVerifier
from hbllm.hcir.kernel.scheduler import KernelInstructionScheduler
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.stores import EventType
from hbllm.hcir.transactions import HCIRTransaction, TransactionOp, TransactionOperation
from hbllm.hcir.types import Scope
from hbllm.hcir.workspace import HCIRWorkspaceState


def _add(node) -> TransactionOperation:
    return TransactionOperation(op=TransactionOp.ADD_NODE, node_data=node.model_dump())


def _system():
    ws = HCIRWorkspaceState()
    bus = KernelEventBus()
    mgr = TransactionManager(ws, event_bus=bus)
    services = KernelServices(
        workspace=ws,
        transaction_manager=mgr,
        capability_resolver=CapabilityResolver(),
        scheduler=KernelInstructionScheduler(),
        event_bus=bus,
    )
    return ws, mgr, services, HCIRInterpreter(ws, services)


class TestAtomicity:
    def test_failed_op_leaves_graph_unchanged(self):
        ws = HCIRWorkspaceState()
        mgr = TransactionManager(ws)
        before_hash = ws.graph.content_hash()
        before_version = ws.current_version
        tx = HCIRTransaction(
            author="t",
            operations=[
                _add(GoalNode(id="g1", description="a")),
                _add(GoalNode(id="g2", description="b")),
                TransactionOperation(
                    op=TransactionOp.MODIFY_NODE, node_id="missing", changes={"x": 1}
                ),
            ],
        )
        res = mgr.commit(tx)
        assert res.is_rejected
        assert ws.get_node("g1") is None
        assert ws.get_node("g2") is None
        assert ws.graph.content_hash() == before_hash
        assert ws.current_version == before_version
        assert mgr.committed_count == 0

    def test_failed_op_restores_previous_node_value(self):
        ws = HCIRWorkspaceState()
        mgr = TransactionManager(ws)
        mgr.commit(
            HCIRTransaction(author="t", operations=[_add(GoalNode(id="g1", description="v1"))])
        )
        tx = HCIRTransaction(
            author="t",
            operations=[
                TransactionOperation(
                    op=TransactionOp.MODIFY_NODE, node_id="g1", changes={"description": "v2"}
                ),
                TransactionOperation(op=TransactionOp.REMOVE_NODE, node_id="g1"),
                TransactionOperation(
                    op=TransactionOp.ADD_EDGE,
                    edge_data=HCIREdge(
                        id="e_bad",
                        edge_type=HCIREdgeType.SUPPORTS,
                        sources=["nope"],
                        targets=["g1"],
                    ).model_dump(),
                ),
            ],
        )
        res = mgr.commit(tx)
        assert res.is_rejected
        g1 = ws.get_node("g1")
        assert g1 is not None and g1.description == "v1"


class TestLosslessEventSourcing:
    def test_modify_event_carries_full_payload(self):
        ws = HCIRWorkspaceState()
        mgr = TransactionManager(ws)
        mgr.commit(
            HCIRTransaction(author="t", operations=[_add(GoalNode(id="g1", description="v1"))])
        )
        mgr.commit(
            HCIRTransaction(
                author="t",
                operations=[
                    TransactionOperation(
                        op=TransactionOp.MODIFY_NODE, node_id="g1", changes={"description": "v2"}
                    )
                ],
            )
        )
        events = [e for e in ws.event_store.get_events() if e.event_type == EventType.NODE_MODIFIED]
        assert events, "expected a NODE_MODIFIED event"
        assert events[-1].data["node_data"]["description"] == "v2"

    def test_snapshot_hash_reflects_payload_changes(self):
        ws = HCIRWorkspaceState()
        mgr = TransactionManager(ws)
        mgr.commit(
            HCIRTransaction(author="t", operations=[_add(GoalNode(id="g1", description="v1"))])
        )
        h1 = ws.snapshot_manager.get_snapshot(ws.current_version).content_hash
        mgr.commit(
            HCIRTransaction(
                author="t",
                operations=[
                    TransactionOperation(
                        op=TransactionOp.MODIFY_NODE, node_id="g1", changes={"description": "v2"}
                    )
                ],
            )
        )
        h2 = ws.snapshot_manager.get_snapshot(ws.current_version).content_hash
        assert h1 != h2


class TestVerifierSinglePass:
    def test_commit_with_verification_runs_stages_once(self):
        ws = HCIRWorkspaceState()
        verifier = ConstitutionalVerifier()
        mgr = TransactionManager(ws, verification_stages=[verifier])
        tx = HCIRTransaction(author="t", operations=[_add(GoalNode(id="g1", description="x"))])
        ok, res, comp = mgr.commit_with_verification(tx)
        assert ok and res.is_committed and comp is None
        assert verifier.evaluations == 1


class TestScopeIsolationOnAdd:
    def test_tenant_cannot_add_node_into_foreign_scope(self):
        ws = HCIRWorkspaceState()
        mgr = TransactionManager(ws, verification_stages=[ConstitutionalVerifier()])
        node = GoalNode(id="g_b", description="x", scope=Scope(tenant_id="tenant_b"))
        tx = HCIRTransaction(
            author="t", operations=[_add(node)], metadata={"tenant_id": "tenant_a"}
        )
        res = mgr.commit(tx)
        assert res.is_rejected
        assert ws.get_node("g_b") is None

    def test_tenant_can_add_node_into_own_scope(self):
        ws = HCIRWorkspaceState()
        mgr = TransactionManager(ws, verification_stages=[ConstitutionalVerifier()])
        node = GoalNode(id="g_a", description="x", scope=Scope(tenant_id="tenant_a"))
        tx = HCIRTransaction(
            author="t", operations=[_add(node)], metadata={"tenant_id": "tenant_a"}
        )
        assert mgr.commit(tx).is_committed


class TestEventBusPublication:
    def test_commit_publishes_transaction_committed(self):
        ws, mgr, services, _ = _system()
        seen: list[str] = []
        services.event_bus.subscribe("transaction.*", lambda e: seen.append(e.event_type))
        mgr.commit(
            HCIRTransaction(author="t", operations=[_add(GoalNode(id="g1", description="x"))])
        )
        assert "transaction.committed" in seen

    def test_rejection_publishes_transaction_rejected(self):
        ws, mgr, services, _ = _system()
        seen: list[str] = []
        services.event_bus.subscribe("transaction.rejected", lambda e: seen.append(e.event_type))
        mgr.commit(
            HCIRTransaction(
                author="t",
                operations=[
                    TransactionOperation(
                        op=TransactionOp.MODIFY_NODE, node_id="nope", changes={"a": 1}
                    )
                ],
            )
        )
        assert seen == ["transaction.rejected"]

    @pytest.mark.asyncio
    async def test_wait_without_event_is_not_fake_resolved(self):
        _, _, _, interp = _system()
        stream = InstructionStream(
            instructions=[Instruction(opcode=Opcode.WAIT, params={"timeout_ms": 10})]
        )
        res = await interp.execute(stream)
        assert res.events[0]["results"][0]["resolved"] is False


class TestIsaMergeRollback:
    @pytest.mark.asyncio
    async def test_merge_goes_through_transaction_manager_and_propagates_removals(self):
        ws, mgr, services, interp = _system()
        mgr.commit(
            HCIRTransaction(
                author="t",
                operations=[
                    _add(GoalNode(id="keep", description="k")),
                    _add(GoalNode(id="drop", description="d")),
                ],
            )
        )
        committed_before = mgr.committed_count
        await interp.execute(
            InstructionStream(
                instructions=[Instruction(opcode=Opcode.FORK, params={"branch_name": "sim"})]
            )
        )
        branch = ws.get_branch("sim")
        TransactionManager(branch).commit(
            HCIRTransaction(
                author="sim",
                operations=[
                    _add(BeliefNode(id="new", claim="n")),
                    TransactionOperation(op=TransactionOp.REMOVE_NODE, node_id="drop"),
                    TransactionOperation(
                        op=TransactionOp.MODIFY_NODE, node_id="keep", changes={"description": "k2"}
                    ),
                ],
            )
        )
        res = await interp.execute(
            InstructionStream(
                instructions=[Instruction(opcode=Opcode.MERGE, params={"branch_name": "sim"})]
            )
        )
        assert res.success
        out = res.events[0]["results"][0]
        assert out["merged"] and out["merged_count"] == 3
        assert mgr.committed_count == committed_before + 1
        assert ws.get_node("new") is not None
        assert ws.get_node("drop") is None
        assert ws.get_node("keep").description == "k2"
        assert mgr.committed_log[-1].author == "merge"

    @pytest.mark.asyncio
    async def test_merge_rejected_by_verifier_leaves_main_untouched(self):
        ws, mgr, services, interp = _system()
        mgr.add_verification_stage(ConstitutionalVerifier(max_operations_per_tx=1))
        await interp.execute(
            InstructionStream(
                instructions=[Instruction(opcode=Opcode.FORK, params={"branch_name": "sim"})]
            )
        )
        branch = ws.get_branch("sim")
        TransactionManager(branch).commit(
            HCIRTransaction(
                author="sim",
                operations=[
                    _add(GoalNode(id="a", description="a")),
                    _add(GoalNode(id="b", description="b")),
                ],
            )
        )
        res = await interp.execute(
            InstructionStream(
                instructions=[Instruction(opcode=Opcode.MERGE, params={"branch_name": "sim"})]
            )
        )
        out = res.events[0]["results"][0]
        assert "error" in out
        assert ws.get_node("a") is None and ws.get_node("b") is None

    @pytest.mark.asyncio
    async def test_rollback_restores_prior_snapshot_state(self):
        ws, mgr, services, interp = _system()
        mgr.commit(
            HCIRTransaction(author="t", operations=[_add(GoalNode(id="g1", description="v1"))])
        )
        target = ws.current_version
        mgr.commit(
            HCIRTransaction(
                author="t",
                operations=[
                    _add(GoalNode(id="g2", description="later")),
                    TransactionOperation(
                        op=TransactionOp.MODIFY_NODE, node_id="g1", changes={"description": "v2"}
                    ),
                ],
            )
        )
        mgr.commit(
            HCIRTransaction(
                author="t",
                operations=[TransactionOperation(op=TransactionOp.REMOVE_NODE, node_id="g1")],
            )
        )
        assert ws.get_node("g1") is None
        res = await interp.execute(
            InstructionStream(
                instructions=[
                    Instruction(opcode=Opcode.ROLLBACK, params={"target_version": target})
                ]
            )
        )
        out = res.events[0]["results"][0]
        assert out["rolled_back"] is True
        assert ws.get_node("g2") is None
        g1 = ws.get_node("g1")
        assert g1 is not None and g1.description == "v1"
        assert out["transactions_reverted"] == 2


class TestBoundedLogs:
    def test_logs_are_bounded(self):
        ws = HCIRWorkspaceState()
        mgr = TransactionManager(ws, max_log_size=5)
        for i in range(12):
            mgr.commit(
                HCIRTransaction(
                    author="t", operations=[_add(GoalNode(id=f"g{i}", description="x"))]
                )
            )
        assert mgr.committed_count == 5
        assert len(mgr._undo_logs) <= 5
