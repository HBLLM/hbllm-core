"""Unit tests for HCIR ExecutionReceipt, Scheduler Policy, Truth/Utility Separation, and Optimizer."""

import pytest

from hbllm.hcir.abi import ExecutionMetrics
from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.graph import (
    BeliefNode,
    CognitiveGraph,
    FactNode,
    GoalNode,
    HCIREdge,
    HCIREdgeType,
    ValueNode,
)
from hbllm.hcir.interpreter import HCIRInterpreter
from hbllm.hcir.kernel.capability_resolver import CapabilityResolver
from hbllm.hcir.kernel.scheduler import CognitiveScheduler
from hbllm.hcir.kernel.scheduler_policy import CognitiveScoreCalculator, TaskScoringFactors
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.optimizer import (
    CostPruningPass,
    DeadInstructionEliminationPass,
    HCIROptimizer,
    QueryMergingPass,
)
from hbllm.hcir.receipt import ExecutionReceipt, ReceiptStore
from hbllm.hcir.validation import GraphValidator, ValidationSeverity
from hbllm.hcir.workspace import HCIRWorkspaceState


# ═══════════════════════════════════════════════════════════════════════════
# ExecutionReceipt Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestExecutionReceipt:
    def test_receipt_creation_and_hash(self):
        receipt = ExecutionReceipt(
            author="test_author",
            instruction_stream_hash="abc123hash",
            input_snapshot_version=1,
            final_snapshot_version=2,
            transactions_committed=["tx_001", "tx_002"],
            capabilities_used=["causal_analysis"],
        )
        assert receipt.execution_id.startswith("rcpt_")
        checksum = receipt.compute_certificate_hash()
        assert len(checksum) == 16

    def test_receipt_store(self):
        store = ReceiptStore()
        receipt = ExecutionReceipt(author="author_1", process_id="proc_123")
        store.store(receipt)

        assert store.count == 1
        assert store.get(receipt.execution_id) is receipt
        assert len(store.list_by_author("author_1")) == 1
        assert len(store.list_by_process("proc_123")) == 1

    @pytest.mark.asyncio
    async def test_interpreter_execute_with_receipt(self):
        ws = HCIRWorkspaceState()
        tx_mgr = TransactionManager(ws)
        resolver = CapabilityResolver()
        scheduler = CognitiveScheduler()
        services = KernelServices(
            workspace=ws,
            transaction_manager=tx_mgr,
            capability_resolver=resolver,
            scheduler=scheduler,
        )
        interpreter = HCIRInterpreter(ws, services)

        stream = InstructionStream(
            author="planner",
            instructions=[
                Instruction(
                    opcode=Opcode.ASSERT,
                    params={
                        "node_data": GoalNode(id="g_receipt", description="test").model_dump(),
                        "author": "planner",
                    },
                )
            ],
        )

        res, receipt = await interpreter.execute_with_receipt(stream, process_id="p1", thread_id="t1")
        assert res.success is True
        assert receipt.process_id == "p1"
        assert receipt.thread_id == "t1"
        assert receipt.author == "planner"
        assert len(receipt.transactions_committed) == 1
        assert ws.get_node("g_receipt") is not None


# ═══════════════════════════════════════════════════════════════════════════
# Scheduler Policy Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSchedulerPolicy:
    def test_task_scoring_factors(self):
        factors = TaskScoringFactors(
            expected_value=0.8,
            urgency=0.9,
            confidence=0.95,
            resource_cost=1.5,
            interruption_cost=1.2,
        )
        # (0.8 * 0.9 * 0.95) / (1.5 * 1.2) = 0.684 / 1.8 = 0.38
        score = factors.compute_score()
        assert abs(score - 0.38) < 0.001

    def test_calculator_helper(self):
        score = CognitiveScoreCalculator.score_task(
            expected_value=1.0, urgency=1.0, confidence=1.0,
            resource_cost=1.0, interruption_cost=1.0
        )
        assert score == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Truth vs Utility Separation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTruthUtilitySeparation:
    def test_valid_graph_passes(self):
        graph = CognitiveGraph()
        fact = FactNode(id="f1", statement="Water boils at 100C")
        belief = BeliefNode(id="b1", claim="Kettle will boil soon")
        graph.add_node(fact)
        graph.add_node(belief)
        edge = HCIREdge(
            id="e1",
            edge_type=HCIREdgeType.SUPPORTS,
            sources=["f1"],
            targets=["b1"],
        )
        graph.add_edge(edge)

        validator = GraphValidator()
        report = validator.validate(graph)
        assert report.is_valid is True

    def test_value_node_corrupting_truth_node_fails(self):
        graph = CognitiveGraph()
        val_node = ValueNode(id="v1", dimension="cost_saving", weight=0.9)
        belief = BeliefNode(id="b1", claim="Cheap components are unbreakable")
        graph.add_node(val_node)
        graph.add_node(belief)

        # Invalid edge: Value directly CAUSES Belief
        edge = HCIREdge(
            id="e_invalid",
            edge_type=HCIREdgeType.CAUSES,
            sources=["v1"],
            targets=["b1"],
        )
        graph.add_edge(edge)

        validator = GraphValidator()
        report = validator.validate(graph)
        assert report.is_valid is False
        assert any(issue.code == "TRUTH_UTILITY_CORRUPTION" for issue in report.issues)


# ═══════════════════════════════════════════════════════════════════════════
# HCIROptimizer Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHCIROptimizer:
    def test_dead_instruction_elimination(self):
        pass_opt = DeadInstructionEliminationPass()
        stream = InstructionStream(
            author="test",
            instructions=[
                Instruction(opcode=Opcode.ASSERT, params={}),  # Empty - should be eliminated
                Instruction(
                    opcode=Opcode.ASSERT,
                    params={"node_data": {"id": "n1", "node_type": "belief"}},
                ),
            ],
        )
        opt_stream = pass_opt.run(stream)
        assert opt_stream.length == 1
        assert opt_stream.instructions[0].params["node_data"]["id"] == "n1"

    def test_query_merging(self):
        pass_opt = QueryMergingPass()
        query_ins = Instruction(opcode=Opcode.QUERY, params={"text_contains": "battery"})
        stream = InstructionStream(
            author="test",
            instructions=[query_ins, query_ins, query_ins],
        )
        opt_stream = pass_opt.run(stream)
        assert opt_stream.length == 1

    def test_cost_pruning(self):
        pass_opt = CostPruningPass(max_cost_budget=50)
        stream = InstructionStream(
            author="test",
            instructions=[
                Instruction(opcode=Opcode.QUERY, cost_estimate=20),
                Instruction(opcode=Opcode.QUERY, cost_estimate=20),
                Instruction(opcode=Opcode.EXECUTE, cost_estimate=100),  # Exceeds 50 total limit
            ],
        )
        opt_stream = pass_opt.run(stream)
        assert opt_stream.length == 2

    def test_full_optimizer_pipeline(self):
        optimizer = HCIROptimizer([
            DeadInstructionEliminationPass(),
            QueryMergingPass(),
        ])
        stream = InstructionStream(
            author="test",
            instructions=[
                Instruction(opcode=Opcode.ASSERT, params={}),  # Dead
                Instruction(opcode=Opcode.QUERY, params={"text": "x"}),
                Instruction(opcode=Opcode.QUERY, params={"text": "x"}),  # Duplicate
            ],
        )
        opt_stream = optimizer.optimize(stream)
        assert opt_stream.length == 1
        assert opt_stream.instructions[0].opcode == Opcode.QUERY
