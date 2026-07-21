"""Unit tests for HCIR Bytecode, Interpreter, and SyscallDispatcher."""

import pytest

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.graph import GoalNode, HCIREdge, HCIREdgeType, HCIRNodeType
from hbllm.hcir.interpreter import HCIRInterpreter, SyscallDispatcher
from hbllm.hcir.kernel.capability_resolver import (
    CapabilityImplementation,
    CapabilityResolver,
)
from hbllm.hcir.kernel.scheduler import CognitiveScheduler
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.workspace import HCIRWorkspaceState


# ═══════════════════════════════════════════════════════════════════════════
# Bytecode Model Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBytecode:
    def test_instruction_creation(self):
        ins = Instruction(opcode=Opcode.ASSERT, params={"node_data": {"id": "g1"}})
        assert ins.opcode == Opcode.ASSERT
        assert "node_data" in ins.params

    def test_instruction_repr(self):
        ins = Instruction(opcode=Opcode.QUERY, params={"node_type": "goal"})
        assert "QUERY" in repr(ins)

    def test_instruction_stream(self):
        stream = InstructionStream(author="planner")
        stream.append(Instruction(opcode=Opcode.ASSERT, cost_estimate=10))
        stream.append(Instruction(opcode=Opcode.QUERY, cost_estimate=5))
        assert stream.length == 2
        assert stream.total_cost_estimate == 15

    def test_all_opcodes_exist(self):
        assert len(Opcode) == 8
        expected = {"ASSERT", "RETRACT", "QUERY", "EXECUTE", "WAIT", "FORK", "MERGE", "ROLLBACK"}
        assert {o.value for o in Opcode} == expected


# ═══════════════════════════════════════════════════════════════════════════
# Syscall Dispatcher Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSyscallDispatcher:
    def test_default_handlers_registered(self):
        dispatcher = SyscallDispatcher()
        for opcode in Opcode:
            assert opcode in dispatcher._handlers

    @pytest.mark.asyncio
    async def test_dispatch_unknown_opcode_raises(self):
        dispatcher = SyscallDispatcher()
        dispatcher._handlers.clear()
        ins = Instruction(opcode=Opcode.ASSERT)
        ws = HCIRWorkspaceState()
        mgr = TransactionManager(ws)
        resolver = CapabilityResolver()
        scheduler = CognitiveScheduler()
        services = KernelServices(
            workspace=ws,
            transaction_manager=mgr,
            capability_resolver=resolver,
            scheduler=scheduler,
        )
        with pytest.raises(ValueError, match="No syscall handler"):
            await dispatcher.dispatch(ins, ws, services)


# ═══════════════════════════════════════════════════════════════════════════
# Interpreter Integration Tests
# ═══════════════════════════════════════════════════════════════════════════


def _make_system():
    """Create a full kernel system for testing."""
    ws = HCIRWorkspaceState()
    mgr = TransactionManager(ws)
    resolver = CapabilityResolver()
    scheduler = CognitiveScheduler()
    services = KernelServices(
        workspace=ws,
        transaction_manager=mgr,
        capability_resolver=resolver,
        scheduler=scheduler,
    )
    interpreter = HCIRInterpreter(ws, services)
    return ws, services, interpreter


class TestInterpreterAssert:
    @pytest.mark.asyncio
    async def test_assert_node(self):
        ws, services, interpreter = _make_system()
        node = GoalNode(id="g1", description="Test goal")
        stream = InstructionStream(instructions=[
            Instruction(
                opcode=Opcode.ASSERT,
                params={"node_data": node.model_dump(), "author": "test"},
            ),
        ])
        result = await interpreter.execute(stream)
        assert result.success
        assert ws.get_node("g1") is not None

    @pytest.mark.asyncio
    async def test_assert_edge(self):
        ws, services, interpreter = _make_system()
        ws.add_node(GoalNode(id="g1", description="a"))
        ws.add_node(GoalNode(id="g2", description="b"))
        edge = HCIREdge(id="e1", edge_type=HCIREdgeType.DEPENDS_ON, sources=["g1"], targets=["g2"])
        stream = InstructionStream(instructions=[
            Instruction(
                opcode=Opcode.ASSERT,
                params={"edge_data": edge.model_dump()},
            ),
        ])
        result = await interpreter.execute(stream)
        assert result.success
        assert ws.get_edge("e1") is not None


class TestInterpreterRetract:
    @pytest.mark.asyncio
    async def test_retract_node(self):
        ws, services, interpreter = _make_system()
        ws.add_node(GoalNode(id="g1", description="to remove"))
        stream = InstructionStream(instructions=[
            Instruction(opcode=Opcode.RETRACT, params={"node_id": "g1"}),
        ])
        result = await interpreter.execute(stream)
        assert result.success
        assert ws.get_node("g1") is None


class TestInterpreterQuery:
    @pytest.mark.asyncio
    async def test_query_by_type(self):
        ws, services, interpreter = _make_system()
        ws.add_node(GoalNode(id="g1", description="a"))
        ws.add_node(GoalNode(id="g2", description="b"))
        stream = InstructionStream(instructions=[
            Instruction(
                opcode=Opcode.QUERY,
                params={"node_type": "goal"},
            ),
        ])
        result = await interpreter.execute(stream)
        assert result.success
        event_data = result.events[0]["results"][0]
        assert event_data["total"] == 2


class TestInterpreterForkMerge:
    @pytest.mark.asyncio
    async def test_fork_and_merge(self):
        ws, services, interpreter = _make_system()
        ws.add_node(GoalNode(id="g1", description="main"))

        # Fork
        fork_stream = InstructionStream(instructions=[
            Instruction(opcode=Opcode.FORK, params={"branch_name": "sim_1"}),
        ])
        result = await interpreter.execute(fork_stream)
        assert result.success

        # Add a node to the forked branch
        branch = ws.get_branch("sim_1")
        assert branch is not None
        branch.add_node(GoalNode(id="g2", description="sim goal"))

        # Merge
        merge_stream = InstructionStream(instructions=[
            Instruction(opcode=Opcode.MERGE, params={"branch_name": "sim_1"}),
        ])
        result = await interpreter.execute(merge_stream)
        assert result.success
        assert ws.get_node("g2") is not None
        assert ws.get_branch("sim_1") is None  # Dropped after merge


class TestInterpreterMultiInstruction:
    @pytest.mark.asyncio
    async def test_multi_instruction_stream(self):
        """Test a complete cognitive cycle: ASSERT → QUERY → RETRACT."""
        ws, services, interpreter = _make_system()
        node = GoalNode(id="g1", description="Multi-step test")
        stream = InstructionStream(instructions=[
            Instruction(
                opcode=Opcode.ASSERT,
                params={"node_data": node.model_dump()},
                cost_estimate=10,
            ),
            Instruction(
                opcode=Opcode.QUERY,
                params={"node_type": "goal"},
                cost_estimate=5,
            ),
            Instruction(
                opcode=Opcode.RETRACT,
                params={"node_id": "g1"},
                cost_estimate=5,
            ),
        ])
        result = await interpreter.execute(stream)
        assert result.success
        assert result.metrics.tokens_consumed == 20
        assert result.metrics.elapsed_ms >= 0
        assert ws.get_node("g1") is None  # Retracted
