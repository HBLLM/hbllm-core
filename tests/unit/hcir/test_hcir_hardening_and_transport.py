"""Unit tests for HCIR Runtime Hardening: Branch Isolation, Replay Debugger, and Distributed Delta Transport."""

import pytest

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.delta_transport import DeltaTransportProtocol
from hbllm.hcir.graph import GoalNode, PredictionErrorNode
from hbllm.hcir.kernel.capability_resolver import CapabilityResolver
from hbllm.hcir.kernel.scheduler import CognitiveScheduler
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.receipt import ExecutionReceipt
from hbllm.hcir.replay_debugger import ReplayDebugger
from hbllm.hcir.transactions import HCIRDelta
from hbllm.hcir.types import BranchMode
from hbllm.hcir.workspace import HCIRWorkspaceState

# ═══════════════════════════════════════════════════════════════════════════
# Branch Isolation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBranchIsolation:
    def test_workspace_branch_mode_live(self):
        ws = HCIRWorkspaceState()
        assert ws.branch_mode == BranchMode.LIVE

    def test_fork_branch_mode_simulation(self):
        ws = HCIRWorkspaceState()
        forked = ws.fork("sim_1", mode=BranchMode.SIMULATION)
        assert forked.branch_mode == BranchMode.SIMULATION

    def test_fork_branch_mode_replay(self):
        ws = HCIRWorkspaceState()
        forked = ws.fork("replay_1", mode=BranchMode.REPLAY)
        assert forked.branch_mode == BranchMode.REPLAY


# ═══════════════════════════════════════════════════════════════════════════
# Prediction Error Node Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPredictionErrorNode:
    def test_prediction_error_node_creation(self):
        node = PredictionErrorNode(
            id="pe_001",
            prediction_id="pred_42",
            predicted_value=25.0,
            observed_value=31.0,
            delta=6.0,
            error_magnitude=0.24,
            suspected_cause="solar gain underestimated",
        )
        assert node.node_type.value == "prediction_error"
        assert node.delta == 6.0
        assert node.suspected_cause == "solar gain underestimated"


# ═══════════════════════════════════════════════════════════════════════════
# Replay Debugger Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestReplayDebugger:
    @pytest.mark.asyncio
    async def test_replay_stream_step_by_step(self):
        ws = HCIRWorkspaceState()
        services = KernelServices(
            workspace=ws,
            transaction_manager=TransactionManager(ws),
            capability_resolver=CapabilityResolver(),
            scheduler=CognitiveScheduler(),
        )

        debugger = ReplayDebugger(services)

        stream = InstructionStream(
            author="tester",
            instructions=[
                Instruction(
                    opcode=Opcode.ASSERT,
                    params={
                        "node_data": GoalNode(
                            id="g_replay", description="replay test"
                        ).model_dump(),
                        "author": "tester",
                    },
                ),
                Instruction(
                    opcode=Opcode.QUERY,
                    params={"node_type": "goal"},
                ),
            ],
        )

        receipt = ExecutionReceipt(execution_id="rcpt_orig", success=True)

        steps = await debugger.replay_stream(stream, expected_receipt=receipt)

        assert len(steps) == 2
        assert steps[0].instruction.opcode == Opcode.ASSERT
        assert steps[1].instruction.opcode == Opcode.QUERY
        # Ensure main workspace was unaffected by replay branch execution
        assert ws.graph.node_count == 0


# ═══════════════════════════════════════════════════════════════════════════
# Distributed Delta Transport Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDeltaTransport:
    def test_create_and_sign_packet(self):
        transport = DeltaTransportProtocol(device_id="robot_01")
        delta = HCIRDelta(nodes_to_add=[GoalNode(id="g_remote", description="remote goal")])

        packet = transport.create_packet(delta, target_device_id="server_node")

        assert packet.source_device_id == "robot_01"
        assert packet.target_device_id == "server_node"
        assert len(packet.signature) == 16

    def test_verify_and_apply_remote_packet(self):
        transport_src = DeltaTransportProtocol(device_id="robot_01", secret_key="shared_secret")
        transport_tgt = DeltaTransportProtocol(device_id="server_node", secret_key="shared_secret")

        delta = HCIRDelta(nodes_to_add=[GoalNode(id="g_shared", description="shared goal")])
        packet = transport_src.create_packet(delta)

        target_ws = HCIRWorkspaceState()
        success = transport_tgt.verify_and_apply(packet.to_dict(), target_ws)

        assert success is True
