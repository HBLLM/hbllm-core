"""Unit tests for BranchModeVerifier, Cognitive Breakpoints, Consensus Engine, and Attention Graph."""


from hbllm.hcir.bytecode import Instruction, Opcode
from hbllm.hcir.graph import BeliefNode, GoalNode, PredictionErrorNode
from hbllm.hcir.kernel.attention_graph import AttentionManager
from hbllm.hcir.kernel.consensus_engine import CandidateBelief, CognitiveConsensusEngine
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.kernel.verification import BranchModeVerifier
from hbllm.hcir.replay_debugger import CognitiveBreakpoint
from hbllm.hcir.transactions import HCIRTransaction
from hbllm.hcir.types import BranchMode
from hbllm.hcir.workspace import HCIRWorkspaceState

# ═══════════════════════════════════════════════════════════════════════════
# BranchModeVerifier Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBranchModeVerifier:
    def test_replay_mode_rejects_transaction(self):
        ws = HCIRWorkspaceState(branch_mode=BranchMode.REPLAY)
        verifier = BranchModeVerifier()
        tx = HCIRTransaction(author="test", operations=[])

        assert verifier.verify(tx, ws) is False
        assert any("REPLAY mode" in a.assertion for a in tx.annotations)

    def test_live_mode_passes(self):
        ws = HCIRWorkspaceState(branch_mode=BranchMode.LIVE)
        verifier = BranchModeVerifier()
        tx = HCIRTransaction(author="test", operations=[])

        assert verifier.verify(tx, ws) is True


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive Breakpoints Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCognitiveBreakpoint:
    def test_breakpoint_matching(self):
        bp = CognitiveBreakpoint(opcode="ASSERT", min_cost=10)
        ins1 = Instruction(opcode=Opcode.ASSERT, cost_estimate=15)
        ins2 = Instruction(opcode=Opcode.QUERY, cost_estimate=15)
        ins3 = Instruction(opcode=Opcode.ASSERT, cost_estimate=5)

        assert bp.matches(ins1) is True
        assert bp.matches(ins2) is False
        assert bp.matches(ins3) is False


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive Consensus Engine Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestConsensusEngine:
    def test_arbitrate_beliefs(self):
        ws = HCIRWorkspaceState()
        tx_mgr = TransactionManager(ws)
        engine = CognitiveConsensusEngine(ws, tx_mgr)

        b1 = BeliefNode(id="b1", claim="humidity_70")
        b1.uncertainty.confidence = 0.6

        b2 = BeliefNode(id="b2", claim="humidity_62")
        b2.uncertainty.confidence = 0.95

        c1 = CandidateBelief(source_id="device_a", belief_node=b1, source_trust=0.7)
        c2 = CandidateBelief(source_id="device_b", belief_node=b2, source_trust=1.0)

        winning_belief = engine.arbitrate_beliefs([c1, c2])

        assert winning_belief is not None
        assert winning_belief.id == "b2"
        assert ws.get_node("b2") is not None


# ═══════════════════════════════════════════════════════════════════════════
# Attention Manager & Graph Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAttentionManager:
    def test_recompute_attention_allocates_goals_and_surprise(self):
        ws = HCIRWorkspaceState()
        ws.add_node(GoalNode(id="g1", description="Yield optimization", priority=0.9))
        ws.add_node(PredictionErrorNode(
            id="pe1", prediction_id="pred1", error_magnitude=0.8, suspected_cause="sensor_drift"
        ))

        attn_mgr = AttentionManager(ws)
        attn_graph = attn_mgr.recompute_attention()

        assert len(attn_graph.focus_nodes) == 2
        assert "g1" in attn_graph.focus_nodes
        assert "pe1" in attn_graph.focus_nodes
        assert attn_graph.focus_nodes["pe1"].focus_reason == "prediction_error_surprise"
        assert ws.attention.salience > 0.5
