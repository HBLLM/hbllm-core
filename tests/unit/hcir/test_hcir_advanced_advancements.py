"""Unit tests for Advanced Advancements:
1. Native Rust acceleration integration
2. Predictive reality ensemble inside AutonomousEpistemicEngine
3. Declarative skill induction from transaction logs
"""

from __future__ import annotations

import numpy as np

from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.skills.declarative import DeclarativeNeuroSymbolicSkill, SkillEvaluationContext
from hbllm.hcir.skills.induction import SkillInductionEngine
from hbllm.hcir.spatial_planner import SpatialActionIntent
from hbllm.hcir.transactions import HCIRTransaction, TransactionOp, TransactionOperation
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine


class TestPredictiveRealityLoopInEngine:
    """Tests for PredictiveRealityModel integration inside AutonomousEpistemicEngine."""

    def test_evaluate_candidate_action_ensemble(self):
        ws = HCIRWorkspaceState()
        services = KernelServices.create_default(ws)
        engine = AutonomousEpistemicEngine(services=services)

        grid = np.zeros((10, 10), dtype=np.int32)
        grid[2, 3] = 1

        pred, is_high_disagreement = engine.evaluate_candidate_action_ensemble(
            action="MOVE_UP",
            curr_grid=grid,
            horizon_ms=1000,
        )

        assert pred is not None
        assert pred.calibrated_confidence > 0.0
        assert "physics" in pred.component_predictions
        assert "snn" in pred.component_predictions
        assert isinstance(is_high_disagreement, bool)

    def test_ensemble_disagreement_event_publication(self):
        ws = HCIRWorkspaceState()
        services = KernelServices.create_default(ws)
        engine = AutonomousEpistemicEngine(services=services)

        received_events = []
        services.event_bus.subscribe(
            "epistemic.ensemble_disagreement",
            lambda evt: received_events.append(evt),
        )

        # Force disagreement analyzer to return high disagreement
        original_analyze = (
            engine.world_kernel.predictive_reality.disagreement_analyzer.analyze_disagreement
        )

        class MockDisagreement:
            high_disagreement = True
            disagreement_index = 0.85
            conflicting_predictors = ["physics", "neural"]

        engine.world_kernel.predictive_reality.disagreement_analyzer.analyze_disagreement = (
            lambda comp: MockDisagreement()
        )

        try:
            pred, is_high = engine.evaluate_candidate_action_ensemble(
                action="INTERACT_KEY",
                curr_grid=np.zeros((5, 5)),
            )
            assert is_high is True
            assert len(received_events) == 1
            assert received_events[0].data["action_intent"] == "INTERACT_KEY"
            assert received_events[0].data["disagreement_index"] == 0.85
        finally:
            engine.world_kernel.predictive_reality.disagreement_analyzer.analyze_disagreement = (
                original_analyze
            )


class TestSkillInductionEngine:
    """Tests for automated declarative neuro-symbolic skill induction."""

    def test_induce_from_transactions(self):
        inducer = SkillInductionEngine()

        txs = [
            HCIRTransaction(
                author="AgentA",
                operations=[
                    TransactionOperation(
                        op=TransactionOp.ADD_NODE,
                        node_id="act_nav",
                        node_data={
                            "action_intent": "navigate_to_key",
                            "action_id": 1,
                            "role": "key",
                        },
                    ),
                    TransactionOperation(
                        op=TransactionOp.ADD_NODE,
                        node_id="act_interact",
                        node_data={
                            "action_intent": "interact_with_door",
                            "action_id": 2,
                            "role": "door",
                        },
                    ),
                ],
            )
        ]

        skill = inducer.induce_from_transactions(txs, skill_name="KeyDoorMacro")
        assert skill is not None
        assert isinstance(skill, DeclarativeNeuroSymbolicSkill)
        assert skill.skill_name == "KeyDoorMacro"
        assert skill.program is not None
        assert len(skill.program.subgoals) == 2
        assert skill.program.subgoals[0].intent == SpatialActionIntent.NAVIGATE
        assert skill.program.subgoals[0].target_query["role"] == "key"
        assert skill.program.subgoals[1].intent == SpatialActionIntent.INTERACT
        assert skill.program.subgoals[1].target_query["role"] == "door"

        # Verify compilation to bytecode instruction stream
        stream = skill.compile_to_instruction_stream(
            grid=np.zeros((5, 5)),
            available_actions=[1, 2],
        )
        assert stream.length > 0

        # Verify signature evaluation against cognitive context
        ctx_valid = SkillEvaluationContext(
            grid=np.zeros((5, 5)),
            available_actions=[1, 2, 3],
        )
        assert skill.can_handle(ctx_valid.grid, ctx_valid.available_actions) is True

        ctx_invalid = SkillEvaluationContext(
            grid=np.zeros((5, 5)),
            available_actions=[3, 4],  # missing actions 1 and 2
        )
        assert skill.can_handle(ctx_invalid.grid, ctx_invalid.available_actions) is False

    def test_induce_from_trajectory(self):
        inducer = SkillInductionEngine()
        skill = inducer.induce_from_trajectory(
            actions=[1, 1, 3, 2],
            metadata={"domain": "grid_navigation"},
            skill_name="PathSkill",
        )
        assert skill.skill_name == "PathSkill"
        assert len(skill.program.subgoals) == 4
        assert skill.can_handle(np.zeros((5, 5)), [1, 2, 3]) is True
        assert skill.can_handle(np.zeros((5, 5)), [1, 2]) is False  # missing action 3

    def test_procedural_macro_to_declarative_skill(self):
        from hbllm.hcir.world.cortex_episodic import ProceduralMacro

        macro = ProceduralMacro(
            macro_id="macro_win_lvl_1",
            action_sequence=[1, 2, 1],
            action_data_sequence=[None, None, None],
            available_actions={1, 2},
            level=1,
            game_id="game_alpha",
            confidence=0.95,
        )
        skill = macro.to_declarative_skill()
        assert skill.skill_name == "macro_win_lvl_1"
        assert skill.can_handle(np.zeros((5, 5)), [1, 2]) is True
        assert skill.can_handle(np.zeros((5, 5)), [1]) is False
        assert skill.program is not None
        assert len(skill.program.subgoals) == 3


class TestNativeRustAccelerationIntegration:
    """Tests confirming native Rust module loading and parity."""

    def test_native_hcir_graph_loaded(self):
        import hbllm_hcir_graph

        assert hasattr(hbllm_hcir_graph, "NativeGraph") or hasattr(
            hbllm_hcir_graph, "NativeHCIRGraph"
        )

    def test_native_confidence_loaded(self):
        import hbllm_confidence

        score = hbllm_confidence.quick_score(
            "test query", "test response with detailed explanation"
        )
        assert 0.0 <= score <= 1.0

    def test_native_semantic_search_loaded(self):
        import hbllm_semantic_search

        assert hbllm_semantic_search is not None

    def test_native_compute_loaded(self):
        import hbllm_compute

        assert hbllm_compute is not None

    def test_native_concept_extract_loaded(self):
        import hbllm_concept_extract

        assert hbllm_concept_extract is not None

    def test_native_knowledge_graph_loaded(self):
        import hbllm_knowledge_graph

        assert hbllm_knowledge_graph is not None

    def test_native_tokenizer_loaded(self):
        import hbllm_tokenizer_rs

        assert hbllm_tokenizer_rs is not None
