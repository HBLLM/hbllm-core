"""Unit tests for Advanced Advancements:
Track A: Curricular Skill Graph & Subgoal Decomposition (BA10)
Track B: Structural Causal Model (SCM) & Pearl's do(a) Interventions
Track D: Cerebellar Temporal SNN Pacing & Hazard Forecasting
"""

from __future__ import annotations

import numpy as np

from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.skills.curriculum import (
    CurricularSkillGraph,
    SkillTier,
)
from hbllm.hcir.skills.declarative import (
    ActionAffordancePredicate,
    DeclarativeNeuroSymbolicSkill,
    SubgoalSequence,
    SymbolicSubgoal,
)
from hbllm.hcir.spatial_planner import SpatialActionIntent
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.causal_scm import CounterfactualOutcome, StructuralCausalModel
from hbllm.hcir.world.cerebellar_phase_clock import CerebellarPhaseClock, PhaseGateDecision
from hbllm.hcir.world.frontopolar_subgoal_stack import (
    FrontopolarSubgoal,
    FrontopolarSubgoalStack,
    SubgoalType,
)
from hbllm.hcir.world.predictors.snn import SNNTemporalPredictor


class TestCurricularSkillGraphAndSubgoalDecomposition:
    """Tests for Track A: Curricular Skill Graph and Frontopolar Subgoal Decomposition."""

    def test_curricular_skill_registration_and_query(self):
        graph = CurricularSkillGraph()

        # 1. Register micro-skill (move/navigate)
        micro_skill = DeclarativeNeuroSymbolicSkill()
        micro_skill.skill_name = "MicroNavigate"
        micro_skill.signature = ActionAffordancePredicate(required={1, 2})
        micro_skill.program = SubgoalSequence(
            SymbolicSubgoal(intent=SpatialActionIntent.NAVIGATE, target_query={"role": "corridor"})
        )
        graph.register_skill(micro_skill, tier=SkillTier.MICRO, target_roles={"corridor"})

        # 2. Register meso-skill (key acquisition)
        meso_skill = DeclarativeNeuroSymbolicSkill()
        meso_skill.skill_name = "AcquireKey"
        meso_skill.signature = ActionAffordancePredicate(required={1, 2, 3})
        meso_skill.program = SubgoalSequence(
            SymbolicSubgoal(intent=SpatialActionIntent.NAVIGATE, target_query={"role": "key"}),
            SymbolicSubgoal(intent=SpatialActionIntent.INTERACT, target_query={"role": "key"}),
        )
        graph.register_skill(
            meso_skill,
            tier=SkillTier.MESO,
            dependencies={"MicroNavigate"},
            target_roles={"key"},
        )

        grid = np.zeros((10, 10))
        # Query with actions {1, 2}
        matches_micro = graph.find_matching_skills(grid, available_actions={1, 2})
        assert len(matches_micro) == 1
        assert matches_micro[0].skill_name == "MicroNavigate"

        # Query with actions {1, 2, 3} targeting "key"
        matches_key = graph.find_matching_skills(
            grid, available_actions={1, 2, 3}, target_role="key"
        )
        assert len(matches_key) == 1
        assert matches_key[0].skill_name == "AcquireKey"
        assert matches_key[0].tier == SkillTier.MESO

    def test_macro_skill_composition_and_decomposition(self):
        graph = CurricularSkillGraph()

        skill_a = DeclarativeNeuroSymbolicSkill()
        skill_a.skill_name = "KeySkill"
        skill_a.signature = ActionAffordancePredicate(required={1})
        skill_a.program = SubgoalSequence(
            SymbolicSubgoal(intent=SpatialActionIntent.NAVIGATE, target_query={"role": "key"})
        )
        graph.register_skill(skill_a, tier=SkillTier.MESO, target_roles={"key"})

        skill_b = DeclarativeNeuroSymbolicSkill()
        skill_b.skill_name = "DoorSkill"
        skill_b.signature = ActionAffordancePredicate(required={2})
        skill_b.program = SubgoalSequence(
            SymbolicSubgoal(intent=SpatialActionIntent.ACTUATE, target_query={"role": "door"})
        )
        graph.register_skill(skill_b, tier=SkillTier.MESO, target_roles={"door"})

        macro = graph.compose_macro_skill("SolveLevelMacro", ["KeySkill", "DoorSkill"])
        assert macro is not None
        assert macro.skill_name == "SolveLevelMacro"
        assert len(macro.program.subgoals) == 2

        # Decompose
        children = graph.decompose_macro_skill("SolveLevelMacro")
        assert len(children) == 2
        assert [c.skill_name for c in children] == ["KeySkill", "DoorSkill"]

    def test_frontopolar_subgoal_skill_binding_and_decomposition(self):
        stack = FrontopolarSubgoalStack(max_depth=8)

        parent_sg = FrontopolarSubgoal(
            subgoal_id="sg_master_0",
            subgoal_type=SubgoalType.DELIVER_TO_GOAL,
            target_entity_pos=(1, 1),
            target_destination=(8, 8),
            priority=1.0,
        )
        stack.push_subgoal(parent_sg)

        # Build declarative skill with 2 subgoals
        skill = DeclarativeNeuroSymbolicSkill()
        skill.skill_name = "TwoStagePush"
        skill.program = SubgoalSequence(
            SymbolicSubgoal(
                intent=SpatialActionIntent.NAVIGATE,
                target_query={"role": "barrier", "pos": (3, 3)},
            ),
            SymbolicSubgoal(
                intent=SpatialActionIntent.ACTUATE,
                target_query={"role": "goal", "pos": (8, 8)},
            ),
        )

        parent_sg.bind_skill(skill)
        assert parent_sg.bound_skill_name == "TwoStagePush"

        # Decompose in stack
        decomposed = stack.decompose_subgoal_with_skill("sg_master_0", skill)
        assert len(decomposed) == 2
        assert decomposed[0].subgoal_type == SubgoalType.CLEAR_OBSTACLE
        assert decomposed[0].target_entity_pos == (3, 3)
        assert decomposed[1].subgoal_type == SubgoalType.DELIVER_TO_GOAL
        assert decomposed[1].target_destination == (8, 8)
        assert stack.depth == 2


class TestStructuralCausalModelAndInterventions:
    """Tests for Track B: Pearl's SCM, do-calculus, and counterfactuals."""

    def test_pearl_graph_surgery_and_do_operator(self):
        scm = StructuralCausalModel(model_id="test_grid_scm")

        # Define structural equations:
        # switch -> door_open -> path_clear
        scm.add_variable(
            name="switch",
            parents=[],
            structural_func=lambda p, u: 0,
        )
        scm.add_variable(
            name="door_open",
            parents=["switch"],
            structural_func=lambda p, u: 1 if p.get("switch") == 1 else 0,
        )
        scm.add_variable(
            name="path_clear",
            parents=["door_open"],
            structural_func=lambda p, u: 1 if p.get("door_open") == 1 else 0,
        )

        # Observational baseline: switch=0 -> door_open=0 -> path_clear=0
        obs = scm.evaluate()
        assert obs["switch"] == 0
        assert obs["door_open"] == 0
        assert obs["path_clear"] == 0

        # Pearl's do-intervention: do(switch = 1)
        intervened = scm.do({"switch": 1})
        res = intervened.evaluate()
        assert res["switch"] == 1
        assert res["door_open"] == 1
        assert res["path_clear"] == 1

        # Direct graph surgery on intermediate variable: do(door_open = 1) even if switch=0
        surgery = scm.do({"door_open": 1})
        res_surgery = surgery.evaluate()
        assert res_surgery["door_open"] == 1
        assert res_surgery["path_clear"] == 1

    def test_counterfactual_query_and_pn_ps_metrics(self):
        scm = StructuralCausalModel()
        scm.add_variable(
            name="action",
            parents=[],
            structural_func=lambda p, u: "WAIT",
        )
        scm.add_variable(
            name="deadlock",
            parents=["action"],
            structural_func=lambda p, u: 1 if p.get("action") == "PUSH_INTO_CORNER" else 0,
        )

        # Factual evidence: action="WAIT", deadlock=0
        evidence = {"action": "WAIT", "deadlock": 0}

        # Counterfactual query: What if action had been PUSH_INTO_CORNER?
        cf = scm.evaluate_counterfactual(
            factual_evidence=evidence,
            interventions={"action": "PUSH_INTO_CORNER"},
            target_variables=["deadlock"],
        )
        assert isinstance(cf, CounterfactualOutcome)
        assert cf.counterfactual_values["deadlock"] == 1
        assert cf.causal_effect == 1.0
        assert cf.probability_of_necessity == 1.0

    def test_synthesize_safety_invariants(self):
        scm = StructuralCausalModel()
        scm.add_variable("action", structural_func=lambda p, u: "NONE")
        scm.add_variable(
            "deadlock",
            parents=["action"],
            structural_func=lambda p, u: True if p.get("action") == "BAD_ACTION" else False,
        )

        invariants = scm.synthesize_safety_invariants(
            action_space=["SAFE_ACTION", "BAD_ACTION"],
            current_state={"action": "SAFE_ACTION", "deadlock": False},
            hazard_variable="deadlock",
        )
        assert len(invariants) == 1
        assert invariants[0].target_entity == "BAD_ACTION"
        assert invariants[0].operator == "!="


class TestCerebellarPhaseClockAndSNNPrediction:
    """Tests for Track D: SNN LIF Dynamics, limit cycles, and cerebellar crossing safety."""

    def test_snn_lif_neuron_spiking(self):
        snn = SNNTemporalPredictor(v_thresh=1.0, tau_m=5.0)

        # Input below threshold
        sub_input = np.full((3, 3), 0.5, dtype=np.float32)
        spikes_1, v_1 = snn.step_lif(sub_input)
        assert np.all(spikes_1 == 0)
        assert np.all(v_1 > 0.0)

        # Strong input exceeding threshold
        strong_input = np.full((3, 3), 2.5, dtype=np.float32)
        spikes_2, v_2 = snn.step_lif(strong_input)
        assert np.all(spikes_2 == 1)
        # Should reset membrane potential
        assert np.all(v_2 == snn.v_reset)

    def test_snn_hazard_density_projection(self):
        snn = SNNTemporalPredictor()

        grid_t0 = np.zeros((6, 6), dtype=np.int32)
        grid_t0[2, 2] = 5  # Moving patroller feature 5
        snn.update_tracks_from_grid(grid_t0)

        grid_t1 = np.zeros((6, 6), dtype=np.int32)
        grid_t1[2, 3] = 5  # Moved right
        snn.update_tracks_from_grid(grid_t1)

        assert 5 in snn.entity_tracks
        assert snn.entity_tracks[5].velocity == (0, 1)

        # Forecast hazard density 2 steps forward
        density = snn.predict_hazard_density(grid_t1, horizon_steps=2)
        assert density.shape == (6, 6)
        # Predicted positions: (2, 4) at step 1, (2, 5) at step 2
        assert density[2, 4] > 0.0 or density[2, 5] > 0.0

    def test_cerebellar_crossing_safety_with_snn(self):
        clock = CerebellarPhaseClock()
        snn = SNNTemporalPredictor()

        # Simulate patroller moving directly toward target coordinate (2, 4)
        grid_t0 = np.zeros((5, 5), dtype=np.int32)
        grid_t0[2, 2] = 9
        snn.update_tracks_from_grid(grid_t0)

        grid_t1 = np.zeros((5, 5), dtype=np.int32)
        grid_t1[2, 3] = 9
        snn.update_tracks_from_grid(grid_t1)

        # Agent is at (2, 0), wants to move to (2, 4).
        # Distance = 4 steps. SNN forecasts target will be occupied/hazardous.
        decision = clock.evaluate_crossing_safety(
            agent_pos=(2, 0),
            target_pos=(2, 4),
            snn_predictor=snn,
            grid=grid_t1,
            current_step=1,
        )
        assert isinstance(decision, PhaseGateDecision)

    def test_autonomous_epistemic_engine_transit_safety(self):
        ws = HCIRWorkspaceState()
        services = KernelServices.create_default(ws)
        engine = AutonomousEpistemicEngine(services=services)

        grid = np.zeros((6, 6), dtype=np.int32)
        grid[1, 1] = 1  # avatar
        engine.avatar_pos = (1, 1)

        decision = engine.evaluate_transit_safety_cerebellar_snn(
            target_pos=(1, 4),
            curr_grid=grid,
        )
        assert isinstance(decision, PhaseGateDecision)
        assert decision.should_wait is False  # Clear grid -> safe to proceed
