"""Unit tests for HCIR Round 18 Cognitive Substrate Enhancements.

Tracks Tested:
1. Track 1: Anterior Insular Cortex (AIC) Subjective Time Dilation & Interoceptive Pacing (anterior_insular_timing.py)
2. Track 2: Orbitofrontal Cortex (OFC) Counterfactual Credit Assignment & Inferred State Value Updating (orbitofrontal_credit_assignment.py)
3. Track 3: Pedunculopontine Nucleus (PPN / MLR) Locomotor Velocity Modulation & Obstacle Deceleration (pedunculopontine_locomotor.py)
4. Track 4: Native Structural Graph Feedback Vertex Set (FVS) & Cycle Transversal Discovery (native_adapter.py & graph.py)
5. Track 5: Autonomous Epistemic Engine Round 18 Integrated Orchestration
"""

from __future__ import annotations

import pytest

from hbllm.hcir.graph import (
    CognitiveGraph,
    FactNode,
    HCIREdge,
    HCIREdgeType,
)
from hbllm.hcir.native_adapter import NativeGraphAdapter
from hbllm.hcir.world.anterior_insular_timing import (
    AnteriorInsularTimingEngine,
    PacingMode,
)
from hbllm.hcir.world.autonomous_epistemic_engine import (
    AutonomousEpistemicEngine,
)
from hbllm.hcir.world.cortex_causal import (
    HCIRSymbolicWorldTheory,
)
from hbllm.hcir.world.experimental import restore_all as _restore_all
from hbllm.hcir.world.orbitofrontal_credit_assignment import (
    OrbitofrontalCreditAssignmentEngine,
)
from hbllm.hcir.world.pedunculopontine_locomotor import (
    LocomotorPacingRegime,
    PedunculopontineLocomotorEngine,
)


class TestAnteriorInsularTiming:
    """Track 1: Anterior insular subjective time dilation and interoceptive pacing."""

    def test_subjective_time_dilation_and_pacing_modes(self) -> None:
        engine = AnteriorInsularTimingEngine(
            base_pacemaker_hz=10.0,
            threat_gain=1.5,
            salience_gain=1.0,
            max_dilation=4.0,
        )

        # 1. Baseline peaceful state: routine slight compression or nominal
        p_base = engine.compute_subjective_dilation(
            objective_dt=1.0, threat_level=0.0, salience_score=0.0
        )
        assert p_base.dilation_factor <= 1.0
        assert p_base.subjective_dt <= 1.0
        assert p_base.pacing_mode in (PacingMode.ROUTINE_COMPRESSED, PacingMode.CALIBRATED_NOMINAL)

        # 2. Acute threat dilation: slow-motion expansion
        p_threat = engine.compute_subjective_dilation(
            objective_dt=1.0,
            threat_level=0.95,
            visceral_conflict=0.80,
            salience_score=0.90,
        )
        assert p_threat.dilation_factor >= 2.50
        assert p_threat.subjective_dt >= 2.50
        assert p_threat.pacing_mode == PacingMode.ACUTE_HYPER_DILATED

        # 3. Pacemaker pulse accumulation
        # Pulse accumulator should have advanced by (base_hz * dilation * dt)
        assert p_threat.pulse_accumulator > 0.0

        # 4. Deliberation budget expansion
        base_budget = 10
        scaled_budget = engine.calibrate_deliberation_budget(base_rollouts=base_budget)
        assert scaled_budget >= int(base_budget * 2.5)

        # 5. HCIR CausalPredicate serialization
        pred = p_threat.to_causal_predicate()
        assert pred.variable == "anterior_insula_subjective_timing"
        assert "dilation:" in str(pred.value)


class TestOrbitofrontalCreditAssignment:
    """Track 2: OFC counterfactual value updates and unchosen branch credit propagation."""

    def test_counterfactual_credit_and_regret_computation(self) -> None:
        engine = OrbitofrontalCreditAssignmentEngine(
            learning_rate=0.40,
            counterfactual_generalization=0.60,
            regret_threshold=0.30,
        )

        actions = ["move_left", "move_right", "jump"]
        similarities = {
            "move_right": 0.80,  # Highly similar spatial traversal
            "jump": 0.10,  # Dissimilar vertical action
        }

        # 1. Experience negative outcome on chosen action (e.g. wall collision or trap)
        report_neg = engine.update_contingent_credit(
            state_key="room_3_obstacle",
            chosen_action="move_left",
            observed_outcome=-1.0,
            available_actions=actions,
            relational_similarities=similarities,
        )

        # Chosen action must be directly penalized
        assert report_neg.chosen_updated_value < 0.0

        # Counterfactual update: "move_right" (high similarity) must receive negative transfer penalty
        branch_map = {b.action_id: b for b in report_neg.branch_updates}
        assert "move_right" in branch_map
        assert branch_map["move_right"].updated_value < 0.0
        assert branch_map["move_right"].counterfactual_delta < 0.0

        # "jump" has very low similarity, so its counterfactual delta is minimal
        assert abs(branch_map["jump"].counterfactual_delta) < abs(
            branch_map["move_right"].counterfactual_delta
        )

        # Regret should be positive since jump (0.0) is superior to move_left (< 0)
        assert report_neg.counterfactual_regret > 0.0
        assert report_neg.best_alternative_action == "jump"
        assert engine.is_branch_switch_indicated(report_neg) is True

        # 2. CausalPredicate serialization
        pred = report_neg.to_causal_predicate()
        assert pred.variable == "orbitofrontal_counterfactual_credit"
        assert "regret:" in str(pred.value)


class TestPedunculopontineLocomotor:
    """Track 3: Pedunculopontine graded velocity modulation and anticipatory proximity deceleration."""

    def test_locomotor_velocity_and_deceleration_curves(self) -> None:
        engine = PedunculopontineLocomotorEngine(
            nominal_velocity=1.0,
            min_velocity=0.20,
            braking_distance=3.0,
        )

        # 1. Open space, clear terrain -> Normal cruise
        cmd_open = engine.modulate_locomotor_velocity(
            distance_to_obstacle=10.0,
            surface_friction=1.0,
        )
        assert cmd_open.velocity == pytest.approx(1.0, abs=1e-2)
        assert cmd_open.pacing_regime == LocomotorPacingRegime.NORMAL_CRUISE

        # 2. Urgent sprint in open space -> Accelerated burst
        cmd_burst = engine.modulate_locomotor_velocity(
            distance_to_obstacle=10.0,
            surface_friction=1.0,
            urgency_score=0.80,
        )
        assert cmd_burst.velocity > 1.0
        assert cmd_burst.pacing_regime == LocomotorPacingRegime.ACCELERATED_BURST

        # 3. Proximity deceleration: obstacle within braking distance (1.5 < 3.0)
        cmd_brake = engine.modulate_locomotor_velocity(
            distance_to_obstacle=1.5,
            surface_friction=1.0,
        )
        assert cmd_brake.velocity < 1.0
        assert cmd_brake.velocity >= 0.20
        assert cmd_brake.pacing_regime == LocomotorPacingRegime.DECELERATING_BRAKE
        assert "proximity_braking" in cmd_brake.deceleration_reason

        # 4. Low friction (ice slip): clamped to prevent traction loss
        cmd_ice = engine.modulate_locomotor_velocity(
            distance_to_obstacle=10.0,
            surface_friction=0.10,
        )
        assert cmd_ice.velocity < 0.60
        assert cmd_ice.pacing_regime == LocomotorPacingRegime.CREEP_CAUTIOUS

        # 5. CausalPredicate serialization
        pred = cmd_brake.to_causal_predicate()
        assert pred.variable == "ppn_locomotor_drive"
        assert "vel:" in str(pred.value)


class TestStructuralGraphFeedbackVertexSet:
    """Track 4: Feedback Vertex Set (FVS) & Cycle Transversal in Native Adapter & CognitiveGraph."""

    def test_feedback_vertex_set_acyclic(self) -> None:
        graph = CognitiveGraph(enable_native_substrate=False)
        for n in ["A", "B", "C"]:
            graph.add_node(FactNode(id=n, claim=n))
        graph.add_edge(
            HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["B"])
        )
        graph.add_edge(
            HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["B"], targets=["C"])
        )

        # Acyclic directed path: FVS should be empty
        assert graph.feedback_vertex_set() == set()

        adapter = NativeGraphAdapter()
        for n in ["A", "B", "C"]:
            adapter.add_node(n, node_type="concept")
        adapter.add_edge("e1", "causes", ["A"], ["B"])
        adapter.add_edge("e2", "causes", ["B"], ["C"])
        assert adapter.feedback_vertex_set() == set()

    def test_feedback_vertex_set_single_and_figure8_cycles(self) -> None:
        # 1. Simple 3-node cycle: A -> B -> C -> A
        graph_cycle = CognitiveGraph(enable_native_substrate=False)
        for n in ["A", "B", "C"]:
            graph_cycle.add_node(FactNode(id=n, claim=n))
        graph_cycle.add_edge(
            HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["B"])
        )
        graph_cycle.add_edge(
            HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["B"], targets=["C"])
        )
        graph_cycle.add_edge(
            HCIREdge(id="e3", edge_type=HCIREdgeType.CAUSES, sources=["C"], targets=["A"])
        )

        fvs_cycle = graph_cycle.feedback_vertex_set()
        assert len(fvs_cycle) == 1
        assert list(fvs_cycle)[0] in {"A", "B", "C"}

        # 2. Figure-8 with shared hub node A:
        # Cycle 1: A -> B -> A
        # Cycle 2: A -> C -> A
        # Removing node A breaks BOTH cycles simultaneously -> minimal FVS = {A}!
        graph_fig8 = CognitiveGraph(enable_native_substrate=False)
        for n in ["A", "B", "C"]:
            graph_fig8.add_node(FactNode(id=n, claim=n))
        graph_fig8.add_edge(
            HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["B"])
        )
        graph_fig8.add_edge(
            HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["B"], targets=["A"])
        )
        graph_fig8.add_edge(
            HCIREdge(id="e3", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["C"])
        )
        graph_fig8.add_edge(
            HCIREdge(id="e4", edge_type=HCIREdgeType.CAUSES, sources=["C"], targets=["A"])
        )

        fvs_fig8 = graph_fig8.feedback_vertex_set()
        assert fvs_fig8 == {"A"}

        # Parity check on NativeGraphAdapter
        adapter_fig8 = NativeGraphAdapter()
        for n in ["A", "B", "C"]:
            adapter_fig8.add_node(n, node_type="concept")
        adapter_fig8.add_edge("e1", "causes", ["A"], ["B"])
        adapter_fig8.add_edge("e2", "causes", ["B"], ["A"])
        adapter_fig8.add_edge("e3", "causes", ["A"], ["C"])
        adapter_fig8.add_edge("e4", "causes", ["C"], ["A"])

        fvs_adapter = adapter_fig8.feedback_vertex_set()
        assert fvs_adapter == {"A"}


class TestAutonomousEpistemicEngineRound18Orchestration:
    """Track 5: Integrated AutonomousEpistemicEngine Round 18 faculties."""

    def test_engine_round18_orchestration(self) -> None:
        engine = AutonomousEpistemicEngine()

        _restore_all(engine)
        engine.symbolic_theory = HCIRSymbolicWorldTheory()

        # 1. Subjective Time Dilation
        time_state = engine.compute_subjective_time_dilation(
            objective_dt=0.5,
            threat_level=0.7,
            visceral_conflict=0.5,
        )
        assert time_state.dilation_factor > 1.0
        assert any(
            p.variable == "anterior_insula_subjective_timing"
            for p in engine.symbolic_theory.causal_predicates
        )

        # 2. Counterfactual Credit Assignment
        cf_report = engine.evaluate_counterfactual_credit(
            state_key="intersection_gamma",
            chosen_action="east",
            observed_outcome=-0.8,
            available_actions=["east", "west", "north"],
            relational_similarities={"west": 0.7, "north": 0.2},
        )
        assert cf_report.chosen_updated_value < 0.0
        assert any(
            p.variable == "orbitofrontal_counterfactual_credit"
            for p in engine.symbolic_theory.causal_predicates
        )

        # 3. Locomotor Velocity Modulation
        loco_cmd = engine.modulate_locomotor_velocity(
            distance_to_obstacle=1.2,
            surface_friction=0.9,
        )
        assert loco_cmd.pacing_regime == LocomotorPacingRegime.DECELERATING_BRAKE
        assert any(
            p.variable == "ppn_locomotor_drive" for p in engine.symbolic_theory.causal_predicates
        )

        # 4. Graph Feedback Vertex Set
        graph = CognitiveGraph(enable_native_substrate=False)
        for n in ["U", "V", "W"]:
            graph.add_node(FactNode(id=n, claim=n))
        graph.add_edge(
            HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["U"], targets=["V"])
        )
        graph.add_edge(
            HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["V"], targets=["W"])
        )
        graph.add_edge(
            HCIREdge(id="e3", edge_type=HCIREdgeType.CAUSES, sources=["W"], targets=["U"])
        )

        fvs = engine.compute_graph_feedback_vertex_set(graph)
        assert len(fvs) == 1
        assert list(fvs)[0] in {"U", "V", "W"}
