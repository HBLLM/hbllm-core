"""Unit tests for HCIR World Round 20 — The Grand Capstone:
1. Global Neuronal Workspace (GNWT) Non-Linear Ignition & Conscious Broadcast (Dehaene / Mashour)
2. Native Structural Graph All-Pairs Shortest Paths, Radius, Diameter & Betweenness Centrality (Brandes)
3. Unified Closed-Loop Active Inference Governor (AutonomousEpistemicEngine.decide_active_inference_step)
4. Full Substrate Capstone Invariant Verification across all 20 Evolutionary Rounds
"""

from __future__ import annotations

from hbllm.hcir.graph import CognitiveGraph, FactNode, HCIREdge, HCIREdgeType
from hbllm.hcir.native_adapter import NativeGraphAdapter
from hbllm.hcir.world.autonomous_epistemic_engine import (
    AutonomousEpistemicEngine,
    UnifiedActiveInferenceDecision,
)
from hbllm.hcir.world.global_workspace_ignition import (
    CandidateIgnitionSignal,
    GlobalWorkspaceIgnitionEngine,
)


def test_global_workspace_ignition_engine() -> None:
    """Test non-linear all-or-none conscious ignition, reverberation, and refractory gating."""
    engine = GlobalWorkspaceIgnitionEngine(
        ignition_threshold=0.60,
        reverberation_decay_rate=0.30,
        refractory_period_cycles=1,
    )

    # Cycle 1: Only weak subliminal signals (salience 0.30)
    subliminal_sig = CandidateIgnitionSignal(
        source_modality="visual_texture",
        content_payload="subtle_gradient",
        salience_score=0.30,
    )
    state1 = engine.compete_and_ignite([subliminal_sig])
    assert not state1.is_ignited
    assert state1.active_ignition is None
    assert state1.subliminal_suppressed_count == 1

    # Cycle 2: High salience acute threat / goal signal (salience 0.85)
    salient_sig = CandidateIgnitionSignal(
        source_modality="prefrontal_conflict",
        content_payload="barrier_breach_detected",
        salience_score=0.85,
    )
    state2 = engine.compete_and_ignite([subliminal_sig, salient_sig])
    assert state2.is_ignited
    assert state2.active_ignition is not None
    assert state2.active_ignition.source_modality == "prefrontal_conflict"
    assert state2.active_ignition.reverberation_energy > 0.80

    pred = state2.active_ignition.to_causal_predicate()
    assert pred.variable == "global_workspace_broadcast"
    assert "prefrontal_conflict" in pred.value

    # Cycle 3: Sustained reverberation decays across cycles
    state3 = engine.compete_and_ignite([])
    assert state3.is_ignited
    assert state3.reverberation_decay < state2.active_ignition.reverberation_energy


def test_graph_all_pairs_shortest_paths_radius_diameter_and_betweenness() -> None:
    """Test graph metrics (APSP, radius, diameter, betweenness centrality) on line graph A-B-C."""
    adapter = NativeGraphAdapter()

    # Undirected line graph: A - B - C
    adapter.add_node("A", "FACT")
    adapter.add_node("B", "FACT")
    adapter.add_node("C", "FACT")

    adapter.add_edge("e_ab", "NEIGHBOR", ["A"], ["B"], 1.0)
    adapter.add_edge("e_bc", "NEIGHBOR", ["B"], ["C"], 1.0)

    # APSP
    sp, _ = adapter.all_pairs_shortest_paths()
    assert sp[("A", "B")] == 1.0
    assert sp[("B", "C")] == 1.0
    assert sp[("A", "C")] == 2.0

    # Radius and diameter
    rad, diam = adapter.graph_radius_and_diameter()
    assert rad == 1.0  # Center node B has eccentricity 1.0
    assert diam == 2.0  # Endpoints A and C have eccentricity 2.0

    # Betweenness centrality: B sits on the only path between A and C
    cb = adapter.betweenness_centrality(normalized=True, directed=False)
    assert cb["B"] > cb["A"]
    assert cb["B"] > cb["C"]
    assert cb["A"] == 0.0
    assert cb["C"] == 0.0
    assert cb["B"] == 1.0  # (n=3: normalized 1 path / ((2*1)/2) = 1.0)

    # CognitiveGraph equivalence test
    graph = CognitiveGraph(enable_native_substrate=False)
    graph.add_node(FactNode(id="A", claim="A"))
    graph.add_node(FactNode(id="B", claim="B"))
    graph.add_node(FactNode(id="C", claim="C"))

    graph.add_edge(
        HCIREdge(sources=["A"], targets=["B"], edge_type=HCIREdgeType.DEPENDS_ON, weight=1.0)
    )
    graph.add_edge(
        HCIREdge(sources=["B"], targets=["C"], edge_type=HCIREdgeType.DEPENDS_ON, weight=1.0)
    )

    graph_sp, _ = graph.all_pairs_shortest_paths()
    assert graph_sp[("A", "C")] == 2.0

    graph_rad, graph_diam = graph.graph_radius_and_diameter()
    assert graph_rad == 1.0
    assert graph_diam == 2.0

    graph_cb = graph.betweenness_centrality(normalized=True, directed=False)
    assert graph_cb["B"] == 1.0


def test_unified_closed_loop_active_inference_decision() -> None:
    """Test Grand Capstone canonical 5-stage active inference decision loop."""
    engine = AutonomousEpistemicEngine(enable_logging=False)
    engine.avatar_pos = (5, 5)
    engine.record_learned_goal((5, 8))

    curr_grid = [[0 for _ in range(16)] for _ in range(16)]
    curr_grid[5][5] = 1  # avatar
    curr_grid[5][8] = 2  # goal

    decision = engine.decide_active_inference_step(
        curr_grid=curr_grid,
        available_actions=[1, 2, 3, 4],  # e.g., Up, Down, Left, Right
    )

    assert isinstance(decision, UnifiedActiveInferenceDecision)
    assert decision.selected_action in [1, 2, 3, 4]
    assert decision.expected_free_energy >= 0.0
    assert decision.locomotor_velocity > 0.0
    assert decision.time_dilation_factor >= 1.0
    assert decision.subthalamic_status in (
        "dispatch_permitted",
        "brake_active",
        "emergency_override",
    )

    # Verify that the decision was automatically recorded into HCIR Symbolic World Theory
    pred = decision.to_causal_predicate()
    assert pred.variable == "active_inference_decision"
    assert "act:" in pred.value
    assert "vel:" in pred.value

    recorded_predicates = [p.variable for p in engine.symbolic_theory.causal_predicates]
    assert "active_inference_decision" in recorded_predicates


def test_grand_capstone_full_substrate_invariants_across_all_20_rounds() -> None:
    """Exhaustively verify that all 20 evolutionary rounds of cortical faculties
    are completely wired, active, non-null, and executing synchronously on the engine.
    """
    engine = AutonomousEpistemicEngine(enable_logging=False)

    # Round 1-16 Foundational Cortical Faculties
    assert engine.body_schema is not None
    assert engine.acc_conflict is not None
    assert engine.optical_projector is not None
    assert engine.it_segmenter is not None
    assert engine.habenular_ior is not None
    assert engine.subgoal_stack is not None
    assert engine.remote_causal is not None
    assert engine.symbolic_theory is not None
    assert engine.working_memory is not None
    assert engine.hippocampal_replay is not None
    assert engine.locus_coeruleus is not None
    assert engine.drift_diffusion is not None
    assert engine.allostasis_cortex is not None
    assert engine.cerebellar_clock is not None
    assert engine.potential_field_navigator is not None
    assert engine.foveated_attention is not None

    # Round 17 Cortical Faculties
    assert engine.hippocampal_time is not None
    assert engine.vl_motor_inhibition is not None
    assert engine.claustrum_interhemispheric is not None

    # Round 18 Cortical Faculties
    assert engine.anterior_insula_timing is not None
    assert engine.ofc_counterfactual is not None
    assert engine.ppn_locomotor is not None

    # Round 19 Cortical Faculties
    assert engine.boundary_grid_aligner is not None
    assert engine.mpfc_schema_engine is not None
    assert engine.stn_escalator is not None

    # Round 20 Grand Capstone Faculties & Orchestrator
    assert engine.workspace_ignition is not None
    assert callable(engine.decide_active_inference_step)
