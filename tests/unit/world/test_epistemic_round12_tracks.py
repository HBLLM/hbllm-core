"""Unit test suite for HCIR World Round 12 Cognitive Faculties & Native Substrates.

Covers:
- Track 1: Cerebellar Forward Dynamics & Microzone LTD Error Adaptation (Marr-Albus-Ito)
- Track 2: Ventral Premotor (F5 / Mirror Neuron) Affordance Resonator (Rizzolatti)
- Track 3: Anterior Cingulate Cortex (ACC) Effort Allocator & EVC Gating (Botvinick / Shenhav)
- Track 4: Native Structural Graph Directed Maximum Flow & Minimum s-t Cut Bottlenecks (Edmonds-Karp)
"""

from __future__ import annotations

import pytest

from hbllm.hcir import algorithms
from hbllm.hcir.graph import CognitiveGraph, FactNode, HCIREdge, HCIREdgeType
from hbllm.hcir.native_adapter import NativeGraphAdapter
from hbllm.hcir.world.autonomous_epistemic_engine import (
    AutonomousEpistemicEngine,
)
from hbllm.hcir.world.experimental import restore_all as _restore_all
from hbllm.hcir.world.experimental.anterior_cingulate_effort_allocator import (
    AnteriorCingulateEffortAllocator,
    ConflictProfile,
    EffortAllocationDecision,
)
from hbllm.hcir.world.experimental.cerebellar_microzone_adaptor import (
    CerebellarMicrozoneAdaptor,
    ClimbingFiberFeedback,
    MotorPrediction,
)
from hbllm.hcir.world.experimental.ventral_premotor_resonator import (
    VentralPremotorAffordanceResonator,
)

# ============================================================================
# Track 1: Cerebellar Microzone Granule-Purkinje LTD Tests
# ============================================================================


def test_cerebellar_microzone_feedforward_prediction_and_ltd() -> None:
    """Test cerebellar Purkinje forward model prediction and climbing fiber LTD adaptation."""
    microzone = CerebellarMicrozoneAdaptor(
        granule_dimension=128,
        top_k_active_granules=16,
        learning_rate_ltd=0.15,
    )

    state = {"r": 5.0, "c": 5.0, "vr": 1.0, "vc": 0.0}
    action = 1  # DOWN

    # 1. Initial prediction
    pred1: MotorPrediction = microzone.predict_outcome(action_id=action, current_state=state)
    assert pred1.active_granule_count == 16
    assert pred1.predicted_latency_ms > 0.0

    # 2. Provide ground truth motor outcome with substantial difference
    actual_disp = (2.5, 0.5)
    feedback: ClimbingFiberFeedback = microzone.adapt_from_feedback(
        action_id=action,
        current_state=state,
        actual_displacement=actual_disp,
        actual_latency_ms=15.0,
    )

    assert feedback.error_magnitude > 0.0
    assert feedback.weights_depressed_count == 16

    # 3. Second prediction: Purkinje readout must move toward actual outcome
    pred2: MotorPrediction = microzone.predict_outcome(action_id=action, current_state=state)
    err_before = (actual_disp[0] - pred1.predicted_displacement[0]) ** 2
    err_after = (actual_disp[0] - pred2.predicted_displacement[0]) ** 2
    assert err_after < err_before


# ============================================================================
# Track 2: Ventral Premotor (F5) Affordance Resonator Tests
# ============================================================================


def test_ventral_premotor_affordance_resonance() -> None:
    """Test vicarious affordance extraction from observing external entity interactions."""
    resonator = VentralPremotorAffordanceResonator(resonance_threshold=0.50)

    # Observe a patroller moving into a door mechanism and toggling its state
    pattern = resonator.observe_interaction(
        entity_id="patroller_9",
        entity_role="patroller",
        initial_pos=(4, 5),
        final_pos=(4, 6),
        target_entity_id="door_mechanism",
        target_role="door_switch",
        state_mutation={"is_open": True},
    )

    assert pattern is not None
    assert pattern.inferred_action_type == "interact"
    assert pattern.resonance_score >= 0.80

    # Verify vicarious causal inference was generated
    predicates = resonator.get_vicarious_predicates()
    assert len(predicates) == 1
    assert "interact" in predicates[0].value
    assert predicates[0].target_entity == "door_mechanism"


# ============================================================================
# Track 3: Anterior Cingulate Cortex (ACC) Effort Allocation Tests
# ============================================================================


def test_anterior_cingulate_conflict_and_effort_allocation() -> None:
    """Test ACC conflict monitoring and Expected Value of Control effort allocation."""
    allocator = AnteriorCingulateEffortAllocator(
        base_depth=2,
        max_depth=6,
        base_simulations=5,
        max_simulations=25,
        conflict_threshold_habit=0.20,
    )

    # 1. Low conflict scenario: clear dominant action, no hazard -> habitual execution
    low_conflict_vals = {0: 10.0, 1: 1.0, 2: 0.5, 3: 0.2}
    profile_low: ConflictProfile = allocator.compute_conflict_profile(
        action_values=low_conflict_vals,
        hazard_distance=10.0,
        state_familiarity=1.0,
    )
    decision_low: EffortAllocationDecision = allocator.allocate_effort(profile_low)

    assert decision_low.is_habitual is True
    assert decision_low.search_depth == 1
    assert decision_low.simulation_budget == 0
    assert decision_low.effort_level <= 0.2

    # 2. High conflict scenario: close competing values, imminent hazard -> deep deliberation
    high_conflict_vals = {0: 5.0, 1: 4.9, 2: 4.8}
    profile_high: ConflictProfile = allocator.compute_conflict_profile(
        action_values=high_conflict_vals,
        hazard_distance=1.2,
        state_familiarity=0.2,
    )
    decision_high: EffortAllocationDecision = allocator.allocate_effort(profile_high)

    assert decision_high.is_habitual is False
    assert decision_high.search_depth >= 4
    assert decision_high.simulation_budget >= 15
    assert decision_high.effort_level > 0.6


# ============================================================================
# Track 4: Native Structural Graph Maximum Flow & Minimum Cut Tests
# ============================================================================


def test_native_graph_adapter_max_flow_min_cut() -> None:
    """Edmonds-Karp on NativeGraphAdapter computes exact max flow and min-cut bottleneck."""
    adapter = NativeGraphAdapter()

    # Flow network:
    # S -> A (cap 3), S -> B (cap 2)
    # A -> B (cap 1), A -> T (cap 2)
    # B -> T (cap 3)
    nodes = ["S", "A", "B", "T"]
    for n in nodes:
        adapter.add_node(node_id=n, node_type="concept")

    adapter.add_edge("e_sa", "flow", sources=["S"], targets=["A"], weight=3.0)
    adapter.add_edge("e_sb", "flow", sources=["S"], targets=["B"], weight=2.0)
    adapter.add_edge("e_ab", "flow", sources=["A"], targets=["B"], weight=1.0)
    adapter.add_edge("e_at", "flow", sources=["A"], targets=["T"], weight=2.0)
    adapter.add_edge("e_bt", "flow", sources=["B"], targets=["T"], weight=3.0)

    max_flow, cut_edges, source_set, sink_set = adapter.max_flow_min_cut(
        source="S", sink="T", capacity_attribute="weight"
    )

    # Max flow in this network is exactly 5.0
    assert max_flow == pytest.approx(5.0)

    # S in source_set, T in sink_set
    assert "S" in source_set
    assert "T" in sink_set
    assert source_set.isdisjoint(sink_set)

    # Min cut capacity must match max flow (5.0)
    cut_capacity = sum(
        adapter.get_edge(eid).get("weight", 1.0)
        for (u, v) in cut_edges
        for eid in ["e_sa", "e_sb", "e_ab", "e_at", "e_bt"]
        if adapter.get_edge(eid).get("sources") == [u]
        and adapter.get_edge(eid).get("targets") == [v]
    )
    assert cut_capacity == pytest.approx(5.0)


def test_cognitive_graph_max_flow_parity() -> None:
    """CognitiveGraph implements identical Edmonds-Karp max flow & min cutset computation."""
    graph = CognitiveGraph(enable_native_substrate=False)

    for nid in ["Source", "Mid", "Sink"]:
        graph.add_node(FactNode(id=nid, claim=nid))

    e1 = HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["Source"], targets=["Mid"])
    e1.weight = 7.0  # type: ignore[attr-defined]
    e2 = HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["Mid"], targets=["Sink"])
    e2.weight = 4.0  # type: ignore[attr-defined]

    graph.add_edge(e1)
    graph.add_edge(e2)

    max_flow, cut_edges, source_set, sink_set = algorithms.max_flow_min_cut(
        graph, source="Source", sink="Sink", capacity_attr="weight"
    )

    # Bottleneck is 4.0 on e2
    assert max_flow == pytest.approx(4.0)
    assert ("Mid", "Sink") in cut_edges
    assert "Source" in source_set
    assert "Sink" in sink_set


# ============================================================================
# Orchestrator Integration Tests: AutonomousEpistemicEngine Round 12
# ============================================================================


def test_epistemic_engine_round12_integration() -> None:
    """Test AutonomousEpistemicEngine delegation to Round 12 faculties."""
    engine = AutonomousEpistemicEngine()

    _restore_all(engine)

    # 1. Test cerebellar motor outcome adaptation
    feedback = engine.adapt_cerebellar_motor_outcome(
        action_id=0,
        state={"r": 2.0, "c": 3.0, "vr": 0.0, "vc": 1.0},
        actual_displacement=(-1.0, 0.0),
        actual_latency_ms=12.0,
    )
    assert feedback.weights_depressed_count > 0
    assert any(
        p.variable == "cerebellar_climbing_error" for p in engine.symbolic_theory.causal_predicates
    )

    # 2. Test mirror affordance interaction observation
    pattern = engine.observe_external_affordance_interaction(
        entity_id="entity_agent_2",
        entity_role="patroller",
        initial_pos=(1, 1),
        final_pos=(1, 2),
        target_id="barrier_lock",
        target_role="barrier_switch",
        state_mutation={"locked": False},
    )
    assert pattern is not None
    assert any(
        p.relation == "vicarious_mirror_resonance" for p in engine.symbolic_theory.causal_predicates
    )

    # 3. Test ACC cognitive deliberation effort allocation
    decision = engine.allocate_cognitive_deliberation_effort(
        action_values={0: 3.0, 1: 2.95},
        hazard_distance=1.0,
    )
    assert decision.search_depth >= 3
    assert any(
        p.variable == "cognitive_effort_allocation"
        for p in engine.symbolic_theory.causal_predicates
    )

    # 4. Test network max-flow/min-cut on cognitive graph
    graph = CognitiveGraph(enable_native_substrate=False)
    graph.add_node(FactNode(id="Start", claim="Start"))
    graph.add_node(FactNode(id="Goal", claim="Goal"))
    e = HCIREdge(id="e_direct", edge_type=HCIREdgeType.CAUSES, sources=["Start"], targets=["Goal"])
    e.weight = 3.5  # type: ignore[attr-defined]
    graph.add_edge(e)

    flow_val, cuts, s_set, t_set = engine.compute_network_max_flow_min_cut(
        source="Start", sink="Goal", graph=graph, capacity_attribute="weight"
    )
    assert flow_val == pytest.approx(3.5)
    assert ("Start", "Goal") in cuts
