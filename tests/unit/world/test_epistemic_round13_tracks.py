"""Unit test suite for HCIR World Round 13 Cognitive Faculties & Native Substrates.

Covers:
- Track 1: Subicular Boundary Vector Cell (BVC) Map (Hartley-Burgess)
- Track 2: Ventromedial Prefrontal Cortex (vmPFC) Somatic Marker Guidance (Damasio-Bechara)
- Track 3: Pulvinar Thalamic Transthalamic Routing & Attentional Gating (Saalmann-Kastner)
- Track 4: Native Structural Graph Feedback Arc Set (FAS) & Cycle Severing (Eades et al.)
- Engine Orchestration Integration: AutonomousEpistemicEngine Round 13 faculties
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hbllm.hcir import algorithms
from hbllm.hcir.graph import CognitiveGraph, FactNode, HCIREdge, HCIREdgeType
from hbllm.hcir.native_adapter import NativeGraphAdapter
from hbllm.hcir.world.autonomous_epistemic_engine import (
    AutonomousEpistemicEngine,
)
from hbllm.hcir.world.experimental import restore_all as _restore_all
from hbllm.hcir.world.experimental.pulvinar_attention_router import (
    PulvinarAttentionRouter,
    PulvinarRoutingState,
)
from hbllm.hcir.world.experimental.subicular_boundary_vector import (
    BoundaryVectorCell,
    BoundaryVectorMap,
    SubicularBoundaryVectorEngine,
)
from hbllm.hcir.world.experimental.ventromedial_somatic_marker import (
    SomaticAffectiveBias,
    VentromedialSomaticMarkerEngine,
)

# ============================================================================
# Track 1: Subicular Boundary Vector Cell (BVC) Map Tests
# ============================================================================


def test_subicular_bvc_tuning_response() -> None:
    """Test single BVC Gaussian tuning curve for distance and allocentric angle."""
    bvc = BoundaryVectorCell(
        cell_id="test_bvc",
        preferred_distance=3.0,
        preferred_angle_rad=0.0,  # North (up)
        radial_tuning=1.0,
        angular_tuning=math.pi / 4.0,
    )

    # 1. Exact match produces maximal firing
    peak_resp = bvc.response(distance=3.0, angle_rad=0.0)
    assert peak_resp == pytest.approx(1.0, rel=1e-3)

    # 2. Distance deviation attenuates response
    dist_deviated = bvc.response(distance=5.0, angle_rad=0.0)
    assert 0.0 < dist_deviated < peak_resp

    # 3. Angle deviation attenuates response
    angle_deviated = bvc.response(distance=3.0, angle_rad=math.pi / 2.0)
    assert 0.0 < angle_deviated < peak_resp

    # 4. Opposite direction produces near-zero firing
    opp_resp = bvc.response(distance=3.0, angle_rad=math.pi)
    assert opp_resp < 0.01


def test_subicular_population_boundary_vector_map() -> None:
    """Test population BVC activation over 2D obstacle grid."""
    engine = SubicularBoundaryVectorEngine(
        distance_bins=[1.0, 3.0, 6.0],
        num_orientations=8,
    )
    assert len(engine.cells) == 24

    # 10x10 empty grid with a north wall at row 0
    grid = np.zeros((10, 10), dtype=int)
    grid[0, :] = 1  # North boundary

    # Avatar placed at (3, 5) -> distance to north wall is 3 units
    bvc_map = engine.compute_bvc_activations(avatar_pos=(3, 5), boundary_grid=grid)

    assert isinstance(bvc_map, BoundaryVectorMap)
    assert bvc_map.nearest_boundary_distance == pytest.approx(3.0, abs=0.1)
    assert bvc_map.active_bvc_count > 0

    # North-tuned BVCs (d=3.0) should have prominent activation
    pred = bvc_map.to_causal_predicate()
    assert pred.variable == "subicular_boundary_vector"
    assert "near_dist:3.00" in pred.value


# ============================================================================
# Track 2: Ventromedial Prefrontal Cortex (vmPFC) Somatic Marker Tests
# ============================================================================


def test_ventromedial_somatic_marker_visceral_alarm_and_appetite() -> None:
    """Test somatic marker recording and pre-deliberative visceral alarm gating."""
    vmpfc = VentromedialSomaticMarkerEngine(alarm_threshold=-0.40)

    state_safe = 1001
    state_lethal = 6666
    state_reward = 9999

    # 1. Record catastrophic outcome for lethal state
    marker_lethal = vmpfc.record_somatic_marker(
        state_hash=state_lethal,
        outcome_valence=-0.80,
        arousal=0.90,
    )
    assert marker_lethal.is_visceral_alarm is True
    assert marker_lethal.valence < -0.40

    # 2. Record rewarding outcome
    marker_reward = vmpfc.record_somatic_marker(
        state_hash=state_reward,
        outcome_valence=0.90,
        arousal=0.80,
    )
    assert marker_reward.is_visceral_alarm is False
    assert marker_reward.valence > 0.0

    # 3. Evaluate candidate actions
    actions = ["move_north", "move_south", "move_east"]
    projected_states = {
        "move_north": state_lethal,
        "move_south": state_reward,
        "move_east": state_safe,
    }

    bias: SomaticAffectiveBias = vmpfc.evaluate_somatic_bias(
        candidate_actions=actions,
        projected_state_hashes=projected_states,
    )

    # Visceral alarm must trigger and inhibit lethal action
    assert bias.alarm_active is True
    assert "move_north" in bias.inhibited_actions
    assert bias.action_biases["move_north"] < -0.50
    assert bias.action_biases["move_south"] > 0.50
    assert bias.action_biases["move_east"] == 0.0
    assert bias.gut_feeling == "visceral_alarm_threat_detected"

    predicate = bias.to_causal_predicate()
    assert predicate.variable == "somatic_affective_bias"
    assert "alarm:True" in predicate.value


# ============================================================================
# Track 3: Pulvinar Thalamic Transthalamic Routing Tests
# ============================================================================


def test_pulvinar_attention_router_cognitive_load_and_threat() -> None:
    """Test pulvinar dynamic bandwidth throttling under high cognitive load and acute threat."""
    pulvinar = PulvinarAttentionRouter()

    # 1. Baseline equilibrium: all channels open
    state_norm: PulvinarRoutingState = pulvinar.update_routing(
        cognitive_load=0.20,
        threat_urgency=0.10,
    )
    assert len(state_norm.throttled_routes) == 0
    assert pulvinar.can_transmit("perception_to_deliberation", message_cost=0.5) is True

    # 2. High cognitive load (MCTS deep search): throttle peripheral perceptual streams
    state_heavy: PulvinarRoutingState = pulvinar.update_routing(
        cognitive_load=0.85,
        threat_urgency=0.10,
    )
    assert "perception_to_deliberation" in state_heavy.throttled_routes
    assert state_heavy.routes["perception_to_deliberation"].bandwidth < 0.50

    # 3. Acute threat: emergency motor escape prioritized, long-term memory throttled
    state_threat: PulvinarRoutingState = pulvinar.update_routing(
        cognitive_load=0.30,
        threat_urgency=0.90,
    )
    assert state_threat.dominant_pathway == "threat_to_motor_escape"
    assert "episodic_to_working_memory" in state_threat.throttled_routes
    assert state_threat.routes["threat_to_motor_escape"].bandwidth == 1.0

    predicate = state_threat.to_causal_predicate()
    assert predicate.variable == "pulvinar_routing_state"
    assert "dominant:threat_to_motor_escape" in predicate.value


# ============================================================================
# Track 4: Feedback Arc Set (FAS) Native Graph Tests
# ============================================================================


def test_feedback_arc_set_on_dag_and_cycle() -> None:
    """Test feedback arc set detection on DAGs and cyclic directed graphs."""
    # 1. Directed Acyclic Graph (DAG): A -> B -> C
    graph_dag = CognitiveGraph(enable_native_substrate=False)
    graph_dag.add_node(FactNode(id="A", claim="Alpha"))
    graph_dag.add_node(FactNode(id="B", claim="Beta"))
    graph_dag.add_node(FactNode(id="C", claim="Gamma"))
    graph_dag.add_edge(
        HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["B"])
    )
    graph_dag.add_edge(
        HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["B"], targets=["C"])
    )

    fas_edges, order = algorithms.feedback_arc_set(graph_dag)
    assert len(fas_edges) == 0
    assert order == ["A", "B", "C"]

    # 2. Directed Cycle: A -> B -> C -> A
    graph_cycle = CognitiveGraph(enable_native_substrate=False)
    graph_cycle.add_node(FactNode(id="A", claim="Alpha"))
    graph_cycle.add_node(FactNode(id="B", claim="Beta"))
    graph_cycle.add_node(FactNode(id="C", claim="Gamma"))
    graph_cycle.add_edge(
        HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["B"])
    )
    graph_cycle.add_edge(
        HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["B"], targets=["C"])
    )
    graph_cycle.add_edge(
        HCIREdge(id="e3", edge_type=HCIREdgeType.CAUSES, sources=["C"], targets=["A"])
    )

    fas_edges, order = algorithms.feedback_arc_set(graph_cycle)
    # Breaking exactly 1 edge severs the 3-cycle
    assert len(fas_edges) == 1
    u, v = fas_edges[0]
    # Edge must be one of the cycle edges
    assert (u, v) in {("A", "B"), ("B", "C"), ("C", "A")}
    # Topological order contains all nodes
    assert set(order) == {"A", "B", "C"}


def test_feedback_arc_set_native_adapter_parity() -> None:
    """Test NativeGraphAdapter FAS matches CognitiveGraph behavior."""
    adapter = NativeGraphAdapter()
    adapter.add_node("N1", "concept")
    adapter.add_node("N2", "concept")
    adapter.add_node("N3", "concept")
    adapter.add_node("N4", "concept")

    # Add cycle: N1 -> N2 -> N3 -> N1, plus branch N3 -> N4
    adapter.add_edge("e1", "causes", ["N1"], ["N2"])
    adapter.add_edge("e2", "causes", ["N2"], ["N3"])
    adapter.add_edge("e3", "causes", ["N3"], ["N1"])
    adapter.add_edge("e4", "causes", ["N3"], ["N4"])

    fas_edges, order = adapter.feedback_arc_set()
    assert len(fas_edges) == 1
    assert set(order) == {"N1", "N2", "N3", "N4"}


# ============================================================================
# Engine Orchestration Tests
# ============================================================================


def test_autonomous_epistemic_engine_round13_orchestration() -> None:
    """Test full integration of Round 13 faculties into AutonomousEpistemicEngine."""
    engine = AutonomousEpistemicEngine()

    _restore_all(engine)

    # 1. Subicular BVC map
    grid = np.zeros((8, 8), dtype=int)
    grid[0, :] = 1  # Boundary at row 0
    bvc_map = engine.compute_subicular_boundary_map(boundary_grid=grid, avatar_pos=(2, 4))
    assert isinstance(bvc_map, BoundaryVectorMap)
    assert bvc_map.nearest_boundary_distance == pytest.approx(2.0, abs=0.1)

    # 2. vmPFC Somatic marker guidance
    engine.somatic_marker.record_somatic_marker(state_hash=999, outcome_valence=-0.9)
    bias = engine.evaluate_somatic_marker_bias(
        candidate_actions=["act_left", "act_right"],
        projected_state_hashes={"act_left": 999, "act_right": 123},
    )
    assert "act_left" in bias.inhibited_actions
    assert bias.alarm_active is True

    # 3. Pulvinar attention routing
    routing = engine.update_pulvinar_routing(cognitive_load=0.9, threat_urgency=0.2)
    assert "perception_to_deliberation" in routing.throttled_routes

    # 4. Minimal feedback arc set extraction
    test_graph = CognitiveGraph(enable_native_substrate=False)
    test_graph.add_node(FactNode(id="X", claim="NodeX"))
    test_graph.add_node(FactNode(id="Y", claim="NodeY"))
    test_graph.add_edge(
        HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["X"], targets=["Y"])
    )
    test_graph.add_edge(
        HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["Y"], targets=["X"])
    )

    fas, order = engine.extract_minimal_feedback_arc_set(test_graph)
    assert len(fas) == 1
    assert set(order) == {"X", "Y"}

    # Verify causal predicates logged into symbolic theory
    assert engine.symbolic_theory is not None
    pred_variables = [p.variable for p in engine.symbolic_theory.causal_predicates]
    assert "subicular_boundary_vector" in pred_variables
    assert "somatic_affective_bias" in pred_variables
    assert "pulvinar_routing_state" in pred_variables
