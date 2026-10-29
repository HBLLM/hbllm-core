"""Unit test suite for HCIR World Round 15 Cognitive Faculties & Native Substrates.

Covers:
- Track 1: Entorhinal Border Cells & Grid Module Rescaling (Solstad-Barry)
- Track 2: Amygdalar Fear Conditioning & Fast Threat Potentiation (LeDoux-Phelps)
- Track 3: Ventral Tegmental Area (VTA) Dopaminergic Epistemic Curiosity Drive (Schultz-Friston)
- Track 4: Native Structural Graph Eulerian Path, Circuit & Chinese Postman Tour (Hierholzer-Edmonds)
- Engine Orchestration Integration: AutonomousEpistemicEngine Round 15 faculties
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hbllm.hcir.graph import CognitiveGraph, FactNode, HCIREdge, HCIREdgeType
from hbllm.hcir.native_adapter import NativeGraphAdapter
from hbllm.hcir.world.amygdalar_threat_conditioning import (
    AmygdalarThreatConditioningEngine,
    ConditionedThreatStimulus,
    DefensiveVigilanceMode,
    DefensiveVigilanceState,
)
from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.entorhinal_border_cell import (
    EntorhinalBorderCellEngine,
    GridModuleRescalingResult,
)
from hbllm.hcir.world.ventral_tegmental_curiosity import (
    CuriosityDriveSignal,
    EpistemicInformationGain,
    VentralTegmentalCuriosityEngine,
)

# ============================================================================
# Track 1: Entorhinal Border Cells & Grid Rescaling Tests
# ============================================================================


def test_entorhinal_border_firing_and_phase_reset() -> None:
    """Test boundary proximity firing and dead-reckoning drift reset."""
    engine = EntorhinalBorderCellEngine(
        reference_dimensions=(10.0, 10.0),
        base_wavelengths=[2.0, 4.0, 8.0],
        border_threshold=1.5,
    )

    # 10x10 grid with perimeter walls
    grid = np.zeros((10, 10), dtype=int)
    grid[0, :] = 1  # North wall
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1

    # Place avatar at (1, 5) -> immediately adjacent to north wall
    engine.record_odometric_step(step_disp=(3.0, 2.0), noise=0.1)
    assert engine.accumulated_drift != (0.0, 0.0)

    result: GridModuleRescalingResult = engine.evaluate_borders_and_rescaling(
        boundary_grid=grid,
        avatar_pos=(1, 5),
    )

    assert "north" in result.active_borders
    assert result.phase_drift_reset is True
    # Accumulated dead-reckoning drift is clamped to zero on border contact
    assert engine.accumulated_drift == (0.0, 0.0)

    # Check geometric rescaling: 10x10 matches reference (10, 10)
    assert result.scale_r == pytest.approx(1.0)
    assert result.scale_c == pytest.approx(1.0)
    assert result.rescaled_wavelengths == [2.0, 4.0, 8.0]

    # Test elongated arena rescaling (10x20)
    grid_elongated = np.zeros((10, 20), dtype=int)
    grid_elongated[0, :] = 1
    grid_elongated[-1, :] = 1
    grid_elongated[:, 0] = 1
    grid_elongated[:, -1] = 1

    res_elongated = engine.evaluate_borders_and_rescaling(
        boundary_grid=grid_elongated,
        avatar_pos=(5, 10),
    )
    assert res_elongated.scale_r == pytest.approx(1.0)
    assert res_elongated.scale_c == pytest.approx(2.0)
    # Wavelength stretched by sqrt(1 * 2) = 1.414
    assert res_elongated.rescaled_wavelengths[0] == pytest.approx(2.0 * math.sqrt(2.0), rel=1e-3)

    pred = res_elongated.to_causal_predicate()
    assert pred.variable == "grid_module_rescaling"
    assert "scale_c:2.00" in pred.value


# ============================================================================
# Track 2: Amygdalar Fear Conditioning & Threat Potentiation Tests
# ============================================================================


def test_amygdalar_fear_conditioning_and_extinction() -> None:
    """Test rapid threat acquisition, defensive cascade triggering, and gradual extinction."""
    engine = AmygdalarThreatConditioningEngine(learning_rate=0.60, extinction_rate=0.20)

    # 1. Baseline vigilance with neutral stimuli
    state_base: DefensiveVigilanceState = engine.evaluate_sensory_vigilance(
        ["stim_flower", "stim_wall"]
    )
    assert state_base.vigilance_mode == DefensiveVigilanceMode.BASELINE_ALERT
    assert state_base.composite_threat_level == 0.0
    assert state_base.evasion_recommended is False

    # 2. Pair neutral stimulus with aversive shock (CS -> US)
    cs: ConditionedThreatStimulus = engine.associate_threat(
        stimulus_key="stim_spike_hazard",
        aversive_magnitude=0.95,
    )
    assert cs.threat_potency >= 0.90
    assert cs.reinforcement_count == 1

    # 3. Present conditioned stimulus in sensory field -> triggers EVASIVE_FLIGHT
    state_threat: DefensiveVigilanceState = engine.evaluate_sensory_vigilance(
        ["stim_flower", "stim_spike_hazard"]
    )
    assert state_threat.vigilance_mode == DefensiveVigilanceMode.EVASIVE_FLIGHT
    assert state_threat.composite_threat_level >= 0.85
    assert "stim_spike_hazard" in state_threat.inhibited_approaches
    assert state_threat.evasion_recommended is True

    # 4. Extinguish threat over unreinforced exposures
    for _ in range(12):
        engine.extinguish_threat("stim_spike_hazard")

    # Threat extinguished below threshold -> returns to baseline alert
    state_extinguished = engine.evaluate_sensory_vigilance(["stim_flower", "stim_spike_hazard"])
    assert state_extinguished.vigilance_mode == DefensiveVigilanceMode.BASELINE_ALERT
    assert state_extinguished.evasion_recommended is False

    pred = state_threat.to_causal_predicate()
    assert pred.variable == "defensive_vigilance_state"
    assert "mode:evasive_flight" in pred.value


# ============================================================================
# Track 3: Ventral Tegmental Area (VTA) Epistemic Curiosity Tests
# ============================================================================


def test_vta_epistemic_curiosity_bonus_and_habituation() -> None:
    """Test curiosity reward calculation on novel states and decay on habituation."""
    vta = VentralTegmentalCuriosityEngine(base_curiosity_coefficient=0.80, novelty_decay_rate=0.50)

    # 1. Unvisited state yields maximum novelty
    gain1: EpistemicInformationGain = vta.record_visit("state_coords_4_4")
    assert gain1.visit_count == 1
    assert gain1.novelty_score == pytest.approx(1.0)
    assert gain1.curiosity_bonus == pytest.approx(0.80)
    assert gain1.information_gain > 0.50

    # 2. Repeated visits quench curiosity
    for _ in range(8):
        gain_repeat = vta.record_visit("state_coords_4_4")

    assert gain_repeat.visit_count == 9
    assert gain_repeat.novelty_score < 0.40
    assert gain_repeat.curiosity_bonus < 0.35

    # 3. Evaluate candidate actions: unvisited state receives high curiosity bonus
    candidates = ["action_up", "action_down"]
    projected = {
        "action_up": "state_coords_4_4",  # Familiar
        "action_down": "state_coords_9_9",  # Brand new
    }

    drive: CuriosityDriveSignal = vta.evaluate_candidate_curiosity(
        candidate_actions=candidates,
        projected_state_keys=projected,
    )
    assert drive.highest_novelty_action == "action_down"
    assert (
        drive.action_curiosity_bonuses["action_down"] > drive.action_curiosity_bonuses["action_up"]
    )
    assert drive.exploration_recommended is True

    pred = gain1.to_causal_predicate()
    assert pred.variable == "epistemic_curiosity_gain"
    assert "bonus:0.80" in pred.value


# ============================================================================
# Track 4: Eulerian Path, Circuit & Chinese Postman Tour Tests
# ============================================================================


def test_find_eulerian_path_and_circuit() -> None:
    """Test Eulerian circuit and path discovery using Hierholzer's algorithm."""
    # 1. Directed Eulerian Circuit: A -> B -> C -> A
    graph_circuit = CognitiveGraph(enable_native_substrate=False)
    graph_circuit.add_node(FactNode(id="A", claim="A"))
    graph_circuit.add_node(FactNode(id="B", claim="B"))
    graph_circuit.add_node(FactNode(id="C", claim="C"))
    graph_circuit.add_edge(
        HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["B"])
    )
    graph_circuit.add_edge(
        HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["B"], targets=["C"])
    )
    graph_circuit.add_edge(
        HCIREdge(id="e3", edge_type=HCIREdgeType.CAUSES, sources=["C"], targets=["A"])
    )

    is_eul, path = graph_circuit.find_eulerian_path_or_circuit()
    assert is_eul is True
    assert len(path) == 3
    # Verifies closed circuit
    assert path[0][0] == path[-1][1]

    # 2. Directed Eulerian Path: A -> B -> C (open path)
    graph_path = CognitiveGraph(enable_native_substrate=False)
    graph_path.add_node(FactNode(id="A", claim="A"))
    graph_path.add_node(FactNode(id="B", claim="B"))
    graph_path.add_node(FactNode(id="C", claim="C"))
    graph_path.add_edge(
        HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["B"])
    )
    graph_path.add_edge(
        HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["B"], targets=["C"])
    )

    is_eul_p, path_p = graph_path.find_eulerian_path_or_circuit()
    assert is_eul_p is True
    assert len(path_p) == 2
    assert path_p == [("A", "B"), ("B", "C")]


def test_chinese_postman_tour_native_adapter_parity() -> None:
    """Test Chinese Postman Tour optimization across NativeGraphAdapter and CognitiveGraph."""
    adapter = NativeGraphAdapter()
    for n in ["A", "B", "C"]:
        adapter.add_node(node_id=n, node_type="concept")

    # Strongly connected graph with unbalanced degrees:
    # A -> B (w=2.0), B -> C (w=1.0), C -> A (w=3.0) forms cycle of cost 6.0
    # Additional edge: B -> A (w=1.0)
    adapter.add_edge("e_ab", "causes", ["A"], ["B"], weight=2.0)
    adapter.add_edge("e_bc", "causes", ["B"], ["C"], weight=1.0)
    adapter.add_edge("e_ca", "causes", ["C"], ["A"], weight=3.0)
    adapter.add_edge("e_ba", "causes", ["B"], ["A"], weight=1.0)

    cost, tour = adapter.chinese_postman_tour()
    assert cost >= 7.0
    assert len(tour) >= 4
    # All 4 original edges must be visited at least once
    visited_pairs = set(tour)
    assert ("A", "B") in visited_pairs
    assert ("B", "C") in visited_pairs
    assert ("C", "A") in visited_pairs
    assert ("B", "A") in visited_pairs


# ============================================================================
# Engine Orchestration Tests
# ============================================================================


def test_autonomous_epistemic_engine_round15_orchestration() -> None:
    """Test full integration of Round 15 faculties into AutonomousEpistemicEngine."""
    engine = AutonomousEpistemicEngine()

    # 1. Entorhinal border cell rescaling
    grid = np.zeros((10, 10), dtype=int)
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    rescaling = engine.rescale_grid_modules_from_borders(boundary_grid=grid, avatar_pos=(1, 4))
    assert "north" in rescaling.active_borders
    assert rescaling.phase_drift_reset is True

    # 2. Amygdala threat conditioning
    engine.condition_threat_stimulus(stimulus_key="hazard_flame", aversive_magnitude=0.90)
    vigilance = engine.evaluate_defensive_vigilance(present_stimuli=["hazard_flame", "safe_stone"])
    assert vigilance.vigilance_mode == DefensiveVigilanceMode.EVASIVE_FLIGHT
    assert vigilance.evasion_recommended is True

    # 3. VTA epistemic curiosity bonus
    gain = engine.compute_epistemic_curiosity_bonus(state_key="coord_5_5")
    assert gain.novelty_score == pytest.approx(1.0)
    assert gain.curiosity_bonus > 0.0

    # 4. Eulerian inspection tour on graph
    test_graph = CognitiveGraph(enable_native_substrate=False)
    test_graph.add_node(FactNode(id="1", claim="N1"))
    test_graph.add_node(FactNode(id="2", claim="N2"))
    test_graph.add_edge(
        HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["1"], targets=["2"])
    )
    test_graph.add_edge(
        HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["2"], targets=["1"])
    )

    is_eul, tour = engine.compute_graph_eulerian_inspection_tour(test_graph)
    assert is_eul is True
    assert len(tour) == 2

    # Chinese Postman tour
    cost, cpp_tour = engine.compute_graph_chinese_postman_tour(test_graph)
    assert cost > 0.0
    assert len(cpp_tour) == 2

    # Verify causal predicates logged into symbolic theory
    assert engine.symbolic_theory is not None
    pred_variables = [p.variable for p in engine.symbolic_theory.causal_predicates]
    assert "grid_module_rescaling" in pred_variables
    assert "defensive_vigilance_state" in pred_variables
    assert "epistemic_curiosity_gain" in pred_variables
