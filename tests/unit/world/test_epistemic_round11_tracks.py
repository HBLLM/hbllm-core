"""Unit test suite for HCIR World Round 11 Cognitive Faculties & Native Substrates.

Covers:
- Track 1: Hippocampal Theta-Gamma Phase Precession & Macro Chunking (O'Keefe / Lisman)
- Track 2: Frontal Polar Cortex (FPC / BA10) Counterfactual Branch Switching (Koechlin / Rushworth)
- Track 3: Superior Colliculus Multisensory Alignment & Saccadic Arbiter (Stein & Meredith)
- Track 4: Native Structural Graph Articulation Points, Bridge Detection & Biconnectivity (Tarjan)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hbllm.hcir.graph import CognitiveGraph, FactNode, HCIREdge, HCIREdgeType
from hbllm.hcir.native_adapter import NativeGraphAdapter
from hbllm.hcir.world.autonomous_epistemic_engine import (
    AutonomousEpistemicEngine,
)
from hbllm.hcir.world.frontopolar_exploration_horizon import (
    FrontopolarExplorationHorizonEngine,
)
from hbllm.hcir.world.hippocampal_phase_precession import (
    HippocampalPhasePrecessionEngine,
)
from hbllm.hcir.world.superior_colliculus_arbiter import (
    MultisensoryPriorityMap,
    SaccadeTarget,
    SuperiorColliculusTargetArbiter,
)

# ============================================================================
# Track 1: Hippocampal Theta-Gamma Phase Precession Tests
# ============================================================================


def test_hippocampal_phase_precession_dynamics() -> None:
    """Test theta cycle stepping, continuous precession, and sequence compression."""
    engine = HippocampalPhasePrecessionEngine(
        theta_frequency_hz=8.0,
        gamma_cycles_per_theta=7,
        place_field_length=10.0,
    )

    # 1. Test oscillatory stepping (125 ms = 1 theta wave cycle at 8 Hz)
    tp1 = engine.step_theta_phase(dt=0.0625)  # Half cycle -> ~pi radians
    assert math.isclose(tp1.phase_rad, math.pi, rel_tol=1e-2)

    tp2 = engine.step_theta_phase(dt=0.0625)  # Full cycle -> wrap to 0 rad, cycle = 1
    assert tp2.cycle_index == 1

    # 2. Test place field phase precession monotonicity: phase advances / decrements as agent moves
    phase_entry = engine.compute_place_field_phase(position_along_field=0.0)
    phase_mid = engine.compute_place_field_phase(position_along_field=5.0)
    phase_exit = engine.compute_place_field_phase(position_along_field=10.0)

    # At entry, phase is late (~320 deg); as it moves, phase precesses to earlier values
    assert phase_entry != phase_mid
    assert phase_mid != phase_exit

    # 3. Test trajectory burst compression
    positions = [(0, 0), (0, 1), (0, 2), (0, 3)]
    actions = [1, 1, 1]
    burst = engine.encode_trajectory_burst(
        positions=positions,
        actions=actions,
        behavioral_duration_s=1.5,
    )

    assert len(burst.ordered_actions) == 3
    assert len(burst.relative_phases) == 3
    assert len(burst.gamma_subcycles) == 3
    # Compression ratio: 1.5s / 0.125s = 12.0x
    assert burst.compression_ratio == pytest.approx(12.0)

    # 4. Test macro primitive chunking
    macro = engine.chunk_into_macro_primitive(burst, min_steps=2)
    assert macro is not None
    assert macro["action_sequence"] == [1, 1, 1]
    assert macro["step_count"] == 3


# ============================================================================
# Track 2: Frontal Polar Cortex (BA10) Exploration Horizon Tests
# ============================================================================


def test_frontopolar_counterfactual_branch_switching() -> None:
    """Test background branch tracking, foreground decay, and executive branch swapping."""
    engine = FrontopolarExplorationHorizonEngine(
        switch_threshold=0.20,
        foreground_decay_rate=0.10,
        exploration_bonus_weight=0.30,
    )

    # Register two background branches
    engine.register_counterfactual_branch(
        branch_id="alt_explore_north",
        intent="explore_unmapped_frontier",
        target_position=(2, 8),
        initial_value=0.55,
        uncertainty=0.60,
    )
    engine.register_counterfactual_branch(
        branch_id="alt_interact_switch",
        intent="manipulate_distant_door",
        target_position=(8, 2),
        initial_value=0.40,
        uncertainty=0.20,
    )

    # Initial evaluation: foreground is dominant (0.70 vs ~0.73) -> no switch
    decision1 = engine.evaluate_branch_switch()
    assert decision1.should_switch is False

    # Simulate foreground plan stalling (e.g. hitting dead end, no progress) for 4 steps
    for _ in range(4):
        engine.step_foreground_progress(reward=0.0, progress_made=False)

    assert engine.foreground_value < 0.50

    # Boost evidence for north exploration branch
    engine.update_background_evidence("alt_explore_north", evidence_delta=0.20)

    # Now counterfactual advantage exceeds threshold -> triggers executive switch
    decision2 = engine.evaluate_branch_switch()
    assert decision2.should_switch is True
    assert decision2.recommended_branch_id == "alt_explore_north"
    assert decision2.value_difference > 0.20

    # Execute switch
    new_active = engine.execute_branch_switch(decision2)
    assert new_active == "alt_explore_north"
    assert engine.active_branch_id == "alt_explore_north"
    assert engine.total_switches == 1
    # Demoted plan should now be tracked as a background counterfactual branch
    assert "foreground_primary" in engine.branches


# ============================================================================
# Track 3: Superior Colliculus Multisensory Alignment Tests
# ============================================================================


def test_superior_colliculus_priority_map_and_saccades() -> None:
    """Test SC sensorimotor priority fusion and discrete saccade target selection."""
    arbiter = SuperiorColliculusTargetArbiter(
        visual_weight=0.40,
        goal_weight=0.40,
        clearance_weight=0.20,
        inhibition_radius=2,
    )

    h, w = 12, 12
    visual_saliency = np.zeros((h, w), dtype=np.float64)
    visual_saliency[3, 3] = 0.9  # Perceptually bright feature

    goals = [(9, 9)]  # Primary goal anchor
    obstacle_clearance = np.ones((h, w), dtype=np.float64) * 0.5

    priority_map: MultisensoryPriorityMap = arbiter.build_priority_map(
        visual_saliency=visual_saliency,
        goal_positions=goals,
        obstacle_clearance=obstacle_clearance,
        avatar_pos=(6, 6),
    )

    assert priority_map.priority_grid.shape == (h, w)
    assert priority_map.primary_target is not None

    # Primary target should be selected at one of the salient peaks (either goal or visual peak)
    target: SaccadeTarget = priority_map.primary_target
    assert target.position in [(9, 9), (3, 3)]
    assert target.saccade_priority > 0.5
    assert target.distance_from_avatar > 0.0


# ============================================================================
# Track 4: Native Structural Graph Articulation Points & Bridges Tests
# ============================================================================


def test_native_adapter_articulation_points_and_bridges() -> None:
    """Tarjan's algorithm on NativeGraphAdapter detects cut vertices, bridges, and biconnectivity."""
    adapter = NativeGraphAdapter()

    # Graph structure:
    # Component 1 (Triangle): A - B - C - A
    # Bridge: C - D
    # Component 2 (Triangle): D - E - F - D
    nodes = ["A", "B", "C", "D", "E", "F"]
    for n in nodes:
        adapter.add_node(node_id=n, node_type="concept")

    # Triangle 1
    adapter.add_edge("e_ab", "link", sources=["A"], targets=["B"])
    adapter.add_edge("e_bc", "link", sources=["B"], targets=["C"])
    adapter.add_edge("e_ca", "link", sources=["C"], targets=["A"])

    # Bridge between C and D
    adapter.add_edge("e_cd", "link", sources=["C"], targets=["D"])

    # Triangle 2
    adapter.add_edge("e_de", "link", sources=["D"], targets=["E"])
    adapter.add_edge("e_ef", "link", sources=["E"], targets=["F"])
    adapter.add_edge("e_fd", "link", sources=["F"], targets=["D"])

    art_points, bridges = adapter.find_articulation_points_and_bridges()

    # Articulation points must be C and D
    assert art_points == {"C", "D"}

    # Bridge edge must be (C, D) or (D, C)
    bridge_pairs = set(bridges)
    assert ("C", "D") in bridge_pairs or ("D", "C") in bridge_pairs

    # Graph has articulation points, so it is NOT biconnected
    assert adapter.is_biconnected() is False

    # Now create a simple 4-cycle graph: 1 - 2 - 3 - 4 - 1
    cycle_adapter = NativeGraphAdapter()
    for v in ["1", "2", "3", "4"]:
        cycle_adapter.add_node(node_id=v, node_type="concept")
    cycle_adapter.add_edge("c1", "link", sources=["1"], targets=["2"])
    cycle_adapter.add_edge("c2", "link", sources=["2"], targets=["3"])
    cycle_adapter.add_edge("c3", "link", sources=["3"], targets=["4"])
    cycle_adapter.add_edge("c4", "link", sources=["4"], targets=["1"])

    c_art, c_bridges = cycle_adapter.find_articulation_points_and_bridges()
    assert len(c_art) == 0
    assert len(c_bridges) == 0
    assert cycle_adapter.is_biconnected() is True


def test_cognitive_graph_biconnectivity_parity() -> None:
    """CognitiveGraph implements identical Tarjan biconnectivity analysis."""
    graph = CognitiveGraph(enable_native_substrate=False)

    for nid in ["A", "B", "C"]:
        graph.add_node(FactNode(id=nid, claim=nid))

    # Path: A - B - C (B is articulation point, A-B and B-C are bridges)
    graph.add_edge(HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["B"]))
    graph.add_edge(HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["B"], targets=["C"]))

    art_points, bridges = graph.find_articulation_points_and_bridges()

    assert art_points == {"B"}
    assert len(bridges) == 2
    assert graph.is_biconnected() is False


# ============================================================================
# Orchestrator Integration Tests: AutonomousEpistemicEngine Round 11
# ============================================================================


def test_epistemic_engine_round11_integration() -> None:
    """Test AutonomousEpistemicEngine delegation to Round 11 faculties."""
    engine = AutonomousEpistemicEngine()

    # 1. Test phase precession trajectory compression
    burst = engine.compress_trajectory_phase_precession(
        positions=[(1, 1), (1, 2), (1, 3)],
        actions=[2, 2],
        duration_s=1.0,
    )
    assert len(burst.ordered_actions) == 2
    assert any(
        p.variable == "phase_precession_burst" for p in engine.symbolic_theory.causal_predicates
    )

    # 2. Test counterfactual branch switching delegation
    engine.exploration_horizon.register_counterfactual_branch(
        branch_id="plan_escape",
        intent="emergency_flee",
        initial_value=0.95,
        uncertainty=0.1,
    )
    engine.exploration_horizon.foreground_value = 0.20  # Stalled foreground
    decision = engine.evaluate_counterfactual_branch_switch()
    assert decision.should_switch is True
    assert decision.recommended_branch_id == "plan_escape"

    # 3. Test superior colliculus saccade selection delegation
    saccade_target = engine.arbitrate_saccadic_target(
        visual_saliency=np.zeros((8, 8)),
        goal_positions=[(4, 4)],
    )
    assert saccade_target is not None
    assert saccade_target.position == (4, 4)

    # 4. Test graph bottleneck and biconnectivity checking
    graph = CognitiveGraph(enable_native_substrate=False)
    graph.add_node(FactNode(id="u1", claim="u1"))
    graph.add_node(FactNode(id="u2", claim="u2"))
    graph.add_edge(
        HCIREdge(id="e_single", edge_type=HCIREdgeType.CAUSES, sources=["u1"], targets=["u2"])
    )

    art_points, bridges = engine.find_graph_bottlenecks_and_bridges(graph=graph)
    assert len(bridges) == 1
    assert engine.is_graph_fault_tolerant_biconnected(graph=graph) is False
