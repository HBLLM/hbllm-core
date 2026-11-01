"""Unit test suite for HCIR World Round 14 Cognitive Faculties & Native Substrates.

Covers:
- Track 1: Parahippocampal Cortex (PHC) Spatial Context Schema & Environmental Topology Recognition (Epstein-Bar)
- Track 2: Locus Coeruleus (LC) Tonic vs. Phasic Noradrenergic Exploration-Exploitation Switch (Aston-Jones-Cohen)
- Track 3: Secondary Somatosensory / Posterior Insular Haptic Texture & Surface Friction Integration (Craig-Kaas)
- Track 4: Native Structural Graph Strongly Connected Components (SCC) & Metagraph Condensation (Tarjan)
- Engine Orchestration Integration: AutonomousEpistemicEngine Round 14 faculties
"""

from __future__ import annotations

import numpy as np
import pytest

from hbllm.hcir.graph import CognitiveGraph, FactNode, HCIREdge, HCIREdgeType
from hbllm.hcir.native_adapter import NativeGraphAdapter
from hbllm.hcir.world.autonomous_epistemic_engine import (
    AutonomousEpistemicEngine,
)
from hbllm.hcir.world.experimental import restore_all as _restore_all
from hbllm.hcir.world.locus_coeruleus_noradrenaline import (
    ExplorationRegime,
    LocusCoeruleusNoradrenalineEngine,
)
from hbllm.hcir.world.parahippocampal_context import (
    ContextTransitionEvent,
    ParahippocampalContextEngine,
    SpatialContextSchema,
)
from hbllm.hcir.world.somatosensory_haptic import (
    HapticAffordanceFeedback,
    SomatosensoryHapticEngine,
)

# ============================================================================
# Track 1: Parahippocampal Scene Schema Recognition Tests
# ============================================================================


def test_parahippocampal_open_arena_and_corridor_classification() -> None:
    """Test classification of open arena vs narrow corridor archetypes."""
    engine = ParahippocampalContextEngine(hysteresis_threshold=0.30)

    # 1. Open Arena: 12x12 grid with only perimeter walls
    arena_grid = np.zeros((12, 12), dtype=int)
    arena_grid[0, :] = 1
    arena_grid[-1, :] = 1
    arena_grid[:, 0] = 1
    arena_grid[:, -1] = 1

    schema_arena = engine.analyze_environmental_envelope(arena_grid)
    assert isinstance(schema_arena, SpatialContextSchema)
    assert schema_arena.archetype_name == "open_arena"
    assert schema_arena.boundary_enclosure >= 0.80
    assert schema_arena.obstacle_density < 0.35
    assert "straight_line_bonus" in schema_arena.heuristic_biases

    # 2. Narrow Corridor: 4x24 highly elongated grid
    corridor_grid = np.zeros((4, 24), dtype=int)
    corridor_grid[0, :] = 1
    corridor_grid[-1, :] = 1
    corridor_grid[:, 0] = 1
    corridor_grid[:, -1] = 1

    # Force a context switch with reset to test direct classification
    corridor_engine = ParahippocampalContextEngine()
    schema_corridor = corridor_engine.analyze_environmental_envelope(corridor_grid)
    assert schema_corridor.archetype_name == "narrow_corridor"
    assert schema_corridor.aspect_ratio >= 3.0
    assert schema_corridor.heuristic_biases["straight_line_bonus"] > 1.2

    # 3. Verify CausalPredicate export
    pred = schema_arena.to_causal_predicate()
    assert pred.variable == "spatial_context_schema"
    assert "archetype:open_arena" in pred.value


def test_parahippocampal_hysteresis_and_context_transition() -> None:
    """Test that context engine applies hysteresis and emits transition events upon major shifts."""
    engine = ParahippocampalContextEngine(hysteresis_threshold=0.35)

    # Start in open arena
    grid1 = np.zeros((10, 10), dtype=int)
    grid1[0, :] = 1
    grid1[-1, :] = 1
    grid1[:, 0] = 1
    grid1[:, -1] = 1
    s1 = engine.analyze_environmental_envelope(grid1)
    assert s1.archetype_name == "open_arena"
    assert len(engine.transitions) == 0

    # Slight modification (adding 2 obstacle blocks) should not trigger a transition due to hysteresis
    grid1[5, 5] = 1
    grid1[5, 6] = 1
    s2 = engine.analyze_environmental_envelope(grid1)
    assert s2.archetype_name == "open_arena"
    assert len(engine.transitions) == 0

    # Drastic shift: transform environment into an elongated corridor
    grid_corridor = np.zeros((3, 25), dtype=int)
    grid_corridor[0, :] = 1
    grid_corridor[-1, :] = 1
    grid_corridor[:, 0] = 1
    grid_corridor[:, -1] = 1
    s3 = engine.analyze_environmental_envelope(grid_corridor)

    assert s3.archetype_name == "narrow_corridor"
    assert len(engine.transitions) == 1
    assert isinstance(engine.transitions[0], ContextTransitionEvent)
    assert engine.transitions[0].previous_schema_id == "schema_open_arena"
    assert engine.transitions[0].new_schema_id == "schema_narrow_corridor"


# ============================================================================
# Track 2: Locus Coeruleus Noradrenergic Adaptive Gain Tests
# ============================================================================


def test_locus_coeruleus_exploitation_and_stagnation_switch() -> None:
    """Test LC transition from focused exploitation to diffuse exploration and disengagement."""
    lc = LocusCoeruleusNoradrenalineEngine(
        base_tonic=0.20,
        stagnation_threshold=4,
        disengagement_threshold=8,
    )

    # 1. Steady progress maintains focused exploitation
    for _ in range(5):
        state = lc.update_gain_state(immediate_utility=0.80, progress_made=True)

    assert state.active_regime == ExplorationRegime.EXPLOITATION_FOCUSED
    assert state.tonic_level <= 0.25
    assert state.phasic_burst >= 0.70
    assert state.exploration_temperature < 0.50

    # 2. Accumulate stagnation -> transitions to diffuse exploration
    for _ in range(4):
        state = lc.update_gain_state(immediate_utility=-0.20, progress_made=False)

    assert state.active_regime == ExplorationRegime.EXPLORATION_DIFFUSE
    assert state.stagnation_counter == 4
    assert state.tonic_level > 0.35
    assert state.exploration_temperature >= 1.0

    # 3. Acute prolonged stagnation -> transitions to disengaged reset
    for _ in range(4):
        state = lc.update_gain_state(immediate_utility=-0.50, progress_made=False)

    assert state.active_regime == ExplorationRegime.DISENGAGED_RESET
    assert state.stagnation_counter >= 8
    assert state.exploration_temperature == pytest.approx(2.50)

    # 4. Major discovery triggers phasic burst and dampens tonic discharge
    old_tonic = state.tonic_level
    burst = lc.trigger_phasic_burst(magnitude=0.90)
    assert burst >= 0.90
    assert lc.current_tonic < old_tonic

    pred = state.to_causal_predicate()
    assert pred.variable == "noradrenergic_state"
    assert "regime:disengaged_reset" in pred.value


# ============================================================================
# Track 3: Secondary Somatosensory Haptic Surface Friction Tests
# ============================================================================


def test_somatosensory_haptic_friction_and_braking_calibration() -> None:
    """Test tactile feedback inference across normal, ice, and viscous surfaces."""
    haptic = SomatosensoryHapticEngine(default_friction=0.70)

    # 1. Normal terrain interaction (expected ~ actual)
    fb_normal: HapticAffordanceFeedback = haptic.process_kinesthetic_feedback(
        position=(3, 3),
        expected_displacement=(1.0, 0.0),
        actual_displacement=(1.0, 0.0),
    )
    assert fb_normal.slip_detected is False
    assert fb_normal.drag_detected is False
    assert fb_normal.inferred_profile.terrain_type == "normal_terrain"
    assert fb_normal.inferred_profile.friction_coefficient == pytest.approx(0.70, abs=0.05)

    # 2. Slippery terrain (actual displacement exceeds expected due to momentum slip)
    fb_ice: HapticAffordanceFeedback = haptic.process_kinesthetic_feedback(
        position=(4, 5),
        expected_displacement=(1.0, 0.0),
        actual_displacement=(1.8, 0.2),
    )
    assert fb_ice.slip_detected is True
    assert fb_ice.drag_detected is False
    assert fb_ice.inferred_profile.terrain_type == "ice_slippery"
    assert fb_ice.inferred_profile.friction_coefficient < 0.40
    assert fb_ice.inferred_profile.slip_probability > 0.50

    # 3. Viscous mud terrain (actual displacement heavily damped by resistive drag)
    fb_mud: HapticAffordanceFeedback = haptic.process_kinesthetic_feedback(
        position=(6, 2),
        expected_displacement=(2.0, 0.0),
        actual_displacement=(0.6, 0.0),
    )
    assert fb_mud.drag_detected is True
    assert fb_mud.slip_detected is False
    assert (
        "viscous" in fb_mud.inferred_profile.terrain_type
        or "adhesive" in fb_mud.inferred_profile.terrain_type
    )
    assert fb_mud.inferred_profile.drag_factor > 0.40

    # 4. Braking distance calibration: ice requires longer stopping distance than normal
    brake_normal = haptic.calibrate_braking_distance(velocity=2.0, position=(3, 3))
    brake_ice = haptic.calibrate_braking_distance(velocity=2.0, position=(4, 5))
    assert brake_ice > brake_normal * 1.5

    pred = fb_ice.inferred_profile.to_causal_predicate()
    assert pred.variable == "surface_friction_profile"
    assert "terrain:ice_slippery" in pred.value


# ============================================================================
# Track 4: Strongly Connected Components (SCC) & Metagraph Tests
# ============================================================================


def test_strongly_connected_components_on_dag_and_cycles() -> None:
    """Test Tarjan's SCC decomposition on acyclic and cyclic directed graphs."""
    graph = CognitiveGraph(enable_native_substrate=False)

    # 1. Directed Acyclic Graph: A -> B -> C (Each node is its own SCC)
    graph.add_node(FactNode(id="A", claim="A"))
    graph.add_node(FactNode(id="B", claim="B"))
    graph.add_node(FactNode(id="C", claim="C"))
    graph.add_edge(HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["B"]))
    graph.add_edge(HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["B"], targets=["C"]))

    sccs = graph.strongly_connected_components()
    assert len(sccs) == 3
    assert all(len(c) == 1 for c in sccs)

    # 2. Close cycle: C -> A, turning {A, B, C} into a single SCC
    graph.add_edge(HCIREdge(id="e3", edge_type=HCIREdgeType.CAUSES, sources=["C"], targets=["A"]))
    sccs_cycle = graph.strongly_connected_components()
    assert len(sccs_cycle) == 1
    assert sccs_cycle[0] == {"A", "B", "C"}


def test_condense_scc_metagraph_parity() -> None:
    """Test SCC metagraph condensation produces an acyclic DAG on CognitiveGraph & NativeGraphAdapter."""
    # Build two separate components:
    # Component 1 (Cycle): N1 <-> N2
    # Component 2 (Cycle): N3 <-> N4
    # Cross bridge: N2 -> N3
    adapter = NativeGraphAdapter()
    for n in ["N1", "N2", "N3", "N4"]:
        adapter.add_node(node_id=n, node_type="concept")

    adapter.add_edge("e12", "causes", ["N1"], ["N2"])
    adapter.add_edge("e21", "causes", ["N2"], ["N1"])
    adapter.add_edge("e34", "causes", ["N3"], ["N4"])
    adapter.add_edge("e43", "causes", ["N4"], ["N3"])
    adapter.add_edge("e23", "causes", ["N2"], ["N3"])

    scc_map, meta_edges = adapter.condense_scc_metagraph()

    # Must condense into exactly 2 supernodes
    assert len(scc_map) == 2
    # One SCC has {N1, N2}, other has {N3, N4}
    comps = list(scc_map.values())
    assert {"N1", "N2"} in comps
    assert {"N3", "N4"} in comps

    # Exactly 1 directed edge between the two SCC supernodes
    assert len(meta_edges) == 1
    u, v = meta_edges[0]
    assert scc_map[u] == {"N1", "N2"}
    assert scc_map[v] == {"N3", "N4"}


# ============================================================================
# Engine Orchestration Tests
# ============================================================================


def test_autonomous_epistemic_engine_round14_orchestration() -> None:
    """Test full integration of Round 14 faculties into AutonomousEpistemicEngine."""
    engine = AutonomousEpistemicEngine()

    _restore_all(engine)

    # 1. Spatial context classification
    grid = np.zeros((10, 10), dtype=int)
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    schema = engine.classify_spatial_context(boundary_grid=grid)
    assert schema.archetype_name == "open_arena"

    # 2. Locus coeruleus noradrenergic gain update
    norad_state = engine.update_noradrenergic_gain(immediate_utility=0.9, progress_made=True)
    assert norad_state.active_regime == ExplorationRegime.EXPLOITATION_FOCUSED

    # 3. Somatosensory haptic surface feedback
    haptic_fb = engine.process_haptic_surface_feedback(
        position=(2, 2),
        expected_displacement=(1.0, 0.0),
        actual_displacement=(2.0, 0.0),
    )
    assert haptic_fb.slip_detected is True
    assert haptic_fb.inferred_profile.terrain_type == "ice_slippery"

    # 4. SCC metagraph condensation
    test_graph = CognitiveGraph(enable_native_substrate=False)
    test_graph.add_node(FactNode(id="X", claim="NodeX"))
    test_graph.add_node(FactNode(id="Y", claim="NodeY"))
    test_graph.add_edge(
        HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["X"], targets=["Y"])
    )
    test_graph.add_edge(
        HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["Y"], targets=["X"])
    )

    scc_map, meta_edges = engine.condense_graph_scc_metagraph(test_graph)
    assert len(scc_map) == 1
    assert len(meta_edges) == 0

    # Verify causal predicates logged into symbolic theory
    assert engine.symbolic_theory is not None
    pred_variables = [p.variable for p in engine.symbolic_theory.causal_predicates]
    assert "spatial_context_schema" in pred_variables
    assert "noradrenergic_state" in pred_variables
    assert "surface_friction_profile" in pred_variables
