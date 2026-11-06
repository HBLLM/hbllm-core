from __future__ import annotations

import pytest

from hbllm.hcir.graph import (
    CognitiveGraph,
    FactNode,
    HCIREdge,
    HCIREdgeType,
)
from hbllm.hcir.native_adapter import (
    NativeGraphAdapter,
)
from hbllm.hcir.world import (
    AutonomousEpistemicEngine,
    PotentialFieldVector,
    ShapeArchetypeSignature,
)
from hbllm.hcir.world.experimental import restore_all as _restore_all

"""Comprehensive Unit Tests for Round 8 Cognitive Architecture Enhancements:

Track 1: Ventral Striatum & Lateral Habenula Dual Value-Opponency Learning
Track 2: Posterior Parietal Cortex (PPC) Potential Field Navigation & Affordance Routing
Track 3: Inferotemporal (IT) Viewpoint- & Scale-Invariant Morphological Shape Archetype Hashing
Track 4: Native Structural Graph Transitive Reduction & Minimal Causal Skeleton Extraction
"""


from hbllm.hcir.world import (
    OpponencyRegime,
    OpponencyValueState,
)
from hbllm.hcir.world.experimental import (
    DualValueOpponencyCortex,
    InferotemporalShapeHasher,
    ParietalPotentialFieldNavigator,
    SpatialDistanceTransformEngine,
)

# ─────────────────────────────────────────────────────────────────────────────
# Track 1: Ventral Striatum & Lateral Habenula Dual Value-Opponency Learning
# ─────────────────────────────────────────────────────────────────────────────


def test_value_opponency_initial_neutral_state() -> None:
    """Opponency circuit initializes in NEUTRAL regime with zeroed values."""
    cortex = DualValueOpponencyCortex()
    st = cortex.get_state((3, 3))

    assert isinstance(st, OpponencyValueState)
    assert st.state_id == "(3,3)"
    assert st.v_plus == 0.0
    assert st.v_minus == 0.0
    assert st.balance == 0.0
    assert st.regime == OpponencyRegime.NEUTRAL
    assert st.is_braked is False

    pred = st.to_causal_predicate()
    assert pred.variable == "opponency:(3,3)"
    assert pred.value == "NEUTRAL"

    ops = st.to_transaction_operations()
    assert len(ops) == 1
    assert ops[0].node_id == "opponency_(3,3)"


def test_value_opponency_appetitive_and_aversive_learning() -> None:
    """Dual TD learning updates appetitive and aversive channels and triggers Habenular brake."""
    cortex = DualValueOpponencyCortex(
        learning_rate_plus=0.5,
        learning_rate_minus=0.5,
        risk_aversion_lambda=1.5,
        brake_threshold=0.6,
    )

    # 1. Appetitive reward updates
    st_reward = cortex.update_transition((2, 2), reward_plus=1.0, punishment_minus=0.0)
    assert st_reward.v_plus > 0.4
    assert st_reward.v_minus == 0.0
    assert st_reward.balance > 0.4
    assert st_reward.regime == OpponencyRegime.APPETITIVE_DOMINANT
    assert st_reward.is_braked is False

    # 2. Aversive punishment updates
    st_hazard = cortex.update_transition((5, 5), reward_plus=0.0, punishment_minus=0.5)
    assert st_hazard.v_minus > 0.2
    assert st_hazard.balance < -0.2
    assert st_hazard.regime == OpponencyRegime.AVERSIVE_DOMINANT

    # 3. Severe punishment triggers Habenular avoidance brake
    st_severe = cortex.update_transition((5, 5), reward_plus=0.0, punishment_minus=1.0)
    assert st_severe.v_minus >= 0.6
    assert st_severe.regime == OpponencyRegime.AVOIDANCE_BRAKE
    assert st_severe.is_braked is True

    # Action inhibition check
    inhibit, reason = cortex.should_inhibit_action((5, 5))
    assert inhibit is True
    assert "lateral habenula brake active" in reason.lower()

    # Action arbitration penalties
    utilities = cortex.arbitrate_actions(
        {
            "move_to_safe": (2, 2),
            "move_to_trap": (5, 5),
        }
    )
    assert utilities["move_to_safe"] > 0.0
    assert utilities["move_to_trap"] < -5.0


def test_value_opponency_engine_integration() -> None:
    """Engine updates value opponency and registers causal predicate."""
    engine = AutonomousEpistemicEngine()
    _restore_all(engine)
    assert hasattr(engine, "value_opponency")

    st = engine.update_value_opponency(
        state_key=(4, 4),
        reward_plus=1.0,
        punishment_minus=0.0,
    )
    assert st.v_plus > 0.0
    assert engine.symbolic_theory is not None
    assert any("opponency:(4,4)" in p.variable for p in engine.symbolic_theory.causal_predicates)


def test_potential_field_pure_attraction_open_space() -> None:
    """In obstacle-free space, potential field points directly towards the goal."""
    edt_engine = SpatialDistanceTransformEngine()
    edt_res = edt_engine.compute_edt(grid_shape=(10, 10), obstacles=set())

    navigator = ParietalPotentialFieldNavigator()
    # Agent at (2, 2), Goal at (2, 8) -> direct East attraction
    vec = navigator.compute_navigation_vector(
        agent_pos=(2, 2),
        goal_pos=(2, 8),
        edt_result=edt_res,
    )

    assert isinstance(vec, PotentialFieldVector)
    assert vec.heading_direction == "EAST"
    assert vec.force_r == pytest.approx(0.0, abs=1e-3)
    assert vec.force_c > 0.5
    assert vec.is_stuck_in_local_minima is False
    assert vec.to_discrete_displacement() == (0, 1)

    # Predicate & Transaction
    pred = vec.to_causal_predicate()
    assert pred.variable == "parietal:potential_field_vector"
    assert pred.value == "EAST"

    ops = vec.to_transaction_operations()
    assert len(ops) == 1
    assert ops[0].node_id == "parietal_navigation_vector"


def test_potential_field_obstacle_repulsion_and_minima_escape() -> None:
    """EDT obstacle repels agent and saddle-point local minimum triggers orthogonal escape."""
    edt_engine = SpatialDistanceTransformEngine()
    # Obstacle placed directly at (2, 5) between agent at (2, 4) and goal at (2, 7)
    obstacles = {(2, 5)}
    edt_res = edt_engine.compute_edt(grid_shape=(10, 10), obstacles=obstacles)

    navigator = ParietalPotentialFieldNavigator(
        attractive_gain=1.0,
        repulsive_gain=3.0,
        obstacle_buffer_radius=2.0,
    )

    # At (2, 4), right next to obstacle at (2, 5):
    # Attractive force points East (+c), but EDT gradient points West away from (2,5)
    # The repulsive force opposes the attractive force
    vec = navigator.compute_navigation_vector(
        agent_pos=(2, 4),
        goal_pos=(2, 7),
        edt_result=edt_res,
    )

    # Since forces directly oppose or create a saddle point, escape perturbation or deflection occurs
    assert vec.magnitude > 0.0
    assert vec.heading_direction in {
        "NORTH",
        "SOUTH",
        "NORTH_EAST",
        "SOUTH_EAST",
        "WEST",
        "NORTH_WEST",
        "SOUTH_WEST",
    }


def test_potential_field_engine_integration() -> None:
    """Engine synthesizes potential navigation vector and records causal predicate."""
    engine = AutonomousEpistemicEngine()
    _restore_all(engine)
    assert hasattr(engine, "potential_field_navigator")

    vec = engine.compute_navigational_force(
        agent_pos=(1, 1),
        goal_pos=(1, 5),
        obstacles={(5, 5)},
        grid_shape=(10, 10),
    )

    assert vec.heading_direction == "EAST"
    assert any(
        "parietal:potential_field_vector" in p.variable
        for p in engine.symbolic_theory.causal_predicates
    )


def test_shape_hasher_translation_invariance() -> None:
    """Translated instances of the same geometric shape produce identical Hu moments and hash."""
    hasher = InferotemporalShapeHasher()

    # Shape 1: 2x3 rectangle at (1, 1)
    coords_1 = {(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)}
    sig_1 = hasher.compute_signature(coords_1)

    # Shape 2: Same 2x3 rectangle translated to (6, 6)
    coords_2 = {(6, 6), (6, 7), (6, 8), (7, 6), (7, 7), (7, 8)}
    sig_2 = hasher.compute_signature(coords_2)

    assert sig_1.archetype_hash == sig_2.archetype_hash
    assert sig_1.aspect_ratio == pytest.approx(sig_2.aspect_ratio)
    assert sig_1.hu_moments[0] == pytest.approx(sig_2.hu_moments[0], rel=1e-3)
    assert sig_1.hu_moments[1] == pytest.approx(sig_2.hu_moments[1], rel=1e-3)

    # Similarity between identical shapes is 1.0
    sim = hasher.compute_similarity(sig_1, sig_2)
    assert sim == pytest.approx(1.0)

    # Predicate & Transaction
    pred = sig_1.to_causal_predicate()
    assert "shape_archetype:" in pred.variable

    ops = sig_1.to_transaction_operations()
    assert len(ops) == 1
    assert ops[0].node_data["area"] == 6


def test_shape_hasher_scale_invariance_and_discrimination() -> None:
    """Square shapes of different scales produce similar H1 values, distinct from a line."""
    hasher = InferotemporalShapeHasher()

    # Small 2x2 square
    sq_small = {(0, 0), (0, 1), (1, 0), (1, 1)}
    sig_small = hasher.compute_signature(sq_small)

    # Larger 4x4 square
    sq_large = {(r, c) for r in range(4) for c in range(4)}
    sig_large = hasher.compute_signature(sq_large)

    # Elongated line 1x6
    line = {(0, c) for c in range(6)}
    sig_line = hasher.compute_signature(line)

    # Both squares have aspect ratio 1.0 and H2 approx 0 (symmetric)
    assert sig_small.aspect_ratio == pytest.approx(1.0)
    assert sig_large.aspect_ratio == pytest.approx(1.0)
    assert sig_small.hu_moments[1] == pytest.approx(0.0, abs=1e-4)
    assert sig_large.hu_moments[1] == pytest.approx(0.0, abs=1e-4)

    # Similarity between squares is high, similarity with line is lower
    sim_squares = hasher.compute_similarity(sig_small, sig_large)
    sim_square_line = hasher.compute_similarity(sig_small, sig_line)
    assert sim_squares > sim_square_line


def test_shape_hasher_engine_integration() -> None:
    """Engine computes shape signature and registers causal predicate."""
    engine = AutonomousEpistemicEngine()
    _restore_all(engine)
    assert hasattr(engine, "shape_hasher")

    coords = {(3, 3), (3, 4), (4, 3), (4, 4)}
    sig = engine.hash_entity_shape(coords)

    assert isinstance(sig, ShapeArchetypeSignature)
    assert sig.area == 4
    assert any("shape_archetype:" in p.variable for p in engine.symbolic_theory.causal_predicates)


def test_native_adapter_transitive_reduction() -> None:
    """NativeGraphAdapter prunes redundant shortcut edges in a DAG."""
    adapter = NativeGraphAdapter()

    # Nodes: A, B, C
    adapter.add_node(node_id="A", node_type="concept")
    adapter.add_node(node_id="B", node_type="concept")
    adapter.add_node(node_id="C", node_type="concept")

    # Edges: A -> B, B -> C, and redundant shortcut A -> C
    adapter.add_edge(edge_id="e_ab", edge_type="causes", sources=["A"], targets=["B"])
    adapter.add_edge(edge_id="e_bc", edge_type="causes", sources=["B"], targets=["C"])
    adapter.add_edge(edge_id="e_ac", edge_type="causes", sources=["A"], targets=["C"])

    # Total 3 edges initially
    assert len(adapter.edges_from("A")) == 2

    # Compute transitive reduction
    reduced = adapter.transitive_reduction()

    # Redundant edge e_ac must be pruned
    edges_from_a = reduced.edges_from("A")
    assert len(edges_from_a) == 1
    assert edges_from_a[0]["id"] == "e_ab"

    # Edge e_bc must be retained
    edges_from_b = reduced.edges_from("B")
    assert len(edges_from_b) == 1
    assert edges_from_b[0]["id"] == "e_bc"

    # Reachability from A must still include B and C!
    reachable = reduced.reachable_nodes("A", max_depth=5)
    assert "B" in reachable
    assert "C" in reachable


def test_cognitive_graph_transitive_reduction_diamond() -> None:
    """CognitiveGraph prunes redundant diagonal edge in a diamond causal DAG."""
    graph = CognitiveGraph(enable_native_substrate=False)

    n_a = FactNode(id="A", claim="Root Cause")
    n_b = FactNode(id="B", claim="Intermediate B")
    n_c = FactNode(id="C", claim="Intermediate C")
    n_d = FactNode(id="D", claim="Terminal Effect")

    graph.add_node(n_a)
    graph.add_node(n_b)
    graph.add_node(n_c)
    graph.add_node(n_d)

    # Edges: A -> B, A -> C, B -> D, C -> D, and redundant direct shortcut A -> D
    graph.add_edge(HCIREdge(id="e_ab", sources=["A"], targets=["B"], edge_type=HCIREdgeType.CAUSES))
    graph.add_edge(HCIREdge(id="e_ac", sources=["A"], targets=["C"], edge_type=HCIREdgeType.CAUSES))
    graph.add_edge(HCIREdge(id="e_bd", sources=["B"], targets=["D"], edge_type=HCIREdgeType.CAUSES))
    graph.add_edge(HCIREdge(id="e_cd", sources=["C"], targets=["D"], edge_type=HCIREdgeType.CAUSES))
    graph.add_edge(HCIREdge(id="e_ad", sources=["A"], targets=["D"], edge_type=HCIREdgeType.CAUSES))

    assert len(list(graph.all_edges())) == 5

    # Compute transitive reduction
    reduced = graph.transitive_reduction()

    # Redundant shortcut e_ad is pruned; the 4 diamond edges remain
    reduced_eids = {e.id for e in reduced.all_edges()}
    assert "e_ad" not in reduced_eids
    assert reduced_eids == {"e_ab", "e_ac", "e_bd", "e_cd"}
    assert len(list(reduced.all_nodes())) == 4


def test_engine_extract_minimal_causal_skeleton() -> None:
    """Engine extracts minimal transitive reduction causal skeleton from graph."""
    engine = AutonomousEpistemicEngine()
    _restore_all(engine)

    graph = CognitiveGraph(enable_native_substrate=False)
    graph.add_node(FactNode(id="X", claim="X"))
    graph.add_node(FactNode(id="Y", claim="Y"))
    graph.add_node(FactNode(id="Z", claim="Z"))
    graph.add_edge(HCIREdge(id="e_xy", sources=["X"], targets=["Y"], edge_type=HCIREdgeType.CAUSES))
    graph.add_edge(HCIREdge(id="e_yz", sources=["Y"], targets=["Z"], edge_type=HCIREdgeType.CAUSES))
    graph.add_edge(HCIREdge(id="e_xz", sources=["X"], targets=["Z"], edge_type=HCIREdgeType.CAUSES))

    skeleton = engine.extract_minimal_causal_skeleton(graph)
    assert len(list(skeleton.all_edges())) == 2
    assert "e_xz" not in {e.id for e in skeleton.all_edges()}
