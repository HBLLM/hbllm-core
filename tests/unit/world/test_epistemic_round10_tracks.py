"""Unit test suite for HCIR World Round 10 Cognitive Faculties & Native Substrates.

Covers:
- Track 1: Hippocampal-Entorhinal Prioritized Experience Replay Buffer (Foster & Wilson / Schaul)
- Track 2: Lateral Prefrontal Cortex (LPFC) Dual Task Switching & Cognitive Flexibility (Monsell / Koechlin)
- Track 3: Retrosplenial Path Integration Drift Recalibration via Visual Landmarks (Jeffery / Burgess)
- Track 4: Native Structural Graph All-Pairs Shortest Paths (APSP) & Distance Matrix Embedding (Floyd-Warshall)
"""

from __future__ import annotations

from hbllm.hcir import algorithms
from hbllm.hcir.graph import CognitiveGraph, FactNode, HCIREdge, HCIREdgeType
from hbllm.hcir.native_adapter import NativeGraphAdapter
from hbllm.hcir.world.autonomous_epistemic_engine import (
    AutonomousEpistemicEngine,
)
from hbllm.hcir.world.experimental import restore_all as _restore_all
from hbllm.hcir.world.hippocampal_replay import (
    HippocampalPrioritizedReplayBuffer,
    ReplayBatch,
    ReplayTransition,
)
from hbllm.hcir.world.lateral_prefrontal_switcher import (
    CognitiveTaskSet,
    LateralPrefrontalTaskSwitcher,
)
from hbllm.hcir.world.retrosplenial_landmark_anchor import (
    LandmarkRecalibrationResult,
    RetrosplenialLandmarkAnchorEngine,
)

# ============================================================================
# Track 1: Hippocampal Prioritized Experience Replay Buffer Tests
# ============================================================================


def test_hippocampal_replay_priority_computation() -> None:
    """Test composite priority calculation from TD-error and predictive surprise."""
    buffer = HippocampalPrioritizedReplayBuffer(alpha=0.6, surprise_weight=0.5, epsilon=1e-5)

    p1 = buffer.compute_priority(td_error=1.0, surprise=0.0)
    p2 = buffer.compute_priority(td_error=0.0, surprise=2.0)
    p3 = buffer.compute_priority(td_error=1.0, surprise=2.0)

    # p3 should exceed both p1 and p2
    assert p3 > p1
    assert p3 > p2
    assert p1 > 0.0


def test_hippocampal_replay_sampling_and_importance_weights() -> None:
    """Test prioritized transition sampling and normalized importance weights."""
    buffer = HippocampalPrioritizedReplayBuffer(capacity=50, alpha=0.6, beta_start=0.4)

    for i in range(20):
        buffer.add(
            state={"step": i},
            action=i % 4,
            reward=float(i % 2),
            next_state={"step": i + 1},
            done=(i == 19),
            td_error=float(i) * 0.1,
            surprise=float(20 - i) * 0.05,
        )

    assert len(buffer) == 20
    batch: ReplayBatch = buffer.sample(batch_size=8)
    assert len(batch) == 8
    assert len(batch.indices) == 8
    assert len(batch.weights) == 8

    # Weights must be normalized in (0, 1] with at least one 1.0 (max weight)
    for w in batch.weights:
        assert 0.0 < w <= 1.0001
    assert any(abs(w - 1.0) < 1e-4 for w in batch.weights)


def test_hippocampal_replay_bidirectional_sequences() -> None:
    """Test reverse replay leading to terminal event and forward preplay sequences."""
    buffer = HippocampalPrioritizedReplayBuffer(capacity=50)

    transitions: list[ReplayTransition] = []
    for i in range(10):
        t = buffer.add(
            state={"pos": i},
            action=1,
            reward=10.0 if i == 9 else 0.0,
            next_state={"pos": i + 1},
            done=(i == 9),
            td_error=2.0 if i == 9 else 0.1,
        )
        transitions.append(t)

    terminal_id = transitions[-1].transition_id

    # Reverse replay should trace backward from terminal transition
    reverse_seq = buffer.generate_reverse_replay(terminal_id=terminal_id, max_depth=5)
    assert len(reverse_seq) == 5
    assert reverse_seq[0].transition_id == terminal_id
    assert reverse_seq[-1].transition_id == transitions[5].transition_id

    # Forward preplay should trace forward from seed
    seed_id = transitions[2].transition_id
    forward_seq = buffer.generate_forward_preplay(start_id=seed_id, max_depth=4)
    assert len(forward_seq) == 4
    assert forward_seq[0].transition_id == seed_id
    assert forward_seq[-1].transition_id == transitions[5].transition_id


def test_hippocampal_schema_distillation() -> None:
    """Test extraction of recurrent high-salience schemas."""
    buffer = HippocampalPrioritizedReplayBuffer(capacity=50)

    for i in range(15):
        buffer.add(
            state={"s": i},
            action=2,  # Frequent winning action
            reward=5.0,
            next_state={"s": i + 1},
            done=False,
            td_error=1.5,
        )

    schemas = buffer.distill_high_priority_schemas(min_priority_percentile=50.0)
    assert len(schemas) > 0
    assert schemas[0]["action"] == 2
    assert schemas[0]["support"] >= 2
    assert schemas[0]["average_reward"] == 5.0


# ============================================================================
# Track 2: Lateral Prefrontal Cortex Task Switching Tests
# ============================================================================


def test_lateral_prefrontal_affinities_and_threat_override() -> None:
    """Test affinity distributions and threat avoidance priority over navigation."""
    switcher = LateralPrefrontalTaskSwitcher(initial_task=CognitiveTaskSet.NAVIGATE)

    # 1. Distant hazard, distant goal -> primarily NAVIGATE
    affinities_nav = switcher.evaluate_affinities(hazard_distance=10.0, goal_distance=5.0)
    assert affinities_nav[CognitiveTaskSet.NAVIGATE] > affinities_nav[CognitiveTaskSet.AVOID]

    # 2. Imminent lethal hazard (dist <= 1.5) -> AVOID must dominate
    affinities_avoid = switcher.evaluate_affinities(hazard_distance=1.0)
    assert affinities_avoid[CognitiveTaskSet.AVOID] > 0.45
    assert affinities_avoid[CognitiveTaskSet.AVOID] > affinities_avoid[CognitiveTaskSet.NAVIGATE]


def test_lateral_prefrontal_hysteresis_and_inertia() -> None:
    """Test task-set inertia preventing rapid chattering unless margin is exceeded."""
    switcher = LateralPrefrontalTaskSwitcher(
        initial_task=CognitiveTaskSet.NAVIGATE,
        inertia_margin=0.20,
    )

    # Candidate has marginal lead (0.35 vs 0.30) - should NOT trigger switch due to inertia
    marginal_affinities = {
        CognitiveTaskSet.NAVIGATE: 0.30,
        CognitiveTaskSet.INTERACT: 0.35,
        CognitiveTaskSet.AVOID: 0.10,
        CognitiveTaskSet.COLLECT: 0.15,
        CognitiveTaskSet.EXPLORE: 0.10,
    }
    res1 = switcher.arbitrate(marginal_affinities)
    assert res1.switched is False
    assert res1.active_task == CognitiveTaskSet.NAVIGATE

    # Competitor clearly beats current task + inertia (0.60 vs 0.20) -> triggers switch
    decisive_affinities = {
        CognitiveTaskSet.NAVIGATE: 0.20,
        CognitiveTaskSet.INTERACT: 0.60,
        CognitiveTaskSet.AVOID: 0.05,
        CognitiveTaskSet.COLLECT: 0.10,
        CognitiveTaskSet.EXPLORE: 0.05,
    }
    res2 = switcher.arbitrate(decisive_affinities)
    assert res2.switched is True
    assert res2.active_task == CognitiveTaskSet.INTERACT
    assert res2.previous_task == CognitiveTaskSet.NAVIGATE
    assert res2.switch_cost > 0.0
    assert res2.inhibitory_suppression > 0.0


# ============================================================================
# Track 3: Retrosplenial Landmark Anchor Engine Tests
# ============================================================================


def test_retrosplenial_landmark_drift_recalibration() -> None:
    """Test RSC landmark-based Kalman coordinate recalibration and variance contraction."""
    engine = RetrosplenialLandmarkAnchorEngine(
        base_landmark_variance=0.1,
        initial_odometry_variance=0.5,
        drift_rate_per_step=0.1,
    )

    # Register immovable boundary landmark at (0, 0)
    engine.register_landmark(landmark_id="corner_0_0", position=(0, 0), stability=1.0)

    # Dead reckoning moves 5 steps -> accumulates odometric variance
    engine.step_odometry(steps=5)
    assert engine.odometry_variance > 0.9

    # Avatar dead reckoning thinks it is at (5, 5), but visual bearing to corner (0, 0)
    # is observed as (-4, -4). Implied avatar position = (0, 0) - (-4, -4) = (4, 4).
    # True position is (4, 4), dead reckoning drifted to (5, 5).
    observations = [("corner_0_0", (-4, -4))]
    result: LandmarkRecalibrationResult = engine.recalibrate(
        odometry_pos=(5, 5),
        landmark_observations=observations,
    )

    assert result.has_drift_correction is True
    assert result.corrected_position == (4, 4)
    assert result.kalman_gain > 0.7
    assert result.variance_after < result.variance_before
    assert "corner_0_0" in result.recalibrated_landmark_ids


# ============================================================================
# Track 4: Graph All-Pairs Shortest Paths (APSP) & Distance Matrix Tests
# ============================================================================


def test_native_graph_adapter_apsp_and_centrality() -> None:
    """Test Floyd-Warshall APSP, diameter, and closeness centrality on NativeGraphAdapter."""
    adapter = NativeGraphAdapter()

    # Create directed path A -> B -> C -> D with weighted edges
    for node in ["A", "B", "C", "D"]:
        adapter.add_node(node_id=node, node_type="concept")

    adapter.add_edge("e1", "causes", sources=["A"], targets=["B"], weight=2.0)
    adapter.add_edge("e2", "causes", sources=["B"], targets=["C"], weight=3.0)
    adapter.add_edge("e3", "causes", sources=["C"], targets=["D"], weight=1.0)

    dist, paths = adapter.all_pairs_shortest_paths(weight_attribute="weight")

    # Verify distances
    assert dist[("A", "A")] == 0.0
    assert dist[("A", "B")] == 2.0
    assert dist[("A", "C")] == 5.0
    assert dist[("A", "D")] == 6.0
    assert dist[("B", "D")] == 4.0
    assert dist[("D", "A")] == float("inf")  # Directed

    # Verify reconstructed path
    assert paths[("A", "D")] == ["A", "B", "C", "D"]
    assert paths[("D", "A")] == []

    # Verify diameter and centrality
    diameter = adapter.graph_diameter(weight_attribute="weight")
    assert diameter == 6.0

    centrality_a = adapter.closeness_centrality("A", weight_attribute="weight")
    assert centrality_a > 0.0


def test_cognitive_graph_apsp_parity() -> None:
    """Test Floyd-Warshall APSP implementation on CognitiveGraph."""
    graph = CognitiveGraph(enable_native_substrate=False)

    for nid in ["X", "Y", "Z"]:
        graph.add_node(FactNode(id=nid, claim=nid))

    e1 = HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["X"], targets=["Y"])
    e1.weight = 4.0  # type: ignore[attr-defined]
    e2 = HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["Y"], targets=["Z"])
    e2.weight = 2.0  # type: ignore[attr-defined]

    graph.add_edge(e1)
    graph.add_edge(e2)

    dist, paths = algorithms.all_pairs_shortest_paths(graph, weight_attr="weight")

    assert dist[("X", "Z")] == 6.0
    assert paths[("X", "Z")] == ["X", "Y", "Z"]
    assert algorithms.graph_radius_and_diameter(graph, weight_attr="weight")[1] == 6.0


# ============================================================================
# Orchestrator Integration Tests: AutonomousEpistemicEngine Round 10
# ============================================================================


def test_epistemic_engine_round10_integration() -> None:
    """Test AutonomousEpistemicEngine delegation to Round 10 faculties."""
    engine = AutonomousEpistemicEngine()

    _restore_all(engine)

    # 1. Test hippocampal replay recording and sampling
    engine.record_replay_transition(
        state={"step": 1},
        action=0,
        reward=1.0,
        next_state={"step": 2},
        done=False,
        td_error=0.5,
        surprise=0.8,
    )
    assert len(engine.hippocampal_replay) == 1
    batch = engine.sample_prioritized_replay(batch_size=1)
    assert len(batch) == 1

    # 2. Test task set arbitration and causal predicate recording
    task_res = engine.arbitrate_cognitive_task_set(
        hazard_distance=0.8,  # Critical hazard triggers AVOID
        goal_distance=10.0,
    )
    assert task_res.active_task == CognitiveTaskSet.AVOID
    assert any(
        p.variable == "active_task_set" and p.value == "avoid"
        for p in engine.symbolic_theory.causal_predicates
    )

    # 3. Test retrosplenial landmark recalibration
    engine.retrosplenial_anchor.register_landmark("gate_1", position=(10, 10), stability=1.0)
    recal = engine.recalibrate_spatial_drift_with_landmarks(
        odometry_pos=(12, 12),
        landmark_observations=[("gate_1", (-2, -2))],  # True pos = (10, 10) - (-2, -2) = (12, 12)
    )
    assert recal.kalman_gain > 0.0

    # 4. Test all-pairs geodesics on brain graph
    graph = CognitiveGraph(enable_native_substrate=False)
    graph.add_node(FactNode(id="n1", claim="n1"))
    graph.add_node(FactNode(id="n2", claim="n2"))
    e = HCIREdge(id="e_test", edge_type=HCIREdgeType.CAUSES, sources=["n1"], targets=["n2"])
    e.weight = 1.5  # type: ignore[attr-defined]
    graph.add_edge(e)

    dist_map, path_map = engine.compute_all_pairs_geodesics(graph=graph)
    assert dist_map[("n1", "n2")] == 1.5
    assert path_map[("n1", "n2")] == ["n1", "n2"]
