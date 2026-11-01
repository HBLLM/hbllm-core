from __future__ import annotations

"""Comprehensive Unit Tests for Round 9 Cognitive Architecture Enhancements:

Track 1: Subthalamic Nucleus (STN) Decision Threshold Modulation via Drift-Diffusion Dynamics
Track 2: Parietal Egocentric Foveation & Spatial Saliency Field Compression
Track 3: Prefrontal Counterfactual Blame Attribution & Causal Credit Assignment
Track 4: Native Structural Graph Minimum Spanning Tree (MST) & Geodesic Spine Extraction
"""


from typing import Any

import numpy as np
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
    DriftDiffusionDecision,
    FoveatedSaliencyMap,
    FoveationResolution,
    PlanAttributionReport,
)
from hbllm.hcir.world.experimental import (
    ParietalFoveatedAttentionEngine,
    PrefrontalBlameAttributor,
    SubthalamicDriftDiffusionEngine,
)

# ─────────────────────────────────────────────────────────────────────────────
# Track 1: Subthalamic Nucleus (STN) Drift-Diffusion Decision Dynamics
# ─────────────────────────────────────────────────────────────────────────────


def test_stn_drift_diffusion_conflict_threshold_elevation() -> None:
    """High ACC conflict elevates STN decision boundary ('hold your horses')."""
    engine = SubthalamicDriftDiffusionEngine(base_threshold=1.0, conflict_gain=2.0)

    thresh_low_conflict = engine.compute_threshold(conflict_level=0.0, elapsed_steps=0)
    thresh_high_conflict = engine.compute_threshold(conflict_level=0.8, elapsed_steps=0)

    assert thresh_low_conflict == pytest.approx(1.0)
    assert thresh_high_conflict == pytest.approx(1.0 * (1.0 + 2.0 * 0.8))  # 2.6
    assert thresh_high_conflict > thresh_low_conflict


def test_stn_drift_diffusion_urgency_collapse_and_accumulation() -> None:
    """Evidence accumulates to bound, while urgency over time collapses threshold."""
    engine = SubthalamicDriftDiffusionEngine(
        base_threshold=1.0,
        conflict_gain=1.0,
        urgency_decay_rate=0.1,
    )

    rates = {"action_left": 0.4, "action_right": 0.1}

    # Step 1: evidence accumulates, threshold not yet reached
    dec1 = engine.step_accumulation(rates, conflict_level=0.0)
    assert isinstance(dec1, DriftDiffusionDecision)
    assert dec1.chosen_action is None
    assert dec1.is_threshold_reached is False
    assert dec1.evidence_level == pytest.approx(0.4)

    # Step 2: evidence reaches ~0.8
    dec2 = engine.step_accumulation(rates, conflict_level=0.0)
    assert dec2.chosen_action is None
    assert dec2.evidence_level == pytest.approx(0.8)

    # Step 3: threshold decays (~0.74) and evidence reaches 1.2 -> threshold crossed!
    dec3 = engine.step_accumulation(rates, conflict_level=0.0)
    assert dec3.is_threshold_reached is True
    assert dec3.chosen_action == "action_left"
    assert dec3.confidence > 0.5

    # Causal predicate and transactions
    pred = dec3.to_causal_predicate()
    assert pred.variable == "stn:drift_diffusion_decision"
    assert pred.value == "action_left"

    ops = dec3.to_transaction_operations()
    assert len(ops) == 1
    assert ops[0].node_id == "stn_decision_state"


def test_stn_drift_diffusion_engine_integration() -> None:
    """Autonomous epistemic engine runs drift diffusion accumulation and records predicate."""
    engine = AutonomousEpistemicEngine()
    assert hasattr(engine, "drift_diffusion")

    rates = {"up": 0.6, "down": 0.2}
    dec = engine.step_drift_diffusion(rates, conflict_level=0.1)

    assert isinstance(dec, DriftDiffusionDecision)
    assert engine.symbolic_theory is not None
    assert any(
        "stn:drift_diffusion_decision" in p.variable
        for p in engine.symbolic_theory.causal_predicates
    )


def test_parietal_foveation_multi_scale_resolution() -> None:
    """Foveation partitions grid into FOVEA, PARAFOVEA, and PERIPHERY tiers."""
    fovea_engine = ParietalFoveatedAttentionEngine(fovea_radius=2, parafovea_radius=4)
    grid = np.zeros((10, 10), dtype=int)
    grid[5, 5] = 1  # Agent
    grid[5, 6] = 2  # Adjacent item in fovea
    grid[1, 1] = 3  # Distant item in periphery

    foveated = fovea_engine.foveate_grid(grid, gaze_center=(5, 5))

    assert isinstance(foveated, FoveatedSaliencyMap)
    assert foveated.grid_shape == (10, 10)
    assert foveated.total_original_cells == 100

    # Center is FOVEA (1x1)
    assert foveated.resolution_grid[5][5] == FoveationResolution.FOVEA
    assert foveated.resolution_grid[5][6] == FoveationResolution.FOVEA
    assert foveated.get_feature_at(5, 6) == 2

    # Corner (0, 0) is PERIPHERY
    assert foveated.resolution_grid[0][0] == FoveationResolution.PERIPHERY

    # Substantial compression achieved
    assert foveated.total_compressed_cells < 100
    assert foveated.compression_ratio > 0.40

    # Serialization
    pred = foveated.to_causal_predicate()
    assert pred.variable == "parietal:foveated_attention"

    ops = foveated.to_transaction_operations()
    assert len(ops) == 1
    assert ops[0].node_id == "foveated_saliency_state"


def test_parietal_foveation_engine_integration() -> None:
    """Engine foveates sensory lattice around avatar position and records predicate."""
    engine = AutonomousEpistemicEngine()
    assert hasattr(engine, "foveated_attention")

    grid = np.zeros((10, 10), dtype=int)
    grid[3, 3] = 7
    foveated = engine.foveate_sensory_lattice(grid=grid, gaze_center=(3, 3))

    assert foveated.gaze_center == (3, 3)
    assert foveated.compression_ratio > 0.0
    assert any(
        "parietal:foveated_attention" in p.variable
        for p in engine.symbolic_theory.causal_predicates
    )


def test_prefrontal_blame_attribution_culprit_isolation() -> None:
    """Isolates pivotal faulty subgoal in a dead-end plan trace using Pearl causality."""
    attributor = PrefrontalBlameAttributor()

    trace: list[dict[str, Any]] = [
        {"id": "sg_1", "intent": "NAVIGATE", "target_coord": (1, 1), "alternatives": []},
        {
            "id": "sg_2",
            "intent": "NAVIGATE",
            "target_coord": (2, 3),
            "alternatives": ["take_west_hall"],
        },
        {"id": "sg_3", "intent": "INTERACT", "target_coord": (5, 5), "alternatives": []},
    ]

    report = attributor.attribute_plan_failure(
        subgoal_trace=trace,
        failure_condition="dead_end",
        blocked_targets={(5, 5)},
    )

    assert isinstance(report, PlanAttributionReport)
    assert report.failed_outcome == "dead_end"
    assert report.primary_culprit_id == "sg_3"

    # sg_3 led directly to blocked target -> rho = 1.0
    blame_map = {b.subgoal_id: b for b in report.subgoal_blames}
    assert blame_map["sg_3"].responsibility_score == 1.0
    assert blame_map["sg_3"].is_primary_cause is True

    # sg_2 had an alternative branch -> contributory responsibility rho = 0.5
    assert blame_map["sg_2"].responsibility_score == 0.5
    assert blame_map["sg_2"].counterfactual_alternative == "take_west_hall"

    # sg_1 had no alternatives -> rho = 0.0
    assert blame_map["sg_1"].responsibility_score == 0.0

    # Serialization
    pred = report.to_causal_predicate()
    assert pred.variable == "blame:plan_attribution_report"
    assert pred.value == "sg_3"

    ops = report.to_transaction_operations()
    assert len(ops) >= 4  # report + 3 subgoals


def test_prefrontal_blame_attribution_engine_integration() -> None:
    """Engine attributes blame to failed trace and registers causal predicates."""
    engine = AutonomousEpistemicEngine()
    assert hasattr(engine, "blame_attributor")

    trace: list[dict[str, Any]] = [
        {"id": "step_a", "intent": "EXPLORE", "target_coord": (2, 2), "alternatives": []},
        {
            "id": "step_b",
            "intent": "MOVE",
            "target_coord": (9, 9),
            "alternatives": ["safe_corridor"],
        },
    ]

    report = engine.attribute_plan_failure(trace, failure_condition="trap")
    assert report.primary_culprit_id == "step_b"
    assert any(
        "blame:plan_attribution_report" in p.variable
        for p in engine.symbolic_theory.causal_predicates
    )


def test_native_adapter_minimum_spanning_tree() -> None:
    """Kruskal's MST on NativeGraphAdapter extracts minimal-cost connectivity spine."""
    adapter = NativeGraphAdapter()

    # Vertices: 1, 2, 3, 4
    for v in ["1", "2", "3", "4"]:
        adapter.add_node(node_id=v, node_type="concept")

    # Weighted edges:
    # 1-2 (w=1), 2-3 (w=2), 3-4 (w=3), 1-3 (w=5), 2-4 (w=4)
    adapter.add_edge(edge_id="e12", edge_type="link", sources=["1"], targets=["2"], weight=1.0)
    adapter.add_edge(edge_id="e23", edge_type="link", sources=["2"], targets=["3"], weight=2.0)
    adapter.add_edge(edge_id="e34", edge_type="link", sources=["3"], targets=["4"], weight=3.0)
    adapter.add_edge(edge_id="e13", edge_type="link", sources=["1"], targets=["3"], weight=5.0)
    adapter.add_edge(edge_id="e24", edge_type="link", sources=["2"], targets=["4"], weight=4.0)

    mst, total_weight = adapter.minimum_spanning_tree()

    # MST must select e12 (1.0), e23 (2.0), e34 (3.0) -> total weight = 6.0
    assert total_weight == pytest.approx(6.0)
    assert mst.has_edge("e12") is True
    assert mst.has_edge("e23") is True
    assert mst.has_edge("e34") is True
    # Cycle-forming edges must NOT be in the tree
    assert mst.has_edge("e13") is False
    assert mst.has_edge("e24") is False


def test_cognitive_graph_minimum_spanning_tree() -> None:
    """CognitiveGraph extracts minimal spanning spine and preserves all nodes."""
    graph = CognitiveGraph(enable_native_substrate=False)

    for nid in ["A", "B", "C"]:
        graph.add_node(FactNode(id=nid, claim=nid))

    # Triangle: A-B (w=1.5), B-C (w=2.5), A-C (w=10.0)
    graph.add_edge(
        HCIREdge(id="e_ab", sources=["A"], targets=["B"], edge_type=HCIREdgeType.CAUSES, weight=1.5)
    )
    graph.add_edge(
        HCIREdge(id="e_bc", sources=["B"], targets=["C"], edge_type=HCIREdgeType.CAUSES, weight=2.5)
    )
    graph.add_edge(
        HCIREdge(
            id="e_ac", sources=["A"], targets=["C"], edge_type=HCIREdgeType.CAUSES, weight=10.0
        )
    )

    mst, total_weight = graph.minimum_spanning_tree()

    assert total_weight == pytest.approx(4.0)
    assert len(list(mst.all_nodes())) == 3
    assert len(list(mst.all_edges())) == 2
    mst_eids = {e.id for e in mst.all_edges()}
    assert mst_eids == {"e_ab", "e_bc"}


def test_engine_extract_geodesic_spine() -> None:
    """Engine extracts geodesic spine from cognitive graph via Kruskal's MST."""
    engine = AutonomousEpistemicEngine()

    graph = CognitiveGraph(enable_native_substrate=False)
    for nid in ["X", "Y", "Z"]:
        graph.add_node(FactNode(id=nid, claim=nid))
    graph.add_edge(
        HCIREdge(id="e1", sources=["X"], targets=["Y"], edge_type=HCIREdgeType.CAUSES, weight=2.0)
    )
    graph.add_edge(
        HCIREdge(id="e2", sources=["Y"], targets=["Z"], edge_type=HCIREdgeType.CAUSES, weight=3.0)
    )
    graph.add_edge(
        HCIREdge(id="e3", sources=["X"], targets=["Z"], edge_type=HCIREdgeType.CAUSES, weight=8.0)
    )

    mst, cost = engine.extract_geodesic_spine(graph)
    assert cost == pytest.approx(5.0)
    assert len(list(mst.all_edges())) == 2
