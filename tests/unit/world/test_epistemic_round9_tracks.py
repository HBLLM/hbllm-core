from __future__ import annotations

"""Unit Tests for Round 9 Surviving Cognitive Architecture Enhancements:

Track 3: Prefrontal Counterfactual Blame Attribution & Causal Credit Assignment
Track 4: Native Structural Graph Minimum Spanning Tree (MST) & Geodesic Spine Extraction
"""

from typing import Any

import pytest

from hbllm.hcir import algorithms
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
    PlanAttributionReport,
)
from hbllm.hcir.world.experimental import restore_all as _restore_all
from hbllm.hcir.world.experimental.prefrontal_blame_attribution import (
    PrefrontalBlameAttributor,
)

# ─────────────────────────────────────────────────────────────────────────────
# Track 3: Prefrontal Counterfactual Blame Attribution
# ─────────────────────────────────────────────────────────────────────────────


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
    _restore_all(engine)
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


# ─────────────────────────────────────────────────────────────────────────────
# Track 4: Native Structural Graph Minimum Spanning Tree (MST)
# ─────────────────────────────────────────────────────────────────────────────


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

    mst, total_weight = algorithms.minimum_spanning_tree_subgraph(graph)

    assert total_weight == pytest.approx(4.0)
    assert len(list(mst.all_nodes())) == 3
    assert len(list(mst.all_edges())) == 2
    mst_eids = {e.id for e in mst.all_edges()}
    assert mst_eids == {"e_ab", "e_bc"}


def test_engine_extract_geodesic_spine() -> None:
    """Engine extracts geodesic spine from cognitive graph via Kruskal's MST."""
    engine = AutonomousEpistemicEngine()
    _restore_all(engine)

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
