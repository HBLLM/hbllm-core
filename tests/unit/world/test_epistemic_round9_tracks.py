from __future__ import annotations

"""Unit Tests for Round 9 Surviving Cognitive Architecture Enhancements:

Track 3: Prefrontal Counterfactual Blame Attribution & Causal Credit Assignment
Track 4: Native Structural Graph Minimum Spanning Tree (MST) & Geodesic Spine Extraction
"""


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
)

# ─────────────────────────────────────────────────────────────────────────────
# Track 3: Prefrontal Counterfactual Blame Attribution
# ─────────────────────────────────────────────────────────────────────────────


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
