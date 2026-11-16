"""Unit tests for HCIR World Round 20 — The Grand Capstone:
1. Global Neuronal Workspace (GNWT) Non-Linear Ignition & Conscious Broadcast (Dehaene / Mashour)
2. Native Structural Graph All-Pairs Shortest Paths, Radius, Diameter & Betweenness Centrality (Brandes)
3. Canonical 5-stage pipeline via AutonomousEpistemicEngine.decide (WP1)
4. Full Substrate Capstone Invariant Verification across all 20 Evolutionary Rounds
"""

from __future__ import annotations

from hbllm.hcir import algorithms
from hbllm.hcir.graph import CognitiveGraph, FactNode, HCIREdge, HCIREdgeType
from hbllm.hcir.native_adapter import NativeGraphAdapter


def test_graph_all_pairs_shortest_paths_radius_diameter_and_betweenness() -> None:
    """Test graph metrics (APSP, radius, diameter, betweenness centrality) on line graph A-B-C."""
    adapter = NativeGraphAdapter()

    # Undirected line graph: A - B - C
    adapter.add_node("A", "FACT")
    adapter.add_node("B", "FACT")
    adapter.add_node("C", "FACT")

    adapter.add_edge("e_ab", "NEIGHBOR", ["A"], ["B"], 1.0)
    adapter.add_edge("e_bc", "NEIGHBOR", ["B"], ["C"], 1.0)

    # APSP
    sp, _ = adapter.all_pairs_shortest_paths()
    assert sp[("A", "B")] == 1.0
    assert sp[("B", "C")] == 1.0
    assert sp[("A", "C")] == 2.0

    # Radius and diameter
    rad, diam = adapter.graph_radius_and_diameter()
    assert rad == 1.0  # Center node B has eccentricity 1.0
    assert diam == 2.0  # Endpoints A and C have eccentricity 2.0

    # Betweenness centrality: B sits on the only path between A and C
    cb = adapter.betweenness_centrality(normalized=True, directed=False)
    assert cb["B"] > cb["A"]
    assert cb["B"] > cb["C"]
    assert cb["A"] == 0.0
    assert cb["C"] == 0.0
    assert cb["B"] == 1.0  # (n=3: normalized 1 path / ((2*1)/2) = 1.0)

    # CognitiveGraph equivalence test
    graph = CognitiveGraph(enable_native_substrate=False)
    graph.add_node(FactNode(id="A", claim="A"))
    graph.add_node(FactNode(id="B", claim="B"))
    graph.add_node(FactNode(id="C", claim="C"))

    graph.add_edge(
        HCIREdge(sources=["A"], targets=["B"], edge_type=HCIREdgeType.DEPENDS_ON, weight=1.0)
    )
    graph.add_edge(
        HCIREdge(sources=["B"], targets=["C"], edge_type=HCIREdgeType.DEPENDS_ON, weight=1.0)
    )

    graph_sp, _ = algorithms.all_pairs_shortest_paths(graph)
    assert graph_sp[("A", "C")] == 2.0

    graph_rad, graph_diam = algorithms.graph_radius_and_diameter(graph)
    assert graph_rad == 1.0
    assert graph_diam == 2.0

    graph_cb = algorithms.betweenness_centrality(graph, normalized=True, directed=False)
    assert graph_cb["B"] == 1.0
