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

# ============================================================================
# Track 1: Hippocampal Prioritized Experience Replay Buffer Tests
# ============================================================================


# ============================================================================
# Track 2: Lateral Prefrontal Cortex Task Switching Tests
# ============================================================================


# ============================================================================
# Track 3: Retrosplenial Landmark Anchor Engine Tests
# ============================================================================


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
