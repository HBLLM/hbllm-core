"""Unit test suite for HCIR World Round 11 Cognitive Faculties & Native Substrates.

Covers:
- Track 1: Hippocampal Theta-Gamma Phase Precession & Macro Chunking (O'Keefe / Lisman)
- Track 2: Frontal Polar Cortex (FPC / BA10) Counterfactual Branch Switching (Koechlin / Rushworth)
- Track 3: Superior Colliculus Multisensory Alignment & Saccadic Arbiter (Stein & Meredith)
- Track 4: Native Structural Graph Articulation Points, Bridge Detection & Biconnectivity (Tarjan)
"""

from __future__ import annotations

from hbllm.hcir import algorithms
from hbllm.hcir.graph import CognitiveGraph, FactNode, HCIREdge, HCIREdgeType
from hbllm.hcir.native_adapter import NativeGraphAdapter

# ============================================================================
# Track 1: Hippocampal Theta-Gamma Phase Precession Tests
# ============================================================================


# ============================================================================
# Track 2: Frontal Polar Cortex (BA10) Exploration Horizon Tests
# ============================================================================


# ============================================================================
# Track 3: Superior Colliculus Multisensory Alignment Tests
# ============================================================================


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

    art_points, bridges = algorithms.articulation_points_and_bridges(graph)

    assert art_points == {"B"}
    assert len(bridges) == 2
    assert algorithms.is_biconnected(graph) is False


# ============================================================================
# Orchestrator Integration Tests: AutonomousEpistemicEngine Round 11
# ============================================================================
