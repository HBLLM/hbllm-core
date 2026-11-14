from __future__ import annotations

"""Unit Tests for Round 8 Surviving Cognitive Architecture Enhancements:

Track 4: Native Structural Graph Transitive Reduction & Minimal Causal Skeleton Extraction
"""


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
from hbllm.hcir.world.experimental import restore_all as _restore_all

# ─────────────────────────────────────────────────────────────────────────────
# Track 4: Native Structural Graph Transitive Reduction
# ─────────────────────────────────────────────────────────────────────────────


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
    reduced = algorithms.transitive_reduction_subgraph(graph)

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
