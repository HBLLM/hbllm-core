"""Unit tests for HCIR Round 17 Cognitive Substrate Enhancements.

Tracks Tested:
1. Track 1: Hippocampal CA1 Temporal Distance Cells & Relative Interval Timing (hippocampal_time_cell.py)
2. Track 2: Ventrolateral Prefrontal Cortex (vlPFC / rIFG) Motor Rule Retrieval & Response Inhibition (ventrolateral_motor_inhibition.py)
3. Track 3: Claustrum Dynamic Inter-Hemispheric Synchrony & Cross-Modal Gestalt Binding (claustrum_interhemispheric.py)
4. Track 4: Native Structural Graph Minimum Vertex Cover & Greedy Edge Coloring (native_adapter.py & graph.py)
5. Track 5: Autonomous Epistemic Engine Round 17 Integrated Orchestration
"""

from __future__ import annotations

from hbllm.hcir import algorithms
from hbllm.hcir.graph import (
    CognitiveGraph,
    FactNode,
    HCIREdge,
    HCIREdgeType,
)
from hbllm.hcir.native_adapter import NativeGraphAdapter


class TestStructuralGraphVertexCoverAndColoring:
    """Track 4: Minimum Vertex Cover & Greedy Edge Coloring in Native Adapter & CognitiveGraph."""

    def test_minimum_vertex_cover(self) -> None:
        graph = CognitiveGraph(enable_native_substrate=False)
        for node in ["A", "B", "C", "D"]:
            graph.add_node(FactNode(id=node, claim=node))

        # Star graph with center A and leaves B, C, D
        graph.add_edge(
            HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["B"])
        )
        graph.add_edge(
            HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["C"])
        )
        graph.add_edge(
            HCIREdge(id="e3", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["D"])
        )

        mvc = algorithms.minimum_vertex_cover(graph)
        # In a star graph, the single center vertex covers all 3 edges
        assert mvc == {"A"}

        # Triangle graph K3: A-B, B-C, C-A
        graph_tri = CognitiveGraph(enable_native_substrate=False)
        for node in ["A", "B", "C"]:
            graph_tri.add_node(FactNode(id=node, claim=node))
        graph_tri.add_edge(
            HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["B"])
        )
        graph_tri.add_edge(
            HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["B"], targets=["C"])
        )
        graph_tri.add_edge(
            HCIREdge(id="e3", edge_type=HCIREdgeType.CAUSES, sources=["C"], targets=["A"])
        )

        mvc_tri = algorithms.minimum_vertex_cover(graph_tri)
        assert len(mvc_tri) == 2
        # Verify valid cover: every edge has at least one endpoint in mvc_tri
        edges = [("A", "B"), ("B", "C"), ("C", "A")]
        for u, v in edges:
            assert u in mvc_tri or v in mvc_tri

        # Parity check on NativeGraphAdapter
        adapter = NativeGraphAdapter()
        for node in ["A", "B", "C"]:
            adapter.add_node(node, node_type="concept")
        adapter.add_edge("e1", "causes", ["A"], ["B"])
        adapter.add_edge("e2", "causes", ["B"], ["C"])
        adapter.add_edge("e3", "causes", ["C"], ["A"])
        mvc_adapter = adapter.minimum_vertex_cover()
        assert len(mvc_adapter) == 2
        for u, v in edges:
            assert u in mvc_adapter or v in mvc_adapter

    def test_greedy_edge_coloring(self) -> None:
        graph = CognitiveGraph(enable_native_substrate=False)
        for node in ["A", "B", "C", "D"]:
            graph.add_node(FactNode(id=node, claim=node))

        # Path graph: A-B, B-C, C-D (max degree = 2)
        graph.add_edge(
            HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["B"])
        )
        graph.add_edge(
            HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["B"], targets=["C"])
        )
        graph.add_edge(
            HCIREdge(id="e3", edge_type=HCIREdgeType.CAUSES, sources=["C"], targets=["D"])
        )

        coloring = algorithms.greedy_edge_coloring(graph)
        assert len(coloring) == 3

        # Edges incident to B: (A, B) and (B, C) must have different colors
        c_ab = coloring[("A", "B")]
        c_bc = coloring[("B", "C")]
        c_cd = coloring[("C", "D")]
        assert c_ab != c_bc
        assert c_bc != c_cd

        # Parity check on NativeGraphAdapter
        adapter = NativeGraphAdapter()
        for node in ["A", "B", "C", "D"]:
            adapter.add_node(node, node_type="concept")
        adapter.add_edge("e1", "causes", ["A"], ["B"])
        adapter.add_edge("e2", "causes", ["B"], ["C"])
        adapter.add_edge("e3", "causes", ["C"], ["D"])

        coloring_adapter = adapter.greedy_edge_coloring()
        assert len(coloring_adapter) == 3
        assert coloring_adapter[("A", "B")] != coloring_adapter[("B", "C")]
        assert coloring_adapter[("B", "C")] != coloring_adapter[("C", "D")]
