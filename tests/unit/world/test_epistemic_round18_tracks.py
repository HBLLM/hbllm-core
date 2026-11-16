"""Unit tests for HCIR Round 18 Cognitive Substrate Enhancements.

Tracks Tested:
1. Track 1: Anterior Insular Cortex (AIC) Subjective Time Dilation & Interoceptive Pacing (anterior_insular_timing.py)
2. Track 2: Orbitofrontal Cortex (OFC) Counterfactual Credit Assignment & Inferred State Value Updating (orbitofrontal_credit_assignment.py)
3. Track 3: Pedunculopontine Nucleus (PPN / MLR) Locomotor Velocity Modulation & Obstacle Deceleration (pedunculopontine_locomotor.py)
4. Track 4: Native Structural Graph Feedback Vertex Set (FVS) & Cycle Transversal Discovery (native_adapter.py & graph.py)
5. Track 5: Autonomous Epistemic Engine Round 18 Integrated Orchestration
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


class TestStructuralGraphFeedbackVertexSet:
    """Track 4: Feedback Vertex Set (FVS) & Cycle Transversal in Native Adapter & CognitiveGraph."""

    def test_feedback_vertex_set_acyclic(self) -> None:
        graph = CognitiveGraph(enable_native_substrate=False)
        for n in ["A", "B", "C"]:
            graph.add_node(FactNode(id=n, claim=n))
        graph.add_edge(
            HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["B"])
        )
        graph.add_edge(
            HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["B"], targets=["C"])
        )

        # Acyclic directed path: FVS should be empty
        assert algorithms.feedback_vertex_set(graph) == set()

        adapter = NativeGraphAdapter()
        for n in ["A", "B", "C"]:
            adapter.add_node(n, node_type="concept")
        adapter.add_edge("e1", "causes", ["A"], ["B"])
        adapter.add_edge("e2", "causes", ["B"], ["C"])
        assert adapter.feedback_vertex_set() == set()

    def test_feedback_vertex_set_single_and_figure8_cycles(self) -> None:
        # 1. Simple 3-node cycle: A -> B -> C -> A
        graph_cycle = CognitiveGraph(enable_native_substrate=False)
        for n in ["A", "B", "C"]:
            graph_cycle.add_node(FactNode(id=n, claim=n))
        graph_cycle.add_edge(
            HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["B"])
        )
        graph_cycle.add_edge(
            HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["B"], targets=["C"])
        )
        graph_cycle.add_edge(
            HCIREdge(id="e3", edge_type=HCIREdgeType.CAUSES, sources=["C"], targets=["A"])
        )

        fvs_cycle = algorithms.feedback_vertex_set(graph_cycle)
        assert len(fvs_cycle) == 1
        assert list(fvs_cycle)[0] in {"A", "B", "C"}

        # 2. Figure-8 with shared hub node A:
        # Cycle 1: A -> B -> A
        # Cycle 2: A -> C -> A
        # Removing node A breaks BOTH cycles simultaneously -> minimal FVS = {A}!
        graph_fig8 = CognitiveGraph(enable_native_substrate=False)
        for n in ["A", "B", "C"]:
            graph_fig8.add_node(FactNode(id=n, claim=n))
        graph_fig8.add_edge(
            HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["B"])
        )
        graph_fig8.add_edge(
            HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["B"], targets=["A"])
        )
        graph_fig8.add_edge(
            HCIREdge(id="e3", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["C"])
        )
        graph_fig8.add_edge(
            HCIREdge(id="e4", edge_type=HCIREdgeType.CAUSES, sources=["C"], targets=["A"])
        )

        fvs_fig8 = algorithms.feedback_vertex_set(graph_fig8)
        assert fvs_fig8 == {"A"}

        # Parity check on NativeGraphAdapter
        adapter_fig8 = NativeGraphAdapter()
        for n in ["A", "B", "C"]:
            adapter_fig8.add_node(n, node_type="concept")
        adapter_fig8.add_edge("e1", "causes", ["A"], ["B"])
        adapter_fig8.add_edge("e2", "causes", ["B"], ["A"])
        adapter_fig8.add_edge("e3", "causes", ["A"], ["C"])
        adapter_fig8.add_edge("e4", "causes", ["C"], ["A"])

        fvs_adapter = adapter_fig8.feedback_vertex_set()
        assert fvs_adapter == {"A"}
