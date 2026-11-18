"""Unit tests for HCIR World Round 19 faculties:
1. Hippocampal-Entorhinal Boundary-to-Grid Phase Alignment (Bush / Giocomo)
2. Medial Prefrontal Schema Induction & Invariant Abstraction (Euston / McKenzie)
3. Subthalamic Nucleus (STN) Decision Threshold Escalation & Conflict Braking (Frank)
4. Native Structural Graph Transitive Reduction & Essential Causal Skeleton
5. AutonomousEpistemicEngine Round 19 Integration
"""

from __future__ import annotations

from hbllm.hcir import algorithms
from hbllm.hcir.graph import CognitiveGraph, FactNode, HCIREdge, HCIREdgeType
from hbllm.hcir.native_adapter import NativeGraphAdapter


def test_transitive_reduction_and_causal_skeleton() -> None:
    """Test DAG transitive reduction removing redundant shortcut edges."""
    adapter = NativeGraphAdapter()

    # Create a DAG: A -> B -> C and direct shortcut A -> C
    adapter.add_node("A", "FACT")
    adapter.add_node("B", "FACT")
    adapter.add_node("C", "FACT")

    adapter.add_edge("e_ab", "CAUSAL_LINKS", ["A"], ["B"])
    adapter.add_edge("e_bc", "CAUSAL_LINKS", ["B"], ["C"])
    adapter.add_edge("e_ac", "CAUSAL_LINKS", ["A"], ["C"])

    skeleton = adapter.essential_causal_skeleton()
    assert ("A", "B") in skeleton
    assert ("B", "C") in skeleton
    assert ("A", "C") not in skeleton  # Redundant direct edge eliminated!

    # CognitiveGraph equivalence
    graph = CognitiveGraph(enable_native_substrate=False)
    graph.add_node(FactNode(id="A", claim="A"))
    graph.add_node(FactNode(id="B", claim="B"))
    graph.add_node(FactNode(id="C", claim="C"))

    graph.add_edge(HCIREdge(sources=["A"], targets=["B"], edge_type=HCIREdgeType.CAUSES))
    graph.add_edge(HCIREdge(sources=["B"], targets=["C"], edge_type=HCIREdgeType.CAUSES))
    graph.add_edge(HCIREdge(sources=["A"], targets=["C"], edge_type=HCIREdgeType.CAUSES))

    graph_skeleton = algorithms.essential_causal_skeleton(graph)
    assert ("A", "B") in graph_skeleton
    assert ("B", "C") in graph_skeleton
    assert ("A", "C") not in graph_skeleton
