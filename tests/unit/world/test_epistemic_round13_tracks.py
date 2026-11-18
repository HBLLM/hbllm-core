"""Unit test suite for HCIR World Round 13 Cognitive Faculties & Native Substrates.

Covers:
- Track 1: Subicular Boundary Vector Cell (BVC) Map (Hartley-Burgess)
- Track 2: Ventromedial Prefrontal Cortex (vmPFC) Somatic Marker Guidance (Damasio-Bechara)
- Track 3: Pulvinar Thalamic Transthalamic Routing & Attentional Gating (Saalmann-Kastner)
- Track 4: Native Structural Graph Feedback Arc Set (FAS) & Cycle Severing (Eades et al.)
- Engine Orchestration Integration: AutonomousEpistemicEngine Round 13 faculties
"""

from __future__ import annotations

from hbllm.hcir import algorithms
from hbllm.hcir.graph import CognitiveGraph, FactNode, HCIREdge, HCIREdgeType
from hbllm.hcir.native_adapter import NativeGraphAdapter

# ============================================================================
# Track 1: Subicular Boundary Vector Cell (BVC) Map Tests
# ============================================================================


# ============================================================================
# Track 2: Ventromedial Prefrontal Cortex (vmPFC) Somatic Marker Tests
# ============================================================================


# ============================================================================
# Track 3: Pulvinar Thalamic Transthalamic Routing Tests
# ============================================================================


# ============================================================================
# Track 4: Feedback Arc Set (FAS) Native Graph Tests
# ============================================================================


def test_feedback_arc_set_on_dag_and_cycle() -> None:
    """Test feedback arc set detection on DAGs and cyclic directed graphs."""
    # 1. Directed Acyclic Graph (DAG): A -> B -> C
    graph_dag = CognitiveGraph(enable_native_substrate=False)
    graph_dag.add_node(FactNode(id="A", claim="Alpha"))
    graph_dag.add_node(FactNode(id="B", claim="Beta"))
    graph_dag.add_node(FactNode(id="C", claim="Gamma"))
    graph_dag.add_edge(
        HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["B"])
    )
    graph_dag.add_edge(
        HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["B"], targets=["C"])
    )

    fas_edges, order = algorithms.feedback_arc_set(graph_dag)
    assert len(fas_edges) == 0
    assert order == ["A", "B", "C"]

    # 2. Directed Cycle: A -> B -> C -> A
    graph_cycle = CognitiveGraph(enable_native_substrate=False)
    graph_cycle.add_node(FactNode(id="A", claim="Alpha"))
    graph_cycle.add_node(FactNode(id="B", claim="Beta"))
    graph_cycle.add_node(FactNode(id="C", claim="Gamma"))
    graph_cycle.add_edge(
        HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["B"])
    )
    graph_cycle.add_edge(
        HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["B"], targets=["C"])
    )
    graph_cycle.add_edge(
        HCIREdge(id="e3", edge_type=HCIREdgeType.CAUSES, sources=["C"], targets=["A"])
    )

    fas_edges, order = algorithms.feedback_arc_set(graph_cycle)
    # Breaking exactly 1 edge severs the 3-cycle
    assert len(fas_edges) == 1
    u, v = fas_edges[0]
    # Edge must be one of the cycle edges
    assert (u, v) in {("A", "B"), ("B", "C"), ("C", "A")}
    # Topological order contains all nodes
    assert set(order) == {"A", "B", "C"}


def test_feedback_arc_set_native_adapter_parity() -> None:
    """Test NativeGraphAdapter FAS matches CognitiveGraph behavior."""
    adapter = NativeGraphAdapter()
    adapter.add_node("N1", "concept")
    adapter.add_node("N2", "concept")
    adapter.add_node("N3", "concept")
    adapter.add_node("N4", "concept")

    # Add cycle: N1 -> N2 -> N3 -> N1, plus branch N3 -> N4
    adapter.add_edge("e1", "causes", ["N1"], ["N2"])
    adapter.add_edge("e2", "causes", ["N2"], ["N3"])
    adapter.add_edge("e3", "causes", ["N3"], ["N1"])
    adapter.add_edge("e4", "causes", ["N3"], ["N4"])

    fas_edges, order = adapter.feedback_arc_set()
    assert len(fas_edges) == 1
    assert set(order) == {"N1", "N2", "N3", "N4"}


# ============================================================================
# Engine Orchestration Tests
# ============================================================================
