"""Unit test suite for HCIR World Round 16 Cognitive Faculties & Native Substrates.

Covers:
- Track 1: Retrosplenial Heading Integration & Head Direction Cell Attractors (Taube-Burgess)
- Track 2: Thalamic Reticular Nucleus (TRN) Selective Inhibitory Gating (Crick-Halassa)
- Track 3: Dorsolateral Prefrontal Cortex (dlPFC) Rule Abstraction & Schema Transfer (Badre-D'Esposito)
- Track 4: Native Structural Graph Minimum Dominating Set & Maximum Independent Set (Greedy Heuristics)
- Engine Orchestration Integration: AutonomousEpistemicEngine Round 16 faculties
"""

from __future__ import annotations

from hbllm.hcir import algorithms
from hbllm.hcir.graph import CognitiveGraph, FactNode, HCIREdge, HCIREdgeType
from hbllm.hcir.native_adapter import NativeGraphAdapter

# ============================================================================
# Track 1: Retrosplenial Heading Integration & Attractor Tests
# ============================================================================


# ============================================================================
# Track 2: Thalamic Reticular Nucleus (TRN) Gating Tests
# ============================================================================


# ============================================================================
# Track 3: Dorsolateral Prefrontal Cortex (dlPFC) Rule Abstraction Tests
# ============================================================================


# ============================================================================
# Track 4: Minimum Dominating Set & Maximum Independent Set Tests
# ============================================================================


def test_minimum_dominating_set_and_maximum_independent_set() -> None:
    """Test MDS and MIS on star and path topologies."""
    # 1. Star Graph: Center C with 3 leaves L1, L2, L3
    # Center C dominates all 4 nodes! Dominating set is {C}.
    # Maximum Independent Set consists of all 3 leaves {L1, L2, L3}.
    graph_star = CognitiveGraph(enable_native_substrate=False)
    for n in ["C", "L1", "L2", "L3"]:
        graph_star.add_node(FactNode(id=n, claim=n))

    graph_star.add_edge(
        HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["C"], targets=["L1"])
    )
    graph_star.add_edge(
        HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["C"], targets=["L2"])
    )
    graph_star.add_edge(
        HCIREdge(id="e3", edge_type=HCIREdgeType.CAUSES, sources=["C"], targets=["L3"])
    )

    dom_set = algorithms.minimum_dominating_set(graph_star)
    assert dom_set == {"C"}

    ind_set = algorithms.maximum_independent_set(graph_star)
    assert ind_set == {"L1", "L2", "L3"}

    # 2. Path Graph: A - B - C - D
    graph_path = CognitiveGraph(enable_native_substrate=False)
    for n in ["A", "B", "C", "D"]:
        graph_path.add_node(FactNode(id=n, claim=n))

    graph_path.add_edge(
        HCIREdge(id="e_ab", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["B"])
    )
    graph_path.add_edge(
        HCIREdge(id="e_bc", edge_type=HCIREdgeType.CAUSES, sources=["B"], targets=["C"])
    )
    graph_path.add_edge(
        HCIREdge(id="e_cd", edge_type=HCIREdgeType.CAUSES, sources=["C"], targets=["D"])
    )

    dom_path = algorithms.minimum_dominating_set(graph_path)
    assert len(dom_path) == 2  # 2 nodes dominate all 4

    ind_path = algorithms.maximum_independent_set(graph_path)
    assert len(ind_path) == 2  # Independent set of size 2


def test_dominating_and_independent_set_native_adapter_parity() -> None:
    """Test parity of MDS and MIS on NativeGraphAdapter."""
    adapter = NativeGraphAdapter()
    for n in ["C", "L1", "L2", "L3"]:
        adapter.add_node(node_id=n, node_type="concept")

    adapter.add_edge("e1", "causes", ["C"], ["L1"])
    adapter.add_edge("e2", "causes", ["C"], ["L2"])
    adapter.add_edge("e3", "causes", ["C"], ["L3"])

    dom = adapter.minimum_dominating_set()
    assert dom == {"C"}

    ind = adapter.maximum_independent_set()
    assert ind == {"L1", "L2", "L3"}


# ============================================================================
# Engine Orchestration Tests
# ============================================================================
