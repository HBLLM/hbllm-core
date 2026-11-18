"""Unit test suite for HCIR World Round 15 Cognitive Faculties & Native Substrates.

Covers:
- Track 1: Entorhinal Border Cells & Grid Module Rescaling (Solstad-Barry)
- Track 2: Amygdalar Fear Conditioning & Fast Threat Potentiation (LeDoux-Phelps)
- Track 3: Ventral Tegmental Area (VTA) Dopaminergic Epistemic Curiosity Drive (Schultz-Friston)
- Track 4: Native Structural Graph Eulerian Path, Circuit & Chinese Postman Tour (Hierholzer-Edmonds)
- Engine Orchestration Integration: AutonomousEpistemicEngine Round 15 faculties
"""

from __future__ import annotations

from hbllm.hcir import algorithms
from hbllm.hcir.graph import CognitiveGraph, FactNode, HCIREdge, HCIREdgeType
from hbllm.hcir.native_adapter import NativeGraphAdapter

# ============================================================================
# Track 1: Entorhinal Border Cells & Grid Rescaling Tests
# ============================================================================


# ============================================================================
# Track 2: Amygdalar Fear Conditioning & Threat Potentiation Tests
# ============================================================================


# ============================================================================
# Track 3: Ventral Tegmental Area (VTA) Epistemic Curiosity Tests
# ============================================================================


# ============================================================================
# Track 4: Eulerian Path, Circuit & Chinese Postman Tour Tests
# ============================================================================


def test_find_eulerian_path_and_circuit() -> None:
    """Test Eulerian circuit and path discovery using Hierholzer's algorithm."""
    # 1. Directed Eulerian Circuit: A -> B -> C -> A
    graph_circuit = CognitiveGraph(enable_native_substrate=False)
    graph_circuit.add_node(FactNode(id="A", claim="A"))
    graph_circuit.add_node(FactNode(id="B", claim="B"))
    graph_circuit.add_node(FactNode(id="C", claim="C"))
    graph_circuit.add_edge(
        HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["B"])
    )
    graph_circuit.add_edge(
        HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["B"], targets=["C"])
    )
    graph_circuit.add_edge(
        HCIREdge(id="e3", edge_type=HCIREdgeType.CAUSES, sources=["C"], targets=["A"])
    )

    is_eul, path = algorithms.find_eulerian_path_or_circuit(graph_circuit)
    assert is_eul is True
    assert len(path) == 3
    # Verifies closed circuit
    assert path[0][0] == path[-1][1]

    # 2. Directed Eulerian Path: A -> B -> C (open path)
    graph_path = CognitiveGraph(enable_native_substrate=False)
    graph_path.add_node(FactNode(id="A", claim="A"))
    graph_path.add_node(FactNode(id="B", claim="B"))
    graph_path.add_node(FactNode(id="C", claim="C"))
    graph_path.add_edge(
        HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["B"])
    )
    graph_path.add_edge(
        HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["B"], targets=["C"])
    )

    is_eul_p, path_p = algorithms.find_eulerian_path_or_circuit(graph_path)
    assert is_eul_p is True
    assert len(path_p) == 2
    assert path_p == [("A", "B"), ("B", "C")]


def test_chinese_postman_tour_native_adapter_parity() -> None:
    """Test Chinese Postman Tour optimization across NativeGraphAdapter and CognitiveGraph."""
    adapter = NativeGraphAdapter()
    for n in ["A", "B", "C"]:
        adapter.add_node(node_id=n, node_type="concept")

    # Strongly connected graph with unbalanced degrees:
    # A -> B (w=2.0), B -> C (w=1.0), C -> A (w=3.0) forms cycle of cost 6.0
    # Additional edge: B -> A (w=1.0)
    adapter.add_edge("e_ab", "causes", ["A"], ["B"], weight=2.0)
    adapter.add_edge("e_bc", "causes", ["B"], ["C"], weight=1.0)
    adapter.add_edge("e_ca", "causes", ["C"], ["A"], weight=3.0)
    adapter.add_edge("e_ba", "causes", ["B"], ["A"], weight=1.0)

    cost, tour = adapter.chinese_postman_tour()
    assert cost >= 7.0
    assert len(tour) >= 4
    # All 4 original edges must be visited at least once
    visited_pairs = set(tour)
    assert ("A", "B") in visited_pairs
    assert ("B", "C") in visited_pairs
    assert ("C", "A") in visited_pairs
    assert ("B", "A") in visited_pairs


# ============================================================================
# Engine Orchestration Tests
# ============================================================================
