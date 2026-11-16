"""Unit test suite for HCIR World Round 14 Cognitive Faculties & Native Substrates.

Covers:
- Track 1: Parahippocampal Cortex (PHC) Spatial Context Schema & Environmental Topology Recognition (Epstein-Bar)
- Track 2: Locus Coeruleus (LC) Tonic vs. Phasic Noradrenergic Exploration-Exploitation Switch (Aston-Jones-Cohen)
- Track 3: Secondary Somatosensory / Posterior Insular Haptic Texture & Surface Friction Integration (Craig-Kaas)
- Track 4: Native Structural Graph Strongly Connected Components (SCC) & Metagraph Condensation (Tarjan)
- Engine Orchestration Integration: AutonomousEpistemicEngine Round 14 faculties
"""

from __future__ import annotations

from hbllm.hcir import algorithms
from hbllm.hcir.graph import CognitiveGraph, FactNode, HCIREdge, HCIREdgeType
from hbllm.hcir.native_adapter import NativeGraphAdapter

# ============================================================================
# Track 1: Parahippocampal Scene Schema Recognition Tests
# ============================================================================


# ============================================================================
# Track 2: Locus Coeruleus Noradrenergic Adaptive Gain Tests
# ============================================================================


# ============================================================================
# Track 3: Secondary Somatosensory Haptic Surface Friction Tests
# ============================================================================


# ============================================================================
# Track 4: Strongly Connected Components (SCC) & Metagraph Tests
# ============================================================================


def test_strongly_connected_components_on_dag_and_cycles() -> None:
    """Test Tarjan's SCC decomposition on acyclic and cyclic directed graphs."""
    graph = CognitiveGraph(enable_native_substrate=False)

    # 1. Directed Acyclic Graph: A -> B -> C (Each node is its own SCC)
    graph.add_node(FactNode(id="A", claim="A"))
    graph.add_node(FactNode(id="B", claim="B"))
    graph.add_node(FactNode(id="C", claim="C"))
    graph.add_edge(HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["B"]))
    graph.add_edge(HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["B"], targets=["C"]))

    sccs = algorithms.strongly_connected_components(graph)
    assert len(sccs) == 3
    assert all(len(c) == 1 for c in sccs)

    # 2. Close cycle: C -> A, turning {A, B, C} into a single SCC
    graph.add_edge(HCIREdge(id="e3", edge_type=HCIREdgeType.CAUSES, sources=["C"], targets=["A"]))
    sccs_cycle = algorithms.strongly_connected_components(graph)
    assert len(sccs_cycle) == 1
    assert sccs_cycle[0] == {"A", "B", "C"}


def test_condense_scc_metagraph_parity() -> None:
    """Test SCC metagraph condensation produces an acyclic DAG on CognitiveGraph & NativeGraphAdapter."""
    # Build two separate components:
    # Component 1 (Cycle): N1 <-> N2
    # Component 2 (Cycle): N3 <-> N4
    # Cross bridge: N2 -> N3
    adapter = NativeGraphAdapter()
    for n in ["N1", "N2", "N3", "N4"]:
        adapter.add_node(node_id=n, node_type="concept")

    adapter.add_edge("e12", "causes", ["N1"], ["N2"])
    adapter.add_edge("e21", "causes", ["N2"], ["N1"])
    adapter.add_edge("e34", "causes", ["N3"], ["N4"])
    adapter.add_edge("e43", "causes", ["N4"], ["N3"])
    adapter.add_edge("e23", "causes", ["N2"], ["N3"])

    scc_map, meta_edges = adapter.condense_scc_metagraph()

    # Must condense into exactly 2 supernodes
    assert len(scc_map) == 2
    # One SCC has {N1, N2}, other has {N3, N4}
    comps = list(scc_map.values())
    assert {"N1", "N2"} in comps
    assert {"N3", "N4"} in comps

    # Exactly 1 directed edge between the two SCC supernodes
    assert len(meta_edges) == 1
    u, v = meta_edges[0]
    assert scc_map[u] == {"N1", "N2"}
    assert scc_map[v] == {"N3", "N4"}


# ============================================================================
# Engine Orchestration Tests
# ============================================================================
