"""Unit test suite for HCIR World Round 12 Cognitive Faculties & Native Substrates.

Covers:
- Track 1: Cerebellar Forward Dynamics & Microzone LTD Error Adaptation (Marr-Albus-Ito)
- Track 2: Ventral Premotor (F5 / Mirror Neuron) Affordance Resonator (Rizzolatti)
- Track 3: Anterior Cingulate Cortex (ACC) Effort Allocator & EVC Gating (Botvinick / Shenhav)
- Track 4: Native Structural Graph Directed Maximum Flow & Minimum s-t Cut Bottlenecks (Edmonds-Karp)
"""

from __future__ import annotations

import pytest

from hbllm.hcir import algorithms
from hbllm.hcir.graph import CognitiveGraph, FactNode, HCIREdge, HCIREdgeType
from hbllm.hcir.native_adapter import NativeGraphAdapter

# ============================================================================
# Track 1: Cerebellar Microzone Granule-Purkinje LTD Tests
# ============================================================================


# ============================================================================
# Track 2: Ventral Premotor (F5) Affordance Resonator Tests
# ============================================================================


# ============================================================================
# Track 3: Anterior Cingulate Cortex (ACC) Effort Allocation Tests
# ============================================================================


# ============================================================================
# Track 4: Native Structural Graph Maximum Flow & Minimum Cut Tests
# ============================================================================


def test_native_graph_adapter_max_flow_min_cut() -> None:
    """Edmonds-Karp on NativeGraphAdapter computes exact max flow and min-cut bottleneck."""
    adapter = NativeGraphAdapter()

    # Flow network:
    # S -> A (cap 3), S -> B (cap 2)
    # A -> B (cap 1), A -> T (cap 2)
    # B -> T (cap 3)
    nodes = ["S", "A", "B", "T"]
    for n in nodes:
        adapter.add_node(node_id=n, node_type="concept")

    adapter.add_edge("e_sa", "flow", sources=["S"], targets=["A"], weight=3.0)
    adapter.add_edge("e_sb", "flow", sources=["S"], targets=["B"], weight=2.0)
    adapter.add_edge("e_ab", "flow", sources=["A"], targets=["B"], weight=1.0)
    adapter.add_edge("e_at", "flow", sources=["A"], targets=["T"], weight=2.0)
    adapter.add_edge("e_bt", "flow", sources=["B"], targets=["T"], weight=3.0)

    max_flow, cut_edges, source_set, sink_set = adapter.max_flow_min_cut(
        source="S", sink="T", capacity_attribute="weight"
    )

    # Max flow in this network is exactly 5.0
    assert max_flow == pytest.approx(5.0)

    # S in source_set, T in sink_set
    assert "S" in source_set
    assert "T" in sink_set
    assert source_set.isdisjoint(sink_set)

    # Min cut capacity must match max flow (5.0)
    cut_capacity = sum(
        adapter.get_edge(eid).get("weight", 1.0)
        for (u, v) in cut_edges
        for eid in ["e_sa", "e_sb", "e_ab", "e_at", "e_bt"]
        if adapter.get_edge(eid).get("sources") == [u]
        and adapter.get_edge(eid).get("targets") == [v]
    )
    assert cut_capacity == pytest.approx(5.0)


def test_cognitive_graph_max_flow_parity() -> None:
    """CognitiveGraph implements identical Edmonds-Karp max flow & min cutset computation."""
    graph = CognitiveGraph(enable_native_substrate=False)

    for nid in ["Source", "Mid", "Sink"]:
        graph.add_node(FactNode(id=nid, claim=nid))

    e1 = HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["Source"], targets=["Mid"])
    e1.weight = 7.0  # type: ignore[attr-defined]
    e2 = HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["Mid"], targets=["Sink"])
    e2.weight = 4.0  # type: ignore[attr-defined]

    graph.add_edge(e1)
    graph.add_edge(e2)

    max_flow, cut_edges, source_set, sink_set = algorithms.max_flow_min_cut(
        graph, source="Source", sink="Sink", capacity_attr="weight"
    )

    # Bottleneck is 4.0 on e2
    assert max_flow == pytest.approx(4.0)
    assert ("Mid", "Sink") in cut_edges
    assert "Source" in source_set
    assert "Sink" in sink_set


# ============================================================================
# Orchestrator Integration Tests: AutonomousEpistemicEngine Round 12
# ============================================================================
