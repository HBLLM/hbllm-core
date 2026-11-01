from __future__ import annotations

import inspect

from hbllm.hcir.algorithms import (
    GraphView,
    articulation_points_and_bridges,
    betweenness_centrality,
    bfs_path,
    closeness_centrality,
    connected_components,
    dijkstra,
    edmonds_karp_max_flow,
    min_st_cut,
    minimum_spanning_tree,
    transitive_closure,
    transitive_reduction,
)
from hbllm.hcir.graph import (
    CognitiveCategory,
    CognitiveGraph,
    HCIREdge,
    HCIREdgeType,
    HCIRNode,
    HCIRNodeType,
)


class MockGraphView:
    """Mock implementation of GraphView protocol."""

    def __init__(self) -> None:
        self.nodes = {"A", "B", "C", "D"}
        self.adj = {"A": ["B", "C"], "B": ["C", "D"], "C": ["D"], "D": []}
        self.weights = {
            ("A", "B"): 1.0,
            ("A", "C"): 2.0,
            ("B", "C"): 1.0,
            ("B", "D"): 3.0,
            ("C", "D"): 1.0,
        }

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.weights)

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    def neighbors(self, node_id: str) -> list[str]:
        return self.adj.get(node_id, [])

    def edges_from(self, node_id: str) -> list[dict]:
        return [
            {"targets": [v], "weight": self.weights.get((node_id, v), 1.0)}
            for v in self.adj.get(node_id, [])
        ]

    def all_nodes(self) -> list[str]:
        return list(self.nodes)

    def all_edges(self) -> list[dict]:
        return [{"sources": [u], "targets": [v], "weight": w} for (u, v), w in self.weights.items()]

    def weight(self, u: str, v: str) -> float:
        return self.weights.get((u, v), 1.0)


def test_mock_graphview_protocol() -> None:
    mock = MockGraphView()
    assert isinstance(mock, GraphView)

    path = bfs_path(mock, "A", "D")
    assert path == ["A", "B", "D"] or path == ["A", "C", "D"]

    dist, _ = dijkstra(mock, "A", "D")
    assert dist["D"] == 3.0  # A -> B -> C -> D = 1 + 1 + 1 = 3.0

    comps = connected_components(mock)
    assert len(comps) == 1

    max_f, _ = edmonds_karp_max_flow(mock, "A", "D")
    assert max_f > 0.0

    s, t, cut_w = min_st_cut(mock, "A", "D")
    assert "A" in s
    assert "D" in t
    assert cut_w == max_f


def test_algorithms_on_cognitive_graph() -> None:
    cg = CognitiveGraph(enable_native_substrate=False)
    for name in ["A", "B", "C", "D"]:
        cg.add_node(
            HCIRNode(
                id=name,
                node_type=HCIRNodeType.CONCEPT,
                category=CognitiveCategory.MEMORY,
            )
        )

    cg.add_edge(
        HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["B"], weight=1.0)
    )
    cg.add_edge(
        HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["B"], targets=["C"], weight=1.0)
    )
    cg.add_edge(
        HCIREdge(id="e3", edge_type=HCIREdgeType.CAUSES, sources=["C"], targets=["D"], weight=1.0)
    )
    cg.add_edge(
        HCIREdge(id="e4", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["C"], weight=5.0)
    )

    # Path algorithms
    path = bfs_path(cg, "A", "D")
    assert path == ["A", "B", "C", "D"] or path == ["A", "C", "D"]

    dist, _ = dijkstra(cg, "A", "D")
    assert dist["D"] == 3.0

    # Spanning tree
    mst, total_w = minimum_spanning_tree(cg)
    assert len(mst) == 3
    assert total_w == 3.0

    # Connectivity
    comps = connected_components(cg)
    assert len(comps) == 1
    art_points, bridges = articulation_points_and_bridges(cg)

    # Transitive reduction & closure
    red = transitive_reduction(cg)
    assert ("A", "C") not in red  # A -> C is bypassed by A -> B -> C

    clos = transitive_closure(cg)
    assert ("A", "D") in clos

    # Centrality
    bc = betweenness_centrality(cg)
    assert isinstance(bc, dict)

    cc = closeness_centrality(cg, "B")
    assert cc > 0.0


def test_cognitive_graph_line_count_gate() -> None:
    """WP5 Acceptance Gate: CognitiveGraph must be < 1800 lines."""
    lines = inspect.getsourcelines(CognitiveGraph)[0]
    assert len(lines) < 1800, f"CognitiveGraph is {len(lines)} lines; must be < 1800"
