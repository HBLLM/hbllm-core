"""WP5 timing report: heaviest hbllm.hcir.algorithms functions on a 2k-node graph.

Writes reports/algorithms/<unix_timestamp>.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

import hbllm.hcir.algorithms as algorithms
from hbllm.hcir.graph import CognitiveGraph, HCIREdge, HCIREdgeType
from hbllm.hcir.nodes import GoalNode

N_NODES = 2000
N_EDGES = 6000


def build_graph() -> CognitiveGraph:
    rng = np.random.default_rng(0)
    g = CognitiveGraph(enable_native_substrate=False)
    for i in range(N_NODES):
        g.add_node(GoalNode(id=f"n{i}"))
    for _ in range(N_EDGES):
        u, v = rng.integers(0, N_NODES, size=2)
        if u != v:
            e = HCIREdge(edge_type=HCIREdgeType.LOCATED_IN, sources=[f"n{u}"], targets=[f"n{v}"])
            g.add_edge(e)
    return g


def main() -> None:
    g = build_graph()
    print(f"graph: {g.node_count} nodes, {g.edge_count} edges")
    out_dir = Path("reports/algorithms")
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, float] = {}
    # NOTE: max_flow_min_cut (Edmonds-Karp O(V·E^2)) is intentionally excluded
    # at this scale — it does not terminate in reasonable time on 2k/6k graphs.
    candidates = {
        "all_pairs_shortest_paths": lambda: algorithms.all_pairs_shortest_paths(g),
        "betweenness_centrality": lambda: algorithms.betweenness_centrality(g),
        "minimum_spanning_tree_subgraph": lambda: algorithms.minimum_spanning_tree_subgraph(g),
        "strongly_connected_components": lambda: algorithms.strongly_connected_components(g),
        "chinese_postman_tour": lambda: algorithms.chinese_postman_tour(g),
        "feedback_vertex_set": lambda: algorithms.feedback_vertex_set(g),
    }
    for name, fn in candidates.items():
        t0 = time.perf_counter()
        fn()
        ms = (time.perf_counter() - t0) * 1000
        results[name] = round(ms, 2)
        print(f"{name}: {ms:.1f}ms")

    report = {
        "n_nodes": N_NODES,
        "n_edges": N_EDGES,
        "timings_ms": results,
        "heaviest": sorted(results, key=lambda k: results[k], reverse=True)[:3],
    }
    ts = int(time.time())
    out = out_dir / f"{ts}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
