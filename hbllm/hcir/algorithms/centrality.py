from __future__ import annotations

"""Pure Centrality Metrics Algorithms (WP5)."""

from collections import deque
from typing import Any

from hbllm.hcir.algorithms.paths import dijkstra


def betweenness_centrality(graph: Any, normalized: bool = True) -> dict[str, float]:
    """Compute Brandes' betweenness centrality for all nodes."""
    all_nodes: list[str] = []
    if hasattr(graph, "all_nodes"):
        all_nodes = [n.id if hasattr(n, "id") else str(n) for n in graph.all_nodes()]
    elif hasattr(graph, "all_node_ids"):
        all_nodes = list(graph.all_node_ids())
    else:
        all_nodes = list(getattr(graph, "_nodes", {}).keys())

    cb: dict[str, float] = {v: 0.0 for v in all_nodes}

    for s in all_nodes:
        S: list[str] = []
        P: dict[str, list[str]] = {w: [] for w in all_nodes}
        sigma: dict[str, int] = {w: 0 for w in all_nodes}
        sigma[s] = 1
        d: dict[str, int] = {w: -1 for w in all_nodes}
        d[s] = 0
        Q = deque([s])

        while Q:
            v = Q.popleft()
            S.append(v)
            nbrs = (
                graph.neighbors(v)
                if hasattr(graph, "neighbors")
                else (graph._neighbors(v) if hasattr(graph, "_neighbors") else [])
            )
            for w in nbrs:
                if w not in d:
                    continue
                if d[w] < 0:
                    Q.append(w)
                    d[w] = d[v] + 1
                if d[w] == d[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)

        delta: dict[str, float] = {w: 0.0 for w in all_nodes}
        while S:
            w = S.pop()
            for v in P[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                cb[w] += delta[w]

    if normalized and len(all_nodes) > 2:
        scale = 1.0 / ((len(all_nodes) - 1) * (len(all_nodes) - 2))
        for v in cb:
            cb[v] *= scale

    return cb


def closeness_centrality(graph: Any, node_id: str, weight_attr: str = "weight") -> float:
    """Compute closeness centrality for a target node."""
    dist, _ = dijkstra(graph, node_id, weight_attr=weight_attr)
    finite_dists = [d for v, d in dist.items() if v != node_id and d < float("inf")]
    if not finite_dists:
        return 0.0
    return float(len(finite_dists)) / sum(finite_dists)
