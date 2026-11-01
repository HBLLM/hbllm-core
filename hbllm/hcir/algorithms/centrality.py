from __future__ import annotations

"""Pure Centrality Metrics Algorithms (WP5)."""

from collections import defaultdict, deque
from typing import Any

from hbllm.hcir.algorithms.paths import dijkstra


def betweenness_centrality(
    graph: Any, normalized: bool = True, directed: bool = False
) -> dict[str, float]:
    """Compute Brandes' betweenness centrality for all nodes."""
    all_nodes: list[str] = []
    if hasattr(graph, "all_nodes"):
        all_nodes = [n.id if hasattr(n, "id") else str(n) for n in graph.all_nodes()]
    elif hasattr(graph, "all_node_ids"):
        all_nodes = list(graph.all_node_ids())
    else:
        all_nodes = list(getattr(graph, "_nodes", {}).keys())

    all_nodes_set = set(all_nodes)
    cb: dict[str, float] = {v: 0.0 for v in all_nodes_set}
    n = len(all_nodes_set)
    if n <= 2:
        return cb

    adj: dict[str, list[str]] = defaultdict(list)
    all_edges = (
        list(graph.all_edges())
        if hasattr(graph, "all_edges")
        else list(getattr(graph, "_edges", {}).values())
    )
    for edge in all_edges:
        sources = edge.sources if hasattr(edge, "sources") else edge.get("sources", [])
        targets = edge.targets if hasattr(edge, "targets") else edge.get("targets", [])
        for s in sources:
            if s in all_nodes_set:
                for t in targets:
                    if t in all_nodes_set and s != t:
                        adj[s].append(t)
                        if not directed:
                            adj[t].append(s)

    for s in all_nodes_set:
        stack: list[str] = []
        p: dict[str, list[str]] = {w: [] for w in all_nodes_set}
        sigma: dict[str, float] = {w: 0.0 for w in all_nodes_set}
        sigma[s] = 1.0
        d: dict[str, int] = {w: -1 for w in all_nodes_set}
        d[s] = 0
        queue: deque[str] = deque([s])

        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in adj[v]:
                if d[w] < 0:
                    d[w] = d[v] + 1
                    queue.append(w)
                if d[w] == d[v] + 1:
                    sigma[w] += sigma[v]
                    p[w].append(v)

        delta: dict[str, float] = {w: 0.0 for w in all_nodes_set}
        while stack:
            w = stack.pop()
            for v in p[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                cb[w] += delta[w]

    if not directed:
        for v in cb:
            cb[v] /= 2.0

    if normalized and n > 2:
        scale = 1.0 / ((n - 1) * (n - 2)) if directed else 2.0 / ((n - 1) * (n - 2))
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
