from __future__ import annotations

"""Pure Network Flow and Cut Algorithms (WP5)."""

from collections import deque
from typing import Any


def edmonds_karp_max_flow(
    graph: Any,
    source: str,
    sink: str,
    capacity_attr: str = "capacity",
) -> tuple[float, dict[tuple[str, str], float]]:
    """Compute maximum flow from source to sink using Edmonds-Karp (BFS augmenting paths)."""
    if source == sink:
        return 0.0, {}

    # Build adjacency and capacity mapping
    capacity: dict[tuple[str, str], float] = {}
    adj: dict[str, set[str]] = {}

    all_edges = (
        list(graph.all_edges())
        if hasattr(graph, "all_edges")
        else list(getattr(graph, "_edges", {}).values())
    )
    for e in all_edges:
        sources = e.sources if hasattr(e, "sources") else e.get("sources", [])
        targets = e.targets if hasattr(e, "targets") else e.get("targets", [])
        cap = float(
            getattr(
                e,
                capacity_attr,
                e.get(capacity_attr, getattr(e, "weight", 1.0))
                if isinstance(e, dict)
                else getattr(e, "weight", 1.0),
            )
        )
        for u in sources:
            for v in targets:
                capacity[(u, v)] = capacity.get((u, v), 0.0) + cap
                capacity.setdefault((v, u), 0.0)
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)

    flow: dict[tuple[str, str], float] = {k: 0.0 for k in capacity}
    max_flow = 0.0

    while True:
        parent: dict[str, str | None] = {source: None}
        queue = deque([source])

        while queue and sink not in parent:
            u = queue.popleft()
            for v in adj.get(u, set()):
                residual = capacity.get((u, v), 0.0) - flow.get((u, v), 0.0)
                if residual > 1e-9 and v not in parent:
                    parent[v] = u
                    queue.append(v)

        if sink not in parent:
            break

        # Find bottleneck capacity
        path_flow = float("inf")
        curr = sink
        while curr != source:
            prev = parent[curr]
            assert prev is not None
            path_flow = min(
                path_flow, capacity.get((prev, curr), 0.0) - flow.get((prev, curr), 0.0)
            )
            curr = prev

        curr = sink
        while curr != source:
            prev = parent[curr]
            assert prev is not None
            flow[(prev, curr)] = flow.get((prev, curr), 0.0) + path_flow
            flow[(curr, prev)] = flow.get((curr, prev), 0.0) - path_flow
            curr = prev

        max_flow += path_flow

    return max_flow, flow


def min_st_cut(
    graph: Any,
    source: str,
    sink: str,
    capacity_attr: str = "capacity",
) -> tuple[set[str], set[str], float]:
    """Compute the minimum s-t cut partitioning nodes into (S, T)."""
    max_f, flow = edmonds_karp_max_flow(graph, source, sink, capacity_attr=capacity_attr)

    # Reachable from source in residual graph
    all_edges = (
        list(graph.all_edges())
        if hasattr(graph, "all_edges")
        else list(getattr(graph, "_edges", {}).values())
    )
    capacity: dict[tuple[str, str], float] = {}
    adj: dict[str, set[str]] = {}
    all_nodes: set[str] = set()

    for e in all_edges:
        sources = e.sources if hasattr(e, "sources") else e.get("sources", [])
        targets = e.targets if hasattr(e, "targets") else e.get("targets", [])
        cap = float(
            getattr(
                e,
                capacity_attr,
                e.get(capacity_attr, getattr(e, "weight", 1.0))
                if isinstance(e, dict)
                else getattr(e, "weight", 1.0),
            )
        )
        for u in sources:
            all_nodes.add(u)
            for v in targets:
                all_nodes.add(v)
                capacity[(u, v)] = capacity.get((u, v), 0.0) + cap
                capacity.setdefault((v, u), 0.0)
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)

    S: set[str] = {source}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in adj.get(u, set()):
            residual = capacity.get((u, v), 0.0) - flow.get((u, v), 0.0)
            if residual > 1e-9 and v not in S:
                S.add(v)
                queue.append(v)

    T = all_nodes - S
    return S, T, max_f
