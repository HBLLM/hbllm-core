from __future__ import annotations

"""Pure Path Traversal and Distance Algorithms (WP5)."""

import heapq
from collections import deque
from typing import Any


def bfs_path(graph: Any, start: str, end: str, max_depth: int = 10) -> list[str]:
    """Find the shortest unweighted path using Breadth-First Search."""
    if start == end:
        return [start]
    queue = deque([(start, [start], 0)])
    visited = {start}

    while queue:
        curr, path, depth = queue.popleft()
        if depth >= max_depth:
            continue
        neighbors = (
            graph.neighbors(curr)
            if hasattr(graph, "neighbors")
            else (graph._neighbors(curr) if hasattr(graph, "_neighbors") else [])
        )
        for nbr in neighbors:
            if nbr == end:
                return path + [nbr]
            if nbr not in visited:
                visited.add(nbr)
                queue.append((nbr, path + [nbr], depth + 1))
    return []


def dijkstra(
    graph: Any,
    start: str,
    end: str | None = None,
    weight_attr: str = "weight",
) -> tuple[dict[str, float], dict[str, str]]:
    """Compute single-source shortest paths using Dijkstra's algorithm."""
    dist: dict[str, float] = {start: 0.0}
    prev: dict[str, str] = {}
    pq = [(0.0, start)]
    visited: set[str] = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if end is not None and u == end:
            break

        edges = graph.edges_from(u) if hasattr(graph, "edges_from") else []
        for e in edges:
            targets = e.targets if hasattr(e, "targets") else e.get("targets", [])
            w = float(
                getattr(e, weight_attr, e.get(weight_attr, 1.0) if isinstance(e, dict) else 1.0)
            )
            for v in targets:
                new_d = d + w
                if new_d < dist.get(v, float("inf")):
                    dist[v] = new_d
                    prev[v] = u
                    heapq.heappush(pq, (new_d, v))
    return dist, prev


def all_pairs_shortest_paths(
    graph: Any, weight_attr: str = "weight"
) -> dict[tuple[str, str], float]:
    """Compute all-pairs shortest paths using Floyd-Warshall or repeated Dijkstra."""
    all_nodes: list[str] = []
    if hasattr(graph, "all_nodes"):
        all_nodes = [n.id if hasattr(n, "id") else str(n) for n in graph.all_nodes()]
    elif hasattr(graph, "all_node_ids"):
        all_nodes = list(graph.all_node_ids())
    else:
        all_nodes = list(getattr(graph, "_nodes", {}).keys())

    distances: dict[tuple[str, str], float] = {}
    for u in all_nodes:
        dist_u, _ = dijkstra(graph, u, weight_attr=weight_attr)
        for v in all_nodes:
            if v in dist_u:
                distances[(u, v)] = dist_u[v]
            elif u == v:
                distances[(u, v)] = 0.0
            else:
                distances[(u, v)] = float("inf")
    return distances


def graph_radius_and_diameter(graph: Any, weight_attr: str = "weight") -> tuple[float, float]:
    """Compute the radius and diameter of the graph."""
    apsp = all_pairs_shortest_paths(graph, weight_attr=weight_attr)
    all_nodes = set()
    for u, v in apsp.keys():
        all_nodes.add(u)
        all_nodes.add(v)

    if not all_nodes:
        return 0.0, 0.0

    eccentricities: dict[str, float] = {}
    for u in all_nodes:
        finite_dists = [
            apsp.get((u, v), float("inf"))
            for v in all_nodes
            if apsp.get((u, v), float("inf")) < float("inf")
        ]
        eccentricities[u] = max(finite_dists) if finite_dists else 0.0

    vals = list(eccentricities.values())
    radius = min(vals) if vals else 0.0
    diameter = max(vals) if vals else 0.0
    return radius, diameter
