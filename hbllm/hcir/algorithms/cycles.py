from __future__ import annotations

"""Pure Cycle Detection and Feedback Arc Algorithms (WP5)."""

from collections import deque
from typing import Any


def fundamental_cycle_basis(graph: Any) -> list[list[str]]:
    """Compute Paton's fundamental cycle basis for undirected view of the graph."""
    visited: set[str] = set()
    parent: dict[str, str | None] = {}
    cycles: list[list[str]] = []

    all_nodes: list[str] = []
    if hasattr(graph, "all_nodes"):
        all_nodes = [n.id if hasattr(n, "id") else str(n) for n in graph.all_nodes()]
    elif hasattr(graph, "all_node_ids"):
        all_nodes = list(graph.all_node_ids())
    else:
        all_nodes = list(getattr(graph, "_nodes", {}).keys())

    for root in all_nodes:
        if root in visited:
            continue
        queue = deque([root])
        visited.add(root)
        parent[root] = None

        while queue:
            u = queue.popleft()
            nbrs = (
                graph.neighbors(u)
                if hasattr(graph, "neighbors")
                else (graph._neighbors(u) if hasattr(graph, "_neighbors") else [])
            )
            for v in nbrs:
                if v == parent.get(u):
                    continue
                if v in visited:
                    # Found a cycle: backtrace path
                    cycle = [u]
                    curr = u
                    while curr is not None and curr != v:
                        curr = parent.get(curr)
                        if curr:
                            cycle.append(curr)
                    if cycle and cycle[-1] == v:
                        cycles.append(cycle)
                else:
                    visited.add(v)
                    parent[v] = u
                    queue.append(v)

    return cycles


def feedback_arc_set(graph: Any) -> list[tuple[str, str]]:
    """Greedy approximation of feedback arc set (edges to remove to make DAG)."""
    all_edges = (
        list(graph.all_edges())
        if hasattr(graph, "all_edges")
        else list(getattr(graph, "_edges", {}).values())
    )
    edges: list[tuple[str, str]] = []
    for e in all_edges:
        sources = e.sources if hasattr(e, "sources") else e.get("sources", [])
        targets = e.targets if hasattr(e, "targets") else e.get("targets", [])
        if sources and targets:
            edges.append((sources[0], targets[0]))

    # Detect back-edges in DFS
    color: dict[str, int] = {}  # 0: unvisited, 1: visiting, 2: visited
    fas: list[tuple[str, str]] = []

    def dfs(u: str) -> None:
        color[u] = 1
        for s, t in edges:
            if s == u:
                if color.get(t, 0) == 1:
                    fas.append((s, t))
                elif color.get(t, 0) == 0:
                    dfs(t)
        color[u] = 2

    all_nodes: set[str] = set()
    for s, t in edges:
        all_nodes.add(s)
        all_nodes.add(t)

    for n in all_nodes:
        if color.get(n, 0) == 0:
            dfs(n)

    return fas
