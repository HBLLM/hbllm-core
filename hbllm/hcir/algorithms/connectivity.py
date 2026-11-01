from __future__ import annotations

"""Pure Connectivity Graph Algorithms (WP5)."""

from collections import deque
from typing import Any


def connected_components(graph: Any) -> list[set[str]]:
    """Compute weakly connected components of the graph."""
    # Get all node IDs
    if hasattr(graph, "all_nodes"):
        all_ids = [n.id if hasattr(n, "id") else str(n) for n in graph.all_nodes()]
    elif hasattr(graph, "all_node_ids"):
        all_ids = list(graph.all_node_ids())
    elif hasattr(graph, "_nodes"):
        all_ids = list(graph._nodes.keys())
    else:
        all_ids = []

    adj: dict[str, set[str]] = {nid: set() for nid in all_ids}
    if hasattr(graph, "all_edges"):
        edges = list(graph.all_edges())
    elif hasattr(graph, "_edges"):
        edges = list(graph._edges.values())
    else:
        edges = []

    for e in edges:
        sources = e.sources if hasattr(e, "sources") else e.get("sources", [])
        targets = e.targets if hasattr(e, "targets") else e.get("targets", [])
        for u in sources:
            for v in targets:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)

    visited: set[str] = set()
    components: list[set[str]] = []

    for nid in all_ids:
        if nid in visited:
            continue
        comp: set[str] = set()
        queue = deque([nid])
        visited.add(nid)
        comp.add(nid)

        while queue:
            curr = queue.popleft()
            for nbr in adj.get(curr, set()):
                if nbr not in visited:
                    visited.add(nbr)
                    comp.add(nbr)
                    queue.append(nbr)
        components.append(comp)

    return components


def articulation_points_and_bridges(graph: Any) -> tuple[set[str], set[str]]:
    """Find articulation points (cut vertices) and bridges (cut edges) using Tarjan's algorithm."""
    tin: dict[str, int] = {}
    low: dict[str, int] = {}
    timer = 0
    art_points: set[str] = set()
    bridges: set[str] = set()

    if hasattr(graph, "all_nodes"):
        all_ids = [n.id if hasattr(n, "id") else str(n) for n in graph.all_nodes()]
    elif hasattr(graph, "all_node_ids"):
        all_ids = list(graph.all_node_ids())
    else:
        all_ids = list(getattr(graph, "_nodes", {}).keys())

    def dfs(u: str, p: str | None = None) -> None:
        nonlocal timer
        timer += 1
        tin[u] = low[u] = timer
        children = 0

        neighbors = (
            graph.neighbors(u)
            if hasattr(graph, "neighbors")
            else (graph._neighbors(u) if hasattr(graph, "_neighbors") else [])
        )
        for v in neighbors:
            if v == p:
                continue
            if v in tin:
                low[u] = min(low[u], tin[v])
            else:
                dfs(v, u)
                low[u] = min(low[u], low[v])
                if low[v] >= tin[u] and p is not None:
                    art_points.add(u)
                if low[v] > tin[u]:
                    bridges.add(f"{u}->{v}")
                children += 1

        if p is None and children > 1:
            art_points.add(u)

    for nid in all_ids:
        if nid not in tin:
            dfs(nid)

    return art_points, bridges


def is_biconnected(graph: Any) -> bool:
    """Check whether graph is biconnected (connected and contains no articulation points)."""
    comps = connected_components(graph)
    if len(comps) != 1:
        return False
    art_points, _ = articulation_points_and_bridges(graph)
    return len(art_points) == 0
