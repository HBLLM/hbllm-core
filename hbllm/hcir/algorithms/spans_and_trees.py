from __future__ import annotations

"""Pure Spanning Tree and Reachability Reduction Algorithms (WP5)."""

from typing import Any


def minimum_spanning_tree(graph: Any, weight_attr: str = "weight") -> tuple[list[Any], float]:
    """Compute Minimum Spanning Tree using Kruskal's algorithm."""
    all_edges = (
        list(graph.all_edges())
        if hasattr(graph, "all_edges")
        else list(getattr(graph, "_edges", {}).values())
    )

    # Sort edges by weight
    sorted_edges = sorted(
        all_edges,
        key=lambda e: float(
            getattr(
                e,
                weight_attr,
                e.get(weight_attr, 1.0) if isinstance(e, dict) else 1.0,
            )
        ),
    )

    parent: dict[str, str] = {}

    def find(i: str) -> str:
        if parent.setdefault(i, i) == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i: str, j: str) -> bool:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            return True
        return False

    mst_edges: list[Any] = []
    total_weight = 0.0

    for e in sorted_edges:
        sources = e.sources if hasattr(e, "sources") else e.get("sources", [])
        targets = e.targets if hasattr(e, "targets") else e.get("targets", [])
        if sources and targets:
            u, v = sources[0], targets[0]
            if union(u, v):
                mst_edges.append(e)
                w = float(
                    getattr(
                        e,
                        weight_attr,
                        e.get(weight_attr, 1.0) if isinstance(e, dict) else 1.0,
                    )
                )
                total_weight += w

    return mst_edges, total_weight


def transitive_reduction(graph: Any) -> set[tuple[str, str]]:
    """Compute minimal direct edge set preserving reachability for a DAG."""
    all_nodes: list[str] = []
    if hasattr(graph, "all_nodes"):
        all_nodes = [n.id if hasattr(n, "id") else str(n) for n in graph.all_nodes()]
    elif hasattr(graph, "all_node_ids"):
        all_nodes = list(graph.all_node_ids())
    else:
        all_nodes = list(getattr(graph, "_nodes", {}).keys())

    adj: dict[str, set[str]] = {n: set() for n in all_nodes}
    all_edges = (
        list(graph.all_edges())
        if hasattr(graph, "all_edges")
        else list(getattr(graph, "_edges", {}).values())
    )
    for e in all_edges:
        sources = e.sources if hasattr(e, "sources") else e.get("sources", [])
        targets = e.targets if hasattr(e, "targets") else e.get("targets", [])
        for u in sources:
            for v in targets:
                adj.setdefault(u, set()).add(v)

    # Compute reachability via DFS
    reach: dict[str, set[str]] = {}
    for u in all_nodes:
        seen: set[str] = set()
        stack = list(adj.get(u, set()))
        while stack:
            curr = stack.pop()
            if curr not in seen:
                seen.add(curr)
                stack.extend(adj.get(curr, set()))
        reach[u] = seen

    # Edge (u, v) is retained if no w exists such that u -> w and w can reach v
    reduced_edges: set[tuple[str, str]] = set()
    for u in all_nodes:
        for v in adj.get(u, set()):
            redundant = any(v in reach.get(w, set()) for w in adj.get(u, set()) if w != v)
            if not redundant:
                reduced_edges.add((u, v))

    return reduced_edges


def transitive_closure(graph: Any) -> set[tuple[str, str]]:
    """Compute transitive closure reachability pairs (u, v)."""
    all_nodes: list[str] = []
    if hasattr(graph, "all_nodes"):
        all_nodes = [n.id if hasattr(n, "id") else str(n) for n in graph.all_nodes()]
    elif hasattr(graph, "all_node_ids"):
        all_nodes = list(graph.all_node_ids())
    else:
        all_nodes = list(getattr(graph, "_nodes", {}).keys())

    adj: dict[str, set[str]] = {n: set() for n in all_nodes}
    all_edges = (
        list(graph.all_edges())
        if hasattr(graph, "all_edges")
        else list(getattr(graph, "_edges", {}).values())
    )
    for e in all_edges:
        sources = e.sources if hasattr(e, "sources") else e.get("sources", [])
        targets = e.targets if hasattr(e, "targets") else e.get("targets", [])
        for u in sources:
            for v in targets:
                adj.setdefault(u, set()).add(v)

    closure: set[tuple[str, str]] = set()
    for u in all_nodes:
        seen: set[str] = set()
        stack = list(adj.get(u, set()))
        while stack:
            curr = stack.pop()
            if curr not in seen:
                seen.add(curr)
                closure.add((u, curr))
                stack.extend(adj.get(curr, set()))

    return closure
