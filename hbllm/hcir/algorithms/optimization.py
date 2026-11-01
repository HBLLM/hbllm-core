from __future__ import annotations

"""Pure Optimization, Independent Sets, and Graph Coloring Algorithms (WP5)."""

from collections import defaultdict
from typing import Any


def minimum_dominating_set(graph: Any) -> set[str]:
    """Compute an approximate Minimum Dominating Set (MDS) using greedy coverage heuristic.

    Returns:
        Set of node IDs such that every node in the graph is either in the set
        or adjacent to a node in the set.
    """
    if hasattr(graph, "all_nodes"):
        all_nodes = {n.id if hasattr(n, "id") else str(n) for n in graph.all_nodes()}
    elif hasattr(graph, "_nodes"):
        all_nodes = set(graph._nodes.keys())
    else:
        all_nodes = set(getattr(graph, "nodes", set()))

    if not all_nodes:
        return set()

    all_edges = (
        list(graph.all_edges())
        if hasattr(graph, "all_edges")
        else list(getattr(graph, "_edges", {}).values())
    )

    neighbors: dict[str, set[str]] = {nid: {nid} for nid in all_nodes}
    for edge in all_edges:
        sources = edge.sources if hasattr(edge, "sources") else edge.get("sources", [])
        targets = edge.targets if hasattr(edge, "targets") else edge.get("targets", [])
        for s in sources:
            if s in all_nodes:
                for t in targets:
                    if t in all_nodes:
                        neighbors[s].add(t)
                        neighbors[t].add(s)

    uncovered = set(all_nodes)
    dominating_set: set[str] = set()

    while uncovered:
        best_node = max(
            sorted(list(all_nodes)),
            key=lambda u: (len(neighbors[u] & uncovered), -ord(u[0]) if u else 0),
        )
        dominating_set.add(best_node)
        uncovered -= neighbors[best_node]

    return dominating_set


def maximum_independent_set(graph: Any) -> set[str]:
    """Compute an approximate Maximum Independent Set (MIS) using minimum-degree greedy heuristic.

    Returns:
        Set of mutually non-adjacent node IDs representing non-conflicting concurrent elements.
    """
    if hasattr(graph, "all_nodes"):
        all_nodes = {n.id if hasattr(n, "id") else str(n) for n in graph.all_nodes()}
    elif hasattr(graph, "_nodes"):
        all_nodes = set(graph._nodes.keys())
    else:
        all_nodes = set(getattr(graph, "nodes", set()))

    if not all_nodes:
        return set()

    all_edges = (
        list(graph.all_edges())
        if hasattr(graph, "all_edges")
        else list(getattr(graph, "_edges", {}).values())
    )

    adj: dict[str, set[str]] = {nid: set() for nid in all_nodes}
    for edge in all_edges:
        sources = edge.sources if hasattr(edge, "sources") else edge.get("sources", [])
        targets = edge.targets if hasattr(edge, "targets") else edge.get("targets", [])
        for s in sources:
            if s in all_nodes:
                for t in targets:
                    if t in all_nodes and s != t:
                        adj[s].add(t)
                        adj[t].add(s)

    candidates = set(all_nodes)
    independent_set: set[str] = set()

    while candidates:
        best_node = min(
            sorted(list(candidates)),
            key=lambda u: (len(adj[u] & candidates), u),
        )
        independent_set.add(best_node)
        candidates.remove(best_node)
        candidates -= adj[best_node]

    return independent_set


def minimum_vertex_cover(graph: Any) -> set[str]:
    """Compute an approximate Minimum Vertex Cover (MVC) using greedy coverage and Gallai duality.

    Returns:
        Minimal set of node IDs such that every undirected edge has at least one endpoint in the set.
    """
    if hasattr(graph, "all_nodes"):
        all_nodes = {n.id if hasattr(n, "id") else str(n) for n in graph.all_nodes()}
    elif hasattr(graph, "_nodes"):
        all_nodes = set(graph._nodes.keys())
    else:
        all_nodes = set(getattr(graph, "nodes", set()))

    if not all_nodes:
        return set()

    all_edges = (
        list(graph.all_edges())
        if hasattr(graph, "all_edges")
        else list(getattr(graph, "_edges", {}).values())
    )

    edges: set[tuple[str, str]] = set()
    for edge in all_edges:
        sources = edge.sources if hasattr(edge, "sources") else edge.get("sources", [])
        targets = edge.targets if hasattr(edge, "targets") else edge.get("targets", [])
        for s in sources:
            if s in all_nodes:
                for t in targets:
                    if t in all_nodes and s != t:
                        edges.add((min(s, t), max(s, t)))

    if not edges:
        return set()

    incident: dict[str, set[tuple[str, str]]] = {nid: set() for nid in all_nodes}
    for u, v in edges:
        incident[u].add((u, v))
        incident[v].add((u, v))

    # Heuristic 1: Greedy incident edge covering
    uncovered = set(edges)
    cover1: set[str] = set()
    while uncovered:
        best_node = max(
            sorted(list(all_nodes)),
            key=lambda u: (len(incident[u] & uncovered), -ord(u[0]) if u else 0),
        )
        cover1.add(best_node)
        uncovered -= incident[best_node]

    # Prune redundant nodes to guarantee minimal vertex cover
    for node in sorted(list(cover1)):
        can_remove = True
        for u, v in incident[node]:
            other = v if u == node else u
            if other not in cover1:
                can_remove = False
                break
        if can_remove:
            cover1.remove(node)

    # Heuristic 2: Gallai duality on active edge vertices (V_active \ MIS)
    active_nodes = {u for u, _ in edges} | {v for _, v in edges}
    adj_active: dict[str, set[str]] = {nid: set() for nid in active_nodes}
    for u, v in edges:
        adj_active[u].add(v)
        adj_active[v].add(u)

    candidates = set(active_nodes)
    mis: set[str] = set()
    while candidates:
        best_c = min(
            sorted(list(candidates)),
            key=lambda u: (len(adj_active[u] & candidates), u),
        )
        mis.add(best_c)
        candidates.remove(best_c)
        candidates -= adj_active[best_c]

    cover2 = active_nodes - mis
    for node in sorted(list(cover2)):
        can_remove = True
        for u, v in incident[node]:
            other = v if u == node else u
            if other not in cover2:
                can_remove = False
                break
        if can_remove:
            cover2.remove(node)

    return cover1 if len(cover1) <= len(cover2) else cover2


def greedy_edge_coloring(graph: Any) -> dict[tuple[str, str], int]:
    """Compute a proper greedy edge coloring (Vizing's theorem) of undirected edges.

    Returns:
        Dictionary mapping canonical edge pairs (min(u, v), max(u, v)) to discrete color indices (0, 1, 2, ...).
    """
    if hasattr(graph, "all_nodes"):
        all_nodes = {n.id if hasattr(n, "id") else str(n) for n in graph.all_nodes()}
    elif hasattr(graph, "_nodes"):
        all_nodes = set(graph._nodes.keys())
    else:
        all_nodes = set(getattr(graph, "nodes", set()))

    if not all_nodes:
        return {}

    all_edges = (
        list(graph.all_edges())
        if hasattr(graph, "all_edges")
        else list(getattr(graph, "_edges", {}).values())
    )

    edges: set[tuple[str, str]] = set()
    for edge in all_edges:
        sources = edge.sources if hasattr(edge, "sources") else edge.get("sources", [])
        targets = edge.targets if hasattr(edge, "targets") else edge.get("targets", [])
        for s in sources:
            if s in all_nodes:
                for t in targets:
                    if t in all_nodes and s != t:
                        edges.add((min(s, t), max(s, t)))

    node_incident_edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for u, v in edges:
        node_incident_edges[u].append((u, v))
        node_incident_edges[v].append((u, v))

    sorted_edges = sorted(
        list(edges),
        key=lambda e: (
            -(len(node_incident_edges[e[0]]) + len(node_incident_edges[e[1]])),
            e[0],
            e[1],
        ),
    )

    edge_colors: dict[tuple[str, str], int] = {}
    node_used_colors: dict[str, set[int]] = {nid: set() for nid in all_nodes}

    for u, v in sorted_edges:
        used = node_used_colors[u] | node_used_colors[v]
        c = 0
        while c in used:
            c += 1
        edge_colors[(u, v)] = c
        node_used_colors[u].add(c)
        node_used_colors[v].add(c)

    return edge_colors
