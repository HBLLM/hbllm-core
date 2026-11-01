from __future__ import annotations

"""Pure Path Traversal and Distance Algorithms (WP5)."""

import heapq
from collections import deque
from typing import Any

from hbllm.hcir.algorithms.protocol import native_of


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
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], list[str]]]:
    """Compute all-pairs shortest paths using repeated Dijkstra or Floyd-Warshall."""
    native = native_of(graph)
    if native is not None:
        return native.all_pairs_shortest_paths(weight_attribute=weight_attr)
    all_nodes: list[str] = []
    if hasattr(graph, "all_nodes"):
        all_nodes = [n.id if hasattr(n, "id") else str(n) for n in graph.all_nodes()]
    elif hasattr(graph, "all_node_ids"):
        all_nodes = list(graph.all_node_ids())
    else:
        all_nodes = list(getattr(graph, "_nodes", {}).keys())

    distances: dict[tuple[str, str], float] = {}
    paths: dict[tuple[str, str], list[str]] = {}
    for u in all_nodes:
        dist_u, prev = dijkstra(graph, u, weight_attr=weight_attr)
        for v in all_nodes:
            if u == v:
                distances[(u, v)] = 0.0
                paths[(u, v)] = [u]
            elif v in dist_u:
                distances[(u, v)] = dist_u[v]
                curr: str | None = v
                p: list[str] = []
                visited: set[str] = set()
                while curr is not None and curr != u and curr not in visited:
                    visited.add(curr)
                    p.append(curr)
                    curr = prev.get(curr)
                if curr == u:
                    p.append(u)
                    p.reverse()
                    paths[(u, v)] = p
                else:
                    paths[(u, v)] = []
            else:
                distances[(u, v)] = float("inf")
                paths[(u, v)] = []
    return distances, paths


def graph_radius_and_diameter(graph: Any, weight_attr: str = "weight") -> tuple[float, float]:
    """Compute the radius and diameter of the graph based on node eccentricities.

    Eccentricity e(u) = max_{v != u, reachable} dist(u, v).
    Radius R = min_u e(u), Diameter D = max_u e(u).
    """
    native = native_of(graph)
    if native is not None:
        return native.graph_radius_and_diameter()
    apsp, _ = all_pairs_shortest_paths(graph, weight_attr=weight_attr)
    all_nodes = set()
    for u, v in apsp.keys():
        all_nodes.add(u)
        all_nodes.add(v)

    if len(all_nodes) <= 1:
        return 0.0, 0.0

    eccentricities: list[float] = []
    for u in all_nodes:
        dists = [d for (src, tgt), d in apsp.items() if src == u and tgt != u and d < float("inf")]
        if dists:
            eccentricities.append(max(dists))

    if not eccentricities:
        return 0.0, 0.0

    return min(eccentricities), max(eccentricities)


def topological_sort(graph: Any) -> list[str]:
    """Compute topological ordering of DAG nodes using Kahn's algorithm."""
    native = native_of(graph)
    if native is not None:
        return native.topological_sort()
    if hasattr(graph, "all_nodes"):
        all_ids = set(n.id if hasattr(n, "id") else str(n) for n in graph.all_nodes())
    elif hasattr(graph, "_nodes"):
        all_ids = set(graph._nodes.keys())
    else:
        all_ids = set(getattr(graph, "nodes", set()))

    if not all_ids:
        return []

    in_degree: dict[str, int] = {nid: 0 for nid in all_ids}
    adj: dict[str, list[str]] = {nid: [] for nid in all_ids}

    all_edges = (
        list(graph.all_edges())
        if hasattr(graph, "all_edges")
        else list(getattr(graph, "_edges", {}).values())
    )

    for edge in all_edges:
        sources = edge.sources if hasattr(edge, "sources") else edge.get("sources", [])
        targets = edge.targets if hasattr(edge, "targets") else edge.get("targets", [])
        for s in sources:
            if s in all_ids:
                for t in targets:
                    if t in all_ids:
                        adj[s].append(t)
                        in_degree[t] += 1

    queue: deque[str] = deque([nid for nid, deg in in_degree.items() if deg == 0])
    order: list[str] = []

    while queue:
        curr = queue.popleft()
        order.append(curr)
        for neighbor in adj[curr]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return order


def find_eulerian_path_or_circuit(
    graph: Any,
) -> tuple[bool, list[tuple[str, str]]]:
    """Find an Eulerian path or circuit traversing every directed edge exactly once (Hierholzer).

    Returns:
        Tuple of (is_eulerian, edge_sequence).
    """
    native = native_of(graph)
    if native is not None:
        return native.find_eulerian_path_or_circuit()
    from collections import defaultdict

    if hasattr(graph, "all_nodes"):
        all_nodes = sorted([n.id if hasattr(n, "id") else str(n) for n in graph.all_nodes()])
    elif hasattr(graph, "_nodes"):
        all_nodes = sorted(list(graph._nodes.keys()))
    else:
        all_nodes = sorted(list(getattr(graph, "nodes", [])))

    if not all_nodes:
        return True, []

    all_directed_edges: list[tuple[str, str]] = []
    in_deg: dict[str, int] = defaultdict(int)
    out_deg: dict[str, int] = defaultdict(int)

    all_edges = (
        list(graph.all_edges())
        if hasattr(graph, "all_edges")
        else list(getattr(graph, "_edges", {}).values())
    )

    for edge in all_edges:
        sources = edge.sources if hasattr(edge, "sources") else edge.get("sources", [])
        targets = edge.targets if hasattr(edge, "targets") else edge.get("targets", [])
        for s in sources:
            if s in all_nodes:
                for t in targets:
                    if t in all_nodes:
                        all_directed_edges.append((s, t))
                        out_deg[s] += 1
                        in_deg[t] += 1

    if not all_directed_edges:
        return True, []

    start_candidates: list[str] = []
    end_candidates: list[str] = []
    unbalanced: list[str] = []

    active_nodes = sorted(
        list({u for u, v in all_directed_edges} | {v for u, v in all_directed_edges})
    )

    for v in active_nodes:
        diff = out_deg[v] - in_deg[v]
        if diff == 1:
            start_candidates.append(v)
        elif diff == -1:
            end_candidates.append(v)
        elif diff != 0:
            unbalanced.append(v)

    if unbalanced:
        return False, []

    if len(start_candidates) == 0 and len(end_candidates) == 0:
        start_node = min(active_nodes, key=lambda n: (out_deg[n] == 0, n))
    elif len(start_candidates) == 1 and len(end_candidates) == 1:
        start_node = start_candidates[0]
    else:
        return False, []

    adj: dict[str, list[str]] = defaultdict(list)
    for u, v in sorted(all_directed_edges, reverse=True):
        adj[u].append(v)

    curr_path: list[str] = [start_node]
    circuit: list[str] = []

    while curr_path:
        curr = curr_path[-1]
        if adj[curr]:
            nxt = adj[curr].pop()
            curr_path.append(nxt)
        else:
            circuit.append(curr_path.pop())

    circuit.reverse()
    path_edges = [(circuit[i], circuit[i + 1]) for i in range(len(circuit) - 1)]

    if len(path_edges) != len(all_directed_edges):
        return False, []

    return True, path_edges


def chinese_postman_tour(
    graph: Any,
    weight_attr: str = "weight",
) -> tuple[float, list[tuple[str, str]]]:
    """Compute minimum-weight edge-covering tour traversing every edge at least once (CPP).

    Returns:
        Tuple of (total_tour_weight, tour_edge_sequence).
    """
    native = native_of(graph)
    if native is not None:
        return native.chinese_postman_tour()
    from collections import defaultdict

    if hasattr(graph, "all_nodes"):
        all_nodes = sorted([n.id if hasattr(n, "id") else str(n) for n in graph.all_nodes()])
    elif hasattr(graph, "_nodes"):
        all_nodes = sorted(list(graph._nodes.keys()))
    else:
        all_nodes = sorted(list(getattr(graph, "nodes", [])))

    if not all_nodes:
        return 0.0, []

    all_edges = (
        list(graph.all_edges())
        if hasattr(graph, "all_edges")
        else list(getattr(graph, "_edges", {}).values())
    )

    directed_edges: list[tuple[str, str, float]] = []
    for edge in all_edges:
        w = max(
            0.1,
            float(
                getattr(
                    edge,
                    weight_attr,
                    edge.get(weight_attr, 1.0) if isinstance(edge, dict) else 1.0,
                )
            ),
        )
        sources = edge.sources if hasattr(edge, "sources") else edge.get("sources", [])
        targets = edge.targets if hasattr(edge, "targets") else edge.get("targets", [])
        for s in sources:
            if s in all_nodes:
                for t in targets:
                    if t in all_nodes:
                        directed_edges.append((s, t, w))

    if not directed_edges:
        return 0.0, []

    is_eul, eul_path = find_eulerian_path_or_circuit(graph)
    if is_eul and eul_path and eul_path[0][0] == eul_path[-1][1]:
        edge_weight_map: dict[tuple[str, str], float] = defaultdict(lambda: 1.0)
        for u, v, w in directed_edges:
            edge_weight_map[(u, v)] = w
        total_w = sum(edge_weight_map.get((u, v), 1.0) for u, v in eul_path)
        return float(total_w), eul_path

    in_deg: dict[str, int] = defaultdict(int)
    out_deg: dict[str, int] = defaultdict(int)
    adj_dist: dict[tuple[str, str], float] = {}

    for u, v, w in directed_edges:
        out_deg[u] += 1
        in_deg[v] += 1
        if (u, v) not in adj_dist or w < adj_dist[(u, v)]:
            adj_dist[(u, v)] = w

    active_nodes = sorted(
        list({u for u, v, _ in directed_edges} | {v for u, v, _ in directed_edges})
    )

    dist: dict[tuple[str, str], float] = {}
    nxt_hop: dict[tuple[str, str], str] = {}
    for u in active_nodes:
        for v in active_nodes:
            if u == v:
                dist[(u, v)] = 0.0
            elif (u, v) in adj_dist:
                dist[(u, v)] = adj_dist[(u, v)]
                nxt_hop[(u, v)] = v
            else:
                dist[(u, v)] = float("inf")

    for k in active_nodes:
        for i in active_nodes:
            for j in active_nodes:
                if dist[(i, k)] + dist[(k, j)] < dist[(i, j)]:
                    dist[(i, j)] = dist[(i, k)] + dist[(k, j)]
                    nxt_hop[(i, j)] = nxt_hop.get((i, k), k)

    deficit = {v: in_deg[v] - out_deg[v] for v in active_nodes}
    demand = [v for v in active_nodes if deficit[v] > 0]
    supply = [v for v in active_nodes if deficit[v] < 0]

    augmented_edges: list[tuple[str, str]] = [(u, v) for u, v, _ in directed_edges]
    added_cost = 0.0

    for d in demand:
        needed = deficit[d]
        while needed > 0 and supply:
            s = min(supply, key=lambda sup: dist.get((d, sup), float("inf")))
            d_s_dist = dist.get((d, s), float("inf"))
            if d_s_dist == float("inf"):
                break
            curr = d
            while curr != s and (curr, s) in nxt_hop:
                nxt = nxt_hop[(curr, s)]
                augmented_edges.append((curr, nxt))
                added_cost += adj_dist.get((curr, nxt), 1.0)
                curr = nxt
            needed -= 1
            deficit[s] += 1
            if deficit[s] >= 0:
                supply.remove(s)

    adj_aug: dict[str, list[str]] = defaultdict(list)
    for u, v in sorted(augmented_edges, reverse=True):
        adj_aug[u].append(v)

    start_node = active_nodes[0]
    curr_path = [start_node]
    circuit: list[str] = []

    while curr_path:
        curr = curr_path[-1]
        if adj_aug[curr]:
            nxt = adj_aug[curr].pop()
            curr_path.append(nxt)
        else:
            circuit.append(curr_path.pop())

    circuit.reverse()
    tour = [(circuit[i], circuit[i + 1]) for i in range(len(circuit) - 1)]
    total_weight = sum(adj_dist.get((u, v), 1.0) for u, v, _ in directed_edges) + added_cost
    return float(total_weight), tour
