from __future__ import annotations

"""Pure Connectivity Graph Algorithms (WP5)."""

from collections import deque
from typing import Any

from hbllm.hcir.algorithms.protocol import native_of


def connected_components(graph: Any) -> list[set[str]]:
    """Compute weakly connected components of the graph."""
    native = native_of(graph)
    if native is not None:
        return native.connected_components()
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


def articulation_points_and_bridges(graph: Any) -> tuple[set[str], list[tuple[str, str]]]:
    """Find articulation points (cut vertices) and bridges (cut edges) using Tarjan's algorithm."""
    native = native_of(graph)
    if native is not None:
        return native.find_articulation_points_and_bridges()
    tin: dict[str, int] = {}
    low: dict[str, int] = {}
    timer = 0
    art_points: set[str] = set()
    bridges: list[tuple[str, str]] = []

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
                    bridges.append((u, v))
                children += 1

        if p is None and children > 1:
            art_points.add(u)

    for nid in all_ids:
        if nid not in tin:
            dfs(nid)

    return art_points, bridges


def is_biconnected(graph: Any) -> bool:
    """Check whether graph is biconnected (connected with >= 2 nodes, 0 articulation points, and 0 bridges)."""
    native = native_of(graph)
    if native is not None:
        return native.is_biconnected()
    comps = connected_components(graph)
    if len(comps) != 1:
        return False
    art_points, bridges = articulation_points_and_bridges(graph)
    return len(art_points) == 0 and len(bridges) == 0


def strongly_connected_components(graph: Any) -> list[set[str]]:
    """Compute Strongly Connected Components (SCC) of the directed graph using Tarjan's algorithm.

    Returns:
        List of sets of node IDs, where each set represents a maximal strongly connected component.
    """
    native = native_of(graph)
    if native is not None:
        return native.strongly_connected_components()
    if hasattr(graph, "all_nodes"):
        all_nodes = sorted([n.id if hasattr(n, "id") else str(n) for n in graph.all_nodes()])
    elif hasattr(graph, "_nodes"):
        all_nodes = sorted(list(graph._nodes.keys()))
    else:
        all_nodes = sorted(list(getattr(graph, "nodes", [])))

    if not all_nodes:
        return []

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
            if s in adj:
                for t in targets:
                    if t in adj and s != t:
                        adj[s].add(t)

    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    sccs: list[set[str]] = []

    for node in all_nodes:
        if node not in indices:
            call_stack: list[tuple[str, int, list[str]]] = [(node, 0, sorted(list(adj[node])))]
            indices[node] = index
            lowlinks[node] = index
            index += 1
            stack.append(node)
            on_stack.add(node)

            while call_stack:
                curr, neighbor_idx, neighbors = call_stack[-1]
                if neighbor_idx < len(neighbors):
                    nxt = neighbors[neighbor_idx]
                    call_stack[-1] = (curr, neighbor_idx + 1, neighbors)
                    if nxt not in indices:
                        indices[nxt] = index
                        lowlinks[nxt] = index
                        index += 1
                        stack.append(nxt)
                        on_stack.add(nxt)
                        call_stack.append((nxt, 0, sorted(list(adj[nxt]))))
                    elif nxt in on_stack:
                        lowlinks[curr] = min(lowlinks[curr], indices[nxt])
                else:
                    call_stack.pop()
                    if call_stack:
                        parent = call_stack[-1][0]
                        lowlinks[parent] = min(lowlinks[parent], lowlinks[curr])

                    if lowlinks[curr] == indices[curr]:
                        component: set[str] = set()
                        while True:
                            w = stack.pop()
                            on_stack.remove(w)
                            component.add(w)
                            if w == curr:
                                break
                        sccs.append(component)

    return sorted(sccs, key=lambda comp: min(comp) if comp else "")


def condense_scc_metagraph(
    graph: Any,
) -> tuple[dict[int, set[str]], list[tuple[int, int]]]:
    """Condense directed graph SCCs into an acyclic metagraph (DAG).

    Returns:
        Tuple of (scc_map, metagraph_edges).
    """
    native = native_of(graph)
    if native is not None:
        return native.condense_scc_metagraph()
    sccs = strongly_connected_components(graph)
    node_to_scc: dict[str, int] = {}
    scc_map: dict[int, set[str]] = {}

    for scc_id, comp in enumerate(sccs):
        scc_map[scc_id] = comp
        for nid in comp:
            node_to_scc[nid] = scc_id

    metagraph_edges_set: set[tuple[int, int]] = set()

    all_edges = (
        list(graph.all_edges())
        if hasattr(graph, "all_edges")
        else list(getattr(graph, "_edges", {}).values())
    )

    for edge in all_edges:
        sources = edge.sources if hasattr(edge, "sources") else edge.get("sources", [])
        targets = edge.targets if hasattr(edge, "targets") else edge.get("targets", [])
        for s in sources:
            src_scc = node_to_scc.get(s)
            if src_scc is None:
                continue
            for t in targets:
                tgt_scc = node_to_scc.get(t)
                if tgt_scc is not None and src_scc != tgt_scc:
                    metagraph_edges_set.add((src_scc, tgt_scc))

    return scc_map, sorted(list(metagraph_edges_set))
