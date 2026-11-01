from __future__ import annotations

"""Pure Cycle Detection and Feedback Arc Algorithms (WP5)."""

from collections import deque
from typing import Any

from hbllm.hcir.algorithms.protocol import native_of


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


def feedback_arc_set(graph: Any) -> tuple[list[tuple[str, str]], list[str]]:
    """Compute the minimal Feedback Arc Set (FAS) to render the graph acyclic using Eades heuristic.

    Returns:
        Tuple of (feedback_edges, topological_ordering), where removing feedback_edges
        yields a Directed Acyclic Graph (DAG).
    """
    native = native_of(graph)
    if native is not None:
        return native.feedback_arc_set()
    all_nodes: list[str] = []
    if hasattr(graph, "all_nodes"):
        all_nodes = [n.id if hasattr(n, "id") else str(n) for n in graph.all_nodes()]
    elif hasattr(graph, "all_node_ids"):
        all_nodes = list(graph.all_node_ids())
    else:
        all_nodes = list(getattr(graph, "_nodes", {}).keys())

    all_nodes.sort()
    if not all_nodes:
        return [], []

    all_edges = (
        list(graph.all_edges())
        if hasattr(graph, "all_edges")
        else list(getattr(graph, "_edges", {}).values())
    )
    adj_out: dict[str, set[str]] = {nid: set() for nid in all_nodes}
    adj_in: dict[str, set[str]] = {nid: set() for nid in all_nodes}
    all_directed_edges: list[tuple[str, str]] = []

    for e in all_edges:
        sources = e.sources if hasattr(e, "sources") else e.get("sources", [])
        targets = e.targets if hasattr(e, "targets") else e.get("targets", [])
        for s in sources:
            if s in adj_out:
                for t in targets:
                    if t in adj_out and s != t:
                        adj_out[s].add(t)
                        adj_in[t].add(s)
                        all_directed_edges.append((s, t))

    in_deg = {nid: len(adj_in[nid]) for nid in all_nodes}
    out_deg = {nid: len(adj_out[nid]) for nid in all_nodes}
    remaining = set(all_nodes)
    s1: list[str] = []
    s2: list[str] = []

    while remaining:
        sinks = [u for u in remaining if out_deg[u] == 0]
        if sinks:
            for u in sorted(sinks):
                if u in remaining:
                    remaining.remove(u)
                    s2.insert(0, u)
                    for p in adj_in[u]:
                        if p in remaining:
                            out_deg[p] -= 1
            continue

        sources_list = [u for u in remaining if in_deg[u] == 0]
        if sources_list:
            for u in sorted(sources_list):
                if u in remaining:
                    remaining.remove(u)
                    s1.append(u)
                    for s in adj_out[u]:
                        if s in remaining:
                            in_deg[s] -= 1
            continue

        best_u = max(
            sorted(list(remaining)),
            key=lambda u: (out_deg[u] - in_deg[u], -ord(u[0]) if u else 0),
        )
        remaining.remove(best_u)
        s1.append(best_u)
        for s in adj_out[best_u]:
            if s in remaining:
                in_deg[s] -= 1
        for p in adj_in[best_u]:
            if p in remaining:
                out_deg[p] -= 1

    topological_order = s1 + s2
    rank = {nid: i for i, nid in enumerate(topological_order)}
    feedback_edges: list[tuple[str, str]] = []
    seen_feedback: set[tuple[str, str]] = set()

    for u, v in all_directed_edges:
        if rank[u] > rank[v]:
            pair = (u, v)
            if pair not in seen_feedback:
                seen_feedback.add(pair)
                feedback_edges.append(pair)

    return feedback_edges, topological_order


def feedback_vertex_set(graph: Any) -> set[str]:
    """Compute an approximate minimal Feedback Vertex Set (FVS) to break all directed cycles.

    Returns:
        Minimal set of node IDs whose removal from the graph leaves a Directed Acyclic Graph (DAG).
    """
    native = native_of(graph)
    if native is not None:
        return native.feedback_vertex_set()
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

    directed_edges: set[tuple[str, str]] = set()
    for edge in all_edges:
        sources = edge.sources if hasattr(edge, "sources") else edge.get("sources", [])
        targets = edge.targets if hasattr(edge, "targets") else edge.get("targets", [])
        for s in sources:
            if s in all_nodes:
                for t in targets:
                    if t in all_nodes:
                        directed_edges.add((s, t))

    if not directed_edges:
        return set()

    def is_subgraph_acyclic(remaining_vertices: set[str]) -> bool:
        in_deg: dict[str, int] = {v: 0 for v in remaining_vertices}
        sub_adj: dict[str, list[str]] = {v: [] for v in remaining_vertices}
        for u, v in directed_edges:
            if u in remaining_vertices and v in remaining_vertices:
                if u == v:
                    return False  # Self-loop
                sub_adj[u].append(v)
                in_deg[v] += 1

        queue = deque([v for v, d in in_deg.items() if d == 0])
        peeled_count = 0
        while queue:
            curr = queue.popleft()
            peeled_count += 1
            for nxt in sub_adj[curr]:
                in_deg[nxt] -= 1
                if in_deg[nxt] == 0:
                    queue.append(nxt)
        return peeled_count == len(remaining_vertices)

    if is_subgraph_acyclic(all_nodes):
        return set()

    fvs: set[str] = set()

    # Step 1: Self-loops must be in FVS immediately
    for u, v in directed_edges:
        if u == v:
            fvs.add(u)

    curr_nodes = all_nodes - fvs

    # Step 2: Greedily pick vertices with highest cyclic degree product
    while not is_subgraph_acyclic(curr_nodes):
        sub_in: dict[str, int] = {v: 0 for v in curr_nodes}
        sub_out: dict[str, int] = {v: 0 for v in curr_nodes}
        for u, v in directed_edges:
            if u in curr_nodes and v in curr_nodes and u != v:
                sub_out[u] += 1
                sub_in[v] += 1

        active_cyclic = {v for v in curr_nodes if sub_in[v] > 0 and sub_out[v] > 0}
        if not active_cyclic:
            break

        best_v = max(
            sorted(list(active_cyclic)),
            key=lambda u: (
                sub_in[u] * sub_out[u],
                sub_in[u] + sub_out[u],
                -ord(u[0]) if u else 0,
            ),
        )
        fvs.add(best_v)
        curr_nodes.remove(best_v)

    # Step 3: Minimalize FVS via reverse pruning
    for node in sorted(list(fvs)):
        candidate_remaining = (all_nodes - fvs) | {node}
        if is_subgraph_acyclic(candidate_remaining):
            fvs.remove(node)

    return fvs


def essential_causal_skeleton(graph: Any) -> set[tuple[str, str]]:
    """Compute the irreducible causal skeleton DAG via transitive reduction."""
    native = native_of(graph)
    if native is not None:
        return native.essential_causal_skeleton()
    from hbllm.hcir.algorithms.spans_and_trees import transitive_reduction

    return transitive_reduction(graph)
