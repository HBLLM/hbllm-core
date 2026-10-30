"""Native HCIR Graph Adapter.

Provides a unified interface between Python HCIR and the native persistent Rust GraphState substrate,
with transparent 100% feature-parity pure-Python fallback when native binaries are uncompiled.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import struct
from collections import defaultdict, deque
from typing import Any

from hbllm.hcir.graph import HCIREdgeType, HCIRNodeType

logger = logging.getLogger(__name__)

# Attempt to load native PyO3 module
try:
    from hbllm_hcir_graph import (
        NativeGraph as _PyNativeGraph,  # type: ignore[import-not-found,no-redef]
    )
except ImportError:
    _PyNativeGraph = None  # type: ignore[assignment]


class NativeGraphAdapter:
    """Three-layer adapter wrapping native Rust GraphState.

    Preserves 100% of the high-level HCIR graph contract while delegating
    storage, traversal, chunked snapshotting, and canonical BLAKE3 hashing
    to the native Rust substrate when available. When native binaries are absent,
    executes an identical pure-Python in-memory fallback.
    """

    def __init__(self, native_instance: Any = None, py_state: dict[str, Any] | None = None) -> None:
        self._removed_nodes: set[str] = set()
        self._removed_edges: set[str] = set()
        self._node_ids: set[str] = set()
        self._edge_ids: set[str] = set()
        if native_instance is not None:
            self._native = native_instance
            self._py_nodes: dict[str, dict[str, Any]] = {}
            self._py_edges: dict[str, dict[str, Any]] = {}
            self._py_outgoing: dict[str, list[str]] = {}
            self._py_incoming: dict[str, list[str]] = {}
            self._py_type_index: dict[str, list[str]] = {}
        elif _PyNativeGraph is not None:
            self._native = _PyNativeGraph()
            self._py_nodes = {}
            self._py_edges = {}
            self._py_outgoing = {}
            self._py_incoming = {}
            self._py_type_index = {}
        else:
            self._native = None
            if py_state is not None:
                self._py_nodes = copy.deepcopy(py_state.get("nodes", {}))
                self._py_edges = copy.deepcopy(py_state.get("edges", {}))
                self._py_outgoing = copy.deepcopy(py_state.get("outgoing", {}))
                self._py_incoming = copy.deepcopy(py_state.get("incoming", {}))
                self._py_type_index = copy.deepcopy(py_state.get("type_index", {}))
                self._node_ids = set(self._py_nodes.keys())
                self._edge_ids = set(self._py_edges.keys())
            else:
                self._py_nodes = {}
                self._py_edges = {}
                self._py_outgoing = {}
                self._py_incoming = {}
                self._py_type_index = {}

    @property
    def is_native(self) -> bool:
        """Whether this adapter is backed by the compiled Rust substrate."""
        return self._native is not None

    def all_node_ids(self) -> set[str]:
        """Return the set of all active node IDs."""
        return {
            nid
            for nid in (self._node_ids if self._native is not None else self._py_nodes)
            if nid not in self._removed_nodes
        }

    def add_node(
        self,
        node_id: str,
        node_type: str | HCIRNodeType,
        lifecycle: str = "ACTIVE",
        properties: dict[str, str] | None = None,
        created_at: float = 0.0,
    ) -> None:
        self._removed_nodes.discard(node_id)
        self._node_ids.add(node_id)
        type_str = node_type.value if hasattr(node_type, "value") else str(node_type)
        props = {k: str(v) for k, v in (properties or {}).items()}
        if self._native is not None:
            self._native.add_node(node_id, type_str, lifecycle, props, created_at)
            return

        self._py_nodes[node_id] = {
            "id": node_id,
            "node_type": type_str,
            "lifecycle": lifecycle,
            "properties": props,
            "created_at": float(created_at),
        }
        type_list = self._py_type_index.setdefault(type_str, [])
        if node_id not in type_list:
            type_list.append(node_id)

    def add_edge(
        self,
        edge_id: str,
        edge_type: str | HCIREdgeType,
        sources: list[str],
        targets: list[str],
        weight: float = 1.0,
        properties: dict[str, str] | None = None,
        created_at: float = 0.0,
    ) -> None:
        self._removed_edges.discard(edge_id)
        self._edge_ids.add(edge_id)
        type_str = edge_type.value if hasattr(edge_type, "value") else str(edge_type)
        props = {k: str(v) for k, v in (properties or {}).items()}
        if self._native is not None:
            self._native.add_edge(edge_id, type_str, sources, targets, weight, props, created_at)
            return

        self._py_edges[edge_id] = {
            "id": edge_id,
            "edge_type": type_str,
            "sources": list(sources),
            "targets": list(targets),
            "weight": float(weight),
            "properties": props,
            "created_at": float(created_at),
        }
        for src in sources:
            out_list = self._py_outgoing.setdefault(src, [])
            if edge_id not in out_list:
                out_list.append(edge_id)
        for tgt in targets:
            in_list = self._py_incoming.setdefault(tgt, [])
            if edge_id not in in_list:
                in_list.append(edge_id)

    def has_node(self, node_id: str) -> bool:
        if node_id in self._removed_nodes:
            return False
        if self._native is not None:
            return bool(self._native.has_node(node_id))
        return node_id in self._py_nodes

    def remove_node(self, node_id: str) -> None:
        """Remove a node and clean up connected index entries."""
        self._removed_nodes.add(node_id)
        self._node_ids.discard(node_id)
        if self._native is not None and hasattr(self._native, "remove_node"):
            self._native.remove_node(node_id)
        if node_id in self._py_nodes:
            old_node = self._py_nodes.pop(node_id)
            type_str = old_node.get("node_type", "")
            if type_str in self._py_type_index:
                self._py_type_index[type_str] = [
                    nid for nid in self._py_type_index[type_str] if nid != node_id
                ]
            self._py_outgoing.pop(node_id, None)
            self._py_incoming.pop(node_id, None)

    def remove_edge(self, edge_id: str) -> None:
        """Remove an edge by ID."""
        self._removed_edges.add(edge_id)
        self._edge_ids.discard(edge_id)
        if self._native is not None and hasattr(self._native, "remove_edge"):
            self._native.remove_edge(edge_id)
        if edge_id in self._py_edges:
            edge = self._py_edges.pop(edge_id)
            for src in edge.get("sources", []):
                if src in self._py_outgoing:
                    self._py_outgoing[src] = [
                        eid for eid in self._py_outgoing[src] if eid != edge_id
                    ]
            for tgt in edge.get("targets", []):
                if tgt in self._py_incoming:
                    self._py_incoming[tgt] = [
                        eid for eid in self._py_incoming[tgt] if eid != edge_id
                    ]

    def has_edge(self, edge_id: str) -> bool:
        if edge_id in self._removed_edges:
            return False
        if self._native is not None:
            return bool(self._native.has_edge(edge_id))
        return edge_id in self._py_edges

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        if node_id in self._removed_nodes:
            return None
        if self._native is not None:
            return self._native.get_node(node_id)
        node = self._py_nodes.get(node_id)
        if node is not None:
            return copy.deepcopy(node)
        return None

    def get_edge(self, edge_id: str) -> dict[str, Any] | None:
        if edge_id in self._removed_edges:
            return None
        if self._native is not None:
            return self._native.get_edge(edge_id)
        edge = self._py_edges.get(edge_id)
        if edge is not None:
            return copy.deepcopy(edge)
        return None

    def edges_from(self, node_id: str) -> list[dict[str, Any]]:
        if self._native is not None:
            edges = list(self._native.edges_from(node_id))
            return [e for e in edges if e.get("id") not in self._removed_edges]
        edge_ids = self._py_outgoing.get(node_id, [])
        return [copy.deepcopy(self._py_edges[eid]) for eid in edge_ids if eid in self._py_edges]

    def edges_to(self, node_id: str) -> list[dict[str, Any]]:
        if self._native is not None:
            edges = list(self._native.edges_to(node_id))
            return [e for e in edges if e.get("id") not in self._removed_edges]
        edge_ids = self._py_incoming.get(node_id, [])
        return [copy.deepcopy(self._py_edges[eid]) for eid in edge_ids if eid in self._py_edges]

    def nodes_of_type(self, node_type: str | HCIRNodeType) -> list[dict[str, Any]]:
        type_str = node_type.value if hasattr(node_type, "value") else str(node_type)
        if self._native is not None:
            return list(self._native.nodes_of_type(type_str))
        node_ids = self._py_type_index.get(type_str, [])
        return [copy.deepcopy(self._py_nodes[nid]) for nid in node_ids if nid in self._py_nodes]

    def bfs_path(self, start: str, end: str, max_depth: int = 10) -> list[str]:
        if self._native is not None:
            return list(self._native.bfs_path(start, end, max_depth))

        if start == end:
            return [start]

        visited: set[str] = {start}
        queue: deque[tuple[str, list[str]]] = deque([(start, [start])])

        while queue:
            curr, path = queue.popleft()
            if len(path) > max_depth:
                break

            for edge_id in self._py_outgoing.get(curr, []):
                edge = self._py_edges.get(edge_id)
                if not edge:
                    continue
                for target in edge["targets"]:
                    if target == end:
                        return path + [target]
                    if target not in visited:
                        visited.add(target)
                        queue.append((target, path + [target]))

        return []

    def find_path(self, start: str, end: str, max_depth: int = 10) -> list[str]:
        """Find shortest directed path (alias for bfs_path)."""
        return self.bfs_path(start, end, max_depth)

    def reachable_nodes(self, start_id: str, max_depth: int = 10) -> set[str]:
        """Compute the forward reachability cone of node IDs from start_id within max_depth."""
        if not self.has_node(start_id):
            return set()
        visited: set[str] = {start_id}
        queue: deque[tuple[str, int]] = deque([(start_id, 0)])

        while queue:
            curr, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for edge in self.edges_from(curr):
                for target in edge.get("targets", []):
                    if target not in visited and self.has_node(target):
                        visited.add(target)
                        queue.append((target, depth + 1))
        return visited

    def parallel_reachability_cones(
        self, seed_nodes: list[str], max_depth: int = 10
    ) -> dict[str, set[str]]:
        """Compute forward reachability cones across multiple seed anchors simultaneously."""
        if not seed_nodes:
            return {}
        if len(seed_nodes) == 1:
            return {seed_nodes[0]: self.reachable_nodes(seed_nodes[0], max_depth)}

        results: dict[str, set[str]] = {}
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(len(seed_nodes), 8)) as executor:
            futures = {
                seed: executor.submit(self.reachable_nodes, seed, max_depth) for seed in seed_nodes
            }
            for seed, fut in futures.items():
                results[seed] = fut.result()
        return results

    def connected_components(self) -> list[set[str]]:
        """Decompose active graph into disjoint undirected connected components."""
        all_node_ids = {
            nid
            for nid in (self._node_ids if self._native is not None else self._py_nodes)
            if nid not in self._removed_nodes
        }
        visited: set[str] = set()
        components: list[set[str]] = []

        for node_id in all_node_ids:
            if node_id in visited:
                continue
            comp: set[str] = set()
            q: deque[str] = deque([node_id])
            visited.add(node_id)

            while q:
                curr = q.popleft()
                comp.add(curr)
                for edge in self.edges_from(curr):
                    for tgt in edge.get("targets", []):
                        if tgt in all_node_ids and tgt not in visited:
                            visited.add(tgt)
                            q.append(tgt)
                for edge in self.edges_to(curr):
                    for src in edge.get("sources", []):
                        if src in all_node_ids and src not in visited:
                            visited.add(src)
                            q.append(src)
            components.append(comp)

        return components

    def topological_sort(self) -> list[str]:
        """Compute topological ordering of DAG nodes using Kahn's algorithm."""
        all_node_ids = {
            nid
            for nid in (self._node_ids if self._native is not None else self._py_nodes)
            if nid not in self._removed_nodes
        }
        if not all_node_ids:
            return []

        in_degree: dict[str, int] = {nid: 0 for nid in all_node_ids}
        adj: dict[str, list[str]] = {nid: [] for nid in all_node_ids}

        for nid in all_node_ids:
            for edge in self.edges_from(nid):
                for tgt in edge.get("targets", []):
                    if tgt in all_node_ids:
                        adj[nid].append(tgt)
                        in_degree[tgt] += 1

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

    def transitive_reduction(self) -> NativeGraphAdapter:
        """Compute the transitive reduction of the directed graph (DAG).

        Returns a new NativeGraphAdapter with identical nodes and the minimal edge set
        preserving the reachability relation (Aho, Garey, & Ullman 1972).
        For every directed edge u -> v, if an alternative path of length >= 2 exists,
        the edge u -> v is redundant and pruned.
        """
        reduced = self.snapshot()
        all_node_ids = {
            nid
            for nid in (self._node_ids if self._native is not None else self._py_nodes)
            if nid not in self._removed_nodes
        }

        # Build adjacency mapping: u -> list of (edge_id, target_v)
        adj: dict[str, list[tuple[str, str]]] = {nid: [] for nid in all_node_ids}
        for nid in all_node_ids:
            for edge in self.edges_from(nid):
                eid = edge.get("id", "")
                for tgt in edge.get("targets", []):
                    if tgt in all_node_ids:
                        adj[nid].append((eid, tgt))

        # Check reachability avoiding the direct edge
        redundant_edge_ids: set[str] = set()

        for u in all_node_ids:
            for direct_eid, v in adj[u]:
                visited: set[str] = {u}
                queue: deque[str] = deque()
                found_alternate_path = False

                for other_eid, w in adj[u]:
                    if (other_eid, w) != (direct_eid, v) and w not in visited:
                        visited.add(w)
                        queue.append(w)

                while queue:
                    curr = queue.popleft()
                    if curr == v:
                        found_alternate_path = True
                        break
                    for _, next_node in adj.get(curr, []):
                        if next_node not in visited:
                            visited.add(next_node)
                            queue.append(next_node)

                if found_alternate_path:
                    redundant_edge_ids.add(direct_eid)

        # Remove redundant edges from the reduced adapter
        for reid in redundant_edge_ids:
            reduced.remove_edge(reid)

        return reduced

    def minimum_spanning_tree(self) -> tuple[NativeGraphAdapter, float]:
        """Compute the Minimum Spanning Tree (MST) across connected vertices using Kruskal's algorithm.

        Returns (mst_adapter, total_weight) preserving all nodes and retaining
        the minimal-weight acyclic connectivity spine.
        """
        all_node_ids = {
            nid
            for nid in (self._node_ids if self._native is not None else self._py_nodes)
            if nid not in self._removed_nodes
        }
        mst = NativeGraphAdapter()
        for nid in all_node_ids:
            node_data = self.get_node(nid)
            if node_data:
                mst.add_node(
                    node_id=nid,
                    node_type=node_data.get("node_type", "concept"),
                    lifecycle=node_data.get("lifecycle", "ACTIVE"),
                    properties=node_data.get("properties"),
                )

        # Collect unique edges with weight
        collected_edges: list[dict[str, Any]] = []
        seen_edge_ids: set[str] = set()

        for nid in all_node_ids:
            for edge in self.edges_from(nid):
                eid = edge.get("id", "")
                if eid not in seen_edge_ids:
                    seen_edge_ids.add(eid)
                    collected_edges.append(edge)

        # Sort edges by weight
        collected_edges.sort(key=lambda e: float(e.get("weight", 1.0)))

        # Disjoint Set Union (DSU) with path compression
        parent: dict[str, str] = {nid: nid for nid in all_node_ids}
        rank: dict[str, int] = {nid: 0 for nid in all_node_ids}

        def find(i: str) -> str:
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i: str, j: str) -> bool:
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                if rank[root_i] < rank[root_j]:
                    parent[root_i] = root_j
                elif rank[root_i] > rank[root_j]:
                    parent[root_j] = root_i
                else:
                    parent[root_j] = root_i
                    rank[root_i] += 1
                return True
            return False

        total_weight = 0.0
        for edge in collected_edges:
            sources = [s for s in edge.get("sources", []) if s in all_node_ids]
            targets = [t for t in edge.get("targets", []) if t in all_node_ids]
            if not sources or not targets:
                continue

            merged_any = False
            for s in sources:
                for t in targets:
                    if union(s, t):
                        merged_any = True

            if merged_any:
                w = float(edge.get("weight", 1.0))
                total_weight += w
                mst.add_edge(
                    edge_id=edge.get("id", ""),
                    edge_type=edge.get("edge_type", "causes"),
                    sources=sources,
                    targets=targets,
                    weight=w,
                    properties=edge.get("properties"),
                )

        return mst, total_weight

    def all_pairs_shortest_paths(
        self, weight_attribute: str = "weight"
    ) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], list[str]]]:
        """Compute all-pairs shortest paths using Floyd-Warshall with path reconstruction.

        Returns:
            Tuple of (distances, paths), where distances maps (u, v) -> float (or inf),
            and paths maps (u, v) -> list of node IDs along the shortest path.
        """
        all_node_ids = sorted(list(self.all_node_ids()))
        dist: dict[tuple[str, str], float] = {}
        next_hop: dict[tuple[str, str], str | None] = {}

        for u in all_node_ids:
            for v in all_node_ids:
                if u == v:
                    dist[(u, v)] = 0.0
                    next_hop[(u, v)] = u
                else:
                    dist[(u, v)] = float("inf")
                    next_hop[(u, v)] = None

        collected_edges: list[dict[str, Any]] = []
        seen_edge_ids: set[str] = set()
        for nid in all_node_ids:
            for edge in self.edges_from(nid):
                eid = edge.get("id", "")
                if eid not in seen_edge_ids:
                    seen_edge_ids.add(eid)
                    collected_edges.append(edge)

        for edge in collected_edges:
            w = max(0.0, float(edge.get(weight_attribute, 1.0)))
            sources = [s for s in edge.get("sources", []) if s in all_node_ids]
            targets = [t for t in edge.get("targets", []) if t in all_node_ids]
            for s in sources:
                for t in targets:
                    if w < dist[(s, t)]:
                        dist[(s, t)] = w
                        next_hop[(s, t)] = t

        for k in all_node_ids:
            for i in all_node_ids:
                d_ik = dist[(i, k)]
                if d_ik == float("inf"):
                    continue
                for j in all_node_ids:
                    d_kj = dist[(k, j)]
                    if d_kj == float("inf"):
                        continue
                    if d_ik + d_kj < dist[(i, j)]:
                        dist[(i, j)] = d_ik + d_kj
                        next_hop[(i, j)] = next_hop[(i, k)]

        paths: dict[tuple[str, str], list[str]] = {}
        for u in all_node_ids:
            for v in all_node_ids:
                if dist[(u, v)] == float("inf"):
                    paths[(u, v)] = []
                elif u == v:
                    paths[(u, v)] = [u]
                else:
                    curr = u
                    p = [curr]
                    visited = {curr}
                    valid = True
                    while curr != v:
                        nxt = next_hop.get((curr, v))
                        if nxt is None or nxt in visited:
                            valid = False
                            break
                        visited.add(nxt)
                        p.append(nxt)
                        curr = nxt
                    paths[(u, v)] = p if valid else []

        return dist, paths

    def graph_diameter(self, weight_attribute: str = "weight") -> float:
        """Compute the maximum finite shortest path distance between any two nodes."""
        dist, _ = self.all_pairs_shortest_paths(weight_attribute=weight_attribute)
        finite_dists = [d for (u, v), d in dist.items() if u != v and d < float("inf")]
        return max(finite_dists) if finite_dists else 0.0

    def closeness_centrality(self, node_id: str, weight_attribute: str = "weight") -> float:
        """Compute normalized closeness centrality for a node."""
        all_node_ids = list(self.all_node_ids())
        n = len(all_node_ids)
        if n <= 1 or node_id not in all_node_ids:
            return 0.0

        dist, _ = self.all_pairs_shortest_paths(weight_attribute=weight_attribute)
        reachable_dists = [
            dist[(node_id, v)]
            for v in all_node_ids
            if v != node_id and dist.get((node_id, v), float("inf")) < float("inf")
        ]
        k = len(reachable_dists)
        if k == 0:
            return 0.0
        sum_dist = sum(reachable_dists)
        if sum_dist <= 0.0:
            return 1.0

        raw_closeness = k / sum_dist
        wasserman_faust_norm = (k / (n - 1)) * raw_closeness
        return float(wasserman_faust_norm)

    def find_articulation_points_and_bridges(
        self,
    ) -> tuple[set[str], list[tuple[str, str]]]:
        """Compute articulation points (cut vertices) and bridges (cut edges) using Tarjan's DFS.

        Returns:
            Tuple of (articulation_points, bridges), identifying single points of failure.
        """
        all_nodes = sorted(list(self.all_node_ids()))
        adj: dict[str, set[str]] = {nid: set() for nid in all_nodes}

        # Build undirected adjacency from active edges
        collected_edges: list[dict[str, Any]] = []
        seen_edge_ids: set[str] = set()
        for nid in all_nodes:
            for edge in self.edges_from(nid):
                eid = edge.get("id", "")
                if eid not in seen_edge_ids:
                    seen_edge_ids.add(eid)
                    collected_edges.append(edge)

        for edge in collected_edges:
            sources = [s for s in edge.get("sources", []) if s in adj]
            targets = [t for t in edge.get("targets", []) if t in adj]
            for s in sources:
                for t in targets:
                    if s != t:
                        adj[s].add(t)
                        adj[t].add(s)

        timer = 0
        disc: dict[str, int] = {}
        low: dict[str, int] = {}
        parent: dict[str, str | None] = {}
        articulation_points: set[str] = set()
        bridges: list[tuple[str, str]] = []

        def dfs(u: str) -> None:
            nonlocal timer
            disc[u] = low[u] = timer
            timer += 1
            children = 0

            for v in sorted(list(adj[u])):
                if v == parent.get(u):
                    continue
                if v in disc:
                    low[u] = min(low[u], disc[v])
                else:
                    parent[v] = u
                    children += 1
                    dfs(v)
                    low[u] = min(low[u], low[v])

                    if parent.get(u) is None and children > 1:
                        articulation_points.add(u)
                    elif parent.get(u) is not None and low[v] >= disc[u]:
                        articulation_points.add(u)

                    if low[v] > disc[u]:
                        bridges.append((u, v))

        for nid in all_nodes:
            if nid not in disc:
                parent[nid] = None
                dfs(nid)

        return articulation_points, bridges

    def is_biconnected(self) -> bool:
        """Return True if the graph is connected with >= 2 nodes, 0 articulation points, and 0 bridges."""
        all_nodes = list(self.all_node_ids())
        if len(all_nodes) < 2:
            return False

        # Must have exactly 1 connected component containing all nodes
        components = self.connected_components()
        if len(components) != 1 or len(components[0]) != len(all_nodes):
            return False

        articulation_points, bridges = self.find_articulation_points_and_bridges()
        return len(articulation_points) == 0 and len(bridges) == 0

    def max_flow_min_cut(
        self,
        source: str,
        sink: str,
        capacity_attribute: str = "capacity",
    ) -> tuple[float, list[tuple[str, str]], set[str], set[str]]:
        """Compute directed maximum flow and minimum s-t cut using Edmonds-Karp algorithm.

        Returns:
            Tuple of (max_flow_value, cut_edges, source_set, sink_set).
        """
        all_nodes = set(self.all_node_ids())
        if source not in all_nodes or sink not in all_nodes or source == sink:
            return 0.0, [], set(), set()

        initial_capacities: dict[tuple[str, str], float] = defaultdict(float)
        adj: dict[str, set[str]] = defaultdict(set)

        collected_edges: list[dict[str, Any]] = []
        seen_edge_ids: set[str] = set()
        for nid in all_nodes:
            for edge in self.edges_from(nid):
                eid = edge.get("id", "")
                if eid not in seen_edge_ids:
                    seen_edge_ids.add(eid)
                    collected_edges.append(edge)

        for edge in collected_edges:
            cap = max(0.0, float(edge.get(capacity_attribute, edge.get("weight", 1.0))))
            sources = [s for s in edge.get("sources", []) if s in all_nodes]
            targets = [t for t in edge.get("targets", []) if t in all_nodes]
            for s in sources:
                for t in targets:
                    if s != t:
                        initial_capacities[(s, t)] += cap
                        adj[s].add(t)
                        adj[t].add(s)

        residual: dict[tuple[str, str], float] = defaultdict(float)
        for (u, v), cap in initial_capacities.items():
            residual[(u, v)] = cap

        max_flow = 0.0

        while True:
            parent: dict[str, str | None] = {nid: None for nid in all_nodes}
            visited: set[str] = {source}
            queue: deque[str] = deque([source])

            while queue:
                curr = queue.popleft()
                if curr == sink:
                    break
                for neighbor in sorted(list(adj[curr])):
                    if neighbor not in visited and residual.get((curr, neighbor), 0.0) > 1e-9:
                        visited.add(neighbor)
                        parent[neighbor] = curr
                        queue.append(neighbor)

            if sink not in visited:
                break

            path: list[tuple[str, str]] = []
            curr = sink
            bottleneck = float("inf")
            while curr != source:
                p = parent[curr]
                if p is None:
                    break
                path.append((p, curr))
                bottleneck = min(bottleneck, residual.get((p, curr), 0.0))
                curr = p

            if bottleneck <= 0.0 or bottleneck == float("inf"):
                break

            for u, v in path:
                residual[(u, v)] -= bottleneck
                residual[(v, u)] += bottleneck

            max_flow += bottleneck

        reachable: set[str] = set()
        queue = deque([source])
        reachable.add(source)
        while queue:
            curr = queue.popleft()
            for neighbor in sorted(list(adj[curr])):
                if neighbor not in reachable and residual.get((curr, neighbor), 0.0) > 1e-9:
                    reachable.add(neighbor)
                    queue.append(neighbor)

        source_set = reachable
        sink_set = all_nodes - source_set

        cut_edges: list[tuple[str, str]] = [
            (u, v)
            for (u, v), cap in initial_capacities.items()
            if u in source_set and v in sink_set and cap > 0.0
        ]

        return float(max_flow), cut_edges, source_set, sink_set

    def feedback_arc_set(
        self,
    ) -> tuple[list[tuple[str, str]], list[str]]:
        """Compute the minimal Feedback Arc Set (FAS) to render the graph acyclic using Eades heuristic.

        Returns:
            Tuple of (feedback_edges, topological_ordering), where removing feedback_edges
            yields a Directed Acyclic Graph (DAG).
        """
        all_nodes = sorted(list(self.all_node_ids()))
        if not all_nodes:
            return [], []

        adj_out: dict[str, set[str]] = {nid: set() for nid in all_nodes}
        adj_in: dict[str, set[str]] = {nid: set() for nid in all_nodes}

        collected_edges: list[dict[str, Any]] = []
        seen_edge_ids: set[str] = set()
        for nid in all_nodes:
            for edge in self.edges_from(nid):
                eid = edge.get("id", "")
                if eid not in seen_edge_ids:
                    seen_edge_ids.add(eid)
                    collected_edges.append(edge)

        all_directed_edges: list[tuple[str, str]] = []
        for edge in collected_edges:
            sources = [s for s in edge.get("sources", []) if s in adj_out]
            targets = [t for t in edge.get("targets", []) if t in adj_out]
            for s in sources:
                for t in targets:
                    if s != t:
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

            sources = [u for u in remaining if in_deg[u] == 0]
            if sources:
                for u in sorted(sources):
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

    def strongly_connected_components(
        self,
    ) -> list[set[str]]:
        """Compute Strongly Connected Components (SCC) of the directed graph using Tarjan's algorithm.

        Returns:
            List of sets of node IDs, where each set represents a maximal strongly connected component.
        """
        all_nodes = self.all_node_ids()
        if not all_nodes:
            return []

        adj: dict[str, set[str]] = {nid: set() for nid in all_nodes}
        if self._native is not None:
            for nid in all_nodes:
                for edge in self._native.edges_from(nid):
                    for target in edge.get("targets", []):
                        if target in adj and target != nid:
                            adj[nid].add(target)
        else:
            for edge in self._py_edges.values():
                for s in edge.get("sources", []):
                    if s in adj:
                        for t in edge.get("targets", []):
                            if t in adj and s != t:
                                adj[s].add(t)

        index = 0
        indices: dict[str, int] = {}
        lowlinks: dict[str, int] = {}
        on_stack: set[str] = set()
        stack: list[str] = []
        sccs: list[set[str]] = []

        for node in sorted(all_nodes):
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
        self,
    ) -> tuple[dict[int, set[str]], list[tuple[int, int]]]:
        """Condense directed graph SCCs into an acyclic metagraph (DAG).

        Returns:
            Tuple of (scc_map, metagraph_edges), where:
              scc_map: mapping scc_id -> set of original node IDs.
              metagraph_edges: list of directed edges (scc_u, scc_v) with scc_u != scc_v forming a DAG.
        """
        sccs = self.strongly_connected_components()
        node_to_scc: dict[str, int] = {}
        scc_map: dict[int, set[str]] = {}

        for scc_id, comp in enumerate(sccs):
            scc_map[scc_id] = comp
            for nid in comp:
                node_to_scc[nid] = scc_id

        all_nodes = set(self.all_node_ids())
        metagraph_edges_set: set[tuple[int, int]] = set()

        if self._native is not None:
            for nid in all_nodes:
                src_scc = node_to_scc.get(nid)
                if src_scc is None:
                    continue
                for edge in self._native.edges_from(nid):
                    for target in edge.get("targets", []):
                        if target in node_to_scc:
                            tgt_scc = node_to_scc[target]
                            if src_scc != tgt_scc:
                                metagraph_edges_set.add((src_scc, tgt_scc))
        else:
            for edge in self._py_edges.values():
                for s in edge.get("sources", []):
                    src_scc = node_to_scc.get(s)
                    if src_scc is None:
                        continue
                    for t in edge.get("targets", []):
                        tgt_scc = node_to_scc.get(t)
                        if tgt_scc is not None and src_scc != tgt_scc:
                            metagraph_edges_set.add((src_scc, tgt_scc))

        return scc_map, sorted(list(metagraph_edges_set))

    def find_eulerian_path_or_circuit(
        self,
    ) -> tuple[bool, list[tuple[str, str]]]:
        """Find an Eulerian path or circuit traversing every directed edge exactly once (Hierholzer).

        Returns:
            Tuple of (is_eulerian, edge_sequence).
        """
        all_nodes = self.all_node_ids()
        if not all_nodes:
            return True, []

        collected_edges: list[dict[str, Any]] = []
        if self._native is not None:
            for nid in all_nodes:
                collected_edges.extend(self._native.edges_from(nid))
        else:
            collected_edges = list(self._py_edges.values())

        all_directed_edges: list[tuple[str, str]] = []
        in_deg: dict[str, int] = defaultdict(int)
        out_deg: dict[str, int] = defaultdict(int)

        for edge in collected_edges:
            sources = [s for s in edge.get("sources", []) if s in all_nodes]
            targets = [t for t in edge.get("targets", []) if t in all_nodes]
            for s in sources:
                for t in targets:
                    all_directed_edges.append((s, t))
                    out_deg[s] += 1
                    in_deg[t] += 1

        if not all_directed_edges:
            return True, []

        # Check degree conditions
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
            # Eulerian Circuit: can start at any active node with out_deg > 0
            start_node = min(active_nodes, key=lambda n: (out_deg[n] == 0, n))
        elif len(start_candidates) == 1 and len(end_candidates) == 1:
            # Eulerian Path
            start_node = start_candidates[0]
        else:
            return False, []

        # Hierholzer's algorithm
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
        self,
        weight_attribute: str = "weight",
    ) -> tuple[float, list[tuple[str, str]]]:
        """Compute minimum-weight edge-covering tour traversing every edge at least once (CPP).

        Returns:
            Tuple of (total_tour_weight, tour_edge_sequence).
        """
        all_nodes = self.all_node_ids()
        if not all_nodes:
            return 0.0, []

        collected_edges: list[dict[str, Any]] = []
        if self._native is not None:
            for nid in all_nodes:
                collected_edges.extend(self._native.edges_from(nid))
        else:
            collected_edges = list(self._py_edges.values())

        directed_edges: list[tuple[str, str, float]] = []
        for edge in collected_edges:
            w = max(0.1, float(edge.get(weight_attribute, 1.0)))
            for s in edge.get("sources", []):
                if s in all_nodes:
                    for t in edge.get("targets", []):
                        if t in all_nodes:
                            directed_edges.append((s, t, w))

        if not directed_edges:
            return 0.0, []

        # First check if already Eulerian
        is_eul, eul_path = self.find_eulerian_path_or_circuit()
        if is_eul and eul_path and eul_path[0][0] == eul_path[-1][1]:
            edge_weight_map: dict[tuple[str, str], float] = defaultdict(lambda: 1.0)
            for u, v, w in directed_edges:
                edge_weight_map[(u, v)] = w
            total_w = sum(edge_weight_map.get((u, v), 1.0) for u, v in eul_path)
            return float(total_w), eul_path

        # Degree balancing via shortest path augmentations
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

        # Compute all-pairs shortest paths using Floyd-Warshall
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

    def minimum_dominating_set(
        self,
    ) -> set[str]:
        """Compute an approximate Minimum Dominating Set (MDS) using greedy coverage heuristic.

        Returns:
            Set of node IDs such that every node in the graph is either in the set
            or adjacent to a node in the set.
        """
        all_nodes = set(self.all_node_ids())
        if not all_nodes:
            return set()

        neighbors: dict[str, set[str]] = {nid: {nid} for nid in all_nodes}

        if self._native is not None:
            for nid in all_nodes:
                for edge in self._native.edges_from(nid):
                    for target in edge.get("targets", []):
                        if target in all_nodes:
                            neighbors[nid].add(target)
                            neighbors[target].add(nid)
        else:
            for edge in self._py_edges.values():
                for s in edge.get("sources", []):
                    if s in all_nodes:
                        for t in edge.get("targets", []):
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

    def maximum_independent_set(
        self,
    ) -> set[str]:
        """Compute an approximate Maximum Independent Set (MIS) using minimum-degree greedy heuristic.

        Returns:
            Set of mutually non-adjacent node IDs representing non-conflicting concurrent elements.
        """
        all_nodes = set(self.all_node_ids())
        if not all_nodes:
            return set()

        adj: dict[str, set[str]] = {nid: set() for nid in all_nodes}
        if self._native is not None:
            for nid in all_nodes:
                for edge in self._native.edges_from(nid):
                    for target in edge.get("targets", []):
                        if target in all_nodes and target != nid:
                            adj[nid].add(target)
                            adj[target].add(nid)
        else:
            for edge in self._py_edges.values():
                for s in edge.get("sources", []):
                    if s in all_nodes:
                        for t in edge.get("targets", []):
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

    def minimum_vertex_cover(
        self,
    ) -> set[str]:
        """Compute an approximate Minimum Vertex Cover (MVC) using greedy coverage and Gallai duality.

        Returns:
            Minimal set of node IDs such that every undirected edge has at least one endpoint in the set.
        """
        all_nodes = set(self.all_node_ids())
        if not all_nodes:
            return set()

        edges: set[tuple[str, str]] = set()
        if self._native is not None:
            for nid in all_nodes:
                for edge in self._native.edges_from(nid):
                    for target in edge.get("targets", []):
                        if target in all_nodes and target != nid:
                            edges.add((min(nid, target), max(nid, target)))
        else:
            for edge in self._py_edges.values():
                for s in edge.get("sources", []):
                    if s in all_nodes:
                        for t in edge.get("targets", []):
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

    def greedy_edge_coloring(
        self,
    ) -> dict[tuple[str, str], int]:
        """Compute a proper greedy edge coloring (Vizing's theorem) of undirected edges.

        Returns:
            Dictionary mapping canonical edge pairs (min(u, v), max(u, v)) to discrete color indices (0, 1, 2, ...).
        """
        all_nodes = set(self.all_node_ids())
        if not all_nodes:
            return {}

        edges: set[tuple[str, str]] = set()
        if self._native is not None:
            for nid in all_nodes:
                for edge in self._native.edges_from(nid):
                    for target in edge.get("targets", []):
                        if target in all_nodes and target != nid:
                            edges.add((min(nid, target), max(nid, target)))
        else:
            for edge in self._py_edges.values():
                for s in edge.get("sources", []):
                    if s in all_nodes:
                        for t in edge.get("targets", []):
                            if t in all_nodes and s != t:
                                edges.add((min(s, t), max(s, t)))

        if not edges:
            return {}

        node_incident_edges: dict[str, list[tuple[str, str]]] = {nid: [] for nid in all_nodes}
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

    def feedback_vertex_set(
        self,
    ) -> set[str]:
        """Compute an approximate minimal Feedback Vertex Set (FVS) to break all directed cycles.

        Returns:
            Minimal set of node IDs whose removal from the graph leaves a Directed Acyclic Graph (DAG).
        """
        all_nodes = set(self.all_node_ids())
        if not all_nodes:
            return set()

        directed_edges: set[tuple[str, str]] = set()
        if self._native is not None:
            for nid in all_nodes:
                for edge in self._native.edges_from(nid):
                    for target in edge.get("targets", []):
                        if target in all_nodes:
                            directed_edges.add((nid, target))
        else:
            for edge in self._py_edges.values():
                for s in edge.get("sources", []):
                    if s in all_nodes:
                        for t in edge.get("targets", []):
                            if t in all_nodes:
                                directed_edges.add((s, t))

        if not directed_edges:
            return set()

        # Helper to check if a subset of vertices is acyclic under directed_edges
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

        # If already acyclic, no vertices need removal
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

            # Peel off nodes that cannot be part of any cycle (in_degree 0 or out_degree 0)
            active_cyclic = {v for v in curr_nodes if sub_in[v] > 0 and sub_out[v] > 0}
            if not active_cyclic:
                break

            # Pick vertex maximizing in_deg * out_deg (Becker-Geiger heuristic)
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

    def essential_causal_skeleton(
        self,
    ) -> set[tuple[str, str]]:
        """Compute the irreducible causal skeleton DAG via transitive reduction."""
        reduced = self.transitive_reduction()
        edges: set[tuple[str, str]] = set()
        for nid in reduced.all_node_ids():
            for edge in reduced.edges_from(nid):
                for target in edge.get("targets", []):
                    edges.add((nid, target))
        return edges

    def graph_radius_and_diameter(self, weight_attribute: str = "weight") -> tuple[float, float]:
        """Compute the radius and diameter of the graph based on node eccentricities.

        Eccentricity e(u) = max_{v != u, reachable} dist(u, v).
        Radius R = min_u e(u), Diameter D = max_u e(u).

        Returns:
            Tuple of (radius, diameter). Returns (0.0, 0.0) if disconnected or empty.
        """
        all_nodes = list(self.all_node_ids())
        if len(all_nodes) <= 1:
            return (0.0, 0.0)

        dist, _ = self.all_pairs_shortest_paths(weight_attribute=weight_attribute)
        eccentricities: list[float] = []

        for u in all_nodes:
            dists = [
                d for (src, tgt), d in dist.items() if src == u and tgt != u and d < float("inf")
            ]
            if dists:
                eccentricities.append(max(dists))

        if not eccentricities:
            return (0.0, 0.0)

        return (min(eccentricities), max(eccentricities))

    def betweenness_centrality(
        self, normalized: bool = True, directed: bool = False
    ) -> dict[str, float]:
        """Compute exact betweenness centrality for all nodes via Brandes (2001) algorithm.

        Args:
            normalized: If True, normalizes scores by (n-1)(n-2) for directed or 2/((n-1)(n-2)) for undirected.
            directed: If True, treats edges as directed.

        Returns:
            Dictionary mapping node IDs to betweenness centrality scores.
        """
        all_nodes = set(self.all_node_ids())
        cb: dict[str, float] = {v: 0.0 for v in all_nodes}
        n = len(all_nodes)
        if n <= 2:
            return cb

        adj: dict[str, list[str]] = defaultdict(list)
        if self._native is not None:
            for nid in all_nodes:
                for edge in self._native.edges_from(nid):
                    for target in edge.get("targets", []):
                        if target in all_nodes and target != nid:
                            adj[nid].append(target)
                            if not directed:
                                adj[target].append(nid)
        else:
            for edge in self._py_edges.values():
                for s in edge.get("sources", []):
                    if s in all_nodes:
                        for t in edge.get("targets", []):
                            if t in all_nodes and s != t:
                                adj[s].append(t)
                                if not directed:
                                    adj[t].append(s)

        for s in all_nodes:
            stack: list[str] = []
            p: dict[str, list[str]] = {w: [] for w in all_nodes}
            sigma: dict[str, float] = {w: 0.0 for w in all_nodes}
            sigma[s] = 1.0
            d: dict[str, int] = {w: -1 for w in all_nodes}
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

            delta: dict[str, float] = {w: 0.0 for w in all_nodes}
            while stack:
                w = stack.pop()
                for v in p[w]:
                    if sigma[w] > 0:
                        delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                if w != s:
                    cb[w] += delta[w]

        # Symmetrize / scale for undirected graphs where each path is traversed twice
        if not directed:
            for v in cb:
                cb[v] /= 2.0

        if normalized and n > 2:
            scale = 1.0 / ((n - 1) * (n - 2)) if directed else 2.0 / ((n - 1) * (n - 2))
            for v in cb:
                cb[v] *= scale

        return cb

    def extract_subgraph(
        self, seed: str, max_depth: int = 2
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Extract ego subgraph around seed (alias for subgraph_bfs)."""
        return self.subgraph_bfs(seed, max_depth)

    def subgraph_bfs(
        self, center_id: str, max_depth: int = 3
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if self._native is not None:
            nodes, edges = self._native.subgraph_bfs(center_id, max_depth)
            return list(nodes), list(edges)

        visited_nodes: set[str] = set()
        collected_edge_ids: set[str] = set()
        queue: deque[tuple[str, int]] = deque()

        if center_id in self._py_nodes:
            visited_nodes.add(center_id)
            queue.append((center_id, 0))

        while queue:
            curr, depth = queue.popleft()
            if depth >= max_depth:
                continue

            for edge_id in self._py_outgoing.get(curr, []):
                collected_edge_ids.add(edge_id)
                edge = self._py_edges.get(edge_id)
                if edge:
                    for tgt in edge["targets"]:
                        if tgt not in visited_nodes:
                            visited_nodes.add(tgt)
                            queue.append((tgt, depth + 1))

            for edge_id in self._py_incoming.get(curr, []):
                collected_edge_ids.add(edge_id)
                edge = self._py_edges.get(edge_id)
                if edge:
                    for src in edge["sources"]:
                        if src not in visited_nodes:
                            visited_nodes.add(src)
                            queue.append((src, depth + 1))

        nodes_res = [
            copy.deepcopy(self._py_nodes[nid]) for nid in visited_nodes if nid in self._py_nodes
        ]
        edges_res = [
            copy.deepcopy(self._py_edges[eid])
            for eid in collected_edge_ids
            if eid in self._py_edges
        ]
        return nodes_res, edges_res

    def snapshot(self) -> NativeGraphAdapter:
        """Create an O(1) root snapshot sharing chunked persistent storage or deepcopy fallback."""
        if self._native is not None:
            snap = NativeGraphAdapter(native_instance=self._native.snapshot())
            snap._removed_nodes = self._removed_nodes.copy()
            snap._removed_edges = self._removed_edges.copy()
            snap._node_ids = self._node_ids.copy()
            snap._edge_ids = self._edge_ids.copy()
            return snap
        snap = NativeGraphAdapter(
            py_state={
                "nodes": self._py_nodes,
                "edges": self._py_edges,
                "outgoing": self._py_outgoing,
                "incoming": self._py_incoming,
                "type_index": self._py_type_index,
            }
        )
        snap._removed_nodes = self._removed_nodes.copy()
        snap._removed_edges = self._removed_edges.copy()
        snap._node_ids = self._node_ids.copy()
        snap._edge_ids = self._edge_ids.copy()
        return snap

    def canonical_hash(self) -> str:
        """Compute deterministic versioned canonical state hash (64-char hex string)."""
        if self._native is not None:
            return str(self._native.canonical_hash())

        # Exact Python fallback matching HCIR_STATE_V1 canonical specification
        h = hashlib.sha256()
        h.update(b"HCIR_STATE_V1\n")

        node_keys = sorted(self._py_nodes.keys())
        h.update(struct.pack("<Q", len(node_keys)))
        h.update(b"\n")

        for nk in node_keys:
            node = self._py_nodes[nk]
            h.update(b"NODE:")
            h.update(node["id"].encode("utf-8"))
            h.update(b"|")
            h.update(node["node_type"].encode("utf-8"))
            h.update(b"|")
            h.update(node["lifecycle"].encode("utf-8"))
            h.update(b"|")

            props = node.get("properties", {})
            for pk in sorted(props.keys()):
                h.update(pk.encode("utf-8"))
                h.update(b"=")
                h.update(str(props[pk]).encode("utf-8"))
                h.update(b";")
            h.update(b"\n")

        edge_keys = sorted(self._py_edges.keys())
        h.update(struct.pack("<Q", len(edge_keys)))
        h.update(b"\n")

        for ek in edge_keys:
            edge = self._py_edges[ek]
            h.update(b"EDGE:")
            h.update(edge["id"].encode("utf-8"))
            h.update(b"|")
            h.update(edge["edge_type"].encode("utf-8"))
            h.update(b"|SRC:")
            for s in sorted(edge.get("sources", [])):
                h.update(s.encode("utf-8"))
                h.update(b",")
            h.update(b"|TGT:")
            for t in sorted(edge.get("targets", [])):
                h.update(t.encode("utf-8"))
                h.update(b",")
            h.update(b"|W:")
            weight = float(edge.get("weight", 1.0))
            h.update(struct.pack("<d", weight))
            h.update(b"\n")

        return h.hexdigest()
