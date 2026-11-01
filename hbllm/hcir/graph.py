"""
Typed Cognitive Hypergraph — HCIR §3.

The core data structure of the HCIR workspace.  A single graph instance
holds all cognitive entities (nodes) and semantic relationships (edges).
Different subsystems access it through **views** (filtered projections)
rather than separate graph objects:

    Graph
      ├── Knowledge View   (Facts, Beliefs, Concepts, Procedures)
      ├── Execution View   (Goals, Actions, active threads)
      ├── Simulation View  (Forked branches)
      └── Memory View      (Episodes, Skills, Values)

Graph nodes are **typed subclasses**, not untyped dictionaries.
Each subclass carries typed payload fields validated at construction.

Design invariants:
    - Node IDs are globally unique within a graph instance.
    - Edge IDs are globally unique within a graph instance.
    - Edges are *hyper*-edges: they can link multiple source/target nodes.
    - The graph never owns persistence.  Storage is delegated to ``IGraphStore``.
"""

# ruff: noqa: I001, F401, F403, F405

from __future__ import annotations

import copy
import hashlib
import json
import logging
from collections import deque
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, Field

import hbllm.hcir.algorithms as algorithms
from hbllm.hcir.nodes import *
from hbllm.hcir.nodes import _new_id
from hbllm.hcir.nodes import __all__ as _nodes_all
from hbllm.hcir.proposition import (
    Proposition,
    SpatialContext,
    TemporalValidity,
)
from hbllm.hcir.types import (
    Attention,
    BeliefConfidence,
    CognitiveMode,
    Confidence,
    CostMetric,
    DiscoveryTrigger,
    EpistemicLifecycle,
    EvidenceStrength,
    ExperimentRealityLevel,
    ExperimentStatus,
    FalsificationStatus,
    KnowledgeValue,
    PerceptualContradictionLevel,
    PerceptualEpistemicProfile,
    Priority,
    Provenance,
    ResearchStrategyType,
    Scope,
    TimeDuration,
    Timestamp,
    UncertaintyVector,
)

logger = logging.getLogger(__name__)

_types_all = [
    "Proposition",
    "SpatialContext",
    "TemporalValidity",
    "Attention",
    "BeliefConfidence",
    "CognitiveMode",
    "Confidence",
    "CostMetric",
    "DiscoveryTrigger",
    "EpistemicLifecycle",
    "EvidenceStrength",
    "ExperimentRealityLevel",
    "ExperimentStatus",
    "FalsificationStatus",
    "KnowledgeValue",
    "PerceptualContradictionLevel",
    "PerceptualEpistemicProfile",
    "Priority",
    "Provenance",
    "ResearchStrategyType",
    "Scope",
    "TimeDuration",
    "Timestamp",
    "UncertaintyVector",
]

__all__ = (
    _nodes_all
    + _types_all
    + [
        "HCIREdge",
        "CognitiveGraph",
        "SubstrateDivergenceError",
    ]
)


class HCIREdge(BaseModel):
    """A typed hyperedge connecting multiple source and target nodes.

    Hyperedges allow representations like "Goal G1 depends on
    Constraint C1, Resource R1, and Capability Cap1" as a single
    relationship.
    """

    id: str = Field(default_factory=lambda: _new_id("e"))
    edge_type: HCIREdgeType
    sources: list[str]  # Node IDs
    targets: list[str]  # Node IDs
    properties: dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance = Field(default_factory=Provenance)
    weight: float = 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive Graph — single graph with views
# ═══════════════════════════════════════════════════════════════════════════


class SubstrateDivergenceError(RuntimeError):
    """Raised when native graph acceleration substrate diverges from authoritative Python graph."""


class CognitiveGraph:
    """The unified typed cognitive hypergraph.

    A single graph instance holds all cognitive entities.  Different
    subsystems access filtered *views* (knowledge, execution, memory,
    simulation) rather than separate graph objects.

    Internally maintains hash-map indexes for O(1) lookups by ID,
    and secondary indexes by type, category, lifecycle, scope, and
    capability for efficient querying.
    """

    def __init__(self, enable_native_substrate: bool = True) -> None:
        # ── Primary storage ──
        self._nodes: dict[str, HCIRNode] = {}
        self._edges: dict[str, HCIREdge] = {}

        # ── Secondary indexes (type → set of node IDs) ──
        self._idx_by_type: dict[HCIRNodeType, set[str]] = {t: set() for t in HCIRNodeType}
        self._idx_by_category: dict[CognitiveCategory, set[str]] = {
            c: set() for c in CognitiveCategory
        }
        self._idx_by_lifecycle: dict[NodeLifecycle, set[str]] = {l: set() for l in NodeLifecycle}
        self._idx_by_scope: dict[str, set[str]] = {}  # tenant_id → node IDs
        self._idx_by_tag: dict[str, set[str]] = {}  # tag → node IDs

        # ── Edge indexes ──
        self._edges_by_source: dict[str, set[str]] = {}  # node_id → edge IDs
        self._edges_by_target: dict[str, set[str]] = {}  # node_id → edge IDs

        # ── Incremental content digest (order-independent XOR of element digests) ──
        self._node_digests: dict[str, int] = {}
        self._edge_digests: dict[str, int] = {}
        self._content_digest: int = 0

        # ── Native Substrate Sync ──
        self._enable_native_substrate = enable_native_substrate
        self._native_adapter: Any | None = None
        if enable_native_substrate:
            try:
                from hbllm.hcir.native_adapter import NativeGraphAdapter

                self._native_adapter = NativeGraphAdapter()
            except Exception as e:
                logger.debug("CognitiveGraph: native adapter initialization skipped: %s", e)
                self._native_adapter = None

    # ── Node Operations ──────────────────────────────────────────────

    def add_node(self, node: HCIRNode, author: str = "system") -> None:
        """Add a node to the graph.  Raises ValueError on duplicate ID."""
        if node.id in self._nodes:
            raise ValueError(f"Duplicate node ID: {node.id}")
        self._nodes[node.id] = node
        self._index_node(node)
        self._track_digest(self._node_digests, "n", node)
        if self._native_adapter is not None:
            ntype = (
                node.node_type.value if hasattr(node.node_type, "value") else str(node.node_type)
            )
            lifecycle = (
                node.lifecycle.value if hasattr(node.lifecycle, "value") else str(node.lifecycle)
            )
            cat = node.category.value if hasattr(node.category, "value") else str(node.category)
            props = {"category": cat}
            if hasattr(node, "properties") and isinstance(node.properties, dict):
                props.update(node.properties)
            elif hasattr(node, "payload") and isinstance(node.payload, dict):
                props.update(node.payload)
            try:
                self._native_adapter.add_node(
                    node_id=node.id,
                    node_type=ntype,
                    lifecycle=lifecycle,
                    properties=props,
                    created_at=getattr(node, "created_at", 0.0),
                )
            except Exception as e:
                raise SubstrateDivergenceError(
                    f"Native substrate diverged on add_node for {node.id}: {e}"
                ) from e

    def upsert_node(self, node: HCIRNode, author: str = "system") -> None:
        """Add or replace a node in the graph."""
        old = self._nodes.get(node.id)
        if old is not None:
            self._deindex_node(old)
            self._untrack_digest(self._node_digests, node.id)
        self._nodes[node.id] = node
        self._index_node(node)
        self._track_digest(self._node_digests, "n", node)
        if self._native_adapter is not None:
            ntype = (
                node.node_type.value if hasattr(node.node_type, "value") else str(node.node_type)
            )
            lifecycle = (
                node.lifecycle.value if hasattr(node.lifecycle, "value") else str(node.lifecycle)
            )
            cat = node.category.value if hasattr(node.category, "value") else str(node.category)
            props = {"category": cat}
            if hasattr(node, "properties") and isinstance(node.properties, dict):
                props.update(node.properties)
            elif hasattr(node, "payload") and isinstance(node.payload, dict):
                props.update(node.payload)
            try:
                if self._native_adapter.has_node(node.id):
                    self._native_adapter.remove_node(node.id)
                self._native_adapter.add_node(
                    node_id=node.id,
                    node_type=ntype,
                    lifecycle=lifecycle,
                    properties=props,
                    created_at=getattr(node, "created_at", 0.0),
                )
            except Exception as e:
                raise SubstrateDivergenceError(
                    f"Native substrate diverged on upsert_node for {node.id}: {e}"
                ) from e

    def remove_node(self, node_id: str) -> HCIRNode | None:
        """Remove a node and all its connected edges.  Returns the removed node."""
        node = self._nodes.pop(node_id, None)
        if node is None:
            return None
        self._deindex_node(node)
        self._untrack_digest(self._node_digests, node_id)
        if self._native_adapter is not None:
            try:
                self._native_adapter.remove_node(node_id)
            except Exception as e:
                raise SubstrateDivergenceError(
                    f"Native substrate diverged on remove_node for {node_id}: {e}"
                ) from e
        # Remove edges connected to this node
        connected_edge_ids = set()
        connected_edge_ids.update(self._edges_by_source.pop(node_id, set()))
        connected_edge_ids.update(self._edges_by_target.pop(node_id, set()))
        for eid in connected_edge_ids:
            edge = self._edges.pop(eid, None)
            if edge:
                self._untrack_digest(self._edge_digests, eid)
                if self._native_adapter is not None:
                    try:
                        self._native_adapter.remove_edge(eid)
                    except Exception as e:
                        raise SubstrateDivergenceError(
                            f"Native substrate diverged on remove_edge for {eid}: {e}"
                        ) from e
                # Clean up reverse indexes for other endpoints
                for src in edge.sources:
                    if src != node_id:
                        self._edges_by_source.get(src, set()).discard(eid)
                for tgt in edge.targets:
                    if tgt != node_id:
                        self._edges_by_target.get(tgt, set()).discard(eid)
        return node

    def get_node(self, node_id: str) -> HCIRNode | None:
        """Retrieve a node by ID.  O(1)."""
        return self._nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    def all_nodes(self) -> Iterator[HCIRNode]:
        """Iterate over all nodes."""
        yield from self._nodes.values()

    # ── Edge Operations ──────────────────────────────────────────────

    def add_edge(self, edge: HCIREdge) -> None:
        """Add an edge.  Raises ValueError on duplicate ID or dangling refs."""
        if edge.id in self._edges:
            raise ValueError(f"Duplicate edge ID: {edge.id}")
        for nid in edge.sources + edge.targets:
            if nid not in self._nodes:
                raise ValueError(f"Dangling edge reference: node '{nid}' not in graph")
        self._edges[edge.id] = edge
        for src in edge.sources:
            self._edges_by_source.setdefault(src, set()).add(edge.id)
        for tgt in edge.targets:
            self._edges_by_target.setdefault(tgt, set()).add(edge.id)
        self._track_digest(self._edge_digests, "e", edge)
        if self._native_adapter is not None:
            etype = (
                edge.edge_type.value if hasattr(edge.edge_type, "value") else str(edge.edge_type)
            )
            props = {}
            if hasattr(edge, "properties") and isinstance(edge.properties, dict):
                props.update(edge.properties)
            try:
                self._native_adapter.add_edge(
                    edge_id=edge.id,
                    edge_type=etype,
                    sources=edge.sources,
                    targets=edge.targets,
                    weight=edge.weight,
                    properties=props,
                    created_at=getattr(edge, "created_at", 0.0),
                )
            except Exception as e:
                raise SubstrateDivergenceError(
                    f"Native substrate diverged on add_edge for {edge.id}: {e}"
                ) from e

    def remove_edge(self, edge_id: str) -> HCIREdge | None:
        """Remove an edge by ID."""
        edge = self._edges.pop(edge_id, None)
        if edge is None:
            return None
        self._untrack_digest(self._edge_digests, edge_id)
        if self._native_adapter is not None:
            try:
                self._native_adapter.remove_edge(edge_id)
            except Exception as e:
                raise SubstrateDivergenceError(
                    f"Native substrate diverged on remove_edge for {edge_id}: {e}"
                ) from e
                logger.debug("CognitiveGraph: native remove_edge error: %s", e)
        for src in edge.sources:
            self._edges_by_source.get(src, set()).discard(edge_id)
        for tgt in edge.targets:
            self._edges_by_target.get(tgt, set()).discard(edge_id)
        return edge

    def get_edge(self, edge_id: str) -> HCIREdge | None:
        return self._edges.get(edge_id)

    def has_edge(self, edge_id: str) -> bool:
        return edge_id in self._edges

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def all_edges(self) -> Iterator[HCIREdge]:
        yield from self._edges.values()

    def edges_from(self, node_id: str) -> list[HCIREdge]:
        """All edges where ``node_id`` is a source."""
        return [
            self._edges[eid]
            for eid in self._edges_by_source.get(node_id, set())
            if eid in self._edges
        ]

    def edges_to(self, node_id: str) -> list[HCIREdge]:
        """All edges where ``node_id`` is a target."""
        return [
            self._edges[eid]
            for eid in self._edges_by_target.get(node_id, set())
            if eid in self._edges
        ]

    def edges_between(self, node_a: str, node_b: str, directed: bool = False) -> list[HCIREdge]:
        """Find edges connecting two specific nodes.

        If directed=True, returns edges where node_a is a source and node_b is a target.
        If directed=False, returns edges connecting them in either direction.
        """
        out_a = self._edges_by_source.get(node_a, set())
        in_b = self._edges_by_target.get(node_b, set())
        matched = out_a & in_b
        if not directed:
            out_b = self._edges_by_source.get(node_b, set())
            in_a = self._edges_by_target.get(node_a, set())
            matched = matched | (out_b & in_a)
        return [self._edges[eid] for eid in matched if eid in self._edges]

    def neighbors(self, node_id: str, direction: str = "both") -> list[str]:
        """Get all directly connected node IDs.

        Args:
            node_id: Center node ID.
            direction: "out" (targets), "in" (sources), or "both" (default).
        """
        result: set[str] = set()
        if direction in ("out", "both"):
            for eid in self._edges_by_source.get(node_id, set()):
                edge = self._edges.get(eid)
                if edge:
                    result.update(edge.targets)
        if direction in ("in", "both"):
            for eid in self._edges_by_target.get(node_id, set()):
                edge = self._edges.get(eid)
                if edge:
                    result.update(edge.sources)
        result.discard(node_id)
        return sorted(result)

    def find_path(self, start_id: str, target_id: str, max_depth: int = 10) -> list[str]:
        """Find the shortest directed path of node IDs from start_id to target_id."""
        if self._native_adapter is not None:
            res = self._native_adapter.find_path(start_id, target_id, max_depth)
            if res:
                return res
        if start_id == target_id:
            return [start_id] if start_id in self._nodes else []
        if start_id not in self._nodes or target_id not in self._nodes:
            return []

        visited: set[str] = {start_id}
        queue: deque[tuple[str, list[str]]] = deque([(start_id, [start_id])])

        while queue:
            curr, path = queue.popleft()
            if len(path) > max_depth:
                break
            for edge in self.edges_from(curr):
                for tgt in edge.targets:
                    if tgt == target_id:
                        return path + [tgt]
                    if tgt not in visited and tgt in self._nodes:
                        visited.add(tgt)
                        queue.append((tgt, path + [tgt]))
        return []

    def reachable_nodes(self, start_id: str, max_depth: int = 10) -> set[str]:
        """Compute the set of forward reachable node IDs from start_id within max_depth."""
        if self._native_adapter is not None:
            return self._native_adapter.reachable_nodes(start_id, max_depth)
        if start_id not in self._nodes:
            return set()

        visited: set[str] = {start_id}
        queue: deque[tuple[str, int]] = deque([(start_id, 0)])

        while queue:
            curr, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for edge in self.edges_from(curr):
                for tgt in edge.targets:
                    if tgt not in visited and tgt in self._nodes:
                        visited.add(tgt)
                        queue.append((tgt, depth + 1))
        return visited

    def parallel_reachability_cones(
        self, seed_nodes: list[str], max_depth: int = 10
    ) -> dict[str, set[str]]:
        """Compute forward reachability cones across multiple seed nodes concurrently."""
        if self._native_adapter is not None:
            return self._native_adapter.parallel_reachability_cones(seed_nodes, max_depth)
        return {s: self.reachable_nodes(s, max_depth) for s in seed_nodes}

    def connected_components(self) -> list[set[str]]:
        """Decompose active graph into disjoint undirected connected components."""
        if self._native_adapter is not None:
            return self._native_adapter.connected_components()
        return algorithms.connected_components(self)

    def topological_sort(self) -> list[str]:
        """Compute topological ordering of DAG nodes using Kahn's algorithm."""
        if self._native_adapter is not None:
            return self._native_adapter.topological_sort()
        return algorithms.topological_sort(self)

    def transitive_reduction(self) -> CognitiveGraph:
        """Compute the transitive reduction of the cognitive graph DAG."""
        surviving = algorithms.transitive_reduction(self)
        reduced_graph = CognitiveGraph(enable_native_substrate=False)
        for node in self.all_nodes():
            reduced_graph.add_node(copy.deepcopy(node))
        for edge in self.all_edges():
            keep = False
            for s in edge.sources:
                for t in edge.targets:
                    if (s, t) in surviving:
                        keep = True
                        break
                if keep:
                    break
            if keep:
                reduced_graph.add_edge(copy.deepcopy(edge))
        return reduced_graph

    def minimum_spanning_tree(self) -> tuple[CognitiveGraph, float]:
        """Compute the Minimum Spanning Tree (MST) across cognitive graph vertices using Kruskal's algorithm."""
        mst_edges, total_weight = algorithms.minimum_spanning_tree(self)
        mst_graph = CognitiveGraph(enable_native_substrate=False)
        for node in self.all_nodes():
            mst_graph.add_node(copy.deepcopy(node))
        for edge in mst_edges:
            mst_graph.add_edge(copy.deepcopy(edge))
        return mst_graph, total_weight

    def all_pairs_shortest_paths(
        self,
        weight_attribute: str = "weight",
    ) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], list[str]]]:
        """Compute all-pairs shortest paths using Floyd-Warshall or repeated Dijkstra."""
        if self._native_adapter is not None:
            return self._native_adapter.all_pairs_shortest_paths(weight_attribute=weight_attribute)
        return algorithms.all_pairs_shortest_paths(self, weight_attr=weight_attribute)

    def graph_diameter(self, weight_attribute: str = "weight") -> float:
        """Compute the diameter (maximum shortest path distance) of the graph."""
        return self.graph_radius_and_diameter(weight_attribute=weight_attribute)[1]

    def closeness_centrality(self, node_id: str, weight_attribute: str = "weight") -> float:
        """Compute normalized closeness centrality for a node."""
        return algorithms.closeness_centrality(self, node_id, weight_attr=weight_attribute)

    def find_articulation_points_and_bridges(
        self,
    ) -> tuple[set[str], list[tuple[str, str]]]:
        """Identify critical articulation points and bridges (Hopcroft-Tarjan)."""
        if self._native_adapter is not None:
            return self._native_adapter.find_articulation_points_and_bridges()
        return algorithms.articulation_points_and_bridges(self)

    def is_biconnected(self) -> bool:
        """Verify if the graph is biconnected (2-vertex-connected)."""
        if self._native_adapter is not None:
            return self._native_adapter.is_biconnected()
        return algorithms.is_biconnected(self)

    def max_flow_min_cut(
        self,
        source: str,
        sink: str,
        capacity_attribute: str = "capacity",
        default_capacity: float = 1.0,
    ) -> tuple[float, list[tuple[str, str]], set[str], set[str]]:
        """Compute the maximum flow and minimum s-t cut using Edmonds-Karp algorithm."""
        if self._native_adapter is not None:
            return self._native_adapter.max_flow_min_cut(
                source, sink, capacity_attribute=capacity_attribute
            )
        max_flow, _ = algorithms.edmonds_karp_max_flow(
            self, source, sink, capacity_attr=capacity_attribute
        )
        s_cut, t_cut, _ = algorithms.min_st_cut(
            self, source, sink, capacity_attr=capacity_attribute
        )
        cut_edges: list[tuple[str, str]] = []
        for edge in self.all_edges():
            for u in edge.sources:
                if u in s_cut:
                    for v in edge.targets:
                        if v in t_cut:
                            cut_edges.append((u, v))
        return max_flow, cut_edges, s_cut, t_cut

    def feedback_arc_set(
        self,
    ) -> tuple[list[tuple[str, str]], list[str]]:
        """Compute approximate minimum feedback arc set to break all directed cycles."""
        if self._native_adapter is not None:
            return self._native_adapter.feedback_arc_set()
        return algorithms.feedback_arc_set(self)

    def strongly_connected_components(
        self,
    ) -> list[set[str]]:
        """Compute Strongly Connected Components (SCC) of the directed graph using Tarjan's algorithm."""
        if self._native_adapter is not None:
            return self._native_adapter.strongly_connected_components()
        return algorithms.strongly_connected_components(self)

    def condense_scc_metagraph(
        self,
    ) -> tuple[dict[int, set[str]], list[tuple[int, int]]]:
        """Condense directed graph SCCs into an acyclic metagraph (DAG)."""
        if self._native_adapter is not None:
            return self._native_adapter.condense_scc_metagraph()
        return algorithms.condense_scc_metagraph(self)

    def find_eulerian_path_or_circuit(
        self,
    ) -> tuple[bool, list[tuple[str, str]]]:
        """Find an Eulerian path or circuit traversing every directed edge exactly once (Hierholzer)."""
        if self._native_adapter is not None:
            return self._native_adapter.find_eulerian_path_or_circuit()
        return algorithms.find_eulerian_path_or_circuit(self)

    def chinese_postman_tour(
        self,
        weight_attribute: str = "weight",
    ) -> tuple[float, list[tuple[str, str]]]:
        """Compute minimum-weight edge-covering tour traversing every edge at least once (CPP)."""
        if self._native_adapter is not None:
            return self._native_adapter.chinese_postman_tour()
        return algorithms.chinese_postman_tour(self, weight_attr=weight_attribute)

    def minimum_dominating_set(
        self,
    ) -> set[str]:
        """Compute an approximate Minimum Dominating Set (MDS) using greedy coverage heuristic."""
        if self._native_adapter is not None:
            return self._native_adapter.minimum_dominating_set()
        return algorithms.minimum_dominating_set(self)

    def maximum_independent_set(
        self,
    ) -> set[str]:
        """Compute an approximate Maximum Independent Set (MIS) using minimum-degree greedy heuristic."""
        if self._native_adapter is not None:
            return self._native_adapter.maximum_independent_set()
        return algorithms.maximum_independent_set(self)

    def minimum_vertex_cover(
        self,
    ) -> set[str]:
        """Compute an approximate Minimum Vertex Cover (MVC) using greedy coverage and Gallai duality."""
        if self._native_adapter is not None:
            return self._native_adapter.minimum_vertex_cover()
        return algorithms.minimum_vertex_cover(self)

    def greedy_edge_coloring(
        self,
    ) -> dict[tuple[str, str], int]:
        """Compute a proper greedy edge coloring (Vizing's theorem) of undirected edges."""
        if self._native_adapter is not None:
            return self._native_adapter.greedy_edge_coloring()
        return algorithms.greedy_edge_coloring(self)

    def feedback_vertex_set(
        self,
    ) -> set[str]:
        """Compute an approximate minimal Feedback Vertex Set (FVS) to break all directed cycles."""
        if self._native_adapter is not None:
            return self._native_adapter.feedback_vertex_set()
        return algorithms.feedback_vertex_set(self)

    def essential_causal_skeleton(
        self,
    ) -> set[tuple[str, str]]:
        """Compute the irreducible causal skeleton DAG via transitive reduction."""
        if self._native_adapter is not None:
            return self._native_adapter.essential_causal_skeleton()
        return algorithms.essential_causal_skeleton(self)

    def graph_radius_and_diameter(
        self,
        weight_attribute: str = "weight",
    ) -> tuple[float, float]:
        """Compute the radius and diameter of the graph based on node eccentricities."""
        if self._native_adapter is not None:
            return self._native_adapter.graph_radius_and_diameter()
        return algorithms.graph_radius_and_diameter(self, weight_attr=weight_attribute)

    def betweenness_centrality(
        self,
        normalized: bool = True,
        directed: bool = False,
    ) -> dict[str, float]:
        """Compute Brandes' betweenness centrality for all vertices in the graph."""
        if self._native_adapter is not None:
            return self._native_adapter.betweenness_centrality(
                normalized=normalized, directed=directed
            )
        return algorithms.betweenness_centrality(self, normalized=normalized, directed=directed)

    def nodes_by_type(self, node_type: HCIRNodeType) -> list[HCIRNode]:
        """O(k) where k is the number of nodes of that type."""
        return [
            self._nodes[nid]
            for nid in self._idx_by_type.get(node_type, set())
            if nid in self._nodes
        ]

    def nodes_by_category(self, category: CognitiveCategory) -> list[HCIRNode]:
        return [
            self._nodes[nid]
            for nid in self._idx_by_category.get(category, set())
            if nid in self._nodes
        ]

    def nodes_by_lifecycle(self, lifecycle: NodeLifecycle) -> list[HCIRNode]:
        return [
            self._nodes[nid]
            for nid in self._idx_by_lifecycle.get(lifecycle, set())
            if nid in self._nodes
        ]

    def nodes_by_scope(self, tenant_id: str) -> list[HCIRNode]:
        return [
            self._nodes[nid]
            for nid in self._idx_by_scope.get(tenant_id, set())
            if nid in self._nodes
        ]

    def nodes_by_tag(self, tag: str) -> list[HCIRNode]:
        return [self._nodes[nid] for nid in self._idx_by_tag.get(tag, set()) if nid in self._nodes]

    # ── Views ────────────────────────────────────────────────────────

    def knowledge_view(self) -> list[HCIRNode]:
        """Facts, Beliefs, Concepts, Procedures — slowly changing knowledge."""
        knowledge_types = {
            HCIRNodeType.FACT,
            HCIRNodeType.BELIEF,
            HCIRNodeType.CONCEPT,
            HCIRNodeType.PROCEDURE,
            HCIRNodeType.EXTERNAL_KNOWLEDGE,
        }
        result: list[HCIRNode] = []
        for t in knowledge_types:
            result.extend(self.nodes_by_type(t))
        return result

    def execution_view(self) -> list[HCIRNode]:
        """Goals, Actions, Resources — rapidly changing execution state."""
        exec_types = {
            HCIRNodeType.GOAL,
            HCIRNodeType.INTENT,
            HCIRNodeType.ACTION,
            HCIRNodeType.RESOURCE,
            HCIRNodeType.CONSTRAINT,
        }
        result: list[HCIRNode] = []
        for t in exec_types:
            result.extend(self.nodes_by_type(t))
        return result

    def memory_view(self) -> list[HCIRNode]:
        """Episodes, Skills, Values — episodic and procedural memory."""
        mem_types = {
            HCIRNodeType.EPISODE,
            HCIRNodeType.SKILL,
            HCIRNodeType.VALUE,
        }
        result: list[HCIRNode] = []
        for t in mem_types:
            result.extend(self.nodes_by_type(t))
        return result

    def discovery_view(self) -> list[HCIRNode]:
        """Hypotheses, Evidence, Experiments, Claims, Contradictions, Unknowns — epistemic state."""
        discovery_types = {
            HCIRNodeType.HYPOTHESIS,
            HCIRNodeType.EVIDENCE,
            HCIRNodeType.EXPERIMENT,
            HCIRNodeType.CLAIM,
            HCIRNodeType.CONTRADICTION,
            HCIRNodeType.UNKNOWN,
            HCIRNodeType.RESEARCH_PROGRAM,
            HCIRNodeType.PREDICTION,
            HCIRNodeType.PREDICTION_ERROR,
        }
        result: list[HCIRNode] = []
        for t in discovery_types:
            result.extend(self.nodes_by_type(t))
        return result

    # ── Internal Indexing ────────────────────────────────────────────

    @staticmethod
    def _element_digest(prefix: str, model: BaseModel) -> int:
        try:
            payload = model.model_dump_json()
        except Exception:
            payload = json.dumps(model.model_dump(), sort_keys=True, default=str)
        return int.from_bytes(
            hashlib.blake2b(f"{prefix}:{payload}".encode(), digest_size=16).digest(), "big"
        )

    def _track_digest(self, table: dict[str, int], prefix: str, model: BaseModel) -> None:
        d = self._element_digest(prefix, model)
        table[model.id] = d  # type: ignore[attr-defined]
        self._content_digest ^= d

    def _untrack_digest(self, table: dict[str, int], element_id: str) -> None:
        d = table.pop(element_id, None)
        if d is not None:
            self._content_digest ^= d

    def content_hash(self) -> str:
        """O(1) payload-aware content hash maintained incrementally on every mutation.

        Unlike ``canonical_state_hash`` (structural: IDs only), this digest changes
        whenever any node or edge payload changes, making it suitable for
        event-sourced snapshot equality checks.
        """
        return f"{self._content_digest:032x}{len(self._nodes):08x}{len(self._edges):08x}"

    def _index_node(self, node: HCIRNode) -> None:
        self._idx_by_type[node.node_type].add(node.id)
        self._idx_by_category[node.category].add(node.id)
        self._idx_by_lifecycle[node.lifecycle].add(node.id)
        self._idx_by_scope.setdefault(node.scope.tenant_id, set()).add(node.id)
        for tag in node.tags:
            self._idx_by_tag.setdefault(tag, set()).add(node.id)

    def _deindex_node(self, node: HCIRNode) -> None:
        self._idx_by_type[node.node_type].discard(node.id)
        self._idx_by_category[node.category].discard(node.id)
        self._idx_by_lifecycle[node.lifecycle].discard(node.id)
        scope_set = self._idx_by_scope.get(node.scope.tenant_id)
        if scope_set:
            scope_set.discard(node.id)
        for tag in node.tags:
            tag_set = self._idx_by_tag.get(tag)
            if tag_set:
                tag_set.discard(node.id)

    # ── Bulk & Graph Utilities ───────────────────────────────────────

    def node_count_by_type(self, include_empty: bool = False) -> dict[HCIRNodeType, int]:
        """Summary stats of node counts grouped by node type."""
        if include_empty:
            return {t: len(self._idx_by_type.get(t, set())) for t in HCIRNodeType}
        return {t: len(ids) for t, ids in self._idx_by_type.items() if ids}

    def clear(self) -> None:
        """Reset the graph to an empty state."""
        self._nodes.clear()
        self._edges.clear()
        for s in self._idx_by_type.values():
            s.clear()
        for s in self._idx_by_category.values():
            s.clear()
        for s in self._idx_by_lifecycle.values():
            s.clear()
        self._idx_by_scope.clear()
        self._idx_by_tag.clear()
        self._edges_by_source.clear()
        self._edges_by_target.clear()
        self._node_digests.clear()
        self._edge_digests.clear()
        self._content_digest = 0
        if self._native_adapter is not None:
            try:
                from hbllm.hcir.native_adapter import NativeGraphAdapter

                self._native_adapter = NativeGraphAdapter()
            except Exception:
                self._native_adapter = None

    def find_shortest_path(self, source_id: str, target_id: str) -> list[str] | None:
        """Unbounded-depth shortest directed path; ``None`` when unreachable.

        Thin wrapper over :meth:`find_path` (single BFS implementation) with the
        Optional return convention used by the planners.
        """
        path = self.find_path(source_id, target_id, max_depth=max(1, len(self._nodes)))
        return path or None

    def extract_subgraph_by_depth(self, seed_id: str, max_depth: int = 2) -> CognitiveGraph:
        """Extract a depth-limited ego subgraph centered on seed_id using the native substrate."""
        if self._native_adapter is not None:
            res = self._native_adapter.extract_subgraph(seed_id, max_depth)
            if isinstance(res, tuple):
                nodes, _ = res
                node_ids = {n["id"] for n in nodes}
            elif isinstance(res, dict):
                node_ids = {n["id"] for n in res.get("nodes", [])}
            else:
                node_ids = set()
            return self.subgraph(node_ids)
        node_ids = {seed_id}
        curr_layer = {seed_id}
        for _ in range(max_depth):
            next_layer: set[str] = set()
            for nid in curr_layer:
                for eid in self._edges_by_source.get(nid, set()):
                    e = self._edges.get(eid)
                    if e:
                        next_layer.update(e.targets)
            next_layer.difference_update(node_ids)
            if not next_layer:
                break
            node_ids.update(next_layer)
            curr_layer = next_layer
        return self.subgraph(node_ids)

    def canonical_state_hash(self) -> str:
        """Deterministic canonical structural state hash."""
        if self._native_adapter is not None:
            return self._native_adapter.canonical_hash()
        import hashlib

        h = hashlib.sha256()
        for nid in sorted(self._nodes.keys()):
            h.update(nid.encode("utf-8"))
        for eid in sorted(self._edges.keys()):
            h.update(eid.encode("utf-8"))
        return h.hexdigest()[:16]

    def native_snapshot(self) -> Any:
        """Create O(1) chunked native snapshot via structural sharing."""
        if self._native_adapter is not None:
            return self._native_adapter.snapshot()
        return None

    def get_native_adapter(self) -> Any:
        """Return the underlying NativeGraphAdapter instance."""
        return self._native_adapter

    @property
    def is_native_substrate_active(self) -> bool:
        """Whether native Rust graph acceleration is compiled and active."""
        return self._native_adapter is not None and getattr(
            self._native_adapter, "is_native", False
        )

    def fork(self) -> CognitiveGraph:
        """Create a structural-sharing copy-on-write fork of this graph in < 1 ms (WP4)."""
        forked = CognitiveGraph(enable_native_substrate=False)
        forked._enable_native_substrate = self._enable_native_substrate
        forked._nodes = dict(self._nodes)
        forked._edges = dict(self._edges)
        forked._edges_by_source = {k: set(v) for k, v in self._edges_by_source.items()}
        forked._edges_by_target = {k: set(v) for k, v in self._edges_by_target.items()}
        forked._idx_by_type = {k: set(v) for k, v in self._idx_by_type.items()}
        forked._idx_by_category = {k: set(v) for k, v in self._idx_by_category.items()}
        forked._node_digests = dict(self._node_digests)
        forked._edge_digests = dict(self._edge_digests)
        if self._native_adapter is not None:
            try:
                forked._native_adapter = self._native_adapter.snapshot()
            except Exception:
                forked._native_adapter = None
        return forked

    def subgraph(self, node_ids: set[str] | list[str] | Iterator[str]) -> CognitiveGraph:
        """Extract an induced subgraph containing only specified nodes and edges between them."""
        sub = CognitiveGraph()
        target_ids = set(node_ids)
        for nid in target_ids:
            node = self._nodes.get(nid)
            if node is not None:
                sub.add_node(node)
        for edge in self._edges.values():
            if all(s in target_ids for s in edge.sources) and all(
                t in target_ids for t in edge.targets
            ):
                sub.add_edge(edge)
        return sub

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire graph into a dictionary."""
        return {
            "nodes": [node.model_dump() for node in self._nodes.values()],
            "edges": [edge.model_dump() for edge in self._edges.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CognitiveGraph:
        """Reconstruct a CognitiveGraph from serialized dictionary data."""
        graph = cls()
        for ndata in data.get("nodes", []):
            ntype_raw = ndata.get("node_type")
            node_cls: type[HCIRNode] = HCIRNode
            if ntype_raw:
                try:
                    ntype = HCIRNodeType(ntype_raw)
                    node_cls = NODE_TYPE_REGISTRY.get(ntype, HCIRNode)
                except ValueError:
                    pass
            node = node_cls.model_validate(ndata)
            graph.add_node(node)
        for edata in data.get("edges", []):
            edge = HCIREdge.model_validate(edata)
            graph.add_edge(edge)
        return graph
