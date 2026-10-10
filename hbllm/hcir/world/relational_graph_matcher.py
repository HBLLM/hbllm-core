"""Relational Structure and Object Graph Matching Engine (W038).

Implements structural correspondence between objects and relations across examples:
1. Object-Relational Scene Graph Construction:
   - Lifted object nodes with spatial, geometric, and topological attributes.
   - Directed relational edges: spatial displacement, containment, adjacency, collinearity.
2. Relational Invariant Signatures:
   - Position-invariant, color-invariant, and size-relative node representations.
   - Topological degree and relation histograms.
3. Graph Correspondence & Isomorphism Matching:
   - Resolves optimal object-to-object mapping across scene variations.
   - Handles transformations with differing positions, recoloring, or scaling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from scipy.optimize import linear_sum_assignment

from hbllm.hcir.world.inferotemporal_segmentation import (
    InferotemporalSegmentationEngine,
    VentralObjectToken,
)

logger = logging.getLogger(__name__)


class SpatialRelation(StrEnum):
    """Categorical spatial and topological relations between object pairs."""

    CONTAINS = "CONTAINS"
    INSIDE = "INSIDE"
    ADJACENT = "ADJACENT"
    ABOVE = "ABOVE"
    BELOW = "BELOW"
    LEFT_OF = "LEFT_OF"
    RIGHT_OF = "RIGHT_OF"
    COLINEAR_H = "COLINEAR_H"
    COLINEAR_V = "COLINEAR_V"
    DISJOINT = "DISJOINT"


@dataclass
class RelationalEdge:
    """Directed edge encoding the spatial-structural relationship between two objects."""

    source_id: int
    target_id: int
    relation: SpatialRelation
    distance: float
    dx: float  # row displacement
    dy: float  # col displacement


@dataclass
class ObjectGraph:
    """Scene graph representing discrete objects and relational topology."""

    nodes: dict[int, VentralObjectToken] = field(default_factory=dict)
    edges: list[RelationalEdge] = field(default_factory=list)
    adjacency: dict[int, list[RelationalEdge]] = field(default_factory=dict)

    def add_node(self, token: VentralObjectToken) -> None:
        self.nodes[token.object_id] = token
        if token.object_id not in self.adjacency:
            self.adjacency[token.object_id] = []

    def add_edge(self, edge: RelationalEdge) -> None:
        self.edges.append(edge)
        self.adjacency[edge.source_id].append(edge)


@dataclass
class ObjectCorrespondence:
    """Mapped correspondence between an object in scene A and scene B."""

    src_object_id: int
    dst_object_id: int
    match_score: float
    matched_relations: int
    invariants_preserved: list[str] = field(default_factory=list)


class RelationalGraphMatcher:
    """Constructs scene graphs and computes structural correspondences (W038)."""

    @classmethod
    def build_scene_graph(
        cls,
        grid: np.ndarray,
        bg_color: int = 0,
        connectivity: int = 4,
    ) -> ObjectGraph:
        """Extract objects and compute all pair-wise relational edges."""
        tokens = InferotemporalSegmentationEngine.segment_objects(
            grid, background_feature=bg_color, connectivity=connectivity
        )
        graph = ObjectGraph()
        for t in tokens:
            graph.add_node(t)

        node_ids = list(graph.nodes.keys())
        for i, id_a in enumerate(node_ids):
            token_a = graph.nodes[id_a]
            ra, ca = token_a.centroid
            for id_b in node_ids[i + 1 :]:
                token_b = graph.nodes[id_b]
                rb, cb = token_b.centroid

                dr = rb - ra
                dc = cb - ca
                dist = float(np.hypot(dr, dc))

                # Compute relations from A -> B and B -> A
                rel_ab = cls._classify_relation(token_a, token_b, dr, dc)
                rel_ba = cls._classify_relation(token_b, token_a, -dr, -dc)

                graph.add_edge(
                    RelationalEdge(
                        source_id=id_a,
                        target_id=id_b,
                        relation=rel_ab,
                        distance=dist,
                        dx=dr,
                        dy=dc,
                    )
                )
                graph.add_edge(
                    RelationalEdge(
                        source_id=id_b,
                        target_id=id_a,
                        relation=rel_ba,
                        distance=dist,
                        dx=-dr,
                        dy=-dc,
                    )
                )

        return graph

    @classmethod
    def _classify_relation(
        cls,
        token_a: VentralObjectToken,
        token_b: VentralObjectToken,
        dr: float,
        dc: float,
    ) -> SpatialRelation:
        """Categorize topological and directional relationship between A and B."""
        # Containment check
        min_ra, min_ca, max_ra, max_ca = token_a.bounding_box
        min_rb, min_cb, max_rb, max_cb = token_b.bounding_box

        if min_ra <= min_rb and max_ra >= max_rb and min_ca <= min_cb and max_ca >= max_cb:
            return SpatialRelation.CONTAINS
        if min_rb <= min_ra and max_rb >= max_ra and min_cb <= min_ca and max_cb >= max_ca:
            return SpatialRelation.INSIDE

        # Adjacency check (distance between bounding boxes <= 1)
        dist_r = max(0, max(min_ra - max_rb, min_rb - max_ra))
        dist_c = max(0, max(min_ca - max_cb, min_cb - max_ca))
        if dist_r <= 1 and dist_c <= 1:
            return SpatialRelation.ADJACENT

        # Collinearity check
        if abs(dr) < 0.5:
            return SpatialRelation.COLINEAR_H
        if abs(dc) < 0.5:
            return SpatialRelation.COLINEAR_V

        # Dominant direction
        if abs(dr) >= abs(dc):
            return SpatialRelation.BELOW if dr > 0 else SpatialRelation.ABOVE
        return SpatialRelation.RIGHT_OF if dc > 0 else SpatialRelation.LEFT_OF

    @classmethod
    def match_graphs(
        cls,
        graph_a: ObjectGraph,
        graph_b: ObjectGraph,
        ignore_color: bool = True,
        ignore_scale: bool = True,
    ) -> list[ObjectCorrespondence]:
        """Find optimal bijective structural correspondence between nodes of graph_a and graph_b.

        Uses relational feature compatibility and bipartite Hungarian assignment.
        """
        nodes_a = list(graph_a.nodes.values())
        nodes_b = list(graph_b.nodes.values())

        if not nodes_a or not nodes_b:
            return []

        cost_matrix = np.zeros((len(nodes_a), len(nodes_b)), dtype=float)

        for i, na in enumerate(nodes_a):
            for j, nb in enumerate(nodes_b):
                cost = 0.0

                # 1. Color matching cost
                if not ignore_color:
                    if na.feature_id != nb.feature_id:
                        cost += 2.0

                # 2. Geometric aspect ratio and compactness cost
                cost += abs(na.aspect_ratio - nb.aspect_ratio) * 1.5
                if na.is_compact != nb.is_compact:
                    cost += 1.0

                # 3. Normalized area ratio cost
                if not ignore_scale:
                    area_diff = abs(na.area - nb.area) / max(na.area, nb.area, 1)
                    cost += area_diff * 2.0

                # 4. Relational structural degree compatibility
                deg_a = len(graph_a.adjacency.get(na.object_id, []))
                deg_b = len(graph_b.adjacency.get(nb.object_id, []))
                cost += abs(deg_a - deg_b) * 0.5

                # 5. Relational profile matching
                rels_a = {e.relation for e in graph_a.adjacency.get(na.object_id, [])}
                rels_b = {e.relation for e in graph_b.adjacency.get(nb.object_id, [])}
                jaccard = (
                    len(rels_a & rels_b) / max(len(rels_a | rels_b), 1)
                    if (rels_a or rels_b)
                    else 1.0
                )
                cost += (1.0 - jaccard) * 3.0

                cost_matrix[i, j] = cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        correspondences: list[ObjectCorrespondence] = []
        for r, c in zip(row_ind, col_ind):
            na = nodes_a[r]
            nb = nodes_b[c]
            match_cost = cost_matrix[r, c]
            match_score = float(max(0.0, 1.0 - (match_cost / 10.0)))

            invariants = []
            if na.feature_id == nb.feature_id:
                invariants.append("COLOR_PRESERVED")
            if abs(na.aspect_ratio - nb.aspect_ratio) < 0.2:
                invariants.append("SHAPE_PRESERVED")
            if na.is_compact == nb.is_compact:
                invariants.append("TOPOLOGY_PRESERVED")

            correspondences.append(
                ObjectCorrespondence(
                    src_object_id=na.object_id,
                    dst_object_id=nb.object_id,
                    match_score=match_score,
                    matched_relations=len(graph_a.adjacency.get(na.object_id, [])),
                    invariants_preserved=invariants,
                )
            )

        return correspondences
