"""Constraint-Aware Relational Graph Matcher."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity

logger = logging.getLogger(__name__)


@dataclass
class RelationalNode:
    """Attributed spatial entity in relational scene graph."""

    node_id: str
    feature_id: int = 0
    area: int = 0
    centroid: tuple[float, float] = (0.0, 0.0)
    bounding_box: tuple[int, int, int, int] = (0, 0, 0, 0)
    role: EntityRole = EntityRole.UNKNOWN
    properties: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        node_id: str,
        feature_id: int = 0,
        area: int = 0,
        centroid: tuple[float, float] = (0.0, 0.0),
        bounding_box: tuple[int, int, int, int] = (0, 0, 0, 0),
        role: EntityRole = EntityRole.UNKNOWN,
        properties: dict[str, Any] | None = None,
        color: int | None = None,
    ) -> None:
        self.node_id = node_id
        self.feature_id = color if color is not None else feature_id
        self.area = area
        self.centroid = centroid
        self.bounding_box = bounding_box
        self.role = role
        self.properties = properties or {}

    @property
    def color(self) -> int:
        """Backward-compatibility alias for feature_id."""
        return self.feature_id

    @classmethod
    def from_spatial_entity(cls, entity: SpatialEntity) -> RelationalNode:
        """Lift an HCIR SpatialEntity component into a RelationalNode."""
        min_r, max_r, min_c, max_c = entity.bounding_box
        return cls(
            node_id=entity.id,
            feature_id=entity.feature_id if entity.feature_id is not None else 0,
            area=entity.area,
            centroid=entity.centroid,
            bounding_box=(min_r, min_c, max_r, max_c),
            role=entity.role,
            properties=dict(entity.properties),
        )


@dataclass
class RelationalEdge:
    """Attributed relation between two entities."""

    source_id: str
    target_id: str
    relation: str  # "adjacent", "contains", "left_of", "right_of", "above", "below"


class RelationalGraphMatcher:
    """Constraint-aware attributed subgraph matcher with exponential worst-case bounding (W038).

    Addresses NP-complete subgraph isomorphism by:
    1. Pruning domain candidates using semantic compatibility (feature_id, role, area, degree).
    2. Enforcing structural relational consistency along attributed edges.
    3. Enforcing a hard search step budget (max_steps) with backtracking to prevent freeze.
    """

    def __init__(self, max_steps: int = 5000) -> None:
        self.max_steps = max_steps

    def find_mapping(
        self,
        source_nodes: Sequence[RelationalNode],
        source_edges: Sequence[RelationalEdge],
        target_nodes: Sequence[RelationalNode],
        target_edges: Sequence[RelationalEdge],
    ) -> dict[str, str] | None:
        """Find bijective mapping f: source_id -> target_id satisfying all constraints."""
        if len(source_nodes) > len(target_nodes):
            return None

        # Build edge relation lookup tables: (u, v) -> set(relations)
        src_adj: dict[tuple[str, str], set[str]] = {}
        for e in source_edges:
            src_adj.setdefault((e.source_id, e.target_id), set()).add(e.relation)

        tgt_adj: dict[tuple[str, str], set[str]] = {}
        for e in target_edges:
            tgt_adj.setdefault((e.source_id, e.target_id), set()).add(e.relation)

        # 1. Candidate domains: for each u in source, find compatible v in target
        domains: dict[str, list[str]] = {}
        for u in source_nodes:
            candidates: list[str] = []
            for v in target_nodes:
                # Feature constraint
                if u.feature_id != v.feature_id:
                    continue
                # Role constraint
                if (
                    u.role != EntityRole.UNKNOWN
                    and v.role != EntityRole.UNKNOWN
                    and u.role != v.role
                ):
                    continue
                # Area constraint (within 20% or equal)
                if abs(u.area - v.area) > max(1, 0.2 * u.area):
                    continue
                candidates.append(v.node_id)
            if not candidates:
                return None
            domains[u.node_id] = candidates

        # Sort source nodes by MRV (Minimum Remaining Values heuristic)
        sorted_u_ids = sorted(domains.keys(), key=lambda uid: len(domains[uid]))

        mapping: dict[str, str] = {}
        used_target_ids: set[str] = set()
        steps = [0]

        def backtrack(idx: int) -> bool:
            if idx == len(sorted_u_ids):
                return True
            steps[0] += 1
            if steps[0] > self.max_steps:
                logger.warning(
                    "RelationalGraphMatcher: search budget (%d steps) exceeded", self.max_steps
                )
                return False

            u = sorted_u_ids[idx]
            for v in domains[u]:
                if v in used_target_ids:
                    continue

                # Relational consistency check with already assigned neighbors
                consistent = True
                for prev_u, prev_v in mapping.items():
                    # Check edge (u, prev_u)
                    if (u, prev_u) in src_adj:
                        required_rels = src_adj[(u, prev_u)]
                        if (v, prev_v) not in tgt_adj or not required_rels.issubset(
                            tgt_adj[(v, prev_v)]
                        ):
                            consistent = False
                            break
                    # Check edge (prev_u, u)
                    if (prev_u, u) in src_adj:
                        required_rels = src_adj[(prev_u, u)]
                        if (prev_v, v) not in tgt_adj or not required_rels.issubset(
                            tgt_adj[(prev_v, v)]
                        ):
                            consistent = False
                            break

                if not consistent:
                    continue

                mapping[u] = v
                used_target_ids.add(v)

                if backtrack(idx + 1):
                    return True

                del mapping[u]
                used_target_ids.remove(v)

            return False

        success = backtrack(0)
        return mapping if success else None
