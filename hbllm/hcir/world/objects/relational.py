"""Constraint-Aware Relational Graph Matcher."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.spatial_planner import EntityRole


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
