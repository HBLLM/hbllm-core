"""Object Models, Enums, and Option Specifications."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

import numpy as np

try:
    from scipy.ndimage import label as _scipy_label
except ImportError:  # pragma: no cover

    def _scipy_label(input_arr: np.ndarray, structure: Any = None) -> tuple[np.ndarray, int]:
        """4-connectivity connected-component labeling fallback when scipy is not installed."""
        h, w = input_arr.shape
        labeled = np.zeros((h, w), dtype=np.int32)
        current_label = 0
        visited = np.zeros((h, w), dtype=bool)
        for r in range(h):
            for c in range(w):
                if input_arr[r, c] and not visited[r, c]:
                    current_label += 1
                    q = [(r, c)]
                    visited[r, c] = True
                    labeled[r, c] = current_label
                    while q:
                        cr, cc = q.pop()
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nr, nc = cr + dr, cc + dc
                            if (
                                0 <= nr < h
                                and 0 <= nc < w
                                and input_arr[nr, nc]
                                and not visited[nr, nc]
                            ):
                                visited[nr, nc] = True
                                labeled[nr, nc] = current_label
                                q.append((nr, nc))
        return labeled, current_label


from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


class ExecutiveDirective(IntEnum):
    """The 5 executive directives guiding object-centric option selection."""

    CHECK_EXIT_OPEN = 1
    GO_TO_EXIT = 2
    EXPLORE_OBJECTS = 3
    USE_UNLOCKING_OBJECT = 4
    INTERACT_UNKNOWN_OBJECT = 5


class ObjectCategory(StrEnum):
    """Semantic category for lifted salient objects."""

    EXIT = "EXIT"
    KEY = "KEY"
    SWITCH = "SWITCH"
    BARRIER = "BARRIER"
    CONTAINER = "CONTAINER"
    MANIPULABLE = "MANIPULABLE"
    UNKNOWN = "UNKNOWN"


@dataclass
class SalientObject:
    """A distinct interactive object in the lifted object-centric graph."""

    object_id: str
    category: ObjectCategory
    feature_id: int
    centroid: tuple[int, int]
    cells: set[tuple[int, int]]
    area: int
    is_solid: bool = False
    interaction_stances: list[tuple[int, int]] = field(default_factory=list)
    interaction_action: int | None = None
    anchor_coord: tuple[int, int] = (0, 0)
    salience_score: float = 0.0
    properties: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_spatial_entity(
        cls,
        entity: SpatialEntity,
        category: ObjectCategory = ObjectCategory.UNKNOWN,
    ) -> SalientObject:
        """Construct a SalientObject directly from an HCIR SpatialEntity component."""
        cells = set(entity.properties.get("cells", [entity.grid_pos]))
        centroid = (int(round(entity.centroid[0])), int(round(entity.centroid[1])))
        anchor = (
            min(cells, key=lambda p: (p[0] - centroid[0]) ** 2 + (p[1] - centroid[1]) ** 2)
            if cells
            else centroid
        )
        salience = float(
            entity.properties.get(
                "salience_score",
                100.0 / math.log2(2 + max(1, entity.area)),
            )
        )
        return cls(
            object_id=entity.id,
            category=category,
            feature_id=entity.feature_id if entity.feature_id is not None else 0,
            centroid=centroid,
            cells=cells,
            area=entity.area,
            is_solid=(entity.role == EntityRole.OBSTACLE),
            interaction_stances=[entity.grid_pos],
            anchor_coord=anchor,
            salience_score=salience,
            properties={"role": str(entity.role), **dict(entity.properties)},
        )

    def to_spatial_entity(self, role: EntityRole | None = None) -> SpatialEntity:
        """Convert salient object back into a formal HCIR SpatialEntity."""
        eff_role = role or (
            EntityRole.GOAL
            if self.category == ObjectCategory.EXIT
            else EntityRole.MANIPULABLE
            if self.category in (ObjectCategory.KEY, ObjectCategory.MANIPULABLE)
            else EntityRole.OBSTACLE
            if self.is_solid
            else EntityRole.UNKNOWN
        )
        min_r = min(r for r, c in self.cells) if self.cells else self.centroid[0]
        max_r = max(r for r, c in self.cells) if self.cells else self.centroid[0]
        min_c = min(c for r, c in self.cells) if self.cells else self.centroid[1]
        max_c = max(c for r, c in self.cells) if self.cells else self.centroid[1]
        return SpatialEntity(
            id=self.object_id,
            role=eff_role,
            centroid=(float(self.centroid[0]), float(self.centroid[1])),
            grid_pos=self.interaction_stances[0] if self.interaction_stances else self.centroid,
            area=self.area,
            bounding_box=(min_r, max_r, min_c, max_c),
            feature_id=self.feature_id,
            properties=dict(self.properties),
        )

    def to_causal_predicate(self) -> CausalPredicate:
        """Export object identity as a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable=f"object_category::{self.category.value}",
            operator="==",
            value=self.feature_id,
            target_entity=self.object_id,
        )


@dataclass
class MacroOption:
    """A high-level macro-action representing an option to interact with a salient object."""

    option_id: str
    directive: ExecutiveDirective | int
    target_object: SalientObject | None
    target_stance: tuple[int, int]
    primitive_steps: list[Any]  # list[MentalSimulationStep]
    score: float = 0.0
    description: str = ""

    def to_subgoal(self) -> dict[str, Any]:
        """Convert macro-option into a structured declarative subgoal dictionary."""
        return {
            "option_id": self.option_id,
            "directive": int(self.directive),
            "target_object_id": self.target_object.object_id if self.target_object else None,
            "target_stance": self.target_stance,
            "step_count": len(self.primitive_steps),
            "score": self.score,
            "description": self.description,
        }
