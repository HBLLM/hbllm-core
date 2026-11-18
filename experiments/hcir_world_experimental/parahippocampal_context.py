"""Parahippocampal Cortex (PHC) Spatial Context Schema & Environmental Topology Recognition.

Biologically grounded in Parahippocampal Place Area (PPA) / PHC scene perception
(Epstein & Kanwisher, 1998; Aminoff, Gronau, & Bar, 2007; Bar, 2004):
1. Global Scene Geometry & Spatial Envelope:
   - Encodes macro-scale topological properties of the spatial environment:
     aspect ratio, boundary enclosure, obstacle density, and room/corridor connectivity.
2. Context Schema Matching & Hysteresis:
   - Matches current environmental perceptual features against archetypal scene schemas
     ("open_arena", "narrow_corridor", "labyrinth_maze", "dense_obstacle_field", "compartmentalized_rooms").
   - Employs hysteresis to prevent rapid context flickering during local movements.
3. Top-Down Heuristic Bias:
   - Emits schema-specific heuristic biases (e.g., corridor bias promotes straight-line pursuit;
     maze bias promotes dead-end pruning and wall-following).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class SpatialContextSchema:
    """An archetypal spatial scene envelope profile."""

    schema_id: str
    archetype_name: (
        str  # e.g., "open_arena", "narrow_corridor", "labyrinth_maze", "compartmentalized_rooms"
    )
    aspect_ratio: float
    obstacle_density: float
    boundary_enclosure: float  # [0.0, 1.0] degree of surrounding perimeter walls
    estimated_corridor_ratio: float
    heuristic_biases: dict[str, float] = field(default_factory=dict)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export spatial context schema as a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable="spatial_context_schema",
            operator="==",
            value=f"archetype:{self.archetype_name}:density:{self.obstacle_density:.2f}",
            relation="parahippocampal_scene_context",
            target_entity="parahippocampal_cortex",
        )


@dataclass
class ContextTransitionEvent:
    """Event emitted when environmental geometry triggers a context schema switch."""

    previous_schema_id: str
    new_schema_id: str
    divergence_score: float
    timestamp: float


class ParahippocampalContextEngine:
    """Parahippocampal cortex engine recognizing macro-scale environmental context schemas."""

    def __init__(
        self,
        hysteresis_threshold: float = 0.35,
        history_window: int = 10,
    ) -> None:
        """Initialize parahippocampal context engine.

        Args:
            hysteresis_threshold: Divergence threshold required to trigger a schema transition.
            history_window: Number of recent frames to consider for smoothing.
        """
        self.hysteresis_threshold = float(hysteresis_threshold)
        self.history_window = max(1, int(history_window))
        self.active_schema: SpatialContextSchema | None = None
        self.transitions: list[ContextTransitionEvent] = []
        self.step_counter: int = 0

    def analyze_environmental_envelope(
        self,
        boundary_grid: np.ndarray,
    ) -> SpatialContextSchema:
        """Extract spatial envelope features and classify environmental context schema.

        Args:
            boundary_grid: 2D binary array where >0 indicates wall/obstacle and 0 indicates traversable space.

        Returns:
            SpatialContextSchema representing active scene archetype.
        """
        self.step_counter += 1
        h, w = boundary_grid.shape
        total_cells = float(h * w)

        if total_cells <= 0.0:
            return SpatialContextSchema(
                schema_id="schema_empty",
                archetype_name="open_arena",
                aspect_ratio=1.0,
                obstacle_density=0.0,
                boundary_enclosure=0.0,
                estimated_corridor_ratio=0.0,
                heuristic_biases={"straight_line_bonus": 1.0},
            )

        # 1. Aspect ratio
        aspect_ratio = float(w) / float(max(1, h))

        # 2. Obstacle / Wall density
        obstacle_count = float(np.count_nonzero(boundary_grid > 0))
        obstacle_density = obstacle_count / total_cells

        # 3. Perimeter boundary enclosure
        top_wall = np.count_nonzero(boundary_grid[0, :] > 0)
        bot_wall = np.count_nonzero(boundary_grid[-1, :] > 0)
        left_wall = np.count_nonzero(boundary_grid[:, 0] > 0)
        right_wall = np.count_nonzero(boundary_grid[:, -1] > 0)
        perimeter_cells = 2 * (h + w) - 4
        boundary_enclosure = float(top_wall + bot_wall + left_wall + right_wall) / float(
            max(1, perimeter_cells)
        )
        boundary_enclosure = min(1.0, max(0.0, boundary_enclosure))

        # 4. Corridor estimation via local traversable width sampling
        # Internal cells traversable with adjacent walls on opposite sides
        corridor_count = 0
        internal_traversable = 0
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if boundary_grid[r, c] == 0:
                    internal_traversable += 1
                    horiz_corridor = (boundary_grid[r - 1, c] > 0) and (boundary_grid[r + 1, c] > 0)
                    vert_corridor = (boundary_grid[r, c - 1] > 0) and (boundary_grid[r, c + 1] > 0)
                    if horiz_corridor or vert_corridor:
                        corridor_count += 1

        corridor_ratio = (
            float(corridor_count) / float(max(1, internal_traversable))
            if internal_traversable > 0
            else 0.0
        )

        if h > 2 and w > 2:
            internal_grid = boundary_grid[1:-1, 1:-1]
            internal_obstacle_count = float(np.count_nonzero(internal_grid > 0))
            internal_obstacle_density = internal_obstacle_count / float((h - 2) * (w - 2))
        else:
            internal_obstacle_density = 0.0

        # 5. Archetype Classification
        archetype, biases = self._classify_archetype(
            aspect_ratio=aspect_ratio,
            obstacle_density=obstacle_density,
            boundary_enclosure=boundary_enclosure,
            corridor_ratio=corridor_ratio,
            internal_obstacle_density=internal_obstacle_density,
        )

        schema_id = f"schema_{archetype}"
        candidate_schema = SpatialContextSchema(
            schema_id=schema_id,
            archetype_name=archetype,
            aspect_ratio=aspect_ratio,
            obstacle_density=obstacle_density,
            boundary_enclosure=boundary_enclosure,
            estimated_corridor_ratio=corridor_ratio,
            heuristic_biases=biases,
        )

        # 6. Apply Hysteresis
        if self.active_schema is None:
            self.active_schema = candidate_schema
        else:
            divergence = self._compute_schema_divergence(self.active_schema, candidate_schema)
            if divergence >= self.hysteresis_threshold:
                transition = ContextTransitionEvent(
                    previous_schema_id=self.active_schema.schema_id,
                    new_schema_id=candidate_schema.schema_id,
                    divergence_score=divergence,
                    timestamp=float(self.step_counter),
                )
                self.transitions.append(transition)
                self.active_schema = candidate_schema

        return self.active_schema

    def _classify_archetype(
        self,
        aspect_ratio: float,
        obstacle_density: float,
        boundary_enclosure: float,
        corridor_ratio: float,
        internal_obstacle_density: float = 0.0,
    ) -> tuple[str, dict[str, float]]:
        """Classify geometric parameters into an environmental schema archetype."""
        # Highly elongated aspect ratio -> corridor
        if aspect_ratio >= 3.0 or aspect_ratio <= 0.33 or corridor_ratio > 0.40:
            return "narrow_corridor", {
                "straight_line_bonus": 1.5,
                "wall_following_weight": 0.3,
                "dead_end_pruning": 1.2,
            }

        # High boundary enclosure with few internal obstacles -> open arena
        if boundary_enclosure >= 0.65 and internal_obstacle_density < 0.15:
            return "open_arena", {
                "straight_line_bonus": 1.3,
                "wall_following_weight": 0.2,
                "geodesic_euclidean_alignment": 1.8,
            }

        # High internal obstacle density + high corridor ratio -> labyrinth maze
        if internal_obstacle_density >= 0.20 and corridor_ratio >= 0.20:
            return "labyrinth_maze", {
                "straight_line_bonus": 0.5,
                "wall_following_weight": 1.4,
                "dead_end_pruning": 2.0,
                "backtracking_tolerance": 1.5,
            }

        # High internal obstacle density with low corridor ratio -> dense obstacle field
        if internal_obstacle_density >= 0.25:
            return "dense_obstacle_field", {
                "straight_line_bonus": 0.8,
                "wall_following_weight": 0.8,
                "obstacle_margin_multiplier": 1.5,
            }

        # Moderate obstacle density with internal partitions -> compartmentalized rooms
        return "compartmentalized_rooms", {
            "straight_line_bonus": 1.0,
            "wall_following_weight": 1.0,
            "doorway_bottleneck_bonus": 1.6,
        }

    def _compute_schema_divergence(
        self,
        schema_a: SpatialContextSchema,
        schema_b: SpatialContextSchema,
    ) -> float:
        """Compute Euclidean distance between normalized feature vectors of two schemas."""
        d_aspect = abs(schema_a.aspect_ratio - schema_b.aspect_ratio) / max(
            1.0, schema_a.aspect_ratio + schema_b.aspect_ratio
        )
        d_density = abs(schema_a.obstacle_density - schema_b.obstacle_density)
        d_enclosure = abs(schema_a.boundary_enclosure - schema_b.boundary_enclosure)
        d_corridor = abs(schema_a.estimated_corridor_ratio - schema_b.estimated_corridor_ratio)
        return float(math.sqrt(d_aspect**2 + d_density**2 + d_enclosure**2 + d_corridor**2))

    def reset(self) -> None:
        """Reset internal history and active schema."""
        self.active_schema = None
        self.transitions.clear()
        self.step_counter = 0
