"""Perceptual Skill Context: Perception-Driven Grid Analysis.

Provides a zero-hardcoded-constant context for all declarative skills.
Instead of assuming ``tile_size = 6`` or ``center_y = 39``, skills query
this context which derives all geometry from actual grid analysis.

This is the bridge between raw grid observations and skill planning —
the core principle is: **perceive, don't assume**.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hbllm.hcir.primitives import (
    EntityComponent,
    detect_background_color,
    find_grid_dividers,
    op_extract_subgrid_panels,
    segment_entities,
)
from hbllm.hcir.skills.common_subskills import (
    LatticeQuantizer,
)

logger = logging.getLogger(__name__)


@dataclass
class PerceptualSkillContext:
    """Perception-derived context for skill planning — zero hardcoded constants.

    All geometry, entity positions, and structural features are derived
    from actual grid analysis at runtime. Skills should use this context
    instead of hardcoded pixel coordinates or tile sizes.

    Attributes:
        grid: Raw 2D numpy grid observation.
        bg_color: Detected background color (most frequent pixel).
        entities: All segmented discrete entity components.
        entity_by_color: Entities grouped by color for fast lookup.
        unique_colors: Set of all distinct non-background colors.
        dividers: Row/column indices of grid divider lines (if any).
        panels: Extracted subgrid panels if grid has dividers.
        lattice_stride: Detected lattice cell size (tile quantization).
        avatar_entity: The controllable agent entity (if identified).
        avatar_color: Color of the avatar entity.
        goal_entities: Entities likely to be targets/goals.
        clickable_entities: Entities that respond to click actions.
        obstacle_colors: Colors identified as impassable.
        obstacle_mask: 2D binary mask indicating impassable cells.
    """

    grid: np.ndarray
    bg_color: int
    entities: list[EntityComponent]
    entity_by_color: dict[int, list[EntityComponent]] = field(default_factory=dict)
    unique_colors: set[int] = field(default_factory=set)
    dividers: tuple[list[int], list[int], int | None] | None = None
    panels: list[np.ndarray] | None = None
    lattice_stride: int = 1
    avatar_entity: EntityComponent | None = None
    avatar_color: int | None = None
    goal_entities: list[EntityComponent] = field(default_factory=list)
    clickable_entities: list[EntityComponent] = field(default_factory=list)
    obstacle_colors: set[int] = field(default_factory=set)
    obstacle_mask: np.ndarray | None = None

    # Derived geometry (computed lazily)
    _panel_count: int | None = None
    _symmetry_axis: str | None = None

    @classmethod
    def from_grid(
        cls,
        grid: np.ndarray,
        available_actions: list[int] | None = None,
        avatar_feature: Any = None,
        learned_obstacles: set[Any] | None = None,
        learned_targets: set[Any] | None = None,
    ) -> PerceptualSkillContext:
        """Build full perceptual context from raw grid — no hardcoded values.

        Args:
            grid: 2D numpy grid observation.
            available_actions: List of valid action IDs.
            avatar_feature: Known avatar color/feature from prior learning.
            learned_obstacles: Colors/features known to be obstacles.
            learned_targets: Colors/features known to be goals.
        """
        if grid.ndim == 3:
            grid = grid[-1]

        bg = detect_background_color(grid)
        entities = segment_entities(grid, bg_color=bg)

        # Group entities by color
        by_color: dict[int, list[EntityComponent]] = {}
        for e in entities:
            by_color.setdefault(e.color, []).append(e)

        unique_colors = set(int(c) for c in np.unique(grid)) - {bg}

        # Detect grid structure & panels
        dividers = find_grid_dividers(grid)
        panels = None
        if dividers:
            try:
                panels = op_extract_subgrid_panels(grid)
            except Exception:
                pass

        # Detect lattice stride
        stride = 1
        try:
            anchor = LatticeQuantizer.detect_lattice_anchor(grid, bg)
            if anchor and anchor > 1:
                stride = anchor
        except Exception:
            pass

        # Identify avatar
        avatar = None
        avatar_color = None
        if avatar_feature is not None:
            candidates = by_color.get(int(avatar_feature), [])
            if candidates:
                avatar = min(candidates, key=lambda e: e.area)
                avatar_color = avatar.color

        if avatar is None:
            # Fallback heuristic: small singleton entities with distinct color
            singletons = [
                elist[0]
                for c, elist in by_color.items()
                if len(elist) == 1 and 2 <= elist[0].area <= 64 and c != bg and c != 0
            ]
            if singletons:
                # Avatar is typically the singleton with lowest vertical position or smallest area
                avatar = max(singletons, key=lambda e: (e.centroid[0], -e.area))
                avatar_color = avatar.color
            elif entities:
                non_bg_ents = [e for e in entities if e.color != bg and e.area >= 2]
                if non_bg_ents:
                    avatar = min(non_bg_ents, key=lambda e: e.area)
                    avatar_color = avatar.color

        # Identify obstacles
        obstacle_colors = set()
        if learned_obstacles:
            obstacle_colors = {int(o) for o in learned_obstacles if isinstance(o, (int, float))}

        # Identify goals — entities matching learned targets or non-avatar/non-obstacle
        goals: list[EntityComponent] = []
        if learned_targets:
            target_colors = {int(t) for t in learned_targets if isinstance(t, (int, float))}
            goals = [e for e in entities if e.color in target_colors]

        if not goals and avatar is not None:
            # Fallback: small distinct entities located near borders/summit
            H, W = grid.shape
            border_candidates = [
                e
                for e in entities
                if e is not avatar
                and e.color != bg
                and e.area <= 50
                and (e.min_r <= 5 or e.max_r >= H - 6 or e.min_c <= 5 or e.max_c >= W - 6)
            ]
            if border_candidates:
                goals = border_candidates

        # Obstacle mask
        obstacle_mask = np.zeros_like(grid, dtype=bool)
        for e in entities:
            if e.color in obstacle_colors or e.area > 200:
                for r, c in e.coords:
                    obstacle_mask[r, c] = True

        # Identify clickable entities (small, non-background, potentially interactive)
        has_click = available_actions and 6 in available_actions
        clickable: list[EntityComponent] = []
        if has_click:
            # Clickable entities are typically small colored objects
            clickable = [
                e
                for e in entities
                if e.color != bg
                and e.area >= 2
                and e.area <= 200
                and e.color not in obstacle_colors
            ]

        return cls(
            grid=grid,
            bg_color=bg,
            entities=entities,
            entity_by_color=by_color,
            unique_colors=unique_colors,
            dividers=dividers if (dividers and (dividers[0] or dividers[1])) else None,
            panels=panels,
            lattice_stride=stride,
            avatar_entity=avatar,
            avatar_color=avatar_color,
            goal_entities=goals,
            clickable_entities=clickable,
            obstacle_colors=obstacle_colors,
            obstacle_mask=obstacle_mask,
        )

    # ── Derived Queries ──────────────────────────────────────────────────

    @property
    def has_panels(self) -> bool:
        """Whether the grid has panel-like subdivisions."""
        return self.dividers is not None and bool(self.dividers[0] or self.dividers[1])

    @property
    def entity_count(self) -> int:
        """Total number of segmented entities."""
        return len(self.entities)

    @property
    def color_count(self) -> int:
        """Number of distinct non-background colors."""
        return len(self.unique_colors)

    def entities_of_color(self, color: int) -> list[EntityComponent]:
        """Get all entities of a specific color."""
        return self.entity_by_color.get(color, [])

    def nearest_entity_to(
        self, pos: tuple[float, float], exclude_colors: set[int] | None = None
    ) -> EntityComponent | None:
        """Find the nearest entity to a given position."""
        best = None
        best_dist = float("inf")
        for e in self.entities:
            if exclude_colors and e.color in exclude_colors:
                continue
            cr, cc = e.centroid
            d = abs(cr - pos[0]) + abs(cc - pos[1])
            if d < best_dist:
                best_dist = d
                best = e
        return best

    def entities_in_region(
        self, r_min: int, r_max: int, c_min: int, c_max: int
    ) -> list[EntityComponent]:
        """Get entities whose centroid falls within a bounding box."""
        return [
            e
            for e in self.entities
            if r_min <= e.centroid[0] <= r_max and c_min <= e.centroid[1] <= c_max
        ]

    def entities_above(self, entity: EntityComponent) -> list[EntityComponent]:
        """Get entities whose centroid is above (lower row) the given entity."""
        return [e for e in self.entities if e.centroid[0] < entity.min_r and e is not entity]

    def entities_sorted_by_distance(self, origin: tuple[float, float]) -> list[EntityComponent]:
        """Return all entities sorted by Manhattan distance from origin."""
        return sorted(
            self.entities,
            key=lambda e: abs(e.centroid[0] - origin[0]) + abs(e.centroid[1] - origin[1]),
        )
