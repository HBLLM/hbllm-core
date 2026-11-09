"""Geometric, Support Stability, and Spatial Path Collision Models for A18.

Derives support stability from grounded surface geometry rather than object names,
and evaluates spatial collision clearance along trajectory waypoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class SurfaceGeometry(str, Enum):
    """Surface curvature and geometric contact characteristics."""

    FLAT = "flat"  # Stable support (e.g. table, box top, floor)
    CONVEX = "convex"  # Unstable support, prone to roll/fall (e.g. sphere, ball, cylinder side)
    CONCAVE = "concave"  # Container interior / cradle (e.g. bowl, hollow cavity)
    IRREGULAR = "irregular"  # Variable support


@dataclass
class BoundingBox:
    """Axis-aligned bounding box representation for spatial entities."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    width: float = 1.0
    depth: float = 1.0
    height: float = 1.0

    def contains_point(self, px: float, py: float, pz: float = 0.0) -> bool:
        return (
            self.x <= px <= self.x + self.width
            and self.y <= py <= self.y + self.depth
            and self.z <= pz <= self.z + self.height
        )

    def intersects(self, other: BoundingBox) -> bool:
        return not (
            self.x + self.width < other.x
            or other.x + other.width < self.x
            or self.y + self.depth < other.y
            or other.y + other.depth < self.y
            or self.z + self.height < other.z
            or other.z + other.height < self.z
        )


def derive_surface_geometry(
    properties: dict[str, Any], entity_type: str = ""
) -> SurfaceGeometry | None:
    """Infer surface geometry from entity properties and prototype shapes."""
    geom_str = str(properties.get("geometry", properties.get("surface", ""))).lower()
    if geom_str in ("flat", "planar"):
        return SurfaceGeometry.FLAT
    if geom_str in ("convex", "curved", "round"):
        return SurfaceGeometry.CONVEX
    if geom_str in ("concave", "hollow", "cavity"):
        return SurfaceGeometry.CONCAVE

    # Fallback from shape / entity type
    shape = str(properties.get("shape", "")).lower()
    if shape in ("unknown", ""):
        etype = entity_type.lower()
        if etype in ("table", "box", "block", "floor", "shelf", "cube", "tray"):
            return SurfaceGeometry.FLAT
        if etype in ("ball", "sphere", "globe", "egg"):
            return SurfaceGeometry.CONVEX
        return None  # Unknown geometry

    if shape in ("ball", "sphere", "globe", "egg"):
        return SurfaceGeometry.CONVEX
    if shape in ("bowl", "basin", "cup_interior", "container"):
        return SurfaceGeometry.CONCAVE
    if shape in ("table", "box", "block", "floor", "shelf", "cube", "tray"):
        return SurfaceGeometry.FLAT

    return None


def evaluate_support_stability(
    supported_props: dict[str, Any],
    supporting_props: dict[str, Any],
    supporting_type: str = "",
) -> tuple[bool, float, str]:
    """Derive whether stacking an object on another is physically stable.

    Returns:
        (is_stable, stability_score, reason)
    """
    base_geom = derive_surface_geometry(supporting_props, supporting_type)

    if base_geom is None:
        return False, 0.20, "unstable_unknown_geometry"

    if base_geom == SurfaceGeometry.FLAT:
        return True, 0.95, "stable_flat_support"

    if base_geom == SurfaceGeometry.CONCAVE:
        return True, 0.90, "stable_concave_cradle"

    if base_geom == SurfaceGeometry.CONVEX:
        # Placing an object on a convex curved surface results in unstable rolling/falling
        return False, 0.08, "unstable_convex_curvature_fall"

    return False, 0.20, "unstable_irregular_support"


def is_path_clear(
    start_pos: tuple[float, float],
    end_pos: tuple[float, float],
    obstacles: list[tuple[str, BoundingBox]],
    step_size: float = 0.2,
) -> tuple[bool, str | None]:
    """Evaluate if a linear waypoint path is free from obstacle collisions.

    Returns:
        (is_clear, colliding_obstacle_id)
    """
    x0, y0 = start_pos
    x1, y1 = end_pos
    dx = x1 - x0
    dy = y1 - y0
    dist = (dx**2 + dy**2) ** 0.5

    if dist < 1e-6:
        return True, None

    steps = max(1, int(dist / step_size))
    for i in range(steps + 1):
        frac = i / float(steps)
        curr_x = x0 + frac * dx
        curr_y = y0 + frac * dy

        for obs_id, obs_box in obstacles:
            if obs_box.contains_point(curr_x, curr_y, 0.0):
                return False, obs_id

    return True, None
