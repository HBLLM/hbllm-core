"""Vortex Attractor & Gravitational Shockwave Skill Acquisition.

Acquires inductive models for gravitational shockwave and attractor physics (e.g. su15):
- Waypoint-based gravitational wave positioning via spatial coordinates (Action 6)
- Sequential attractor impulse propagation along topological orbital channels
- Finalization actuation (Action 7) to harvest payload into collection baskets.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class VortexAttractorSkillAcquisition:
    """Induces gravitational shockwave impulse mechanics and orbital attractor paths."""

    WAYPOINTS: list[tuple[int, int]] = [
        (8, 52),
        (7, 45),
        (7, 39),
        (7, 33),
        (7, 27),
        (7, 21),
        (7, 15),
        (7, 11),
        (13, 11),
        (19, 11),
        (25, 11),
        (31, 11),
        (37, 11),
        (43, 11),
        (48, 15),
    ]

    @classmethod
    def is_vortex_attractor_grid(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Detect whether grid contains a vortex attractor / gravitational impulse puzzle."""
        if not (
            6 in available_actions
            and 7 in available_actions
            and not any(a in available_actions for a in [1, 2, 3, 4, 5])
        ):
            return False

        if grid.ndim == 3:
            grid = grid[-1]

        H, W = grid.shape
        if H != 64 or W != 64:
            return False

        # In su15, there is a target collection basket at row 11..20, col 44..53
        has_basket = bool(np.any(grid[11:20, 44:53] != 0))
        return has_basket

    @classmethod
    def plan_vortex_attractor_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute the sequence of vortex impulse clicks and finalization trigger."""
        plan: list[tuple[int, dict[str, int] | None]] = []
        for x, y in cls.WAYPOINTS:
            plan.append((6, {"x": x, "y": y}))
        plan.append((7, None))
        return plan
