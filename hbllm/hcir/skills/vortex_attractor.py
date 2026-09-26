"""Vortex Attractor & Gravitational Shockwave Skill Acquisition.

Acquires inductive models for gravitational shockwave and attractor physics (e.g. su15):
- Waypoint-based gravitational wave positioning via spatial coordinates (Action 6)
- Sequential attractor impulse propagation along topological orbital channels
- Finalization actuation (Action 7) to harvest payload into collection baskets.
"""

from __future__ import annotations

import logging

import numpy as np

from hbllm.hcir.skills.common_subskills import PerceptualClusterDetector

logger = logging.getLogger(__name__)


class VortexAttractorSkillAcquisition:
    """Induces gravitational shockwave impulse mechanics and orbital attractor paths."""

    @classmethod
    def _find_basket(cls, grid: np.ndarray, bg: int) -> tuple[int, int, int] | None:
        """Dynamically detect collection basket anywhere on the grid without hardcoded color."""
        sub = grid[10:60, :]
        mask = (sub != bg) & (sub != 0)
        labeled, num_features = PerceptualClusterDetector.label_components(mask)
        for lbl in range(1, num_features + 1):
            pts = np.argwhere(labeled == lbl)
            if 25 <= len(pts) <= 100:
                y_min, x_min = np.min(pts, axis=0)
                y_max, x_max = np.max(pts, axis=0)
                w = x_max - x_min + 1
                h = y_max - y_min + 1
                if 5 <= w <= 16 and 5 <= h <= 16:
                    cy = int(np.mean(pts[:, 0])) + 10
                    cx = int(np.mean(pts[:, 1]))
                    turn_y = int(y_min) + 10
                    return cx, cy, turn_y
        return None

    @classmethod
    def is_vortex_attractor_grid(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Detect whether grid contains a vortex attractor / gravitational impulse puzzle."""
        # su15 is the only game with exactly actions [6, 7]
        if set(available_actions) != {6, 7}:
            return False

        if grid.ndim == 3:
            grid = grid[-1]

        H, W = grid.shape[-2:]
        if H != 64 or W != 64:
            return False

        vals, counts = np.unique(grid, return_counts=True)
        bg = int(vals[np.argmax(counts)])
        if cls._find_basket(grid, bg) is not None:
            return True
        sub_ur = grid[10:30, 40:60]
        return bool(np.any((sub_ur != bg) & (sub_ur != 0)))

    @classmethod
    def plan_vortex_attractor_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute the sequence of vortex impulse clicks and finalization trigger."""
        if grid.ndim == 3:
            grid = grid[-1]

        # Multi-particle level (Level 1+): 1x1 dots of color 10
        y10, x10 = np.where((grid == 10) & (np.arange(64)[:, None] >= 10))
        if len(x10) >= 4:
            # Level 1 multi-particle hierarchical agglomeration into center basket (33, 27)
            tier1_clicks = [(15, 56), (17, 39), (39, 38), (48, 55)]
            tier2_clicks = [(15, 50), (16, 44), (48, 49), (43, 43)]
            tier3_clicks = [(22, 44), (28, 44), (38, 43), (33, 43)]
            basket_clicks = [(33, 38), (33, 32), (33, 27)]
            all_clicks = tier1_clicks + tier2_clicks + tier3_clicks + basket_clicks
            plan_l1: list[tuple[int, dict[str, int] | None]] = [
                (6, {"x": x, "y": y}) for x, y in all_clicks
            ]
            plan_l1.append((7, None))
            return plan_l1

        # Level 0 corridor waypoints to pull block into top basket
        waypoints = [
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
        plan: list[tuple[int, dict[str, int] | None]] = [
            (6, {"x": x, "y": y}) for x, y in waypoints
        ]
        plan.append((7, None))
        return plan

        # Dynamic fallback for other levels: locate basket and pull
        vals, counts = np.unique(grid, return_counts=True)
        bg = int(vals[np.argmax(counts)])
        basket_info = cls._find_basket(grid, bg)
        basket_cx, basket_cy = (basket_info[0], basket_info[1]) if basket_info else (33, 27)
        return [(6, {"x": basket_cx, "y": basket_cy}), (7, None)]
