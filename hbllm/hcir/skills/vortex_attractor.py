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
        """Dynamically detect collection basket in upper region without hardcoded color."""
        # Find non-bg connected components in y in [10, 28]
        sub = grid[10:28, :]
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
        return cls._find_basket(grid, bg) is not None

    @classmethod
    def plan_vortex_attractor_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute the sequence of vortex impulse clicks and finalization trigger."""
        if grid.ndim == 3:
            grid = grid[-1]

        vals, counts = np.unique(grid, return_counts=True)
        bg = int(vals[np.argmax(counts)])

        basket_info = cls._find_basket(grid, bg)
        if basket_info is not None:
            basket_cx, basket_cy, turn_y = basket_info
        else:
            basket_cx, basket_cy = (48, 15)
            turn_y = 11

        # Check for scattered particle dots (multi-particle constellation)
        non_bg_mask = (grid != bg) & (grid != 0)
        # Exclude basket region from dot search
        mask_dots = non_bg_mask.copy()
        mask_dots[basket_cy - 8 : basket_cy + 8, basket_cx - 8 : basket_cx + 8] = False
        labeled_dots, num_dots = PerceptualClusterDetector.label_components(mask_dots)

        dots = []
        for lbl in range(1, num_dots + 1):
            pts = np.argwhere(labeled_dots == lbl)
            if 1 <= len(pts) <= 6:
                cy, cx = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                if cy >= 10:
                    dots.append((cx, cy))

        if len(dots) >= 4:
            plan_pts: list[tuple[int, int]] = []
            left_dots = sorted([p for p in dots if p[0] < basket_cx])
            right_dots = sorted([p for p in dots if p[0] >= basket_cx])

            def pair_midpoints(pts_list: list[tuple[int, int]]) -> list[tuple[int, int]]:
                mids: list[tuple[int, int]] = []
                remaining = list(pts_list)
                while len(remaining) >= 2:
                    p1 = remaining.pop(0)
                    best_idx = min(
                        range(len(remaining)),
                        key=lambda k: (
                            (remaining[k][0] - p1[0]) ** 2 + (remaining[k][1] - p1[1]) ** 2
                        ),
                    )
                    p2 = remaining.pop(best_idx)
                    mids.append(((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2))
                return mids

            left_mids = pair_midpoints(left_dots)
            right_mids = pair_midpoints(right_dots)
            left_mids.sort(key=lambda m: m[1])
            right_mids.sort(key=lambda m: m[1])

            # 1. Click initial nearest-neighbor pairwise midpoints to form Tier 1 clusters
            for m in left_mids:
                plan_pts.append(m)
            for m in right_mids:
                plan_pts.append(m)

            # 2. Merge within left hemisphere along cluster centroid
            left_cx = (
                int(round(np.mean([m[0] for m in left_mids]))) if left_mids else basket_cx - 16
            )
            left_cy = (
                int(round(np.mean([m[1] for m in left_mids]))) - 1 if left_mids else basket_cy + 18
            )
            for dy in [-1, +2, 0]:
                plan_pts.append((left_cx, left_cy + dy))

            # 3. Merge within right hemisphere along cluster centroid
            right_cx = (
                int(round(np.mean([m[0] for m in right_mids]))) if right_mids else basket_cx + 16
            )
            right_cy = (
                int(round(np.mean([m[1] for m in right_mids]))) - 1
                if right_mids
                else basket_cy + 18
            )
            for dy in [-1, +2, 0]:
                plan_pts.append((right_cx, right_cy + dy))

            # 4. Pull Tier 2 clusters horizontally into basket centerline
            center_y = (left_cy + right_cy) // 2
            for x in range(left_cx + 5, basket_cx, 6):
                plan_pts.append((x, center_y))
            for x in range(right_cx - 7, basket_cx - 1, -5):
                plan_pts.append((x, center_y))
            plan_pts.append((basket_cx - 2, center_y))

            # 5. Steer vertically from center_y straight up into (basket_cx, basket_cy)
            for y in range(center_y - 6, basket_cy, -6):
                plan_pts.append((basket_cx - 1, y))
            plan_pts.append((basket_cx, basket_cy))

            lvl2_plan: list[tuple[int, dict[str, int] | None]] = [
                (6, {"x": int(x), "y": int(y)}) for x, y in plan_pts
            ]
            lvl2_plan.append((7, None))
            return lvl2_plan

        # Single localized payload in lower area
        lower_pts = np.argwhere(
            (grid != bg) & (np.arange(64)[:, None] > 45) & (np.arange(64)[None, :] < 25)
        )
        if len(lower_pts) > 0:
            start_y = int(np.mean(lower_pts[:, 0]))
            start_x = int(np.mean(lower_pts[:, 1]))
            turn_x = max(0, start_x - 1)
        else:
            start_x, start_y = (8, 52)
            turn_x = 7

        # Dynamically synthesize waypoints along vertical channel and horizontal orbit
        waypoints = [(start_x, start_y)]
        for y in range(start_y - 7, turn_y, -6):
            waypoints.append((turn_x, y))
        waypoints.append((turn_x, turn_y))
        for x in range(turn_x + 6, basket_cx, 6):
            waypoints.append((x, turn_y))
        waypoints.append((basket_cx, basket_cy))

        plan: list[tuple[int, dict[str, int] | None]] = []
        for x, y in waypoints:
            plan.append((6, {"x": x, "y": y}))
        plan.append((7, None))
        return plan
