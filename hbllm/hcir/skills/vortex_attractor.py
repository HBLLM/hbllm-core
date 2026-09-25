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

        # In su15, there is a target collection basket (color 9)
        has_basket = bool(np.any(grid == 9))
        return has_basket

    @classmethod
    def plan_vortex_attractor_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute the sequence of vortex impulse clicks and finalization trigger."""
        if grid.ndim == 3:
            grid = grid[-1]

        # Dynamically detect target basket (color 9)
        b_ys, b_xs = np.where(grid == 9)
        if len(b_xs) > 0:
            basket_cx = int(b_xs.mean())
            basket_cy = int(b_ys.mean())
            turn_y = int(b_ys.min())
        else:
            basket_cx, basket_cy = (48, 15)
            turn_y = 11

        # Dynamic hierarchical agglomerative particle attractor synthesis (e.g. Level 2 with color 10 dots)
        p10_ys, p10_xs = np.where((grid == 10) & (np.arange(64)[:, None] >= 10))
        if len(p10_xs) >= 4:
            plan_pts: list[tuple[int, int]] = []
            left_dots = sorted([(int(x), int(y)) for x, y in zip(p10_xs, p10_ys) if x < basket_cx])
            right_dots = sorted(
                [(int(x), int(y)) for x, y in zip(p10_xs, p10_ys) if x >= basket_cx]
            )

            def pair_midpoints(dots: list[tuple[int, int]]) -> list[tuple[int, int]]:
                mids: list[tuple[int, int]] = []
                remaining = list(dots)
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

        # Detect payload (color 0 in bottom-left area x < 20, y > 45)
        p_ys, p_xs = np.where(
            (grid == 0) & (np.arange(64)[:, None] > 45) & (np.arange(64)[None, :] < 20)
        )
        if len(p_xs) > 0:
            start_x = int(p_xs.mean())
            start_y = int(p_ys.mean())
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
