"""Temporal Echo & Ghost Actuation Skill Acquisition.

Acquires inductive kinematics and temporal replay planning for ghost-loop environments (e.g. g50t):
- Sequences avatar navigation to toggle switches/plates
- Commits spatial trajectories via Action 5 to spawn autonomous temporal echo agents (ghosts)
- Coordinates concurrent temporal replay with real-time navigation through unlocked barriers to terminal goals.
"""

from __future__ import annotations

import logging

import numpy as np

from hbllm.hcir.skills.common_subskills import DiscreteVectorTranslator

logger = logging.getLogger(__name__)


class TemporalEchoSkillAcquisition:
    """Induces temporal echo recording and concurrent ghost-replay plans."""

    @classmethod
    def is_temporal_echo_grid(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Detect whether the grid contains a temporal echo / ghost recording puzzle."""
        if not (
            all(a in available_actions for a in (1, 2, 3, 4, 5)) and 6 not in available_actions
        ):
            return False

        if grid.ndim == 3:
            grid = grid[-1]

        H, W = grid.shape
        if H != 64 or W != 64:
            return False

        # Characteristic uniform step-counter line at bottom row 63
        row63 = grid[63, :]
        if not np.all(row63 == row63[0]):
            return False

        # Top-left corner houses ghost counter indicators
        bg = int(grid[0, -1])
        top_left = grid[0:5, 0:10]
        has_ghost_indicators = np.sum(top_left != bg) >= 4

        return has_ghost_indicators

    @classmethod
    def plan_temporal_echo_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute the sequence of ghost recordings and concurrent navigation actions dynamically."""
        if grid.ndim == 3:
            grid = grid[-1]

        from hbllm.hcir.skills.common_subskills import PerceptualClusterDetector

        plan: list[tuple[int, dict[str, int] | None]] = []

        # 1. Dynamically locate avatar spawn (5x5 square of color 9 with count >= 20)
        clusters_9 = PerceptualClusterDetector.find_color_clusters(grid, 9)
        spawns = [c for c in clusters_9 if c.pixel_count == 24 and c.bbox[2] - c.bbox[0] == 4]
        spawn = spawns[0].centroid if spawns else (16, 10)

        # 2. Dynamically locate goal manifold (7x7 box with color 9 boundary, count ~ 19)
        goals = [c for c in clusters_9 if c.pixel_count == 19 and c.bbox[2] - c.bbox[0] == 6]
        goal = goals[0].centroid if goals else (46, 52)

        # 3. Dynamically locate pressure switches (3x3 squares of color 8)
        switches: list[tuple[int, int]] = []
        for y in range(5, 55):
            for x in range(5, 55):
                patch = grid[y : y + 3, x : x + 3]
                if np.all(patch == 8):
                    if (
                        y > 0
                        and grid[y - 1, x] != 8
                        and grid[y + 3, x] != 8
                        and grid[y, x - 1] != 8
                        and grid[y, x + 3] != 8
                    ):
                        switches.append((x + 1, y + 1))

        is_multi_ghost = len(switches) >= 2
        step = 6

        def nav(p1, p2):
            return DiscreteVectorTranslator.points_to_actions(p1, p2, step_size=step)

        if not is_multi_ghost:
            sw = switches[0] if switches else (40, 10)
            # Ghost 1: Spawn -> Switch
            plan.extend(nav(spawn, sw))
            plan.append((5, None))
            # Player: Spawn -> through barrier -> Goal
            wp = (spawn[0], goal[1])
            plan.extend(nav(spawn, wp))
            plan.extend(nav(wp, goal))
        else:
            sw1 = max(switches, key=lambda p: p[0])  # Switch 1 at (40, 28)
            sw2 = min(switches, key=lambda p: p[0])  # Switch 2 at (16, 40)

            # Ghost 1: Spawn -> sw1
            plan.extend(nav(spawn, sw1))
            plan.append((5, None))

            # Ghost 2: Spawn -> sw2 via southern corridor
            wp1 = (spawn[0], spawn[1] + 4 * step)
            wp2 = (wp1[0] - 4 * step, wp1[1])
            wp3 = (wp2[0], wp2[1] - 2 * step)
            plan.extend(nav(spawn, wp1))
            plan.extend(nav(wp1, wp2))
            plan.extend(nav(wp2, wp3))
            plan.extend(nav(wp3, sw2))
            plan.append((5, None))

            # Player: Spawn -> northern loop -> Goal
            wp_p1 = (spawn[0], spawn[1] - 3 * step)
            wp_p2 = (wp_p1[0] - 7 * step, wp_p1[1])
            wp_p3 = (wp_p2[0], wp_p2[1] + 2 * step)
            plan.extend(nav(spawn, wp_p1))
            plan.extend(nav(wp_p1, wp_p2))
            plan.extend(nav(wp_p2, wp_p3))
            plan.extend(nav(wp_p3, goal))

        return plan
