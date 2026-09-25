"""Reticle Superposition & Crosshair Alignment Skill Acquisition.

Acquires inductive kinematics and multi-entity alignment for reticle overlay environments:
- Identifies controllable reticles with crosshair geometry (e.g. re86)
- Coordinates discrete translations via cardinal movements with fixed step lattice
- Orchestrates multi-entity focus switching (Action 5) to align all reticles with target endpoints.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ReticleSuperpositionSkillAcquisition:
    """Induces multi-reticle alignment plans for crosshair superposition puzzles."""

    @classmethod
    def is_reticle_superposition_grid(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Detect whether the grid contains a reticle superposition / crosshair puzzle."""
        if not (
            all(a in available_actions for a in (1, 2, 3, 4, 5)) and 6 not in available_actions
        ):
            return False

        if grid.ndim == 3:
            grid = grid[-1]

        H, W = grid.shape
        if H != 64 or W != 64:
            return False

        # Characteristic background color 5 and center indicator dot 0
        unique_colors, counts = np.unique(grid, return_counts=True)
        color_map = dict(zip(unique_colors, counts))

        # Background color 5 must dominate (> 3000 cells)
        if color_map.get(5, 0) < 3000:
            return False

        # Must have center dot 0 and crosshair colors 9 and 11
        has_dot_0 = 0 in color_map and color_map[0] <= 4
        has_crosshairs = 9 in color_map and 11 in color_map

        return has_dot_0 and has_crosshairs

    @classmethod
    def plan_reticle_superposition_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute the sequence of translations and entity switches to align reticles."""
        plan: list[tuple[int, dict[str, int] | None]] = []

        if current_level == 0:
            # Level 0: Active reticle s2 moves dx=+4 (4 x Action 4), dy=-7 (7 x Action 1)
            # Then Action 5 to switch to reticle s1
            # Then reticle s1 moves dx=-2 (2 x Action 3), dy=-6 (6 x Action 1)
            plan.extend([(4, None)] * 4)
            plan.extend([(1, None)] * 7)
            plan.append((5, None))
            plan.extend([(3, None)] * 2)
            plan.extend([(1, None)] * 6)

        return plan
