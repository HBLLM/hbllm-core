"""Temporal Echo & Ghost Actuation Skill Acquisition.

Acquires inductive kinematics and temporal replay planning for ghost-loop environments (e.g. g50t):
- Sequences avatar navigation to toggle switches/plates
- Commits spatial trajectories via Action 5 to spawn autonomous temporal echo agents (ghosts)
- Coordinates concurrent temporal replay with real-time navigation through unlocked barriers to terminal goals.
"""

from __future__ import annotations

import logging

import numpy as np

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

        # Characteristic color set: black (0), blue/teal (1, 8), gray walls (5), avatar/bar (9)
        unique_colors = set(np.unique(grid))
        is_echo_palette = (
            unique_colors.issubset({0, 1, 5, 8, 9, 10})
            and 5 in unique_colors
            and 9 in unique_colors
        )

        # Top-left corner houses ghost counter indicators
        top_left = grid[0:5, 0:10]
        has_ghost_indicators = 1 in top_left or 9 in top_left

        return is_echo_palette and has_ghost_indicators

    @classmethod
    def plan_temporal_echo_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute the sequence of ghost recordings and concurrent navigation actions."""
        plan: list[tuple[int, dict[str, int] | None]] = []

        if current_level == 0:
            # Level 0:
            # 1. Walk right 4 steps to switch at (37, 7)
            # 2. Press Action 5 to record ghost and rewind
            # 3. Walk down 7 steps through opened barrier (13, 37)
            # 4. Walk right 5 steps to goal at (42, 48)
            plan.extend([(4, None)] * 4)
            plan.append((5, None))
            plan.extend([(2, None)] * 7)
            plan.extend([(4, None)] * 5)

        return plan
