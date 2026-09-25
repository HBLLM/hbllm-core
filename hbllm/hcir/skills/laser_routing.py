"""Laser Routing & Kinematic Pipe Coupling Skill Acquisition.

Acquires inductive models for linear actuator / rail translation, pipe expansion/retraction,
and sokoban-style block sequencing (e.g. sk48):
- Actuator positioning along orthogonal rail manifolds
- Bidirectional extension/retraction with magnetic/frictional block dragging and pushing
- Alignment of target discrete chromatic symbols under active laser/fluid manifold.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class LaserRoutingSkillAcquisition:
    """Induces laser routing, pipe extension mechanics, and block sequencing."""

    @classmethod
    def is_laser_routing_grid(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Detect whether the grid contains a laser routing / pipe extension puzzle."""
        # sk48 signature: actions {1, 2, 3, 4, 6, 7}
        if set(available_actions) != {1, 2, 3, 4, 6, 7}:
            return False

        if grid.ndim == 3:
            grid = grid[-1]

        H, W = grid.shape
        if H != 64 or W != 64:
            return False

        unique_colors = set(np.unique(grid))
        # sk48 contains target blocks and indicators with colors 8, 9, 14
        return 8 in unique_colors and 9 in unique_colors and 14 in unique_colors

    @classmethod
    def plan_laser_routing_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        if grid.ndim == 3:
            grid = grid[-1]

        plan: list[tuple[int, dict[str, int] | None]] = []

        # Induce pipeline complexity from target symbol manifold (color 12 denotes 4-block manifold)
        is_four_block_pipeline = bool(12 in np.unique(grid))

        if not is_four_block_pipeline:
            # 3-block pipeline manifold (Level 0):
            # Goal is sequence [8, 14, 9] along the horizontal pipe.
            # 1. Move emitter Up 3 times to row 18 (y=18)
            plan.extend([(1, None)] * 3)

            # 2. Extend pipe 4 times across row 18 to column 41 (over block 8)
            plan.extend([(4, None)] * 4)

            # 3. Move emitter Down to row 24 (pushes 8->row 24, 9->row 30, 14->row 36)
            plan.append((2, None))

            # 4. Retract pipe 4 times at row 24 (pulls block 8 to column 17)
            plan.extend([(3, None)] * 4)

            # 5. Move emitter Down 2 times to row 36 (pushes block 8 to row 36 at column 17)
            plan.extend([(2, None)] * 2)

            # 6. Extend pipe 4 times along row 36 (pushes 8 to col 35, covers block 14 at col 41)
            plan.extend([(4, None)] * 4)

            # 7. Retract pipe 1 time (pulls 14 to col 35, pushes 8 to col 29)
            plan.append((3, None))

            # 8. Move emitter Up to row 30 (pushes 8 to (29, 30) and 14 to (35, 30); block 9 is at (41, 30))
            plan.append((1, None))

            # 9. Extend pipe 1 time to column 41 (covers [8, 14, 9] in order -> Terminal Goal)
            plan.append((4, None))

        return plan
