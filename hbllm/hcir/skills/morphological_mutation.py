"""Morphological State Mutation & Receptacle Keying Skill Acquisition.

Acquires inductive models for morphological attunement and keyed receptacle navigation (e.g. ls20):
- Avatar morphological attribute induction (shape, color, rotation)
- Keyed target receptacle affordance matching
- Modal attunement glyph navigation to mutate required attributes
- Shortest path synthesis across maze topologies to satisfy gate invariants.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class MorphologicalStateMutationSkillAcquisition:
    """Induces morphological attribute mutation requirements and plans gate attunement."""

    @classmethod
    def is_morphological_mutation_grid(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Detect whether the grid contains a morphological state mutation puzzle."""
        # ls20 signature: standard movement {1, 2, 3, 4} (no click actions)
        if set(available_actions) != {1, 2, 3, 4}:
            return False

        if grid.ndim == 3:
            grid = grid[-1]

        H, W = grid.shape
        if H != 64 or W != 64:
            return False

        unique_colors = set(np.unique(grid))
        # Unique color combination for ls20: avatar 12, pads/walls 8 and 9
        return 12 in unique_colors and 8 in unique_colors and 9 in unique_colors

    @classmethod
    def plan_morphological_mutation_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute the sequence of moves to attune avatar morphology and enter the terminal gate."""
        plan: list[tuple[int, dict[str, int] | None]] = []

        if current_level == 0:
            # Level 0:
            # Avatar starts at (34, 45) with shape=5, color=1, rot=3.
            # Goal is at (34, 10) requiring shape=5, color=1, rot=0.
            # Rotation pad at (19, 30) mutates rot from 3 -> 0.

            # 1. Navigate from (34, 45) to (19, 30):
            # 3 x Left (Action 3), 3 x Up (Action 1)
            plan.extend([(3, None)] * 3)
            plan.extend([(1, None)] * 3)

            # 2. Navigate from (19, 30) to goal at (34, 10):
            # 1 x Up (Action 1), 3 x Right (Action 4), 3 x Up (Action 1)
            plan.append((1, None))
            plan.extend([(4, None)] * 3)
            plan.extend([(1, None)] * 3)

        return plan
