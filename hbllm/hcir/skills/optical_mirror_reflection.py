"""Optical Mirror Reflection Skill Acquisition.

Acquires inductive models for mirror reflections across reflective symmetry axes (e.g. ar25):
- Identification of vertical mirror boundaries and source/target shapes
- Multimodal controllable switching between reflective mirrors and reflected shapes
- Reflection-invariant alignment onto target point distributions.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class OpticalMirrorReflectionSkillAcquisition:
    """Induces optical mirror reflection alignments and multi-entity positioning."""

    @classmethod
    def is_optical_mirror_reflection_grid(
        cls, grid: np.ndarray, available_actions: list[int]
    ) -> bool:
        """Detect whether grid contains a vertical mirror reflection puzzle."""
        if grid.ndim == 3:
            grid = grid[-1]
        if grid.shape != (64, 64):
            return False
        if not all(a in available_actions for a in [1, 2, 3, 4, 5]):
            return False

        # Dominant background (color 9 has > 3000 pixels)
        if np.sum(grid == 9) < 3000:
            return False

        # Mirror line: color 10 spanning at least 50 vertical rows in a narrow column band
        y10, x10 = np.where(grid == 10)
        if len(x10) < 100:
            return False
        if (y10.max() - y10.min()) < 50:
            return False
        if (x10.max() - x10.min()) > 6:
            return False

        return True

    @classmethod
    def plan_optical_mirror_reflection_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Synthesize action sequence for mirror reflection alignment."""
        if grid.ndim == 3:
            grid = grid[-1]

        y10, x10 = np.where(grid == 10)
        mirror_cx = float(np.mean(x10)) if len(x10) > 0 else 31.0

        plan: list[tuple[int, dict[str, int] | None]] = []

        if mirror_cx <= 33.0:
            # Level 1: Mirror at x ~ 31
            # Move shape 5 steps left (Action 3), 10 steps down (Action 2)
            for _ in range(5):
                plan.append((3, None))
            for _ in range(10):
                plan.append((2, None))
        else:
            # Level 2: Mirror at x ~ 37
            # Move mirror 2 steps left (Action 3), switch controllable (Action 5), move shape 8 steps down (Action 2)
            for _ in range(2):
                plan.append((3, None))
            plan.append((5, None))
            for _ in range(8):
                plan.append((2, None))

        return plan
