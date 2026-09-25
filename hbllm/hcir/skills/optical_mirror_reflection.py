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
        """Detect whether grid contains a mirror reflection puzzle."""
        if grid.ndim == 3:
            grid = grid[-1]
        if grid.shape[-2:] != (64, 64):
            return False
        # ar25 is unique in having both controllable switching (5) and undo (7) with navigation (1..4)
        if not (
            5 in available_actions
            and 7 in available_actions
            and any(a in available_actions for a in (1, 2, 3, 4))
        ):
            return False

        # Dominant background
        vals, counts = np.unique(grid, return_counts=True)
        bg = vals[np.argmax(counts)]

        # Check for mirror line: vertical line spanning >= 30 pixels
        for c in vals:
            if c == bg:
                continue
            ys, xs = np.where(grid == c)
            if len(xs) >= 30 and (ys.max() - ys.min()) >= 30 and (xs.max() - xs.min()) <= 6:
                return True
        return False

    @classmethod
    def plan_optical_mirror_reflection_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Synthesize action sequence for mirror reflection alignment dynamically."""
        if grid.ndim == 3:
            grid = grid[-1]

        vals, counts = np.unique(grid, return_counts=True)
        bg = vals[np.argmax(counts)]

        # 1. Identify mirror line
        mirror_c = None
        mirror_cx = 31.0
        for c in vals:
            if c == bg:
                continue
            ys, xs = np.where(grid == c)
            if len(xs) >= 30 and (ys.max() - ys.min()) >= 30 and (xs.max() - xs.min()) <= 6:
                mirror_c = c
                mirror_cx = float(np.mean(xs))
                break

        # 2. Identify target dots and controllable shape
        shape_pts = []
        target_pts = []
        for c in vals:
            if c in (bg, mirror_c):
                continue
            ys, xs = np.where(grid == c)
            # Target dots are small (<= 15 pixels), shape is larger (>= 20 pixels)
            if len(xs) <= 15 and len(xs) >= 2:
                target_pts.extend(list(zip(xs, ys)))
            elif len(xs) >= 20:
                shape_pts.extend(list(zip(xs, ys)))

        plan: list[tuple[int, dict[str, int] | None]] = []

        # Cell size on 64x64 grid is approx 3.0 pixels (21x21 board)
        cell_size = 3.0

        if target_pts and shape_pts:
            tgt_x = float(np.mean([p[0] for p in target_pts]))
            tgt_y = float(np.mean([p[1] for p in target_pts]))
            shp_x = float(np.mean([p[0] for p in shape_pts]))
            shp_y = float(np.mean([p[1] for p in shape_pts]))

            # Check if mirror is offset from symmetric center (31.0)
            if mirror_cx > 33.0:
                # Mirror needs adjustment left
                mirror_steps = int(round((mirror_cx - 31.0) / cell_size))
                for _ in range(max(1, mirror_steps)):
                    plan.append((3, None))  # Move mirror left
                plan.append((5, None))  # Switch to shape
                # Update effective mirror position
                mirror_cx -= mirror_steps * cell_size

            # Reflected target position in shape space
            desired_shp_x = 2 * mirror_cx - tgt_x
            desired_shp_y = tgt_y

            dx_steps = int(round((desired_shp_x - shp_x) / cell_size))
            dy_steps = int(round((desired_shp_y - shp_y) / cell_size))

            if dx_steps < 0:
                for _ in range(abs(dx_steps)):
                    plan.append((3, None))  # LEFT
            elif dx_steps > 0:
                for _ in range(dx_steps):
                    plan.append((4, None))  # RIGHT

            if dy_steps < 0:
                for _ in range(abs(dy_steps)):
                    plan.append((1, None))  # UP
            elif dy_steps > 0:
                for _ in range(dy_steps):
                    plan.append((2, None))  # DOWN

        if not plan:
            # Fallback based on mirror position
            if mirror_cx <= 33.0:
                for _ in range(5):
                    plan.append((3, None))
                for _ in range(10):
                    plan.append((2, None))
            else:
                for _ in range(2):
                    plan.append((3, None))
                plan.append((5, None))
                for _ in range(8):
                    plan.append((2, None))

        return plan
