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

        # Downsample 64x64 grid to 21x21 board
        board = np.zeros((21, 21), dtype=int)
        for r in range(21):
            for c in range(21):
                cell = grid[r * 3 : (r + 1) * 3, c * 3 : (c + 1) * 3]
                u, uc = np.unique(cell, return_counts=True)
                board[r, c] = u[np.argmax(uc)]

        vals, counts = np.unique(board, return_counts=True)
        bg = vals[np.argmax(counts)]

        # Mirror: vertical line of single non-bg color spanning 21 rows
        mirror_col = None
        for c in range(21):
            col_vals = board[:, c]
            if len(np.unique(col_vals)) == 1 and col_vals[0] != bg:
                mirror_col = c
                break

        if mirror_col is None:
            return []

        shape_pts = set(zip(*np.where(board == 5)))
        target_pts = set(zip(*np.where(board == 11)))

        if not shape_pts or not target_pts:
            return []

        min_tr = min(r for r, c in target_pts)
        min_sr = min(r for r, c in shape_pts)
        dr = min_tr - min_sr

        allowed_dm = [0] if current_level == 0 else list(range(-5, 6))

        best_plan: tuple[int, int, int] | None = None
        best_cost = 9999

        for dm in allowed_dm:
            new_m = mirror_col + dm
            if not (0 <= new_m < 21):
                continue
            for dc in range(-20, 21):
                valid = True
                for sr, sc in shape_pts:
                    nr = sr + dr
                    nc = sc + dc
                    if not (0 <= nr < 21 and 0 <= nc < 21):
                        valid = False
                        break
                    refl_c = 2 * new_m - nc
                    if (nr, refl_c) not in target_pts:
                        valid = False
                        break
                if valid:
                    cost = abs(dm) + abs(dc) + abs(dr) + (1 if dm != 0 else 0)
                    if cost < best_cost:
                        best_cost = cost
                        best_plan = (dm, dc, dr)

        if best_plan is None:
            return []

        dm, dc, dr = best_plan
        actions: list[tuple[int, dict[str, int] | None]] = []
        if dm < 0:
            actions.extend([(3, None)] * (-dm))
        elif dm > 0:
            actions.extend([(4, None)] * dm)

        if current_level >= 1:
            actions.append((5, None))

        if dc < 0:
            actions.extend([(3, None)] * (-dc))
        elif dc > 0:
            actions.extend([(4, None)] * dc)

        if dr < 0:
            actions.extend([(1, None)] * (-dr))
        elif dr > 0:
            actions.extend([(2, None)] * dr)

        return actions
