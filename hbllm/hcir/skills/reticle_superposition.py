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

        # Dominant background color
        bg = int(np.bincount(grid.flatten().astype(np.int64)).argmax())
        if np.sum(grid == bg) < 3000:
            return False

        border_cols = set(grid[0, :]) | set(grid[-1, :]) | set(grid[:, 0]) | set(grid[:, -1])
        unique_colors, counts = np.unique(grid, return_counts=True)
        color_map = dict(zip(unique_colors, counts))

        dot_candidates = [c for c, count in color_map.items() if c != bg and count <= 4]
        has_dot = len(dot_candidates) >= 1

        crosshair_colors = [
            c
            for c, count in color_map.items()
            if c != bg and c not in dot_candidates and c not in border_cols and count >= 15
        ]
        has_crosshairs = len(crosshair_colors) >= 1
        if not (has_dot and has_crosshairs):
            return False

        # Genuine reticle superposition puzzles contain crosshair colors with isolated alignment dots
        has_isolated_targets = False
        for c in crosshair_colors:
            pts = np.argwhere(grid == c)
            for r, col in pts:
                dists = np.max(np.abs(pts - np.array([r, col])), axis=1)
                neighbors = np.sum((dists > 0) & (dists <= 2))
                if neighbors == 0:
                    has_isolated_targets = True
                    break
            if has_isolated_targets:
                break

        return has_isolated_targets

    @classmethod
    def plan_reticle_superposition_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute closed-loop translations and entity switches to align active reticle."""
        if grid.ndim == 3:
            grid = grid[-1]

        bg = int(np.bincount(grid.flatten().astype(np.int64)).argmax())
        border_cols = set(grid[0, :]) | set(grid[-1, :]) | set(grid[:, 0]) | set(grid[:, -1])
        unique_colors, counts = np.unique(grid, return_counts=True)
        color_map = dict(zip(unique_colors, counts))

        dot_candidates = [c for c, count in color_map.items() if c != bg and count <= 4]
        if not dot_candidates:
            return []
        dot_col = dot_candidates[0]
        dot0_pts = np.argwhere(grid == dot_col)
        if len(dot0_pts) == 0:
            return []
        dot0 = tuple(dot0_pts[0])

        crosshair_colors = [
            c
            for c, count in color_map.items()
            if c != bg and c != dot_col and c not in border_cols and count >= 15
        ]
        active_color: int | None = None
        reticle_info: dict[int, tuple[int, int]] = {}

        for c in crosshair_colors:
            pts = np.argwhere(grid == c)
            isolated: list[tuple[int, int]] = []
            crosshair: list[tuple[int, int]] = []
            for r, col in pts:
                dists = np.max(np.abs(pts - np.array([r, col])), axis=1)
                neighbors = np.sum((dists > 0) & (dists <= 2))
                if neighbors == 0:
                    isolated.append((int(r), int(col)))
                else:
                    crosshair.append((int(r), int(col)))

            if not crosshair or not isolated:
                continue

            min_r = min(r for r, _ in crosshair)
            max_r = max(r for r, _ in crosshair)
            min_c = min(col for _, col in crosshair)
            max_c = max(col for _, col in crosshair)

            if min_r <= dot0[0] <= max_r and min_c <= dot0[1] <= max_c:
                active_color = int(c)

            c_set = set(crosshair)
            best_dr: int | None = None
            best_dc: int | None = None
            for dr in range(-60, 61, 3):
                for dc in range(-60, 61, 3):
                    shifted = {(r + dr, col + dc) for r, col in c_set}
                    if all((tr, tc) in shifted for tr, tc in isolated):
                        best_dr, best_dc = dr, dc
                        break
                if best_dr is not None:
                    break

            if best_dr is not None and best_dc is not None:
                reticle_info[int(c)] = (best_dr, best_dc)

        if not reticle_info:
            return []

        all_aligned = all(dr == 0 and dc == 0 for dr, dc in reticle_info.values())
        if all_aligned:
            return []

        if active_color is None or active_color not in reticle_info:
            return [(5, None)]

        act_dr, act_dc = reticle_info[active_color]
        if act_dr == 0 and act_dc == 0:
            return [(5, None)]

        subplan: list[tuple[int, dict[str, int] | None]] = []
        step_r = act_dr // 3
        step_c = act_dc // 3

        if step_c > 0:
            subplan.extend([(4, None)] * step_c)
        elif step_c < 0:
            subplan.extend([(3, None)] * (-step_c))

        if step_r > 0:
            subplan.extend([(2, None)] * step_r)
        elif step_r < 0:
            subplan.extend([(1, None)] * (-step_r))

        subplan.append((5, None))
        return subplan
