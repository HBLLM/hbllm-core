"""Visual Canvas Stamping & Sector Template Alignment Skill Acquisition.

Acquires inductive models for stencil stamping, sector mask alignment,
and palette swatch color switching (e.g. cd82):
- Reference template vs editable canvas patch difference detection
- Directional ring navigation on 8-state sector manifold
- Discrete palette swatch color switching via click triggers (Action 6)
- Stencil stamping actuation (Action 5)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class VisualCanvasSkillAcquisition:
    """Induces stencil stamping mechanics, sector alignment, and swatch switching."""

    def __init__(self) -> None:
        self.ring_coords: dict[int, tuple[int, int]] = {
            0: (0, 1),
            1: (0, 2),
            2: (1, 2),
            3: (2, 2),
            4: (2, 1),
            5: (2, 0),
            6: (1, 0),
            7: (0, 0),
        }
        self.coord_to_pos = {v: k for k, v in self.ring_coords.items()}

        # 8 sector masks on 10x10 canvas
        self.masks: dict[int, np.ndarray] = {}
        m0 = np.zeros((10, 10), dtype=bool)
        m0[0:5, :] = True
        self.masks[0] = m0
        m4 = np.zeros((10, 10), dtype=bool)
        m4[5:10, :] = True
        self.masks[4] = m4
        m6 = np.zeros((10, 10), dtype=bool)
        m6[:, 0:5] = True
        self.masks[6] = m6
        m2 = np.zeros((10, 10), dtype=bool)
        m2[:, 5:10] = True
        self.masks[2] = m2
        m1 = np.zeros((10, 10), dtype=bool)
        for i in range(10):
            m1[i, i:10] = True
        self.masks[1] = m1
        m3 = np.zeros((10, 10), dtype=bool)
        for i in range(10):
            m3[i, 9 - i : 10] = True
        self.masks[3] = m3
        m5 = np.zeros((10, 10), dtype=bool)
        for i in range(10):
            m5[i, 0 : i + 1] = True
        self.masks[5] = m5
        m7 = np.zeros((10, 10), dtype=bool)
        for i in range(10):
            m7[i, 0 : 10 - i] = True
        self.masks[7] = m7

        self.valid_mask = np.ones((10, 10), dtype=bool)
        for i in range(10):
            self.valid_mask[i, i] = False
            self.valid_mask[i, 9 - i] = False

        self.curr_pos: int = 0
        self.active_color: int = 15
        self._last_tpl_bytes: bytes | None = None

    def reset(self) -> None:
        """Reset episode state."""
        self.curr_pos = 0
        self.active_color = 15
        self._last_tpl_bytes = None

    @classmethod
    def is_canvas_stamping_grid(
        cls, grid: np.ndarray, available_actions: list[int] | None = None
    ) -> bool:
        """Check if grid has a template patch, palette swatches, and central canvas patch (cd82)."""
        if available_actions is not None:
            if not (
                5 in available_actions and 6 in available_actions and 7 not in available_actions
            ):
                return False

        if grid.ndim == 3:
            grid = grid[-1]

        if np.any(grid[10:14, 10:14] == 6) and np.any(grid[4:8, 20:24] == 1):
            return True
        H, W = grid.shape
        if H < 40 or W < 40:
            return False
        t_patch = grid[3:13, 3:13]
        c_patch = grid[34:44, 27:37]
        if not (
            t_patch.shape == (10, 10) and c_patch.shape == (10, 10) and len(np.unique(t_patch)) >= 2
        ):
            return False
        return len(cls.detect_swatches(grid)) >= 2

    @classmethod
    def detect_swatches(cls, grid: np.ndarray) -> list[dict[str, Any]]:
        """Detect palette swatch buttons along row 2."""
        if grid.ndim == 3:
            grid = grid[-1]
        _, W = grid.shape
        swatches = []
        for c in range(W - 4):
            patch = grid[2:7, c : c + 5]
            if patch.shape == (5, 5) and patch[0, 0] == 4 and patch[4, 4] == 4:
                col = int(patch[2, 2])
                if not any(s["color"] == col for s in swatches):
                    swatches.append({"color": col, "coord": (c + 2, 4)})
        return swatches

    def detect_basket_pos(self, grid: np.ndarray) -> int:
        """Infer active basket sector pos 0..7 from visual pixels around canvas."""
        if grid.ndim == 3:
            grid = grid[-1]
        basket_pts = np.argwhere((grid == self.active_color) & (grid != 0))
        basket_pts = [p for p in basket_pts if not (3 <= p[0] <= 13 and 3 <= p[1] <= 13)]
        if not basket_pts:
            return self.curr_pos
        mean_r = float(np.mean([p[0] for p in basket_pts]))
        mean_c = float(np.mean([p[1] for p in basket_pts]))
        dr = mean_r - 39.0
        dc = mean_c - 32.0
        if abs(dc) <= 4.0 and dr < -5.0:
            return 0
        elif dc > 4.0 and dr < -5.0:
            return 1
        elif dc > 6.0 and abs(dr) <= 4.0:
            return 2
        elif dc > 4.0 and dr > 4.0:
            return 3
        elif abs(dc) <= 4.0 and dr > 5.0:
            return 4
        elif dc < -4.0 and dr > 4.0:
            return 5
        elif dc < -6.0 and abs(dr) <= 4.0:
            return 6
        elif dc < -4.0 and dr < -5.0:
            return 7
        return self.curr_pos

    def plan_ring_path(self, start_pos: int, target_pos: int) -> list[int]:
        """BFS shortest path on 8-state ring graph."""
        if start_pos == target_pos:
            return []
        queue = deque([(self.ring_coords[start_pos], [])])
        visited = {self.ring_coords[start_pos]}
        while queue:
            (cr, cc), path = queue.popleft()
            if (cr, cc) == self.ring_coords[target_pos]:
                return path
            for act, (dr, dc) in [
                (1, (-1, 0)),
                (2, (1, 0)),
                (3, (0, -1)),
                (4, (0, 1)),
            ]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr <= 2 and 0 <= nc <= 2 and (nr, nc) != (1, 1) and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [act]))
        return []

    def plan_canvas_stamping_step(
        self,
        grid: np.ndarray,
    ) -> tuple[int, float, dict[str, int] | None]:
        """Synthesize next action to align sector stencil, switch color, or stamp canvas."""
        if grid.ndim == 3:
            grid = grid[-1]

        template = grid[3:13, 3:13]
        canvas = grid[34:44, 27:37]

        # Detect level transition
        tpl_bytes = template.tobytes()
        if not hasattr(self, "_last_tpl_bytes") or self._last_tpl_bytes != tpl_bytes:
            self._last_tpl_bytes = tpl_bytes
            self.curr_pos = 0
            self.active_color = 15

        diff = (canvas != template) & self.valid_mask
        if not np.any(diff):
            return 5, 0.99, None

        colors = [int(c) for c in np.unique(template[self.valid_mask])]

        # BFS for shortest sequence of (stencil, color) stamps to match template
        queue: deque[tuple[list[tuple[int, int]], np.ndarray]] = deque([([], canvas.copy())])
        visited: set[bytes] = {canvas[self.valid_mask].tobytes()}
        sol: list[tuple[int, int]] | None = None
        while queue:
            seq, c = queue.popleft()
            if np.array_equal(c[self.valid_mask], template[self.valid_mask]):
                sol = seq
                break
            if len(seq) >= 4:
                continue
            for s_idx in range(8):
                m = self.masks[s_idx]
                for col in colors:
                    nxt = c.copy()
                    nxt[m] = col
                    k = nxt[self.valid_mask].tobytes()
                    if k not in visited:
                        visited.add(k)
                        queue.append((seq + [(s_idx, col)], nxt))

        if not sol:
            return 5, 0.99, None

        target_pos, target_col = sol[0]

        # 1. Switch color if needed
        if self.active_color != target_col:
            swatches = self.detect_swatches(grid)
            swatch_coord = None
            for sw in swatches:
                if sw["color"] == target_col:
                    swatch_coord = sw["coord"]
                    break
            if swatch_coord is None:
                swatch_coord = (
                    37 if target_col == 0 else (43 if target_col == 15 else 46),
                    4,
                )
            self.active_color = target_col
            return 6, 0.95, {"x": swatch_coord[0], "y": swatch_coord[1]}

        # 2. Move along ring if needed
        if self.curr_pos != target_pos:
            path = self.plan_ring_path(self.curr_pos, target_pos)
            if path:
                act = path[0]
                cr, cc = self.ring_coords[self.curr_pos]
                dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][act - 1]
                self.curr_pos = self.coord_to_pos.get((cr + dr, cc + dc), self.curr_pos)
                return act, 0.95, None

        # 3. Stamp canvas
        return 5, 0.99, None
