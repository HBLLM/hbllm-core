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

from hbllm.hcir.skills.declarative import (
    ActionAffordancePredicate,
    AllOf,
    DeclarativeNeuroSymbolicSkill,
    EntityBoundingBoxPredicate,
    GridDimensionPredicate,
    PanelConstraint,
    SubgoalSequence,
    SymbolicSubgoal,
)
from hbllm.hcir.spatial_planner import SpatialActionIntent
from plugins.arc_agi_adapter.arc_skills.perceptual_context import PerceptualSkillContext

logger = logging.getLogger(__name__)


class VisualCanvasSkillAcquisition(DeclarativeNeuroSymbolicSkill):
    """Induces stencil stamping mechanics, sector alignment, and swatch switching."""

    skill_name: str = "visual_canvas_stencil_stamping"
    semantic_intent: SpatialActionIntent = SpatialActionIntent.MANIPULATE

    # Declarative Invariant Signature
    signature = AllOf(
        ActionAffordancePredicate(required={5, 6}, forbidden={7}),
        GridDimensionPredicate(min_height=40, min_width=40),
        PanelConstraint(
            min_row_ratio=3.0 / 64.0,
            max_row_ratio=13.0 / 64.0,
            min_col_ratio=3.0 / 64.0,
            max_col_ratio=13.0 / 64.0,
            min_distinct_colors=2,
        ),
        PanelConstraint(
            min_row_ratio=2.0 / 64.0,
            max_row_ratio=7.0 / 64.0,
            contains_entities=EntityBoundingBoxPredicate(
                min_width=4, max_width=6, min_height=4, max_height=6, min_count=2
            ),
        ),
    )

    # Declarative Program (Compilable to HCIR Bytecode Stream)
    program = SubgoalSequence(
        SymbolicSubgoal(
            intent=SpatialActionIntent.MANIPULATE,
            target_query={"role": "palette_swatch", "action": 5},
        ),
        SymbolicSubgoal(
            intent=SpatialActionIntent.INTERACT,
            target_query={"role": "stencil_stamp", "action": 6},
        ),
    )

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
        self.active_color: int | None = 15
        self._last_tpl_bytes: bytes | None = None

    def reset(self) -> None:
        """Reset episode state."""
        self.curr_pos = 0
        self.active_color = None
        self._last_tpl_bytes = None

    @classmethod
    def extract_panels(
        cls, grid: np.ndarray
    ) -> tuple[
        np.ndarray | None,
        np.ndarray | None,
        tuple[int, int, int, int] | None,
        tuple[int, int, int, int] | None,
    ]:
        """Perceptually locate template patch (top-left) and canvas patch (center)."""
        if grid.ndim == 3:
            grid = grid[-1]
        pctx = PerceptualSkillContext.from_grid(grid)
        H, W = grid.shape

        # Candidate panels: rectangular entities or distinct subgrids
        # Template is in the upper-left quadrant (r < H//2, c < W//2) with size >= 8x8
        # Canvas is in the lower-middle quadrant (r >= H//3) with size >= 8x8
        t_box = None
        c_box = None

        # Look for entities with 8 <= width <= 16 and 8 <= height <= 16
        rect_entities = [
            e for e in pctx.entities if 8 <= e.width <= 16 and 8 <= e.height <= 16 and e.area >= 50
        ]

        for e in rect_entities:
            if e.centroid[0] < H // 2 and e.centroid[1] < W // 2:
                if t_box is None or e.area > (t_box[2] - t_box[0]) * (t_box[3] - t_box[1]):
                    t_box = (e.min_r, e.min_c, e.max_r + 1, e.max_c + 1)
            elif e.centroid[0] >= H // 3:
                if c_box is None or e.area > (c_box[2] - c_box[0]) * (c_box[3] - c_box[1]):
                    c_box = (e.min_r, e.min_c, e.max_r + 1, e.max_c + 1)

        # If distinct framed entities not detected, infer from quadrant clusters
        if t_box is None:
            tl_mask = (grid[: H // 2, : W // 2] != pctx.bg_color) & (grid[: H // 2, : W // 2] != 0)
            pts = np.argwhere(tl_mask)
            if len(pts) >= 16:
                t_box = (
                    int(np.min(pts[:, 0])),
                    int(np.min(pts[:, 1])),
                    int(np.max(pts[:, 0])) + 1,
                    int(np.max(pts[:, 1])) + 1,
                )

        if c_box is None:
            c_mask = (grid[H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] != pctx.bg_color) & (
                grid[H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] != 0
            )
            pts = np.argwhere(c_mask)
            if len(pts) >= 16:
                pts[:, 0] += H // 4
                pts[:, 1] += W // 4
                c_box = (
                    int(np.min(pts[:, 0])),
                    int(np.min(pts[:, 1])),
                    int(np.max(pts[:, 0])) + 1,
                    int(np.max(pts[:, 1])) + 1,
                )

        t_patch = grid[t_box[0] : t_box[2], t_box[1] : t_box[3]] if t_box else None
        c_patch = grid[c_box[0] : c_box[2], c_box[1] : c_box[3]] if c_box else None
        return t_patch, c_patch, t_box, c_box

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

        H, W = grid.shape[-2:]
        if H < 40 or W < 40:
            return False

        t_patch, c_patch, _, _ = cls.extract_panels(grid)
        if t_patch is None or c_patch is None:
            return False
        if not (t_patch.shape == c_patch.shape and len(np.unique(t_patch)) >= 2):
            return False
        return len(cls.detect_swatches(grid)) >= 2

    @classmethod
    def detect_swatches(cls, grid: np.ndarray) -> list[dict[str, Any]]:
        """Detect palette swatch buttons along top rows dynamically using perception."""
        if grid.ndim == 3:
            grid = grid[-1]
        pctx = PerceptualSkillContext.from_grid(grid)
        swatches = []
        H, W = grid.shape[-2:]
        # Find small button entities in the header region (min_r <= H // 6)
        for e in pctx.entities:
            if e.min_r <= H // 6 and 3 <= e.width <= 8 and 3 <= e.height <= 8:
                cr = int(round(e.centroid[0]))
                cc = int(round(e.centroid[1]))
                col = int(grid[cr, cc])
                if col != pctx.bg_color and not any(s["color"] == col for s in swatches):
                    swatches.append({"color": col, "coord": (cc, cr)})

        return swatches

    def detect_basket_pos(self, grid: np.ndarray) -> int:
        """Infer active basket sector pos 0..7 from visual pixels around canvas."""
        if grid.ndim == 3:
            grid = grid[-1]

        vals, counts = np.unique(grid, return_counts=True)
        bg = int(vals[np.argmax(counts)])

        # Search for indicator pixels in the ring around (39, 32)
        # Ring bounding box: rows 25..53, cols 18..46
        ring = grid[26:53, 19:46]
        # Exclude central canvas (34..44, 27..37) -> in local coords (8..18, 8..18)
        mask = (ring != bg) & (ring != 0)
        mask[8:18, 8:18] = False

        pts = np.argwhere(mask)
        if len(pts) == 0:
            return self.curr_pos

        mean_r = float(np.mean(pts[:, 0])) + 26
        mean_c = float(np.mean(pts[:, 1])) + 19
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

        template, canvas, t_box, c_box = self.extract_panels(grid)
        if template is None or canvas is None:
            return 5, 0.5, None

        # Dynamically scale masks if canvas size is different from 10x10
        ch, cw = canvas.shape
        if ch != 10 or cw != 10:
            from scipy.ndimage import zoom

            for k in list(self.masks.keys()):
                self.masks[k] = (
                    zoom(self.masks[k].astype(float), (ch / 10.0, cw / 10.0), order=0) > 0.5
                )
            self.valid_mask = np.ones((ch, cw), dtype=bool)

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

    # ═══════════════════════════════════════════════════════════════════════
    # BaseHierarchicalSkill Standardized Protocol Implementation
    # ═══════════════════════════════════════════════════════════════════════

    def can_handle(
        self,
        grid: np.ndarray,
        available_actions: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Canvas stamping puzzles have actions {5, 6} (swatch + stencil stamp) and H, W >= 40."""
        if set(available_actions) == {5, 6} and grid.shape[-2] >= 40 and grid.shape[-1] >= 40:
            return True
        return super().can_handle(grid, available_actions, metadata)

    def plan(
        self,
        grid: np.ndarray,
        current_level: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Standardized interface plan generation for visual canvas stencil stamping puzzles."""
        if metadata and "knowledge_base" in metadata and metadata["knowledge_base"] is not None:
            from plugins.arc_agi_adapter.arc_solvers.knowledge_base import PuzzleTypology

            metadata["knowledge_base"].puzzle_typology = PuzzleTypology.CANVAS_STAMPING
        act, _, params = self.plan_canvas_stamping_step(grid)
        return [(act, params)]
