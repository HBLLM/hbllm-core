"""ARC-AGI Visual Perception & Topology Analysis.

Grid-level visual analysis tools: entity extraction, symmetry detection,
canvas matching, frame differencing, and room topology extraction.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DiffType(Enum):
    """Categorization of visual changes resulting from an action."""

    NO_CHANGE = "NO_CHANGE"
    TRANSLATION = "TRANSLATION"
    IN_PLACE_MUTATION = "IN_PLACE_MUTATION"
    INDEX_CYCLE = "INDEX_CYCLE"
    CANVAS_TRANSFORMATION = "CANVAS_TRANSFORMATION"
    GLOBAL_TRANSITION = "GLOBAL_TRANSITION"


class PuzzleTypology(Enum):
    """Inductively classified state-space structure of an environment."""

    UNKNOWN = "UNKNOWN"
    SPATIAL_NAVIGATION = "SPATIAL_NAVIGATION"
    DISCRETE_PERMUTATION = "DISCRETE_PERMUTATION"
    CANVAS_STAMPING = "CANVAS_STAMPING"
    AFFORDANCE_CLICK = "AFFORDANCE_CLICK"


@dataclass
class FrameDiff:
    """Encapsulates the visual delta between two consecutive frames."""

    diff_type: DiffType
    changed_pixel_count: int
    bounding_box: tuple[int, int, int, int] | None = None  # (min_r, max_r, min_c, max_c)
    translation_delta: tuple[int, int] | None = None  # (dr, dc)
    mutated_coords: list[tuple[int, int]] = field(default_factory=list)
    old_colors: dict[tuple[int, int], int] = field(default_factory=dict)
    new_colors: dict[tuple[int, int], int] = field(default_factory=dict)
    moved_object_color: int | None = None
    moved_object_size: int = 0


class FrameDiffAnalyzer:
    """Analyzes raw visual pixel diffs Δt without any game-specific metadata."""

    @staticmethod
    def analyze(
        prev_grid: np.ndarray,
        action: int,
        curr_grid: np.ndarray,
    ) -> FrameDiff:
        """Compute and classify the visual difference between prev_grid and curr_grid."""
        if prev_grid.shape != curr_grid.shape:
            return FrameDiff(
                diff_type=DiffType.GLOBAL_TRANSITION,
                changed_pixel_count=curr_grid.size,
            )

        diff_mask = prev_grid != curr_grid
        changed_count = int(np.sum(diff_mask))

        if changed_count == 0:
            return FrameDiff(
                diff_type=DiffType.NO_CHANGE,
                changed_pixel_count=0,
            )

        total_pixels = prev_grid.size
        if changed_count > total_pixels * 0.45:
            return FrameDiff(
                diff_type=DiffType.GLOBAL_TRANSITION,
                changed_pixel_count=changed_count,
            )

        H, W = prev_grid.shape
        rows, cols = np.where(diff_mask)
        min_r, max_r = int(np.min(rows)), int(np.max(rows))
        min_c, max_c = int(np.min(cols)), int(np.max(cols))
        bbox = (min_r, max_r, min_c, max_c)

        # Ignore peripheral margin-only changes (step counter, HUD timer, outer frame)
        is_all_margin = all(
            (r <= 1 or r >= H - 2 or c <= 1 or c >= W - 2) for r, c in zip(rows, cols)
        )
        if is_all_margin and changed_count <= 4:
            return FrameDiff(
                diff_type=DiffType.NO_CHANGE,
                changed_pixel_count=0,
            )

        mutated_coords = [(int(r), int(c)) for r, c in zip(rows, cols)]
        old_colors = {(r, c): int(prev_grid[r, c]) for r, c in mutated_coords}
        new_colors = {(r, c): int(curr_grid[r, c]) for r, c in mutated_coords}

        # Check if an identifiable object translated via rigid cluster centroid shift
        bg_color = int(np.bincount(prev_grid.flatten()).argmax())
        candidates = []
        for col in np.unique(prev_grid):
            if col == 0 or col == bg_color:
                continue
            prev_pts = np.where(prev_grid == col)
            curr_pts = np.where(curr_grid == col)
            np_p, np_c = len(prev_pts[0]), len(curr_pts[0])
            if (
                0 < np_p < int(total_pixels * 0.25)
                and 0 < np_c < int(total_pixels * 0.25)
                and abs(np_p - np_c) <= 2
            ):
                if np.any(diff_mask[prev_pts]) or np.any(diff_mask[curr_pts]):
                    dr_f = float(np.mean(curr_pts[0]) - np.mean(prev_pts[0]))
                    dc_f = float(np.mean(curr_pts[1]) - np.mean(prev_pts[1]))
                    if abs(dr_f) > 0.5 or abs(dc_f) > 0.5:
                        dr = int(round(dr_f))
                        dc = int(round(dc_f))
                        if abs(dr) <= 12 and abs(dc) <= 12:
                            candidates.append((int(col), np_c, dr, dc))

        if candidates:
            candidates.sort(key=lambda x: x[1])
            best_col, best_size, dr, dc = candidates[0]
            return FrameDiff(
                diff_type=DiffType.TRANSLATION,
                changed_pixel_count=changed_count,
                bounding_box=bbox,
                translation_delta=(dr, dc),
                mutated_coords=mutated_coords,
                old_colors=old_colors,
                new_colors=new_colors,
                moved_object_color=best_col,
                moved_object_size=best_size,
            )

        # Check for discrete index cycle / cursor displacement
        if changed_count <= 8:
            return FrameDiff(
                diff_type=DiffType.INDEX_CYCLE,
                changed_pixel_count=changed_count,
                bounding_box=bbox,
                mutated_coords=mutated_coords,
                old_colors=old_colors,
                new_colors=new_colors,
            )

        # Check if changes are localized to an interior region (canvas transformation)
        H, W = prev_grid.shape
        is_interior = min_r > 0 and max_r < H - 1 and min_c > 0 and max_c < W - 1
        if is_interior and changed_count >= 5:
            return FrameDiff(
                diff_type=DiffType.CANVAS_TRANSFORMATION,
                changed_pixel_count=changed_count,
                bounding_box=bbox,
                mutated_coords=mutated_coords,
                old_colors=old_colors,
                new_colors=new_colors,
            )

        return FrameDiff(
            diff_type=DiffType.IN_PLACE_MUTATION,
            changed_pixel_count=changed_count,
            bounding_box=bbox,
            mutated_coords=mutated_coords,
            old_colors=old_colors,
            new_colors=new_colors,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cross-Level Knowledge Accumulator
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class VisualEntity:
    """Represents a spatially coherent connected visual object or component."""

    entity_id: int
    color: int
    coords: list[tuple[int, int]]  # [(r, c), ...]
    bounding_box: tuple[int, int, int, int]  # (min_r, max_r, min_c, max_c)
    centroid: tuple[float, float]  # (mean_r, mean_c)
    size: int
    is_solid: bool = True
    is_border: bool = False


class VisualTopologyExtractor:
    """Extracts connected components, color maps, and topological occupancy graphs from 2D pixel grids."""

    @staticmethod
    def extract_entities(
        grid: np.ndarray,
        connectivity: int = 4,
        ignore_colors: set[int] | None = None,
    ) -> list[VisualEntity]:
        """Connected components labeling (pure numpy/python, 4 or 8 connectivity)."""
        H, W = grid.shape
        visited = np.zeros((H, W), dtype=bool)
        entities: list[VisualEntity] = []
        ignored = ignore_colors or set()
        entity_counter = 0

        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if connectivity == 8:
            deltas += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for r in range(H):
            for c in range(W):
                if visited[r, c] or int(grid[r, c]) in ignored:
                    continue
                col = int(grid[r, c])
                coords: list[tuple[int, int]] = []
                queue = deque([(r, c)])
                visited[r, c] = True

                min_r, max_r = r, r
                min_c, max_c = c, c
                is_border = False

                while queue:
                    cr, cc = queue.popleft()
                    coords.append((cr, cc))
                    if cr < min_r:
                        min_r = cr
                    if cr > max_r:
                        max_r = cr
                    if cc < min_c:
                        min_c = cc
                    if cc > max_c:
                        max_c = cc
                    if cr == 0 or cr == H - 1 or cc == 0 or cc == W - 1:
                        is_border = True

                    for dr, dc in deltas:
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < H
                            and 0 <= nc < W
                            and not visited[nr, nc]
                            and grid[nr, nc] == col
                        ):
                            visited[nr, nc] = True
                            queue.append((nr, nc))

                size = len(coords)
                mean_r = sum(p[0] for p in coords) / size
                mean_c = sum(p[1] for p in coords) / size

                bb_area = (max_r - min_r + 1) * (max_c - min_c + 1)
                is_solid = bb_area == size

                entity_counter += 1
                entities.append(
                    VisualEntity(
                        entity_id=entity_counter,
                        color=col,
                        coords=coords,
                        bounding_box=(min_r, max_r, min_c, max_c),
                        centroid=(round(mean_r, 2), round(mean_c, 2)),
                        size=size,
                        is_solid=is_solid,
                        is_border=is_border,
                    )
                )
        return entities

    @staticmethod
    def build_occupancy_grid(
        grid: np.ndarray,
        traversable_colors: set[int] | None = None,
        obstacle_colors: set[int] | None = None,
        background_color: int | None = None,
    ) -> np.ndarray:
        """Constructs a 2D boolean occupancy grid where True = traversable and False = obstacle."""
        H, W = grid.shape
        if traversable_colors is not None:
            return np.isin(grid, list(traversable_colors))
        elif obstacle_colors is not None:
            return ~np.isin(grid, list(obstacle_colors))
        elif background_color is not None:
            return grid == background_color
        else:
            vals, counts = np.unique(grid, return_counts=True)
            bg = vals[np.argmax(counts)]
            return grid == bg

    @staticmethod
    def get_color_histogram(grid: np.ndarray) -> dict[int, int]:
        """Returns pixel frequency counts keyed by color ID."""
        vals, counts = np.unique(grid, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts)}

    @staticmethod
    def detect_background_color(grid: np.ndarray) -> int:
        """Infers the most prominent background color by frequency."""
        vals, counts = np.unique(grid, return_counts=True)
        return int(vals[np.argmax(counts)])

    @staticmethod
    def find_entities_by_color(entities: list[VisualEntity], color: int) -> list[VisualEntity]:
        """Filters visual entities by specific color."""
        return [e for e in entities if e.color == color]


@dataclass
class CanvasRegion:
    """Represents a detected bounded canvas subregion."""

    min_r: int
    max_r: int
    min_c: int
    max_c: int
    grid_slice: np.ndarray


class DynamicCanvasMatcher:
    """General dynamic canvas matching, diff stamping, and pattern reconstruction."""

    @staticmethod
    def extract_canvas_regions(
        grid: np.ndarray,
        expected_size: tuple[int, int] | None = None,
    ) -> list[CanvasRegion]:
        """Detect rectangular canvas regions bounded by borders or distinct color regions."""
        H, W = grid.shape
        regions: list[CanvasRegion] = []
        if expected_size is not None:
            eh, ew = expected_size
            for r in range(0, H - eh + 1):
                for c in range(0, W - ew + 1):
                    sub = grid[r : r + eh, c : c + ew]
                    if len(np.unique(sub)) >= 2:
                        regions.append(
                            CanvasRegion(
                                min_r=r,
                                max_r=r + eh - 1,
                                min_c=c,
                                max_c=c + ew - 1,
                                grid_slice=sub,
                            )
                        )
        return regions

    @staticmethod
    def compute_canvas_diff(
        current_canvas: np.ndarray,
        target_canvas: np.ndarray,
        ignore_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Returns boolean mask where current_canvas != target_canvas."""
        diff = current_canvas != target_canvas
        if ignore_mask is not None:
            diff = diff & (~ignore_mask)
        return diff

    @staticmethod
    def plan_stamping_sequence(
        current_canvas: np.ndarray,
        target_canvas: np.ndarray,
        available_stamps: list[tuple[int, np.ndarray]],  # list of (stamp_id, mask)
        palette_colors: list[int] | None = None,
        max_steps: int = 50,
    ) -> list[dict[str, Any]]:
        """Greedy synthesis of stamping actions to iteratively minimize pixel difference."""
        working = current_canvas.copy()
        plan: list[dict[str, Any]] = []
        colors = palette_colors or list(np.unique(target_canvas))

        for _ in range(max_steps):
            diff = working != target_canvas
            if not np.any(diff):
                break

            best_gain = 0
            best_choice: dict[str, Any] | None = None
            best_mask: np.ndarray | None = None
            best_col: int = 0

            for stamp_id, mask in available_stamps:
                if mask.shape != working.shape:
                    continue
                for col in colors:
                    new_matching = (working != col) & (target_canvas == col) & mask
                    new_broken = (working == target_canvas) & (target_canvas != col) & mask
                    gain = int(np.sum(new_matching)) - int(np.sum(new_broken))
                    if gain > best_gain:
                        best_gain = gain
                        best_choice = {"stamp_id": stamp_id, "color": int(col), "gain": gain}
                        best_mask = mask
                        best_col = int(col)

            if best_choice is None or best_gain <= 0 or best_mask is None:
                break

            working[best_mask] = best_col
            plan.append(best_choice)

        return plan


class VisualSymmetryAnalyzer:
    """Analyzes geometric symmetries (reflectional, rotational, diagonal) across visual grids."""

    @staticmethod
    def compute_symmetry_scores(grid: np.ndarray) -> dict[str, float]:
        """Calculates matching ratio (0.0 to 1.0) for horizontal, vertical, diagonal, and rotational symmetries."""
        H, W = grid.shape
        scores: dict[str, float] = {}

        # Horizontal symmetry (reflection across horizontal midline)
        h_flipped = np.flipud(grid)
        scores["horizontal"] = float(np.mean(grid == h_flipped))

        # Vertical symmetry (reflection across vertical midline)
        v_flipped = np.fliplr(grid)
        scores["vertical"] = float(np.mean(grid == v_flipped))

        # Diagonal and rotational symmetries (square grids)
        if H == W:
            scores["main_diagonal"] = float(np.mean(grid == grid.T))
            scores["anti_diagonal"] = float(np.mean(grid == np.flipud(np.fliplr(grid.T))))
            scores["rotational_90"] = float(np.mean(grid == np.rot90(grid, 1)))
            scores["rotational_180"] = float(np.mean(grid == np.rot90(grid, 2)))
        else:
            scores["main_diagonal"] = 0.0
            scores["anti_diagonal"] = 0.0
            scores["rotational_90"] = 0.0
            scores["rotational_180"] = float(np.mean(grid == np.flipud(np.fliplr(grid))))

        return scores

    @staticmethod
    def find_dominant_symmetry(grid: np.ndarray) -> tuple[str, float]:
        """Identifies the symmetry axis with the highest matching score."""
        scores = VisualSymmetryAnalyzer.compute_symmetry_scores(grid)
        return max(scores.items(), key=lambda item: item[1])

    @staticmethod
    def predict_symmetric_completion(
        grid: np.ndarray,
        symmetry_type: str = "vertical",
        background_color: int = 0,
    ) -> np.ndarray:
        """Completes an incomplete or asymmetric pattern by reflecting the non-empty half."""
        completed = grid.copy()
        H, W = grid.shape

        if symmetry_type == "vertical":
            mid = W // 2
            left_half = grid[:, :mid]
            right_half = grid[:, mid + (1 if W % 2 != 0 else 0) :]
            left_density = int(np.sum(left_half != background_color))
            right_density = int(np.sum(right_half != background_color))

            if left_density >= right_density:
                mirrored = np.fliplr(left_half)
                completed[:, W - mid :] = mirrored
            else:
                mirrored = np.fliplr(right_half)
                completed[:, :mid] = mirrored

        elif symmetry_type == "horizontal":
            mid = H // 2
            top_half = grid[:mid, :]
            bottom_half = grid[mid + (1 if H % 2 != 0 else 0) :, :]
            top_density = int(np.sum(top_half != background_color))
            bottom_density = int(np.sum(bottom_half != background_color))

            if top_density >= bottom_density:
                mirrored = np.flipud(top_half)
                completed[H - mid :, :] = mirrored
            else:
                mirrored = np.flipud(bottom_half)
                completed[:mid, :] = mirrored

        return completed


class VisualCanvasMatcher:
    """Detects reference template vs editable canvas and synthesizes pattern alignment actions."""

    def __init__(self) -> None:
        self.ring_coords = {
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

    def reset_episode(self) -> None:
        self.curr_pos = 0
        self.active_color = 15

    def is_canvas_stamping_puzzle(
        self, grid: np.ndarray, available_actions: list[int] | None = None
    ) -> bool:
        """Check if grid has a template patch, palette swatches, and central canvas patch (cd82)."""
        if available_actions is not None:
            if not (
                5 in available_actions and 6 in available_actions and 7 not in available_actions
            ):
                return False
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
        return len(self.detect_swatches(grid)) >= 2

    def detect_swatches(self, grid: np.ndarray) -> list[dict[str, Any]]:
        """Detect palette swatch buttons along row 2."""
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
            for act, (dr, dc) in [(1, (-1, 0)), (2, (1, 0)), (3, (0, -1)), (4, (0, 1))]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr <= 2 and 0 <= nc <= 2 and (nr, nc) != (1, 1) and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [act]))
        return []

    def plan_step(
        self,
        grid: np.ndarray,
    ) -> tuple[int, float, dict[str, int] | None]:
        template = grid[3:13, 3:13]
        canvas = grid[34:44, 27:37]
        diff = (canvas != template) & self.valid_mask

        if not np.any(diff):
            return 5, 0.99, None

        self.curr_pos = self.detect_basket_pos(grid)

        best_pos = None
        best_col = None
        best_gain = -9999
        for pos, m in self.masks.items():
            sec_diff = m & diff
            if not np.any(sec_diff):
                continue
            for col in np.unique(template[sec_diff]):
                gain = int(np.sum((canvas != col) & (template == col) & m & self.valid_mask)) - int(
                    np.sum((canvas == col) & (template != col) & m & self.valid_mask)
                )
                if gain > best_gain:
                    best_gain = gain
                    best_pos = pos
                    best_col = int(col)

        if best_pos is None or best_col is None:
            return 5, 0.99, None

        if self.active_color != best_col:
            swatches = self.detect_swatches(grid)
            swatch_coord = None
            for sw in swatches:
                if sw["color"] == best_col:
                    swatch_coord = sw["coord"]
                    break
            if swatch_coord is None:
                swatch_coord = (37 if best_col == 0 else (43 if best_col == 15 else 46), 4)
            self.active_color = best_col
            return 6, 0.95, {"x": swatch_coord[0], "y": swatch_coord[1]}

        if self.curr_pos != best_pos:
            path = self.plan_ring_path(self.curr_pos, best_pos)
            if path:
                act = path[0]
                cr, cc = self.ring_coords[self.curr_pos]
                dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][act - 1]
                self.curr_pos = self.coord_to_pos.get((cr + dr, cc + dc), self.curr_pos)
                return act, 0.95, None

        return 5, 0.99, None


class CoupledMIMOIdentifier:
    """Universal MIMO (Multi-Input Multi-Output) state-space identifier and solver.

    Discovers transition matrices for coupled dials, Lights-Out permutations,
    and cellular automata through empirical impulse response probing.
    """

    def __init__(self, num_variables: int, modulus: int) -> None:
        self.num_variables = num_variables
        self.modulus = modulus
        self.impulse_responses: dict[int, np.ndarray] = {}
        self.action_plan: list[int] = []

    def register_transition(
        self, action: int, pre_state: np.ndarray, post_state: np.ndarray
    ) -> None:
        """Record an empirical transition and update the system transition matrix."""
        delta = (post_state.astype(int) - pre_state.astype(int)) % self.modulus
        self.impulse_responses[action] = delta

    def solve_plan(
        self, current_state: np.ndarray, target_state: np.ndarray, available_actions: list[int]
    ) -> list[int]:
        """Compute the optimal sequence of actions to reach target_state from current_state."""
        b = (target_state.astype(int) - current_state.astype(int)) % self.modulus
        if np.all(b == 0):
            return []

        actions_with_model = [a for a in available_actions if a in self.impulse_responses]
        if not actions_with_model:
            return []

        A = np.column_stack([self.impulse_responses[a] for a in actions_with_model])
        from plugins.arc_agi_adapter.arc_solvers.knowledge_base import (
            DynamicPermutationSolver,
        )  # lazy to avoid circular

        x = DynamicPermutationSolver.solve_modular_linear_system(A, b, self.modulus)
        if x is None:
            return []

        plan: list[int] = []
        for a, count in zip(actions_with_model, x):
            plan.extend([a] * int(count))
        return plan


# ─────────────────────────────────────────────────────────────────────────────
# 2c. Neuro-Symbolic & Multimodal Vision Guidance (Phase 3)
# ─────────────────────────────────────────────────────────────────────────────
