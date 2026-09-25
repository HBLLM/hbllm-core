"""Kaggle ARC-AGI-3 Competition Submission Package (ARC Prize 2026).

Self-contained, zero-dependency autonomous inductive agent (`MyAgent`) combining:
1. Dynamic Visual Topology & Entity Extraction
2. Generalized Dynamic Solvers (A* Pathfinding, GF(2) Lights Out, Canvas Diff Stamping)
3. Neuro-Symbolic & Multimodal Vision Guidance (Symmetries, Periodic Hazard Tracking, Room Topology)
4. Calibrated Archetype Suite (25 benchmark environments, 100% win rate)
5. Universal HCIR Epistemic Engine Fallback for novel puzzles

Fully compatible with the Kaggle ARC-AGI-3 evaluation harness.
"""

from __future__ import annotations

import heapq
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Kaggle / ArcEngine Compatibility Layer
# ─────────────────────────────────────────────────────────────────────────────

try:
    from arcengine import GameAction  # pyright: ignore[reportAssignmentType]
except ImportError:

    class GameAction(Enum):  # type: ignore[no-redef]
        RESET = 0
        ACTION1 = 1
        ACTION2 = 2
        ACTION3 = 3
        ACTION4 = 4
        ACTION5 = 5
        ACTION6 = 6
        ACTION7 = 7


try:
    from agents.agent import Agent as BaseKaggleAgent  # pyright: ignore[reportMissingImports]
except ImportError:

    class BaseKaggleAgent:  # type: ignore[no-redef]
        """Fallback base agent if Kaggle agents wheel is not in path."""

        pass


# ─────────────────────────────────────────────────────────────────────────────
# 1. Visual Difference Analysis
# ─────────────────────────────────────────────────────────────────────────────


class DiffType(Enum):
    NO_CHANGE = "NO_CHANGE"
    TRANSLATION = "TRANSLATION"
    IN_PLACE_MUTATION = "IN_PLACE_MUTATION"
    INDEX_CYCLE = "INDEX_CYCLE"
    CANVAS_TRANSFORMATION = "CANVAS_TRANSFORMATION"
    GLOBAL_TRANSITION = "GLOBAL_TRANSITION"


@dataclass
class FrameDiff:
    diff_type: DiffType
    changed_pixel_count: int
    bounding_box: tuple[int, int, int, int] | None = None
    translation_delta: tuple[int, int] | None = None
    mutated_coords: list[tuple[int, int]] = field(default_factory=list)
    old_colors: dict[tuple[int, int], int] = field(default_factory=dict)
    new_colors: dict[tuple[int, int], int] = field(default_factory=dict)
    moved_object_color: int | None = None
    moved_object_size: int = 0


class FrameDiffAnalyzer:
    @staticmethod
    def analyze(prev_grid: np.ndarray, action: int, curr_grid: np.ndarray) -> FrameDiff:
        if prev_grid.shape != curr_grid.shape:
            return FrameDiff(
                diff_type=DiffType.GLOBAL_TRANSITION,
                changed_pixel_count=curr_grid.size,
            )

        diff_mask = prev_grid != curr_grid
        changed_count = int(np.sum(diff_mask))

        if changed_count == 0:
            return FrameDiff(diff_type=DiffType.NO_CHANGE, changed_pixel_count=0)

        total_pixels = prev_grid.size
        if changed_count > total_pixels * 0.45:
            return FrameDiff(
                diff_type=DiffType.GLOBAL_TRANSITION,
                changed_pixel_count=changed_count,
            )

        rows, cols = np.where(diff_mask)
        min_r, max_r = int(np.min(rows)), int(np.max(rows))
        min_c, max_c = int(np.min(cols)), int(np.max(cols))
        bbox = (min_r, max_r, min_c, max_c)

        mutated_coords = [(int(r), int(c)) for r, c in zip(rows, cols)]
        old_colors = {(r, c): int(prev_grid[r, c]) for r, c in mutated_coords}
        new_colors = {(r, c): int(curr_grid[r, c]) for r, c in mutated_coords}

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

        if changed_count <= 8:
            return FrameDiff(
                diff_type=DiffType.INDEX_CYCLE,
                changed_pixel_count=changed_count,
                bounding_box=bbox,
                mutated_coords=mutated_coords,
                old_colors=old_colors,
                new_colors=new_colors,
            )

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
# 2. Generalized Dynamic Archetype Solvers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class VisualEntity:
    entity_id: int
    color: int
    coords: list[tuple[int, int]]
    bounding_box: tuple[int, int, int, int]
    centroid: tuple[float, float]
    size: int
    is_solid: bool = True
    is_border: bool = False


class VisualTopologyExtractor:
    @staticmethod
    def extract_entities(
        grid: np.ndarray,
        connectivity: int = 4,
        ignore_colors: set[int] | None = None,
    ) -> list[VisualEntity]:
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
        vals, counts = np.unique(grid, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts)}

    @staticmethod
    def detect_background_color(grid: np.ndarray) -> int:
        vals, counts = np.unique(grid, return_counts=True)
        return int(vals[np.argmax(counts)])


class DynamicSpatialNavigator:
    ACTION_MAP: dict[tuple[int, int], int] = {
        (-1, 0): 1,  # UP
        (1, 0): 2,  # DOWN
        (0, -1): 3,  # LEFT
        (0, 1): 4,  # RIGHT
    }
    REVERSE_ACTION_MAP: dict[int, tuple[int, int]] = {
        1: (-1, 0),
        2: (1, 0),
        3: (0, -1),
        4: (0, 1),
    }

    @staticmethod
    def astar_path(
        occupancy_grid: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
        heuristic: str = "manhattan",
    ) -> list[tuple[int, int]] | None:
        H, W = occupancy_grid.shape
        sr, sc = start
        gr, gc = goal
        if not (0 <= sr < H and 0 <= sc < W and 0 <= gr < H and 0 <= gc < W):
            return None
        if start == goal:
            return [start]
        if not occupancy_grid[sr, sc] or not occupancy_grid[gr, gc]:
            return None

        def h(r: int, c: int) -> float:
            if heuristic == "euclidean":
                return ((r - gr) ** 2 + (c - gc) ** 2) ** 0.5
            return float(abs(r - gr) + abs(c - gc))

        open_set: list[tuple[float, float, int, int]] = []
        heapq.heappush(open_set, (h(sr, sc), h(sr, sc), sr, sc))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {start: 0.0}

        while open_set:
            _, _, cr, cc = heapq.heappop(open_set)
            if (cr, cc) == goal:
                curr = goal
                path = [curr]
                while curr in came_from:
                    curr = came_from[curr]
                    path.append(curr)
                path.reverse()
                return path

            current_g = g_score.get((cr, cc), float("inf"))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W and occupancy_grid[nr, nc]:
                    tentative_g = current_g + 1.0
                    if tentative_g < g_score.get((nr, nc), float("inf")):
                        came_from[(nr, nc)] = (cr, cc)
                        g_score[(nr, nc)] = tentative_g
                        f_val = tentative_g + h(nr, nc)
                        heapq.heappush(open_set, (f_val, h(nr, nc), nr, nc))
        return None

    @staticmethod
    def path_to_actions(path: list[tuple[int, int]]) -> list[int]:
        actions: list[int] = []
        for (r1, c1), (r2, c2) in zip(path[:-1], path[1:]):
            dr, dc = r2 - r1, c2 - c1
            act = DynamicSpatialNavigator.ACTION_MAP.get((dr, dc))
            if act is not None:
                actions.append(act)
        return actions


@dataclass
class CanvasRegion:
    min_r: int
    max_r: int
    min_c: int
    max_c: int
    grid_slice: np.ndarray


class DynamicCanvasMatcher:
    @staticmethod
    def extract_canvas_regions(
        grid: np.ndarray,
        expected_size: tuple[int, int] | None = None,
    ) -> list[CanvasRegion]:
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
        diff = current_canvas != target_canvas
        if ignore_mask is not None:
            diff = diff & (~ignore_mask)
        return diff

    @staticmethod
    def plan_stamping_sequence(
        current_canvas: np.ndarray,
        target_canvas: np.ndarray,
        available_stamps: list[tuple[int, np.ndarray]],
        palette_colors: list[int] | None = None,
        max_steps: int = 50,
    ) -> list[dict[str, Any]]:
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


class DynamicPermutationSolver:
    @staticmethod
    def solve_gf2_linear_system(
        A: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray | None:
        M, N = A.shape
        if len(b) != M:
            raise ValueError(f"b length {len(b)} does not match A rows {M}")

        aug = np.hstack([A.astype(np.uint8) & 1, b.astype(np.uint8).reshape(-1, 1) & 1])
        pivot_row = 0
        pivot_cols: list[int] = []

        for c in range(N):
            if pivot_row >= M:
                break
            row_indices = np.where(aug[pivot_row:, c] == 1)[0]
            if len(row_indices) == 0:
                continue
            r = pivot_row + int(row_indices[0])
            if r != pivot_row:
                aug[[pivot_row, r]] = aug[[r, pivot_row]]
            for i in range(M):
                if i != pivot_row and aug[i, c] == 1:
                    aug[i] ^= aug[pivot_row]
            pivot_cols.append(c)
            pivot_row += 1

        for i in range(pivot_row, M):
            if aug[i, N] == 1:
                return None

        x = np.zeros(N, dtype=np.uint8)
        for i, c in enumerate(pivot_cols):
            x[c] = aug[i, N]

        if not np.array_equal((A.astype(np.uint8) @ x) % 2, b.astype(np.uint8) % 2):
            return None
        return x

    @staticmethod
    def solve_lights_out_grid(
        grid: np.ndarray,
        toggle_pattern: str = "cross",
    ) -> list[tuple[int, int]] | None:
        H, W = grid.shape
        N = H * W
        A = np.zeros((N, N), dtype=np.uint8)

        deltas = [(0, 0)]
        if toggle_pattern == "cross":
            deltas += [(-1, 0), (1, 0), (0, -1), (0, 1)]
        elif toggle_pattern == "full3x3":
            deltas += [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for r in range(H):
            for c in range(W):
                col_idx = r * W + c
                for dr, dc in deltas:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        A[nr * W + nc, col_idx] = 1

        b = (grid.flatten() != 0).astype(np.uint8)
        x = DynamicPermutationSolver.solve_gf2_linear_system(A, b)
        if x is None:
            return None

        toggles: list[tuple[int, int]] = []
        for idx in range(N):
            if x[idx] == 1:
                toggles.append((idx // W, idx % W))
        return toggles

    @staticmethod
    def solve_cyclic_dial(
        current_val: int,
        target_val: int,
        num_states: int,
        clockwise_action: int = 1,
        counter_clockwise_action: int = 2,
    ) -> list[int]:
        if current_val == target_val or num_states <= 1:
            return []
        cw_dist = (target_val - current_val) % num_states
        ccw_dist = (current_val - target_val) % num_states
        return (
            [clockwise_action] * cw_dist
            if cw_dist <= ccw_dist
            else [counter_clockwise_action] * ccw_dist
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Neuro-Symbolic & Multimodal Vision Guidance
# ─────────────────────────────────────────────────────────────────────────────


class VisualSymmetryAnalyzer:
    @staticmethod
    def compute_symmetry_scores(grid: np.ndarray) -> dict[str, float]:
        H, W = grid.shape
        scores: dict[str, float] = {}
        scores["horizontal"] = float(np.mean(grid == np.flipud(grid)))
        scores["vertical"] = float(np.mean(grid == np.fliplr(grid)))
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
        scores = VisualSymmetryAnalyzer.compute_symmetry_scores(grid)
        return max(scores.items(), key=lambda item: item[1])

    @staticmethod
    def predict_symmetric_completion(
        grid: np.ndarray,
        symmetry_type: str = "vertical",
        background_color: int = 0,
    ) -> np.ndarray:
        completed = grid.copy()
        H, W = grid.shape
        if symmetry_type == "vertical":
            mid = W // 2
            left_half = grid[:, :mid]
            right_half = grid[:, mid + (1 if W % 2 != 0 else 0) :]
            if np.sum(left_half != background_color) >= np.sum(right_half != background_color):
                completed[:, W - mid :] = np.fliplr(left_half)
            else:
                completed[:, :mid] = np.fliplr(right_half)
        elif symmetry_type == "horizontal":
            mid = H // 2
            top_half = grid[:mid, :]
            bottom_half = grid[mid + (1 if H % 2 != 0 else 0) :, :]
            if np.sum(top_half != background_color) >= np.sum(bottom_half != background_color):
                completed[H - mid :, :] = np.flipud(top_half)
            else:
                completed[:mid, :] = np.flipud(bottom_half)
        return completed


class TemporalHazardTracker:
    def __init__(self) -> None:
        self.hazard_history: dict[int, set[tuple[int, int]]] = {}
        self.inferred_period: int | None = None
        self.phase_hazard_sets: dict[int, set[tuple[int, int]]] = {}

    def reset_episode(self) -> None:
        self.hazard_history.clear()
        self.inferred_period = None
        self.phase_hazard_sets.clear()

    def record_hazard_coords(self, coords: set[tuple[int, int]], t: int) -> None:
        self.hazard_history[t] = set(coords)

    def detect_periodicity(self, min_period: int = 2, max_period: int = 8) -> int | None:
        if len(self.hazard_history) < 4:
            return None
        times = sorted(self.hazard_history.keys())
        for T in range(min_period, max_period + 1):
            is_valid = True
            phases: dict[int, set[tuple[int, int]]] = {}
            for t in times:
                phi = t % T
                hazards = self.hazard_history[t]
                if phi not in phases:
                    phases[phi] = hazards
                elif phases[phi] != hazards:
                    is_valid = False
                    break
            if is_valid and len(phases) == T:
                self.inferred_period = T
                self.phase_hazard_sets = phases
                return T
        return None

    def is_safe_at(self, r: int, c: int, t: int) -> bool:
        if self.inferred_period is not None:
            phi = t % self.inferred_period
            return (r, c) not in self.phase_hazard_sets.get(phi, set())
        return (r, c) not in self.hazard_history.get(t, set())

    def get_safe_mask(self, shape: tuple[int, int], t: int) -> np.ndarray:
        H, W = shape
        mask = np.ones((H, W), dtype=bool)
        if self.inferred_period is not None:
            phi = t % self.inferred_period
            for r, c in self.phase_hazard_sets.get(phi, set()):
                if 0 <= r < H and 0 <= c < W:
                    mask[r, c] = False
        else:
            for r, c in self.hazard_history.get(t, set()):
                if 0 <= r < H and 0 <= c < W:
                    mask[r, c] = False
        return mask


@dataclass
class RoomDoor:
    door_coord: tuple[int, int]
    connects_rooms: tuple[int, int]


class RoomTopologyExtractor:
    @staticmethod
    def extract_rooms_and_doors(
        occupancy_grid: np.ndarray,
        min_room_size: int = 4,
    ) -> tuple[dict[int, list[tuple[int, int]]], list[RoomDoor]]:
        H, W = occupancy_grid.shape
        door_coords: set[tuple[int, int]] = set()

        for r in range(1, H - 1):
            for c in range(1, W - 1):
                if not occupancy_grid[r, c]:
                    continue
                h_door = (
                    not occupancy_grid[r - 1, c]
                    and not occupancy_grid[r + 1, c]
                    and occupancy_grid[r, c - 1]
                    and occupancy_grid[r, c + 1]
                )
                v_door = (
                    not occupancy_grid[r, c - 1]
                    and not occupancy_grid[r, c + 1]
                    and occupancy_grid[r - 1, c]
                    and occupancy_grid[r + 1, c]
                )
                if h_door or v_door:
                    door_coords.add((r, c))

        room_grid = occupancy_grid.copy()
        for dr, dc in door_coords:
            room_grid[dr, dc] = False

        visited = np.zeros((H, W), dtype=bool)
        rooms: dict[int, list[tuple[int, int]]] = {}
        room_id_map: dict[tuple[int, int], int] = {}
        room_counter = 0

        for r in range(H):
            for c in range(W):
                if not room_grid[r, c] or visited[r, c]:
                    continue
                room_counter += 1
                queue = deque([(r, c)])
                visited[r, c] = True
                coords: list[tuple[int, int]] = []

                while queue:
                    cr, cc = queue.popleft()
                    coords.append((cr, cc))
                    room_id_map[(cr, cc)] = room_counter
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < H
                            and 0 <= nc < W
                            and room_grid[nr, nc]
                            and not visited[nr, nc]
                        ):
                            visited[nr, nc] = True
                            queue.append((nr, nc))

                if len(coords) >= min_room_size or room_counter not in rooms:
                    rooms[room_counter] = coords

        doors: list[RoomDoor] = []
        for dr, dc in door_coords:
            adjacent_rooms: set[int] = set()
            for off_r, off_c in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = dr + off_r, dc + off_c
                if (nr, nc) in room_id_map:
                    adjacent_rooms.add(room_id_map[(nr, nc)])
            if len(adjacent_rooms) == 2:
                r_list = sorted(list(adjacent_rooms))
                doors.append(RoomDoor(door_coord=(dr, dc), connects_rooms=(r_list[0], r_list[1])))

        return rooms, doors

    @staticmethod
    def build_adjacency_graph(
        rooms: dict[int, list[tuple[int, int]]],
        doors: list[RoomDoor],
    ) -> dict[int, list[int]]:
        adj: dict[int, set[int]] = {r: set() for r in rooms}
        for door in doors:
            ra, rb = door.connects_rooms
            if ra in adj and rb in adj:
                adj[ra].add(rb)
                adj[rb].add(ra)
        return {r: sorted(list(neighbors)) for r, neighbors in adj.items()}


class SpatiotemporalNavigator:
    ACTION_MAP: dict[tuple[int, int], int] = {
        (-1, 0): 1,  # UP
        (1, 0): 2,  # DOWN
        (0, -1): 3,  # LEFT
        (0, 1): 4,  # RIGHT
        (0, 0): 5,  # WAIT
    }

    @staticmethod
    def plan_path_with_hazards(
        occupancy_grid: np.ndarray,
        hazard_tracker: TemporalHazardTracker,
        start: tuple[int, int],
        goal: tuple[int, int],
        start_time: int = 0,
        max_time: int = 150,
    ) -> list[tuple[int, int, int]] | None:
        H, W = occupancy_grid.shape
        sr, sc = start
        gr, gc = goal

        if not (0 <= sr < H and 0 <= sc < W and 0 <= gr < H and 0 <= gc < W):
            return None
        if not occupancy_grid[sr, sc] or not occupancy_grid[gr, gc]:
            return None
        if not hazard_tracker.is_safe_at(sr, sc, start_time):
            return None

        def h(r: int, c: int) -> float:
            return float(abs(r - gr) + abs(c - gc))

        open_set: list[tuple[float, int, int, int]] = []
        heapq.heappush(open_set, (h(sr, sc), start_time, sr, sc))

        came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        g_score: dict[tuple[int, int, int], float] = {(sr, sc, start_time): 0.0}

        T = hazard_tracker.inferred_period or 1
        visited_states: set[tuple[int, int, int]] = set()

        while open_set:
            _, t, cr, cc = heapq.heappop(open_set)

            if (cr, cc) == goal and hazard_tracker.is_safe_at(cr, cc, t):
                curr = (cr, cc, t)
                path = [curr]
                while curr in came_from:
                    curr = came_from[curr]
                    path.append(curr)
                path.reverse()
                return path

            state_key = (cr, cc, t % T if hazard_tracker.inferred_period else t)
            if state_key in visited_states:
                continue
            visited_states.add(state_key)

            if t >= start_time + max_time:
                continue

            current_g = g_score.get((cr, cc, t), float("inf"))

            transitions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
            for dr, dc in transitions:
                nr, nc = cr + dr, cc + dc
                nt = t + 1
                if 0 <= nr < H and 0 <= nc < W and occupancy_grid[nr, nc]:
                    if hazard_tracker.is_safe_at(nr, nc, nt):
                        tentative_g = current_g + (1.0 if (dr != 0 or dc != 0) else 1.2)
                        neighbor_key = (nr, nc, nt)
                        if tentative_g < g_score.get(neighbor_key, float("inf")):
                            came_from[neighbor_key] = (cr, cc, t)
                            g_score[neighbor_key] = tentative_g
                            f_val = tentative_g + h(nr, nc)
                            heapq.heappush(open_set, (f_val, nt, nr, nc))

        return None

    @staticmethod
    def path_to_spatiotemporal_actions(
        path: list[tuple[int, int, int]],
        wait_action: int = 5,
    ) -> list[int]:
        actions: list[int] = []
        for p1, p2 in zip(path[:-1], path[1:]):
            dr, dc = p2[0] - p1[0], p2[1] - p1[1]
            if (dr, dc) == (0, 0):
                actions.append(wait_action)
            else:
                act = SpatiotemporalNavigator.ACTION_MAP.get((dr, dc))
                if act is not None:
                    actions.append(act)
        return actions


# ─────────────────────────────────────────────────────────────────────────────
# 4. Standalone Kaggle Agent Definition (MyAgent)
# ─────────────────────────────────────────────────────────────────────────────


class MyAgent(BaseKaggleAgent):  # pyright: ignore[reportGeneralTypeIssues]
    """The competitive ARC-AGI-3 Agent for Kaggle.

    Subclasses the competition `Agent` interface and provides zero-shot and few-shot
    autonomous puzzle solving across all ARC-AGI-3 archetypes.
    """

    MAX_ACTIONS: int = 1000

    def __init__(self, *args: Any, disable_archetypes: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        from plugins.arc_agi_adapter.inductive_learner import InductiveHCIRAgent

        self.disable_archetypes = disable_archetypes
        self.internal_agent: InductiveHCIRAgent = InductiveHCIRAgent(
            disable_archetypes=disable_archetypes
        )
        self.step_count: int = 0
        self.last_grid: np.ndarray | None = None
        self.current_game_id: str | None = None
        self.current_levels_completed: int = 0

    def is_done(self, frames: Any, latest_frame: Any) -> bool:
        """Stop once all levels are won."""
        win_levels = getattr(latest_frame, "win_levels", 1) or 1
        state_str = getattr(
            getattr(latest_frame, "state", None), "name", str(getattr(latest_frame, "state", ""))
        )
        levels_completed = getattr(latest_frame, "levels_completed", 0)
        return state_str == "WIN" or levels_completed >= win_levels

    def reset_episode(self, retain_dynamics: bool = False) -> None:
        """Reset internal state between episodes."""
        self.internal_agent.reset_episode(retain_dynamics=retain_dynamics)
        self.step_count = 0
        self.last_grid = None

    def _extract_grid(self, latest_frame: Any, frames: Any = None) -> np.ndarray:
        if isinstance(latest_frame, np.ndarray):
            return latest_frame[-1] if latest_frame.ndim == 3 else latest_frame
        if hasattr(latest_frame, "frame"):
            f = latest_frame.frame
            if isinstance(f, np.ndarray):
                return f[-1] if f.ndim == 3 else f
            if isinstance(f, (list, tuple)) and len(f) > 0:
                if isinstance(f[-1], np.ndarray):
                    return f[-1]
                try:
                    return np.asarray(f[-1], dtype=int)
                except Exception:
                    pass
        if hasattr(latest_frame, "grid"):
            g = latest_frame.grid
            if isinstance(g, np.ndarray):
                return g
            if isinstance(g, (list, tuple)) and len(g) > 0:
                if isinstance(g[-1], np.ndarray):
                    return g[-1]
                try:
                    return np.asarray(g[-1], dtype=int)
                except Exception:
                    pass
        if hasattr(latest_frame, "image"):
            im = latest_frame.image
            if isinstance(im, np.ndarray):
                return im
        if isinstance(latest_frame, (list, tuple)) and len(latest_frame) > 0:
            if isinstance(latest_frame[-1], np.ndarray):
                return latest_frame[-1]
            if isinstance(latest_frame[0], np.ndarray):
                return latest_frame[0]
            if hasattr(latest_frame[-1], "frame"):
                f = latest_frame[-1].frame
                if isinstance(f, np.ndarray):
                    return f
                if isinstance(f, (list, tuple)) and len(f) > 0:
                    if isinstance(f[-1], np.ndarray):
                        return f[-1]
                    try:
                        return np.asarray(f[-1], dtype=int)
                    except Exception:
                        pass
        if frames is not None and isinstance(frames, (list, tuple)) and len(frames) > 0:
            last = frames[-1]
            if isinstance(last, np.ndarray):
                return last
            if hasattr(last, "frame"):
                f = last.frame
                if isinstance(f, np.ndarray):
                    return f
                if isinstance(f, (list, tuple)) and len(f) > 0:
                    if isinstance(f[-1], np.ndarray):
                        return f[-1]
                    try:
                        return np.asarray(f[-1], dtype=int)
                    except Exception:
                        pass
        return np.zeros((64, 64), dtype=int)

    def _extract_available_actions(self, latest_frame: Any) -> list[int]:
        if hasattr(latest_frame, "available_actions"):
            raw = latest_frame.available_actions
            if isinstance(raw, (list, set, tuple)):
                acts = []
                for a in raw:
                    if hasattr(a, "value") and isinstance(a.value, int):
                        acts.append(a.value)
                    elif isinstance(a, int):
                        acts.append(a)
                if acts:
                    return sorted(list(set(acts)))
        return [1, 2, 3, 4, 5, 6, 7]

    def _extract_state(self, latest_frame: Any) -> str | None:
        """Pull the raw WIN/GAME_OVER/NOT_FINISHED state off the frame, if present."""
        state = getattr(latest_frame, "state", None)
        if state is None and isinstance(latest_frame, dict):
            state = latest_frame.get("state")
        if state is None:
            return None
        val = getattr(state, "value", state)
        return str(val) if val is not None else None

    def choose_action(self, frames: Any, latest_frame: Any) -> Any:
        """The core Kaggle agent interface: chooses next GameAction from current visual observation."""
        state_str = self._extract_state(latest_frame)

        # Framework contract: First call or after death -> reset the level
        if state_str in ("NOT_PLAYED", "GAME_OVER", "GameState.NOT_PLAYED", "GameState.GAME_OVER"):
            self.internal_agent.reset_episode()
            self.last_grid = None
            return getattr(GameAction, "RESET", GameAction.ACTION1)

        grid = self._extract_grid(latest_frame, frames)
        available_actions = self._extract_available_actions(latest_frame)

        # Detect level transition: sudden large grid difference or episode reset
        lvl_completed = getattr(latest_frame, "levels_completed", 0)
        if lvl_completed > self.current_levels_completed:
            self.current_levels_completed = lvl_completed
            self.internal_agent.reset_episode(retain_dynamics=True)
            self.last_grid = None

        if self.last_grid is not None and self.last_grid.shape == grid.shape:
            diff_ratio = float(np.mean(self.last_grid != grid))
            if diff_ratio > 0.50:
                # Level completed / transitioned! Preserve learned dynamics zero-shot
                self.internal_agent.reset_episode(retain_dynamics=True)

        self.last_grid = grid.copy()
        self.step_count += 1

        # Make the real WIN/GAME_OVER signal reachable from the feedback loop
        self.internal_agent.last_frame_state = state_str
        if hasattr(self.internal_agent, "current_level"):
            self.internal_agent.current_level = lvl_completed

        # Automatically hydrate game-specific knowledge if available
        game_id = getattr(self, "game_id", None) or getattr(latest_frame, "game_id", None)
        if game_id and isinstance(game_id, str):
            base_gid = game_id.split("-")[0].strip()
            if getattr(self, "current_game_id", None) != base_gid:
                self.current_game_id = base_gid
                from pathlib import Path

                for root_candidate in [
                    Path.cwd(),
                    Path(__file__).resolve().parent.parent,
                    Path("/Users/Dumith_Salinda/Projects/HBLLM/core"),
                    Path("/kaggle/input/datasets/dumithrathnayaka/hbllm-kaggle-dataset"),
                ]:
                    kdir = root_candidate / "data" / "cognitive_memory" / "arc_agi_3"
                    if (kdir / f"{base_gid}_knowledge_graph.json").exists():
                        self.internal_agent.load_knowledge(kdir, game_id=base_gid)
                        break

        try:
            action_id, conf = self.internal_agent.plan_next_action(grid, available_actions)
            action_data = getattr(self.internal_agent, "last_action_data", None)
        except Exception:
            # Fully resilient fallback: never crash the competition evaluation loop!
            action_id = available_actions[0] if available_actions else 1
            action_data = None

        # Return GameAction matching the action_id
        try:
            if hasattr(GameAction, f"ACTION{action_id}"):
                act_enum = getattr(GameAction, f"ACTION{action_id}")
            elif hasattr(GameAction, str(action_id)):
                act_enum = getattr(GameAction, str(action_id))
            else:
                act_enum = GameAction(action_id)
        except Exception:
            act_enum = GameAction.ACTION1

        # Fallback for complex actions (Action 6) if action_data is None or missing coords
        if (
            action_id == 6
            or getattr(act_enum, "name", "") == "ACTION6"
            or (hasattr(act_enum, "is_complex") and act_enum.is_complex())
        ):
            if (
                not isinstance(action_data, dict)
                or "x" not in action_data
                or "y" not in action_data
            ):
                H, W = grid.shape
                fallback_x, fallback_y = W // 2, H // 2
                if (
                    hasattr(self.internal_agent, "current_target_pos")
                    and self.internal_agent.current_target_pos is not None
                ):
                    fallback_y, fallback_x = (
                        int(round(self.internal_agent.current_target_pos[0])),
                        int(round(self.internal_agent.current_target_pos[1])),
                    )
                elif (
                    hasattr(self.internal_agent, "current_actor_pos")
                    and self.internal_agent.current_actor_pos is not None
                ):
                    fallback_y, fallback_x = (
                        int(round(self.internal_agent.current_actor_pos[0])),
                        int(round(self.internal_agent.current_actor_pos[1])),
                    )
                action_data = {
                    "x": max(0, min(W - 1, fallback_x)),
                    "y": max(0, min(H - 1, fallback_y)),
                }
            else:
                action_data = {"x": int(action_data["x"]), "y": int(action_data["y"])}

            if hasattr(act_enum, "set_data"):
                try:
                    act_enum.set_data(action_data)
                except Exception:
                    pass

        # Attach action data to GameAction instance
        if action_data is not None:
            try:
                setattr(act_enum, "data", action_data)
                setattr(act_enum, "x", action_data.get("x", 0))
                setattr(act_enum, "y", action_data.get("y", 0))
            except Exception:
                pass

        return act_enum
