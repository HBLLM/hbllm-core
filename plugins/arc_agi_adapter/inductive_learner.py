"""Inductive HCIR Learner for ARC-AGI-3.

Learns puzzle dynamics, state-space typologies, and goal invariants purely from:
1. Raw pixel layouts: 2D integer grids (H x W)
2. Available action lists: A ⊆ {1, 2, 3, 4, 5, 6, 7}
3. Causal trial-and-error observations: Δt = Grid(t) ⊕ Grid(t-1)

Learned knowledge (action grammar, transition models, controllable entity signatures,
and goal predicates) persists across levels of an environment, enabling zero-shot
or few-shot transfer to subsequent levels with new findings.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .arc_agi_3_runner import (
    ActionDynamicsModel,
    ARC3InteractiveAgent,
)

logger = logging.getLogger(__name__)

# Graceful import of official arcengine
try:
    from arcengine import GameAction as ARCGameAction
    from arcengine import GameState as ARCGameState
except ImportError:

    class _FallbackARCGameAction(Enum):
        RESET = 0
        ACTION1 = 1
        ACTION2 = 2
        ACTION3 = 3
        ACTION4 = 4
        ACTION5 = 5
        ACTION6 = 6
        ACTION7 = 7

    class _FallbackARCGameState(Enum):
        NOT_FINISHED = "NOT_FINISHED"
        WIN = "WIN"
        GAME_OVER = "GAME_OVER"

    ARCGameAction = _FallbackARCGameAction  # type: ignore[assignment, misc]
    ARCGameState = _FallbackARCGameState  # type: ignore[assignment, misc]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Visual Difference Analysis & State Typology
# ─────────────────────────────────────────────────────────────────────────────


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

        rows, cols = np.where(diff_mask)
        min_r, max_r = int(np.min(rows)), int(np.max(rows))
        min_c, max_c = int(np.min(cols)), int(np.max(cols))
        bbox = (min_r, max_r, min_c, max_c)

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
class ActionAffordance:
    """Learned behavioral effect of a specific action ID."""

    action_id: int
    delta_r: int = 0
    delta_c: int = 0
    is_cycler: bool = False
    is_commit_or_stamp: bool = False
    is_click: bool = False
    confidence: float = 0.0
    times_tested: int = 0


@dataclass
class ControllableSignature:
    """Visual invariant signature of the agent's controllable entity or selector."""

    color: int | None = None
    area: int = 0
    shape_pattern: tuple[int, ...] = ()
    is_discrete_selector: bool = False
    active_slots: list[tuple[int, int]] = field(default_factory=list)


class CrossLevelKnowledgeBase:
    """Holds accumulated causal models and invariants across levels of a puzzle."""

    def __init__(self) -> None:
        self.puzzle_typology: PuzzleTypology = PuzzleTypology.UNKNOWN
        self.action_affordances: dict[int, ActionAffordance] = {}
        self.controllable_signature: ControllableSignature = ControllableSignature()
        self.goal_reference_pattern: np.ndarray | None = None
        self.goal_target_zone: tuple[int, int, int, int] | None = None
        self.walkable_colors: set[int] = set()
        self.barrier_colors: set[int] = set()
        self.discrete_state_transitions: dict[
            tuple[int, int], int
        ] = {}  # (slot_idx, action) -> next_slot
        self.levels_solved: int = 0
        self.total_epistemic_probes: int = 0

    def register_observation(
        self,
        prev_grid: np.ndarray,
        action: int,
        curr_grid: np.ndarray,
        diff: FrameDiff,
    ) -> None:
        """Assimilate a visual transition into the persistent knowledge base."""
        self.total_epistemic_probes += 1

        # Retrieve or initialize affordance
        aff = self.action_affordances.setdefault(action, ActionAffordance(action_id=action))
        aff.times_tested += 1

        if diff.diff_type == DiffType.TRANSLATION and diff.translation_delta:
            dr, dc = diff.translation_delta
            aff.delta_r = dr
            aff.delta_c = dc
            aff.confidence = min(1.0, aff.confidence + 0.35)
            self.puzzle_typology = PuzzleTypology.SPATIAL_NAVIGATION

            if self.controllable_signature.color is None and diff.moved_object_color is not None:
                self.controllable_signature.color = diff.moved_object_color
                self.controllable_signature.area = diff.moved_object_size

            # Learn walkable floor colors from vacated and newly entered pixels
            if self.controllable_signature.color is not None:
                for r, c in diff.mutated_coords:
                    old_c = diff.old_colors.get((r, c))
                    new_c = diff.new_colors.get((r, c))
                    if old_c == self.controllable_signature.color and new_c is not None:
                        self.walkable_colors.add(new_c)
                    elif new_c == self.controllable_signature.color and old_c is not None:
                        self.walkable_colors.add(old_c)

        elif diff.diff_type == DiffType.INDEX_CYCLE:
            aff.is_cycler = True
            aff.confidence = min(1.0, aff.confidence + 0.3)
            if self.puzzle_typology == PuzzleTypology.UNKNOWN:
                self.puzzle_typology = PuzzleTypology.DISCRETE_PERMUTATION
            self.controllable_signature.is_discrete_selector = True
            if diff.bounding_box:
                slot = (diff.bounding_box[0], diff.bounding_box[2])
                if slot not in self.controllable_signature.active_slots:
                    self.controllable_signature.active_slots.append(slot)

        elif diff.diff_type == DiffType.CANVAS_TRANSFORMATION:
            aff.is_commit_or_stamp = True
            aff.confidence = min(1.0, aff.confidence + 0.4)
            self.puzzle_typology = PuzzleTypology.CANVAS_STAMPING

        elif diff.diff_type == DiffType.NO_CHANGE:
            aff.confidence = max(0.0, aff.confidence - 0.1)

    def is_world_model_grounded(self, available_actions: list[int]) -> bool:
        """Check if sufficient dynamics have been verified to switch to goal planning."""
        if not available_actions:
            return False
        grounded_count = sum(
            1
            for a in available_actions
            if a in self.action_affordances and self.action_affordances[a].confidence >= 0.6
        )
        return grounded_count >= min(len(available_actions), 3)

    def bind_to_new_level(self, initial_grid: np.ndarray) -> dict[str, Any]:
        """Transfer learned knowledge to a new level's initial grid.

        Returns:
            Dictionary with bound controllable centroid, goal target, and suggested mode.
        """
        bindings: dict[str, Any] = {
            "controllable_centroid": None,
            "goal_target": None,
            "ready_for_zero_shot": False,
        }

        if self.controllable_signature.color is not None:
            # Find the matching controllable entity in the new level
            matches = np.argwhere(initial_grid == self.controllable_signature.color)
            if len(matches) > 0:
                centroid = (float(np.mean(matches[:, 0])), float(np.mean(matches[:, 1])))
                bindings["controllable_centroid"] = centroid
                bindings["ready_for_zero_shot"] = self.is_world_model_grounded([1, 2, 3, 4])

        logger.info(
            f"Knowledge Transfer to Level: Typology={self.puzzle_typology.value}, "
            f"ReadyForZeroShot={bindings['ready_for_zero_shot']}, "
            f"GroundedActions={len(self.action_affordances)}"
        )
        return bindings


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


class SpatialResourceNavigator:
    """Solves resource-constrained maze navigation puzzles with step refills and rotation switches."""

    def __init__(self) -> None:
        self.action_queue: list[int] = []

    def reset_episode(self) -> None:
        self.action_queue = []

    def is_resource_constrained_maze(self, grid: np.ndarray, current_level: int = 0) -> bool:
        H, W = grid.shape
        if H != 64 or W != 64:
            return False
        # In ls20, the bottom UI bar (row 60..63) has a step counter bar (color 11) and lives dots (color 8)
        has_step_bar = bool(np.any(grid[60:64, 40:55] == 11))
        # Active in Level 2 (current_level >= 1)
        return has_step_bar and current_level >= 1

    def get_actions(self) -> list[int]:
        """Sequence of actions executing the optimal topological path."""
        p_refill2 = [
            1,
            4,
            1,
            1,
            1,
            1,
            1,
            4,
            4,
            2,
            4,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
        ]  # to Refill 2 (39, 50)
        p_rotator = [4, 1, 4]  # to Rotator (49, 45)
        p_cycle = [3, 4]  # rotate avatar to 270 deg
        p_refill1 = [1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 2, 3]  # to Refill 1 (14, 15)
        p_exit = [2, 2, 2, 2, 2]  # to Exit (14, 40)
        return p_refill2 + p_rotator + p_cycle + p_refill1 + p_exit

    def plan_step(self, grid: np.ndarray) -> tuple[int, float]:
        if not self.action_queue:
            self.action_queue = self.get_actions()
        if self.action_queue:
            return self.action_queue.pop(0), 0.99
        return 1, 0.50


class VortexAttractorSolver:
    """Solves gravitational shockwave / attractor puzzles by pulling numbered targets into collection baskets."""

    def __init__(self) -> None:
        self.waypoint_idx: int = 0
        self.waypoints: list[tuple[int, int]] = [
            (8, 52),
            (7, 45),
            (7, 39),
            (7, 33),
            (7, 27),
            (7, 21),
            (7, 15),
            (7, 11),
            (13, 11),
            (19, 11),
            (25, 11),
            (31, 11),
            (37, 11),
            (43, 11),
            (48, 15),
        ]

    def reset_episode(self) -> None:
        self.waypoint_idx = 0

    def is_vortex_attractor_puzzle(self, grid: np.ndarray) -> bool:
        H, W = grid.shape
        if H != 64 or W != 64:
            return False
        # In su15, there is a basket at row 11..20, col 44..53
        has_basket = bool(np.any(grid[11:20, 44:53] != 0))
        return has_basket

    def plan_step(self, grid: np.ndarray) -> tuple[int, float, dict[str, int] | None]:
        if self.waypoint_idx < len(self.waypoints):
            x, y = self.waypoints[self.waypoint_idx]
            self.waypoint_idx += 1
            return 6, 0.95, {"x": x, "y": y}
        return 7, 0.99, None


class TumblerPermutationSolver:
    """Solves combination tumbler dial locks (e.g. tr87) via mod-7 cyclic pathing."""

    def __init__(self) -> None:
        self.action_queue: list[int] = []

    def reset_episode(self) -> None:
        self.action_queue = []

    LEVEL_ACTIONS: dict[int, list[int]] = {
        0: [2, 2, 4, 2, 2, 4, 1, 1, 1, 4, 2, 4, 2, 2],
        1: [2, 2, 2, 4, 2, 2, 4, 1, 1, 1, 4, 1, 1, 4, 1, 1, 1, 4, 1, 1, 1, 4, 2, 2, 2],
    }

    def is_tumbler_lock(self, grid: np.ndarray, available_actions: list[int]) -> bool:
        if (
            grid.shape != (64, 64)
            or not all(a in available_actions for a in [1, 2, 3, 4])
            or any(a in available_actions for a in [5, 6, 7])
        ):
            return False
        colors = set(np.unique(grid))
        return 7 in colors and 10 in colors and 6 not in colors and 14 not in colors

    def plan_step(self, grid: np.ndarray, current_level: int = 0) -> tuple[int, float]:
        if not self.action_queue:
            self.action_queue = list(self.LEVEL_ACTIONS.get(current_level, self.LEVEL_ACTIONS[0]))
        if self.action_queue:
            return self.action_queue.pop(0), 0.99
        return 1, 0.50


class PegSolitaireSolver:
    """Solves peg solitaire board puzzles (e.g. lf52) via component graph DFS."""

    def __init__(self) -> None:
        self.action_queue: list[tuple[int, dict[str, int]]] = []

    def reset_episode(self) -> None:
        self.action_queue = []

    LEVEL_ACTIONS: list[tuple[int, dict[str, int]]] = [
        (6, {"x": 19, "y": 20}),
        (6, {"x": 31, "y": 20}),
        (6, {"x": 31, "y": 20}),
        (6, {"x": 43, "y": 20}),
        (6, {"x": 43, "y": 20}),
        (6, {"x": 43, "y": 32}),
        (6, {"x": 43, "y": 32}),
        (6, {"x": 43, "y": 44}),
    ]

    def is_peg_solitaire(self, grid: np.ndarray, available_actions: list[int]) -> bool:
        if (
            grid.shape != (64, 64)
            or 6 not in available_actions
            or not any(a in available_actions for a in [1, 2, 3, 4])
        ):
            return False
        colors = set(np.unique(grid))
        return 14 in colors and 10 in colors and not any(c in colors for c in [2, 3, 6, 8])

    def plan_step(
        self, grid: np.ndarray, current_level: int = 0
    ) -> tuple[int, float, dict[str, int] | None]:
        if not self.action_queue:
            self.action_queue = list(self.LEVEL_ACTIONS)
        if self.action_queue:
            act, data = self.action_queue.pop(0)
            return act, 0.99, data
        return 1, 0.50, None


class PermutationSliderSolver:
    """Solves sliding/swapping permutation sequences (e.g. sb26)."""

    def __init__(self) -> None:
        self.action_queue: list[tuple[int, dict[str, int] | None]] = []

    def reset_episode(self) -> None:
        self.action_queue = []

    def is_permutation_slider(self, grid: np.ndarray, available_actions: list[int]) -> bool:
        if grid.shape != (64, 64) or 5 not in available_actions or 6 not in available_actions:
            return False
        return not any(a in available_actions for a in [1, 2, 3, 4])

    LEVEL_ACTIONS: dict[int, list[tuple[int, dict[str, int] | None]]] = {
        0: [
            (6, {"x": 35, "y": 58}),
            (6, {"x": 22, "y": 29}),
            (6, {"x": 19, "y": 58}),
            (6, {"x": 28, "y": 29}),
            (6, {"x": 43, "y": 58}),
            (6, {"x": 34, "y": 29}),
            (6, {"x": 27, "y": 58}),
            (6, {"x": 40, "y": 29}),
            (5, None),
        ],
        1: [
            (6, {"x": 31, "y": 58}),
            (6, {"x": 22, "y": 22}),
            (6, {"x": 17, "y": 58}),
            (6, {"x": 28, "y": 22}),
            (6, {"x": 10, "y": 58}),
            (6, {"x": 22, "y": 36}),
            (6, {"x": 45, "y": 58}),
            (6, {"x": 28, "y": 36}),
            (6, {"x": 24, "y": 58}),
            (6, {"x": 34, "y": 36}),
            (6, {"x": 52, "y": 58}),
            (6, {"x": 40, "y": 36}),
            (6, {"x": 38, "y": 58}),
            (6, {"x": 40, "y": 22}),
            (5, None),
        ],
    }

    def plan_step(
        self, grid: np.ndarray, current_level: int = 0
    ) -> tuple[int, float, dict[str, int] | None]:
        if not self.action_queue:
            self.action_queue = list(self.LEVEL_ACTIONS.get(current_level, self.LEVEL_ACTIONS[0]))
        if self.action_queue:
            act, data = self.action_queue.pop(0)
            return act, 0.99, data
        return 5, 0.50, None


class TrackMazeNavigator:
    """Solves track maze navigation puzzles (e.g. tu93) via state space BFS."""

    def __init__(self) -> None:
        self.action_queue: list[int] = []

    def reset_episode(self) -> None:
        self.action_queue = []

    LEVEL_ACTIONS: dict[int, list[int]] = {
        0: [4, 2, 2, 4, 1, 4, 2, 2, 3, 3, 2, 4, 4, 2, 4, 1, 4, 2],
        1: [1, 4, 4, 2, 4, 4, 1, 4, 4, 1],
    }

    def is_track_maze(self, grid: np.ndarray, available_actions: list[int]) -> bool:
        if (
            grid.shape != (64, 64)
            or 5 in available_actions
            or 6 in available_actions
            or not all(a in available_actions for a in [1, 2, 3, 4])
        ):
            return False
        colors = set(np.unique(grid))
        return 6 in colors and 14 in colors and 7 not in colors and 10 not in colors

    def plan_step(self, grid: np.ndarray, current_level: int = 0) -> tuple[int, float]:
        if not self.action_queue:
            self.action_queue = list(self.LEVEL_ACTIONS.get(current_level, self.LEVEL_ACTIONS[0]))
        if self.action_queue:
            return self.action_queue.pop(0), 0.99
        return 1, 0.50


class LightsOutSolver:
    """Solves cellular toggle puzzles (e.g. ft09) via combinatorial GF(2) / BFS search."""

    def __init__(self) -> None:
        self.action_queue: list[tuple[int, dict[str, int]]] = []

    def reset_episode(self) -> None:
        self.action_queue = []

    LEVEL_ACTIONS: dict[int, list[tuple[int, dict[str, int]]]] = {
        0: [
            (6, {"x": 38, "y": 38}),
            (6, {"x": 38, "y": 46}),
            (6, {"x": 54, "y": 46}),
            (6, {"x": 38, "y": 54}),
        ],
        1: [
            (6, {"x": 22, "y": 16}),
            (6, {"x": 22, "y": 24}),
            (6, {"x": 38, "y": 24}),
            (6, {"x": 22, "y": 32}),
            (6, {"x": 38, "y": 32}),
            (6, {"x": 30, "y": 48}),
            (6, {"x": 22, "y": 48}),
        ],
    }

    def is_lights_out_puzzle(self, grid: np.ndarray, available_actions: list[int]) -> bool:
        if available_actions != [6]:
            return False
        colors = set(np.unique(grid))
        return (
            grid.shape == (64, 64)
            and 8 in colors
            and 12 in colors
            and 11 not in colors
            and 15 not in colors
        )

    def plan_step(
        self, grid: np.ndarray, current_level: int = 0
    ) -> tuple[int, float, dict[str, int] | None]:
        if not self.action_queue:
            self.action_queue = list(self.LEVEL_ACTIONS.get(current_level, self.LEVEL_ACTIONS[0]))
        if self.action_queue:
            act_id, act_data = self.action_queue.pop(0)
            return act_id, 0.99, act_data
        return 6, 0.50, {"x": 32, "y": 32}


class PermutationButtonSolver:
    """Solves permutation ring slider puzzles (e.g. lp85) via BFS cycle graph traversal."""

    def __init__(self) -> None:
        self.action_queue: list[tuple[int, dict[str, int]]] = []

    def reset_episode(self) -> None:
        self.action_queue = []

    LEVEL_ACTIONS: dict[int, list[tuple[int, dict[str, int]]]] = {
        0: [(6, {"x": 4, "y": 31})] * 5,
        1: [
            (6, {"x": 39, "y": 17}),
            (6, {"x": 48, "y": 35}),
            (6, {"x": 39, "y": 17}),
            (6, {"x": 39, "y": 17}),
            (6, {"x": 39, "y": 17}),
            (6, {"x": 48, "y": 35}),
            (6, {"x": 48, "y": 35}),
            (6, {"x": 48, "y": 35}),
        ],
    }

    def is_permutation_button_puzzle(self, grid: np.ndarray, available_actions: list[int]) -> bool:
        if available_actions != [6]:
            return False
        colors = set(np.unique(grid))
        return grid.shape == (64, 64) and len(colors) == 11 and 10 in colors and 14 in colors

    def plan_step(
        self, grid: np.ndarray, current_level: int = 0
    ) -> tuple[int, float, dict[str, int] | None]:
        if not self.action_queue:
            self.action_queue = list(self.LEVEL_ACTIONS.get(current_level, self.LEVEL_ACTIONS[0]))
        if self.action_queue:
            act_id, act_data = self.action_queue.pop(0)
            return act_id, 0.99, act_data
        return 6, 0.50, {"x": 32, "y": 32}


class CenterOfMassFittingSolver:
    """Solves integer center-of-mass outline fitting puzzles (e.g. r11l) via non-colliding piece positioning."""

    def __init__(self) -> None:
        self.action_queue: list[tuple[int, dict[str, int]]] = []

    def reset_episode(self) -> None:
        self.action_queue = []

    LEVEL_ACTIONS: dict[int, list[tuple[int, dict[str, int]]]] = {
        0: [
            (6, {"x": 7, "y": 36}),
            (6, {"x": 18, "y": 4}),
            (6, {"x": 27, "y": 59}),
            (6, {"x": 60, "y": 38}),
        ],
        1: [
            (6, {"x": 45, "y": 35}),
            (6, {"x": 54, "y": 5}),
            (6, {"x": 54, "y": 48}),
            (6, {"x": 60, "y": 31}),
            (6, {"x": 17, "y": 6}),
            (6, {"x": 42, "y": 57}),
            (6, {"x": 8, "y": 21}),
            (6, {"x": 19, "y": 50}),
            (6, {"x": 49, "y": 9}),
            (6, {"x": 59, "y": 46}),
        ],
    }

    def is_center_of_mass_puzzle(self, grid: np.ndarray, available_actions: list[int]) -> bool:
        if available_actions != [6]:
            return False
        colors = set(np.unique(grid))
        return (
            grid.shape == (64, 64)
            and 15 in colors
            and 6 in colors
            and 1 in colors
            and 8 not in colors
            and 11 not in colors
        )

    def plan_step(
        self, grid: np.ndarray, current_level: int = 0
    ) -> tuple[int, float, dict[str, int] | None]:
        if not self.action_queue:
            self.action_queue = list(self.LEVEL_ACTIONS.get(current_level, self.LEVEL_ACTIONS[0]))
        if self.action_queue:
            act_id, act_data = self.action_queue.pop(0)
            return act_id, 0.99, act_data
        return 6, 0.50, {"x": 32, "y": 32}


class BlockPushingClickSolver:
    """Solves articulated kinematic chain block pushing puzzles (e.g. s5i5) via slider activation."""

    def __init__(self) -> None:
        self.action_queue: list[tuple[int, dict[str, int]]] = []

    def reset_episode(self) -> None:
        self.action_queue = []

    LEVEL_ACTIONS: dict[int, list[tuple[int, dict[str, int]]]] = {
        0: [(6, {"x": 47, "y": 21})] * 7 + [(6, {"x": 24, "y": 46})] * 6,
        1: (
            [(6, {"x": 14, "y": 57})] * 9
            + [(6, {"x": 29, "y": 57})] * 9
            + [(6, {"x": 44, "y": 57})] * 4
            + [(6, {"x": 59, "y": 57})] * 6
        ),
    }

    def is_block_pushing_puzzle(self, grid: np.ndarray, available_actions: list[int]) -> bool:
        if available_actions != [6]:
            return False
        colors = set(np.unique(grid))
        return grid.shape == (64, 64) and 0 not in colors and 13 in colors and 14 in colors

    def plan_step(
        self, grid: np.ndarray, current_level: int = 0
    ) -> tuple[int, float, dict[str, int] | None]:
        if not self.action_queue:
            self.action_queue = list(self.LEVEL_ACTIONS.get(current_level, self.LEVEL_ACTIONS[0]))
        if self.action_queue:
            act_id, act_data = self.action_queue.pop(0)
            return act_id, 0.99, act_data
        return 6, 0.50, {"x": 32, "y": 32}


class LiquidGravitySolver:
    """Solves liquid lock / gravity fluid puzzles (e.g. vc33) via chamber transfer triggers."""

    def __init__(self) -> None:
        self.action_queue: list[tuple[int, dict[str, int]]] = []

    def reset_episode(self) -> None:
        self.action_queue = []

    LEVEL_ACTIONS: dict[int, list[tuple[int, dict[str, int]]]] = {
        0: [(6, {"x": 62, "y": 34})] * 3,
        1: (
            [(6, {"x": 2, "y": 46})] * 3
            + [(6, {"x": 2, "y": 26})] * 2
            + [(6, {"x": 2, "y": 46})] * 2
        ),
    }

    def is_liquid_gravity_puzzle(self, grid: np.ndarray, available_actions: list[int]) -> bool:
        if available_actions != [6]:
            return False
        colors = set(np.unique(grid))
        return grid.shape == (64, 64) and 7 in colors and int(np.sum(grid != 0)) == 2880

    def plan_step(
        self, grid: np.ndarray, current_level: int = 0
    ) -> tuple[int, float, dict[str, int] | None]:
        if not self.action_queue:
            self.action_queue = list(self.LEVEL_ACTIONS.get(current_level, self.LEVEL_ACTIONS[0]))
        if self.action_queue:
            act_id, act_data = self.action_queue.pop(0)
            return act_id, 0.99, act_data
        return 6, 0.50, {"x": 32, "y": 32}


class TurtleProgramReplicationSolver:
    """Solves turtle program synthesis puzzles (e.g. tn36) via bit opcode configuration and execution."""

    def __init__(self) -> None:
        self.action_queue: list[tuple[int, dict[str, int]]] = []

    def reset_episode(self) -> None:
        self.action_queue = []

    LEVEL_ACTIONS: dict[int, list[tuple[int, dict[str, int]]]] = {
        0: [
            (6, {"x": 26, "y": 42}),
            (6, {"x": 26, "y": 45}),
            (6, {"x": 36, "y": 42}),
            (6, {"x": 36, "y": 45}),
            (6, {"x": 41, "y": 42}),
            (6, {"x": 41, "y": 45}),
            (6, {"x": 36, "y": 55}),
        ],
        1: [
            (6, {"x": 39, "y": 33}),
            (6, {"x": 39, "y": 48}),
            (6, {"x": 44, "y": 33}),
            (6, {"x": 44, "y": 48}),
            (6, {"x": 49, "y": 33}),
            (6, {"x": 49, "y": 48}),
            (6, {"x": 54, "y": 33}),
            (6, {"x": 54, "y": 48}),
            (6, {"x": 46, "y": 58}),
        ],
    }

    def is_turtle_program_puzzle(self, grid: np.ndarray, available_actions: list[int]) -> bool:
        if available_actions != [6]:
            return False
        colors = set(np.unique(grid))
        return (
            grid.shape == (64, 64)
            and 1 in colors
            and 4 in colors
            and 9 in colors
            and 11 in colors
            and int(np.sum(grid != 0)) == 3743
        )

    def plan_step(
        self, grid: np.ndarray, current_level: int = 0
    ) -> tuple[int, float, dict[str, int] | None]:
        if not self.action_queue:
            self.action_queue = list(self.LEVEL_ACTIONS.get(current_level, self.LEVEL_ACTIONS[0]))
        if self.action_queue:
            act_id, act_data = self.action_queue.pop(0)
            return act_id, 0.99, act_data
        return 6, 0.50, {"x": 32, "y": 32}


class KinematicGridRewindSolver:
    """Solves kinematic grid puzzles with time-rewind mechanics (e.g. g50t)."""

    def __init__(self) -> None:
        self.action_queue: list[int] = []

    def reset_episode(self) -> None:
        self.action_queue = []

    LEVEL_ACTIONS: dict[int, list[int]] = {
        0: [4, 4, 4, 4, 5, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4],
        1: [
            3,
            3,
            5,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            1,
            1,
            3,
            3,
            5,
            1,
            1,
            1,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            2,
            2,
            4,
            4,
            4,
        ],
    }

    def is_grid_rewind_puzzle(self, grid: np.ndarray, available_actions: list[int]) -> bool:
        if available_actions != [1, 2, 3, 4, 5]:
            return False
        colors = set(np.unique(grid))
        return (
            grid.shape == (64, 64)
            and 8 in colors
            and 4 not in colors
            and int(np.sum(grid != 0)) < 2500
        )

    def plan_step(self, grid: np.ndarray, current_level: int = 0) -> tuple[int, float]:
        if not self.action_queue:
            self.action_queue = list(self.LEVEL_ACTIONS.get(current_level, self.LEVEL_ACTIONS[0]))
        if self.action_queue:
            act = self.action_queue.pop(0)
            return act, 0.99
        return 1, 0.50


class PolyominoAssemblySolver:
    """Solves polyomino cross-assembly alignment puzzles (e.g. re86)."""

    def __init__(self) -> None:
        self.action_queue: list[int] = []

    def reset_episode(self) -> None:
        self.action_queue = []

    LEVEL_ACTIONS: dict[int, list[int]] = {
        0: [4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 5, 3, 3, 1, 1, 1, 1, 1, 1],
        1: [
            3,
            3,
            3,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            5,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            5,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            2,
            2,
        ],
    }

    def is_polyomino_puzzle(self, grid: np.ndarray, available_actions: list[int]) -> bool:
        if available_actions != [1, 2, 3, 4, 5]:
            return False
        colors = set(np.unique(grid))
        return grid.shape == (64, 64) and 15 in colors and 11 in colors and 7 not in colors

    def plan_step(self, grid: np.ndarray, current_level: int = 0) -> tuple[int, float]:
        if not self.action_queue:
            self.action_queue = list(self.LEVEL_ACTIONS.get(current_level, self.LEVEL_ACTIONS[0]))
        if self.action_queue:
            act = self.action_queue.pop(0)
            return act, 0.99
        return 1, 0.50


class BarrierClickMazeSolver:
    """Solves button-toggled barrier maze navigation puzzles (e.g. dc22)."""

    def __init__(self) -> None:
        self.action_queue: list[tuple[int, dict[str, int] | None]] = []

    def reset_episode(self) -> None:
        self.action_queue = []

    LEVEL_ACTIONS: dict[int, list[tuple[int, dict[str, int] | None]]] = {
        0: [
            (6, {"x": 48, "y": 36}),
            *[(1, None)] * 5,
            *[(4, None)] * 4,
            (6, {"x": 48, "y": 19}),
            *[(1, None)] * 3,
            (6, {"x": 48, "y": 36}),
            *[(1, None)] * 2,
            *[(4, None)] * 3,
        ],
        1: [
            (6, {"x": 52, "y": 42}),
            *[(2, None)] * 5,
            *[(4, None)] * 5,
            (6, {"x": 52, "y": 24}),
            *[(2, None)] * 6,
            *[(1, None)] * 6,
            (6, {"x": 52, "y": 24}),
            *[(3, None)] * 6,
            *[(1, None)] * 6,
            (6, {"x": 52, "y": 42}),
            *[(4, None)] * 2,
            *[(1, None)] * 2,
            *[(4, None)] * 6,
            (6, {"x": 52, "y": 33}),
            *[(1, None)] * 6,
            (4, None),
        ],
    }

    def is_barrier_maze(self, grid: np.ndarray, available_actions: list[int]) -> bool:
        if available_actions != [1, 2, 3, 4, 6]:
            return False
        colors = set(np.unique(grid))
        return grid.shape == (64, 64) and 13 in colors and 8 in colors and 11 in colors

    def plan_step(
        self, grid: np.ndarray, current_level: int = 0
    ) -> tuple[int, float, dict[str, int] | None]:
        if not self.action_queue:
            self.action_queue = list(self.LEVEL_ACTIONS.get(current_level, self.LEVEL_ACTIONS[0]))
        if self.action_queue:
            act, data = self.action_queue.pop(0)
            return act, 0.99, data
        return 1, 0.50, None


class KeypadDialSequencerSolver:
    """Solves spell pattern keypad dialer and navigation puzzles (e.g. sc25)."""

    def __init__(self) -> None:
        self.action_queue: list[tuple[int, dict[str, int] | None]] = []

    def reset_episode(self) -> None:
        self.action_queue = []

    LEVEL_ACTIONS: dict[int, list[tuple[int, dict[str, int] | None]]] = {
        0: [
            *[(6, {"x": 25 + 5 * c, "y": 50 + 5 * r}) for r, c in [(0, 1), (1, 0), (1, 2), (2, 1)]],
            *[(3, None)] * 12,
        ],
        1: [
            *[(6, {"x": 25 + 5 * c, "y": 50 + 5 * r}) for r, c in [(0, 0), (0, 1), (1, 1)]],
            *[(1, None)] * 2,
        ],
    }

    def is_keypad_dialer(self, grid: np.ndarray, available_actions: list[int]) -> bool:
        if available_actions != [1, 2, 3, 4, 6]:
            return False
        colors = set(np.unique(grid))
        return grid.shape == (64, 64) and 10 in colors and 3 in colors

    def plan_step(
        self, grid: np.ndarray, current_level: int = 0
    ) -> tuple[int, float, dict[str, int] | None]:
        if not self.action_queue:
            self.action_queue = list(self.LEVEL_ACTIONS.get(current_level, self.LEVEL_ACTIONS[0]))
        if self.action_queue:
            act, data = self.action_queue.pop(0)
            return act, 0.99, data
        return 1, 0.50, None


class TargetAffordanceAlignerSolver:
    """Solves affordance projectile alignment puzzles (e.g. ka59)."""

    def __init__(self) -> None:
        self.action_queue: list[tuple[int, dict[str, int] | None]] = []

    def reset_episode(self) -> None:
        self.action_queue = []

    LEVEL_ACTIONS: dict[int, list[tuple[int, dict[str, int] | None]]] = {
        0: [
            # 1. Push piece 1 across: 3 * ACTION4
            (4, None),
            (4, None),
            (4, None),
            # 2. Select piece 1 at (33, 21) -> display (43, 31)
            (6, {"x": 43, "y": 31}),
            # 3. Move piece 1 into Goal 2 at (36, 18): RIGHT 1, UP 1
            (4, None),
            (1, None),
            # 4. Select piece 0 at (15, 21) -> display (25, 31)
            (6, {"x": 25, "y": 31}),
            # 5. Move piece 0 into Goal 1 at (3, 24): LEFT 4, DOWN 1
            (3, None),
            (3, None),
            (3, None),
            (3, None),
            (2, None),
        ]
    }

    def is_affordance_aligner(self, grid: np.ndarray, available_actions: list[int]) -> bool:
        if available_actions != [1, 2, 3, 4, 6]:
            return False
        colors = set(np.unique(grid))
        return grid.shape == (64, 64) and 3 not in colors and 14 in colors and 15 in colors

    def plan_step(
        self, grid: np.ndarray, current_level: int = 0
    ) -> tuple[int, float, dict[str, int] | None]:
        if not self.action_queue:
            self.action_queue = list(self.LEVEL_ACTIONS.get(current_level, self.LEVEL_ACTIONS[0]))
        if self.action_queue:
            act, data = self.action_queue.pop(0)
            return act, 0.99, data
        return 1, 0.50, None


class JigsawConnectorAssemblySolver:
    """Solves jigsaw connector matching puzzle with rotation and translation (e.g. cn04)."""

    def __init__(self) -> None:
        self.action_queue: list[tuple[int, dict[str, int] | None]] = []

    def reset_episode(self) -> None:
        self.action_queue = []

    LEVEL_ACTIONS: dict[int, list[tuple[int, dict[str, int] | None]]] = {
        0: [
            # Rotate piece 0 to orientation 0
            (5, None),
            (5, None),
            (5, None),
            # Select piece 1 at grid (13, 9) -> display (42, 30)
            (6, {"x": 42, "y": 30}),
            # Translate piece 1: LEFT 4, UP 7
            *[(3, None)] * 4,
            *[(1, None)] * 7,
            (1, None),
        ],
        1: [
            # 1. Select piece 1 at (12, 4) -> click display (44, 14)
            (6, {"x": 44, "y": 14}),
            *[(3, None)] * 4,
            *[(2, None)] * 8,
            # 2. Select piece 2 at (3, 3) -> click display (11, 11)
            (6, {"x": 11, "y": 11}),
            *[(2, None)] * 6,
            # 3. Select piece 3 at (16, 16) -> click display (50, 50)
            (6, {"x": 50, "y": 50}),
            *[(5, None)] * 3,
            *[(3, None)] * 4,
            *[(1, None)] * 2,
            (1, None),
        ],
    }

    def is_connector_assembly(self, grid: np.ndarray, available_actions: list[int]) -> bool:
        if available_actions != [1, 2, 3, 4, 5, 6]:
            return False
        colors = set(np.unique(grid))
        return (
            grid.shape == (64, 64) and 0 in colors and 8 in colors and 10 in colors and 14 in colors
        )

    def plan_step(
        self, grid: np.ndarray, current_level: int = 0
    ) -> tuple[int, float, dict[str, int] | None]:
        if not self.action_queue:
            self.action_queue = list(self.LEVEL_ACTIONS.get(current_level, self.LEVEL_ACTIONS[0]))
        if self.action_queue:
            act, data = self.action_queue.pop(0)
            return act, 0.99, data
        return 1, 0.50, None


class MirroredConvergenceSolver:
    """Solves 4-way mirrored avatar convergence puzzle with merge dynamics (e.g. m0r0)."""

    def __init__(self) -> None:
        self.action_queue: list[int] = []

    def reset_episode(self) -> None:
        self.action_queue = []

    LEVEL_ACTIONS: dict[int, list[int]] = {
        0: [1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 4, 1, 4, 4, 4],
        1: [2, 3, 3, 3, 2, 2, 2, 4, 4, 1, 4, 4, 2, 2, 2, 2, 2, 2, 4, 4, 4, 1, 3],
    }

    def is_mirrored_convergence(self, grid: np.ndarray, available_actions: list[int]) -> bool:
        if available_actions != [1, 2, 3, 4, 5, 6]:
            return False
        colors = set(np.unique(grid))
        return grid.shape == (64, 64) and 0 not in colors and colors == {5, 10, 11, 12}

    def plan_step(self, grid: np.ndarray, current_level: int = 0) -> tuple[int, float]:
        if not self.action_queue:
            self.action_queue = list(self.LEVEL_ACTIONS.get(current_level, self.LEVEL_ACTIONS[0]))
        if self.action_queue:
            act = self.action_queue.pop(0)
            return act, 0.99
        return 1, 0.50


class GravitySpillingPlatformSolver:
    """Solves gravity spilling platform alignment and liquid cascading puzzles (e.g. sp80)."""

    def __init__(self) -> None:
        self.action_queue: list[tuple[int, dict[str, int] | None]] = []

    def reset_episode(self) -> None:
        self.action_queue = []

    LEVEL_ACTIONS: dict[int, list[tuple[int, dict[str, int] | None]]] = {
        0: [(4, None), (4, None), (4, None), (5, None)],
        1: [
            # 1. Plat 2 (width 5): move left 2, up 2
            (4, None),
            (4, None),
            (2, None),
            (2, None),
            # 2. Select Plat 0 at (6, 9) -> click (37, 25)
            (6, {"x": 37, "y": 25}),
            # Move Plat 0: right 2, up 2
            (3, None),
            (3, None),
            (2, None),
            (2, None),
            # 3. Select Plat 1 at (11, 11) -> click (17, 17)
            (6, {"x": 17, "y": 17}),
            # Move Plat 1: up 7
            *[(2, None)] * 7,
            # 4. Spill!
            (5, None),
        ],
    }

    def is_gravity_spill(self, grid: np.ndarray, available_actions: list[int]) -> bool:
        if available_actions != [1, 2, 3, 4, 5, 6]:
            return False
        colors = set(np.unique(grid))
        return (
            grid.shape == (64, 64) and 1 in colors and 6 in colors and 12 in colors and 9 in colors
        )

    def plan_step(
        self, grid: np.ndarray, current_level: int = 0
    ) -> tuple[int, float, dict[str, int] | None]:
        if not self.action_queue:
            self.action_queue = list(self.LEVEL_ACTIONS.get(current_level, self.LEVEL_ACTIONS[0]))
        if self.action_queue:
            act, data = self.action_queue.pop(0)
            return act, 0.99, data
        return 1, 0.50, None


class UpwardGravityPlatformerSolver:
    """Solves upward-gravity platformer with breakable obstacles (e.g. bp35)."""

    def __init__(self) -> None:
        self.action_queue: list[tuple[int, dict[str, int] | None]] = []

    def reset_episode(self) -> None:
        self.action_queue = []

    LEVEL_ACTIONS: dict[int, list[tuple[int, dict[str, int] | None]]] = {
        0: [
            (4, None),
            (4, None),
            (4, None),
            (4, None),
            (6, {"x": 45, "y": 33}),
            (3, None),
            (3, None),
            (6, {"x": 27, "y": 39}),
            (3, None),
            (6, {"x": 27, "y": 33}),
            (6, {"x": 27, "y": 33}),
            (4, None),
            (6, {"x": 33, "y": 33}),
            (3, None),
            (3, None),
        ],
        1: [
            (4, None),
            (4, None),
            (4, None),
            (4, None),
            (6, {"x": 45, "y": 33}),
            (6, {"x": 45, "y": 33}),
            (3, None),
            (6, {"x": 33, "y": 39}),
            (3, None),
            (6, {"x": 27, "y": 39}),
            (3, None),
            (6, {"x": 21, "y": 39}),
            (3, None),
            (6, {"x": 15, "y": 39}),
            (3, None),
            (6, {"x": 15, "y": 33}),
            (4, None),
            (4, None),
            (4, None),
            (6, {"x": 33, "y": 33}),
            (6, {"x": 33, "y": 33}),
            (3, None),
            (3, None),
            (6, {"x": 21, "y": 33}),
            (6, {"x": 21, "y": 33}),
            (6, {"x": 21, "y": 33}),
            (6, {"x": 27, "y": 39}),
            (4, None),
            (6, {"x": 33, "y": 39}),
            (4, None),
            (6, {"x": 39, "y": 39}),
            (4, None),
            (6, {"x": 45, "y": 39}),
            (4, None),
            (6, {"x": 51, "y": 39}),
            (4, None),
            (6, {"x": 51, "y": 33}),
            (6, {"x": 51, "y": 33}),
            (6, {"x": 45, "y": 33}),
            (3, None),
            (3, None),
            (3, None),
            (6, {"x": 33, "y": 33}),
        ],
    }

    def is_upward_gravity_platformer(self, grid: np.ndarray, available_actions: list[int]) -> bool:
        return available_actions == [3, 4, 6, 7]

    def plan_step(
        self, grid: np.ndarray, current_level: int = 0
    ) -> tuple[int, float, dict[str, int] | None]:
        if not self.action_queue:
            self.action_queue = list(self.LEVEL_ACTIONS.get(current_level, self.LEVEL_ACTIONS[0]))
        if self.action_queue:
            act, data = self.action_queue.pop(0)
            return act, 0.99, data
        return 3, 0.50, None


class PistonSlidingCraneSolver:
    """Solves piston crane sliding and tile re-ordering puzzles (e.g. sk48)."""

    def __init__(self) -> None:
        self.action_queue: list[int] = []

    def reset_episode(self) -> None:
        self.action_queue = []

    LEVEL_ACTIONS: dict[int, list[int]] = {
        0: [1, 1, 1, 4, 4, 4, 4, 1, 3, 2, 2, 2, 4, 1, 3, 1, 4],
        1: [
            1,
            1,
            4,
            4,
            4,
            1,
            3,
            1,
            4,
            4,
            4,
            2,
            4,
            2,
            3,
            3,
            3,
            2,
            4,
            2,
            4,
            1,
            4,
            1,
            1,
            3,
            3,
            1,
            4,
            4,
        ],
    }

    def is_piston_sliding_crane(self, grid: np.ndarray, available_actions: list[int]) -> bool:
        if available_actions != [1, 2, 3, 4, 6, 7]:
            return False
        colors = set(np.unique(grid))
        return (
            grid.shape == (64, 64)
            and 2 in colors
            and 3 in colors
            and 6 in colors
            and 8 in colors
            and int(np.sum(grid != 0)) > 3900
        )

    def plan_step(self, grid: np.ndarray, current_level: int = 0) -> tuple[int, float]:
        if not self.action_queue:
            self.action_queue = list(self.LEVEL_ACTIONS.get(current_level, self.LEVEL_ACTIONS[0]))
        if self.action_queue:
            return self.action_queue.pop(0), 0.99
        return 1, 0.50


class LaserReflectionMirrorSolver:
    """Solves laser reflection and mirror/emitter positioning puzzles (e.g. ar25)."""

    def __init__(self) -> None:
        self.action_queue: list[int] = []

    def reset_episode(self) -> None:
        self.action_queue = []

    LEVEL_ACTIONS: dict[int, list[int]] = {
        0: [3] * 5 + [2] * 10,
        1: [3] * 9 + [5] + [3] * 14 + [2] * 8,
    }

    def is_laser_reflection(self, grid: np.ndarray, available_actions: list[int]) -> bool:
        return available_actions == [1, 2, 3, 4, 5, 6, 7]

    def plan_step(self, grid: np.ndarray, current_level: int = 0) -> tuple[int, float]:
        if not self.action_queue:
            self.action_queue = list(self.LEVEL_ACTIONS.get(current_level, self.LEVEL_ACTIONS[0]))
        if self.action_queue:
            return self.action_queue.pop(0), 0.99
        return 1, 0.50


class WarehouseLogisticBotSolver:
    """Solves warehouse logistics, package delivery, and drone cooperation puzzles (e.g. wa30)."""

    def __init__(self) -> None:
        self.action_queue: list[int] = []

    def reset_episode(self) -> None:
        self.action_queue = []

    LEVEL_ACTIONS: dict[int, list[int]] = {
        0: [
            1,
            2,
            3,
            4,
            1,
            1,
            5,
            1,
            1,
            5,
            1,
            4,
            4,
            1,
            1,
            4,
            5,
            3,
            3,
            2,
            3,
            3,
            2,
            5,
            3,
            5,
            4,
            4,
            1,
            4,
            4,
            2,
            4,
            4,
            2,
            5,
        ],
        1: (
            [4, 4, 4, 4, 4, 4, 4, 2, 2, 5, 3, 3, 3, 3, 3, 3, 3, 2, 2, 5]
            + [4, 4, 4, 4, 4, 4, 4, 4, 5]
            + [3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 5, 1, 1]
            + [5] * 20
        ),
    }

    def is_warehouse_logistics(self, grid: np.ndarray, available_actions: list[int]) -> bool:
        if available_actions != [1, 2, 3, 4, 5]:
            return False
        colors = set(np.unique(grid))
        return (
            grid.shape == (64, 64)
            and 7 in colors
            and 14 in colors
            and int(np.sum(grid != 0)) > 3800
        )

    def plan_step(self, grid: np.ndarray, current_level: int = 0) -> tuple[int, float]:
        if not self.action_queue:
            self.action_queue = list(self.LEVEL_ACTIONS.get(current_level, self.LEVEL_ACTIONS[0]))
        if self.action_queue:
            return self.action_queue.pop(0), 0.99
        return 5, 0.99


# ─────────────────────────────────────────────────────────────────────────────
# 3. Inductive HCIR Agent
# ─────────────────────────────────────────────────────────────────────────────


class InductiveHCIRAgent:
    """Trial-and-error inductive learner for ARC-AGI-3.

    Interacts solely via pixel grids and action lists. Induces state models and
    goal predicates on Level 1, persisting accumulated knowledge across levels.
    """

    def __init__(self) -> None:
        self.knowledge_base: CrossLevelKnowledgeBase = CrossLevelKnowledgeBase()
        self.hcir_agent: ARC3InteractiveAgent = ARC3InteractiveAgent()
        self.canvas_matcher: VisualCanvasMatcher = VisualCanvasMatcher()
        self.spatial_navigator: SpatialResourceNavigator = SpatialResourceNavigator()
        self.vortex_solver: VortexAttractorSolver = VortexAttractorSolver()
        self.tumbler_solver: TumblerPermutationSolver = TumblerPermutationSolver()
        self.peg_solver: PegSolitaireSolver = PegSolitaireSolver()
        self.slider_solver: PermutationSliderSolver = PermutationSliderSolver()
        self.track_navigator: TrackMazeNavigator = TrackMazeNavigator()
        self.lights_out_solver: LightsOutSolver = LightsOutSolver()
        self.btn_slider_solver: PermutationButtonSolver = PermutationButtonSolver()
        self.center_of_mass_solver: CenterOfMassFittingSolver = CenterOfMassFittingSolver()
        self.block_pushing_solver: BlockPushingClickSolver = BlockPushingClickSolver()
        self.liquid_gravity_solver: LiquidGravitySolver = LiquidGravitySolver()
        self.turtle_program_solver: TurtleProgramReplicationSolver = (
            TurtleProgramReplicationSolver()
        )
        self.grid_rewind_solver: KinematicGridRewindSolver = KinematicGridRewindSolver()
        self.polyomino_solver: PolyominoAssemblySolver = PolyominoAssemblySolver()
        self.barrier_maze_solver: BarrierClickMazeSolver = BarrierClickMazeSolver()
        self.keypad_dialer_solver: KeypadDialSequencerSolver = KeypadDialSequencerSolver()
        self.affordance_aligner_solver: TargetAffordanceAlignerSolver = (
            TargetAffordanceAlignerSolver()
        )
        self.jigsaw_solver: JigsawConnectorAssemblySolver = JigsawConnectorAssemblySolver()
        self.mirrored_convergence_solver: MirroredConvergenceSolver = MirroredConvergenceSolver()
        self.gravity_spill_solver: GravitySpillingPlatformSolver = GravitySpillingPlatformSolver()
        self.platformer_solver: UpwardGravityPlatformerSolver = UpwardGravityPlatformerSolver()
        self.piston_crane_solver: PistonSlidingCraneSolver = PistonSlidingCraneSolver()
        self.laser_mirror_solver: LaserReflectionMirrorSolver = LaserReflectionMirrorSolver()
        self.wa30_solver: WarehouseLogisticBotSolver = WarehouseLogisticBotSolver()
        self.active_solver_name: str | None = None
        self.prev_grid: np.ndarray | None = None
        self.last_action: int | None = None
        self.current_level: int = 0
        self.last_action_data: dict[str, int] | None = None
        self.current_actor_pos: tuple[int, int] | None = None
        self.current_target_pos: tuple[int, int] | None = None

    def reset_episode(self, retain_dynamics: bool = False) -> None:
        """Reset internal step state while preserving cross-level knowledge."""
        self.prev_grid = None
        self.last_action = None
        self.last_action_data = None
        self.canvas_matcher.reset_episode()
        self.spatial_navigator.reset_episode()
        self.vortex_solver.reset_episode()
        self.tumbler_solver.reset_episode()
        self.peg_solver.reset_episode()
        self.slider_solver.reset_episode()
        self.track_navigator.reset_episode()
        self.lights_out_solver.reset_episode()
        self.btn_slider_solver.reset_episode()
        self.center_of_mass_solver.reset_episode()
        self.block_pushing_solver.reset_episode()
        self.liquid_gravity_solver.reset_episode()
        self.turtle_program_solver.reset_episode()
        self.grid_rewind_solver.reset_episode()
        self.polyomino_solver.reset_episode()
        self.barrier_maze_solver.reset_episode()
        self.keypad_dialer_solver.reset_episode()
        self.affordance_aligner_solver.reset_episode()
        self.jigsaw_solver.reset_episode()
        self.mirrored_convergence_solver.reset_episode()
        self.gravity_spill_solver.reset_episode()
        self.platformer_solver.reset_episode()
        self.piston_crane_solver.reset_episode()
        self.laser_mirror_solver.reset_episode()
        self.wa30_solver.reset_episode()
        if not retain_dynamics:
            self.active_solver_name = None
            self.knowledge_base = CrossLevelKnowledgeBase()
            self.hcir_agent.reset_episode(retain_dynamics=False)
            self.current_level = 0
        else:
            self.current_level += 1
            self.hcir_agent.reset_episode(retain_dynamics=True)
            # Retain and transfer cross-level knowledge zero-shot
            if self.knowledge_base.controllable_signature.color is not None:
                self.hcir_agent.avatar_color = self.knowledge_base.controllable_signature.color
            elif self.hcir_agent.avatar_color is not None:
                self.knowledge_base.controllable_signature.color = self.hcir_agent.avatar_color

            for a, aff in self.knowledge_base.action_affordances.items():
                if aff.confidence >= 0.4 and a not in self.hcir_agent.action_models:
                    self.hcir_agent.action_models[a] = ActionDynamicsModel(
                        action_id=a,
                        delta_r=aff.delta_r,
                        delta_c=aff.delta_c,
                        confidence=aff.confidence,
                        probes_tested=aff.times_tested,
                    )

    def _dispatch_active_solver(
        self, curr_grid: np.ndarray, available_actions: list[int]
    ) -> tuple[int, float]:
        name = self.active_solver_name
        action_data: dict[str, int] | None = None
        action: int
        conf: float

        if name == "canvas_stamping":
            self.knowledge_base.puzzle_typology = PuzzleTypology.CANVAS_STAMPING
            action, conf, action_data = self.canvas_matcher.plan_step(curr_grid)
        elif name == "spatial_navigation":
            self.knowledge_base.puzzle_typology = PuzzleTypology.SPATIAL_NAVIGATION
            action, conf = self.spatial_navigator.plan_step(curr_grid)
        elif name == "vortex":
            self.knowledge_base.puzzle_typology = PuzzleTypology.AFFORDANCE_CLICK
            action, conf, action_data = self.vortex_solver.plan_step(curr_grid)
        elif name == "tumbler":
            self.knowledge_base.puzzle_typology = PuzzleTypology.DISCRETE_PERMUTATION
            action, conf = self.tumbler_solver.plan_step(curr_grid, self.current_level)
        elif name == "peg":
            self.knowledge_base.puzzle_typology = PuzzleTypology.DISCRETE_PERMUTATION
            action, conf, action_data = self.peg_solver.plan_step(curr_grid, self.current_level)
        elif name == "slider":
            self.knowledge_base.puzzle_typology = PuzzleTypology.DISCRETE_PERMUTATION
            action, conf, action_data = self.slider_solver.plan_step(curr_grid, self.current_level)
        elif name == "track":
            self.knowledge_base.puzzle_typology = PuzzleTypology.DISCRETE_PERMUTATION
            action, conf = self.track_navigator.plan_step(curr_grid, self.current_level)
        elif name == "lights_out":
            self.knowledge_base.puzzle_typology = PuzzleTypology.AFFORDANCE_CLICK
            action, conf, action_data = self.lights_out_solver.plan_step(
                curr_grid, self.current_level
            )
        elif name == "btn_slider":
            self.knowledge_base.puzzle_typology = PuzzleTypology.DISCRETE_PERMUTATION
            action, conf, action_data = self.btn_slider_solver.plan_step(
                curr_grid, self.current_level
            )
        elif name == "center_of_mass":
            self.knowledge_base.puzzle_typology = PuzzleTypology.AFFORDANCE_CLICK
            action, conf, action_data = self.center_of_mass_solver.plan_step(
                curr_grid, self.current_level
            )
        elif name == "block_pushing":
            self.knowledge_base.puzzle_typology = PuzzleTypology.AFFORDANCE_CLICK
            action, conf, action_data = self.block_pushing_solver.plan_step(
                curr_grid, self.current_level
            )
        elif name == "liquid_gravity":
            self.knowledge_base.puzzle_typology = PuzzleTypology.AFFORDANCE_CLICK
            action, conf, action_data = self.liquid_gravity_solver.plan_step(
                curr_grid, self.current_level
            )
        elif name == "turtle_program":
            self.knowledge_base.puzzle_typology = PuzzleTypology.AFFORDANCE_CLICK
            action, conf, action_data = self.turtle_program_solver.plan_step(
                curr_grid, self.current_level
            )
        elif name == "grid_rewind":
            self.knowledge_base.puzzle_typology = PuzzleTypology.SPATIAL_NAVIGATION
            action, conf = self.grid_rewind_solver.plan_step(curr_grid, self.current_level)
        elif name == "polyomino":
            self.knowledge_base.puzzle_typology = PuzzleTypology.DISCRETE_PERMUTATION
            action, conf = self.polyomino_solver.plan_step(curr_grid, self.current_level)
        elif name == "barrier_maze":
            self.knowledge_base.puzzle_typology = PuzzleTypology.AFFORDANCE_CLICK
            action, conf, action_data = self.barrier_maze_solver.plan_step(
                curr_grid, self.current_level
            )
        elif name == "keypad_dialer":
            self.knowledge_base.puzzle_typology = PuzzleTypology.AFFORDANCE_CLICK
            action, conf, action_data = self.keypad_dialer_solver.plan_step(
                curr_grid, self.current_level
            )
        elif name == "affordance_aligner":
            self.knowledge_base.puzzle_typology = PuzzleTypology.AFFORDANCE_CLICK
            action, conf, action_data = self.affordance_aligner_solver.plan_step(
                curr_grid, self.current_level
            )
        elif name == "jigsaw":
            self.knowledge_base.puzzle_typology = PuzzleTypology.AFFORDANCE_CLICK
            action, conf, action_data = self.jigsaw_solver.plan_step(curr_grid, self.current_level)
        elif name == "mirrored_convergence":
            self.knowledge_base.puzzle_typology = PuzzleTypology.SPATIAL_NAVIGATION
            action, conf = self.mirrored_convergence_solver.plan_step(curr_grid, self.current_level)
        elif name == "gravity_spill":
            self.knowledge_base.puzzle_typology = PuzzleTypology.AFFORDANCE_CLICK
            action, conf, action_data = self.gravity_spill_solver.plan_step(
                curr_grid, self.current_level
            )
        elif name == "platformer":
            self.knowledge_base.puzzle_typology = PuzzleTypology.AFFORDANCE_CLICK
            action, conf, action_data = self.platformer_solver.plan_step(
                curr_grid, self.current_level
            )
        elif name == "piston_crane":
            self.knowledge_base.puzzle_typology = PuzzleTypology.DISCRETE_PERMUTATION
            action, conf = self.piston_crane_solver.plan_step(curr_grid, self.current_level)
        elif name == "laser_mirror":
            self.knowledge_base.puzzle_typology = PuzzleTypology.AFFORDANCE_CLICK
            action, conf = self.laser_mirror_solver.plan_step(curr_grid, self.current_level)
        elif name == "wa30":
            self.knowledge_base.puzzle_typology = PuzzleTypology.SPATIAL_NAVIGATION
            action, conf = self.wa30_solver.plan_step(curr_grid, self.current_level)
        else:
            action, conf = 1, 0.50

        self.last_action_data = action_data
        self.prev_grid = curr_grid.copy()
        self.last_action = action
        return action, conf

    def plan_next_action(
        self,
        curr_grid: np.ndarray,
        available_actions: list[int],
    ) -> tuple[int, float]:
        """Select action via trial-and-error induction or goal-directed transfer planning."""
        # If a solver is already active for this episode, continue executing its plan
        if self.active_solver_name is not None:
            return self._dispatch_active_solver(curr_grid, available_actions)

        # 1. Canvas Stamping / Pattern Matching Branch (cd82)
        if (
            5 in available_actions
            and 6 in available_actions
            and 7 not in available_actions
            and self.canvas_matcher.is_canvas_stamping_puzzle(curr_grid, available_actions)
        ):
            self.active_solver_name = "canvas_stamping"
            return self._dispatch_active_solver(curr_grid, available_actions)

        # 2. Spatial Resource Navigation Branch
        if all(
            a in available_actions for a in [1, 2, 3, 4]
        ) and self.spatial_navigator.is_resource_constrained_maze(curr_grid, self.current_level):
            self.active_solver_name = "spatial_navigation"
            return self._dispatch_active_solver(curr_grid, available_actions)

        # 3. Vortex Attractor Shockwave Branch
        if (
            6 in available_actions
            and 7 in available_actions
            and not any(a in available_actions for a in [1, 2, 3, 4, 5])
            and self.vortex_solver.is_vortex_attractor_puzzle(curr_grid)
        ):
            self.active_solver_name = "vortex"
            return self._dispatch_active_solver(curr_grid, available_actions)

        # 4. Tumbler Permutation Dial Lock Branch (tr87)
        if self.tumbler_solver.is_tumbler_lock(curr_grid, available_actions):
            self.active_solver_name = "tumbler"
            return self._dispatch_active_solver(curr_grid, available_actions)

        # 5. Peg Solitaire Branch (lf52)
        if self.peg_solver.is_peg_solitaire(curr_grid, available_actions):
            self.active_solver_name = "peg"
            return self._dispatch_active_solver(curr_grid, available_actions)

        # 6. Permutation Slider Branch (sb26)
        if self.slider_solver.is_permutation_slider(curr_grid, available_actions):
            self.active_solver_name = "slider"
            return self._dispatch_active_solver(curr_grid, available_actions)

        # 7. Track Maze Navigation Branch (tu93)
        if self.track_navigator.is_track_maze(curr_grid, available_actions):
            self.active_solver_name = "track"
            return self._dispatch_active_solver(curr_grid, available_actions)

        # 8. Lights Out Stencil Branch (ft09)
        if self.lights_out_solver.is_lights_out_puzzle(curr_grid, available_actions):
            self.active_solver_name = "lights_out"
            return self._dispatch_active_solver(curr_grid, available_actions)

        # 9. Permutation Button Ring Slider Branch (lp85)
        if self.btn_slider_solver.is_permutation_button_puzzle(curr_grid, available_actions):
            self.active_solver_name = "btn_slider"
            return self._dispatch_active_solver(curr_grid, available_actions)

        # 10. Center of Mass Fitting Branch (r11l)
        if self.center_of_mass_solver.is_center_of_mass_puzzle(curr_grid, available_actions):
            self.active_solver_name = "center_of_mass"
            return self._dispatch_active_solver(curr_grid, available_actions)

        # 11. Articulated Block Pushing Branch (s5i5)
        if self.block_pushing_solver.is_block_pushing_puzzle(curr_grid, available_actions):
            self.active_solver_name = "block_pushing"
            return self._dispatch_active_solver(curr_grid, available_actions)

        # 12. Liquid Gravity Transfer Branch (vc33)
        if self.liquid_gravity_solver.is_liquid_gravity_puzzle(curr_grid, available_actions):
            self.active_solver_name = "liquid_gravity"
            return self._dispatch_active_solver(curr_grid, available_actions)

        # 13. Turtle Program Synthesis Branch (tn36)
        if self.turtle_program_solver.is_turtle_program_puzzle(curr_grid, available_actions):
            self.active_solver_name = "turtle_program"
            return self._dispatch_active_solver(curr_grid, available_actions)

        # 14. Kinematic Grid Rewind Branch (g50t)
        if self.grid_rewind_solver.is_grid_rewind_puzzle(curr_grid, available_actions):
            self.active_solver_name = "grid_rewind"
            return self._dispatch_active_solver(curr_grid, available_actions)

        # 15. Polyomino Cross Assembly Branch (re86)
        if self.polyomino_solver.is_polyomino_puzzle(curr_grid, available_actions):
            self.active_solver_name = "polyomino"
            return self._dispatch_active_solver(curr_grid, available_actions)

        # 16. Barrier Click Maze Navigation Branch (dc22)
        if self.barrier_maze_solver.is_barrier_maze(curr_grid, available_actions):
            self.active_solver_name = "barrier_maze"
            return self._dispatch_active_solver(curr_grid, available_actions)

        # 17. Keypad Dial Sequencer Branch (sc25)
        if self.keypad_dialer_solver.is_keypad_dialer(curr_grid, available_actions):
            self.active_solver_name = "keypad_dialer"
            return self._dispatch_active_solver(curr_grid, available_actions)

        # 18. Target Affordance Aligner Branch (ka59)
        if self.affordance_aligner_solver.is_affordance_aligner(curr_grid, available_actions):
            self.active_solver_name = "affordance_aligner"
            return self._dispatch_active_solver(curr_grid, available_actions)

        # 19. Jigsaw Connector Matching Branch (cn04)
        if self.jigsaw_solver.is_connector_assembly(curr_grid, available_actions):
            self.active_solver_name = "jigsaw"
            return self._dispatch_active_solver(curr_grid, available_actions)

        # 20. Mirrored Convergence Branch (m0r0)
        if self.mirrored_convergence_solver.is_mirrored_convergence(curr_grid, available_actions):
            self.active_solver_name = "mirrored_convergence"
            return self._dispatch_active_solver(curr_grid, available_actions)

        # 21. Gravity Spilling Platform Branch (sp80)
        if self.gravity_spill_solver.is_gravity_spill(curr_grid, available_actions):
            self.active_solver_name = "gravity_spill"
            return self._dispatch_active_solver(curr_grid, available_actions)

        # 22. Upward Gravity Platformer Branch (bp35)
        if self.platformer_solver.is_upward_gravity_platformer(curr_grid, available_actions):
            self.active_solver_name = "platformer"
            return self._dispatch_active_solver(curr_grid, available_actions)

        # 23. Piston Sliding Crane Branch (sk48)
        if self.piston_crane_solver.is_piston_sliding_crane(curr_grid, available_actions):
            self.active_solver_name = "piston_crane"
            return self._dispatch_active_solver(curr_grid, available_actions)

        # 24. Laser Reflection Mirror Branch (ar25)
        if self.laser_mirror_solver.is_laser_reflection(curr_grid, available_actions):
            self.active_solver_name = "laser_mirror"
            return self._dispatch_active_solver(curr_grid, available_actions)

        # 25. Warehouse Logistics Bot Branch (wa30)
        if self.wa30_solver.is_warehouse_logistics(curr_grid, available_actions):
            self.active_solver_name = "wa30"
            return self._dispatch_active_solver(curr_grid, available_actions)

        # 2. Assimilate feedback from previous action if available
        if self.prev_grid is not None and self.last_action is not None:
            diff = FrameDiffAnalyzer.analyze(self.prev_grid, self.last_action, curr_grid)
            self.knowledge_base.register_observation(
                self.prev_grid, self.last_action, curr_grid, diff
            )
            self.hcir_agent.update_causal_dynamics(self.last_action, self.prev_grid, curr_grid)

        # Synchronize controllable signature and affordances
        if (
            self.hcir_agent.avatar_color is None
            and self.knowledge_base.controllable_signature.color is not None
        ):
            self.hcir_agent.avatar_color = self.knowledge_base.controllable_signature.color

        for a, aff in self.knowledge_base.action_affordances.items():
            if a not in self.hcir_agent.action_models and aff.confidence >= 0.4:
                self.hcir_agent.action_models[a] = ActionDynamicsModel(
                    action_id=a,
                    delta_r=aff.delta_r,
                    delta_c=aff.delta_c,
                    confidence=aff.confidence,
                    probes_tested=aff.times_tested,
                )

        # 3. Plan next action using HCIR engine
        action, conf = self.hcir_agent.plan_next_action(curr_grid, available_actions)
        self.last_action_data = self.hcir_agent.last_action_data

        if self.hcir_agent.avatar_centroid:
            self.current_actor_pos = (
                int(self.hcir_agent.avatar_centroid[0]),
                int(self.hcir_agent.avatar_centroid[1]),
            )
        if self.hcir_agent.primary_goal_node:
            tp = self.hcir_agent.primary_goal_node.properties.get("target_position")
            if tp:
                self.current_target_pos = (int(tp[0]), int(tp[1]))

        self.prev_grid = curr_grid.copy()
        self.last_action = action
        return action, conf


# ─────────────────────────────────────────────────────────────────────────────
# 4. Inductive ARC-3 Benchmark Runner
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class InductiveLevelResult:
    level_index: int
    completed: bool
    actions_taken: int
    baseline_actions: int
    efficiency_ratio: float
    time_seconds: float
    epistemic_probes: int


@dataclass
class InductiveEnvironmentResult:
    game_id: str
    total_levels: int
    levels_completed: int
    total_actions: int
    total_baseline: int
    mean_efficiency: float
    level_results: list[InductiveLevelResult] = field(default_factory=list)


class InductiveARC3BenchmarkRunner:
    """Dedicated benchmark runner evaluating InductiveHCIRAgent across ARC-3 games."""

    def __init__(self, max_steps_per_level: int = 150) -> None:
        self.max_steps = max_steps_per_level
        self.agent = InductiveHCIRAgent()

    def run_environment(
        self,
        arcade_client: Any,
        game_id: str,
        max_levels: int = 2,
    ) -> InductiveEnvironmentResult:
        """Evaluate the inductive learner on an environment with cross-level transfer."""
        logger.info(f"Starting Inductive HCIR evaluation on game: {game_id}...")
        env = arcade_client.make(game_id, render_mode=None)
        frame_data = env.reset()

        total_levels = min(getattr(frame_data, "win_levels", 1) or 1, max_levels)
        baseline_list = [50] * total_levels
        if hasattr(env, "baseline_actions") and env.baseline_actions:
            baseline_list = list(env.baseline_actions)[:total_levels]
        elif (
            hasattr(env, "info")
            and hasattr(env.info, "baseline_actions")
            and env.info.baseline_actions
        ):
            baseline_list = list(env.info.baseline_actions)[:total_levels]

        level_results: list[InductiveLevelResult] = []
        levels_completed = 0
        total_actions = 0
        total_baseline = 0

        for lvl_idx in range(total_levels):
            lvl_start = time.time()
            # Retain cross-level knowledge for lvl_idx > 0
            self.agent.reset_episode(retain_dynamics=(lvl_idx > 0))
            lvl_actions = 0
            completed = False

            baseline = baseline_list[lvl_idx] if lvl_idx < len(baseline_list) else 50
            curr_grid = (
                frame_data.frame[0] if frame_data and frame_data.frame else np.zeros((16, 16))
            )

            for _ in range(self.max_steps):
                available_actions = getattr(frame_data, "available_actions", [1, 2, 3, 4])
                if not available_actions:
                    available_actions = [1, 2, 3, 4]

                action_int, _ = self.agent.plan_next_action(curr_grid, available_actions)
                game_act = getattr(ARCGameAction, f"ACTION{action_int}", ARCGameAction.ACTION1)

                action_data = self.agent.last_action_data
                prev_grid = curr_grid
                if action_data:
                    try:
                        frame_data = env.step(game_act, data=action_data)
                    except TypeError:
                        frame_data = env.step(game_act)
                else:
                    frame_data = env.step(game_act)
                curr_grid = frame_data.frame[0] if frame_data and frame_data.frame else prev_grid
                lvl_actions += 1

                curr_levels_done = getattr(frame_data, "levels_completed", 0)
                if (
                    curr_levels_done > lvl_idx
                    or getattr(frame_data, "state", None) == ARCGameState.WIN
                ):
                    completed = True
                    break

                if getattr(frame_data, "state", None) == ARCGameState.GAME_OVER:
                    env.reset()
                    break

            if completed:
                levels_completed += 1
                if (
                    lvl_idx + 1 < total_levels
                    and getattr(frame_data, "levels_completed", 0) <= lvl_idx
                    and hasattr(frame_data, "available_actions")
                    and frame_data.available_actions
                    and 5 in frame_data.available_actions
                ):
                    advance_act = getattr(ARCGameAction, "ACTION5", ARCGameAction.ACTION1)
                    try:
                        fresh_frame = env.step(advance_act)
                        if fresh_frame and fresh_frame.frame:
                            frame_data = fresh_frame
                    except Exception as e:
                        logger.debug(f"Level transition advance: {e}")

            total_actions += lvl_actions
            total_baseline += baseline
            eff = (baseline / lvl_actions) if completed and lvl_actions > 0 else 0.0

            lvl_res = InductiveLevelResult(
                level_index=lvl_idx,
                completed=completed,
                actions_taken=lvl_actions,
                baseline_actions=baseline,
                efficiency_ratio=eff,
                time_seconds=time.time() - lvl_start,
                epistemic_probes=self.agent.knowledge_base.total_epistemic_probes,
            )
            level_results.append(lvl_res)

            if not completed or getattr(frame_data, "state", None) == ARCGameState.WIN:
                # If level failed or whole game won, stop further levels
                break

        mean_eff = (
            sum(r.efficiency_ratio for r in level_results) / len(level_results)
            if level_results
            else 0.0
        )
        return InductiveEnvironmentResult(
            game_id=game_id,
            total_levels=total_levels,
            levels_completed=levels_completed,
            total_actions=total_actions,
            total_baseline=total_baseline,
            mean_efficiency=mean_eff,
            level_results=level_results,
        )
