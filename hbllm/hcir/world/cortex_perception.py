from __future__ import annotations

"""Thalamocortical Visual Perception Faculty for HCIR World Kernel.

Biological Modeling:
1. Sensory Thalamus: Universal sensory normalization across camera, depth, lidar, and 2D arrays.
2. Ventral Visual Stream (V1-V4, IT):
   - Figure-ground segregation and background estimation.
   - Connected-component morphological entity segmentation using native vectorized labeling.
   - Gestalt grouping and relational alignment affordances (symmetry, collinearity).
   - Homogeneous interactive control cluster detection.
3. Dorsal Visual Stream & Theory of Mind:
   - Gaze cone and directional orientation tracking of sentient creatures.
   - Frame difference mutation analysis (translation, index cycle, global transition).
"""

import logging
from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.ndimage import label

from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.cortical_stage import StageResult
from hbllm.hcir.world.visual_symmetry import VisualSymmetryAnalyzer

if TYPE_CHECKING:
    from hbllm.hcir.world.engine_state import CognitiveContext

logger = logging.getLogger(__name__)

_STRUCT_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


def _fast_unique_counts(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """O(N) bincount-based unique value & frequency extractor for bounded integer lattices."""
    if arr.size > 0 and np.issubdtype(arr.dtype, np.integer):
        amin, amax = int(arr.min()), int(arr.max())
        if 0 <= amin and amax < 256:
            bc = np.bincount(arr.ravel(), minlength=amax + 1)
            nz = np.flatnonzero(bc)
            return nz, bc[nz]
    return np.unique(arr, return_counts=True)


class FrameDiffType(StrEnum):
    """Categorization of visual changes resulting from an action."""

    NO_CHANGE = "NO_CHANGE"
    TRANSLATION = "TRANSLATION"
    IN_PLACE_MUTATION = "IN_PLACE_MUTATION"
    INDEX_CYCLE = "INDEX_CYCLE"
    CANVAS_TRANSFORMATION = "CANVAS_TRANSFORMATION"
    GLOBAL_TRANSITION = "GLOBAL_TRANSITION"


@dataclass
class EpistemicObservationDiff:
    """Detailed structural difference between consecutive observation frames."""

    changed_pixel_count: int = 0
    changed_mask: np.ndarray | None = None
    displaced_entities: list[tuple[SpatialEntity, tuple[int, int]]] = field(default_factory=list)
    mutated_pixels: list[tuple[int, int, int, int]] = field(
        default_factory=list
    )  # (r, c, old_val, new_val)
    disappeared_features: set[int] = field(default_factory=set)
    appeared_features: set[int] = field(default_factory=set)

    diff_type: FrameDiffType = FrameDiffType.NO_CHANGE
    bounding_box: tuple[int, int, int, int] | None = None  # (min_r, max_r, min_c, max_c)
    translation_delta: tuple[int, int] | None = None  # (dr, dc) if rigid translation detected
    moved_object_feature: int | None = None  # feature/color of the translated object
    moved_object_size: int = 0  # pixel count of the translated object

    def to_causal_predicate(self) -> CausalPredicate:
        """Export visual frame difference as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"observation_diff:{self.diff_type.value}",
            operator="==",
            value=self.changed_pixel_count,
            relation=f"trans:{self.translation_delta}" if self.translation_delta else "in_place",
            target_entity=(
                str(self.moved_object_feature) if self.moved_object_feature is not None else "none"
            ),
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording observation difference."""
        node_id = f"diff_{self.diff_type.value}_{self.changed_pixel_count}"
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=node_id,
                node_data={
                    "id": node_id,
                    "node_type": "observation",
                    "category": "perception",
                    "diff_type": self.diff_type.value,
                    "changed_pixel_count": self.changed_pixel_count,
                    "translation_delta": self.translation_delta,
                    "moved_object_feature": self.moved_object_feature,
                    "moved_object_size": self.moved_object_size,
                    "disappeared_features": sorted(self.disappeared_features),
                    "appeared_features": sorted(self.appeared_features),
                },
            )
        ]


@dataclass(frozen=True)
class OrientedThreat:
    """An embodied creature or entity with directional facing and visual gaze cone."""

    pos: tuple[int, int]
    facing: tuple[int, int]
    gaze_pos: tuple[int, int]
    feature_id: int

    def to_causal_predicate(self) -> CausalPredicate:
        """Export oriented creature threat as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"oriented_threat:{self.feature_id}",
            operator="==",
            value=f"facing:{self.facing}",
            relation=f"gaze:{self.gaze_pos}",
            target_entity=f"({self.pos[0]},{self.pos[1]})",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording oriented threat."""
        node_id = f"threat_{self.feature_id}_{self.pos[0]}_{self.pos[1]}"
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=node_id,
                node_data={
                    "id": node_id,
                    "node_type": "physical_entity",
                    "category": "perception",
                    "feature_id": self.feature_id,
                    "pos": self.pos,
                    "facing": self.facing,
                    "gaze_pos": self.gaze_pos,
                },
            )
        ]


class SensoryThalamus:
    """Universal sensory normalization converting multimodal signals to 2D spatial lattices."""

    @staticmethod
    def normalize(raw: Any) -> np.ndarray:
        """Normalize arbitrary sensory observations (camera, depth, lidar, 2D grid) into a spatial 2D array.

        Supports:
        - 2D integer grids (e.g. ARC, Roguelike, Atari RAM/grid)
        - 2D float arrays (e.g. Depth maps, LiDAR range grids) -> quantized to perceptual levels
        - 3D arrays (e.g. RGB camera (H, W, 3), LiDAR multi-channel, or temporal stack (T, H, W))
        - 3D Point Cloud (N, 3) (x, y, z) -> 2D Bird's-Eye View (BEV) occupancy grid
        - 1D LiDAR range arrays (N,) -> (1, N) range profile
        - Objects with .frame, .grid, .image, or .raw_data
        """
        if raw is None:
            return np.zeros((1, 1), dtype=int)

        if hasattr(raw, "raw_data") and raw.raw_data is not None:
            raw = raw.raw_data
        elif hasattr(raw, "frame") and raw.frame is not None:
            raw = raw.frame
        elif hasattr(raw, "grid") and raw.grid is not None:
            raw = raw.grid
        elif hasattr(raw, "image") and raw.image is not None:
            raw = raw.image

        if isinstance(raw, (list, tuple)):
            if len(raw) == 0:
                return np.zeros((1, 1), dtype=int)
            raw = np.asarray(raw)

        if not isinstance(raw, np.ndarray):
            try:
                raw = np.asarray(raw)
            except Exception:
                return np.zeros((1, 1), dtype=int)

        if raw.ndim == 3 and raw.shape[0] < min(raw.shape[1], raw.shape[2]):
            raw = raw[-1]

        if raw.ndim == 3 and raw.shape[2] in (1, 3, 4):
            if raw.shape[2] == 1:
                raw = raw[:, :, 0]
            elif raw.shape[2] in (3, 4):
                r = raw[:, :, 0].astype(float)
                g = raw[:, :, 1].astype(float)
                b = raw[:, :, 2].astype(float)
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                raw = np.digitize(lum, np.linspace(0, 255, 16)).astype(int)

        if raw.ndim == 2 and np.issubdtype(raw.dtype, np.floating):
            min_v, max_v = float(np.nanmin(raw)), float(np.nanmax(raw))
            if max_v > min_v:
                raw = np.digitize(raw, np.linspace(min_v, max_v, 16)).astype(int)
            else:
                raw = np.zeros(raw.shape, dtype=int)

        # 1D LiDAR range arrays (N,) -> (1, N) range profile
        if raw.ndim == 1:
            if np.issubdtype(raw.dtype, np.floating):
                min_v, max_v = float(np.nanmin(raw)), float(np.nanmax(raw))
                if max_v > min_v:
                    raw = np.digitize(raw, np.linspace(min_v, max_v, 16)).astype(int)
                else:
                    raw = np.zeros(raw.shape, dtype=int)
            raw = raw.reshape((1, -1))

        # 3D Point Cloud (N, 3) (x, y, z) -> 2D Bird's-Eye View (BEV) occupancy grid
        elif raw.ndim == 2 and raw.shape[1] == 3 and raw.shape[0] > 3:
            xs, ys, _zs = raw[:, 0], raw[:, 1], raw[:, 2]
            grid_res = 32
            min_x, max_x = float(np.min(xs)), float(np.max(xs))
            min_y, max_y = float(np.min(ys)), float(np.max(ys))
            if max_x > min_x and max_y > min_y:
                xi = np.clip(
                    np.digitize(xs, np.linspace(min_x, max_x, grid_res)) - 1, 0, grid_res - 1
                )
                yi = np.clip(
                    np.digitize(ys, np.linspace(min_y, max_y, grid_res)) - 1, 0, grid_res - 1
                )
                bev = np.zeros((grid_res, grid_res), dtype=int)
                bev[yi, xi] = 1
                raw = bev
            else:
                raw = np.zeros((grid_res, grid_res), dtype=int)

        if raw.ndim != 2:
            if raw.ndim > 2:
                raw = raw.reshape((raw.shape[0], -1))
            else:
                raw = np.zeros((1, 1), dtype=int)

        return raw.astype(int)


class FigureGroundSegregator:
    """Domain-agnostic background color estimation based on border dominance and frequency."""

    _BG_CACHE: OrderedDict[tuple[Any, ...], int] = OrderedDict()
    _BG_CACHE_MAX = 64

    @staticmethod
    def estimate_background(
        grid: np.ndarray,
        barrier_features: set[int] | None = None,
        avatar_features: set[int] | None = None,
        bg_feature: int = 0,
        walkable_features: set[int] | None = None,
    ) -> int:
        grid = SensoryThalamus.normalize(grid)
        barriers = barrier_features or set()
        av_feats = avatar_features or set()
        walkables = walkable_features or set()
        cache_key = (
            grid.shape,
            str(grid.dtype),
            frozenset(barriers),
            frozenset(av_feats),
            frozenset(walkables),
            grid.tobytes(),
        )
        cached = FigureGroundSegregator._BG_CACHE.get(cache_key)
        if cached is not None:
            FigureGroundSegregator._BG_CACHE.move_to_end(cache_key)
            return cached
        res = FigureGroundSegregator._estimate_background_uncached(
            grid, barriers, av_feats, bg_feature, walkables
        )
        FigureGroundSegregator._BG_CACHE[cache_key] = res
        if len(FigureGroundSegregator._BG_CACHE) > FigureGroundSegregator._BG_CACHE_MAX:
            FigureGroundSegregator._BG_CACHE.popitem(last=False)
        return res

    @staticmethod
    def _estimate_background_uncached(
        grid: np.ndarray,
        barriers: set[int],
        av_feats: set[int],
        bg_feature: int,
        walkables: set[int],
    ) -> int:
        H, W = grid.shape
        border_pixels = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
        vals, counts = _fast_unique_counts(border_pixels)
        top_val = int(vals[np.argmax(counts)])

        g_vals, g_counts = _fast_unique_counts(grid)
        g_dominant = int(g_vals[np.argmax(g_counts)])

        # 1. Empirically Verified Walkable Ground:
        # If the organism has traversed a feature present in the grid, that navigable ground
        # takes priority as the spatial background canvas over non-walkables.
        if walkables:
            valid_walkables = [w for w in walkables if w not in av_feats and w not in barriers]
            if valid_walkables:
                walk_counts = {
                    w: int(np.sum(grid == w)) for w in valid_walkables if np.any(grid == w)
                }
                if walk_counts:
                    best_w, best_cnt = max(walk_counts.items(), key=lambda item: item[1])
                    if top_val not in valid_walkables and (
                        top_val in barriers
                        or top_val != g_dominant
                        or best_cnt >= max(3, int(0.15 * grid.size))
                    ):
                        return best_w

        # 2. Gestalt Perimeter Enclosure vs Canvas Ground:
        # If the outer border is dominated by top_val (an enclosing bounding wall/frame),
        # verify if the interior canvas is dominated by a non-barrier/non-avatar floor feature.
        if H > 2 and W > 2:
            interior = grid[1:-1, 1:-1]
            int_vals, int_counts = _fast_unique_counts(interior)
            int_dominant = int(int_vals[np.argmax(int_counts)])
            int_dom_frac = float(np.max(int_counts)) / max(1, interior.size)
            if (
                top_val != g_dominant
                and g_dominant not in av_feats
                and g_dominant not in barriers
                and int_dominant == g_dominant
                and int_dom_frac > 0.40
            ):
                return g_dominant
            top_border_frac = float(np.max(counts)) / max(1, border_pixels.size)
            top_interior_frac = float(np.sum(interior == top_val)) / max(1, interior.size)
            int_border_frac = float(np.sum(border_pixels == int_dominant)) / max(
                1, border_pixels.size
            )
            if (
                int_dominant != top_val
                and int_dominant not in av_feats
                and int_dominant not in barriers
                and top_border_frac >= 0.60
                and top_interior_frac < 0.30
                and int_dom_frac >= 0.65
                and int_border_frac < 0.10
                and int(np.sum(grid == int_dominant)) >= 0.50 * int(np.sum(grid == top_val))
            ):
                return int_dominant

        if top_val in barriers:
            if top_val == g_dominant and g_counts[np.argmax(g_counts)] > 0.35 * grid.size:
                pass
            else:
                non_barriers = [
                    i
                    for i, v in enumerate(vals)
                    if int(v) not in barriers and int(v) not in av_feats
                ]
                if non_barriers:
                    return int(vals[non_barriers[np.argmax(counts[non_barriers])]])
                g_valid = [
                    i
                    for i, v in enumerate(g_vals)
                    if int(v) not in barriers and int(v) not in av_feats
                ]
                if g_valid:
                    return int(g_vals[g_valid[np.argmax(g_counts[g_valid])]])

        if av_feats and len(vals) > 1:
            other_idx = [i for i, v in enumerate(vals) if v not in av_feats and v not in barriers]
            if other_idx:
                return int(vals[other_idx[np.argmax(counts[other_idx])]])
        return top_val


class SpatialRoleClassifier:
    """Classifies segmented spatial visual entities into functional cognitive roles."""

    @staticmethod
    def classify_role(
        val: int,
        area: int,
        bounding_box: tuple[int, int, int, int],
        grid_pos: tuple[int, int],
        H: int,
        W: int,
        symbolic_theory: Any = None,
        av_feats: set[int] | None = None,
        mutations: list[Any] | None = None,
        goal_pos: set[tuple[int, int]] | None = None,
        receptacle_pos: set[tuple[int, int]] | None = None,
        avatar_pos: tuple[int, int] | None = None,
        cells: list[tuple[int, int]] | None = None,
        bg: int = 0,
    ) -> EntityRole:
        min_r, max_r, min_c, max_c = bounding_box
        av_set = av_feats or set()
        mutation_list = mutations or []
        goal_set = goal_pos or set()
        receptacle_set = receptacle_pos or set()

        is_border = (
            (min_r <= 1 and max_r >= H - 2 and min_c <= 1 and max_c >= W - 2)
            or (max_r - min_r >= H - 2 and max_c - min_c >= W - 2)
            or (area > H * W * 0.20)
            or (
                area >= max(4, int(H * W * 0.025))
                and (max_r - min_r >= H // 3 or max_c - min_c >= W // 3)
            )
        )

        is_theory_barrier = symbolic_theory.is_barrier(val) if symbolic_theory else False
        is_theory_actuator = val in symbolic_theory.actuator_features if symbolic_theory else False
        is_theory_cargo = symbolic_theory.is_cargo(val) if symbolic_theory else False
        is_theory_walkable = val in symbolic_theory.walkable_features if symbolic_theory else False
        is_candidate_goal = (
            val != bg
            and symbolic_theory is not None
            and (
                val in symbolic_theory.goal_features
                or val in symbolic_theory.candidate_goal_features
                or val in symbolic_theory.receptacle_features
            )
        )
        cell_set = set(cells) if cells is not None else None
        touches_goal = cell_set is not None and not goal_set.isdisjoint(cell_set)
        is_spatial_goal = (
            (grid_pos in goal_set or touches_goal) and (cell_set is None or touches_goal)
        ) or grid_pos in receptacle_set

        is_confirmed_goal = bool(symbolic_theory and val in symbolic_theory.goal_features)
        has_known_goals = bool(
            symbolic_theory
            and (symbolic_theory.goal_features or symbolic_theory.receptacle_features)
        )
        if val in av_set or (
            avatar_pos is not None
            and (grid_pos == avatar_pos or (cell_set is not None and avatar_pos in cell_set))
        ):
            return EntityRole.AGENT
        elif is_confirmed_goal:
            return EntityRole.GOAL
        elif is_theory_barrier or is_border:
            return EntityRole.OBSTACLE
        elif is_theory_cargo:
            return EntityRole.MANIPULABLE
        elif is_spatial_goal and (not has_known_goals or val in symbolic_theory.goal_features):
            return EntityRole.GOAL
        elif is_candidate_goal:
            return EntityRole.UNKNOWN
        elif is_theory_actuator or (
            not is_theory_walkable
            and any(
                (
                    getattr(m, "trigger_feature", None) is not None
                    and getattr(m, "trigger_feature", None) == val
                )
                or (
                    getattr(m, "trigger_pos", None) == grid_pos
                    and getattr(m, "occurrences", 0) >= 2
                    and getattr(m, "confidence", 0.0) >= 0.75
                )
                for m in mutation_list
            )
        ):
            return EntityRole.ACTUATOR
        else:
            return EntityRole.UNKNOWN


class GestaltAffordanceDetector:
    """Extracts structural goals, affordance panels, and oriented threats using Gestalt principles."""

    _STRUCTURAL_GOAL_CACHE: OrderedDict[tuple[Any, ...], list[dict[str, Any]]] = OrderedDict()
    _STRUCTURAL_GOAL_CACHE_MAX = 64

    @staticmethod
    def detect_structural_goals(
        grid: np.ndarray,
        bg: int = 0,
        avatar_features: set[int] | None = None,
        avatar_feature: int | None = None,
        barrier_features: set[int] | None = None,
        known_lethal_features: set[int] | None = None,
        avatar_pos: tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        grid = SensoryThalamus.normalize(grid)
        H, W = grid.shape
        av_feats = set(avatar_features) if avatar_features else set()
        if avatar_feature is not None:
            av_feats.add(avatar_feature)

        cache_key: tuple[Any, ...] | None = None
        if not kwargs:
            cache_key = (
                (H, W),
                str(grid.dtype),
                int(bg),
                frozenset(av_feats),
                frozenset(barrier_features or ()),
                frozenset(known_lethal_features or ()),
                avatar_pos,
                grid.tobytes(),
            )
            cached = GestaltAffordanceDetector._STRUCTURAL_GOAL_CACHE.get(cache_key)
            if cached is not None:
                GestaltAffordanceDetector._STRUCTURAL_GOAL_CACHE.move_to_end(cache_key)
                return list(cached)

        goals: list[dict[str, Any]] = []
        unique_feats, feat_counts = _fast_unique_counts(grid)
        max_pts = int(H * W * 0.3)

        # 1. Detect target zones and salient interior constellations
        for feat, fcount in zip(unique_feats.tolist(), feat_counts.tolist()):
            feat_int = int(feat)
            if (
                feat_int == bg
                or feat_int in av_feats
                or (barrier_features and feat_int in barrier_features)
                or (known_lethal_features and feat_int in known_lethal_features)
                or fcount < 1
                or fcount > max_pts
            ):
                continue
            pts = np.argwhere(grid == feat_int)

            min_r, min_c = pts.min(axis=0)
            max_r, max_c = pts.max(axis=0)

            is_interior = min_r > 1 and max_r < H - 2 and min_c > 1 and max_c < W - 2
            rect_area = (max_r - min_r + 1) * (max_c - min_c + 1)
            fill_ratio = len(pts) / max(1, rect_area)

            is_target_zone = fill_ratio > 0.7 and 4 <= len(pts) <= H * W * 0.15
            is_constellation = (
                1 <= len(pts) <= 12 and (max_r - min_r < H // 3) and (max_c - min_c < W // 3)
            )

            if is_interior and (is_target_zone or is_constellation):
                centroid = (
                    int(round(float(pts[:, 0].mean()))),
                    int(round(float(pts[:, 1].mean()))),
                )
                target_cells = list(zip(pts[:, 0].tolist(), pts[:, 1].tolist()))
                if avatar_pos is not None:
                    target_cells = [
                        c
                        for c in target_cells
                        if not (abs(c[0] - avatar_pos[0]) <= 1 and abs(c[1] - avatar_pos[1]) <= 1)
                    ]
                if not target_cells:
                    continue
                if (
                    avatar_pos is not None
                    and abs(centroid[0] - avatar_pos[0]) <= 1
                    and abs(centroid[1] - avatar_pos[1]) <= 1
                ):
                    continue
                goals.append(
                    {
                        "type": "target_zone" if is_target_zone else "constellation_target",
                        "position": centroid,
                        "cells": target_cells,
                        "feature": feat_int,
                        "size": len(target_cells),
                        "confidence": 0.85 if is_constellation else 0.8,
                    }
                )

        # 2. Detect edge and near-edge exit markers
        for feat, fcount in zip(unique_feats.tolist(), feat_counts.tolist()):
            feat_int = int(feat)
            if (
                feat_int == bg
                or feat_int in av_feats
                or (barrier_features and feat_int in barrier_features)
                or (known_lethal_features and feat_int in known_lethal_features)
            ):
                continue

            if fcount == 1:
                rs1, cs1 = np.nonzero(grid == feat_int)
                if rs1.size == 0:
                    continue
                r0, c0 = int(rs1[0]), int(cs1[0])
                if r0 == 0 or r0 == H - 1 or c0 == 0 or c0 == W - 1:
                    if not (min(H, W) <= 3):
                        goals.append(
                            {
                                "type": "exit_marker",
                                "position": (r0, c0),
                                "cells": [(r0, c0)],
                                "feature": feat_int,
                                "size": 1,
                                "confidence": 0.75,
                            }
                        )
                continue

            feat_mask = grid == feat_int
            labeled_feat, num_comp = label(feat_mask, structure=_STRUCT_4)
            for c_id in range(1, num_comp + 1):
                pts = np.argwhere(labeled_feat == c_id)
                max_strip_len = max(16, max(H, W) * 2)
                if len(pts) < 1 or len(pts) > max_strip_len:
                    continue

                touches_edge = any(r == 0 or r == H - 1 or c == 0 or c == W - 1 for r, c in pts)
                if not touches_edge:
                    continue

                # Filter out complete perimeter walls and full-edge boundary borders
                r_coords = pts[:, 0]
                c_coords = pts[:, 1]
                covers_all_4 = (
                    np.any(r_coords == 0)
                    and np.any(r_coords == H - 1)
                    and np.any(c_coords == 0)
                    and np.any(c_coords == W - 1)
                    and len(pts) >= 2 * (H + W) - 8
                )
                spans_full_edge = (
                    (int(r_coords.max()) - int(r_coords.min()) >= H - 2)
                    or (int(c_coords.max()) - int(c_coords.min()) >= W - 2)
                    or len(pts) >= min(H, W) - 2
                )
                if covers_all_4 or spans_full_edge:
                    continue

                cells = list(zip(r_coords.tolist(), c_coords.tolist()))
                centroid = (
                    int(round(float(r_coords.mean()))),
                    int(round(float(c_coords.mean()))),
                )
                goals.append(
                    {
                        "type": "exit_marker",
                        "position": centroid,
                        "cells": cells,
                        "feature": feat_int,
                        "size": len(pts),
                        "confidence": 0.75,
                    }
                )

        # 2B. Detect perimeter gateway breach / doorway notches
        from collections import Counter

        for edge_cells in [
            [(0, c) for c in range(W)],
            [(H - 1, c) for c in range(W)],
            [(r, 0) for r in range(H)],
            [(r, W - 1) for r in range(H)],
        ]:
            edge_vals = [int(grid[r, c]) for r, c in edge_cells]
            val_counts = Counter(edge_vals)
            if not val_counts:
                continue
            dom_val, count = val_counts.most_common(1)[0]
            is_valid_wall = dom_val in barrier_features if barrier_features else (dom_val != bg)
            if count >= len(edge_cells) * 0.6 and is_valid_wall:
                gap_cells = [
                    edge_cells[i]
                    for i, v in enumerate(edge_vals)
                    if v != dom_val and 0 < i < len(edge_cells) - 1
                ]
                if 1 <= len(gap_cells) <= 4:
                    mid_r, mid_c = gap_cells[len(gap_cells) // 2]
                    interior_r = 1 if mid_r == 0 else (H - 2 if mid_r == H - 1 else mid_r)
                    interior_c = 1 if mid_c == 0 else (W - 2 if mid_c == W - 1 else mid_c)
                    int_val = int(grid[interior_r, interior_c])
                    if (
                        (barrier_features and int_val in barrier_features)
                        or (known_lethal_features and int_val in known_lethal_features)
                        or (dom_val in (barrier_features or set()) and int_val == dom_val)
                    ):
                        continue
                    goals.append(
                        {
                            "type": "gateway_notch",
                            "position": (mid_r, mid_c),
                            "cells": gap_cells,
                            "feature": int(grid[gap_cells[0]]),
                            "confidence": 0.85,
                        }
                    )

        # 3. Check for symmetry-completion goals on foreground pattern
        sym_type, sym_score = VisualSymmetryAnalyzer.find_dominant_symmetry(
            grid, background_color=bg
        )
        if 0.60 <= sym_score < 0.99:
            completed_grid = VisualSymmetryAnalyzer.predict_symmetric_completion(
                grid, symmetry_type=sym_type, background_color=bg
            )
            diff_mask = (grid != completed_grid) & (completed_grid != bg)
            missing_pts = np.argwhere(diff_mask)
            if 1 <= len(missing_pts) <= 16:
                for pt in missing_pts:
                    pr, pc = int(pt[0]), int(pt[1])
                    target_val = int(completed_grid[pr, pc])
                    goals.append(
                        {
                            "type": "symmetry_completion",
                            "position": (pr, pc),
                            "feature": target_val,
                            "symmetry_axis": sym_type,
                            "symmetry_score": sym_score,
                            "confidence": float(sym_score * 0.95),
                        }
                    )

        # 4. Gestalt Relational Affordances: Symmetry completion and collinear alignment
        if 0.50 <= sym_score:
            salient_clusters: list[tuple[int, tuple[int, int], np.ndarray]] = []
            max_rel_area = max(48, int(H * W * 0.08))
            for feat, fcount in zip(unique_feats.tolist(), feat_counts.tolist()):
                feat_int = int(feat)
                if (
                    feat_int == bg
                    or feat_int in av_feats
                    or (barrier_features and feat_int in barrier_features)
                    or (known_lethal_features and feat_int in known_lethal_features)
                    or fcount > max_rel_area
                ):
                    continue
                mask = grid == feat_int
                labeled_mask, num_clusters = label(mask, structure=_STRUCT_4)
                # Pervasive substrates, lattices, or maze partitions (>4 clusters or >8% screen area)
                # cannot be isolated salient relational endpoints
                if num_clusters > 4:
                    continue
                for c_idx in range(1, num_clusters + 1):
                    c_pts = np.argwhere(labeled_mask == c_idx)
                    if 1 <= len(c_pts) <= 36:
                        c_rs = c_pts[:, 0]
                        c_cs = c_pts[:, 1]
                        min_cr, max_cr = int(c_rs.min()), int(c_rs.max())
                        min_cc, max_cc = int(c_cs.min()), int(c_cs.max())
                        # Filter outer enclosing boundary edges
                        if min_cr <= 0 or max_cr >= H - 1 or min_cc <= 0 or max_cc >= W - 1:
                            continue
                        if (max_cr - min_cr >= H - 4) or (max_cc - min_cc >= W - 4):
                            continue
                        c_centroid = (
                            int(round(float(c_rs.mean()))),
                            int(round(float(c_cs.mean()))),
                        )
                        salient_clusters.append((feat_int, c_centroid, c_pts))

            # 3a. Dominant Axis Symmetry Relational Projections
            for feat_int, (cr, cc), _ in salient_clusters:
                ref_r, ref_c = cr, cc
                if sym_type == "vertical":
                    ref_c = W - 1 - cc
                elif sym_type == "horizontal":
                    ref_r = H - 1 - cr
                elif sym_type == "main_diagonal" and H == W:
                    ref_r, ref_c = cc, cr
                elif sym_type == "anti_diagonal" and H == W:
                    ref_r, ref_c = W - 1 - cc, H - 1 - cr
                else:
                    continue

                if 0 <= ref_r < H and 0 <= ref_c < W and (ref_r, ref_c) != (cr, cc):
                    target_val = int(grid[ref_r, ref_c])
                    if target_val != feat_int:
                        is_void = target_val == bg
                        if is_void and sym_score < 0.85:
                            continue
                        goals.append(
                            {
                                "type": "relational_alignment",
                                "position": (ref_r, ref_c),
                                "feature": feat_int,
                                "confidence": float(sym_score * 0.90)
                                if not is_void
                                else float(sym_score * 0.35),
                                "is_void": is_void,
                            }
                        )

            # 3b. Collinear Midpoints & Relational Alignment between distinct entity pairs
            for i in range(len(salient_clusters)):
                for j in range(i + 1, len(salient_clusters)):
                    f1, p1, _ = salient_clusters[i]
                    f2, p2, _ = salient_clusters[j]
                    if f1 == f2 or f1 in av_feats or f2 in av_feats:
                        continue
                    is_collinear_h = (p1[0] == p2[0]) and abs(p1[1] - p2[1]) >= 4
                    is_collinear_v = (p1[1] == p2[1]) and abs(p1[0] - p2[0]) >= 4
                    if is_collinear_h or is_collinear_v:
                        mid_r = (p1[0] + p2[0]) // 2
                        mid_c = (p1[1] + p2[1]) // 2
                        if 0 <= mid_r < H and 0 <= mid_c < W:
                            goals.append(
                                {
                                    "type": "relational_alignment",
                                    "position": (mid_r, mid_c),
                                    "feature": None,
                                    "confidence": 0.85,
                                    "is_midpoint": True,
                                }
                            )

        if cache_key is not None:
            GestaltAffordanceDetector._STRUCTURAL_GOAL_CACHE[cache_key] = list(goals)
            if (
                len(GestaltAffordanceDetector._STRUCTURAL_GOAL_CACHE)
                > GestaltAffordanceDetector._STRUCTURAL_GOAL_CACHE_MAX
            ):
                GestaltAffordanceDetector._STRUCTURAL_GOAL_CACHE.popitem(last=False)
        return goals

    @staticmethod
    def detect_affordance_clusters(
        entities: Sequence[SpatialEntity],
        grid: np.ndarray,
        bg: int = 0,
    ) -> list[dict[str, Any]]:
        """Identify spatially grouped entities sharing bounding-box morphology (interactive arrays)."""
        by_dim: dict[tuple[int, int], list[SpatialEntity]] = defaultdict(list)
        for e in entities:
            if getattr(e, "role", None) in (EntityRole.OBSTACLE, EntityRole.AGENT):
                continue
            r0, r1, c0, c1 = e.bounding_box
            h, w = r1 - r0 + 1, c1 - c0 + 1
            by_dim[(h, w)].append(e)

        clusters: list[dict[str, Any]] = []
        for (h, w), group in by_dim.items():
            if len(group) < 2 or len(group) > 16:
                continue

            feat_counts: dict[int, int] = defaultdict(int)
            for e in group:
                feat_counts[e.feature_id] += 1
            majority_feat = max(feat_counts.keys(), key=lambda k: feat_counts[k])
            minority_items = [e for e in group if e.feature_id != majority_feat]

            clusters.append(
                {
                    "shape": (h, w),
                    "items": group,
                    "majority_feature": majority_feat,
                    "minority_items": minority_items,
                    "item_coords": [e.grid_pos for e in group],
                }
            )
        return clusters

    _ORIENTED_THREAT_CACHE: OrderedDict[tuple[Any, ...], list[OrientedThreat]] = OrderedDict()
    _ORIENTED_THREAT_CACHE_MAX: int = 64

    @staticmethod
    def detect_oriented_threats(
        grid: np.ndarray,
        entities: list[SpatialEntity],
        bg: int,
        step_size: int = 1,
        avatar_pos: tuple[int, int] | None = None,
        avatar_features: set[int] | None = None,
        goals: set[tuple[int, int]] | None = None,
        cargo_features: set[int] | None = None,
    ) -> list[OrientedThreat]:
        av_feats = avatar_features or set()
        goal_set = goals or set()
        cargos = cargo_features or set()

        # Fast short-circuit if no entity has a plausible multi-cell body bounding box
        candidate_sig: list[tuple[int, int, tuple[int, int, int, int]]] = []
        for e in entities:
            if e.area < 2 or e.area > 36:
                continue
            if getattr(e, "role", None) == EntityRole.MANIPULABLE:
                continue
            r0, r1, c0, c1 = e.bounding_box
            if not (1 <= r1 - r0 <= 6 and 1 <= c1 - c0 <= 6):
                continue
            candidate_sig.append((e.feature_id, e.area, e.bounding_box))
        if not candidate_sig:
            return []

        cache = GestaltAffordanceDetector._ORIENTED_THREAT_CACHE
        cache_key = (
            grid.shape,
            grid.dtype.str,
            grid.tobytes(),
            int(bg),
            int(step_size),
            avatar_pos,
            frozenset(av_feats),
            frozenset(goal_set),
            frozenset(cargos),
            tuple(candidate_sig),
        )
        cached = cache.get(cache_key)
        if cached is not None:
            cache.move_to_end(cache_key)
            return list(cached)

        oriented_threats: list[OrientedThreat] = []

        for e in entities:
            if e.area < 2 or e.area > 36:
                continue
            if getattr(e, "role", None) == EntityRole.MANIPULABLE:
                continue
            r0, r1, c0, c1 = e.bounding_box
            if not (1 <= r1 - r0 <= 6 and 1 <= c1 - c0 <= 6):
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            vals, counts = np.unique(patch, return_counts=True)
            if len(vals) == 2 and np.min(counts) <= max(2, int(e.area * 0.35)):
                body_val = int(vals[np.argmax(counts)])
                eye_val = int(vals[np.argmin(counts)])
                if (
                    body_val == bg
                    or body_val in av_feats
                    or body_val in cargos
                    or e.feature_id in cargos
                ):
                    continue
                center = ((r0 + r1) // 2, (c0 + c1) // 2)
                if center == avatar_pos or center in goal_set:
                    continue
                eye_coords = np.argwhere(patch == eye_val)
                hdr = int(np.sign(np.mean(eye_coords[:, 0]) - (r1 - r0) / 2))
                hdc = int(np.sign(np.mean(eye_coords[:, 1]) - (c1 - c0) / 2))
                if hdr != 0 or hdc != 0:
                    if step_size >= max(r1 - r0 + 1, c1 - c0 + 1):
                        gaze_cell = (center[0] + hdr * step_size, center[1] + hdc * step_size)
                    else:
                        front_r = r1 if hdr > 0 else (r0 if hdr < 0 else center[0])
                        front_c = c1 if hdc > 0 else (c0 if hdc < 0 else center[1])
                        gaze_cell = (front_r + hdr * step_size, front_c + hdc * step_size)
                    oriented_threats.append(
                        OrientedThreat(
                            pos=center,
                            facing=(hdr, hdc),
                            gaze_pos=gaze_cell,
                            feature_id=body_val,
                        )
                    )
        cache[cache_key] = list(oriented_threats)
        if len(cache) > GestaltAffordanceDetector._ORIENTED_THREAT_CACHE_MAX:
            cache.popitem(last=False)
        return oriented_threats


class PerceptionEngine:
    """Domain-agnostic visual perception, background estimation, and entity segmentation."""

    @staticmethod
    def normalize_sensory_input(raw: Any) -> np.ndarray:
        return SensoryThalamus.normalize(raw)

    @staticmethod
    def estimate_background(
        grid: np.ndarray,
        barrier_features: set[int] | None = None,
        avatar_features: set[int] | None = None,
        bg_feature: int = 0,
        walkable_features: set[int] | None = None,
    ) -> int:
        return FigureGroundSegregator.estimate_background(
            grid=grid,
            barrier_features=barrier_features,
            avatar_features=avatar_features,
            bg_feature=bg_feature,
            walkable_features=walkable_features,
        )

    _COMPONENT_GEOM_CACHE: OrderedDict[
        tuple[tuple[int, ...], str, int, bytes],
        tuple[
            tuple[
                int,
                int,
                tuple[int, int, int, int],
                tuple[float, float],
                tuple[int, int],
                tuple[tuple[int, int], ...],
            ],
            ...,
        ],
    ] = OrderedDict()
    _COMPONENT_GEOM_CACHE_MAX = 128

    @classmethod
    def _get_cached_component_geometry(
        cls,
        grid: np.ndarray,
        bg: int,
        unique_vals: list[int],
    ) -> tuple[
        tuple[
            int,
            int,
            tuple[int, int, int, int],
            tuple[float, float],
            tuple[int, int],
            tuple[tuple[int, int], ...],
        ],
        ...,
    ]:
        """Compute or reuse pure geometric connected components for `(grid, bg)`."""
        cache_key = (grid.shape, str(grid.dtype), int(bg), grid.tobytes())
        cached = cls._COMPONENT_GEOM_CACHE.get(cache_key)
        if cached is not None:
            cls._COMPONENT_GEOM_CACHE.move_to_end(cache_key)
            return cached

        H, W = grid.shape
        components: list[
            tuple[
                int,
                int,
                tuple[int, int, int, int],
                tuple[float, float],
                tuple[int, int],
                tuple[tuple[int, int], ...],
            ]
        ] = []
        for val in unique_vals:
            mask = grid == val
            nz_r, nz_c = np.nonzero(mask)
            n_px = int(nz_r.size)
            if n_px == 0:
                continue
            if n_px == 1:
                r0 = int(nz_r[0])
                c0 = int(nz_c[0])
                components.append(
                    (val, 1, (r0, r0, c0, c0), (float(r0), float(c0)), (r0, c0), ((r0, c0),))
                )
                continue
            labeled_mask, num_components = label(mask, structure=_STRUCT_4)
            _flat_labels = labeled_mask.ravel()
            _nz = np.nonzero(_flat_labels)[0]
            _order = np.argsort(_flat_labels[_nz], kind="stable")
            _nz = _nz[_order]
            _nz_labels = _flat_labels[_nz]
            _splits = np.flatnonzero(np.diff(_nz_labels)) + 1
            _groups = np.split(_nz, _splits)
            for comp_idx in range(1, num_components + 1):
                flat = _groups[comp_idx - 1]
                rows_arr = flat // W
                cols_arr = flat % W
                area = int(flat.size)
                if area == 0:
                    continue
                cells = tuple(zip(rows_arr.tolist(), cols_arr.tolist()))
                min_r = int(rows_arr.min())
                max_r = int(rows_arr.max())
                min_c = int(cols_arr.min())
                max_c = int(cols_arr.max())
                mean_r = float(rows_arr.mean())
                mean_c = float(cols_arr.mean())
                centroid = (mean_r, mean_c)
                grid_pos = (int(round(mean_r)), int(round(mean_c)))
                components.append(
                    (val, area, (min_r, max_r, min_c, max_c), centroid, grid_pos, cells)
                )

        result = tuple(components)
        cls._COMPONENT_GEOM_CACHE[cache_key] = result
        if len(cls._COMPONENT_GEOM_CACHE) > cls._COMPONENT_GEOM_CACHE_MAX:
            cls._COMPONENT_GEOM_CACHE.popitem(last=False)
        return result

    @staticmethod
    def extract_entities(
        grid: np.ndarray,
        bg: int,
        symbolic_theory: Any = None,
        avatar_features: set[int] | None = None,
        state_mutations: list[Any] | None = None,
        learned_barriers: set[tuple[int, int]] | None = None,
        learned_goal_positions: set[tuple[int, int]] | None = None,
        learned_receptacle_positions: set[tuple[int, int]] | None = None,
        avatar_pos: tuple[int, int] | None = None,
    ) -> list[SpatialEntity]:
        """Domain-agnostic connected-component entity segmentation using native substrate labeling."""
        grid = SensoryThalamus.normalize(grid)
        H, W = grid.shape
        av_feats = avatar_features or set()
        mutations = state_mutations or []
        goal_pos = learned_goal_positions or set()
        receptacle_pos = learned_receptacle_positions or set()
        entities: list[SpatialEntity] = []

        unique_vals = [int(v) for v in _fast_unique_counts(grid)[0].tolist() if int(v) != bg]
        geom_comps = PerceptionEngine._get_cached_component_geometry(grid, bg, unique_vals)
        for val, area, bbox, centroid, grid_pos, cells in geom_comps:
            cells_list = list(cells)
            min_r, max_r, min_c, max_c = bbox

            role = SpatialRoleClassifier.classify_role(
                val=val,
                area=area,
                bounding_box=bbox,
                grid_pos=grid_pos,
                H=H,
                W=W,
                symbolic_theory=symbolic_theory,
                av_feats=av_feats,
                mutations=mutations,
                goal_pos=goal_pos,
                receptacle_pos=receptacle_pos,
                avatar_pos=avatar_pos,
                cells=cells_list,
                bg=bg,
            )

            ent = SpatialEntity(
                id=f"ent_{val}_{len(entities)}_{min_r}_{min_c}",
                role=role,
                centroid=centroid,
                grid_pos=grid_pos,
                area=area,
                bounding_box=bbox,
                feature_id=val,
                properties={"cells": cells_list, "feature": val, "feature_id": val},
            )
            if role == EntityRole.OBSTACLE and symbolic_theory is not None:
                if val in symbolic_theory.candidate_goal_features:
                    symbolic_theory.retract_candidate_goal(val, reason="perimeter_border_obstacle")
                if not symbolic_theory.is_barrier(val):
                    symbolic_theory.induce_barrier(val, evidence="topology:border")
            entities.append(ent)

        # Disambiguate when multiple distinct entities share avatar features (e.g. templates, twins, sockets)
        agent_entities = [e for e in entities if e.feature_id in av_feats]
        if len(agent_entities) > 1 and avatar_pos is not None:
            exact_matches = [
                e
                for e in agent_entities
                if avatar_pos in e.properties.get("cells", [e.grid_pos]) or e.grid_pos == avatar_pos
            ]
            best_agent = (
                exact_matches[0]
                if exact_matches
                else min(
                    agent_entities,
                    key=lambda e: (
                        abs(e.grid_pos[0] - avatar_pos[0]) + abs(e.grid_pos[1] - avatar_pos[1])
                    ),
                )
            )
            for e in agent_entities:
                if e is not best_agent:
                    if (
                        e.grid_pos in goal_pos
                        or e.grid_pos in receptacle_pos
                        or (symbolic_theory and symbolic_theory.is_goal(e.feature_id))
                    ):
                        e.role = EntityRole.GOAL
                    else:
                        e.role = EntityRole.MANIPULABLE

        return entities

    @staticmethod
    def compute_frame_diff(
        prev_grid: np.ndarray, curr_grid: np.ndarray
    ) -> EpistemicObservationDiff:
        """Compute structural difference between two observation frames."""
        if prev_grid.shape != curr_grid.shape:
            return EpistemicObservationDiff(
                changed_pixel_count=curr_grid.size,
                diff_type=FrameDiffType.GLOBAL_TRANSITION,
            )

        diff_mask = prev_grid != curr_grid
        changed_count = int(np.sum(diff_mask))

        if changed_count == 0:
            return EpistemicObservationDiff(
                changed_pixel_count=0,
                changed_mask=diff_mask,
                diff_type=FrameDiffType.NO_CHANGE,
            )

        total_pixels = prev_grid.size
        H, W = prev_grid.shape

        if changed_count > total_pixels * 0.45:
            return EpistemicObservationDiff(
                changed_pixel_count=changed_count,
                changed_mask=diff_mask,
                diff_type=FrameDiffType.GLOBAL_TRANSITION,
            )

        rows, cols = np.where(diff_mask)
        min_r, max_r = int(np.min(rows)), int(np.max(rows))
        min_c, max_c = int(np.min(cols)), int(np.max(cols))
        bbox = (min_r, max_r, min_c, max_c)

        if H >= 32 and W >= 32:
            is_all_margin = all(
                (r <= 1 or r >= H - 2 or c <= 1 or c >= W - 2) for r, c in zip(rows, cols)
            )
            if is_all_margin and changed_count <= 4:
                return EpistemicObservationDiff(
                    changed_pixel_count=0,
                    changed_mask=diff_mask,
                    diff_type=FrameDiffType.NO_CHANGE,
                )

        mutated: list[tuple[int, int, int, int]] = []
        for r, c in zip(rows, cols):
            mutated.append((int(r), int(c), int(prev_grid[r, c]), int(curr_grid[r, c])))

        prev_vals, prev_counts = _fast_unique_counts(prev_grid)
        curr_vals, _ = _fast_unique_counts(curr_grid)
        prev_features = set(int(v) for v in prev_vals.tolist())
        curr_features = set(int(v) for v in curr_vals.tolist())

        bg = int(prev_vals[np.argmax(prev_counts)]) if prev_vals.size else 0
        translation_candidates: list[tuple[int, int, int, int]] = []
        changed_feats = {old_v for _, _, old_v, _ in mutated} | {
            new_v for _, _, _, new_v in mutated
        }

        for feat_int in changed_feats:
            if feat_int == bg or feat_int not in prev_features:
                continue
            prev_pts = np.where(prev_grid == feat_int)
            curr_pts = np.where(curr_grid == feat_int)
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
                            translation_candidates.append((feat_int, np_c, dr, dc))

        if translation_candidates:
            translation_candidates.sort(key=lambda x: x[1])
            best_feat, best_size, dr, dc = translation_candidates[0]
            return EpistemicObservationDiff(
                changed_pixel_count=changed_count,
                changed_mask=diff_mask,
                mutated_pixels=mutated,
                disappeared_features=prev_features - curr_features,
                appeared_features=curr_features - prev_features,
                diff_type=FrameDiffType.TRANSLATION,
                bounding_box=bbox,
                translation_delta=(dr, dc),
                moved_object_feature=best_feat,
                moved_object_size=best_size,
            )

        if changed_count <= 8:
            diff_type = FrameDiffType.INDEX_CYCLE
        elif min_r > 0 and max_r < H - 1 and min_c > 0 and max_c < W - 1 and changed_count >= 5:
            diff_type = FrameDiffType.CANVAS_TRANSFORMATION
        else:
            diff_type = FrameDiffType.IN_PLACE_MUTATION

        return EpistemicObservationDiff(
            changed_pixel_count=changed_count,
            changed_mask=diff_mask,
            mutated_pixels=mutated,
            disappeared_features=prev_features - curr_features,
            appeared_features=curr_features - prev_features,
            diff_type=diff_type,
            bounding_box=bbox,
        )

    @staticmethod
    def detect_structural_goals(
        grid: np.ndarray,
        bg: int = 0,
        avatar_features: set[int] | None = None,
        avatar_feature: int | None = None,
        barrier_features: set[int] | None = None,
        known_lethal_features: set[int] | None = None,
        avatar_pos: tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return GestaltAffordanceDetector.detect_structural_goals(
            grid=grid,
            bg=bg,
            avatar_features=avatar_features,
            avatar_feature=avatar_feature,
            barrier_features=barrier_features,
            known_lethal_features=known_lethal_features,
            avatar_pos=avatar_pos,
            **kwargs,
        )

    @staticmethod
    def detect_affordance_clusters(
        entities: Sequence[SpatialEntity],
        grid: np.ndarray,
        bg: int = 0,
    ) -> list[dict[str, Any]]:
        return GestaltAffordanceDetector.detect_affordance_clusters(
            entities=entities,
            grid=grid,
            bg=bg,
        )

    @staticmethod
    def detect_oriented_threats(
        grid: np.ndarray,
        entities: list[SpatialEntity],
        bg: int,
        step_size: int = 1,
        avatar_pos: tuple[int, int] | None = None,
        avatar_features: set[int] | None = None,
        goals: set[tuple[int, int]] | None = None,
        cargo_features: set[int] | None = None,
    ) -> list[OrientedThreat]:
        return GestaltAffordanceDetector.detect_oriented_threats(
            grid=grid,
            entities=entities,
            bg=bg,
            step_size=step_size,
            avatar_pos=avatar_pos,
            avatar_features=avatar_features,
            goals=goals,
            cargo_features=cargo_features,
        )

    @classmethod
    def prune_topological_dead_ends(
        cls,
        passable: np.ndarray,
        goals: set[tuple[int, int]],
        avatar_pos: tuple[int, int] | None = None,
        step_size: int = 1,
    ) -> np.ndarray:
        """Iteratively prune non-goal dead-end branches (degree <= 1) from passable topology."""
        if not goals:
            return passable
        H, W = passable.shape
        pruned = passable.copy()
        protected = set(goals)
        if avatar_pos is not None:
            protected.add(avatar_pos)

        step = max(1, step_size)
        changed = True
        iterations = 0
        max_iters = max(H, W) * 2

        while changed and iterations < max_iters:
            changed = False
            iterations += 1
            for r in range(0, H, step):
                for c in range(0, W, step):
                    if not pruned[r, c] or (r, c) in protected:
                        continue
                    deg = 0
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr * step, c + dc * step
                        if 0 <= nr < H and 0 <= nc < W and pruned[nr, nc]:
                            deg += 1
                    if deg <= 1:
                        for br in range(r, min(H, r + step)):
                            for bc in range(c, min(W, c + step)):
                                pruned[br, bc] = False
                        changed = True
        return pruned

    @classmethod
    def to_causal_predicate(cls, view: Any = None) -> CausalPredicate:
        """Export perception engine capability and runtime sensory state as a declarative causal predicate."""
        val = "ACTIVE"
        rel = "thalamocortical_segmentation"
        if view is not None:
            bg = getattr(view, "bg_feature", None)
            av = getattr(view, "avatar_pos", None)
            rel = f"thalamocortical_segmentation:bg:{bg}:avatar:{av}"
        return CausalPredicate(
            variable="cortex_perception_engine",
            operator="==",
            value=val,
            relation=rel,
            target_entity="cortex_perception",
        )

    @classmethod
    def to_transaction_operations(cls) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction registering perception engine."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="perception_engine",
                node_data={
                    "id": "perception_engine",
                    "node_type": "capability",
                    "category": "perception",
                    "faculty": "cortex_perception",
                    "status": "ACTIVE",
                },
            )
        ]


@dataclass
class PerceptionStage:
    """Cortical Stage 1: Sensory Perception & Coordinate Lifting.

    Transforms raw sensory arrays into SpatialEntity objects and coordinate frames.
    """

    name: str = "perception_stage"
    engine: Any | None = None

    def bind_engine(self, engine: Any) -> None:
        self.engine = engine

    def observe(self, ctx: CognitiveContext) -> StageResult:
        grid = ctx.curr_grid if ctx.curr_grid is not None else ctx.signals.get("_curr_grid")
        if grid is None:
            grid = ctx.prev_grid
        entities: list[SpatialEntity] = []
        if grid is not None:
            try:
                import numpy as _np

                g = _np.asarray(grid)
                entities = ctx.extract_entities(g, ctx.bg_feature)
            except Exception:
                entities = []
        avatar_candidate = ctx.avatar_pos
        ops = [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="perception_observation",
                node_data={
                    "id": "perception_observation",
                    "node_type": "observation",
                    "category": "perception",
                    "entity_count": len(entities),
                    "step": ctx.step_counter,
                },
            )
        ]
        return StageResult(
            author=self.name,
            operations=ops,
            signals={
                "extracted_entities": entities,
                "entity_count": len(entities),
                "avatar_candidate": avatar_candidate,
                "step": ctx.step_counter,
            },
            proposed_action=None,
        )
