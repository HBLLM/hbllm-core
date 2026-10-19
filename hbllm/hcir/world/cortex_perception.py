from __future__ import annotations

"""Thalamocortical Visual Perception Faculty for HCIR World Kernel.

Biological Modeling:
1. Sensory Thalamus: Universal sensory normalization across camera, depth, lidar, and 2D arrays.
2. Ventral Visual Stream (V1-V4, IT):
   - Figure-ground segregation and background estimation.
   - Connected-component morphological entity segmentation using native vectorized labeling.
   - Gestalt grouping and relational alignment affordances (symmetry, collinearity).
   - Affordance panel array detection.
3. Dorsal Visual Stream & Theory of Mind:
   - Gaze cone and directional orientation tracking of sentient creatures.
   - Frame difference mutation analysis (translation, index cycle, global transition).
"""

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
from scipy.ndimage import label

from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.visual_symmetry import VisualSymmetryAnalyzer

logger = logging.getLogger(__name__)


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
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"diff_{self.diff_type.value}_{self.changed_pixel_count}",
                node_data={
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
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"threat_{self.feature_id}_{self.pos[0]}_{self.pos[1]}",
                node_data={
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

    @staticmethod
    def estimate_background(
        grid: np.ndarray,
        barrier_features: set[int] | None = None,
        avatar_features: set[int] | None = None,
        bg_feature: int = 0,
    ) -> int:
        H, W = grid.shape
        border_pixels = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
        vals, counts = np.unique(border_pixels, return_counts=True)
        top_val = int(vals[np.argmax(counts)])

        barriers = barrier_features or set()
        av_feats = avatar_features or set()

        if top_val in barriers:
            g_vals, g_counts = np.unique(grid, return_counts=True)
            g_dominant = int(g_vals[np.argmax(g_counts)])
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
    ) -> EntityRole:
        min_r, max_r, min_c, max_c = bounding_box
        av_set = av_feats or set()
        mutation_list = mutations or []
        goal_set = goal_pos or set()
        receptacle_set = receptacle_pos or set()

        is_border = (
            (min_r <= 1 and max_r >= H - 2 and min_c <= 1 and max_c >= W - 2)
            or (max_r - min_r >= H - 2 and max_c - min_c >= W - 2)
            or (area > H * W * 0.35)
        )

        is_edge_hud = (
            min_c == max_c and (min_c == 0 or min_c == W - 1) and (max_r - min_r >= 4)
        ) or (min_r == max_r and (min_r == 0 or min_r == H - 1) and (max_c - min_c >= 4))

        is_theory_barrier = symbolic_theory.is_barrier(val) if symbolic_theory else False
        is_theory_cargo = (
            symbolic_theory.is_cargo(
                SpatialEntity(
                    id="",
                    role=EntityRole.MANIPULABLE,
                    centroid=(float(grid_pos[0]), float(grid_pos[1])),
                    grid_pos=grid_pos,
                    area=area,
                    bounding_box=bounding_box,
                    feature_id=val,
                )
            )
            if symbolic_theory
            else False
        )
        is_theory_walkable = val in symbolic_theory.walkable_features if symbolic_theory else False
        is_theory_goal = (
            (
                val != 0
                and (
                    val in symbolic_theory.goal_features
                    or val in symbolic_theory.receptacle_features
                )
            )
            if symbolic_theory
            else False
        )

        if is_border or is_edge_hud or is_theory_barrier:
            return EntityRole.OBSTACLE
        elif val in av_set:
            return EntityRole.AGENT
        elif any(
            getattr(m, "trigger_feature", None) == val
            or getattr(m, "trigger_pos", None) == grid_pos
            for m in mutation_list
        ):
            return EntityRole.ACTUATOR
        elif is_theory_cargo and not is_theory_walkable:
            return EntityRole.MANIPULABLE
        elif (
            is_theory_goal
            or (
                (grid_pos in goal_set or any(p in goal_set for p in (cells or [])))
                and (cells is None or any(p in goal_set for p in cells))
            )
            or grid_pos in receptacle_set
        ):
            return EntityRole.GOAL
        elif area <= 16:
            return EntityRole.UNKNOWN
        else:
            return EntityRole.UNKNOWN


class GestaltAffordanceDetector:
    """Extracts structural goals, affordance panels, and oriented threats using Gestalt principles."""

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
        H, W = grid.shape
        goals: list[dict[str, Any]] = []
        av_feats = set(avatar_features) if avatar_features else set()
        if avatar_feature is not None:
            av_feats.add(avatar_feature)

        threat_feats = set(known_lethal_features or ())
        if kwargs.get("mobile_threat_features"):
            threat_feats.update(kwargs["mobile_threat_features"])
        if kwargs.get("threat_features"):
            threat_feats.update(kwargs["threat_features"])
        if kwargs.get("oriented_threat_features"):
            threat_feats.update(kwargs["oriented_threat_features"])

        # 1. Detect target zones (medium-sized non-border rectangular entities)
        for feat in np.unique(grid):
            feat_int = int(feat)
            if (
                feat_int == bg
                or feat_int == 0
                or feat_int in av_feats
                or feat_int in threat_feats
                or (barrier_features and feat_int in barrier_features)
            ):
                continue
            labeled_feat, num_comps = label(grid == feat_int)
            for c_idx in range(1, num_comps + 1):
                pts = np.argwhere(labeled_feat == c_idx)
                if len(pts) < 4 or len(pts) > H * W * 0.3:
                    continue

                min_r, min_c = pts.min(axis=0)
                max_r, max_c = pts.max(axis=0)

                is_interior = min_r > 1 and max_r < H - 2 and min_c > 1 and max_c < W - 2
                rect_area = (max_r - min_r + 1) * (max_c - min_c + 1)
                fill_ratio = len(pts) / max(1, rect_area)

                patch = grid[min_r : max_r + 1, min_c : max_c + 1]
                patch_vals = np.unique(patch)
                if any(pv in threat_feats for pv in patch_vals):
                    continue
                non_bg_vals = [v for v in patch_vals if v != bg and v != 0 and v != feat_int]
                if len(non_bg_vals) >= 2:
                    v_counts = [np.count_nonzero(patch == v) for v in non_bg_vals]
                    if min(v_counts) <= 2:
                        continue

                if is_interior and fill_ratio > 0.60 and 4 <= len(pts) <= H * W * 0.15:
                    centroid = (
                        int(round(float(np.mean(pts[:, 0])))),
                        int(round(float(np.mean(pts[:, 1])))),
                    )
                    goals.append(
                        {
                            "type": "target_zone",
                            "position": centroid,
                            "feature": feat_int,
                            "size": len(pts),
                            "confidence": 0.8,
                        }
                    )

        # 2. Detect edge and near-edge exit markers
        for feat in np.unique(grid):
            feat_int = int(feat)
            if (
                feat_int == bg
                or feat_int == 0
                or feat_int in av_feats
                or feat_int in threat_feats
                or (barrier_features and feat_int in barrier_features)
            ):
                continue
            pts = np.argwhere(grid == feat_int)
            if len(pts) < 1 or len(pts) > 4:
                continue

            # Telemetry Guard: Points strictly on bottom HUD row are status counter pixels, not maze exits
            if H >= 16 and all(r >= H - 2 for r, _ in pts):
                continue

            touches_edge = any(r == 0 or r == H - 1 or c == 0 or c == W - 1 for r, c in pts)
            cells = [(int(r), int(c)) for r, c in pts]
            centroid = (
                int(round(float(np.mean(pts[:, 0])))),
                int(round(float(np.mean(pts[:, 1])))),
            )
            if touches_edge:
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
            if H >= 16 and edge_cells[0][0] == H - 1:
                # Bottom status bar row is not a maze exit gateway
                continue
            edge_vals = [int(grid[r, c]) for r, c in edge_cells]
            val_counts = Counter(edge_vals)
            if not val_counts:
                continue
            dom_val, count = val_counts.most_common(1)[0]
            if count >= len(edge_cells) * 0.6 and (
                (barrier_features and dom_val in barrier_features) or dom_val != bg
            ):
                gap_cells = [edge_cells[i] for i, v in enumerate(edge_vals) if v != dom_val]
                if 1 <= len(gap_cells) <= 4:
                    goals.append(
                        {
                            "type": "gateway_notch",
                            "position": gap_cells[len(gap_cells) // 2],
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

        # 4. Gestalt Relational Affordances: Physical Axis Symmetry & Alignment
        # Core Knowledge: Prominent linear boundaries partition space into reflective hemifields.
        divider_axes: list[tuple[str, int, int]] = []  # (orientation, axis_coord, feat)
        struct_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)

        for feat in np.unique(grid):
            feat_int = int(feat)
            if feat_int == bg or feat_int == 0 or feat_int in av_feats:
                continue
            mask = grid == feat_int
            labeled_mask, num_clusters = label(mask, structure=struct_4)
            for c_idx in range(1, num_clusters + 1):
                pts = np.argwhere(labeled_mask == c_idx)
                min_r, min_c = pts.min(axis=0)
                max_r, max_c = pts.max(axis=0)
                span_r = max_r - min_r + 1
                span_c = max_c - min_c + 1
                # Vertical divider axis (interior: not touching outer border)
                if span_r >= max(12, int(H * 0.35)) and span_c <= 4:
                    ax_c = int(round(float(np.mean(pts[:, 1]))))
                    if 2 < ax_c < W - 3:
                        divider_axes.append(("vertical", ax_c, feat_int))
                # Horizontal divider axis
                elif span_c >= max(12, int(W * 0.35)) and span_r <= 4:
                    ax_r = int(round(float(np.mean(pts[:, 0]))))
                    if 2 < ax_r < H - 3:
                        divider_axes.append(("horizontal", ax_r, feat_int))

        # If an explicit divider axis exists or global symmetry is present
        if divider_axes or 0.50 <= sym_score:
            salient_clusters: list[tuple[int, tuple[int, int], np.ndarray]] = []
            axis_feats = {feat for _, _, feat in divider_axes}
            for feat in np.unique(grid):
                feat_int = int(feat)
                if (
                    feat_int == bg
                    or feat_int == 0
                    or feat_int in av_feats
                    or feat_int in axis_feats
                    or (barrier_features and feat_int in barrier_features)
                    or (known_lethal_features and feat_int in known_lethal_features)
                ):
                    continue
                mask = grid == feat_int
                labeled_mask, num_clusters = label(mask, structure=struct_4)
                for c_idx in range(1, num_clusters + 1):
                    c_pts = np.argwhere(labeled_mask == c_idx)
                    if 1 <= len(c_pts) <= int(H * W * 0.25):
                        min_cr, min_cc = c_pts.min(axis=0)
                        max_cr, max_cc = c_pts.max(axis=0)
                        if (max_cr - min_cr >= H - 4) or (max_cc - min_cc >= W - 4):
                            continue
                        c_centroid = (
                            int(round(float(np.mean(c_pts[:, 0])))),
                            int(round(float(np.mean(c_pts[:, 1])))),
                        )
                        salient_clusters.append((feat_int, c_centroid, c_pts))

            # 4a. Projections across explicit divider axes
            for ax_type, ax_coord, _ in divider_axes:
                for feat_int, (cr, cc), c_pts in salient_clusters:
                    if ax_type == "vertical":
                        ref_r = cr
                        ref_c = 2 * ax_coord - cc
                    else:
                        ref_r = 2 * ax_coord - cr
                        ref_c = cc

                    if 0 <= ref_r < H and 0 <= ref_c < W and (ref_r, ref_c) != (cr, cc):
                        target_val = int(grid[ref_r, ref_c])
                        is_void = target_val == bg
                        goals.append(
                            {
                                "type": "reflection_target",
                                "position": (ref_r, ref_c),
                                "feature": feat_int,
                                "confidence": 0.88 if not is_void else 0.70,
                                "is_void": is_void,
                            }
                        )

            # 4b. Projections across global dominant symmetry (if no physical axis)
            if 0.50 <= sym_score and not divider_axes:
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
                    if f1 == f2:
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
                                    "feature": f1,
                                    "confidence": 0.85,
                                    "is_midpoint": True,
                                }
                            )

        # 5. Gestalt Receptacle Enclaves & Docking Constellations (Closure & Containment)
        # Core Knowledge: Bounded perimeter components enclosing cavities or interior markers
        # serve as docking receptacles or target templates for movable pieces.
        struct_conn = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
        enclaves_by_target: dict[int, list[tuple[int, int]]] = defaultdict(list)
        enclave_border_feats: set[int] = set()

        for feat in np.unique(grid):
            feat_int = int(feat)
            if (
                feat_int == bg
                or feat_int == 0
                or (known_lethal_features and feat_int in known_lethal_features)
                or (barrier_features and feat_int in barrier_features)
            ):
                continue
            mask = grid == feat_int
            labeled_mask, num_clusters = label(mask, structure=struct_conn)
            for c_idx in range(1, num_clusters + 1):
                c_pts = np.argwhere(labeled_mask == c_idx)
                min_cr, min_cc = c_pts.min(axis=0)
                max_cr, max_cc = c_pts.max(axis=0)
                h_c = max_cr - min_cr + 1
                w_c = max_cc - min_cc + 1
                if 3 <= h_c <= 24 and 3 <= w_c <= 24:
                    sub = grid[min_cr : max_cr + 1, min_cc : max_cc + 1]
                    interior = sub[1:-1, 1:-1]
                    unique_int = np.unique(interior)
                    for u in unique_int:
                        u_int = int(u)
                        if u_int != feat_int and u_int != bg and u_int != 0:
                            int_pts = np.argwhere(interior == u_int)
                            for ip in int_pts:
                                gr = min_cr + 1 + ip[0]
                                gc = min_cc + 1 + ip[1]
                                enclaves_by_target[u_int].append((int(gr), int(gc)))
                                enclave_border_feats.add(feat_int)

        for target_feat, coords in enclaves_by_target.items():
            if len(coords) >= 2:
                pts_arr = np.array(coords)
                constellation_centroid = (
                    int(round(float(np.median(pts_arr[:, 0])))),
                    int(round(float(np.median(pts_arr[:, 1])))),
                )
                goals.append(
                    {
                        "type": "target_constellation",
                        "position": constellation_centroid,
                        "feature": target_feat,
                        "cells": coords,
                        "border_features": list(enclave_border_feats),
                        "confidence": 0.95,
                    }
                )
            for gr, gc in coords:
                goals.append(
                    {
                        "type": "receptacle_dock",
                        "position": (gr, gc),
                        "feature": target_feat,
                        "border_features": list(enclave_border_feats),
                        "confidence": 0.90,
                    }
                )

        return goals

    @staticmethod
    def detect_affordance_panels(
        entities: Sequence[SpatialEntity],
        grid: np.ndarray,
        bg: int = 0,
    ) -> list[dict[str, Any]]:
        by_dim: dict[tuple[int, int], list[SpatialEntity]] = defaultdict(list)
        for e in entities:
            if getattr(e, "role", None) in (EntityRole.OBSTACLE, EntityRole.AGENT):
                continue
            r0, r1, c0, c1 = e.bounding_box
            h, w = r1 - r0 + 1, c1 - c0 + 1
            by_dim[(h, w)].append(e)

        panels: list[dict[str, Any]] = []
        for (h, w), group in by_dim.items():
            if len(group) < 2 or len(group) > 16:
                continue

            feat_counts: dict[int, int] = defaultdict(int)
            for e in group:
                feat_counts[e.feature_id] += 1
            majority_feat = max(feat_counts.keys(), key=lambda k: feat_counts[k])
            minority_items = [e for e in group if e.feature_id != majority_feat]

            panels.append(
                {
                    "shape": (h, w),
                    "items": group,
                    "majority_feature": majority_feat,
                    "minority_items": minority_items,
                    "item_coords": [e.grid_pos for e in group],
                }
            )
        return panels

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
        barrier_features: set[int] | None = None,
        walkable_features: set[int] | None = None,
    ) -> list[OrientedThreat]:
        av_feats = avatar_features or set()
        goal_set = goals or set()
        cargos = cargo_features or set()
        oriented_threats: list[OrientedThreat] = []

        for e in entities:
            if e.area < 4 or e.area > 20:
                continue
            if getattr(e, "role", None) == EntityRole.MANIPULABLE:
                continue
            r0, r1, c0, c1 = e.bounding_box
            if not (2 <= r1 - r0 <= 4 and 2 <= c1 - c0 <= 4):
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            vals, counts = np.unique(patch, return_counts=True)
            if len(vals) == 2 and np.min(counts) <= 2:
                body_val = int(vals[np.argmax(counts)])
                eye_val = int(vals[np.argmin(counts)])
                if (
                    body_val == bg
                    or body_val == 0
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
    ) -> int:
        return FigureGroundSegregator.estimate_background(
            grid=grid,
            barrier_features=barrier_features,
            avatar_features=avatar_features,
            bg_feature=bg_feature,
        )

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
        H, W = grid.shape
        av_feats = avatar_features or set()
        mutations = state_mutations or []
        goal_pos = learned_goal_positions or set()
        receptacle_pos = learned_receptacle_positions or set()
        entities: list[SpatialEntity] = []

        unique_vals = [int(v) for v in np.unique(grid) if int(v) != bg]
        # 4-connectivity structuring element
        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)

        for val in unique_vals:
            mask = grid == val
            labeled_mask, num_components = label(mask, structure=struct)
            for comp_idx in range(1, num_components + 1):
                coords = np.argwhere(labeled_mask == comp_idx)
                area = int(len(coords))
                if area == 0:
                    continue

                cells = [(int(r), int(c)) for r, c in coords]
                min_r = int(np.min(coords[:, 0]))
                max_r = int(np.max(coords[:, 0]))
                min_c = int(np.min(coords[:, 1]))
                max_c = int(np.max(coords[:, 1]))
                mean_r = float(np.mean(coords[:, 0]))
                mean_c = float(np.mean(coords[:, 1]))
                centroid = (mean_r, mean_c)
                grid_pos = (int(round(mean_r)), int(round(mean_c)))

                role = SpatialRoleClassifier.classify_role(
                    val=val,
                    area=area,
                    bounding_box=(min_r, max_r, min_c, max_c),
                    grid_pos=grid_pos,
                    H=H,
                    W=W,
                    symbolic_theory=symbolic_theory,
                    av_feats=av_feats,
                    mutations=mutations,
                    goal_pos=goal_pos,
                    receptacle_pos=receptacle_pos,
                    avatar_pos=avatar_pos,
                    cells=cells,
                )

                ent = SpatialEntity(
                    id=f"ent_{val}_{len(entities)}_{min_r}_{min_c}",
                    role=role,
                    centroid=centroid,
                    grid_pos=grid_pos,
                    area=area,
                    bounding_box=(min_r, max_r, min_c, max_c),
                    feature_id=val,
                    properties={"cells": cells, "feature": val, "feature_id": val},
                )
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

        prev_features = set(int(v) for v in np.unique(prev_grid))
        curr_features = set(int(v) for v in np.unique(curr_grid))

        bg = int(np.bincount(prev_grid.flatten()).argmax())
        translation_candidates: list[tuple[int, int, int, int]] = []

        for feat in np.unique(prev_grid):
            feat_int = int(feat)
            if feat_int == 0 or feat_int == bg:
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
    def detect_affordance_panels(
        entities: Sequence[SpatialEntity],
        grid: np.ndarray,
        bg: int = 0,
    ) -> list[dict[str, Any]]:
        return GestaltAffordanceDetector.detect_affordance_panels(
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
        barrier_features: set[int] | None = None,
        walkable_features: set[int] | None = None,
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
            barrier_features=barrier_features,
            walkable_features=walkable_features,
        )

    @classmethod
    def to_causal_predicate(cls) -> CausalPredicate:
        """Export perception engine capability as a declarative causal predicate."""
        return CausalPredicate(
            variable="cortex_perception_engine",
            operator="==",
            value="ACTIVE",
            relation="thalamocortical_segmentation",
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
                    "faculty": "cortex_perception",
                    "status": "ACTIVE",
                },
            )
        ]
