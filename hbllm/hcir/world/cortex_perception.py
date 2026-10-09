"""Thalamocortical Visual Perception Faculty for HCIR World Kernel.

Biological Modeling:
1. Sensory Thalamus: Universal sensory normalization across camera, depth, lidar, and 2D arrays.
2. Ventral Visual Stream (V1-V4, IT):
   - Figure-ground segregation and background estimation.
   - Connected-component morphological entity segmentation.
   - Gestalt grouping and relational alignment affordances (symmetry, collinearity).
   - Affordance panel array detection.
3. Dorsal Visual Stream & Theory of Mind:
   - Gaze cone and directional orientation tracking of sentient creatures.
   - Frame difference mutation analysis (translation, index cycle, global transition).
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
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


@dataclass(frozen=True)
class OrientedThreat:
    """An embodied creature or entity with directional facing and visual gaze cone."""

    pos: tuple[int, int]
    facing: tuple[int, int]
    gaze_pos: tuple[int, int]
    feature_id: int


class PerceptionEngine:
    """Domain-agnostic visual perception, background estimation, and entity segmentation."""

    @staticmethod
    def normalize_sensory_input(raw: Any) -> np.ndarray:
        """Normalize arbitrary sensory observations into a spatial 2D array."""
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

        if isinstance(raw, np.ndarray):
            if raw.ndim == 2:
                if np.issubdtype(raw.dtype, np.floating):
                    u_vals = np.unique(raw)
                    if len(u_vals) <= 32:
                        return np.round(raw).astype(int)
                    bins = np.linspace(float(raw.min()), float(raw.max()), 16)
                    return np.digitize(raw, bins).astype(int)
                return raw.astype(int)
            elif raw.ndim == 3:
                H, W, C = raw.shape
                if C == 3:
                    # Grayscale luminance quantization (Rec. 601)
                    gray = 0.299 * raw[:, :, 0] + 0.587 * raw[:, :, 1] + 0.114 * raw[:, :, 2]
                    return (gray / 25.5).astype(int)
                return raw[:, :, 0].astype(int)
            elif raw.ndim == 1:
                side = int(math.isqrt(raw.size))
                if side * side == raw.size:
                    return raw.reshape((side, side)).astype(int)
                return raw.reshape((1, -1)).astype(int)

        if isinstance(raw, (list, tuple)):
            try:
                arr = np.array(raw)
                return PerceptionEngine.normalize_sensory_input(arr)
            except Exception:
                pass

        return np.zeros((1, 1), dtype=int)

    @staticmethod
    def estimate_background(
        grid: np.ndarray,
        barrier_features: set[int] | None = None,
        avatar_features: set[int] | None = None,
        bg_feature: int = 0,
    ) -> int:
        """Domain-agnostic background color estimation based on border dominance and frequency."""
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
    ) -> list[SpatialEntity]:
        """Domain-agnostic connected-component entity segmentation."""
        H, W = grid.shape
        visited = np.zeros((H, W), dtype=bool)
        entities: list[SpatialEntity] = []

        av_feats = avatar_features or set()
        mutations = state_mutations or []
        goal_pos = learned_goal_positions or set()
        receptacle_pos = learned_receptacle_positions or set()

        for r in range(H):
            for c in range(W):
                if visited[r, c]:
                    continue
                val = int(grid[r, c])
                if val == bg:
                    visited[r, c] = True
                    continue

                cells: list[tuple[int, int]] = []
                queue = [(r, c)]
                visited[r, c] = True
                while queue:
                    cr, cc = queue.pop()
                    cells.append((cr, cc))
                    for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc]:
                            if int(grid[nr, nc]) == val:
                                visited[nr, nc] = True
                                queue.append((nr, nc))

                min_r = min(p[0] for p in cells)
                max_r = max(p[0] for p in cells)
                min_c = min(p[1] for p in cells)
                max_c = max(p[1] for p in cells)
                area = len(cells)
                centroid = (sum(p[0] for p in cells) / area, sum(p[1] for p in cells) / area)
                grid_pos = (int(round(centroid[0])), int(round(centroid[1])))

                is_border = (
                    (min_r <= 1 and max_r >= H - 2 and min_c <= 1 and max_c >= W - 2)
                    or (max_r - min_r >= H - 2 and max_c - min_c >= W - 2)
                    or (area > H * W * 0.35)
                )

                is_theory_barrier = (
                    symbolic_theory.is_barrier(val)
                    if (symbolic_theory and hasattr(symbolic_theory, "is_barrier"))
                    else False
                )
                is_theory_cargo = (
                    symbolic_theory.is_cargo(
                        SpatialEntity(
                            id="",
                            role=EntityRole.MANIPULABLE,
                            centroid=centroid,
                            grid_pos=grid_pos,
                            area=area,
                            bounding_box=(min_r, max_r, min_c, max_c),
                            feature_id=val,
                        )
                    )
                    if (symbolic_theory and hasattr(symbolic_theory, "is_cargo"))
                    else False
                )
                is_theory_walkable = (
                    val in symbolic_theory.walkable_features
                    if (symbolic_theory and hasattr(symbolic_theory, "walkable_features"))
                    else False
                )
                is_theory_goal = (
                    (
                        val in symbolic_theory.goal_features
                        or val in symbolic_theory.receptacle_features
                    )
                    if (symbolic_theory and hasattr(symbolic_theory, "goal_features"))
                    else False
                )

                if is_border or is_theory_barrier:
                    role = EntityRole.OBSTACLE
                elif val in av_feats:
                    role = EntityRole.AGENT
                elif any(
                    getattr(m, "trigger_feature", None) == val
                    or getattr(m, "trigger_pos", None) == grid_pos
                    for m in mutations
                ):
                    role = EntityRole.ACTUATOR
                elif is_theory_cargo and not is_theory_walkable:
                    role = EntityRole.MANIPULABLE
                elif is_theory_goal or grid_pos in goal_pos or grid_pos in receptacle_pos:
                    role = EntityRole.GOAL
                elif area <= 16:
                    role = EntityRole.UNKNOWN
                else:
                    role = EntityRole.UNKNOWN

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
        """Infer goal zones from initial visual structure using Gestalt principles."""
        H, W = grid.shape
        goals: list[dict[str, Any]] = []
        av_feats = set(avatar_features) if avatar_features else set()
        if avatar_feature is not None:
            av_feats.add(avatar_feature)

        # 1. Detect target zones (medium-sized non-border rectangular entities)
        for feat in np.unique(grid):
            feat_int = int(feat)
            if (
                feat_int == bg
                or feat_int == 0
                or feat_int in av_feats
                or (barrier_features and feat_int in barrier_features)
                or (known_lethal_features and feat_int in known_lethal_features)
            ):
                continue
            pts = np.argwhere(grid == feat_int)
            if len(pts) < 4 or len(pts) > H * W * 0.3:
                continue

            min_r, min_c = pts.min(axis=0)
            max_r, max_c = pts.max(axis=0)

            is_interior = min_r > 1 and max_r < H - 2 and min_c > 1 and max_c < W - 2
            rect_area = (max_r - min_r + 1) * (max_c - min_c + 1)
            fill_ratio = len(pts) / max(1, rect_area)

            if is_interior and fill_ratio > 0.7 and 4 <= len(pts) <= H * W * 0.15:
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

        # 2. Detect edge exit markers
        for feat in np.unique(grid):
            feat_int = int(feat)
            if (
                feat_int == bg
                or feat_int == 0
                or feat_int in av_feats
                or (barrier_features and feat_int in barrier_features)
                or (known_lethal_features and feat_int in known_lethal_features)
            ):
                continue
            pts = np.argwhere(grid == feat_int)
            if len(pts) < 1 or len(pts) > 4:
                continue

            touches_edge = any(r == 0 or r == H - 1 or c == 0 or c == W - 1 for r, c in pts)
            if touches_edge:
                centroid = (
                    int(round(float(np.mean(pts[:, 0])))),
                    int(round(float(np.mean(pts[:, 1])))),
                )
                goals.append(
                    {
                        "type": "exit_marker",
                        "position": centroid,
                        "feature": feat_int,
                        "size": len(pts),
                        "confidence": 0.6,
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
            from scipy.ndimage import label

            salient_clusters: list[tuple[int, tuple[int, int], np.ndarray]] = []
            for feat in np.unique(grid):
                feat_int = int(feat)
                if (
                    feat_int == bg
                    or feat_int == 0
                    or feat_int in av_feats
                    or (barrier_features and feat_int in barrier_features)
                    or (known_lethal_features and feat_int in known_lethal_features)
                ):
                    continue
                mask = grid == feat_int
                labeled_mask, num_clusters = label(mask)
                for c_idx in range(1, num_clusters + 1):
                    c_pts = np.argwhere(labeled_mask == c_idx)
                    # Only consider discrete salient objects (never sprawling walls or borders)
                    if 1 <= len(c_pts) <= 36:
                        min_cr, min_cc = c_pts.min(axis=0)
                        max_cr, max_cc = c_pts.max(axis=0)
                        if (max_cr - min_cr >= H - 4) or (max_cc - min_cc >= W - 4):
                            continue
                        c_centroid = (
                            int(round(float(np.mean(c_pts[:, 0])))),
                            int(round(float(np.mean(c_pts[:, 1])))),
                        )
                        salient_clusters.append((feat_int, c_centroid, c_pts))

            # 3a. Dominant Axis Symmetry Relational Projections
            for feat_int, (cr, cc), c_pts in salient_clusters:
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
                    if int(grid[ref_r, ref_c]) != feat_int:
                        goals.append(
                            {
                                "type": "relational_alignment",
                                "position": (ref_r, ref_c),
                                "feature": feat_int,
                                "confidence": float(sym_score * 0.90),
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

        return goals

    @staticmethod
    def detect_affordance_panels(
        entities: Sequence[SpatialEntity],
        grid: np.ndarray,
        bg: int = 0,
    ) -> list[dict[str, Any]]:
        """Detect regular rectangular affordance button/panel arrays."""
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
    ) -> list[OrientedThreat]:
        """Detect embodied creatures possessing visual gaze cones / directional facing."""
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
                    gaze_cell = (center[0] + hdr * step_size, center[1] + hdc * step_size)
                    oriented_threats.append(
                        OrientedThreat(
                            pos=center,
                            facing=(hdr, hdc),
                            gaze_pos=gaze_cell,
                            feature_id=body_val,
                        )
                    )
        return oriented_threats
