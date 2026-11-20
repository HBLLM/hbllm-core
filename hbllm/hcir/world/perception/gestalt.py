"""Gestalt Affordances, Relational Alignment & Spatial Role Classification."""

from __future__ import annotations

import logging
from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.ndimage import label

from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
from hbllm.hcir.world.perception.diff import OrientedThreat
from hbllm.hcir.world.perception.thalamus import SensoryThalamus, _fast_unique_counts
from hbllm.hcir.world.visual_symmetry import VisualSymmetryAnalyzer

logger = logging.getLogger(__name__)

_STRUCT_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


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
