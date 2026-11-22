"""PerceptionEngine: Composite Thalamocortical Perceptual Processing."""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.ndimage import label

from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.perception.diff import (
    EpistemicObservationDiff,
    FrameDiffType,
    OrientedThreat,
)
from hbllm.hcir.world.perception.gestalt import (
    GestaltAffordanceDetector,
    SpatialRoleClassifier,
)
from hbllm.hcir.world.perception.thalamus import (
    FigureGroundSegregator,
    SensoryThalamus,
    _fast_unique_counts,
)

logger = logging.getLogger(__name__)

_STRUCT_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


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
        obstacle_vals: set[int] = set()
        non_obstacle_vals: set[int] = set()
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
            if role == EntityRole.OBSTACLE:
                obstacle_vals.add(val)
            else:
                non_obstacle_vals.add(val)
            entities.append(ent)

        if symbolic_theory is not None:
            for border_val in obstacle_vals - non_obstacle_vals:
                if border_val in symbolic_theory.candidate_goal_features:
                    symbolic_theory.retract_candidate_goal(
                        border_val, reason="perimeter_border_obstacle"
                    )
                if not symbolic_theory.is_barrier(border_val):
                    symbolic_theory.induce_barrier(border_val, evidence="topology:border")

        # Disambiguate when multiple distinct entities share avatar features (e.g. templates, twins, sockets)
        agent_entities = [
            e for e in entities if e.feature_id in av_feats or e.role == EntityRole.AGENT
        ]
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

            def _boxes_touch(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> bool:
                r_gap = max(0, b1[0] - b2[1] - 1, b2[0] - b1[1] - 1)
                c_gap = max(0, b1[2] - b2[3] - 1, b2[2] - b1[3] - 1)
                return (b1[0] <= b2[1] and b2[0] <= b1[1] and c_gap == 0) or (
                    b1[2] <= b2[3] and b2[2] <= b1[3] and r_gap == 0
                )

            avatar_body: list[SpatialEntity] = [best_agent]
            body_ids = {id(best_agent)}
            grew = True
            while grew:
                grew = False
                for e in agent_entities:
                    if id(e) in body_ids:
                        continue
                    if any(_boxes_touch(e.bounding_box, b.bounding_box) for b in avatar_body):
                        avatar_body.append(e)
                        body_ids.add(id(e))
                        grew = True
            for e in agent_entities:
                if id(e) in body_ids:
                    e.role = EntityRole.AGENT
                elif (
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
