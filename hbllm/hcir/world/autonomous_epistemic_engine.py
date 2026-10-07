"""Autonomous Epistemic World Engine — Domain-Agnostic Exploration, Causal Induction, and Mental Simulation.

Implements the fundamental cognitive loop:
1. Observe raw pixel frames I_t and available actions A.
2. Analyze pixel difference layouts Δ(I_t, I_{t-1}) to ground motor dynamics (avatar self-identification)
   and environmental state mutations (switches, gates, consumables).
3. If current knowledge is insufficient to reach the target (WIN), generate causal hypotheses
   and explore unknown entities via active curiosity-driven trial and error.
4. When sufficient rules are confirmed, simulate candidate action plans purely in mental imagination
   (forward search over learned world transition models).
5. Switch to exploitation and execute the validated winning plan directly.
"""

from __future__ import annotations

import heapq
import logging
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from hbllm.hcir.graph import ActionNode
from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
from hbllm.hcir.world.active_inference import ActiveInferenceEngine
from hbllm.hcir.world.causal_discovery import (
    BeliefTransitionEvent,
    CausalHypothesis,
    CausalPredicate,
)
from hbllm.hcir.world.intuitive_physics import IntuitivePhysicsEngine
from hbllm.hcir.world.motor_calibration import (
    ActionDynamicsModel,
    StateMutationModel,
)
from hbllm.hcir.world.object_state_graph import ObjectStateGraphPlanner
from hbllm.hcir.world.prefrontal_working_memory import PrefrontalWorkingMemory
from hbllm.hcir.world.spatial_containment import RoomDoor, RoomTopologyExtractor
from hbllm.hcir.world.spatiotemporal_tracker import SpatiotemporalHazardTracker
from hbllm.hcir.world.surprise_engine import SurpriseEngine, SurpriseEvaluation
from hbllm.perception.saccadic_attention import SaccadicAttentionSystem

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Enums and Core Data Transfer Objects
# ─────────────────────────────────────────────────────────────────────────────


class EpistemicPhase(StrEnum):
    """Cognitive lifecycle phases for autonomous exploration and problem solving."""

    MOTOR_GROUNDING = "motor_grounding"  # Self-identification & basic action calibration
    EPISTEMIC_EXPLORATION = (
        "epistemic_exploration"  # Trial-and-error hypothesis testing on unknown objects
    )
    MENTAL_SIMULATION = "mental_simulation"  # Forward planning in mind
    EXPLOITATION = "exploitation"  # Direct execution of verified solution
    REPLANNING = "replanning"  # Recovering from unexpected contradiction


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


@dataclass
class ActionAffordance:
    """Declared or empirically discovered motor affordance for an embodied action.

    Decouples cognitive reasoning from specific game environments, hardware drivers,
    cameras, depth sensors, or LiDAR peripherals.
    """

    action_id: Any
    name: str = ""
    requires_spatial_target: bool = False
    target_param_keys: tuple[str, ...] = ("x", "y")
    is_displacement: bool = False
    delta: tuple[int, ...] | None = None
    is_state_transform: bool = False
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MentalSimulationStep:
    """A step planned purely within internal mental simulation."""

    action: Any
    action_data: dict[str, Any] | None = None
    predicted_avatar_pos: tuple[int, int] = (0, 0)
    expected_mutation: str | None = None


@dataclass(frozen=True)
class OrientedThreat:
    """An embodied creature or entity with directional facing and visual gaze cone."""

    pos: tuple[int, int]
    facing: tuple[int, int]
    gaze_pos: tuple[int, int]
    feature_id: int


# ─────────────────────────────────────────────────────────────────────────────
# 2. HCIR Neuro-Symbolic World Theory
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class HCIRSymbolicWorldTheory:
    """Neuro-symbolic domain-agnostic world theory grounded in HCIR CausalPredicates.

    Maintains intensional causal rules and first-order predicates across level transitions,
    eliminating fragile procedural heuristics.
    """

    goal_predicates: list[CausalPredicate] = field(default_factory=list)
    barrier_predicates: list[CausalPredicate] = field(default_factory=list)
    walkable_predicates: list[CausalPredicate] = field(default_factory=list)
    cargo_predicates: list[CausalPredicate] = field(default_factory=list)
    receptacle_predicates: list[CausalPredicate] = field(default_factory=list)

    # Fast-lookup cached sets
    goal_features: set[int] = field(default_factory=set)
    candidate_goal_features: set[int] = field(default_factory=set)
    barrier_features: set[int] = field(default_factory=set)
    walkable_features: set[int] = field(default_factory=set)
    cargo_features: set[int] = field(default_factory=set)
    receptacle_features: set[int] = field(default_factory=set)

    def is_goal(self, entity: SpatialEntity) -> bool:
        """True if entity satisfies any induced goal predicate or confirmed goal feature."""
        if entity.role == EntityRole.GOAL:
            return True
        if entity.feature_id in self.goal_features:
            return True
        props = getattr(entity, "properties", {})
        return any(pred.evaluate(props) for pred in self.goal_predicates)

    def is_barrier(self, feature_id: int) -> bool:
        """True if feature satisfies any induced barrier predicate (enforcing HCIR Attractor Invariant)."""
        if feature_id in self.goal_features or feature_id in self.candidate_goal_features:
            return False
        if feature_id in self.barrier_features:
            return True
        props = {"feature_id": feature_id, "visual_id": feature_id}
        return any(pred.evaluate(props) for pred in self.barrier_predicates)

    def is_walkable(self, feature_id: int) -> bool:
        """True if feature satisfies any induced walkable space predicate."""
        if feature_id in self.walkable_features:
            return True
        props = {"feature_id": feature_id, "visual_id": feature_id}
        return any(pred.evaluate(props) for pred in self.walkable_predicates)

    def is_cargo(self, entity: SpatialEntity) -> bool:
        """True if entity satisfies any pushable cargo predicate."""
        if entity.feature_id in self.cargo_features:
            return True
        props = getattr(entity, "properties", {})
        return any(pred.evaluate(props) for pred in self.cargo_predicates)

    def induce_goal(self, feature_id: int) -> None:
        """Induce universal goal predicate with minimum description length."""
        self.barrier_features.discard(feature_id)
        if feature_id not in self.goal_features:
            self.goal_features.add(feature_id)
            pred = CausalPredicate(variable="feature_id", operator="==", value=feature_id)
            self.goal_predicates.append(pred)
            logger.info("HCIRSymbolicWorldTheory: Induced Goal Rule: %s", pred.describe())
        self.walkable_features.add(feature_id)

    def induce_barrier(self, feature_id: int) -> None:
        """Induce universal barrier predicate (enforcing HCIR Attractor Invariant)."""
        if feature_id in self.goal_features or feature_id in self.candidate_goal_features:
            return
        if feature_id not in self.barrier_features:
            self.barrier_features.add(feature_id)
            pred = CausalPredicate(variable="feature_id", operator="==", value=feature_id)
            self.barrier_predicates.append(pred)
            self.walkable_features.discard(feature_id)
            logger.info("HCIRSymbolicWorldTheory: Induced Barrier Rule: %s", pred.describe())

    def induce_walkable(self, feature_id: int) -> None:
        """Induce universal walkable space predicate."""
        if feature_id in self.barrier_features:
            return
        if feature_id not in self.walkable_features:
            self.walkable_features.add(feature_id)
            pred = CausalPredicate(variable="feature_id", operator="==", value=feature_id)
            self.walkable_predicates.append(pred)

    def induce_cargo(self, feature_id: int) -> None:
        """Induce pushable cargo predicate."""
        if feature_id not in self.cargo_features:
            self.cargo_features.add(feature_id)
            pred = CausalPredicate(variable="feature_id", operator="==", value=feature_id)
            self.cargo_predicates.append(pred)
            logger.info("HCIRSymbolicWorldTheory: Induced Cargo Rule: %s", pred.describe())

    def clear(self) -> None:
        """Clear all intensional predicates when resetting across distinct games."""
        self.goal_predicates.clear()
        self.barrier_predicates.clear()
        self.walkable_predicates.clear()
        self.cargo_predicates.clear()
        self.receptacle_predicates.clear()
        self.goal_features.clear()
        self.candidate_goal_features.clear()
        self.barrier_features.clear()
        self.walkable_features.clear()
        self.cargo_features.clear()
        self.receptacle_features.clear()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Perception Engine
# ─────────────────────────────────────────────────────────────────────────────


class PerceptionEngine:
    """Domain-agnostic visual perception, background estimation, and entity segmentation."""

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
            non_barriers = [
                i for i, v in enumerate(vals) if int(v) not in barriers and int(v) not in av_feats
            ]
            if non_barriers:
                return int(vals[non_barriers[np.argmax(counts[non_barriers])]])
            g_vals, g_counts = np.unique(grid, return_counts=True)
            g_valid = [
                i for i, v in enumerate(g_vals) if int(v) not in barriers and int(v) not in av_feats
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
        symbolic_theory: HCIRSymbolicWorldTheory,
        avatar_features: set[int] | None = None,
        state_mutations: list[StateMutationModel] | None = None,
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

                if is_border or symbolic_theory.is_barrier(val):
                    role = EntityRole.OBSTACLE
                elif val in av_feats:
                    role = EntityRole.AGENT
                elif any(m.trigger_feature == val or m.trigger_pos == grid_pos for m in mutations):
                    role = EntityRole.ACTUATOR
                elif symbolic_theory.is_cargo(
                    SpatialEntity(
                        id="",
                        role=EntityRole.MANIPULABLE,
                        centroid=centroid,
                        grid_pos=grid_pos,
                        area=area,
                        bounding_box=(min_r, max_r, min_c, max_c),
                        feature_id=val,
                    )
                ):
                    role = EntityRole.MANIPULABLE
                elif (
                    val in symbolic_theory.goal_features
                    or val in symbolic_theory.receptacle_features
                    or grid_pos in goal_pos
                    or grid_pos in receptacle_pos
                ):
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
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Infer goal zones from initial grid structure before solving.

        Identifies likely goal locations from structural cues:
        1. Target zones: medium-sized non-border rectangular entities
        2. Exit markers: small edge-touching entities
        3. Symmetry-based goals: high symmetry score suggests completion goals
        """
        from hbllm.hcir.world.visual_symmetry import VisualSymmetryAnalyzer

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
                centroid = (int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1])))
                goals.append(
                    {
                        "type": "target_zone",
                        "position": centroid,
                        "feature": feat_int,
                        "size": len(pts),
                        "bounds": (int(min_r), int(max_r), int(min_c), int(max_c)),
                        "cells": [(int(p[0]), int(p[1])) for p in pts],
                        "confidence": min(1.0, fill_ratio),
                    }
                )
            # Gestalt Receptacle Outline / Hollow Container Frame:
            # Interior entity forming a hollow boundary or container surrounding an interior cavity
            elif is_interior and 0.04 <= fill_ratio <= 0.60 and 8 <= rect_area <= H * W * 0.25:
                subgrid = grid[min_r : max_r + 1, min_c : max_c + 1]
                if np.any(subgrid != feat_int):
                    centroid = (int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1])))
                    goals.append(
                        {
                            "type": "receptacle_slot",
                            "position": centroid,
                            "feature": feat_int,
                            "size": len(pts),
                            "bounds": (int(min_r), int(max_r), int(min_c), int(max_c)),
                            "cells": [(int(p[0]), int(p[1])) for p in pts],
                            "confidence": 0.85,
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
                centroid = (int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1])))
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
            # A true completion puzzle involves completing a small number of missing tiles
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

        return goals

    @staticmethod
    def detect_affordance_panels(
        entities: Sequence[SpatialEntity],
        grid: np.ndarray,
        bg: int = 0,
    ) -> list[dict[str, Any]]:
        """Domain-agnostic visual Gestalt grouping of regular affordance arrays (keypads, toggles, slots).

        Detects groups of >= 3 small entities sharing similar dimensions (area in [2, 64])
        and regular spatial arrangement.
        Identifies pop-out / minority state features representing toggled or active targets.
        """
        from collections import defaultdict

        small_entities = [e for e in entities if 2 <= e.area <= 64 and e.feature_id != bg]
        if len(small_entities) < 2:
            return []

        by_dim: dict[tuple[int, int], list[SpatialEntity]] = defaultdict(list)
        for e in small_entities:
            r0, r1, c0, c1 = e.bounding_box
            h, w = r1 - r0 + 1, c1 - c0 + 1
            by_dim[(h, w)].append(e)

        panels: list[dict[str, Any]] = []
        for (h, w), group in by_dim.items():
            if len(group) < 2:
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
    ) -> list[OrientedThreat]:
        """Detect embodied creatures possessing visual gaze cones / directional facing.

        Theory of Mind / Perspective Taking:
        Compact entities with asymmetric minority pixels (e.g. eyes, pupils, pointers)
        possess an intrinsic facing direction. Biological agents avoid crossing their
        line of sight, but can flank or ambush them from behind.
        """
        av_feats = avatar_features or set()
        goal_set = goals or set()
        oriented_threats: list[OrientedThreat] = []

        for e in entities:
            if e.area < 4 or e.area > 20:
                continue
            r0, r1, c0, c1 = e.bounding_box
            if not (2 <= r1 - r0 <= 4 and 2 <= c1 - c0 <= 4):
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            vals, counts = np.unique(patch, return_counts=True)
            if len(vals) == 2 and np.min(counts) <= 2:
                body_val = int(vals[np.argmax(counts)])
                eye_val = int(vals[np.argmin(counts)])
                if body_val == bg or body_val in av_feats:
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


# ─────────────────────────────────────────────────────────────────────────────
# 4. Epistemic Feedback Assimilator
# ─────────────────────────────────────────────────────────────────────────────


class EpistemicFeedbackAssimilator:
    """Assimilates environmental feedback into motor dynamics and neuro-symbolic theory."""

    @staticmethod
    def _record_safe_traversal(
        engine: AutonomousEpistemicEngine,
        landing_cells: list[tuple[int, int]],
        delta: tuple[int, int],
        H: int,
        W: int,
    ) -> None:
        """Mark features on the swept trajectory of a survived displacement as safe.

        For each landing cell, walk back along the displacement vector to the
        origin. Every cell crossed (in the pre-move frame) was traversed without
        dying, which is direct experiential evidence of safety.
        """
        if engine.prev_grid is None:
            return
        dr, dc = delta
        steps = max(abs(dr), abs(dc))
        sr = (dr > 0) - (dr < 0)
        sc = (dc > 0) - (dc < 0)
        av = engine.avatar_features or (
            {engine.avatar_feature} if engine.avatar_feature is not None else set()
        )
        lethal = engine.hazard_tracker.known_lethal_features
        for cr, cc in landing_cells:
            for k in range(0, max(steps, 1)):
                r, c = cr - sr * k, cc - sc * k
                if not (0 <= r < H and 0 <= c < W):
                    continue
                wf = int(engine.prev_grid[r, c])
                if wf in av or wf in lethal:
                    continue
                engine.symbolic_theory.induce_walkable(wf)
                engine.verified_safe_features.add(wf)

    @staticmethod
    def _commit_avatar_identity(
        engine: AutonomousEpistemicEngine,
        pairs: list[tuple[SpatialEntity, SpatialEntity]],
        delta: tuple[int, int],
        action: int | None,
        H: int,
        W: int,
        calibrate_dynamics: bool = True,
    ) -> None:
        """Commit an avatar identity from entity pairs sharing a displacement delta.

        This is the shared commitment step used by both fresh-start (single-frame
        heuristic) and controllability-based (multi-frame correlation) identification.
        """
        dr, dc = delta

        def _adjacent(a: SpatialEntity, b: SpatialEntity) -> bool:
            b1, b2 = a.bounding_box, b.bounding_box
            r_gap = max(0, b1[0] - b2[1] - 1, b2[0] - b1[1] - 1)
            c_gap = max(0, b1[2] - b2[3] - 1, b2[2] - b1[3] - 1)
            return r_gap <= 1 and c_gap <= 1

        comp_pairs: list[tuple[SpatialEntity, SpatialEntity]] = [pairs[0]]
        for pe, ce in pairs[1:]:
            if any(_adjacent(ce, c_ce) for _, c_ce in comp_pairs):
                comp_pairs.append((pe, ce))

        engine.avatar_features = {ce.feature_id for _, ce in comp_pairs}
        engine.avatar_feature = next(iter(engine.avatar_features))
        engine.avatar_size = sum(ce.area for _, ce in comp_pairs)
        all_cells = [
            cell for _, ce in comp_pairs for cell in ce.properties.get("cells", [ce.grid_pos])
        ]
        engine.avatar_pos = (
            int(round(sum(c[0] for c in all_cells) / len(all_cells))),
            int(round(sum(c[1] for c in all_cells) / len(all_cells))),
        )
        # Features the avatar just traversed are verified safe
        EpistemicFeedbackAssimilator._record_safe_traversal(engine, all_cells, delta, H, W)

        if calibrate_dynamics and action is not None:
            engine.action_dynamics[action] = ActionDynamicsModel(
                action_id=action,
                delta_r=dr,
                delta_c=dc,
                confidence=0.7,
                probes_tested=1,
            )
            if action in engine.action_affordances:
                engine.action_affordances[action].is_displacement = dr != 0 or dc != 0
                engine.action_affordances[action].delta = (dr, dc)
            else:
                engine.action_affordances[action] = ActionAffordance(
                    action_id=action,
                    is_displacement=(dr != 0 or dc != 0),
                    delta=(dr, dc),
                )

        logger.info(
            "AutonomousEpistemicEngine: Avatar identified (feats=%s, size=%d). "
            "Action %s -> delta=(%d, %d), calibrate=%s",
            engine.avatar_features,
            engine.avatar_size,
            action,
            dr,
            dc,
            calibrate_dynamics,
        )

    @staticmethod
    def assimilate(
        engine: AutonomousEpistemicEngine,
        curr_grid: Any,
        available_actions: Sequence[Any],
        is_win: bool = False,
        is_lost: bool = False,
        action: Any | None = None,
        action_data: dict[str, Any] | None = None,
    ) -> None:
        """Assimilate sensory feedback from previous action into empirical world models."""
        if action is not None:
            engine.last_action = action
        if action_data is not None:
            engine.last_action_data = action_data

        if engine.prev_grid is None or engine.last_action is None:
            return

        engine._feedback_assimilated = True
        prev_avatar_pos = engine.avatar_pos

        curr_grid = engine.normalize_sensory_input(curr_grid)

        if not is_win:
            engine.hazard_tracker.record_frame(
                engine.step_counter,
                curr_grid,
                background_feature=engine.bg_feature,
                avatar_features=engine.avatar_features,
            )

        diff = engine.compute_frame_diff(engine.prev_grid, curr_grid)
        action = engine.last_action
        action_data = engine.last_action_data

        aff = engine.action_affordances.get(action)
        if aff is not None:
            is_effector_action = aff.requires_spatial_target
        else:
            is_effector_action = bool(
                action is not None
                and (
                    engine.is_spatial_effector(action)
                    or (
                        action_data
                        and isinstance(action_data, dict)
                        and any(k in action_data for k in ("x", "y", "col", "row"))
                    )
                )
            )

        if is_effector_action and action not in engine.action_affordances:
            engine.action_affordances[action] = ActionAffordance(
                action_id=action,
                requires_spatial_target=True,
            )

        act_dyn = engine.action_dynamics.get(action) if not is_effector_action else None
        is_known_displacement = bool(
            not is_effector_action and act_dyn is not None and act_dyn.is_displacement_action()
        )

        target_coord: tuple[int, int] | None = None
        if action_data and isinstance(action_data, dict):
            tc = action_data.get("x", action_data.get("col", action_data.get("c")))
            tr = action_data.get("y", action_data.get("row", action_data.get("r")))
            if tc is not None and tr is not None:
                try:
                    target_coord = (int(tr), int(tc))
                except (ValueError, TypeError):
                    pass

        if diff.changed_pixel_count == 0:
            engine.consecutive_quiescent_actions += 1
            engine.consecutive_effective_clicks = 0
            engine.last_effective_click_coord = None
            engine.active_goal_converging_coord = None
            engine.consecutive_goal_converging_clicks = 0
            if target_coord is not None:
                engine.quiescent_click_targets.add(target_coord)
                engine.effective_click_targets.discard(target_coord)
            if action is not None:
                prev_inh = engine.inhibited_actions.get(action, 0)
                engine.inhibited_actions[action] = max(prev_inh + 2, 3)

            if not is_win and not is_lost:
                return
        else:
            engine.consecutive_quiescent_actions = 0
            if target_coord is not None:
                if getattr(engine, "last_effective_click_coord", None) == target_coord:
                    engine.consecutive_effective_clicks = (
                        getattr(engine, "consecutive_effective_clicks", 0) + 1
                    )
                else:
                    engine.last_effective_click_coord = target_coord
                    engine.consecutive_effective_clicks = 1
                engine.effective_click_targets.add(target_coord)
                engine.quiescent_click_targets.discard(target_coord)
                if engine.prev_grid is not None:
                    r, c = target_coord
                    if 0 <= r < engine.prev_grid.shape[0] and 0 <= c < engine.prev_grid.shape[1]:
                        target_feat = int(engine.prev_grid[r, c])
                        engine.effective_features.add(target_feat)
                if diff.changed_mask is not None:
                    mut_cells = [(int(r), int(c)) for r, c in zip(*np.where(diff.changed_mask))]
                    engine.click_affordances[target_coord] = mut_cells

                # Dorsal Visual Stream (V4/MT): Object motion tracking & Teleological Distance Gradient
                if diff.changed_mask is not None and engine.prev_grid is not None:
                    bg = engine.bg_feature if engine.bg_feature is not None else 0
                    disappeared_pts = np.argwhere(diff.changed_mask & (engine.prev_grid != bg))
                    appeared_pts = np.argwhere(diff.changed_mask & (curr_grid != bg))
                    if 1 <= len(disappeared_pts) <= 36 and 1 <= len(appeared_pts) <= 36:
                        p_old = np.mean(disappeared_pts, axis=0)
                        p_new = np.mean(appeared_pts, axis=0)
                        goal_positions: list[tuple[int, int]] = list(engine.learned_goal_positions)
                        for g in PerceptionEngine.detect_structural_goals(curr_grid, bg=bg):
                            if "position" in g:
                                goal_positions.append(g["position"])
                        for gf in engine.learned_goal_features:
                            g_pts = np.argwhere(curr_grid == gf)
                            if len(g_pts) > 0:
                                goal_positions.append(
                                    (int(np.mean(g_pts[:, 0])), int(np.mean(g_pts[:, 1])))
                                )
                        if goal_positions:
                            min_d_old = min(
                                abs(p_old[0] - gp[0]) + abs(p_old[1] - gp[1])
                                for gp in goal_positions
                            )
                            min_d_new = min(
                                abs(p_new[0] - gp[0]) + abs(p_new[1] - gp[1])
                                for gp in goal_positions
                            )
                            if min_d_new < min_d_old:
                                engine.active_goal_converging_coord = target_coord
                                engine.consecutive_goal_converging_clicks = (
                                    getattr(engine, "consecutive_goal_converging_clicks", 0) + 1
                                )
                            elif min_d_new > min_d_old:
                                if (
                                    getattr(engine, "active_goal_converging_coord", None)
                                    == target_coord
                                ):
                                    engine.active_goal_converging_coord = None
                                    engine.consecutive_goal_converging_clicks = 0

                # Numerical / cardinality constraint discovery (e.g. Minesweeper / local count grids):
                tr, tc = target_coord
                if 0 <= tr < curr_grid.shape[0] and 0 <= tc < curr_grid.shape[1]:
                    new_feat = int(curr_grid[tr, tc])
                    if 0 <= new_feat <= 8 and new_feat != engine.bg_feature:
                        new_safe, new_hazards = (
                            engine.working_memory.register_cardinality_constraint(
                                center=target_coord,
                                count=new_feat,
                                grid_shape=curr_grid.shape,
                                radius=1,
                            )
                        )
                        for hz in new_hazards:
                            engine.hazard_tracker.register_lethal_feature(
                                int(curr_grid[hz[0], hz[1]])
                            )
                            engine.learned_barriers.add(hz)

        # ── Intuitive Physics Transition Tracking ────────────────────────────
        engine.physics_engine.record_transition(
            prev_pos=prev_avatar_pos,
            curr_pos=engine.avatar_pos,
            is_displacement_action=is_known_displacement,
            commanded_delta=act_dyn.get_displacement() if act_dyn else (0, 0),
        )

        # ── A. Motor Calibration & Proprioception ────────────────────────────
        H, W = curr_grid.shape
        bg = engine.estimate_background(curr_grid)
        prev_entities = engine.extract_entities(engine.prev_grid, bg)
        curr_entities = engine.extract_entities(curr_grid, bg)

        unmatched_prev: list[SpatialEntity] = []
        unmatched_curr: dict[int, SpatialEntity] = {id(ce): ce for ce in curr_entities}

        for pe in prev_entities:
            stat_match = None
            for ce_id, ce in unmatched_curr.items():
                if (
                    pe.feature_id == ce.feature_id
                    and pe.area == ce.area
                    and pe.grid_pos == ce.grid_pos
                ):
                    stat_match = ce_id
                    break
            if stat_match is not None:
                del unmatched_curr[stat_match]
            else:
                unmatched_prev.append(pe)

        moved_entities: list[tuple[SpatialEntity, SpatialEntity, tuple[int, int]]] = []
        for pe in unmatched_prev:
            best_ce_id = None
            min_dist = float("inf")
            best_delta = (0, 0)
            for ce_id, ce in unmatched_curr.items():
                if pe.feature_id == ce.feature_id and pe.area == ce.area and pe.area <= 144:
                    dr = int(round(ce.centroid[0] - pe.centroid[0]))
                    dc = int(round(ce.centroid[1] - pe.centroid[1]))
                    dist = abs(dr) + abs(dc)
                    is_valid_displacement = (0 < dist <= 16 and (dr == 0 or dc == 0)) or (
                        0 < dist <= 8 and abs(dr) <= 4 and abs(dc) <= 4
                    )
                    if is_valid_displacement:
                        if dist < min_dist:
                            min_dist = dist
                            best_ce_id = ce_id
                            best_delta = (dr, dc)
            if best_ce_id is not None:
                matched_ce = unmatched_curr.pop(best_ce_id)
                moved_entities.append((pe, matched_ce, best_delta))

        def _are_adjacent(e1: SpatialEntity, e2: SpatialEntity) -> bool:
            bb1 = e1.bounding_box
            bb2 = e2.bounding_box
            r_gap = max(0, bb1[0] - bb2[1] - 1, bb2[0] - bb1[1] - 1)
            c_gap = max(0, bb1[2] - bb2[3] - 1, bb2[2] - bb1[3] - 1)
            return r_gap <= 1 and c_gap <= 1

        if (
            not is_win
            and not is_effector_action
            and not engine.avatar_features
            and engine.avatar_feature is None
        ):
            if moved_entities:
                delta_groups: dict[tuple[int, int], list[tuple[SpatialEntity, SpatialEntity]]] = {}
                for pe, ce, delta in moved_entities:
                    delta_groups.setdefault(delta, []).append((pe, ce))

                has_retained_dynamics = bool(engine.action_dynamics)

                if not has_retained_dynamics:
                    # ── Fresh start (no prior knowledge) ──────────────────────
                    # No dynamics to cross-reference. Use the original heuristic:
                    # the largest group of entities sharing the same displacement
                    # is most likely the avatar (first level, typically no enemies).
                    best_delta, pairs = max(delta_groups.items(), key=lambda item: len(item[1]))
                    EpistemicFeedbackAssimilator._commit_avatar_identity(
                        engine,
                        pairs,
                        best_delta,
                        action,
                        H,
                        W,
                        calibrate_dynamics=True,
                    )
                else:
                    # ── Controllability Test (retained dynamics) ───────────────
                    # The human brain identifies its avatar not from a single
                    # observation but from AGENCY: "I pressed up and THIS thing
                    # moved up. I pressed right and THIS SAME thing moved right."
                    #
                    # Accumulate (action, observed_delta) evidence per feature.
                    # Commit only when ONE feature consistently matches the
                    # expected displacement across 2+ different actions.
                    # This naturally disambiguates the avatar from enemies that
                    # move independently of player input.
                    for delta_val, group in delta_groups.items():
                        for _, ce in group:
                            feat = ce.feature_id
                            if feat != bg and feat != engine.bg_feature:
                                engine._avatar_controllability_evidence.setdefault(feat, []).append(
                                    (action, delta_val)
                                )

                    # Check if any feature has demonstrated controllability:
                    # its observed delta matches the expected action delta across
                    # at least 2 DIFFERENT displacement actions.
                    best_candidate: int | None = None
                    best_score = 0
                    for feat, observations in engine._avatar_controllability_evidence.items():
                        confirmed_actions: set[int] = set()
                        for obs_action, obs_delta in observations:
                            dyn = engine.action_dynamics.get(obs_action)
                            if dyn is not None and dyn.is_displacement_action():
                                expected = dyn.get_displacement()
                                if obs_delta == expected:
                                    confirmed_actions.add(obs_action)
                        if len(confirmed_actions) > best_score:
                            best_score = len(confirmed_actions)
                            best_candidate = feat

                    if best_score >= 2 and best_candidate is not None:
                        # Found a feature that consistently responds to our
                        # actions — this is the avatar. Retrieve its last
                        # observation to commit identity.
                        last_obs = engine._avatar_controllability_evidence[best_candidate][-1]
                        obs_action, obs_delta = last_obs
                        matching_pairs = delta_groups.get(obs_delta, [])
                        # Filter to only the confirmed feature
                        avatar_pairs = [
                            (pe, ce) for pe, ce in matching_pairs if ce.feature_id == best_candidate
                        ]
                        if not avatar_pairs:
                            # Feature was confirmed by prior frame, find it now
                            for dg_pairs in delta_groups.values():
                                for pe, ce in dg_pairs:
                                    if ce.feature_id == best_candidate:
                                        avatar_pairs.append((pe, ce))
                        if avatar_pairs:
                            commit_delta = obs_delta
                            EpistemicFeedbackAssimilator._commit_avatar_identity(
                                engine,
                                avatar_pairs,
                                commit_delta,
                                action,
                                H,
                                W,
                                calibrate_dynamics=False,  # keep retained dynamics
                            )
                            engine._avatar_controllability_evidence.clear()
                            logger.info(
                                "AutonomousEpistemicEngine: Avatar confirmed via controllability "
                                "test — feature %s responded to %d different actions.",
                                best_candidate,
                                best_score,
                            )
                    else:
                        logger.debug(
                            "AutonomousEpistemicEngine: Avatar identification deferred — "
                            "accumulating controllability evidence (best_score=%d, candidates=%d)",
                            best_score,
                            len(engine._avatar_controllability_evidence),
                        )
        elif not is_effector_action or bool(
            engine.avatar_features or engine.avatar_feature is not None
        ):
            known_av_feats = engine.avatar_features or (
                {engine.avatar_feature} if engine.avatar_feature is not None else set()
            )
            av_moved = [item for item in moved_entities if item[1].feature_id in known_av_feats]
            if (
                not is_win
                and not is_lost
                and not av_moved
                and not is_effector_action
                and moved_entities
                and (action in engine.action_dynamics or not action_data)
            ):
                valid_moved = [
                    (pe, ce, delta)
                    for pe, ce, delta in moved_entities
                    if ce.feature_id != bg
                    and ce.feature_id != engine.bg_feature
                    and ce.feature_id != 0
                    and ce.area <= 49
                ]
                if valid_moved:
                    delta_groups_remap: dict[
                        tuple[int, int], list[tuple[SpatialEntity, SpatialEntity]]
                    ] = {}
                    for pe, ce, delta in valid_moved:
                        delta_groups_remap.setdefault(delta, []).append((pe, ce))
                    best_delta, pairs = max(
                        delta_groups_remap.items(), key=lambda item: len(item[1])
                    )
                    comp_pairs = [pairs[0]]
                    for pe, ce in pairs[1:]:
                        if any(_are_adjacent(ce, c_ce) for _, c_ce in comp_pairs):
                            comp_pairs.append((pe, ce))
                    engine.avatar_features = {ce.feature_id for _, ce in comp_pairs}
                    engine.avatar_feature = next(iter(engine.avatar_features))
                    engine.avatar_size = sum(ce.area for _, ce in comp_pairs)
                    known_av_feats = engine.avatar_features
                    av_moved = [
                        item for item in moved_entities if item[1].feature_id in known_av_feats
                    ]

            if av_moved:
                if len(av_moved) > 1 and prev_avatar_pos is not None:
                    av_moved.sort(
                        key=lambda item: (
                            abs(item[0].grid_pos[0] - prev_avatar_pos[0])
                            + abs(item[0].grid_pos[1] - prev_avatar_pos[1])
                        )
                    )
                    primary_ce = av_moved[0][1]
                    filtered_av_moved = [av_moved[0]]
                    for item in av_moved[1:]:
                        if _are_adjacent(item[1], primary_ce):
                            filtered_av_moved.append(item)
                    av_moved = filtered_av_moved

                av_pe, av_ce, (dr, dc) = av_moved[0]
                all_cells = [
                    cell
                    for _, ce, _ in av_moved
                    for cell in ce.properties.get("cells", [ce.grid_pos])
                ]
                engine.avatar_pos = (
                    int(round(sum(c[0] for c in all_cells) / len(all_cells))),
                    int(round(sum(c[1] for c in all_cells) / len(all_cells))),
                )
                if not is_win:
                    EpistemicFeedbackAssimilator._record_safe_traversal(
                        engine, all_cells, (dr, dc), H, W
                    )

                if not is_effector_action:
                    if action not in engine.action_dynamics:
                        engine.action_dynamics[action] = ActionDynamicsModel(
                            action_id=action,
                            delta_r=dr,
                            delta_c=dc,
                            confidence=0.6,
                            probes_tested=1,
                        )
                    else:
                        engine.action_dynamics[action].update_from_trial(
                            (dr, dc), success=True, learning_rate=0.5
                        )
                    if action in engine.action_affordances:
                        engine.action_affordances[action].is_displacement = dr != 0 or dc != 0
                        engine.action_affordances[action].delta = (dr, dc)
                    else:
                        engine.action_affordances[action] = ActionAffordance(
                            action_id=action,
                            is_displacement=(dr != 0 or dc != 0),
                            delta=(dr, dc),
                        )

                for pe, ce, delta in moved_entities:
                    if ce.feature_id in known_av_feats:
                        continue
                    # Pushing requires physical contact: entity must be adjacent and ahead in motion direction
                    is_contact = _are_adjacent(pe, av_pe)
                    is_ahead = False
                    if dr != 0 and dc == 0:
                        is_ahead = (pe.centroid[0] - av_pe.centroid[0]) * dr > 0 and abs(
                            pe.centroid[1] - av_pe.centroid[1]
                        ) <= 3
                    elif dc != 0 and dr == 0:
                        is_ahead = (pe.centroid[1] - av_pe.centroid[1]) * dc > 0 and abs(
                            pe.centroid[0] - av_pe.centroid[0]
                        ) <= 3
                    else:
                        is_ahead = (pe.centroid[0] - av_pe.centroid[0]) * dr + (
                            pe.centroid[1] - av_pe.centroid[1]
                        ) * dc > 0

                    is_pushed = is_contact and is_ahead and (delta == (dr, dc))
                    if is_pushed:
                        engine.symbolic_theory.induce_cargo(ce.feature_id)
                        logger.info(
                            "AutonomousEpistemicEngine: Discovered PUSHABLE CARGO (feat=%d, size=%d)",
                            ce.feature_id,
                            ce.area,
                        )
                    elif delta != (0, 0):
                        # Autonomous dynamic entity (moved under its own agency) — not passive cargo
                        engine.symbolic_theory.cargo_features.discard(ce.feature_id)
                    elif delta == (dr, dc) and any(
                        _are_adjacent(ce, c_ce) for _, c_ce, _ in av_moved
                    ):
                        engine.avatar_features.add(ce.feature_id)
                        engine.avatar_size += ce.area
                        logger.info(
                            "AutonomousEpistemicEngine: Discovered compound avatar component (feat=%d, size=%d)",
                            ce.feature_id,
                            ce.area,
                        )
        # ── Proprioceptive Efference Copy Verification & Obstacle Grounding ──
        if is_known_displacement and prev_avatar_pos is not None:
            avatar_moved = bool(
                engine.avatar_pos is not None and engine.avatar_pos != prev_avatar_pos
            )

            # 1. Physical resistance detection (Avatar did not move upon directional motor command)
            if not avatar_moved and not is_win:
                if act_dyn is not None and act_dyn.is_displacement_action():
                    # Motor command failed to displace avatar! Ground obstacle cell regardless of other background motion
                    exp_dr, exp_dc = act_dyn.get_displacement()
                    step_r = int(np.sign(exp_dr))
                    step_c = int(np.sign(exp_dc))
                    dist = max(abs(exp_dr), abs(exp_dc))

                    if dist > 0 and prev_avatar_pos is not None:
                        av_feats = engine.avatar_features or (
                            {engine.avatar_feature} if engine.avatar_feature is not None else set()
                        )
                        # HCIR Attractor Invariant: Gather candidate goal features
                        cand_goals = engine.detect_structural_goals(curr_grid)
                        cand_goal_feats = {
                            int(g["feature"])
                            for g in cand_goals
                            if "feature" in g and g["feature"] is not None
                        }
                        cand_goal_positions = {
                            g["position"]
                            for g in cand_goals
                            if "position" in g and g["position"] is not None
                        }
                        engine.symbolic_theory.candidate_goal_features.update(cand_goal_feats)

                        for k in range(1, dist + 1):
                            kr = prev_avatar_pos[0] + step_r * k
                            kc = prev_avatar_pos[1] + step_c * k
                            if 0 <= kr < H and 0 <= kc < W:
                                val = (
                                    int(engine.prev_grid[kr, kc])
                                    if engine.prev_grid is not None
                                    else int(curr_grid[kr, kc])
                                )
                                # HCIR Attractor Invariant: Candidate goals are NEVER barriers
                                # Proprioceptive obstacle grounding: any non-walkable feature blocking movement is an empirical barrier
                                if (
                                    val not in av_feats
                                    and val not in engine.learned_goal_features
                                    and val not in cand_goal_feats
                                    and (kr, kc) not in cand_goal_positions
                                    and not engine.symbolic_theory.is_walkable(val)
                                ):
                                    engine.learned_barriers.add((kr, kc))
                                    if not engine.symbolic_theory.is_barrier(val):
                                        engine.symbolic_theory.induce_barrier(val)
                                        logger.info(
                                            "AutonomousEpistemicEngine: Proprioceptive obstacle grounded along ray at (%d, %d) with feature %d",
                                            kr,
                                            kc,
                                            val,
                                        )
                                        if engine.bg_feature == val:
                                            engine.bg_feature = engine.estimate_background(
                                                curr_grid
                                            )
                                    break
                        else:
                            end_r = prev_avatar_pos[0] + step_r * dist
                            end_c = prev_avatar_pos[1] + step_c * dist
                            if 0 <= end_r < H and 0 <= end_c < W:
                                engine.learned_barriers.add((end_r, end_c))

                    if engine.last_action is not None and prev_avatar_pos is not None:
                        engine.failed_transitions.add((prev_avatar_pos, engine.last_action))
                        prev_inh = engine.inhibited_actions.get(engine.last_action, 0)
                        engine.inhibited_actions[engine.last_action] = max(prev_inh + 2, 3)

                if prev_avatar_pos is not None and engine.avatar_pos is not None:
                    if engine.avatar_pos == prev_avatar_pos and not is_win and not is_lost:
                        engine.consecutive_stuck_steps += 1
                    else:
                        engine.consecutive_stuck_steps = 0
            else:
                engine.consecutive_stuck_steps = 0

            # 2. Predictive coding efference copy divergence check & surprise computation
            expected_dr = (
                engine.last_predicted_pos[0] - prev_avatar_pos[0]
                if engine.last_predicted_pos is not None and prev_avatar_pos is not None
                else (act_dyn.delta_r if act_dyn is not None and is_known_displacement else 0)
            )
            expected_dc = (
                engine.last_predicted_pos[1] - prev_avatar_pos[1]
                if engine.last_predicted_pos is not None and prev_avatar_pos is not None
                else (act_dyn.delta_c if act_dyn is not None and is_known_displacement else 0)
            )
            actual_dr = (
                engine.avatar_pos[0] - prev_avatar_pos[0]
                if engine.avatar_pos is not None and prev_avatar_pos is not None
                else 0
            )
            actual_dc = (
                engine.avatar_pos[1] - prev_avatar_pos[1]
                if engine.avatar_pos is not None and prev_avatar_pos is not None
                else 0
            )

            expected_state = {
                "dr": expected_dr,
                "dc": expected_dc,
                "avatar_moved": 1
                if (is_known_displacement or expected_dr != 0 or expected_dc != 0)
                else 0,
            }
            actual_state = {
                "dr": actual_dr,
                "dc": actual_dc,
                "avatar_moved": 1 if avatar_moved else 0,
            }

            confidence = 0.85 if engine.mental_plan else (0.70 if is_known_displacement else 0.40)
            salience = 1.3 if diff.changed_pixel_count > 0 else 0.8
            surprise_eval = engine.surprise_engine.evaluate_surprise(
                prediction_id=f"step_{engine.step_counter}",
                expected_state=expected_state,
                actual_state=actual_state,
                confidence=confidence,
                attention_salience=salience,
                prediction_source="motor_calibration",
                context_signature=f"act_{action}",
            )
            engine.last_surprise = surprise_eval.surprise_score
            engine.last_surprise_eval = surprise_eval

            discrepancy = False
            if not is_win:
                if not avatar_moved and diff.changed_pixel_count == 0:
                    discrepancy = True
                elif (
                    engine.last_predicted_pos is not None
                    and engine.avatar_pos is not None
                    and max(
                        abs(engine.avatar_pos[0] - engine.last_predicted_pos[0]),
                        abs(engine.avatar_pos[1] - engine.last_predicted_pos[1]),
                    )
                    > 1
                ):
                    discrepancy = True
                elif surprise_eval.is_surprising and not avatar_moved and is_known_displacement:
                    discrepancy = True

            # Involuntary Orienting Reflex: When surprising visual change occurs,
            # shift prefrontal attentional focus to the centroid of the mutation
            if (
                surprise_eval.is_surprising
                and diff.changed_mask is not None
                and np.any(diff.changed_mask)
            ):
                mut_cells = np.argwhere(diff.changed_mask)
                if len(mut_cells) > 0:
                    sal_r = int(round(float(np.mean([pt[0] for pt in mut_cells]))))
                    sal_c = int(round(float(np.mean([pt[1] for pt in mut_cells]))))
                    engine.working_memory.orient_attention((sal_r, sal_c))

            if discrepancy:
                if engine.last_action is not None and prev_avatar_pos is not None:
                    engine.failed_transitions.add((prev_avatar_pos, engine.last_action))
                    prev_inh = engine.inhibited_actions.get(engine.last_action, 0)
                    engine.inhibited_actions[engine.last_action] = max(prev_inh + 2, 3)
                if engine.mental_plan:
                    logger.debug(
                        "AutonomousEpistemicEngine: Sensory discrepancy/surprise detected (pred=%s, actual=%s, surprise=%.3f). Invalidating %d-step mental plan.",
                        engine.last_predicted_pos,
                        engine.avatar_pos,
                        surprise_eval.surprise_score,
                        len(engine.mental_plan),
                    )
                    engine.mental_plan.clear()
                    engine.phase = EpistemicPhase.REPLANNING
                    engine.consecutive_plan_failures += 1
                    if engine.consecutive_plan_failures >= 2:
                        engine.exploration_cooldown = 2
                        engine.consecutive_plan_failures = 0
            else:
                if avatar_moved or diff.changed_pixel_count > 0:
                    engine.consecutive_plan_failures = 0

        # ── B. Environmental Mutation Induction ──────────────────────────────
        movable_features = set(engine.symbolic_theory.cargo_features)
        movable_features.update(engine.avatar_features)
        if engine.avatar_feature is not None:
            movable_features.add(engine.avatar_feature)

        def is_hud_mutation(r: int, c: int) -> bool:
            if H >= 24:
                if engine.avatar_pos is not None:
                    if r >= H - 6 and engine.avatar_pos[0] < H - 8:
                        return True
                    if r < 3 and engine.avatar_pos[0] >= 5:
                        return True
                    min_r, max_r = min(engine.avatar_pos[0], r), max(engine.avatar_pos[0], r)
                    for div_r in range(min_r + 1, max_r):
                        div_val = int(engine.prev_grid[div_r, 0])
                        if np.all(engine.prev_grid[div_r, :] == div_val):
                            if div_r >= H - 16 or div_r <= 16:
                                return True
                    min_c, max_c = min(engine.avatar_pos[1], c), max(engine.avatar_pos[1], c)
                    for div_c in range(min_c + 1, max_c):
                        div_val = int(engine.prev_grid[0, div_c])
                        if np.all(engine.prev_grid[:, div_c] == div_val):
                            if div_c >= W - 16 or div_c <= 16:
                                return True
                else:
                    if r >= H - 6 or r < 3:
                        return True
            return False

        av_radius = max(2, int(round(math.sqrt(max(1, engine.avatar_size)))))

        def is_local_to_avatar(r: int, c: int) -> bool:
            if engine.avatar_pos is not None:
                if max(abs(r - engine.avatar_pos[0]), abs(c - engine.avatar_pos[1])) <= av_radius:
                    return True
            if prev_avatar_pos is not None:
                if max(abs(r - prev_avatar_pos[0]), abs(c - prev_avatar_pos[1])) <= av_radius:
                    return True
            return False

        distant_mutations = [
            (r, c, old_v, new_v)
            for r, c, old_v, new_v in diff.mutated_pixels
            if old_v not in movable_features
            and new_v not in movable_features
            and not is_hud_mutation(r, c)
            and not is_local_to_avatar(r, c)
        ]

        if distant_mutations and len(distant_mutations) <= 64:
            trigger_pos = (
                (int(action_data["y"]), int(action_data["x"]))
                if action_data and "x" in action_data
                else engine.avatar_pos
            )
            trigger_feat = (
                int(engine.prev_grid[trigger_pos[0], trigger_pos[1]])
                if trigger_pos and 0 <= trigger_pos[0] < H and 0 <= trigger_pos[1] < W
                else None
            )

            if (
                trigger_feat is None
                or trigger_feat == bg
                or trigger_feat == engine.bg_feature
                or engine.symbolic_theory.is_walkable(trigger_feat)
            ):
                trigger_feat = None

            existing_mut = next(
                (
                    m
                    for m in engine.state_mutations
                    if m.trigger_pos == trigger_pos
                    and m.trigger_feature == trigger_feat
                    and m.prior_value == distant_mutations[0][2]
                ),
                None,
            )
            if existing_mut is not None:
                existing_mut.record_observation(distant_mutations[0][3])
            else:
                mutation_model = StateMutationModel(
                    trigger_type="CONTACT" if is_known_displacement else "ACTION",
                    trigger_pos=trigger_pos,
                    trigger_feature=trigger_feat,
                    mutation_type="ENVIRONMENTAL_TOGGLE",
                    prior_value=distant_mutations[0][2],
                    posterior_value=distant_mutations[0][3],
                    confidence=0.8,
                    occurrences=1,
                    metadata={"action_id": action},
                )
                engine.state_mutations.append(mutation_model)
                engine.exhausted_candidate_goals.clear()
                logger.info(
                    "AutonomousEpistemicEngine: Induced StateMutationModel! Trigger %s at %s changed %d distant pixels.",
                    mutation_model.trigger_type,
                    trigger_pos,
                    len(distant_mutations),
                )

        # ── C. Win / Loss Feedback Assimilation ──────────────────────────────
        av_feats = engine.avatar_features or (
            {engine.avatar_feature} if engine.avatar_feature is not None else set()
        )

        if is_win:
            win_target = None
            if prev_avatar_pos is not None:
                dr, dc = 0, 0
                if action in engine.action_dynamics:
                    dr, dc = engine.action_dynamics[action].get_displacement()
                elif action_data and "x" in action_data and "y" in action_data:
                    win_target = (int(action_data["y"]), int(action_data["x"]))
                if win_target is None:
                    tr, tc = prev_avatar_pos[0] + dr, prev_avatar_pos[1] + dc
                    if 0 <= tr < H and 0 <= tc < W:
                        win_target = (tr, tc)
            elif engine.current_simulated_goal is not None:
                win_target = engine.current_simulated_goal
            elif engine.avatar_pos is not None:
                win_target = engine.avatar_pos
            elif action_data and "x" in action_data and "y" in action_data:
                win_target = (int(action_data["y"]), int(action_data["x"]))

            if win_target is not None:
                engine.learned_goal_positions.add(win_target)
                if (
                    engine.prev_grid is not None
                    and 0 <= win_target[0] < H
                    and 0 <= win_target[1] < W
                ):
                    goal_feat = int(engine.prev_grid[win_target[0], win_target[1]])
                    is_distant_target = (
                        prev_avatar_pos is None
                        or abs(win_target[0] - prev_avatar_pos[0]) > 0
                        or abs(win_target[1] - prev_avatar_pos[1]) > 0
                    )
                    if (
                        goal_feat != bg
                        and goal_feat != engine.bg_feature
                        and (goal_feat not in av_feats or is_distant_target)
                        and not engine.symbolic_theory.is_barrier(goal_feat)
                    ):
                        engine.symbolic_theory.induce_goal(goal_feat)
                        logger.info(
                            "AutonomousEpistemicEngine: Grounded invariant WIN GOAL FEATURE %d at %s",
                            goal_feat,
                            win_target,
                        )

                for pe in prev_entities:
                    is_distant_pe = (
                        prev_avatar_pos is None
                        or abs(pe.grid_pos[0] - prev_avatar_pos[0]) > 2
                        or abs(pe.grid_pos[1] - prev_avatar_pos[1]) > 2
                    )
                    if (
                        (pe.feature_id not in av_feats or is_distant_pe)
                        and pe.feature_id != bg
                        and pe.feature_id != engine.bg_feature
                        and not engine.symbolic_theory.is_barrier(pe.feature_id)
                        and 1 <= pe.area <= 64
                    ):
                        pe_cells = pe.properties.get("cells", [pe.grid_pos])
                        if win_target in pe_cells or pe.grid_pos == win_target:
                            engine.symbolic_theory.induce_goal(pe.feature_id)
                            logger.info(
                                "AutonomousEpistemicEngine: Grounded invariant WIN GOAL ENTITY feature %d (area=%d)",
                                pe.feature_id,
                                pe.area,
                            )

            if engine.current_simulated_goal is not None and engine.prev_grid is not None:
                sr, sc = engine.current_simulated_goal
                if 0 <= sr < H and 0 <= sc < W:
                    sim_feat = int(engine.prev_grid[sr, sc])
                    if (
                        sim_feat != bg
                        and sim_feat != engine.bg_feature
                        and sim_feat not in av_feats
                        and not engine.symbolic_theory.is_barrier(sim_feat)
                    ):
                        engine.symbolic_theory.induce_goal(sim_feat)

            if engine.active_hypothesis:
                engine.active_hypothesis.confirmed = True
                engine.active_hypothesis.confidence = 1.0

            # Level transition: cleanly reset episodic spatial memory for the next level
            engine.reset_episode(
                retain_dynamics=True, is_new_level=True, level=engine.current_level + 1
            )
            return
        elif engine.avatar_pos is not None and not is_lost:
            engine.exhausted_candidate_goals.add(engine.avatar_pos)

        if is_lost:
            if prev_avatar_pos is not None and action is not None:
                engine.failed_transitions.add((prev_avatar_pos, action))
                logger.info(
                    "AutonomousEpistemicEngine: Grounded lethal failed transition (%s, %s)",
                    prev_avatar_pos,
                    action,
                )

            target_pos = None
            if prev_avatar_pos is not None:
                dr, dc = 0, 0
                if action in engine.action_dynamics:
                    dr, dc = engine.action_dynamics[action].get_displacement()
                elif action_data and "x" in action_data and "y" in action_data:
                    target_pos = (int(action_data["y"]), int(action_data["x"]))
                if target_pos is None:
                    tr, tc = prev_avatar_pos[0] + dr, prev_avatar_pos[1] + dc
                    if 0 <= tr < H and 0 <= tc < W:
                        target_pos = (tr, tc)
            elif engine.avatar_pos is not None:
                target_pos = engine.avatar_pos
            elif action_data and "x" in action_data and "y" in action_data:
                target_pos = (int(action_data["y"]), int(action_data["x"]))

            if target_pos is not None:
                if (
                    engine.prev_grid is not None
                    and 0 <= target_pos[0] < H
                    and 0 <= target_pos[1] < W
                ):
                    # Check for lethal hazard features in a 5x5 window around target_pos
                    r0 = max(0, target_pos[0] - 2)
                    r1 = min(H, target_pos[0] + 3)
                    c0 = max(0, target_pos[1] - 2)
                    c1 = min(W, target_pos[1] + 3)
                    candidate_cells = set(np.unique(curr_grid[r0:r1, c0:c1])).union(
                        set(np.unique(engine.prev_grid[r0:r1, c0:c1]))
                    )
                    # Also inspect unobstructed cardinal line-of-sight rays for remote turrets/projectiles
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        curr_r, curr_c = target_pos[0] + dr, target_pos[1] + dc
                        while 0 <= curr_r < H and 0 <= curr_c < W:
                            val = int(engine.prev_grid[curr_r, curr_c])
                            if engine.symbolic_theory.is_barrier(val):
                                break
                            if (
                                val != engine.bg_feature
                                and val not in av_feats
                                and not engine.symbolic_theory.is_walkable(val)
                            ):
                                candidate_cells.add(val)
                            curr_r += dr
                            curr_c += dc
                    for cand_feat in candidate_cells:
                        cand_feat_int = int(cand_feat)
                        if (
                            cand_feat_int != engine.bg_feature
                            and cand_feat_int not in av_feats
                            and not engine.symbolic_theory.is_walkable(cand_feat_int)
                            and int(np.sum(engine.prev_grid == cand_feat_int)) < 40
                        ):
                            engine.hazard_tracker.register_lethal_feature(cand_feat_int)
                            if int(np.sum(engine.prev_grid == cand_feat_int)) >= 25:
                                engine.symbolic_theory.induce_barrier(cand_feat_int)
                            engine.symbolic_theory.cargo_features.discard(cand_feat_int)
                            engine.symbolic_theory.goal_features.discard(cand_feat_int)
                            engine.symbolic_theory.candidate_goal_features.discard(cand_feat_int)
                            logger.info(
                                "AutonomousEpistemicEngine: Grounded lethal feature %d at barrier %s",
                                cand_feat_int,
                                target_pos,
                            )
                    engine.exhausted_candidate_goals.add(target_pos)

                    # Trajectory-level aversive conditioning:
                    # The destination cell target_pos resulted in catastrophic loss/death.
                    # Ground it as a static lethal position and barrier so neither mental simulation
                    # nor exploratory probes step into this coordinate again!
                    engine.hazard_tracker.register_lethal_position(target_pos)
                    engine.learned_barriers.add(target_pos)
                    engine.level_learned_barriers.setdefault(engine.current_level, set()).add(
                        target_pos
                    )
                    engine.level_lethal_positions.setdefault(engine.current_level, set()).add(
                        target_pos
                    )

                    # Blacklist all cardinal incoming moves into target_pos from immediate neighbors
                    for m_dr, m_dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nbr = (target_pos[0] - m_dr, target_pos[1] - m_dc)
                        if 0 <= nbr[0] < H and 0 <= nbr[1] < W:
                            for cand_a, dyn in engine.action_dynamics.items():
                                if dyn.is_displacement_action() and dyn.get_displacement() == (
                                    m_dr,
                                    m_dc,
                                ):
                                    engine.failed_transitions.add((nbr, cand_a))

            if action is not None:
                engine.inhibited_actions[action] = 5

            if engine.active_hypothesis:
                engine.active_hypothesis.confidence = 0.0

        # Prefrontal item acquisition
        if engine.avatar_pos is not None:
            for pe in prev_entities:
                if (
                    pe.role in (EntityRole.MANIPULABLE, EntityRole.RESOURCE)
                    and pe.feature_id not in av_feats
                    and pe.feature_id != 0
                    and not engine.symbolic_theory.is_walkable(pe.feature_id)
                ):
                    dist_to_av = abs(pe.grid_pos[0] - engine.avatar_pos[0]) + abs(
                        pe.grid_pos[1] - engine.avatar_pos[1]
                    )
                    if dist_to_av <= 1:
                        engine.working_memory.acquire_item(
                            feature_id=pe.feature_id,
                            role=pe.role.value,
                            step=engine.step_counter,
                            position=pe.grid_pos,
                        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Mental Simulation Planner (Forward Imagination Search)
# ─────────────────────────────────────────────────────────────────────────────


class MentalSimulationPlanner:
    """Simulates candidate action sequences internally in imagination using learned world theory."""

    @staticmethod
    def is_corner_deadlock(
        pos: tuple[int, int],
        goals: set[tuple[int, int]],
        static_barriers: set[tuple[int, int]],
        H: int,
        W: int,
    ) -> bool:
        """Check whether a cargo block at pos is trapped in an irreversible corner deadlock.

        In push-delivery / Sokoban dynamics, when a block is placed into a corner
        formed by two orthogonal immovable barriers and is not already on a goal target,
        it can never be extracted or redirected. Pruning this state prevents thousands
        of fruitless search expansions.
        """
        if pos in goals:
            return False
        r, c = pos
        blocked_up = (r - 1 < 0) or ((r - 1, c) in static_barriers)
        blocked_down = (r + 1 >= H) or ((r + 1, c) in static_barriers)
        blocked_left = (c - 1 < 0) or ((r, c - 1) in static_barriers)
        blocked_right = (c + 1 >= W) or ((r, c + 1) in static_barriers)

        if (blocked_up or blocked_down) and (blocked_left or blocked_right):
            return True
        return False

    @staticmethod
    def is_wall_deadlock(
        pos: tuple[int, int],
        goals: set[tuple[int, int]],
        H: int,
        W: int,
    ) -> bool:
        """Check whether a cargo block is trapped along a boundary wall with no goals on it."""
        if pos in goals:
            return False
        r, c = pos
        if r == 0 and not any(g[0] == 0 for g in goals):
            return True
        if r == H - 1 and not any(g[0] == H - 1 for g in goals):
            return True
        if c == 0 and not any(g[1] == 0 for g in goals):
            return True
        if c == W - 1 and not any(g[1] == W - 1 for g in goals):
            return True
        return False

    @staticmethod
    def simulate_in_mind(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
        target_role: EntityRole = EntityRole.GOAL,
        blocked_cells: set[tuple[int, int]] | None = None,
    ) -> list[MentalSimulationStep] | None:
        """Simulate candidate action sequences internally in memory without taking physical steps."""
        if not engine.is_motor_grounded() or engine.avatar_pos is None:
            return None

        H, W = curr_grid.shape
        bg = engine.estimate_background(curr_grid)
        entities = engine.extract_entities(curr_grid, bg)

        av_feats = engine.avatar_features or (
            {engine.avatar_feature} if engine.avatar_feature is not None else set()
        )

        # 1. Identify goals using a principled epistemic priority hierarchy.
        #    A human brain resolves goal uncertainty in this order:
        #    (a) Confirmed goals — causal evidence from past wins
        #    (b) Hypothesized goals — prior beliefs carried from experience
        #    (c) Structural inference — geometric/topological patterns
        #    (d) Exploratory guesses — unknown entities worth investigating

        # (a) Confirmed goals: entities whose features have CAUSED wins
        confirmed_goals: list[tuple[int, int]] = []
        for e in entities:
            if engine.avatar_pos is not None and e.grid_pos == engine.avatar_pos:
                continue
            if e.feature_id in av_feats:
                if (
                    engine.avatar_pos is not None
                    and (
                        abs(e.grid_pos[0] - engine.avatar_pos[0])
                        + abs(e.grid_pos[1] - engine.avatar_pos[1])
                    )
                    > 4
                    and (e.role == EntityRole.GOAL or engine.symbolic_theory.is_goal(e))
                ):
                    confirmed_goals.append(e.grid_pos)
                continue
            if (
                engine.symbolic_theory.is_goal(e)
                or e.feature_id in engine.symbolic_theory.goal_features
                or (e.feature_id in engine.learned_goal_features and 1 <= e.area <= 64)
            ):
                if e.grid_pos not in confirmed_goals:
                    confirmed_goals.append(e.grid_pos)
                    engine.learned_goal_positions.add(e.grid_pos)

        # Object Permanence: Retain known goal positions if currently occluded by mobile entities
        for gp in engine.learned_goal_positions:
            if 0 <= gp[0] < H and 0 <= gp[1] < W and gp != engine.avatar_pos:
                if gp not in confirmed_goals:
                    confirmed_goals.append(gp)

        goals: list[tuple[int, int]] = []
        if confirmed_goals:
            goals = confirmed_goals

        # (b) Hypothesized goals: features believed to be goals from prior
        #     experience (candidate_goal_features retained across levels).
        panels = PerceptionEngine.detect_affordance_panels(entities, curr_grid, bg=bg)
        panel_coords: set[tuple[int, int]] = set()
        for p in panels:
            for item in p["items"]:
                r0, r1, c0, c1 = item.bounding_box
                for rr in range(max(0, r0 - 2), min(H, r1 + 3)):
                    for cc in range(max(0, c0 - 2), min(W, c1 + 3)):
                        panel_coords.add((rr, cc))

        # (b) Hypothesized goals from previous level knowledge or role inference.
        #     A human seeing a familiar-looking object assumes "that's probably
        #     the goal again" until evidence contradicts it.
        if not goals:
            hypothesized_goals: list[tuple[int, int]] = []
            for e in entities:
                if engine.avatar_pos is not None and e.grid_pos == engine.avatar_pos:
                    continue
                if engine.avatar_pos is not None and e.feature_id in av_feats:
                    if (
                        abs(e.grid_pos[0] - engine.avatar_pos[0]) <= 3
                        and abs(e.grid_pos[1] - engine.avatar_pos[1]) <= 3
                    ):
                        continue
                elif e.feature_id in av_feats:
                    continue
                if e.grid_pos in panel_coords:
                    continue
                if e.feature_id in engine.symbolic_theory.candidate_goal_features:
                    if e.grid_pos not in hypothesized_goals:
                        hypothesized_goals.append(e.grid_pos)
                elif (
                    e.role == target_role
                    or e.feature_id in engine.learned_receptacle_features
                    or e.grid_pos in engine.learned_goal_positions
                    or e.grid_pos in engine.learned_receptacle_positions
                ):
                    if e.grid_pos not in hypothesized_goals:
                        hypothesized_goals.append(e.grid_pos)
            if hypothesized_goals:
                goals = hypothesized_goals

        # (c) Structural inference: geometric patterns (unique small entities,
        #     isolated objects) that suggest goal-hood
        if not goals and engine.learned_goal_positions:
            goals = [
                g
                for g in engine.learned_goal_positions
                if 0 <= g[0] < H and 0 <= g[1] < W and g not in panel_coords
            ]
        if not goals:
            structural_goals = engine.detect_structural_goals(curr_grid)
            # Prioritize target zones and exit markers over speculative symmetry completion
            primary_sgs = [
                sg for sg in structural_goals if sg.get("type") in ("target_zone", "exit_marker")
            ]
            cand_sgs = primary_sgs if primary_sgs else structural_goals
            for sg in cand_sgs:
                feat = sg.get("feature")
                if feat is not None and (
                    feat in av_feats
                    or feat in engine.hazard_tracker.known_lethal_features
                    or engine.symbolic_theory.is_barrier(feat)
                ):
                    continue
                cells = sg.get("cells")
                if cells:
                    for cp in cells:
                        if (
                            0 <= cp[0] < H
                            and 0 <= cp[1] < W
                            and cp != engine.avatar_pos
                            and cp not in panel_coords
                        ):
                            if cp not in goals:
                                goals.append(cp)
                else:
                    pos = sg.get("position")
                    if (
                        pos
                        and 0 <= pos[0] < H
                        and 0 <= pos[1] < W
                        and pos != engine.avatar_pos
                        and pos not in panel_coords
                    ):
                        if pos not in goals:
                            goals.append(pos)

        # (d) Exploratory guesses: unknown entities worth investigating.
        #     This is the weakest prior — "I don't know what these are, but
        #     interacting with them might teach me something."
        if not goals:
            exploratory_goals = [
                e.grid_pos
                for e in entities
                if e.grid_pos not in panel_coords
                and e.role in (EntityRole.UNKNOWN, EntityRole.AGENT)
                and 1 <= e.area <= 64
                and e.feature_id != bg
                and e.feature_id not in engine.hazard_tracker.known_lethal_features
                and not engine.symbolic_theory.is_barrier(e.feature_id)
                and (
                    e.feature_id not in av_feats
                    or (
                        engine.avatar_pos is not None
                        and (
                            abs(e.grid_pos[0] - engine.avatar_pos[0])
                            + abs(e.grid_pos[1] - engine.avatar_pos[1])
                        )
                        > 4
                    )
                )
            ]
            if exploratory_goals:
                goals = exploratory_goals
                for e in entities:
                    if (
                        e.grid_pos in goals
                        and e.feature_id not in av_feats
                        and e.grid_pos not in panel_coords
                    ):
                        engine.symbolic_theory.candidate_goal_features.add(e.feature_id)

        av_feats = engine.avatar_features or (
            {engine.avatar_feature} if engine.avatar_feature is not None else set()
        )
        engine.symbolic_theory.goal_features.difference_update(av_feats)
        if engine.avatar_pos is not None:
            goals = [
                g
                for g in goals
                if g != engine.avatar_pos and (g in confirmed_goals or g not in panel_coords)
            ]

        unexhausted_goals = [g for g in goals if g not in engine.exhausted_candidate_goals]
        if unexhausted_goals:
            goals = unexhausted_goals
        elif goals:
            engine.exhausted_candidate_goals.clear()

        if engine.avatar_pos is not None and len(goals) > 8:
            goals.sort(
                key=lambda g: abs(g[0] - engine.avatar_pos[0]) + abs(g[1] - engine.avatar_pos[1])
            )
            goals = goals[:8]

        if not goals:
            return None

        # 2. Identify candidate pushable blocks / cargo
        pushable_blocks: list[tuple[int, int]] = [
            e.grid_pos
            for e in entities
            if (
                engine.symbolic_theory.is_cargo(e)
                or e.feature_id in engine.learned_cargo_features
                or (
                    e.role == EntityRole.MANIPULABLE
                    and 1 <= e.area <= 64
                    and e.feature_id != bg
                    and e.feature_id not in av_feats
                    and e.grid_pos not in goals
                )
            )
            and e.feature_id not in engine.hazard_tracker.known_lethal_features
            and not engine.symbolic_theory.is_barrier(e.feature_id)
        ]

        # 3. Static barriers
        static_barriers: set[tuple[int, int]] = set()
        for r, c in engine.learned_barriers:
            if 0 <= r < H and 0 <= c < W:
                val = int(curr_grid[r, c])
                if (r, c) in engine.hazard_tracker.static_lethal_positions or (
                    val != bg and not engine.symbolic_theory.is_walkable(val)
                ):
                    static_barriers.add((r, c))

        u_vals, u_counts = np.unique(curr_grid, return_counts=True)
        feat_counts = dict(zip(u_vals, u_counts))
        for r in range(H):
            for c in range(W):
                val = int(curr_grid[r, c])
                if engine.symbolic_theory.is_barrier(val) or (
                    val in engine.hazard_tracker.known_lethal_features
                    and feat_counts.get(val, 0) >= 25
                ):
                    static_barriers.add((r, c))

        # HCIR Attractor Invariant: Goals must never be static barriers
        static_barriers.difference_update(goals)
        if blocked_cells:
            static_barriers.update(blocked_cells)

        # Check known state mutation triggers (switches that open doors)
        mutation_triggers: dict[tuple[int, int], set[tuple[int, int]]] = {}
        for m in engine.state_mutations:
            opened_cells: set[tuple[int, int]] = set()
            for r in range(H):
                for c in range(W):
                    if int(curr_grid[r, c]) == m.prior_value and (
                        m.posterior_value == bg or m.posterior_value == 0
                    ):
                        opened_cells.add((r, c))
            if not opened_cells:
                continue

            if (
                m.trigger_pos is not None
                and len(m.trigger_pos) >= 2
                and 0 <= m.trigger_pos[0] < H
                and 0 <= m.trigger_pos[1] < W
            ):
                tr_p = (int(m.trigger_pos[0]), int(m.trigger_pos[1]))
                mutation_triggers.setdefault(tr_p, set()).update(opened_cells)

            if (
                m.trigger_feature is not None
                and m.trigger_feature != bg
                and m.trigger_feature != engine.bg_feature
                and not engine.symbolic_theory.is_walkable(m.trigger_feature)
            ):
                for r in range(H):
                    for c in range(W):
                        if int(curr_grid[r, c]) == m.trigger_feature:
                            mutation_triggers.setdefault((r, c), set()).update(opened_cells)

        # 4. Available directional movements in mental model
        movable_actions: list[tuple[Any, int, int]] = []
        for act in available_actions:
            if engine.is_spatial_effector(act):
                continue
            dyn = engine.action_dynamics.get(act)
            if dyn is not None and dyn.is_displacement_action():
                dr, dc = dyn.get_displacement()
                movable_actions.append((act, dr, dc))

        if not movable_actions:
            return None

        # Unknown solid entity cells: risk-aware path planning
        unknown_entity_cells: set[tuple[int, int]] = set()
        for e in entities:
            if (
                e.feature_id != bg
                and e.feature_id != 0
                and e.feature_id not in av_feats
                and not engine.symbolic_theory.is_walkable(e.feature_id)
                and not engine.symbolic_theory.is_goal(e)
                and not engine.symbolic_theory.is_barrier(e.feature_id)
                and not engine.symbolic_theory.is_cargo(e)
                and e.feature_id not in engine.learned_cargo_features
                and e.role != EntityRole.MANIPULABLE
                and e.grid_pos not in goals
            ):
                for cell in e.properties.get("cells", [e.grid_pos]):
                    unknown_entity_cells.add(cell)

        # Determine movement displacement scale
        step_size = 1
        for act, dr, dc in movable_actions:
            dist = max(abs(dr), abs(dc))
            if dist > step_size:
                step_size = dist

        # Theory of Mind / Perspective Taking: Detect creatures with directional gaze cones
        oriented_threats = PerceptionEngine.detect_oriented_threats(
            curr_grid,
            entities,
            bg=bg,
            step_size=step_size,
            avatar_pos=engine.avatar_pos,
            avatar_features=av_feats,
            goals=set(goals),
        )
        # Sentries (stationary, lethal gaze cone) vs patrollers (observed to move:
        # they advance one cell per avatar move along their facing, turning
        # around when the corridor ahead ends).
        static_threats = [
            t for t in oriented_threats if t.feature_id not in engine.mobile_threat_features
        ]
        init_threats = frozenset(t.pos for t in static_threats)
        threat_map = {t.pos: t for t in static_threats}
        init_patrols: frozenset[tuple[tuple[int, int], tuple[int, int]]] = frozenset(
            (t.pos, t.facing)
            for t in oriented_threats
            if t.feature_id in engine.mobile_threat_features
        )
        half_step = max(1, step_size // 2)

        # Build a set of cells that are known walls/barriers on the observed
        # grid.  This supplements static_barriers (entity-level) with
        # pixel-level barrier features the symbolic theory has confirmed.
        _barrier_feats = engine.symbolic_theory.barrier_features
        _grid_walls: set[tuple[int, int]] = set(static_barriers)
        for rr in range(H):
            for cc in range(W):
                if int(curr_grid[rr, cc]) in _barrier_feats:
                    _grid_walls.add((rr, cc))

        def _corridor_blocked(p: tuple[int, int], f: tuple[int, int]) -> bool:
            """Check whether a patroller at *p* facing *f* is blocked.

            Mirrors the game's ``rgwzxyjuqc`` logic: it probes one half-step
            ahead (the intermediate map cell) and one full step ahead (the
            destination grid cell).  If either is out-of-bounds or a known
            wall/barrier, the corridor is blocked.
            """
            probe = (p[0] + f[0] * half_step, p[1] + f[1] * half_step)
            dest = (p[0] + f[0] * step_size, p[1] + f[1] * step_size)
            for q in (probe, dest):
                if not (0 <= q[0] < H and 0 <= q[1] < W) or q in _grid_walls:
                    return True
            return False

        def advance_patrols(
            patrols: frozenset[tuple[tuple[int, int], tuple[int, int]]],
            avatar_new: tuple[int, int],
        ) -> frozenset[tuple[tuple[int, int], tuple[int, int]]] | None:
            """Predict patrollers after one avatar move.  ``None`` = avatar dies.

            Mirrors the game's actual step sequence:
              1. Phase 1 — patroller moves one cell in its current facing.
              2. Phase 2 — patroller checks if the NEXT cell ahead is blocked;
                 if so it reverses facing for the *next* turn.

            The previous implementation reversed direction *before* moving,
            which caused off-by-one prediction errors near corridor ends.
            """
            nxt: set[tuple[tuple[int, int], tuple[int, int]]] = set()
            for p, f in patrols:
                # Avatar stepped onto patroller → patroller eliminated.
                if p == avatar_new:
                    continue

                # 1. Move one cell forward in current facing.
                dest = (p[0] + f[0] * step_size, p[1] + f[1] * step_size)
                if not (0 <= dest[0] < H and 0 <= dest[1] < W) or dest in _grid_walls:
                    # Can't actually move forward — stay and reverse.
                    new_f = (-f[0], -f[1])
                    nxt.add((p, new_f))
                    continue

                # Patroller walks into the avatar → avatar dies.
                if dest == avatar_new:
                    return None

                # 2. After arriving at dest, check if the NEXT cell ahead is
                #    blocked.  If so, reverse facing for the subsequent turn.
                new_f = f
                if _corridor_blocked(dest, f):
                    new_f = (-f[0], -f[1])
                nxt.add((dest, new_f))
            return frozenset(nxt)

        # Pre-compute the set of cells patrollers will occupy next turn.
        # Used for proximity penalties during A* expansion.
        def _patrol_next_cells(
            patrols: frozenset[tuple[tuple[int, int], tuple[int, int]]],
        ) -> set[tuple[int, int]]:
            """Return the set of cells patrollers will move INTO next turn."""
            cells: set[tuple[int, int]] = set()
            for p, f in patrols:
                dest = (p[0] + f[0] * step_size, p[1] + f[1] * step_size)
                if 0 <= dest[0] < H and 0 <= dest[1] < W and dest not in _grid_walls:
                    cells.add(dest)
                else:
                    cells.add(p)  # staying put after reversal
            return cells

        start_pos = engine.avatar_pos
        init_blocks = frozenset(pushable_blocks)
        init_open: frozenset[tuple[int, int]] = frozenset()

        # Theory of Mind: Detect active chasing adversaries (agents moving towards avatar)
        init_chasers: frozenset[tuple[int, int]] = frozenset(
            e.grid_pos
            for e in entities
            if (
                e.feature_id in engine.mobile_threat_features
                or (
                    e.role == EntityRole.AGENT
                    and e.grid_pos != start_pos
                    and e.feature_id not in av_feats
                    and e.feature_id != bg
                )
            )
            and e.grid_pos not in init_threats
            and not any(p[0] == e.grid_pos for p in init_patrols)
        )

        def advance_chasers(
            chasers: frozenset[tuple[int, int]],
            avatar_target: tuple[int, int],
        ) -> frozenset[tuple[int, int]] | None:
            """Advance chasing adversaries one step towards avatar using greedy pursuit."""
            nxt: set[tuple[int, int]] = set()
            for cp in chasers:
                if cp == avatar_target:
                    return None
                dr = 1 if avatar_target[0] > cp[0] else (-1 if avatar_target[0] < cp[0] else 0)
                dc = 1 if avatar_target[1] > cp[1] else (-1 if avatar_target[1] < cp[1] else 0)

                dest = cp
                if abs(avatar_target[0] - cp[0]) >= abs(avatar_target[1] - cp[1]) and dr != 0:
                    cand = (cp[0] + dr, cp[1])
                    if cand not in _grid_walls:
                        dest = cand
                    elif dc != 0 and (cp[0], cp[1] + dc) not in _grid_walls:
                        dest = (cp[0], cp[1] + dc)
                elif dc != 0:
                    cand = (cp[0], cp[1] + dc)
                    if cand not in _grid_walls:
                        dest = cand
                    elif dr != 0 and (cp[0] + dr, cp[1]) not in _grid_walls:
                        dest = (cp[0] + dr, cp[1])

                if dest == avatar_target:
                    return None
                nxt.add(dest)
            return frozenset(nxt)

        is_block_delivery = bool(
            1 <= len(pushable_blocks) <= 5
            and any(
                engine.symbolic_theory.is_cargo(e)
                or e.feature_id in engine.learned_cargo_features
                or e.role == EntityRole.MANIPULABLE
                for e in entities
            )
        )

        def heuristic(pos: tuple[int, int], blocks: frozenset[tuple[int, int]]) -> float:
            if is_block_delivery and blocks:
                b_list = list(blocks)
                goal_dist = 0.0
                for g in goals:
                    goal_dist += min(abs(g[0] - b[0]) + abs(g[1] - b[1]) for b in b_list)
                avatar_to_b = min(abs(pos[0] - b[0]) + abs(pos[1] - b[1]) for b in b_list)
                return float(goal_dist * 2.0 + avatar_to_b)
            else:
                return float(min(abs(g[0] - pos[0]) + abs(g[1] - pos[1]) for g in goals))

        counter = 0
        open_set: list[
            tuple[
                float,
                float,
                int,
                tuple[int, int],
                frozenset[tuple[int, int]],
                frozenset[tuple[int, int]],
                frozenset[tuple[int, int]],
                frozenset[tuple[tuple[int, int], tuple[int, int]]],
                frozenset[tuple[int, int]],
                list[MentalSimulationStep],
            ]
        ] = []
        h0 = heuristic(start_pos, init_blocks)
        heapq.heappush(
            open_set,
            (
                h0,
                0,
                counter,
                start_pos,
                init_blocks,
                init_open,
                init_threats,
                init_patrols,
                init_chasers,
                [],
            ),
        )

        visited_states: set[
            tuple[
                tuple[int, int],
                frozenset[tuple[int, int]],
                frozenset[tuple[int, int]],
                frozenset[tuple[int, int]],
                frozenset[tuple[tuple[int, int], tuple[int, int]]],
                frozenset[tuple[int, int]],
                int,
            ]
        ] = set()
        has_periodic = (
            bool(engine.hazard_tracker.periodic_cells)
            and engine.hazard_tracker.environmental_period >= 2
        )
        has_temporal = bool(init_patrols) or bool(init_chasers) or has_periodic
        max_expansions = 1500 if not has_temporal else 5000

        # When patrollers or periodic hazards are present, add a NO_OP "wait" action.
        # A human watching an oscillating hazard or patroller knows to WAIT for it to clear.
        wait_action: Any | None = None
        if has_temporal:
            for a in available_actions:
                if a in engine.action_dynamics:
                    if not engine.action_dynamics[a].is_displacement_action():
                        wait_action = a
                        break
                elif a == 5:
                    wait_action = 5
                    break
            if wait_action is None and movable_actions:
                for act, dr, dc in movable_actions:
                    adj_r, adj_c = start_pos[0] + dr, start_pos[1] + dc
                    if (adj_r, adj_c) in static_barriers or not (0 <= adj_r < H and 0 <= adj_c < W):
                        wait_action = act
                        break
                if wait_action is None:
                    wait_action = movable_actions[0][0]

        while open_set and max_expansions > 0:
            max_expansions -= 1
            (
                f_score,
                g_cost,
                _,
                cur_pos,
                cur_blocks,
                cur_open,
                cur_threats,
                cur_patrols,
                cur_chasers,
                path,
            ) = heapq.heappop(open_set)

            time_mod = len(path) % engine.hazard_tracker.environmental_period if has_periodic else 0
            state_key = (
                cur_pos,
                cur_blocks,
                cur_open,
                cur_threats,
                cur_patrols,
                cur_chasers,
                time_mod,
            )
            if state_key in visited_states:
                continue
            visited_states.add(state_key)

            # Direct goal reach: avatar reaches goal position (only if not a block delivery task)
            if not is_block_delivery and cur_pos in goals and len(path) > 0:
                logger.info(
                    "AutonomousEpistemicEngine: Mental Simulation SUCCEEDED! Synthesized %d-step path to goal %s.",
                    len(path),
                    cur_pos,
                )
                engine.current_simulated_goal = cur_pos
                return path

            # Block delivery: all cargo pushed into goals/receptacles
            if is_block_delivery and all(g in cur_blocks for g in goals) and len(path) > 0:
                logger.info(
                    "AutonomousEpistemicEngine: Compound Mental Simulation SUCCEEDED! Synthesized %d-step block delivery plan.",
                    len(path),
                )
                engine.current_simulated_goal = list(goals)[0] if goals else None
                return path

            # ── Candidate actions: regular moves + optional wait ────────────
            candidate_actions: list[tuple[Any, int, int]] = list(movable_actions)
            # Add the wait option when patrollers or periodic hazards are present,
            # and cap consecutive waits so the planner doesn't idle forever.
            consecutive_waits = 0
            if path:
                for s in reversed(path):
                    if s.predicted_avatar_pos == cur_pos:
                        consecutive_waits += 1
                    else:
                        break
            if wait_action is not None and (cur_patrols or has_periodic) and consecutive_waits < 4:
                candidate_actions.append(("__WAIT__", 0, 0))  # sentinel ID

            for act, dr, dc in candidate_actions:
                is_wait = act == "__WAIT__"
                if is_wait:
                    if 5 in available_actions and (
                        5 not in engine.action_dynamics
                        or not engine.action_dynamics[5].is_displacement_action()
                    ):
                        real_act = 5
                    else:
                        bump_act = None
                        for act_cand, mdr, mdc in movable_actions:
                            br, bc = cur_pos[0] + mdr, cur_pos[1] + mdc
                            if (br, bc) in static_barriers or not (0 <= br < H and 0 <= bc < W):
                                bump_act = act_cand
                                break
                        real_act = bump_act if bump_act is not None else wait_action
                else:
                    real_act = act
                if not is_wait and (cur_pos, act) in engine.failed_transitions:
                    dest_r, dest_c = cur_pos[0] + dr, cur_pos[1] + dc
                    if not (
                        has_periodic and (dest_r, dest_c) in engine.hazard_tracker.periodic_cells
                    ):
                        continue
                nr, nc = cur_pos[0] + dr, cur_pos[1] + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue

                step_r = int(np.sign(dr))
                step_c = int(np.sign(dc))
                dist = max(abs(dr), abs(dc))
                ray_blocked = False
                for k in range(1, dist + 1):
                    kr = cur_pos[0] + step_r * k
                    kc = cur_pos[1] + step_c * k
                    if (kr, kc) in static_barriers and (kr, kc) not in cur_open:
                        ray_blocked = True
                        break
                if ray_blocked:
                    continue

                sim_step = len(path) + 1
                if engine.hazard_tracker.is_hazard_at(
                    nr, nc, sim_step, bg, avatar_features=av_feats
                ):
                    continue
                if (
                    sim_step <= 1
                    and int(curr_grid[nr, nc]) in engine.hazard_tracker.known_lethal_features
                    and int(curr_grid[nr, nc]) not in engine.mobile_threat_features
                ):
                    continue

                # Living threat gaze cones (Theory of Mind: avoid creature sightlines)
                living_gaze: set[tuple[int, int]] = set()
                for t_pos in cur_threats:
                    t = threat_map.get(t_pos)
                    if t is not None:
                        living_gaze.add(t.gaze_pos)

                if (nr, nc) in living_gaze:
                    continue

                # Stepping onto an active threat:
                # If approaching from its front gaze cell -> charging into its face -> suicidal!
                # If approaching from flank or rear -> stealth ambush / bypass -> eliminate threat!
                new_threats = cur_threats
                if (nr, nc) in cur_threats:
                    t = threat_map.get((nr, nc))
                    if t is not None:
                        if cur_pos == t.gaze_pos:
                            continue  # Suicidal frontal charge!
                        new_threats = cur_threats - {(nr, nc)}

                # Patrollers: imagine where every patrolling creature will be
                # after this move. If one walks into us, this branch is fatal.
                new_patrols = cur_patrols
                patrol_proximity_cost = 0.0
                if cur_patrols:
                    predicted = advance_patrols(cur_patrols, (nr, nc))
                    if predicted is None:
                        continue
                    new_patrols = predicted

                    # Proximity penalty: stepping into a cell that a patroller
                    # is about to enter next turn is extremely dangerous
                    # even if the patroller doesn't collide THIS turn.
                    upcoming = _patrol_next_cells(cur_patrols)
                    if (nr, nc) in upcoming:
                        patrol_proximity_cost += 15.0
                    # Also penalize cells adjacent to current patrol positions
                    for pp, _pf in cur_patrols:
                        dist_to_patrol = abs(nr - pp[0]) + abs(nc - pp[1])
                        if 0 < dist_to_patrol <= step_size:
                            patrol_proximity_cost += 5.0

                # Chasing adversaries: simulate greedy pursuit towards new avatar pos
                new_chasers = cur_chasers
                chaser_proximity_cost = 0.0
                if cur_chasers:
                    predicted_chasers = advance_chasers(cur_chasers, (nr, nc))
                    if predicted_chasers is None:
                        continue  # Captured by chaser! Fatal branch
                    new_chasers = predicted_chasers
                    for cp in new_chasers:
                        dist_to_chaser = abs(nr - cp[0]) + abs(nc - cp[1])
                        if dist_to_chaser <= 1:
                            chaser_proximity_cost += 20.0
                        elif dist_to_chaser <= 2:
                            chaser_proximity_cost += 5.0

                # ── Epistemic Uncertainty Cost ──────────────────────────────────
                # A human doesn't REFUSE to enter unknown territory — they're
                # cautious, preferring verified routes but willing to try
                # unverified ones when no safe alternative exists. Only
                # CONFIRMED lethal features (empirical evidence: "I died on
                # this") are hard-blocked. Everything else gets a risk cost
                # proportional to epistemic uncertainty.
                epistemic_cost = 0.0
                for k in range(1, dist + 1):
                    kr = cur_pos[0] + step_r * k
                    kc = cur_pos[1] + step_c * k
                    if 0 <= kr < H and 0 <= kc < W:
                        cell_feat = int(curr_grid[kr, kc])
                        if cell_feat in engine.hazard_tracker.known_lethal_features:
                            if cell_feat in engine.mobile_threat_features:
                                # Snapshot position of a patroller; its real
                                # trajectory is simulated dynamically above.
                                pass
                            else:
                                is_flanked_threat = (
                                    (kr, kc) in cur_threats
                                    and (kr, kc) == (nr, nc)
                                    and (kr, kc) in threat_map
                                    and cur_pos != threat_map[(kr, kc)].gaze_pos
                                )
                                if not is_flanked_threat:
                                    epistemic_cost = float("inf")  # hard-block confirmed lethal
                                    break
                        if engine.is_feature_unverified(cell_feat, bg):
                            epistemic_cost += engine.UNVERIFIED_FEATURE_COST
                        if (kr, kc) in unknown_entity_cells:
                            epistemic_cost += 10.0  # additional cost for entity presence
                if epistemic_cost == float("inf"):
                    continue

                new_blocks = cur_blocks
                new_pos = (nr, nc)

                # Intuitive Physics: simulate falling under gravity if unsupported
                if engine.physics_engine.has_gravity:
                    if not engine.physics_engine.is_supported(
                        nr, nc, curr_grid, engine.symbolic_theory.barrier_features, static_barriers
                    ):
                        land_pos, fall_traj, is_lethal = engine.physics_engine.project_fall(
                            nr,
                            nc,
                            curr_grid,
                            engine.symbolic_theory.barrier_features,
                            static_barriers,
                            lethal_features=engine.hazard_tracker.known_lethal_features,
                        )
                        if is_lethal:
                            continue
                        new_pos = land_pos

                if (nr, nc) in cur_blocks:
                    pushed_r, pushed_c = nr + dr, nc + dc
                    if not (0 <= pushed_r < H and 0 <= pushed_c < W):
                        continue
                    if (
                        pushed_r,
                        pushed_c,
                    ) in static_barriers and (pushed_r, pushed_c) not in cur_open:
                        continue
                    if (pushed_r, pushed_c) in cur_blocks:
                        continue
                    if engine.hazard_tracker.is_hazard_at(
                        pushed_r, pushed_c, sim_step, bg, avatar_features=av_feats
                    ):
                        continue

                    # Fall under gravity if unsupported
                    if engine.physics_engine.has_gravity:
                        if not engine.physics_engine.is_supported(
                            pushed_r,
                            pushed_c,
                            curr_grid,
                            engine.symbolic_theory.barrier_features,
                            static_barriers,
                        ):
                            land_b, _, is_b_lethal = engine.physics_engine.project_fall(
                                pushed_r,
                                pushed_c,
                                curr_grid,
                                engine.symbolic_theory.barrier_features,
                                static_barriers,
                            )
                            if is_b_lethal:
                                continue
                            pushed_r, pushed_c = land_b

                    # Irreversibility & Deadlock Detection:
                    # Prune branches where cargo is shoved into a non-goal corner or dead wall
                    if is_block_delivery and (pushed_r, pushed_c) not in goals:
                        if MentalSimulationPlanner.is_corner_deadlock(
                            (pushed_r, pushed_c),
                            set(goals),
                            static_barriers,
                            H,
                            W,
                        ):
                            continue
                        if MentalSimulationPlanner.is_wall_deadlock(
                            (pushed_r, pushed_c),
                            set(goals),
                            H,
                            W,
                        ):
                            continue

                    block_set = set(cur_blocks)
                    block_set.remove((nr, nc))
                    block_set.add((pushed_r, pushed_c))
                    new_blocks = frozenset(block_set)

                new_open = cur_open
                if new_pos in mutation_triggers:
                    new_open = cur_open | frozenset(mutation_triggers[new_pos])

                # Wait actions cost slightly more to discourage unnecessary idling
                wait_cost = 2.0 if is_wait else 0.0

                # Total step cost = base (1) + epistemic + patrol proximity + chaser proximity + wait
                step_cost = (
                    1 + epistemic_cost + patrol_proximity_cost + chaser_proximity_cost + wait_cost
                )
                new_g = g_cost + step_cost
                new_h = heuristic(new_pos, new_blocks)
                new_step = MentalSimulationStep(action=real_act, predicted_avatar_pos=new_pos)
                counter += 1
                heapq.heappush(
                    open_set,
                    (
                        new_g + new_h,
                        new_g,
                        counter,
                        new_pos,
                        new_blocks,
                        new_open,
                        new_threats,
                        new_patrols,
                        new_chasers,
                        path + [new_step],
                    ),
                )

        # Hierarchical Subgoal Decomposition Fallback
        if mutation_triggers or goals:
            subgoal_plan = MentalSimulationPlanner._plan_hierarchical_subgoals(
                engine,
                curr_grid,
                available_actions,
                start_pos,
                set(goals),
                static_barriers,
                mutation_triggers,
                movable_actions,
            )
            if subgoal_plan:
                return subgoal_plan

        return None

    @staticmethod
    def _plan_hierarchical_subgoals(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
        start_pos: tuple[int, int],
        goals: set[tuple[int, int]],
        static_barriers: set[tuple[int, int]],
        mutation_triggers: dict[tuple[int, int], set[tuple[int, int]]],
        movable_actions: list[tuple[int, int, int]],
    ) -> list[MentalSimulationStep] | None:
        """Recursive multi-stage subgoal decomposition for locked barrier doors and remote switches."""
        H, W = curr_grid.shape
        bg = engine.estimate_background(curr_grid)
        cur_pos = start_pos
        open_barriers: set[tuple[int, int]] = set()
        accumulated_plan: list[MentalSimulationStep] = []
        max_stages = 8

        def find_shortest_path(
            s_pos: tuple[int, int],
            target_positions: set[tuple[int, int]],
            active_barriers: set[tuple[int, int]],
        ) -> list[MentalSimulationStep] | None:
            q: deque[tuple[tuple[int, int], list[MentalSimulationStep]]] = deque([(s_pos, [])])
            visited = {s_pos}
            while q:
                p, path = q.popleft()
                if p in target_positions:
                    return path
                for act, dr, dc in movable_actions:
                    nr, nc = p[0] + dr, p[1] + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if (nr, nc) in visited:
                        continue

                    step_r = int(np.sign(dr))
                    step_c = int(np.sign(dc))
                    dist = max(abs(dr), abs(dc))
                    ray_blocked = False
                    for k in range(1, dist + 1):
                        kr = p[0] + step_r * k
                        kc = p[1] + step_c * k
                        if (kr, kc) in active_barriers:
                            ray_blocked = True
                            break
                        # Same epistemic caution as the main planner:
                        # never route through unverified features.
                        if 0 <= kr < H and 0 <= kc < W:
                            if engine.is_feature_unverified(int(curr_grid[kr, kc]), bg):
                                ray_blocked = True
                                break
                    if ray_blocked:
                        continue

                    visited.add((nr, nc))
                    step = MentalSimulationStep(action=act, predicted_avatar_pos=(nr, nc))
                    q.append(((nr, nc), path + [step]))
            return None

        for _ in range(max_stages):
            effective_barriers = static_barriers - open_barriers
            goal_path = find_shortest_path(cur_pos, goals, effective_barriers)
            if goal_path is not None:
                accumulated_plan.extend(goal_path)
                logger.info(
                    "AutonomousEpistemicEngine: Hierarchical Subgoal Decomposition SUCCEEDED with %d total steps!",
                    len(accumulated_plan),
                )
                return accumulated_plan

            candidate_triggers: list[tuple[tuple[int, int], list[MentalSimulationStep], int]] = []
            for tr_pos, opened_set in mutation_triggers.items():
                unopened = opened_set - open_barriers
                if not unopened:
                    continue
                tr_path = find_shortest_path(cur_pos, {tr_pos}, effective_barriers)
                if tr_path is not None:
                    candidate_triggers.append((tr_pos, tr_path, len(unopened)))

            if not candidate_triggers:
                if engine.working_memory.held_items:
                    adjacent_barrier_cells: set[tuple[int, int]] = set()
                    for br, bc in effective_barriers:
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ar, ac = br + dr, bc + dc
                            if (ar, ac) not in effective_barriers and 0 <= ar < H and 0 <= ac < W:
                                adjacent_barrier_cells.add((ar, ac))
                    if adjacent_barrier_cells:
                        approach_path = find_shortest_path(
                            cur_pos, adjacent_barrier_cells, effective_barriers
                        )
                        if approach_path:
                            accumulated_plan.extend(approach_path)
                            return accumulated_plan

                bg = engine.estimate_background(curr_grid)
                entities = engine.extract_entities(curr_grid, bg)
                av_feats = engine.avatar_features or (
                    {engine.avatar_feature} if engine.avatar_feature is not None else set()
                )
                candidate_tools = [
                    e.grid_pos
                    for e in entities
                    if e.role in (EntityRole.MANIPULABLE, EntityRole.RESOURCE, EntityRole.UNKNOWN)
                    and 1 <= e.area <= 64
                    and e.feature_id not in av_feats
                    and e.grid_pos not in goals
                ]
                for t_pos in candidate_tools:
                    t_path = find_shortest_path(cur_pos, {t_pos}, effective_barriers)
                    if t_path is not None:
                        candidate_triggers.append((t_pos, t_path, 1))

            if not candidate_triggers:
                break

            candidate_triggers.sort(key=lambda x: (len(x[1]), -x[2]))
            chosen_pos, switch_path, _ = candidate_triggers[0]

            accumulated_plan.extend(switch_path)
            cur_pos = chosen_pos
            open_barriers.update(mutation_triggers.get(chosen_pos, set()))

            matching_mutations = [
                m
                for m in engine.state_mutations
                if (
                    m.trigger_pos == chosen_pos
                    or (
                        m.trigger_feature is not None
                        and int(curr_grid[chosen_pos[0], chosen_pos[1]]) == m.trigger_feature
                    )
                )
            ]
            if matching_mutations and matching_mutations[0].trigger_type == "ACTION":
                act_id = int(matching_mutations[0].metadata.get("action_id", 5))
                if act_id in available_actions:
                    accumulated_plan.append(
                        MentalSimulationStep(
                            action=act_id,
                            predicted_avatar_pos=cur_pos,
                            expected_mutation="ACTION_TRIGGER",
                        )
                    )

        return None


# ─────────────────────────────────────────────────────────────────────────────
# 6. Epistemic Curiosity Explorer
# ─────────────────────────────────────────────────────────────────────────────


class EpistemicCuriosityExplorer:
    """Active curiosity-driven hypothesis testing and information-gain exploration."""

    @staticmethod
    def plan_epistemic_probe(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
    ) -> tuple[int, dict[str, Any] | None]:
        """Select an exploratory probe action to resolve epistemic uncertainty."""
        H, W = curr_grid.shape
        bg = engine.estimate_background(curr_grid)
        entities = engine.extract_entities(curr_grid, bg)
        engine.level_epistemic_probes += 1
        engine.total_epistemic_probes += 1

        # 1. Uncalibrated actions take absolute priority for motor grounding
        uncalibrated = [a for a in available_actions if not engine.is_action_calibrated(a)]
        untested_at_pos = [
            a for a in uncalibrated if (a, engine.avatar_pos) not in engine.tested_action_positions
        ]
        if untested_at_pos:
            chosen_action = untested_at_pos[0]
            engine.tested_action_positions.add((chosen_action, engine.avatar_pos))
            engine.tested_actions.add(chosen_action)

            action_data = None
            if engine.is_spatial_effector(chosen_action):
                action_data = engine.ground_effector_action(curr_grid, chosen_action)
            return chosen_action, action_data

        # 2. Dynamic Affordance Partitioning (Modality-Agnostic, No Hardcoded Action IDs)
        displacement_actions = [a for a in available_actions if engine.is_displacement_action(a)]
        spatial_effector_actions = [a for a in available_actions if engine.is_spatial_effector(a)]
        discrete_transform_actions = [
            a
            for a in available_actions
            if a not in displacement_actions and a not in spatial_effector_actions
        ]

        # 2b. Motor babbling for self-identification. Agency is established by
        # observing which entity co-varies with DIFFERENT self-generated actions,
        # so vary actions round-robin until the controllability test commits.
        if displacement_actions and not engine.avatar_features and engine.avatar_feature is None:
            babble_idx = engine.level_epistemic_probes % len(displacement_actions)
            return displacement_actions[babble_idx], None

        # A. Non-displacement or Spatial Effector Environments
        if spatial_effector_actions:
            should_probe_effector = (
                not displacement_actions
                or (engine.avatar_pos is None and engine.level_epistemic_probes > 12)
                or (engine.avatar_pos is not None and engine.consecutive_stuck_steps >= 4)
            )
            if should_probe_effector:
                chosen_eff = spatial_effector_actions[0]
                return chosen_eff, engine.ground_effector_action(curr_grid, chosen_eff)

        # 3. Spatial Movement Curiosity with Target Commitment & Loop Breaking
        if engine.avatar_pos is not None:
            engine.recent_positions.append(engine.avatar_pos)
            engine.position_visit_counts[engine.avatar_pos] = (
                engine.position_visit_counts.get(engine.avatar_pos, 0) + 1
            )

            for e in entities:
                if (
                    abs(e.grid_pos[0] - engine.avatar_pos[0])
                    + abs(e.grid_pos[1] - engine.avatar_pos[1])
                    <= 1
                ):
                    engine.probed_entity_ids.add(e.id)

            if engine.active_probe_target is not None:
                dist_to_target = abs(engine.active_probe_target[0] - engine.avatar_pos[0]) + abs(
                    engine.active_probe_target[1] - engine.avatar_pos[1]
                )
                engine.probe_target_steps += 1
                if dist_to_target <= 1 or engine.probe_target_steps > 12:
                    if engine.active_probe_id:
                        engine.probed_entity_ids.add(engine.active_probe_id)
                    engine.active_probe_target = None
                    engine.active_probe_id = None
                    engine.probe_target_steps = 0

            # Oscillation & Attractor Loop Breaking
            is_stuck = engine.consecutive_stuck_steps >= 2
            is_oscillating = engine.recent_positions.count(engine.avatar_pos) >= 3 or is_stuck
            if is_oscillating:
                logger.debug(
                    "AutonomousEpistemicEngine: Attractor loop/stuck state detected at %s! Breaking loop...",
                    engine.avatar_pos,
                )
                engine.active_probe_target = None
                engine.active_probe_id = None
                uncalibrated = [
                    a
                    for a in available_actions
                    if a not in engine.action_dynamics
                    or not engine.action_dynamics[a].is_displacement_action()
                ]
                untested_here = [
                    a
                    for a in uncalibrated
                    if (a, engine.avatar_pos) not in engine.tested_action_positions
                ]
                if untested_here:
                    act = untested_here[0]
                    engine.tested_action_positions.add((act, engine.avatar_pos))
                    return act, None
                best_act = None
                min_visits = float("inf")
                disp_actions = [
                    (act, engine.action_dynamics[act].get_displacement())
                    for act in available_actions
                    if act in engine.action_dynamics
                    and engine.action_dynamics[act].is_displacement_action()
                ]
                for act, (dr_cal, dc_cal) in disp_actions:
                    if engine.inhibited_actions.get(act, 0) > 0:
                        continue
                    if (engine.avatar_pos, act) in engine.failed_transitions:
                        continue
                    nr = engine.avatar_pos[0] + dr_cal
                    nc = engine.avatar_pos[1] + dc_cal
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if (
                        (nr, nc) in engine.hazard_tracker.static_lethal_positions
                        or (nr, nc) in engine.learned_barriers
                        or engine.symbolic_theory.is_barrier(int(curr_grid[nr, nc]))
                    ):
                        continue
                    # Known-lethal features are hard-blocked; unverified features
                    # carry an uncertainty cost (caution, not prohibition).
                    _osc_feat = int(curr_grid[nr, nc])
                    if _osc_feat in engine.hazard_tracker.known_lethal_features:
                        continue
                    uncertainty_pen = engine._epistemic_uncertainty_cost(_osc_feat, bg)
                    visits = engine.position_visit_counts.get((nr, nc), 0)
                    recency_pen = 20.0 if (nr, nc) in list(engine.recent_positions)[-6:] else 0.0
                    total_score = visits + recency_pen + uncertainty_pen
                    if total_score < min_visits:
                        min_visits = total_score
                        best_act = act
                if best_act is not None:
                    return best_act, None
                if spatial_effector_actions and min_visits == float("inf"):
                    chosen_eff = spatial_effector_actions[0]
                    return chosen_eff, engine.ground_effector_action(curr_grid, chosen_eff)
                return available_actions[0], None

            # 3a. Update room topology and extract topological subgoals
            engine.update_room_topology(curr_grid)

            candidate_entities = [
                e
                for e in entities
                if e.id not in engine.probed_entity_ids
                and not engine.symbolic_theory.is_barrier(e.feature_id)
                and e.feature_id not in engine.hazard_tracker.known_lethal_features
                and (
                    abs(e.grid_pos[0] - engine.avatar_pos[0])
                    + abs(e.grid_pos[1] - engine.avatar_pos[1])
                )
                > 1
            ]

            if engine.active_probe_target is None and candidate_entities:
                scored: list[tuple[SpatialEntity, float]] = []
                current_room_cells = (
                    set(engine.topology_rooms.get(engine.current_room_id, []))
                    if engine.current_room_id is not None
                    else set()
                )
                for e in candidate_entities:
                    dist = abs(e.grid_pos[0] - engine.avatar_pos[0]) + abs(
                        e.grid_pos[1] - engine.avatar_pos[1]
                    )
                    visits = engine.entity_visit_counts.get(e.id, 0)
                    info_val = 10.0 / (visits + 1.0)
                    if current_room_cells and e.grid_pos in current_room_cells:
                        info_val *= 1.5  # Room-locality preference
                    cost = dist + 1.0
                    scored.append((e, info_val / cost))
                scored.sort(key=lambda x: x[1], reverse=True)
                chosen = scored[0][0]
                engine.active_probe_target = chosen.grid_pos
                engine.active_probe_id = chosen.id
                engine.probe_target_steps = 0
                engine.entity_visit_counts[chosen.id] = (
                    engine.entity_visit_counts.get(chosen.id, 0) + 1
                )

            # 3b. Room Topology Doorway Macro-Subgoal Selection:
            # If no probe target is active, and the avatar is in a partitioned room,
            # select connecting doorway leading to the least-explored adjacent chamber!
            if (
                engine.active_probe_target is None
                and engine.topology_doors
                and engine.current_room_id is not None
            ):
                connecting_doors: list[tuple[RoomDoor, int, float]] = []
                for door in engine.topology_doors:
                    if engine.current_room_id in door.connects_rooms:
                        nbr_rooms = [r for r in door.connects_rooms if r != engine.current_room_id]
                        if not nbr_rooms:
                            continue
                        nbr_room = nbr_rooms[0]
                        nbr_cells = engine.topology_rooms.get(nbr_room, [])
                        unvisited_count = sum(
                            1
                            for cell in nbr_cells
                            if engine.position_visit_counts.get(cell, 0) == 0
                        )
                        dist_to_door = abs(door.door_coord[0] - engine.avatar_pos[0]) + abs(
                            door.door_coord[1] - engine.avatar_pos[1]
                        )
                        door_score = float(unvisited_count) / (dist_to_door + 1.0)
                        connecting_doors.append((door, nbr_room, door_score))

                if connecting_doors:
                    connecting_doors.sort(key=lambda item: item[2], reverse=True)
                    best_door, target_room, score = connecting_doors[0]
                    if score > 0.1:
                        engine.active_probe_target = best_door.door_coord
                        engine.active_probe_id = (
                            f"door_{best_door.door_coord[0]}_{best_door.door_coord[1]}"
                        )
                        engine.probe_target_steps = 0
                        engine.working_memory.register_topological_doorway(
                            best_door.door_coord, target_room
                        )

            target_pos = engine.active_probe_target or (H // 2, W // 2)
            tr, tc = target_pos

            # 3c. Free Energy Active Inference Action Selection
            best_action = None
            disp_actions = [
                (act, engine.action_dynamics[act].get_displacement())
                for act in displacement_actions
            ]
            candidate_action_nodes: list[ActionNode] = []
            info_gain_map: dict[str, float] = {}
            node_to_act_map: dict[str, int] = {}

            for act, (dr_cal, dc_cal) in disp_actions:
                if (engine.avatar_pos, act) in engine.failed_transitions:
                    continue
                if engine.inhibited_actions.get(act, 0) > 0:
                    continue
                nr = engine.avatar_pos[0] + dr_cal
                nc = engine.avatar_pos[1] + dc_cal
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if (nr, nc) in engine.learned_barriers or engine.symbolic_theory.is_barrier(
                    int(curr_grid[nr, nc])
                ):
                    continue

                cell_feat = int(curr_grid[nr, nc])
                if cell_feat in engine.hazard_tracker.known_lethal_features:
                    continue

                # Epistemic information gain
                visit_count = engine.position_visit_counts.get((nr, nc), 0)
                info_gain = 1.0 / (1.0 + float(visit_count))
                if (nr, nc) == (tr, tc):
                    info_gain += 0.50

                # Risk factor (uncertainty & danger)
                uncertainty_pen = (
                    0.0
                    if (nr, nc) == (tr, tc)
                    else engine._epistemic_uncertainty_cost(cell_feat, bg)
                )
                risk_factor = min(1.0, uncertainty_pen / 40.0)

                # Estimated cost (progress to probe target + recency penalty)
                dist_to_target = abs(tr - nr) + abs(tc - nc)
                recency_penalty = 10.0 if (nr, nc) in list(engine.recent_positions)[-4:] else 0.0
                estimated_cost = float(dist_to_target) + recency_penalty

                act_id = f"act_{act}_{nr}_{nc}"
                node = ActionNode(
                    id=act_id,
                    intent=f"move_{dr_cal}_{dc_cal}",
                    risk_factor=risk_factor,
                    estimated_cost=int(round(estimated_cost)),
                )
                candidate_action_nodes.append(node)
                info_gain_map[act_id] = info_gain
                node_to_act_map[act_id] = act

            if candidate_action_nodes:
                eval_results = engine.active_inference.evaluate_candidates(
                    candidate_action_nodes,
                    information_gain_map=info_gain_map,
                )
                if eval_results:
                    best_action = node_to_act_map[eval_results[0].action.id]

            if best_action is not None:
                return best_action, None

            # Metacognitive Refractory Relaxation: if uninhibited actions were all blocked, try candidate with lowest inhibition
            if best_action is None:
                sorted_disp = sorted(
                    disp_actions, key=lambda x: engine.inhibited_actions.get(x[0], 0)
                )
                for act, (dr_cal, dc_cal) in sorted_disp:
                    if (engine.avatar_pos, act) in engine.failed_transitions:
                        continue
                    nr = engine.avatar_pos[0] + dr_cal
                    nc = engine.avatar_pos[1] + dc_cal
                    if 0 <= nr < H and 0 <= nc < W:
                        if (
                            (nr, nc) in engine.hazard_tracker.static_lethal_positions
                            or (nr, nc) in engine.learned_barriers
                            or engine.symbolic_theory.is_barrier(int(curr_grid[nr, nc]))
                        ):
                            continue
                        if (
                            int(curr_grid[nr, nc])
                            not in engine.hazard_tracker.known_lethal_features
                        ):
                            best_action = act
                            break

            if best_action is not None:
                return best_action, None

            # Frontier Backtracking: find shortest path to nearest unexhausted walkable cell / junction
            if disp_actions and engine.avatar_pos is not None:
                bfs_q: deque[tuple[tuple[int, int], list[Any]]] = deque([(engine.avatar_pos, [])])
                bfs_visited = {engine.avatar_pos}
                best_frontier_path: list[Any] | None = None
                least_visited_path: list[Any] | None = None
                least_visits = float("inf")

                while bfs_q and len(bfs_visited) < 300:
                    p, path = bfs_q.popleft()
                    p_visits = engine.position_visit_counts.get(p, 0)
                    if p != engine.avatar_pos and p_visits < least_visits:
                        least_visits = p_visits
                        least_visited_path = path

                    has_unvisited_neighbor = False
                    for act, (dr_cal, dc_cal) in disp_actions:
                        nbr = (p[0] + dr_cal, p[1] + dc_cal)
                        if 0 <= nbr[0] < H and 0 <= nbr[1] < W:
                            if (
                                nbr not in engine.learned_barriers
                                and not engine.symbolic_theory.is_barrier(
                                    int(curr_grid[nbr[0], nbr[1]])
                                )
                            ):
                                if (
                                    int(curr_grid[nbr[0], nbr[1]])
                                    not in engine.hazard_tracker.known_lethal_features
                                ):
                                    if engine.position_visit_counts.get(nbr, 0) == 0:
                                        has_unvisited_neighbor = True
                                        break
                    if has_unvisited_neighbor and path:
                        best_frontier_path = path
                        break

                    for act, (dr_cal, dc_cal) in disp_actions:
                        nr = p[0] + dr_cal
                        nc = p[1] + dc_cal
                        if (nr, nc) in bfs_visited or not (0 <= nr < H and 0 <= nc < W):
                            continue
                        if (nr, nc) in engine.learned_barriers or engine.symbolic_theory.is_barrier(
                            int(curr_grid[nr, nc])
                        ):
                            continue
                        if int(curr_grid[nr, nc]) in engine.hazard_tracker.known_lethal_features:
                            continue
                        if p == engine.avatar_pos and (p, act) in engine.failed_transitions:
                            continue
                        bfs_visited.add((nr, nc))
                        bfs_q.append(((nr, nc), path + [act]))

                chosen_escape = best_frontier_path or least_visited_path
                if chosen_escape:
                    return chosen_escape[0], None

            if spatial_effector_actions:
                chosen_eff = spatial_effector_actions[0]
                return chosen_eff, engine.ground_effector_action(curr_grid, chosen_eff)
            elif discrete_transform_actions:
                return discrete_transform_actions[
                    engine.step_counter % len(discrete_transform_actions)
                ], None
            uncalibrated = [a for a in available_actions if not engine.is_displacement_action(a)]
            untested_here = [
                a
                for a in uncalibrated
                if (a, engine.avatar_pos) not in engine.tested_action_positions
            ]
            if untested_here:
                act = untested_here[0]
                engine.tested_action_positions.add((act, engine.avatar_pos))
                return act, None
            return available_actions[engine.step_counter % len(available_actions)], None

        # 4. Fallback probe
        if spatial_effector_actions:
            chosen_eff = spatial_effector_actions[0]
            return chosen_eff, engine.ground_effector_action(curr_grid, chosen_eff)
        return available_actions[0], None


# ─────────────────────────────────────────────────────────────────────────────
# 7. Master Autonomous Epistemic Engine Orchestrator
# ─────────────────────────────────────────────────────────────────────────────


class AutonomousEpistemicEngine:
    """Domain-agnostic cognitive engine for autonomous exploration, hypothesis testing,
    and forward mental simulation.
    """

    MIN_PROBES_PER_ACTION: int = 3

    def __init__(
        self,
        exploration_budget: int = 150,
        enable_logging: bool = True,
        instructions: Sequence[str] | str | None = None,
    ) -> None:
        self.exploration_budget = exploration_budget
        self.enable_logging = enable_logging

        # Cognitive state
        self.phase: EpistemicPhase = EpistemicPhase.MOTOR_GROUNDING
        self.step_counter: int = 0

        # Memory of frames & actions
        self.prev_grid: np.ndarray | None = None
        self.last_action: int | None = None
        self.last_action_data: dict[str, Any] | None = None
        self.consecutive_quiescent_actions: int = 0

        # Avatar self-model (Motor Grounding)
        self._avatar_feature: int | None = None
        self.avatar_features: set[int] = set()
        self.avatar_pos: tuple[int, int] | None = None
        self.avatar_size: int = 1

        # Controllability-based avatar identification: accumulates evidence
        # across multiple frames/actions to distinguish the avatar (responds
        # to player input) from other moving entities (autonomous movement).
        # Maps feature_id -> list of (action_id, observed_delta) observations.
        self._avatar_controllability_evidence: dict[int, list[tuple[int, tuple[int, int]]]] = {}

        # Experiential verification: features the avatar has successfully
        # traversed without dying. Replaces hardcoded pixel-count thresholds
        # with empirical "I walked on this and survived" knowledge.
        self.verified_safe_features: set[int] = set()

        # Intensional Neuro-Symbolic World Theory
        self.symbolic_theory: HCIRSymbolicWorldTheory = HCIRSymbolicWorldTheory()

        # Motor calibration & dynamics
        self.action_dynamics: dict[int, ActionDynamicsModel] = {}
        self.tested_actions: set[int] = set()
        self.tested_action_positions: set[tuple[int, tuple[int, int] | None]] = set()

        # State Mutation Models (Switches, buttons, door toggles)
        self.state_mutations: list[StateMutationModel] = []

        # Episodic Ground Facts (Layout coordinates, preserved per-level across attempts)
        self._current_level: int = 0
        self.level_failed_transitions: dict[int, set[tuple[tuple[int, int], int]]] = {}
        self.level_learned_barriers: dict[int, set[tuple[int, int]]] = {}
        self.level_lethal_positions: dict[int, set[tuple[int, int]]] = {}
        self.learned_barriers: set[tuple[int, int]] = self.level_learned_barriers.setdefault(
            0, set()
        )
        self.learned_goal_positions: set[tuple[int, int]] = set()
        self.learned_receptacle_positions: set[tuple[int, int]] = set()

        # Mental simulation plan
        self.mental_plan: deque[MentalSimulationStep] = deque()
        self.active_hypothesis: CausalHypothesis | None = None
        self.failed_transitions: set[tuple[tuple[int, int], int]] = (
            self.level_failed_transitions.setdefault(0, set())
        )

        # Theory of Mind: creature kinds observed to patrol autonomously
        # (feature-level concept, retained across levels like a human would).
        self.mobile_threat_features: set[int] = set()
        self._prev_oriented_threats: list[OrientedThreat] = []
        self.consecutive_plan_failures: int = 0
        self.consecutive_stuck_steps: int = 0
        self.exploration_cooldown: int = 0
        self.current_simulated_goal: tuple[int, int] | None = None
        self.last_predicted_pos: tuple[int, int] | None = None

        # Prefrontal working memory & biological hazard tracker
        self.working_memory: PrefrontalWorkingMemory = PrefrontalWorkingMemory()
        if instructions:
            self.load_instructions(instructions)
        self.hazard_tracker: SpatiotemporalHazardTracker = SpatiotemporalHazardTracker()
        self.saccadic_attention: SaccadicAttentionSystem = SaccadicAttentionSystem()
        self.physics_engine: IntuitivePhysicsEngine = IntuitivePhysicsEngine()

        # Room topology & doorway subgoal reasoning (spatial containment)
        self.room_topology: RoomTopologyExtractor = RoomTopologyExtractor()

        # Object-Centric Macro-Action State Graph Planner (5 Executive Directives)
        self.object_planner: ObjectStateGraphPlanner = ObjectStateGraphPlanner()
        self.topology_rooms: dict[int, list[tuple[int, int]]] = {}
        self.topology_doors: list[RoomDoor] = []
        self.room_adjacency: dict[int, list[int]] = {}
        self.current_room_id: int | None = None

        # Confidence-scaled predictive coding surprise engine
        self.surprise_engine: SurpriseEngine = SurpriseEngine(surprise_threshold=0.15)
        self.last_surprise: float = 0.0
        self.last_surprise_eval: SurpriseEvaluation | None = None

        # Free energy active inference decision selection
        self.active_inference: ActiveInferenceEngine = ActiveInferenceEngine(
            w_reward=0.35,
            w_info_gain=0.30,
            w_future_val=0.15,
            w_risk=0.10,
            w_cost=0.10,
        )

        # Active exploration & curiosity state
        self.active_probe_target: tuple[int, int] | None = None
        self.active_probe_id: str | None = None
        self.probe_target_steps: int = 0
        self.probed_entity_ids: set[str] = set()
        self.entity_visit_counts: dict[str, int] = {}
        self.position_visit_counts: dict[tuple[int, int], int] = {}
        self.recent_positions: deque[tuple[int, int]] = deque(maxlen=16)
        self.exhausted_candidate_goals: set[tuple[int, int]] = set()
        self.quiescent_click_targets: set[tuple[int, int]] = set()
        self.effective_click_targets: set[tuple[int, int]] = set()
        self.last_effective_click_coord: tuple[int, int] | None = None
        self.consecutive_effective_clicks: int = 0
        self.click_affordances: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self.last_effector_target_step: dict[tuple[int, int], int] = {}
        self.active_goal_converging_coord: tuple[int, int] | None = None
        self.consecutive_goal_converging_clicks: int = 0

        # Feature-level interventional causal falsification (Piagetian Stage D3)
        self.quiescent_features: set[int] = set()
        self.effective_features: set[int] = set()

        # Metacognitive Refractory Action Inhibition (Stage D13)
        self.inhibited_actions: dict[Any, int] = {}
        self.prior_avatar_feature: int | None = None

        # Universal Dynamic Action Affordance Registry
        self.action_affordances: dict[Any, ActionAffordance] = {}

        self.bg_feature: int = 0
        self.level_epistemic_probes: int = 0
        self.total_epistemic_probes: int = 0
        self._feedback_assimilated: bool = False

        self.hypotheses: list[CausalHypothesis] = []
        self.belief_history: list[BeliefTransitionEvent] = []

    def is_action_sufficiently_probed(self, action_id: Any) -> bool:
        """Check if an action has been tested enough times to have a reliable dynamics model."""
        dyn = self.action_dynamics.get(action_id)
        if dyn is None:
            return False
        probes = getattr(dyn, "probes_tested", 0)
        return probes >= self.MIN_PROBES_PER_ACTION

    def register_action_space(self, action_specs: Sequence[Any]) -> None:
        """Register external action definitions dynamically from peripheral driver or device descriptor.

        Accepts:
        - List of dicts (from SynapticDeviceDescriptor.action_schema):
          e.g. [{'action_id': 1, 'name': 'MOVE_UP'}, {'action_id': 6, 'name': 'CLICK_CELL', 'parameters': {'x': 'int', 'y': 'int'}}]
        - List of ActionAffordance instances
        - List of bare action IDs (int, str, Enum)
        """
        for spec in action_specs:
            aff = self._parse_action_spec(spec)
            self.action_affordances[aff.action_id] = aff

    def _parse_action_spec(self, spec: Any) -> ActionAffordance:
        if isinstance(spec, ActionAffordance):
            return spec
        if isinstance(spec, dict):
            aid = spec.get("action_id", spec.get("id", spec.get("name", 0)))
            name = str(spec.get("name", aid))
            params = spec.get("parameters") or {}
            param_keys = tuple(params.keys()) if isinstance(params, dict) else ("x", "y")
            req_spatial = bool(
                isinstance(params, dict)
                and any(
                    k in params
                    for k in (
                        "x",
                        "y",
                        "col",
                        "row",
                        "lat",
                        "lon",
                        "azimuth",
                        "elevation",
                        "distance",
                        "c",
                        "r",
                    )
                )
            )
            return ActionAffordance(
                action_id=aid,
                name=name,
                requires_spatial_target=req_spatial,
                target_param_keys=param_keys if param_keys else ("x", "y"),
            )
        return ActionAffordance(action_id=spec, name=str(spec))

    def is_spatial_effector(self, action: Any) -> bool:
        """Returns True if the action is configured or observed to accept spatial coordinates."""
        aff = self.action_affordances.get(action)
        if aff is not None:
            return aff.requires_spatial_target
        return False

    def is_displacement_action(self, action: Any) -> bool:
        """Returns True if this action causes physical self/ego-motion displacement."""
        dyn = self.action_dynamics.get(action)
        if dyn is not None and dyn.is_displacement_action():
            return True
        aff = self.action_affordances.get(action)
        if aff is not None and aff.is_displacement:
            return True
        return False

    def is_action_calibrated(self, action: Any) -> bool:
        """Returns True if empirical dynamics for this action have been observed."""
        return action in self.action_dynamics

    @staticmethod
    def normalize_sensory_input(raw: Any) -> np.ndarray:
        """Normalize arbitrary sensory observations (camera, depth, lidar, 2D grid) into a spatial 2D array.

        Supports:
        - 2D integer grids (e.g. ARC, Roguelike, Atari RAM/grid)
        - 2D float arrays (e.g. Depth maps, LiDAR range grids) -> quantized to perceptual levels
        - 3D arrays (e.g. RGB camera (H, W, 3), LiDAR multi-channel, or temporal stack (T, H, W))
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

    # ── Property Pass-Throughs to HCIRSymbolicWorldTheory ─────────────────────

    @property
    def avatar_feature(self) -> int | None:
        if self._avatar_feature is not None:
            return self._avatar_feature
        if self.avatar_features:
            return next(iter(self.avatar_features))
        return None

    @avatar_feature.setter
    def avatar_feature(self, val: int | None) -> None:
        self._avatar_feature = val
        if val is not None:
            if not self.avatar_features or val not in self.avatar_features:
                self.avatar_features = {val}
        else:
            self.avatar_features.clear()

    @property
    def learned_goal_features(self) -> set[int]:
        return self.symbolic_theory.goal_features

    @learned_goal_features.setter
    def learned_goal_features(self, val: set[int]) -> None:
        self.symbolic_theory.goal_features = val

    @property
    def learned_barrier_features(self) -> set[int]:
        return self.symbolic_theory.barrier_features

    @learned_barrier_features.setter
    def learned_barrier_features(self, val: set[int]) -> None:
        self.symbolic_theory.barrier_features = val

    @property
    def learned_walkable_features(self) -> set[int]:
        return self.symbolic_theory.walkable_features

    @learned_walkable_features.setter
    def learned_walkable_features(self, val: set[int]) -> None:
        self.symbolic_theory.walkable_features = val

    @property
    def learned_cargo_features(self) -> set[int]:
        return self.symbolic_theory.cargo_features

    @learned_cargo_features.setter
    def learned_cargo_features(self, val: set[int]) -> None:
        self.symbolic_theory.cargo_features = val

    @property
    def learned_receptacle_features(self) -> set[int]:
        return self.symbolic_theory.receptacle_features

    @learned_receptacle_features.setter
    def learned_receptacle_features(self, val: set[int]) -> None:
        self.symbolic_theory.receptacle_features = val

    # ── Epistemic status of feature values ──────────────────────────────────────
    UNVERIFIED_FEATURE_COST: float = 8.0

    def is_feature_unverified(self, feat: int, bg: int) -> bool:
        """True if the agent has neither experience nor theory about ``feat``.

        A feature is 'known' if it is background, part of the avatar, walked on
        and survived, or classified by the symbolic theory (walkable, barrier,
        goal, candidate goal, cargo). Anything else is epistemically unknown.
        """
        if feat == bg or feat in self.verified_safe_features:
            return False
        av = self.avatar_features or (
            {self.avatar_feature} if self.avatar_feature is not None else set()
        )
        if feat in av:
            return False
        th = self.symbolic_theory
        if th.is_walkable(feat) or th.is_barrier(feat):
            return False
        if (
            feat in th.goal_features
            or feat in th.candidate_goal_features
            or feat in th.cargo_features
        ):
            return False
        return True

    def _epistemic_uncertainty_cost(self, feat: int, bg: int) -> float:
        """Soft exploration cost for stepping onto an unverified feature."""
        return self.UNVERIFIED_FEATURE_COST if self.is_feature_unverified(feat, bg) else 0.0

    def load_instructions(self, instructions: Sequence[str] | str | None) -> None:
        """Load executive cognitive directives into working memory to guide epistemic policies."""
        self.working_memory.load_instructions(instructions)

    @property
    def current_level(self) -> int:
        return self._current_level

    @current_level.setter
    def current_level(self, level: int) -> None:
        self.set_current_level(level)

    def set_current_level(self, level: int) -> None:
        if self._current_level != level:
            self._current_level = level
            self.failed_transitions = self.level_failed_transitions.setdefault(level, set())
            self.learned_barriers = self.level_learned_barriers.setdefault(level, set())
            self.hazard_tracker.static_lethal_positions = self.level_lethal_positions.setdefault(
                level, set()
            )

    # ── Episodic State Management ─────────────────────────────────────────────

    def reset_episode(
        self,
        retain_dynamics: bool = True,
        is_new_level: bool = False,
        level: int | None = None,
    ) -> None:
        """Reset episodic state upon level transition or death."""
        if level is not None:
            self.set_current_level(level)
        self.prev_grid = None
        self._prev_oriented_threats = []
        self.avatar_pos = None
        self.last_action = None
        self.last_action_data = None
        self.last_predicted_pos = None
        self.consecutive_quiescent_actions = 0
        self.consecutive_stuck_steps = 0
        self.last_effective_click_coord = None
        self.consecutive_effective_clicks = 0
        self.mental_plan.clear()
        self.active_hypothesis = None
        self.active_probe_target = None
        self.active_probe_id = None
        self.probe_target_steps = 0
        self.recent_positions.clear()
        self.level_epistemic_probes = 0
        self.working_memory.reset_episode(retain_long_term=retain_dynamics)
        self.consecutive_plan_failures = 0
        self.exploration_cooldown = 0
        self.current_simulated_goal = None
        self._feedback_assimilated = False

        if not retain_dynamics:
            self.level_failed_transitions.clear()
            self.level_learned_barriers.clear()
            self.level_lethal_positions.clear()
            self._current_level = 0 if level is None else level
            self.failed_transitions = self.level_failed_transitions.setdefault(
                self._current_level, set()
            )
            self.learned_barriers = self.level_learned_barriers.setdefault(
                self._current_level, set()
            )
            self.hazard_tracker.clear_lethal_positions()
            self.hazard_tracker.reset_episode()
            self.physics_engine.reset_episode()
            self.exhausted_candidate_goals.clear()
            self.tested_action_positions.clear()
            self.learned_goal_positions.clear()
            self.learned_receptacle_positions.clear()
            self.topology_rooms.clear()
            self.topology_doors.clear()
            self.room_adjacency.clear()
            self.current_room_id = None
            self.surprise_engine.reset_ledger()
            self.last_surprise = 0.0
            self.last_surprise_eval = None
        elif is_new_level:
            # Level transition: point episodic spatial memory to the target level
            self.failed_transitions = self.level_failed_transitions.setdefault(
                self._current_level, set()
            )
            self.learned_barriers = self.level_learned_barriers.setdefault(
                self._current_level, set()
            )
            self.hazard_tracker.static_lethal_positions = self.level_lethal_positions.setdefault(
                self._current_level, set()
            )
            self.hazard_tracker.reset_episode()
            self.physics_engine.reset_episode()
            self.exhausted_candidate_goals.clear()
            self.tested_action_positions.clear()
            self.learned_goal_positions.clear()
            self.learned_receptacle_positions.clear()
            self.topology_rooms.clear()
            self.topology_doors.clear()
            self.room_adjacency.clear()
            self.current_room_id = None
            self.surprise_engine.reset_ledger()
            self.last_surprise = 0.0
            self.last_surprise_eval = None
        else:
            # Same level retry on death
            self.hazard_tracker.grid_history.clear()
            self.hazard_tracker.step_history.clear()
            self.tested_action_positions.clear()
            self.last_surprise = 0.0
            self.last_surprise_eval = None

        if is_new_level and retain_dynamics:
            # ── Principled Hypothesis Management on Level Transition ──────────
            #
            # The human brain carries forward hypotheses and VERIFIES them through experience.
            # 1. Avatar identity is retained as a prior hypothesis: in almost all multi-level games,
            #    the player character maintains their visual identity across level stages.
            #    We record prior_avatar_feature, keep avatar_feature as an active prior,
            #    and clear controllability evidence so discrepancy can trigger re-grounding if needed.
            if self.avatar_feature is not None:
                self.prior_avatar_feature = self.avatar_feature
            self._avatar_controllability_evidence.clear()

            # 2. Feature-value semantics are carried forward as PRIORS, not
            #    facts. People assume "same appearance → same meaning" in a new
            #    level until experience contradicts it. Contradictions are
            #    already handled online: colliding marks a barrier, dying marks
            #    a feature lethal, surviving a traversal marks it safe. Wiping
            #    priors would discard e.g. which feature is the goal.
            #    (symbolic_theory, verified_safe_features and known lethal
            #    features are intentionally retained.)
            #
            #    Action-conditioned state mutations are layout-specific
            #    (a particular switch at a particular place) and are cleared.
            self.state_mutations.clear()

            # 3. Spatial/episodic state is level-specific
            self.entity_visit_counts.clear()
            self.probed_entity_ids.clear()
            self.quiescent_click_targets.clear()
            self.effective_click_targets.clear()
            self.click_affordances.clear()
            self.quiescent_features.clear()
            self.last_effector_target_step.clear()
            self.active_goal_converging_coord = None
            self.consecutive_goal_converging_clicks = 0
            # self.effective_features is retained as semantic affordance prior across levels
            self.consecutive_quiescent_actions = 0
            self.inhibited_actions.clear()

            # 4. Action dynamics and affordances are RETAINED — they represent
            #    structural motor knowledge (e.g., "action 1 moves up by 6 pixels")
            #    that is typically level-invariant.
            self.phase = (
                EpistemicPhase.MENTAL_SIMULATION
                if self.is_motor_grounded()
                else EpistemicPhase.MOTOR_GROUNDING
            )

        elif not retain_dynamics:
            # Full Cross-Game Isolation: Zero cross-game leakage
            self.total_epistemic_probes = 0
            self.avatar_feature = None
            self.avatar_features.clear()
            self.prior_avatar_feature = None
            self._avatar_controllability_evidence.clear()
            self.verified_safe_features.clear()
            self.avatar_pos = None
            self.action_dynamics.clear()
            self.action_affordances.clear()
            self.tested_actions.clear()
            self.symbolic_theory.clear()
            self.state_mutations.clear()
            self.entity_visit_counts.clear()
            self.position_visit_counts.clear()
            self.probed_entity_ids.clear()
            self.quiescent_click_targets.clear()
            self.effective_click_targets.clear()
            self.click_affordances.clear()
            self.quiescent_features.clear()
            self.effective_features.clear()
            self.consecutive_quiescent_actions = 0
            self.inhibited_actions.clear()
            self.topology_rooms.clear()
            self.topology_doors.clear()
            self.room_adjacency.clear()
            self.current_room_id = None
            self.surprise_engine.reset_ledger()
            self.last_surprise = 0.0
            self.last_surprise_eval = None
            self.hazard_tracker.known_lethal_features.clear()
            self.hazard_tracker.periodic_cells.clear()
            self.mobile_threat_features.clear()
            self.phase = EpistemicPhase.MOTOR_GROUNDING
        else:
            self.phase = (
                EpistemicPhase.MENTAL_SIMULATION
                if self.is_motor_grounded()
                else EpistemicPhase.MOTOR_GROUNDING
            )

        if hasattr(self, "object_planner"):
            self.object_planner.reset_episode(is_new_level=is_new_level)

    def is_motor_grounded(self) -> bool:
        """True if the agent has identified its avatar and calibrated directional actions."""
        return (self.avatar_feature is not None or bool(self.avatar_features)) and any(
            dyn.confidence >= 0.7 for dyn in self.action_dynamics.values()
        )

    def compute_exploration_budget(self, n_actions: int) -> int:
        """Adaptive budget: more probing when more unknowns exist."""
        if not self.is_motor_grounded():
            calibrated = sum(1 for d in self.action_dynamics.values() if d.confidence >= 0.7)
            uncalibrated = max(0, n_actions - calibrated)
            return max(
                self.exploration_budget,
                uncalibrated * self.MIN_PROBES_PER_ACTION * 3,
            )
        confirmed_count = (
            len(self.symbolic_theory.goal_features)
            + len(self.symbolic_theory.barrier_features)
            + len(self.state_mutations)
        )
        scaling = max(0.2, 1.0 - 0.15 * confirmed_count)
        return max(15, int(self.exploration_budget * scaling))

    # ── Delegated Public Interfaces ───────────────────────────────────────────

    def estimate_background(self, grid: np.ndarray) -> int:
        return PerceptionEngine.estimate_background(
            grid,
            barrier_features=self.symbolic_theory.barrier_features,
            avatar_features=self.avatar_features,
            bg_feature=self.bg_feature,
        )

    def extract_entities(self, grid: np.ndarray, bg: int) -> list[SpatialEntity]:
        return PerceptionEngine.extract_entities(
            grid,
            bg,
            symbolic_theory=self.symbolic_theory,
            avatar_features=self.avatar_features,
            state_mutations=self.state_mutations,
            learned_barriers=self.learned_barriers,
            learned_goal_positions=self.learned_goal_positions,
            learned_receptacle_positions=self.learned_receptacle_positions,
        )

    def compute_frame_diff(
        self, prev_grid: np.ndarray, curr_grid: np.ndarray
    ) -> EpistemicObservationDiff:
        return PerceptionEngine.compute_frame_diff(prev_grid, curr_grid)

    def detect_structural_goals(
        self, grid: np.ndarray, bg: int | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        bg_val = bg if bg is not None else self.bg_feature
        return PerceptionEngine.detect_structural_goals(
            grid,
            bg=bg_val,
            avatar_features=self.avatar_features,
            avatar_feature=self.avatar_feature,
            barrier_features=self.symbolic_theory.barrier_features,
            known_lethal_features=self.hazard_tracker.known_lethal_features,
            **kwargs,
        )

    def assimilate_feedback(
        self,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
        is_win: bool = False,
        is_lost: bool = False,
        action: int | None = None,
        action_data: dict[str, Any] | None = None,
    ) -> None:
        EpistemicFeedbackAssimilator.assimilate(
            self,
            curr_grid,
            available_actions,
            is_win=is_win,
            is_lost=is_lost,
            action=action,
            action_data=action_data,
        )

    def simulate_in_mind(
        self,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
        target_role: EntityRole = EntityRole.GOAL,
        blocked_cells: set[tuple[int, int]] | None = None,
    ) -> list[MentalSimulationStep] | None:
        return MentalSimulationPlanner.simulate_in_mind(
            self,
            curr_grid,
            available_actions,
            target_role=target_role,
            blocked_cells=blocked_cells,
        )

    def plan_epistemic_probe(
        self,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
    ) -> tuple[int, dict[str, Any] | None]:
        return EpistemicCuriosityExplorer.plan_epistemic_probe(self, curr_grid, available_actions)

    def update_room_topology(self, curr_grid: np.ndarray) -> None:
        """Extract rooms and doorways from current occupancy grid knowledge."""
        H, W = curr_grid.shape
        occupancy = np.ones((H, W), dtype=bool)
        for r in range(H):
            for c in range(W):
                val = int(curr_grid[r, c])
                if (
                    (r, c) in self.learned_barriers
                    or self.symbolic_theory.is_barrier(val)
                    or val in self.hazard_tracker.known_lethal_features
                ):
                    occupancy[r, c] = False

        rooms, doors = self.room_topology.extract_rooms_and_doors(occupancy, min_room_size=4)
        self.topology_rooms = rooms
        self.topology_doors = doors
        self.room_adjacency = self.room_topology.build_adjacency_graph(rooms, doors)

        # Localize current room
        self.current_room_id = None
        if self.avatar_pos is not None:
            for rid, coords in rooms.items():
                if self.avatar_pos in coords:
                    self.current_room_id = rid
                    break

    def _update_avatar_position_from_grid(
        self, curr_grid: np.ndarray, known_av_feats: set[int]
    ) -> None:
        """Visually localize the avatar position on the current grid."""
        H, W = curr_grid.shape
        from scipy.ndimage import label

        mask = np.isin(curr_grid, list(known_av_feats))
        if not np.any(mask):
            return

        labeled, num_features = label(mask)
        if num_features == 0:
            return

        # If avatar was already precisely localized by assimilate_feedback via motion tracking,
        # and current avatar_pos contains an avatar feature and matches avatar_size, verify and keep it
        if self.avatar_pos is not None:
            r, c = self.avatar_pos
            if 0 <= r < H and 0 <= c < W and mask[r, c]:
                comp_lbl = labeled[r, c]
                if comp_lbl > 0:
                    comp_area = int(np.sum(labeled == comp_lbl))
                    if self.avatar_size == 0 or (
                        0.25 * self.avatar_size <= comp_area <= 2.5 * self.avatar_size
                    ):
                        return

        best_pos = None
        min_score = float("inf")

        for lbl in range(1, num_features + 1):
            coords = np.argwhere(labeled == lbl)
            area = len(coords)
            if 1 <= area <= 49:
                centroid = (int(np.mean(coords[:, 0])), int(np.mean(coords[:, 1])))
                if self.avatar_pos is not None:
                    dist = abs(centroid[0] - self.avatar_pos[0]) + abs(
                        centroid[1] - self.avatar_pos[1]
                    )
                    size_diff = abs(area - self.avatar_size) if self.avatar_size > 0 else 0
                    score = dist + size_diff * 0.5
                else:
                    # Fresh localization: choose component best matching known avatar_size
                    score = abs(area - self.avatar_size) if self.avatar_size > 0 else 0
                if score < min_score:
                    min_score = score
                    best_pos = centroid

        if best_pos is not None:
            self.avatar_pos = best_pos

    def ground_effector_action(self, curr_grid: np.ndarray, action: Any = None) -> dict[str, Any]:
        """Spatially ground an allocentric effector command onto salient affordances.

        Dynamically maps targeting parameters based on the action's declared affordance
        (e.g. 'x', 'y' for cell/pixel clicks, or 'azimuth', 'elevation' for directional sensors).
        """
        curr_grid = self.normalize_sensory_input(curr_grid)
        H, W = curr_grid.shape
        aff = self.action_affordances.get(action)
        param_keys = aff.target_param_keys if aff else ("x", "y")

        bg = self.estimate_background(curr_grid)
        entities = self.extract_entities(curr_grid, bg)
        av_feats = self.avatar_features or (
            {self.avatar_feature} if self.avatar_feature is not None else set()
        )

        click_candidates: list[tuple[int, int, float]] = []

        # dlPFC Constraint Propagation: Target guaranteed safe cells
        deduced_safe = self.working_memory.get_unrevealed_safe_cells()
        for sr, sc in deduced_safe:
            if (sr, sc) not in self.quiescent_click_targets and 0 <= sr < H and 0 <= sc < W:
                visit_count = self.entity_visit_counts.get(f"click_{sr}_{sc}", 0)
                if visit_count == 0:
                    cand_score = 300.0  # Top priority! Guaranteed safe progress!
                    click_candidates.append((sr, sc, cand_score))

        # Visual Symmetry: Target asymmetric completion coordinates
        structural_goals = PerceptionEngine.detect_structural_goals(
            curr_grid,
            bg=bg,
            avatar_features=av_feats,
            avatar_feature=self.avatar_feature,
            barrier_features=self.symbolic_theory.barrier_features,
            known_lethal_features=self.hazard_tracker.known_lethal_features,
        )
        sym_goals = [
            g
            for g in structural_goals
            if g.get("type") == "symmetry_completion" and g.get("position") is not None
        ]
        for sg in sym_goals:
            sp = sg["position"]
            if sp not in self.quiescent_click_targets and 0 <= sp[0] < H and 0 <= sp[1] < W:
                visit_count = self.entity_visit_counts.get(f"click_{sp[0]}_{sp[1]}", 0)
                if visit_count < 3:
                    cand_score = (
                        250.0 + (sg.get("confidence", 0.8) * 50.0) - float(visit_count) * 20.0
                    )
                    click_candidates.append((sp[0], sp[1], cand_score))

        # Gestalt Affordance Panels: Prioritize regular interactive arrays & pop-out targets
        panels = PerceptionEngine.detect_affordance_panels(entities, curr_grid, bg=bg)
        for panel in panels:
            panel_coords = set(panel["item_coords"])
            has_effective_item = any(p in self.effective_click_targets for p in panel_coords)

            # Minority pop-out items are top priority targets (+220.0)
            for m_item in panel["minority_items"]:
                mr, mc = m_item.grid_pos
                if (mr, mc) not in self.quiescent_click_targets and 0 <= mr < H and 0 <= mc < W:
                    visit_count = self.entity_visit_counts.get(f"click_{mr}_{mc}", 0)
                    last_step = getattr(self, "last_effector_target_step", {}).get((mr, mc), -999)
                    delta_t = max(1, getattr(self, "step_counter", 0) - last_step)
                    refractory = 35.0 if delta_t == 1 else (35.0 / float(delta_t))
                    visit_damping = min(30.0, float(visit_count) * 4.0)
                    score = 220.0 - refractory - visit_damping
                    click_candidates.append((mr, mc, score))

            # Other panel items (+120.0 or +160.0 if confirmed effective)
            for item in panel["items"]:
                ir, ic = item.grid_pos
                if (ir, ic) not in self.quiescent_click_targets and 0 <= ir < H and 0 <= ic < W:
                    visit_count = self.entity_visit_counts.get(f"click_{ir}_{ic}", 0)
                    bonus = 160.0 if has_effective_item else 120.0
                    feat_bonus = 40.0 if item.feature_id in self.effective_features else 0.0
                    last_step = getattr(self, "last_effector_target_step", {}).get((ir, ic), -999)
                    delta_t = max(1, getattr(self, "step_counter", 0) - last_step)
                    is_goal_converging = (ir, ic) == getattr(
                        self, "active_goal_converging_coord", None
                    ) and getattr(self, "consecutive_goal_converging_clicks", 0) < 12
                    is_active_momentum = (ir, ic) == getattr(
                        self, "last_effective_click_coord", None
                    ) and getattr(self, "consecutive_effective_clicks", 0) < 6
                    if is_goal_converging:
                        momentum_bonus = 120.0
                        ior_penalty = 0.0
                    elif is_active_momentum:
                        momentum_bonus = 60.0
                        ior_penalty = 0.0
                    else:
                        momentum_bonus = 0.0
                        refractory = 35.0 if delta_t == 1 else (35.0 / float(delta_t))
                        visit_damping = min(30.0, float(visit_count) * 4.0)
                        ior_penalty = refractory + visit_damping
                    score = bonus + feat_bonus + momentum_bonus - ior_penalty
                    click_candidates.append((ir, ic, score))

        # Control Panel Primacy: If viable candidates exist in affordance panels,
        # focus execution strictly within the control interface! Do NOT dilute with background/walls!
        if not click_candidates:
            for e in entities:
                if e.feature_id == bg or e.feature_id in av_feats or e.area > 120:
                    continue
                cr, cc = e.grid_pos
                if (cr, cc) in self.quiescent_click_targets:
                    continue
                visit_count = self.entity_visit_counts.get(f"click_{cr}_{cc}", 0)
                saliency = 100.0 / math.log2(2 + max(1, e.area))
                affordance_bonus = 60.0 if (cr, cc) in self.effective_click_targets else 0.0
                feat_bias = 0.0
                if e.feature_id in self.quiescent_features:
                    feat_bias -= 80.0
                elif e.feature_id in self.effective_features:
                    feat_bias += 40.0
                ior_penalty = float(visit_count) * 30.0 + (float(visit_count) ** 2) * 15.0
                cand_score = saliency + affordance_bonus + feat_bias - ior_penalty
                click_candidates.append((cr, cc, cand_score))

        if not click_candidates:
            fixations = self.saccadic_attention.extract_fixations(
                grid=curr_grid,
                prev_grid=self.prev_grid,
                background_feature=bg,
                top_k=24,
            )
            for f in fixations:
                cr, cc = f.r, f.c
                feat = int(curr_grid[cr, cc])
                if (cr, cc) in self.quiescent_click_targets or feat == bg:
                    continue
                visit_count = self.entity_visit_counts.get(f"click_{cr}_{cc}", 0)
                affordance_bonus = 60.0 if (cr, cc) in self.effective_click_targets else 0.0
                feat_bias = 0.0
                if feat in self.quiescent_features:
                    feat_bias -= 80.0
                elif feat in self.effective_features:
                    feat_bias += 40.0
                cand_score = (
                    f.salience
                    + affordance_bonus
                    + feat_bias
                    - float(visit_count) * (0.1 if affordance_bonus > 0 else 0.35)
                )
                click_candidates.append((cr, cc, cand_score))

        # Fallback: scan any unvisited non-background cells not in quiescent targets
        if not click_candidates:
            non_bg = np.argwhere(curr_grid != bg)
            for r, c in non_bg:
                cr, cc = int(r), int(c)
                feat = int(curr_grid[cr, cc])
                if (cr, cc) in self.quiescent_click_targets or (cr, cc) in av_feats:
                    continue
                visit_count = self.entity_visit_counts.get(f"click_{cr}_{cc}", 0)
                feat_bias = 0.0
                if feat in self.quiescent_features:
                    feat_bias -= 80.0
                elif feat in self.effective_features:
                    feat_bias += 40.0
                cand_score = 20.0 + feat_bias - float(visit_count) * 2.0
                click_candidates.append((cr, cc, cand_score))

        # Fallback: re-probe confirmed effective targets
        if not click_candidates and self.effective_click_targets:
            eff_sorted = sorted(
                self.effective_click_targets,
                key=lambda p: self.entity_visit_counts.get(f"click_{p[0]}_{p[1]}", 0),
            )
            click_candidates.append((eff_sorted[0][0], eff_sorted[0][1], 50.0))

        if click_candidates:
            click_candidates.sort(key=lambda x: x[2], reverse=True)
            best_r, best_c, _ = click_candidates[0]
            self.entity_visit_counts[f"click_{best_r}_{best_c}"] = (
                self.entity_visit_counts.get(f"click_{best_r}_{best_c}", 0) + 1
            )
            target_r, target_c = int(best_r), int(best_c)
            if not hasattr(self, "last_effector_target_step"):
                self.last_effector_target_step = {}
            self.last_effector_target_step[(target_r, target_c)] = getattr(self, "step_counter", 0)
        else:
            # Fallback when all known candidates are quiescent: find any unprobed coordinate
            unprobed = [
                (r, c)
                for r in range(H)
                for c in range(W)
                if (r, c) not in self.quiescent_click_targets
            ]
            if unprobed:
                target_r, target_c = unprobed[0]
            else:
                self.quiescent_click_targets.clear()
                target_r, target_c = H // 2, W // 2

        coords: dict[str, Any] = {}
        for k in param_keys:
            if k in ("x", "col", "c", "column", "azimuth"):
                coords[k] = target_c
            elif k in ("y", "row", "r", "elevation", "distance"):
                coords[k] = target_r
            else:
                coords[k] = 0
        return coords

    # ── Theory of Mind: Creature Motion Observation ──────────────────────────

    def infer_motor_step_size(self, available_actions: Sequence[Any]) -> int:
        """Largest calibrated displacement quantum (the world's 'cell' size)."""
        step = 1
        for act in available_actions:
            dyn = self.action_dynamics.get(act)
            if dyn is not None and dyn.is_displacement_action():
                dr, dc = dyn.get_displacement()
                step = max(step, abs(dr), abs(dc))
        return step

    def _observe_oriented_threat_motion(
        self,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
        av_feats: set[int],
    ) -> None:
        """Learn which creature kinds patrol by watching them translate.

        A human watching a creature notices: "that one walked one cell along its
        facing direction while I moved — it patrols". Stationary sentries never
        translate. This is a feature-level concept, so it generalizes to other
        creatures of the same kind in later levels.
        """
        if not av_feats:
            self._prev_oriented_threats = []
            return
        try:
            bg = self.estimate_background(curr_grid)
            entities = self.extract_entities(curr_grid, bg)
            threats = PerceptionEngine.detect_oriented_threats(
                curr_grid,
                entities,
                bg=bg,
                step_size=self.infer_motor_step_size(available_actions),
                avatar_pos=self.avatar_pos,
                avatar_features=av_feats,
            )
        except Exception:
            self._prev_oriented_threats = []
            return

        prev = self._prev_oriented_threats
        if prev:
            prev_by_feat: dict[int, set[tuple[int, int]]] = {}
            for t in prev:
                prev_by_feat.setdefault(t.feature_id, set()).add(t.pos)
            curr_by_feat: dict[int, set[tuple[int, int]]] = {}
            for t in threats:
                curr_by_feat.setdefault(t.feature_id, set()).add(t.pos)
            for feat, cur_positions in curr_by_feat.items():
                if feat in av_feats or feat in self.mobile_threat_features:
                    continue
                old_positions = prev_by_feat.get(feat, set())
                appeared = cur_positions - old_positions
                vanished = old_positions - cur_positions
                for a in appeared:
                    if any(
                        (a[0] == v[0]) != (a[1] == v[1])  # pure cardinal translation
                        for v in vanished
                    ):
                        self.mobile_threat_features.add(feat)
                        logger.info(
                            "AutonomousEpistemicEngine: Observed creature kind %d patrolling "
                            "autonomously — modeling its trajectory.",
                            feat,
                        )
                        break
        self._prev_oriented_threats = threats

    # ── Master Decision Loop ──────────────────────────────────────────────────

    def decide(
        self,
        curr_grid: Any,
        available_actions: Sequence[Any],
        is_win: bool = False,
        is_lost: bool = False,
        action_schemas: Sequence[Any] | None = None,
    ) -> tuple[Any, dict[str, Any] | None]:
        """Unified cognitive decision function:
        Perceive -> Assimilate -> Simulate in Mind -> Exploit / Epistemically Probe.
        """
        self.step_counter += 1
        if action_schemas:
            self.register_action_space(action_schemas)

        curr_grid = self.normalize_sensory_input(curr_grid)

        # 1. Update background & assimilate sensory feedback from previous action
        self.bg_feature = self.estimate_background(curr_grid)
        if self.prev_grid is not None and not self._feedback_assimilated:
            self.assimilate_feedback(curr_grid, available_actions, is_win=is_win, is_lost=is_lost)
        self._feedback_assimilated = False

        # Metacognitive Refractory Inhibition: decay refractory timers
        to_uninhibited = [act for act, timer in self.inhibited_actions.items() if timer <= 1]
        for act in self.inhibited_actions:
            self.inhibited_actions[act] -= 1
        for act in to_uninhibited:
            self.inhibited_actions.pop(act, None)

        # Visual avatar localization
        known_av_feats = set(self.avatar_features) if self.avatar_features else set()
        if self.avatar_feature is not None:
            known_av_feats.add(self.avatar_feature)
        elif self.prior_avatar_feature is not None:
            known_av_feats.add(self.prior_avatar_feature)
        if known_av_feats:
            self._update_avatar_position_from_grid(curr_grid, known_av_feats)
            if (
                self.avatar_pos is not None
                and self.avatar_feature is None
                and self.prior_avatar_feature is not None
            ):
                self.avatar_feature = self.prior_avatar_feature
                self.avatar_features.add(self.prior_avatar_feature)

        # Observe creature motion: oriented entities that translated since the
        # previous frame are autonomous patrollers, not stationary sentries.
        self._observe_oriented_threat_motion(curr_grid, available_actions, known_av_feats)

        chosen_action: Any
        chosen_data: dict[str, Any] | None = None
        predicted_pos: tuple[int, int] | None = None

        # Prefrontal Affordance Panel Sequence Chunking:
        # If an interactive control panel has unvisited pop-out/minority items, commit to completing the pattern.
        # CRITICAL GUARD: Only trigger panel clicking if there are NO active displacement actions (i.e. pure click games),
        # so physical avatar navigation is never hijacked by decorative background panels.
        active_panel_target: tuple[int, int] | None = None
        has_displacement_actions = any(self.is_displacement_action(a) for a in available_actions)
        spatial_effector_actions = [a for a in available_actions if self.is_spatial_effector(a)]
        if spatial_effector_actions and not has_displacement_actions and self.step_counter > 1:
            bg = self.estimate_background(curr_grid)
            entities = self.extract_entities(curr_grid, bg)
            panels = PerceptionEngine.detect_affordance_panels(entities, curr_grid, bg=bg)
            for p in panels:
                unvisited_minority = [
                    m.grid_pos
                    for m in p["minority_items"]
                    if self.entity_visit_counts.get(f"click_{m.grid_pos[0]}_{m.grid_pos[1]}", 0)
                    == 0
                    and m.grid_pos not in self.quiescent_click_targets
                ]
                if unvisited_minority:
                    active_panel_target = unvisited_minority[0]
                    break

        if active_panel_target is not None:
            chosen_action = spatial_effector_actions[0]
            tr, tc = active_panel_target
            self.entity_visit_counts[f"click_{tr}_{tc}"] = (
                self.entity_visit_counts.get(f"click_{tr}_{tc}", 0) + 1
            )
            aff = self.action_affordances.get(chosen_action)
            param_keys = aff.target_param_keys if aff else ("x", "y")
            chosen_data = {}
            for k in param_keys:
                if k in ("x", "col", "c", "column", "azimuth"):
                    chosen_data[k] = tc
                elif k in ("y", "row", "r", "elevation", "distance"):
                    chosen_data[k] = tr
                else:
                    chosen_data[k] = 0
            self.phase = EpistemicPhase.EPISTEMIC_EXPLORATION

        # 2. Check exploration cooldown
        elif self.exploration_cooldown > 0:
            self.exploration_cooldown -= 1
            self.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
            chosen_action, chosen_data = self.plan_epistemic_probe(curr_grid, available_actions)

        # 3. Check active mental plan during EXPLOITATION phase
        elif self.phase == EpistemicPhase.EXPLOITATION and self.mental_plan:
            next_step = self.mental_plan.popleft()
            if (
                not self.mental_plan
                and hasattr(self, "object_planner")
                and self.object_planner.active_target_object_id
            ):
                self.object_planner.ledger.record_interaction(
                    self.object_planner.active_target_object_id,
                    next_step.predicted_avatar_pos or self.avatar_pos or (0, 0),
                )
                self.object_planner.active_target_object_id = None

            # ── Reactive Safety Check ──────────────────────────────────────
            # Before executing each plan step, validate safety with TWO
            # complementary checks:
            #
            # A) STATIC CHECK: Verify no stationary lethal entity has appeared
            #    on a cell the plan will traverse (e.g. a sentry we hadn't
            #    seen before, or a barrier that opened/closed).
            #
            # B) DYNAMIC PATROL CHECK: Forward-simulate the current patroller
            #    positions step-by-step along the remaining plan.  The mental
            #    plan was computed with patrol prediction, but patrollers may
            #    have deviated from the predicted trajectory since then.
            #    Re-simulating from the ACTUAL current positions detects
            #    collisions the stale plan can't anticipate.
            plan_safe = True
            H, W = curr_grid.shape
            bg = self.estimate_background(curr_grid)
            all_future_steps = [next_step] + list(self.mental_plan)

            # (A) Static feature check along the planned path
            for future_step in all_future_steps:
                fr, fc = future_step.predicted_avatar_pos
                if 0 <= fr < H and 0 <= fc < W:
                    feat = int(curr_grid[fr, fc])
                    if feat in self.mobile_threat_features:
                        # Patrollers are modeled dynamically below; their
                        # CURRENT grid position says nothing about future safety.
                        continue
                    if feat in self.hazard_tracker.known_lethal_features:
                        plan_safe = False
                        break

            # (B) Dynamic patrol trajectory simulation
            #     Re-detect current patroller positions and forward-simulate
            #     their movement along the planned avatar trajectory.
            if plan_safe and self.mobile_threat_features:
                try:
                    step_size = self.infer_motor_step_size(available_actions)
                    entities = self.extract_entities(curr_grid, bg)
                    av_feats = self.avatar_features or (
                        {self.avatar_feature} if self.avatar_feature is not None else set()
                    )
                    live_threats = PerceptionEngine.detect_oriented_threats(
                        curr_grid,
                        entities,
                        bg=bg,
                        step_size=step_size,
                        avatar_pos=self.avatar_pos,
                        avatar_features=av_feats,
                    )
                    live_patrols: set[tuple[tuple[int, int], tuple[int, int]]] = set()
                    for t in live_threats:
                        if t.feature_id in self.mobile_threat_features:
                            live_patrols.add((t.pos, t.facing))

                    if live_patrols:
                        # Build patrol-aware wall set for corridor checks
                        half_step = max(1, step_size // 2)
                        barrier_feats = self.symbolic_theory.barrier_features
                        patrol_walls: set[tuple[int, int]] = set()
                        for rr in range(H):
                            for cc in range(W):
                                v = int(curr_grid[rr, cc])
                                if self.symbolic_theory.is_barrier(v) or v in barrier_feats:
                                    patrol_walls.add((rr, cc))

                        def _blocked(p: tuple[int, int], f: tuple[int, int]) -> bool:
                            probe = (p[0] + f[0] * half_step, p[1] + f[1] * half_step)
                            dest = (p[0] + f[0] * step_size, p[1] + f[1] * step_size)
                            for q in (probe, dest):
                                if not (0 <= q[0] < H and 0 <= q[1] < W) or q in patrol_walls:
                                    return True
                            return False

                        sim_patrols = frozenset(live_patrols)
                        for future_step in all_future_steps:
                            avatar_dest = future_step.predicted_avatar_pos
                            new_patrol_set: set[tuple[tuple[int, int], tuple[int, int]]] = set()
                            collision = False
                            for pat_pos, pat_facing in sim_patrols:
                                if pat_pos == avatar_dest:
                                    continue  # avatar eliminates patroller
                                dest = (
                                    pat_pos[0] + pat_facing[0] * step_size,
                                    pat_pos[1] + pat_facing[1] * step_size,
                                )
                                if (
                                    not (0 <= dest[0] < H and 0 <= dest[1] < W)
                                    or dest in patrol_walls
                                ):
                                    new_f = (-pat_facing[0], -pat_facing[1])
                                    new_patrol_set.add((pat_pos, new_f))
                                    continue
                                if dest == avatar_dest:
                                    collision = True
                                    break
                                new_f = pat_facing
                                if _blocked(dest, pat_facing):
                                    new_f = (-pat_facing[0], -pat_facing[1])
                                new_patrol_set.add((dest, new_f))
                            if collision:
                                plan_safe = False
                                break
                            sim_patrols = frozenset(new_patrol_set)
                except Exception:
                    pass  # If patrol simulation fails, rely on the static check

            if not plan_safe:
                self.mental_plan.clear()
                self.phase = EpistemicPhase.REPLANNING
                # Attempt forward simulation from current state with updated patroller positions
                simulated_plan = self.simulate_in_mind(curr_grid, available_actions)
                if not simulated_plan:
                    # If direct path fails, try blocking the immediate hazardous step to seek an alternative branch
                    danger_cells = {next_step.predicted_avatar_pos}
                    simulated_plan = self.simulate_in_mind(
                        curr_grid, available_actions, blocked_cells=danger_cells
                    )
                if simulated_plan:
                    self.phase = EpistemicPhase.EXPLOITATION
                    self.mental_plan = deque(simulated_plan)
                    next_step = self.mental_plan.popleft()
                    chosen_action = next_step.action
                    chosen_data = next_step.action_data
                    predicted_pos = next_step.predicted_avatar_pos
                else:
                    macro_plan = (
                        self.object_planner.plan_macro_option(self, curr_grid, available_actions)
                        if hasattr(self, "object_planner")
                        else None
                    )
                    if macro_plan:
                        self.phase = EpistemicPhase.EXPLOITATION
                        self.mental_plan = deque(macro_plan)
                        next_step = self.mental_plan.popleft()
                        chosen_action = next_step.action
                        chosen_data = next_step.action_data
                        predicted_pos = next_step.predicted_avatar_pos
                    else:
                        chosen_action, chosen_data = self.plan_epistemic_probe(
                            curr_grid, available_actions
                        )
            elif next_step.action in available_actions:
                chosen_action = next_step.action
                chosen_data = next_step.action_data
                predicted_pos = next_step.predicted_avatar_pos
            else:
                self.mental_plan.clear()
                self.phase = EpistemicPhase.REPLANNING
                chosen_action, chosen_data = self.plan_epistemic_probe(curr_grid, available_actions)

        # 4. If motor grounded, attempt Forward Mental Simulation first (direct verified path to goal)
        elif self.is_motor_grounded():
            simulated_plan = self.simulate_in_mind(curr_grid, available_actions)
            if simulated_plan:
                self.phase = EpistemicPhase.EXPLOITATION
                self.mental_plan = deque(simulated_plan)
                next_step = self.mental_plan.popleft()
                chosen_action = next_step.action
                chosen_data = next_step.action_data
                predicted_pos = next_step.predicted_avatar_pos
            else:
                # Direct path to goal blocked or goals not yet reachable;
                # consult Object-Centric Macro-Action State Graph Planner (5 Executive Directives)
                macro_plan = (
                    self.object_planner.plan_macro_option(self, curr_grid, available_actions)
                    if hasattr(self, "object_planner")
                    else None
                )
                if macro_plan:
                    self.phase = EpistemicPhase.EXPLOITATION
                    self.mental_plan = deque(macro_plan)
                    next_step = self.mental_plan.popleft()
                    chosen_action = next_step.action
                    chosen_data = next_step.action_data
                    predicted_pos = next_step.predicted_avatar_pos
                    if (
                        not self.mental_plan
                        and hasattr(self, "object_planner")
                        and self.object_planner.active_target_object_id
                    ):
                        self.object_planner.ledger.record_interaction(
                            self.object_planner.active_target_object_id,
                            predicted_pos or self.avatar_pos or (0, 0),
                        )
                        self.object_planner.active_target_object_id = None
                else:
                    self.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
                    chosen_action, chosen_data = self.plan_epistemic_probe(
                        curr_grid, available_actions
                    )

        # 5. Fallback to Epistemic Curiosity Probing
        else:
            self.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
            chosen_action, chosen_data = self.plan_epistemic_probe(curr_grid, available_actions)

        # Effector Grounding Invariant: Effector actions MUST have spatial target coordinates
        if self.is_spatial_effector(chosen_action):
            aff = self.action_affordances.get(chosen_action)
            req_keys = aff.target_param_keys if aff else ("x", "y")
            if (
                chosen_data is None
                or not isinstance(chosen_data, dict)
                or not all(k in chosen_data for k in req_keys)
            ):
                chosen_data = self.ground_effector_action(curr_grid, chosen_action)

        self.prev_grid = curr_grid.copy()
        self.last_action = chosen_action
        self.last_action_data = chosen_data
        self.last_predicted_pos = predicted_pos
        return chosen_action, chosen_data
