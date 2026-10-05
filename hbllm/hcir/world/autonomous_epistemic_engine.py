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

from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
from hbllm.hcir.world.causal_discovery import (
    BeliefTransitionEvent,
    CausalHypothesis,
    CausalPredicate,
)
from hbllm.hcir.world.motor_calibration import (
    ActionDynamicsModel,
    StateMutationModel,
)
from hbllm.hcir.world.prefrontal_working_memory import PrefrontalWorkingMemory
from hbllm.hcir.world.spatiotemporal_tracker import SpatiotemporalHazardTracker
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
                        "confidence": min(1.0, fill_ratio),
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

        # 3. Check for symmetry-completion goal
        sym_type, sym_score = VisualSymmetryAnalyzer.find_dominant_symmetry(grid)
        if 0.6 < sym_score < 0.95:
            goals.append(
                {
                    "type": "symmetry_completion",
                    "symmetry_axis": sym_type,
                    "symmetry_score": sym_score,
                    "confidence": sym_score * 0.8,
                }
            )

        return goals

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
            if target_coord is not None:
                engine.quiescent_click_targets.add(target_coord)

            if (
                not is_win
                and not is_lost
                and not is_known_displacement
                and bool(engine.action_dynamics)
            ):
                return

        engine.consecutive_quiescent_actions = 0
        if target_coord is not None:
            engine.effective_click_targets.add(target_coord)

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
        elif not is_effector_action:
            known_av_feats = engine.avatar_features or (
                {engine.avatar_feature} if engine.avatar_feature is not None else set()
            )
            av_moved = [item for item in moved_entities if item[1].feature_id in known_av_feats]
            if (
                not is_win
                and not is_lost
                and not av_moved
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
                if diff.changed_pixel_count > 0:
                    # HCIR Causal Transition: Internal state mutation / orientation transition detected
                    logger.debug(
                        "AutonomousEpistemicEngine: Internal state mutation / orientation transition detected (pixels changed: %d). Not a collision.",
                        diff.changed_pixel_count,
                    )
                elif (
                    diff.changed_pixel_count == 0
                    and act_dyn is not None
                    and act_dyn.is_displacement_action()
                ):
                    # True quiescent wall collision: no pixel changed anywhere in the environment
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

            # 2. Predictive coding efference copy divergence check
            # In HCIR, only trigger discrepancy invalidation if:
            # a) Action failed completely with quiescent resistance (changed_pixel_count == 0), OR
            # b) Real spatial divergence: avatar arrived at a different location than predicted
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

            if discrepancy:
                if (
                    engine.last_action is not None
                    and prev_avatar_pos is not None
                    and diff.changed_pixel_count == 0
                ):
                    engine.failed_transitions.add((prev_avatar_pos, engine.last_action))
                if engine.mental_plan:
                    logger.debug(
                        "AutonomousEpistemicEngine: Sensory discrepancy detected (pred=%s, actual=%s). Invalidating %d-step mental plan.",
                        engine.last_predicted_pos,
                        engine.avatar_pos,
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
                    if (
                        goal_feat != bg
                        and goal_feat != engine.bg_feature
                        and goal_feat not in av_feats
                        and not engine.symbolic_theory.is_barrier(goal_feat)
                    ):
                        engine.symbolic_theory.induce_goal(goal_feat)
                        logger.info(
                            "AutonomousEpistemicEngine: Grounded invariant WIN GOAL FEATURE %d at %s",
                            goal_feat,
                            win_target,
                        )

                for pe in prev_entities:
                    if (
                        pe.feature_id not in av_feats
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
            engine.reset_episode(retain_dynamics=True, is_new_level=True)
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

                    dead_feat = int(engine.prev_grid[target_pos[0], target_pos[1]])
                    if (
                        not engine.symbolic_theory.is_walkable(dead_feat)
                        and dead_feat != engine.bg_feature
                        and dead_feat not in av_feats
                    ):
                        # Only mark spatial coordinate as a static barrier if it was an impassable solid obstacle
                        engine.learned_barriers.add(target_pos)

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

        goals: list[tuple[int, int]] = []
        if confirmed_goals:
            goals = confirmed_goals

        # (b) Hypothesized goals: features believed to be goals from prior
        #     experience (candidate_goal_features retained across levels).
        #     A human seeing a familiar-looking object assumes "that's probably
        #     the goal again" until evidence contradicts it.
        if not goals:
            hypothesized_goals: list[tuple[int, int]] = []
            for e in entities:
                if engine.avatar_pos is not None and e.grid_pos == engine.avatar_pos:
                    continue
                if e.feature_id in av_feats:
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
            goals = [g for g in engine.learned_goal_positions if 0 <= g[0] < H and 0 <= g[1] < W]
        if not goals:
            structural_goals = engine.detect_structural_goals(curr_grid)
            for sg in structural_goals:
                feat = sg.get("feature")
                if feat is not None and (
                    feat in av_feats
                    or feat in engine.hazard_tracker.known_lethal_features
                    or engine.symbolic_theory.is_barrier(feat)
                ):
                    continue
                pos = sg.get("position")
                if pos and 0 <= pos[0] < H and 0 <= pos[1] < W and pos != engine.avatar_pos:
                    goals.append(pos)

        # (d) Exploratory guesses: unknown entities worth investigating.
        #     This is the weakest prior — "I don't know what these are, but
        #     interacting with them might teach me something."
        if not goals:
            exploratory_goals = [
                e.grid_pos
                for e in entities
                if e.role in (EntityRole.UNKNOWN, EntityRole.AGENT)
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
                    if e.grid_pos in goals and e.feature_id not in av_feats:
                        engine.symbolic_theory.candidate_goal_features.add(e.feature_id)

        av_feats = engine.avatar_features or (
            {engine.avatar_feature} if engine.avatar_feature is not None else set()
        )
        engine.symbolic_theory.goal_features.difference_update(av_feats)
        if engine.avatar_pos is not None:
            goals = [g for g in goals if g != engine.avatar_pos]

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
        static_barriers: set[tuple[int, int]] = set(engine.learned_barriers)
        for r in range(H):
            for c in range(W):
                val = int(curr_grid[r, c])
                if engine.symbolic_theory.is_barrier(val) or (
                    val in engine.hazard_tracker.known_lethal_features
                    and int(np.sum(curr_grid == val)) >= 25
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

        def _corridor_blocked(p: tuple[int, int], f: tuple[int, int]) -> bool:
            probe = (p[0] + f[0] * half_step, p[1] + f[1] * half_step)
            dest = (p[0] + f[0] * step_size, p[1] + f[1] * step_size)
            for q in (probe, dest):
                if not (0 <= q[0] < H and 0 <= q[1] < W) or q in static_barriers:
                    return True
            return False

        def advance_patrols(
            patrols: frozenset[tuple[tuple[int, int], tuple[int, int]]],
            avatar_new: tuple[int, int],
        ) -> frozenset[tuple[tuple[int, int], tuple[int, int]]] | None:
            """Predict patrollers after one avatar move. None = avatar gets caught."""
            nxt: set[tuple[tuple[int, int], tuple[int, int]]] = set()
            for p, f in patrols:
                if p == avatar_new:
                    continue  # avatar landed on it first -> eliminated
                if _corridor_blocked(p, f):
                    f = (-f[0], -f[1])
                    if _corridor_blocked(p, f):
                        nxt.add((p, f))  # boxed in: stays put
                        continue
                p2 = (p[0] + f[0] * step_size, p[1] + f[1] * step_size)
                if p2 == avatar_new:
                    return None  # patroller walks into the avatar
                nxt.add((p2, f))
            return frozenset(nxt)

        start_pos = engine.avatar_pos
        init_blocks = frozenset(pushable_blocks)
        init_open: frozenset[tuple[int, int]] = frozenset()

        is_block_delivery = bool(
            1 <= len(pushable_blocks) <= 5
            and any(
                engine.symbolic_theory.is_cargo(e) or e.role == EntityRole.MANIPULABLE
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
                int,
                int,
                tuple[int, int],
                frozenset[tuple[int, int]],
                frozenset[tuple[int, int]],
                frozenset[tuple[int, int]],
                frozenset[tuple[tuple[int, int], tuple[int, int]]],
                list[MentalSimulationStep],
            ]
        ] = []
        h0 = heuristic(start_pos, init_blocks)
        heapq.heappush(
            open_set,
            (h0, 0, counter, start_pos, init_blocks, init_open, init_threats, init_patrols, []),
        )

        visited_states: set[
            tuple[
                tuple[int, int],
                frozenset[tuple[int, int]],
                frozenset[tuple[int, int]],
                frozenset[tuple[int, int]],
                frozenset[tuple[tuple[int, int], tuple[int, int]]],
            ]
        ] = set()
        max_expansions = 1500 if not init_patrols else 4000

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
                path,
            ) = heapq.heappop(open_set)

            state_key = (cur_pos, cur_blocks, cur_open, cur_threats, cur_patrols)
            if state_key in visited_states:
                continue
            visited_states.add(state_key)

            # Direct goal reach: avatar reaches goal position
            if cur_pos in goals and len(path) > 0:
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

            for act, dr, dc in movable_actions:
                if (cur_pos, act) in engine.failed_transitions:
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
                if cur_patrols:
                    predicted = advance_patrols(cur_patrols, (nr, nc))
                    if predicted is None:
                        continue
                    new_patrols = predicted

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

                    block_set = set(cur_blocks)
                    block_set.remove((nr, nc))
                    block_set.add((pushed_r, pushed_c))
                    new_blocks = frozenset(block_set)

                new_open = cur_open
                if new_pos in mutation_triggers:
                    new_open = cur_open | frozenset(mutation_triggers[new_pos])

                # Total step cost = base (1) + epistemic uncertainty penalty
                step_cost = 1 + epistemic_cost
                new_g = g_cost + step_cost
                new_h = heuristic(new_pos, new_blocks)
                new_step = MentalSimulationStep(action=act, predicted_avatar_pos=new_pos)
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
                        path + [new_step],
                    ),
                )

        # Hierarchical Subgoal Decomposition Fallback
        if mutation_triggers:
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

        # A. Non-displacement or Pure Spatial Effector Environments
        if spatial_effector_actions and (
            not displacement_actions or engine.consecutive_quiescent_actions > 0
        ):
            if discrete_transform_actions and engine.consecutive_quiescent_actions > 1:
                return discrete_transform_actions[
                    engine.step_counter % len(discrete_transform_actions)
                ], None
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

            # Oscillation Loop Breaking
            is_oscillating = engine.recent_positions.count(engine.avatar_pos) >= 3
            if is_oscillating:
                logger.debug(
                    "AutonomousEpistemicEngine: Oscillation detected at %s! Breaking loop...",
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
                best_act = available_actions[0]
                min_visits = float("inf")
                disp_actions = [
                    (act, engine.action_dynamics[act].get_displacement())
                    for act in available_actions
                    if act in engine.action_dynamics
                    and engine.action_dynamics[act].is_displacement_action()
                ]
                for act, (dr_cal, dc_cal) in disp_actions:
                    nr = engine.avatar_pos[0] + dr_cal
                    nc = engine.avatar_pos[1] + dc_cal
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if (nr, nc) in engine.learned_barriers or engine.symbolic_theory.is_barrier(
                        int(curr_grid[nr, nc])
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
                if min_visits == float("inf") and spatial_effector_actions:
                    chosen_eff = spatial_effector_actions[0]
                    return chosen_eff, engine.ground_effector_action(curr_grid, chosen_eff)
                return best_act, None

            candidate_entities = [
                e
                for e in entities
                if e.id not in engine.probed_entity_ids
                and (
                    abs(e.grid_pos[0] - engine.avatar_pos[0])
                    + abs(e.grid_pos[1] - engine.avatar_pos[1])
                )
                > 1
            ]

            if engine.active_probe_target is None and candidate_entities:
                scored: list[tuple[SpatialEntity, float]] = []
                for e in candidate_entities:
                    dist = abs(e.grid_pos[0] - engine.avatar_pos[0]) + abs(
                        e.grid_pos[1] - engine.avatar_pos[1]
                    )
                    visits = engine.entity_visit_counts.get(e.id, 0)
                    info_val = 10.0 / (visits + 1.0)
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

            target_pos = engine.active_probe_target or (H // 2, W // 2)
            tr, tc = target_pos

            best_action = available_actions[0]
            min_dist = float("inf")
            disp_actions = [
                (act, engine.action_dynamics[act].get_displacement())
                for act in displacement_actions
            ]
            for act, (dr_cal, dc_cal) in disp_actions:
                if (engine.avatar_pos, act) in engine.failed_transitions:
                    continue
                nr = engine.avatar_pos[0] + dr_cal
                nc = engine.avatar_pos[1] + dc_cal
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if (nr, nc) in engine.learned_barriers or engine.symbolic_theory.is_barrier(
                    int(curr_grid[nr, nc])
                ):
                    continue

                # Known-lethal features are hard-blocked; unverified features
                # carry an uncertainty cost (caution, not prohibition).
                cell_feat = int(curr_grid[nr, nc])
                if cell_feat in engine.hazard_tracker.known_lethal_features:
                    continue
                # The probe target itself is the thing being investigated — no penalty.
                uncertainty_pen = (
                    0.0
                    if (nr, nc) == (tr, tc)
                    else engine._epistemic_uncertainty_cost(cell_feat, bg)
                )

                visit_penalty = float(engine.position_visit_counts.get((nr, nc), 0)) * 1.5
                recency_penalty = 10.0 if (nr, nc) in list(engine.recent_positions)[-4:] else 0.0
                d = abs(tr - nr) + abs(tc - nc) + visit_penalty + recency_penalty + uncertainty_pen
                if d < min_dist:
                    min_dist = d
                    best_action = act

            if min_dist == float("inf"):
                if spatial_effector_actions:
                    chosen_eff = spatial_effector_actions[0]
                    return chosen_eff, engine.ground_effector_action(curr_grid, chosen_eff)
                elif discrete_transform_actions:
                    return discrete_transform_actions[
                        engine.step_counter % len(discrete_transform_actions)
                    ], None
                uncalibrated = [
                    a for a in available_actions if not engine.is_displacement_action(a)
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
                return available_actions[engine.step_counter % len(available_actions)], None

            return best_action, None

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

        # Episodic Ground Facts (Layout coordinates, wiped on level transition)
        self.learned_barriers: set[tuple[int, int]] = set()
        self.learned_goal_positions: set[tuple[int, int]] = set()
        self.learned_receptacle_positions: set[tuple[int, int]] = set()

        # Mental simulation plan
        self.mental_plan: deque[MentalSimulationStep] = deque()
        self.active_hypothesis: CausalHypothesis | None = None
        self.failed_transitions: set[tuple[tuple[int, int], int]] = set()

        # Theory of Mind: creature kinds observed to patrol autonomously
        # (feature-level concept, retained across levels like a human would).
        self.mobile_threat_features: set[int] = set()
        self._prev_oriented_threats: list[OrientedThreat] = []
        self.consecutive_plan_failures: int = 0
        self.exploration_cooldown: int = 0
        self.current_simulated_goal: tuple[int, int] | None = None
        self.last_predicted_pos: tuple[int, int] | None = None

        # Prefrontal working memory & biological hazard tracker
        self.working_memory: PrefrontalWorkingMemory = PrefrontalWorkingMemory()
        self.hazard_tracker: SpatiotemporalHazardTracker = SpatiotemporalHazardTracker()
        self.saccadic_attention: SaccadicAttentionSystem = SaccadicAttentionSystem()

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

    # ── Episodic State Management ─────────────────────────────────────────────

    def reset_episode(self, retain_dynamics: bool = True, is_new_level: bool = False) -> None:
        """Reset episodic state upon level transition or death."""
        self.prev_grid = None
        self._prev_oriented_threats = []
        self.avatar_pos = None
        self.last_action = None
        self.last_action_data = None
        self.last_predicted_pos = None
        self.consecutive_quiescent_actions = 0
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

        if not retain_dynamics or is_new_level:
            self.hazard_tracker.reset_episode()
            self.exhausted_candidate_goals.clear()
            self.tested_action_positions.clear()
            self.failed_transitions.clear()
            # Episodic spatial coordinates are layout-specific and MUST be cleared between levels
            self.learned_barriers.clear()
            self.learned_goal_positions.clear()
            self.learned_receptacle_positions.clear()

        if is_new_level and retain_dynamics:
            # ── Principled Hypothesis Management on Level Transition ──────────
            #
            # The human brain doesn't blindly retain or blindly clear knowledge.
            # It carries forward hypotheses and VERIFIES them through experience.
            #
            # 1. Avatar identity MUST be re-verified: the player always needs to
            #    discover "which thing am I?" in a new level. The controllability
            #    test will naturally re-identify the avatar by testing correlation
            #    between actions and entity movements across multiple frames.
            self.avatar_feature = None
            self.avatar_features.clear()
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

            # 4. Action dynamics and affordances are RETAINED — they represent
            #    structural motor knowledge (e.g., "action 1 moves up by 6 pixels")
            #    that is typically level-invariant.

            # Force re-grounding via controllability test
            self.phase = EpistemicPhase.MOTOR_GROUNDING

        elif not retain_dynamics:
            self.total_epistemic_probes = 0
            self.avatar_feature = None
            self.avatar_features.clear()
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
            self.phase = EpistemicPhase.MOTOR_GROUNDING
        else:
            self.phase = (
                EpistemicPhase.MENTAL_SIMULATION
                if self.is_motor_grounded()
                else EpistemicPhase.MOTOR_GROUNDING
            )

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

    def detect_structural_goals(self, grid: np.ndarray) -> list[dict[str, Any]]:
        return PerceptionEngine.detect_structural_goals(
            grid,
            bg=self.bg_feature,
            avatar_features=self.avatar_features,
            avatar_feature=self.avatar_feature,
            barrier_features=self.symbolic_theory.barrier_features,
            known_lethal_features=self.hazard_tracker.known_lethal_features,
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

    def _update_avatar_position_from_grid(
        self, curr_grid: np.ndarray, known_av_feats: set[int]
    ) -> None:
        """Visually localize the avatar position on the current grid."""
        H, W = curr_grid.shape
        # If avatar was already precisely localized by assimilate_feedback via motion tracking,
        # and current avatar_pos contains an avatar feature, verify and keep it
        if self.avatar_pos is not None:
            r, c = self.avatar_pos
            if 0 <= r < H and 0 <= c < W and curr_grid[r, c] in known_av_feats:
                return

        from scipy.ndimage import label

        mask = np.isin(curr_grid, list(known_av_feats))
        if not np.any(mask):
            return

        labeled, num_features = label(mask)
        best_pos = None
        min_dist = float("inf")

        for lbl in range(1, num_features + 1):
            coords = np.argwhere(labeled == lbl)
            if 1 <= len(coords) <= 49:
                centroid = (int(np.mean(coords[:, 0])), int(np.mean(coords[:, 1])))
                if self.avatar_pos is not None:
                    dist = abs(centroid[0] - self.avatar_pos[0]) + abs(
                        centroid[1] - self.avatar_pos[1]
                    )
                    if dist < min_dist:
                        min_dist = dist
                        best_pos = centroid
                else:
                    best_pos = centroid
                    break

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
        for e in entities:
            if e.feature_id == bg or e.feature_id in av_feats or e.area > 120:
                continue
            cr, cc = e.grid_pos
            if (cr, cc) in self.quiescent_click_targets:
                continue
            visit_count = self.entity_visit_counts.get(f"click_{cr}_{cc}", 0)
            saliency = 100.0 / math.log2(2 + max(1, e.area))
            cand_score = saliency - float(visit_count) * 20.0
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
                if (cr, cc) in self.quiescent_click_targets or curr_grid[cr, cc] == bg:
                    continue
                visit_count = self.entity_visit_counts.get(f"click_{cr}_{cc}", 0)
                cand_score = f.salience - float(visit_count) * 0.35
                click_candidates.append((cr, cc, cand_score))

        if click_candidates:
            click_candidates.sort(key=lambda x: x[2], reverse=True)
            best_r, best_c, _ = click_candidates[0]
            self.entity_visit_counts[f"click_{best_r}_{best_c}"] = (
                self.entity_visit_counts.get(f"click_{best_r}_{best_c}", 0) + 1
            )
            target_r, target_c = int(best_r), int(best_c)
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

        # Visual avatar localization
        known_av_feats = self.avatar_features or (
            {self.avatar_feature} if self.avatar_feature is not None else set()
        )
        if known_av_feats:
            self._update_avatar_position_from_grid(curr_grid, known_av_feats)

        # Observe creature motion: oriented entities that translated since the
        # previous frame are autonomous patrollers, not stationary sentries.
        self._observe_oriented_threat_motion(curr_grid, available_actions, known_av_feats)

        chosen_action: Any
        chosen_data: dict[str, Any] | None = None
        predicted_pos: tuple[int, int] | None = None

        # 2. Check exploration cooldown
        if self.exploration_cooldown > 0:
            self.exploration_cooldown -= 1
            self.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
            chosen_action, chosen_data = self.plan_epistemic_probe(curr_grid, available_actions)

        # 3. Check active mental plan during EXPLOITATION phase
        elif self.phase == EpistemicPhase.EXPLOITATION and self.mental_plan:
            next_step = self.mental_plan.popleft()

            # ── Reactive Safety Check ──────────────────────────────────────
            # Before executing each plan step, scan the ENTIRE remaining
            # plan path on the CURRENT grid. A human doesn't just check the
            # next step — they look ahead down the corridor. If a moving
            # entity has entered the planned path anywhere, the human stops
            # and waits rather than charging into a head-on collision.
            plan_safe = True
            H, W = curr_grid.shape
            bg = self.estimate_background(curr_grid)
            all_future_steps = [next_step] + list(self.mental_plan)
            cur_r, cur_c = self.avatar_pos if self.avatar_pos is not None else (-1, -1)
            nr, nc = next_step.predicted_avatar_pos
            dr, dc = nr - cur_r, nc - cur_c

            for future_step in all_future_steps:
                fr, fc = future_step.predicted_avatar_pos
                if 0 <= fr < H and 0 <= fc < W:
                    feat = int(curr_grid[fr, fc])
                    if feat in self.mobile_threat_features:
                        # Patrollers are already modeled dynamically in the plan;
                        # their CURRENT position says nothing about future safety.
                        continue
                    if feat in self.hazard_tracker.known_lethal_features:
                        plan_safe = False
                        break
                    if self.is_feature_unverified(feat, bg):
                        # An unverified entity threatens the plan if it lies ahead
                        # in the SAME corridor that the avatar is currently advancing along
                        # (beyond the immediate destination step).
                        in_corridor_ahead = (
                            dr != 0
                            and fc == cur_c
                            and (fr - cur_r) * dr > 0
                            and (fr, fc) != (nr, nc)
                        ) or (
                            dc != 0
                            and fr == cur_r
                            and (fc - cur_c) * dc > 0
                            and (fr, fc) != (nr, nc)
                        )
                        if in_corridor_ahead:
                            plan_safe = False
                            break

            if not plan_safe:
                self.mental_plan.clear()
                self.phase = EpistemicPhase.REPLANNING
                # The immediate next step leads towards the detected danger zone.
                # In order to replan an alternative safe route, block this dangerous corridor step!
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

        # 4. If motor grounded, attempt Forward Mental Simulation
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
                self.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
                chosen_action, chosen_data = self.plan_epistemic_probe(curr_grid, available_actions)

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
