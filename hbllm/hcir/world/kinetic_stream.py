"""Dorsal Visual Stream (Area MT / V5) — Kinetic Figure-Ground Segregation.

Modeled on primate dorsal visual cortex (Area MT/V5) and corollary discharge:
1. Retinal Motion Differencing: Computes optical displacement fields between successive frames.
2. Corollary Discharge Subtraction: Compares observed visual motion against motor efference
   copies to isolate the controlled self-avatar via common-fate motion, even in the presence
   of camouflage, multi-palette textures, or background-matching pixel values.
3. Autonomous Kinetic Agent Segregation: Isolates moving entities that deviate from the
   efference copy as autonomous dynamic agents / hazards, tracking their velocity vectors
   for predictive trajectory extrapolation.
4. Declarative Neuro-Symbolic Integration: Converts kinetic entities into typed SpatialEntity
   components, CausalPredicates, and atomic TransactionOperations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
from scipy.ndimage import label

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.cortex_causal import CausalPredicate
from hbllm.hcir.world.cortex_perception import EntityRole, SpatialEntity

logger = logging.getLogger(__name__)


class KineticRole(StrEnum):
    """Categorical role of a segregated kinetic entity."""

    SELF_AVATAR = "self_avatar"
    AUTONOMOUS_AGENT = "autonomous_agent"
    DYNAMIC_HAZARD = "dynamic_hazard"
    PASSIVE_DEBRIS = "passive_debris"
    UNKNOWN = "unknown"


class MotionType(StrEnum):
    """Classified kinematic deformation of motion cluster."""

    TRANSLATIONAL = "translational"
    EXPANSIVE = "expansive"
    CONTRACTIVE = "contractive"
    STATIONARY = "stationary"


@dataclass
class KineticEntity:
    """An entity segregated purely through temporal motion dynamics."""

    centroid: tuple[float, float]
    velocity: tuple[float, float]  # (dr, dc) per time step
    cells: list[tuple[int, int]]
    bounding_box: tuple[int, int, int, int]  # (min_r, max_r, min_c, max_c)
    area: int
    features: set[int]
    is_self: bool = False
    confidence: float = 1.0
    role: KineticRole = KineticRole.UNKNOWN
    motion_type: MotionType = MotionType.TRANSLATIONAL
    entity_id: str = ""

    def __post_init__(self) -> None:
        if self.is_self and self.role == KineticRole.UNKNOWN:
            self.role = KineticRole.SELF_AVATAR
        elif not self.is_self and self.role == KineticRole.UNKNOWN:
            v_mag = abs(self.velocity[0]) + abs(self.velocity[1])
            self.role = KineticRole.DYNAMIC_HAZARD if v_mag > 0 else KineticRole.PASSIVE_DEBRIS
        if not self.entity_id:
            prefix = "avatar" if self.is_self else "kinetic"
            self.entity_id = (
                f"{prefix}_{int(round(self.centroid[0]))}_{int(round(self.centroid[1]))}"
            )

    def to_spatial_entity(self, entity_id: str | None = None) -> SpatialEntity:
        """Lift kinetic entity into a formal typed HCIR SpatialEntity."""
        role_enum = EntityRole.AGENT if self.is_self else EntityRole.DYNAMIC_HAZARD
        target_id = entity_id or self.entity_id
        min_r, max_r, min_c, max_c = self.bounding_box
        color_val = next(iter(self.features)) if self.features else 0
        return SpatialEntity(
            id=target_id,
            role=role_enum,
            centroid=self.centroid,
            grid_pos=(int(round(self.centroid[0])), int(round(self.centroid[1]))),
            area=self.area,
            bounding_box=(min_r, max_r, min_c, max_c),
            feature_id=color_val,
            color=color_val,
            properties={
                "velocity": self.velocity,
                "is_self": self.is_self,
                "confidence": self.confidence,
                "role": str(self.role),
                "motion_type": str(self.motion_type),
            },
        )

    def to_causal_predicate(self) -> CausalPredicate:
        """Export dynamic velocity and trajectory as a formal declarative CausalPredicate."""
        return CausalPredicate(
            variable="kinetic_velocity",
            operator="==",
            value=(round(self.velocity[0], 2), round(self.velocity[1], 2)),
            relation=str(self.role),
            target_entity="avatar" if self.is_self else self.entity_id,
        )

    def extrapolate_trajectory(
        self, horizon: int = 3, grid_shape: tuple[int, int] | None = None
    ) -> list[tuple[int, int]]:
        """Project forward extrapolated trajectory for dynamic collision cones."""
        trajectory: list[tuple[int, int]] = []
        cur_r, cur_c = self.centroid[0], self.centroid[1]
        dr, dc = self.velocity[0], self.velocity[1]

        for step in range(1, horizon + 1):
            nr = int(round(cur_r + dr * step))
            nc = int(round(cur_c + dc * step))
            if grid_shape is not None:
                H, W = grid_shape
                nr = max(0, min(H - 1, nr))
                nc = max(0, min(W - 1, nc))
            trajectory.append((nr, nc))
        return trajectory


@dataclass
class KineticSegregationResult:
    """Result of MT/V5 kinetic figure-ground decomposition."""

    self_avatar: KineticEntity | None = None
    external_agents: list[KineticEntity] = field(default_factory=list)
    motion_energy: int = 0
    raw_delta: tuple[int, int] | None = None

    def get_moving_agents(self) -> list[KineticEntity]:
        """Retrieve external agents displaying non-zero translational velocity."""
        return [
            agent
            for agent in self.external_agents
            if abs(agent.velocity[0]) > 0.1 or abs(agent.velocity[1]) > 0.1
        ]

    def has_avatar_motion(self) -> bool:
        """True if the self avatar underwent displacement during this frame."""
        return self.self_avatar is not None and (
            abs(self.self_avatar.velocity[0]) > 0.1 or abs(self.self_avatar.velocity[1]) > 0.1
        )

    def to_causal_predicates(self) -> list[CausalPredicate]:
        """Export all segregated kinetic states into declarative causal predicates."""
        predicates: list[CausalPredicate] = []
        if self.self_avatar is not None:
            predicates.append(self.self_avatar.to_causal_predicate())
        for agent in self.external_agents:
            predicates.append(agent.to_causal_predicate())
        return predicates

    def to_spatial_entities(self) -> list[SpatialEntity]:
        """Lift all segregated kinetic entities into typed SpatialEntity components."""
        entities: list[SpatialEntity] = []
        if self.self_avatar is not None:
            entities.append(self.self_avatar.to_spatial_entity())
        for agent in self.external_agents:
            entities.append(agent.to_spatial_entity())
        return entities

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR state updates for all segregated entities."""
        ops: list[TransactionOperation] = []
        for entity in self.to_spatial_entities():
            ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=entity.id,
                    node_data=entity.properties or {},
                )
            )
        return ops


class DorsalKineticStream:
    """Dorsal stream MT/V5 motion processor and corollary discharge segregator."""

    def __init__(self, velocity_tolerance: float = 1.5) -> None:
        self.velocity_tolerance = velocity_tolerance
        self.tracked_external_agents: list[KineticEntity] = []

    def segregate_motion(
        self,
        prev_grid: np.ndarray,
        curr_grid: np.ndarray,
        commanded_delta: tuple[int, int] | None = None,
        background_feature: int = 0,
        known_avatar_features: set[int] | None = None,
    ) -> KineticSegregationResult:
        """Decompose frame difference into self-effector vs. external kinetic entities.

        Args:
            prev_grid: Visual frame at t-1 (H, W).
            curr_grid: Visual frame at t (H, W).
            commanded_delta: Motor efference copy (dr, dc) expected from executed action.
            background_feature: Estimated background color.
            known_avatar_features: Previously learned avatar features, if any.

        Returns:
            KineticSegregationResult containing segregated self-avatar and external agents.
        """
        if prev_grid.shape != curr_grid.shape:
            return KineticSegregationResult()

        diff_mask = prev_grid != curr_grid
        motion_energy = int(np.sum(diff_mask))

        if motion_energy == 0:
            return KineticSegregationResult(motion_energy=0)

        # 8-connected morphological labeling of changed visual clusters (supports diagonal contours and outlines)
        labeled_diff, num_features = label(diff_mask, structure=np.ones((3, 3)))

        candidates: list[KineticEntity] = []
        vacated_clusters: list[dict[str, Any]] = []
        occupied_clusters: list[dict[str, Any]] = []

        for feat_id in range(1, num_features + 1):
            cluster_coords = np.argwhere(labeled_diff == feat_id)
            if len(cluster_coords) == 0:
                continue

            cells = [(int(r), int(c)) for r, c in cluster_coords]
            min_r, max_r = int(np.min(cluster_coords[:, 0])), int(np.max(cluster_coords[:, 0]))
            min_c, max_c = int(np.min(cluster_coords[:, 1])), int(np.max(cluster_coords[:, 1]))
            H, W = curr_grid.shape
            is_edge_hud = (
                min_c == max_c and (min_c == 0 or min_c == W - 1) and (max_r - min_r >= 4)
            ) or (min_r == max_r and (min_r == 0 or min_r == H - 1) and (max_c - min_c >= 4))
            if is_edge_hud:
                continue
            bbox = (min_r, max_r, min_c, max_c)

            # Separate vacated (disappeared from prev) vs occupied (appeared in curr)
            vacated = [
                (r, c)
                for r, c in cells
                if curr_grid[r, c] == background_feature and prev_grid[r, c] != background_feature
            ]
            occupied = [
                (r, c)
                for r, c in cells
                if curr_grid[r, c] != background_feature and prev_grid[r, c] == background_feature
            ]

            # If background_feature did not capture the substrate:
            if not vacated and not occupied:
                prev_unique = {int(prev_grid[r, c]) for r, c in cells}
                curr_unique = {int(curr_grid[r, c]) for r, c in cells}
                if len(curr_unique) == 1 and len(prev_unique) > 1:
                    vacated = cells
                elif len(prev_unique) == 1 and len(curr_unique) > 1:
                    occupied = cells

            if vacated and occupied:
                # Overlapping footprint (e.g. 1-cell step with overlapping area)
                c_vacated = (
                    float(np.mean([p[0] for p in vacated])),
                    float(np.mean([p[1] for p in vacated])),
                )
                c_occupied = (
                    float(np.mean([p[0] for p in occupied])),
                    float(np.mean([p[1] for p in occupied])),
                )
                velocity = (c_occupied[0] - c_vacated[0], c_occupied[1] - c_vacated[1])
                curr_cells = occupied
                feats = {
                    int(curr_grid[r, c])
                    for r, c in curr_cells
                    if curr_grid[r, c] != background_feature
                } or {int(curr_grid[r, c]) for r, c in curr_cells}
                cand = KineticEntity(
                    centroid=c_occupied,
                    velocity=velocity,
                    cells=curr_cells,
                    bounding_box=bbox,
                    area=len(curr_cells),
                    features=feats,
                )
                candidates.append(cand)
            elif vacated and not occupied:
                c_vac = (
                    float(np.mean([p[0] for p in vacated])),
                    float(np.mean([p[1] for p in vacated])),
                )
                feats = {
                    int(prev_grid[r, c])
                    for r, c in vacated
                    if prev_grid[r, c] != background_feature
                } or {int(prev_grid[r, c]) for r, c in vacated}
                sub_feats = {int(curr_grid[r, c]) for r, c in vacated}
                vacated_clusters.append(
                    {
                        "id": feat_id,
                        "centroid": c_vac,
                        "cells": vacated,
                        "bbox": bbox,
                        "area": len(vacated),
                        "features": feats,
                        "substrate": sub_feats,
                    }
                )
            elif occupied and not vacated:
                c_occ = (
                    float(np.mean([p[0] for p in occupied])),
                    float(np.mean([p[1] for p in occupied])),
                )
                feats = {
                    int(curr_grid[r, c])
                    for r, c in occupied
                    if curr_grid[r, c] != background_feature
                } or {int(curr_grid[r, c]) for r, c in occupied}
                sub_feats = {int(prev_grid[r, c]) for r, c in occupied}
                occupied_clusters.append(
                    {
                        "id": feat_id,
                        "centroid": c_occ,
                        "cells": occupied,
                        "bbox": bbox,
                        "area": len(occupied),
                        "features": feats,
                        "substrate": sub_feats,
                    }
                )
            else:
                # Direct centroid comparison or single color change
                prev_non_bg = [(r, c) for r, c in cells if prev_grid[r, c] != background_feature]
                curr_non_bg = [(r, c) for r, c in cells if curr_grid[r, c] != background_feature]
                if prev_non_bg and curr_non_bg:
                    c_prev = (
                        float(np.mean([p[0] for p in prev_non_bg])),
                        float(np.mean([p[1] for p in prev_non_bg])),
                    )
                    c_curr = (
                        float(np.mean([p[0] for p in curr_non_bg])),
                        float(np.mean([p[1] for p in curr_non_bg])),
                    )
                    velocity = (c_curr[0] - c_prev[0], c_curr[1] - c_prev[1])
                    curr_cells = curr_non_bg
                    feats = {
                        int(curr_grid[r, c])
                        for r, c in curr_cells
                        if curr_grid[r, c] != background_feature
                    } or {int(curr_grid[r, c]) for r, c in cells}
                    candidates.append(
                        KineticEntity(
                            centroid=c_curr,
                            velocity=velocity,
                            cells=curr_cells,
                            bounding_box=bbox,
                            area=len(curr_cells),
                            features=feats,
                        )
                    )
                else:
                    c_mid = (
                        float(np.mean(cluster_coords[:, 0])),
                        float(np.mean(cluster_coords[:, 1])),
                    )
                    p_feats = {int(prev_grid[r, c]) for r, c in cells}
                    c_feats = {int(curr_grid[r, c]) for r, c in cells}
                    vacated_clusters.append(
                        {
                            "id": feat_id,
                            "centroid": c_mid,
                            "cells": cells,
                            "bbox": bbox,
                            "area": len(cells),
                            "features": p_feats,
                            "substrate": c_feats,
                        }
                    )
                    occupied_clusters.append(
                        {
                            "id": feat_id,
                            "centroid": c_mid,
                            "cells": cells,
                            "bbox": bbox,
                            "area": len(cells),
                            "features": c_feats,
                            "substrate": p_feats,
                        }
                    )

        # Match disjoint vacated clusters to occupied clusters
        used_vac: set[int] = set()
        used_occ: set[int] = set()

        for o_idx, occ in enumerate(occupied_clusters):
            best_v_idx: int | None = None
            min_cost = float("inf")
            for v_idx, vac in enumerate(vacated_clusters):
                if v_idx in used_vac or vac["id"] == occ["id"]:
                    continue
                common_feats = occ["features"] & vac["features"]
                if not common_feats:
                    continue
                area_diff = abs(occ["area"] - vac["area"])
                if area_diff > max(3, int(0.5 * max(occ["area"], vac["area"]))):
                    continue
                dr = occ["centroid"][0] - vac["centroid"][0]
                dc = occ["centroid"][1] - vac["centroid"][1]
                dist = (dr**2 + dc**2) ** 0.5
                if dist < 0.5:
                    continue
                common_sub = occ["substrate"] & vac["substrate"]
                sub_bonus = 20.0 if common_sub else 0.0
                cost = dist + 5.0 * area_diff - 10.0 * len(common_feats) - sub_bonus
                if cost < min_cost:
                    min_cost = cost
                    best_v_idx = v_idx

            if best_v_idx is not None:
                vac = vacated_clusters[best_v_idx]
                used_vac.add(best_v_idx)
                used_occ.add(o_idx)
                velocity = (
                    occ["centroid"][0] - vac["centroid"][0],
                    occ["centroid"][1] - vac["centroid"][1],
                )
                cand = KineticEntity(
                    centroid=occ["centroid"],
                    velocity=velocity,
                    cells=occ["cells"],
                    bounding_box=occ["bbox"],
                    area=occ["area"],
                    features=occ["features"],
                )
                candidates.append(cand)

        # Unmatched occupied clusters: stationary or unlinked entities
        for o_idx, occ in enumerate(occupied_clusters):
            if o_idx not in used_occ:
                already_covered = any(
                    abs(c.centroid[0] - occ["centroid"][0]) < 0.5
                    and abs(c.centroid[1] - occ["centroid"][1]) < 0.5
                    for c in candidates
                )
                if not already_covered:
                    cand = KineticEntity(
                        centroid=occ["centroid"],
                        velocity=(0.0, 0.0),
                        cells=occ["cells"],
                        bounding_box=occ["bbox"],
                        area=occ["area"],
                        features=occ["features"],
                    )
                    candidates.append(cand)

        # Corollary Discharge Matching:
        self_avatar: KineticEntity | None = None
        external_agents: list[KineticEntity] = []

        if commanded_delta is not None and (commanded_delta[0] != 0 or commanded_delta[1] != 0):
            exp_dr, exp_dc = float(commanded_delta[0]), float(commanded_delta[1])
            best_match_idx: int | None = None
            min_err = float("inf")

            # Ventral body schema gating: require velocity consistency before gating to prior features
            has_feat_match = known_avatar_features and any(
                bool(cand.features & known_avatar_features)
                and (
                    abs(cand.velocity[0] - exp_dr) + abs(cand.velocity[1] - exp_dc)
                    <= self.velocity_tolerance
                )
                for cand in candidates
            )

            for i, cand in enumerate(candidates):
                if has_feat_match and not (cand.features & known_avatar_features):
                    continue

                # Stationary candidate cannot explain non-zero commanded displacement without prior feature match
                if cand.velocity == (0.0, 0.0) and not has_feat_match:
                    continue

                err = abs(cand.velocity[0] - exp_dr) + abs(cand.velocity[1] - exp_dc)
                if known_avatar_features and (cand.features & known_avatar_features):
                    err *= 0.5

                if err <= self.velocity_tolerance or has_feat_match:
                    if err < min_err:
                        min_err = err
                        best_match_idx = i

            if best_match_idx is not None:
                self_avatar = candidates[best_match_idx]
                self_avatar.is_self = True
                self_avatar.role = KineticRole.SELF_AVATAR
                self_avatar.confidence = max(
                    0.6, 1.0 - (min_err / (self.velocity_tolerance + 1e-4))
                )
                external_agents = [c for j, c in enumerate(candidates) if j != best_match_idx]
            else:
                external_agents = list(candidates)
        else:
            # Stationary or uncalibrated command: check known avatar feature overlap
            if known_avatar_features:
                for cand in candidates:
                    if cand.features & known_avatar_features:
                        if self_avatar is None:
                            self_avatar = cand
                            self_avatar.is_self = True
                            self_avatar.role = KineticRole.SELF_AVATAR
                        else:
                            external_agents.append(cand)
                    else:
                        external_agents.append(cand)
            else:
                external_agents = list(candidates)

        self.tracked_external_agents = external_agents
        raw_delta = (
            (int(round(self_avatar.velocity[0])), int(round(self_avatar.velocity[1])))
            if self_avatar is not None
            else None
        )

        return KineticSegregationResult(
            self_avatar=self_avatar,
            external_agents=external_agents,
            motion_energy=motion_energy,
            raw_delta=raw_delta,
        )
