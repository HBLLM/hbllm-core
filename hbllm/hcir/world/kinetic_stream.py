"""Dorsal Visual Stream (Area MT / V5) — Kinetic Figure-Ground Segregation.

Modeled on primate dorsal visual cortex (Area MT/V5) and corollary discharge:
1. Retinal Motion Differencing: Computes optical displacement fields between successive frames.
2. Corollary Discharge Subtraction: Compares observed visual motion against motor efference
   copies to isolate the controlled self-avatar via common-fate motion, even in the presence
   of camouflage, multi-palette textures, or background-matching pixel values.
3. Autonomous Kinetic Agent Segregation: Isolates moving entities that deviate from the
   efference copy as autonomous dynamic agents / hazards, tracking their velocity vectors
   for predictive trajectory extrapolation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import label

logger = logging.getLogger(__name__)


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


@dataclass
class KineticSegregationResult:
    """Result of MT/V5 kinetic figure-ground decomposition."""

    self_avatar: KineticEntity | None = None
    external_agents: list[KineticEntity] = field(default_factory=list)
    motion_energy: int = 0
    raw_delta: tuple[int, int] | None = None


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

        H, W = curr_grid.shape
        diff_mask = prev_grid != curr_grid
        motion_energy = int(np.sum(diff_mask))

        if motion_energy == 0:
            return KineticSegregationResult(motion_energy=0)

        # 4-connected morphological labeling of changed visual clusters
        labeled_diff, num_features = label(diff_mask)

        candidates: list[KineticEntity] = []
        vacated_clusters: list[dict] = []
        occupied_clusters: list[dict] = []

        for feat_id in range(1, num_features + 1):
            cluster_coords = np.argwhere(labeled_diff == feat_id)
            if len(cluster_coords) == 0:
                continue

            cells = [(int(r), int(c)) for r, c in cluster_coords]
            min_r, max_r = int(np.min(cluster_coords[:, 0])), int(np.max(cluster_coords[:, 0]))
            min_c, max_c = int(np.min(cluster_coords[:, 1])), int(np.max(cluster_coords[:, 1]))
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

            # If background_feature did not capture the substrate (e.g. floor substrate is 0 while bg is canvas color):
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

        # Match disjoint vacated clusters to occupied clusters (stride / jump motion >= entity size)
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

            # Ventral body schema gating: if known_avatar_features is established,
            # prioritize candidates that possess the avatar's visual features
            has_feat_match = known_avatar_features and any(
                bool(cand.features & known_avatar_features) for cand in candidates
            )

            for i, cand in enumerate(candidates):
                if has_feat_match and not (cand.features & known_avatar_features):
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
