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

            # If pure color swap without clear background appearance, inspect feature counts
            if not vacated or not occupied:
                # Direct centroid comparison of previous vs current non-background points in cluster
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
                else:
                    c_curr = (
                        float(np.mean(cluster_coords[:, 0])),
                        float(np.mean(cluster_coords[:, 1])),
                    )
                    velocity = (0.0, 0.0)
                    curr_cells = cells
            else:
                c_vacated = (
                    float(np.mean([p[0] for p in vacated])),
                    float(np.mean([p[1] for p in vacated])),
                )
                c_occupied = (
                    float(np.mean([p[0] for p in occupied])),
                    float(np.mean([p[1] for p in occupied])),
                )
                velocity = (c_occupied[0] - c_vacated[0], c_occupied[1] - c_vacated[1])
                c_curr = c_occupied
                curr_cells = occupied

            feats = {
                int(curr_grid[r, c]) for r, c in curr_cells if curr_grid[r, c] != background_feature
            }
            if not feats:
                feats = {int(curr_grid[r, c]) for r, c in cells}

            cand = KineticEntity(
                centroid=c_curr,
                velocity=velocity,
                cells=curr_cells,
                bounding_box=bbox,
                area=len(curr_cells),
                features=feats,
            )
            candidates.append(cand)

        # Corollary Discharge Matching:
        self_avatar: KineticEntity | None = None
        external_agents: list[KineticEntity] = []

        if commanded_delta is not None and (commanded_delta[0] != 0 or commanded_delta[1] != 0):
            exp_dr, exp_dc = float(commanded_delta[0]), float(commanded_delta[1])
            best_match_idx: int | None = None
            min_err = float("inf")

            for i, cand in enumerate(candidates):
                err = abs(cand.velocity[0] - exp_dr) + abs(cand.velocity[1] - exp_dc)
                # Feature prior bonus if matching known avatar
                if known_avatar_features and (cand.features & known_avatar_features):
                    err *= 0.5

                if err <= self.velocity_tolerance:
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
