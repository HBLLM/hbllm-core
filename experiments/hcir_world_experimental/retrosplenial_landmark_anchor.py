"""Retrosplenial Path Integration Drift Correction via Visual Landmark Anchoring.

Biologically grounded in mammalian retrosplenial cortex (RSC) and postsubiculum
spatial anchoring (Jeffery et al., 2016; Bicanski & Burgess, 2018):
1. Odometric Dead-Reckoning Drift:
   - Path integration (e.g. from Entorhinal grid cells or motor efference-copies)
     accumulates spatial variance over time: sigma_odometry^2(t) = sigma_0^2 + lambda * t.
2. Allocentric Visual Landmark Anchoring:
   - RSC identifies invariant, high-stability environmental features (boundary corners,
     static walls, fixed mechanisms) as allocentric reference anchors.
3. Bayesian / Kalman Coordinate Recalibration:
   - For an observed landmark at known allocentric position L_i and relative bearing
     d_i = L_i - avatar_pos, the landmark-implied position is z_i = L_i - d_i.
   - Computes Kalman gain K = sigma_odometry^2 / (sigma_odometry^2 + sigma_landmark^2).
   - Corrects estimated position: x_hat = x_odometry + K * (z_mean - x_odometry).
   - Resets cumulative path integration drift and contracts spatial uncertainty.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AllocentricLandmark:
    """A stable spatial anchor in the allocentric coordinate frame."""

    landmark_id: str
    position: tuple[int, int]
    landmark_type: str = "static"
    stability_score: float = 1.0  # [0, 1] confidence in spatial immobility
    observed_count: int = 1


@dataclass
class LandmarkRecalibrationResult:
    """The outcome of an RSC landmark-based coordinate recalibration."""

    odometry_position: tuple[int, int]
    corrected_position: tuple[int, int]
    drift_vector: tuple[float, float]
    kalman_gain: float
    confidence: float
    recalibrated_landmark_ids: list[str]
    variance_before: float
    variance_after: float

    @property
    def has_drift_correction(self) -> bool:
        """Return True if significant spatial drift was detected and corrected."""
        return abs(self.drift_vector[0]) > 0.05 or abs(self.drift_vector[1]) > 0.05


class RetrosplenialLandmarkAnchorEngine:
    """Retrosplenial anchoring engine recalibrating path integration using visual landmarks."""

    def __init__(
        self,
        base_landmark_variance: float = 0.2,
        initial_odometry_variance: float = 0.5,
        drift_rate_per_step: float = 0.05,
    ) -> None:
        """Initialize retrosplenial landmark anchor engine.

        Args:
            base_landmark_variance: Measurement noise variance for high-stability landmarks.
            initial_odometry_variance: Starting variance of dead-reckoning state.
            drift_rate_per_step: Linear variance growth per odometric step without visual anchor.
        """
        self.base_landmark_variance = max(0.01, float(base_landmark_variance))
        self.odometry_variance = max(0.01, float(initial_odometry_variance))
        self.initial_odometry_variance = self.odometry_variance
        self.drift_rate_per_step = max(0.001, float(drift_rate_per_step))

        self.landmarks: dict[str, AllocentricLandmark] = {}
        self.total_corrections: int = 0

    def register_landmark(
        self,
        landmark_id: str,
        position: tuple[int, int],
        landmark_type: str = "static",
        stability: float = 1.0,
    ) -> AllocentricLandmark:
        """Register or update an allocentric landmark anchor."""
        if landmark_id in self.landmarks:
            lm = self.landmarks[landmark_id]
            lm.observed_count += 1
            lm.stability_score = min(1.0, lm.stability_score * 0.9 + stability * 0.1)
            return lm

        lm = AllocentricLandmark(
            landmark_id=landmark_id,
            position=position,
            landmark_type=landmark_type,
            stability_score=max(0.1, min(1.0, float(stability))),
            observed_count=1,
        )
        self.landmarks[landmark_id] = lm
        return lm

    def register_landmarks_from_percept(
        self,
        entities: list[Any],
    ) -> list[AllocentricLandmark]:
        """Automatically register stable static entities as landmarks."""
        registered: list[AllocentricLandmark] = []
        for ent in entities:
            # Handle SpatialEntity or dict
            role = getattr(ent, "role", None) or (ent.get("role") if isinstance(ent, dict) else "")
            eid = getattr(ent, "id", None) or (ent.get("id") if isinstance(ent, dict) else "")
            pos = getattr(ent, "grid_pos", None) or (
                ent.get("grid_pos") if isinstance(ent, dict) else None
            )
            if pos is None:
                pos = getattr(ent, "centroid", None) or (
                    ent.get("centroid") if isinstance(ent, dict) else None
                )

            if pos is None or not eid:
                continue

            r, c = int(pos[0]), int(pos[1])
            role_str = str(role).lower()

            # Mark barriers, walls, and switches as stable landmarks
            if any(k in role_str for k in ("barrier", "wall", "switch", "door", "goal", "static")):
                lm = self.register_landmark(
                    landmark_id=str(eid),
                    position=(r, c),
                    landmark_type=role_str,
                    stability=0.95,
                )
                registered.append(lm)

        return registered

    def step_odometry(self, steps: int = 1) -> float:
        """Accumulate dead-reckoning drift variance when moving without recalibration."""
        self.odometry_variance += self.drift_rate_per_step * max(1, steps)
        return self.odometry_variance

    def recalibrate(
        self,
        odometry_pos: tuple[int, int],
        landmark_observations: list[tuple[str, tuple[int, int]]],
    ) -> LandmarkRecalibrationResult:
        """Recalibrate spatial position against visual landmark observations.

        Args:
            odometry_pos: Current estimated position from dead reckoning (r, c).
            landmark_observations: List of (landmark_id, relative_vector_from_avatar)
                where relative_vector = landmark_pos - avatar_pos.

        Returns:
            LandmarkRecalibrationResult containing corrected coordinates and drift.
        """
        if not landmark_observations or not self.landmarks:
            return LandmarkRecalibrationResult(
                odometry_position=odometry_pos,
                corrected_position=odometry_pos,
                drift_vector=(0.0, 0.0),
                kalman_gain=0.0,
                confidence=1.0 / (1.0 + self.odometry_variance),
                recalibrated_landmark_ids=[],
                variance_before=self.odometry_variance,
                variance_after=self.odometry_variance,
            )

        implied_r_list: list[float] = []
        implied_c_list: list[float] = []
        used_ids: list[str] = []
        effective_landmark_variances: list[float] = []

        for lm_id, rel_offset in landmark_observations:
            if lm_id not in self.landmarks:
                continue

            lm = self.landmarks[lm_id]
            used_ids.append(lm_id)

            # Allocentric position implied by landmark: avatar = landmark - relative_offset
            # where rel_offset = (lm_r - avatar_r, lm_c - avatar_c)
            implied_r = lm.position[0] - rel_offset[0]
            implied_c = lm.position[1] - rel_offset[1]
            implied_r_list.append(float(implied_r))
            implied_c_list.append(float(implied_c))

            # Variance scaled inversely with landmark stability and observation count
            lm_var = self.base_landmark_variance / (
                lm.stability_score * math.log(1.0 + lm.observed_count)
            )
            effective_landmark_variances.append(max(0.01, lm_var))

        if not used_ids:
            return LandmarkRecalibrationResult(
                odometry_position=odometry_pos,
                corrected_position=odometry_pos,
                drift_vector=(0.0, 0.0),
                kalman_gain=0.0,
                confidence=1.0 / (1.0 + self.odometry_variance),
                recalibrated_landmark_ids=[],
                variance_before=self.odometry_variance,
                variance_after=self.odometry_variance,
            )

        # Fused landmark estimate (precision-weighted average)
        precisions = [1.0 / v for v in effective_landmark_variances]
        total_prec = sum(precisions)
        mean_implied_r = sum(r * p for r, p in zip(implied_r_list, precisions)) / total_prec
        mean_implied_c = sum(c * p for c, p in zip(implied_c_list, precisions)) / total_prec
        fused_meas_variance = 1.0 / total_prec

        # Compute Kalman Gain: K = var_odo / (var_odo + var_meas)
        var_before = self.odometry_variance
        k = var_before / (var_before + fused_meas_variance)

        # Correct position toward fused landmark observation
        odo_r, odo_c = float(odometry_pos[0]), float(odometry_pos[1])
        corr_r = odo_r + k * (mean_implied_r - odo_r)
        corr_c = odo_c + k * (mean_implied_c - odo_c)

        # New contracted variance
        var_after = (1.0 - k) * var_before
        self.odometry_variance = max(self.initial_odometry_variance * 0.5, var_after)
        self.total_corrections += 1

        drift_r = corr_r - odo_r
        drift_c = corr_c - odo_c

        return LandmarkRecalibrationResult(
            odometry_position=odometry_pos,
            corrected_position=(int(round(corr_r)), int(round(corr_c))),
            drift_vector=(drift_r, drift_c),
            kalman_gain=float(k),
            confidence=1.0 / (1.0 + var_after),
            recalibrated_landmark_ids=used_ids,
            variance_before=var_before,
            variance_after=var_after,
        )

    def reset(self) -> None:
        """Reset internal landmarks and variance to default state."""
        self.landmarks.clear()
        self.odometry_variance = self.initial_odometry_variance
        self.total_corrections = 0
