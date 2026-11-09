"""Retrosplenial Egocentric-Allocentric Heading Integration & Head Direction Cell Attractors.

Biologically grounded in retrosplenial cortex (RSC) and thalamo-cortical head direction cells
(Taube, 2007; Byrne, Becker, & Burgess, 2007; Burgess, 2008):
1. Continuous Ring Attractor Dynamics:
   - Maintains an internal heading representation theta in [0, 2*pi) driven by vestibular / motor
     angular velocity integration:
     theta(t + dt) = (theta(t) + omega * dt) mod 2*pi.
2. Egocentric-to-Allocentric Coordinate Transformation:
   - Rotates egocentric visual vectors (rho, theta_ego) into global allocentric Cartesian displacements:
     dx = rho * cos(theta_allo + theta_ego), dy = rho * sin(theta_allo + theta_ego).
3. Landmark Bearing Recalibration:
   - Compares internal heading against allocentric visual bearings to observed landmarks,
     correcting accumulated angular drift with a circular Kalman innovation filter.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class AllocentricHeadingVector:
    """Estimated allocentric heading orientation and angular confidence."""

    heading_rad: float  # Orientation in [0, 2*pi) relative to reference North (0 rad)
    heading_degrees: float
    confidence: float  # [0.0, 1.0] where 1.0 indicates zero angular uncertainty
    angular_velocity: float  # rad/s
    drift_variance: float

    def to_causal_predicate(self) -> CausalPredicate:
        """Export allocentric heading state as a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable="allocentric_heading_vector",
            operator="==",
            value=f"deg:{self.heading_degrees:.1f}:conf:{self.confidence:.2f}",
            relation="rsc_heading_integration",
            target_entity="retrosplenial_cortex",
        )


@dataclass
class HeadDirectionAttractorState:
    """Ring attractor neural population state representing directional tuning."""

    num_bins: int
    tuning_curve: list[float]
    preferred_heading_rad: float
    bump_amplitude: float
    is_settled: bool


class RetrosplenialHeadingEngine:
    """Retrosplenial heading engine transforming egocentric vectors and tracking head direction."""

    def __init__(
        self,
        initial_heading_rad: float = 0.0,
        num_ring_bins: int = 36,
        base_drift_rate: float = 0.02,
    ) -> None:
        """Initialize retrosplenial heading engine.

        Args:
            initial_heading_rad: Starting orientation in radians [0, 2*pi).
            num_ring_bins: Angular discretization resolution for the ring attractor.
            base_drift_rate: Angular drift variance accumulated per unit time.
        """
        self.heading_rad: float = float(initial_heading_rad) % (2.0 * math.pi)
        self.num_bins: int = max(8, int(num_ring_bins))
        self.base_drift_rate: float = float(base_drift_rate)

        self.drift_variance: float = 0.01
        self.last_angular_velocity: float = 0.0
        self.total_updates: int = 0

    def integrate_angular_velocity(
        self,
        omega_rad: float,
        dt: float = 1.0,
    ) -> AllocentricHeadingVector:
        """Integrate self-motion angular velocity into the head direction attractor.

        Args:
            omega_rad: Angular rotation velocity in radians.
            dt: Time delta.

        Returns:
            AllocentricHeadingVector detailing updated heading and confidence.
        """
        self.total_updates += 1
        self.last_angular_velocity = float(omega_rad)

        # Update heading via circular dead-reckoning
        self.heading_rad = (self.heading_rad + omega_rad * dt) % (2.0 * math.pi)
        self.drift_variance = min(1.0, self.drift_variance + self.base_drift_rate * dt)

        confidence = max(0.05, 1.0 - math.sqrt(self.drift_variance))

        return AllocentricHeadingVector(
            heading_rad=self.heading_rad,
            heading_degrees=math.degrees(self.heading_rad),
            confidence=confidence,
            angular_velocity=self.last_angular_velocity,
            drift_variance=self.drift_variance,
        )

    def recalibrate_from_landmark(
        self,
        observed_bearing_rad: float,
        landmark_true_bearing_rad: float,
        kalman_gain: float = 0.40,
    ) -> AllocentricHeadingVector:
        """Recalibrate heading by comparing observed egocentric bearing to known allocentric landmark bearing.

        Formula: heading_err = (landmark_true - (heading + observed_bearing))
        """
        expected_bearing = (self.heading_rad + observed_bearing_rad) % (2.0 * math.pi)
        # Circular angular difference in [-pi, pi]
        ang_err = (landmark_true_bearing_rad - expected_bearing + math.pi) % (
            2.0 * math.pi
        ) - math.pi

        # Innovation step
        kg = max(0.05, min(0.95, float(kalman_gain)))
        self.heading_rad = (self.heading_rad + kg * ang_err) % (2.0 * math.pi)
        self.drift_variance = max(0.005, self.drift_variance * (1.0 - kg))

        confidence = max(0.05, 1.0 - math.sqrt(self.drift_variance))

        return AllocentricHeadingVector(
            heading_rad=self.heading_rad,
            heading_degrees=math.degrees(self.heading_rad),
            confidence=confidence,
            angular_velocity=self.last_angular_velocity,
            drift_variance=self.drift_variance,
        )

    def egocentric_to_allocentric(
        self,
        distance: float,
        ego_angle_rad: float,
    ) -> tuple[float, float]:
        """Convert egocentric polar vector (distance, ego_angle) to allocentric displacement (dr, dc).

        Assuming 0 rad North is along negative row axis (-r), pi/2 East is along positive col axis (+c).
        """
        total_angle = (self.heading_rad + ego_angle_rad) % (2.0 * math.pi)
        dr = -distance * math.cos(total_angle)
        dc = distance * math.sin(total_angle)
        return float(dr), float(dc)

    def allocentric_to_egocentric(
        self,
        dr: float,
        dc: float,
    ) -> tuple[float, float]:
        """Convert allocentric Cartesian displacement (dr, dc) to egocentric polar coordinates (distance, ego_angle)."""
        dist = math.sqrt(dr * dr + dc * dc)
        if dist < 1e-6:
            return 0.0, 0.0

        allo_angle = math.atan2(dc, -dr) % (2.0 * math.pi)
        ego_angle = (allo_angle - self.heading_rad + math.pi) % (2.0 * math.pi) - math.pi
        return float(dist), float(ego_angle)

    def get_attractor_population(self) -> HeadDirectionAttractorState:
        """Synthesize ring attractor population firing rates."""
        tuning: list[float] = []
        bin_width = (2.0 * math.pi) / self.num_bins
        sigma = bin_width * 1.5

        for b in range(self.num_bins):
            pref_angle = b * bin_width
            diff = (self.heading_rad - pref_angle + math.pi) % (2.0 * math.pi) - math.pi
            act = math.exp(-(diff * diff) / (2.0 * sigma * sigma))
            tuning.append(float(act))

        return HeadDirectionAttractorState(
            num_bins=self.num_bins,
            tuning_curve=tuning,
            preferred_heading_rad=self.heading_rad,
            bump_amplitude=max(tuning) if tuning else 1.0,
            is_settled=self.drift_variance < 0.20,
        )

    def reset(self) -> None:
        """Reset heading to zero and clear drift."""
        self.heading_rad = 0.0
        self.drift_variance = 0.01
        self.last_angular_velocity = 0.0
        self.total_updates = 0
