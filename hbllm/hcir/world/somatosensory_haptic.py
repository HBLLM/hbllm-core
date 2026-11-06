"""Secondary Somatosensory (S2) & Posterior Insular Haptic Surface Friction Integration.

Biologically grounded in S2 and posterior dorsal insular haptic perception
(Craig, 2002; Kaas, 2004; Johansen-Berg et al., 2000):
1. Tactile Surface Impedance & Friction Inference:
   - Compares commanded motor displacement against sensed kinesthetic displacement
     to infer local surface friction coefficients mu in [0.0, 2.0], surface drag, and slippage.
2. Terrain Archetype Classification:
   - Categorizes surfaces into physical archetypes:
     "ice_slippery" (low friction, high slip), "normal_terrain" (optimal traction),
     "viscous_mud" (high drag, decelerated motion), "conveyor_assist" (directional bias).
3. Motor Impulse & Braking Calibration:
   - Dynamically recalibrates motor thrust and deceleration distance predictions,
     preventing overshooting on slippery surfaces or stall on viscous terrain.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class SurfaceFrictionProfile:
    """Estimated physical friction and tactile properties of a surface terrain."""

    terrain_type: str  # e.g., "normal_terrain", "ice_slippery", "viscous_drag", "adhesive_sticky"
    friction_coefficient: float  # mu in [0.05, 2.0]
    drag_factor: float  # [0.0, 2.0] resistive drag
    slip_probability: float  # [0.0, 1.0] probability of uncommanded inertia slip
    traction_efficiency: float  # [0.0, 1.0] effective motor power transfer

    def to_causal_predicate(self) -> CausalPredicate:
        """Export surface friction profile as a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable="surface_friction_profile",
            operator="==",
            value=f"terrain:{self.terrain_type}:mu:{self.friction_coefficient:.2f}:traction:{self.traction_efficiency:.2f}",
            relation="haptic_surface_impedance",
            target_entity="somatosensory_cortex",
        )


@dataclass
class HapticAffordanceFeedback:
    """Feedback emitted following a motor action execution over a surface."""

    position: tuple[int, int]
    expected_displacement: tuple[float, float]
    actual_displacement: tuple[float, float]
    displacement_ratio: float  # |actual| / |expected|
    slip_detected: bool
    drag_detected: bool
    inferred_profile: SurfaceFrictionProfile


class SomatosensoryHapticEngine:
    """Secondary somatosensory engine estimating surface friction and tactile impedances."""

    def __init__(
        self,
        default_friction: float = 0.70,
        learning_rate: float = 0.25,
    ) -> None:
        """Initialize somatosensory haptic engine.

        Args:
            default_friction: Baseline friction coefficient for unvisited terrain.
            learning_rate: Online update rate for terrain tactile parameters.
        """
        self.default_friction = float(default_friction)
        self.learning_rate = max(0.01, min(1.0, float(learning_rate)))

        # Position (r, c) -> SurfaceFrictionProfile
        self.terrain_map: dict[tuple[int, int], SurfaceFrictionProfile] = {}
        # Global terrain archetype beliefs
        self.canonical_archetypes: dict[str, SurfaceFrictionProfile] = {
            "normal_terrain": SurfaceFrictionProfile(
                terrain_type="normal_terrain",
                friction_coefficient=0.70,
                drag_factor=0.10,
                slip_probability=0.05,
                traction_efficiency=0.95,
            ),
            "ice_slippery": SurfaceFrictionProfile(
                terrain_type="ice_slippery",
                friction_coefficient=0.10,
                drag_factor=0.02,
                slip_probability=0.85,
                traction_efficiency=0.40,
            ),
            "viscous_drag": SurfaceFrictionProfile(
                terrain_type="viscous_drag",
                friction_coefficient=1.40,
                drag_factor=0.80,
                slip_probability=0.0,
                traction_efficiency=0.55,
            ),
            "adhesive_sticky": SurfaceFrictionProfile(
                terrain_type="adhesive_sticky",
                friction_coefficient=1.80,
                drag_factor=0.95,
                slip_probability=0.0,
                traction_efficiency=0.30,
            ),
        }
        self.total_evaluations: int = 0

    def process_kinesthetic_feedback(
        self,
        position: tuple[int, int],
        expected_displacement: tuple[float, float],
        actual_displacement: tuple[float, float],
    ) -> HapticAffordanceFeedback:
        """Compare expected displacement with actual displacement to infer surface friction.

        Args:
            position: Grid coordinate (r, c) where the motor interaction occurred.
            expected_displacement: Predicted motor delta (dr, dc).
            actual_displacement: Realized sensory displacement (dr, dc).

        Returns:
            HapticAffordanceFeedback containing inferred friction profile and tactile anomalies.
        """
        self.total_evaluations += 1
        exp_mag = math.sqrt(expected_displacement[0] ** 2 + expected_displacement[1] ** 2)
        act_mag = math.sqrt(actual_displacement[0] ** 2 + actual_displacement[1] ** 2)

        ratio = (act_mag / exp_mag) if exp_mag > 1e-6 else 1.0

        # Detect slippage (actual displacement exceeds expected or exhibits lateral drift)
        slip = ratio > 1.25 or (exp_mag > 0.0 and act_mag > exp_mag + 0.2)
        # Detect viscous drag (actual displacement significantly falls short of expected)
        drag = ratio < 0.70 and exp_mag > 0.0

        # Infer friction coefficient
        if slip:
            inferred_mu = max(0.05, 0.25 / ratio)
            inferred_type = "ice_slippery"
            slip_prob = 0.80
            traction = max(0.2, 1.0 - (ratio - 1.0))
            drag_f = 0.05
        elif drag:
            inferred_mu = min(2.0, 1.0 + (1.0 - ratio))
            inferred_type = "viscous_drag" if ratio > 0.40 else "adhesive_sticky"
            slip_prob = 0.0
            traction = max(0.2, ratio)
            drag_f = min(1.0, 1.0 - ratio)
        else:
            inferred_mu = self.default_friction
            inferred_type = "normal_terrain"
            slip_prob = 0.05
            traction = 0.95
            drag_f = 0.10

        # Update per-cell map with exponential moving average
        if position in self.terrain_map:
            prior = self.terrain_map[position]
            new_mu = (
                1.0 - self.learning_rate
            ) * prior.friction_coefficient + self.learning_rate * inferred_mu
            new_traction = (
                1.0 - self.learning_rate
            ) * prior.traction_efficiency + self.learning_rate * traction
            new_drag = (1.0 - self.learning_rate) * prior.drag_factor + self.learning_rate * drag_f
            profile = SurfaceFrictionProfile(
                terrain_type=inferred_type,
                friction_coefficient=float(new_mu),
                drag_factor=float(new_drag),
                slip_probability=slip_prob,
                traction_efficiency=float(new_traction),
            )
        else:
            profile = SurfaceFrictionProfile(
                terrain_type=inferred_type,
                friction_coefficient=float(inferred_mu),
                drag_factor=float(drag_f),
                slip_probability=slip_prob,
                traction_efficiency=float(traction),
            )

        self.terrain_map[position] = profile

        return HapticAffordanceFeedback(
            position=position,
            expected_displacement=expected_displacement,
            actual_displacement=actual_displacement,
            displacement_ratio=float(ratio),
            slip_detected=slip,
            drag_detected=drag,
            inferred_profile=profile,
        )

    def get_surface_profile_at(self, position: tuple[int, int]) -> SurfaceFrictionProfile:
        """Query haptic friction profile at specified coordinates."""
        return self.terrain_map.get(position, self.canonical_archetypes["normal_terrain"])

    def calibrate_braking_distance(
        self,
        velocity: float,
        position: tuple[int, int],
    ) -> float:
        """Calculate required stopping / braking distance given local surface friction.

        Formula: d = v^2 / (2 * mu * g), normalized with g = 1.0.
        """
        profile = self.get_surface_profile_at(position)
        mu = max(0.05, profile.friction_coefficient)
        stopping_dist = (velocity * velocity) / (2.0 * mu)
        return float(stopping_dist)

    def reset(self) -> None:
        """Clear dynamic terrain map."""
        self.terrain_map.clear()
        self.total_evaluations = 0
