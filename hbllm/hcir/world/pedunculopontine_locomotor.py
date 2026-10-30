"""Pedunculopontine Nucleus (PPN / MLR) Locomotor Velocity Modulation & Obstacle Deceleration.

Biological Foundations:
- Garcia-Rill (1991): "The pedunculopontine nucleus" (Prog Neurobiol 36: 267-289).
- Takakusaki (2017): "Functional neuroanatomy for posture and locomotion controls"
  (Neurosci Res 121: 1-17).
- Shik, Severin, Orlovsky (1966): "Control of walking and running by means of electrical
  stimulation of the mid-brain" (Biophysics 11: 756-765) — Mesencephalic Locomotor Region (MLR).

Key Cognitive Capabilities:
1. Graded Locomotor Drive & Velocity Scaling:
   - Modulates motor execution velocity continuously from cautious creep/crawl to normal
     cruise to burst sprinting based on target distance and cognitive urgency.
2. Anticipatory Proximity Deceleration:
   - Anticipates approaching boundaries, obstacles, or sharp corners, applying smooth
     algebraic braking curves to eliminate momentum overshoots.
3. Terrain Friction Adaptation:
   - Couples with somatosensory haptic friction (mu) to clamp velocity ceilings on icy or
     slippery surfaces, preventing traction loss and drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from hbllm.hcir.world.causal_discovery import CausalPredicate


class LocomotorPacingRegime(str, Enum):
    """Pacing regimes of the pedunculopontine locomotor drive."""

    CREEP_CAUTIOUS = "creep_cautious"
    NORMAL_CRUISE = "normal_cruise"
    ACCELERATED_BURST = "accelerated_burst"
    DECELERATING_BRAKE = "decelerating_brake"


@dataclass(frozen=True)
class LocomotorDriveCommand:
    """Continuous locomotor command emitted by the pedunculopontine nucleus."""

    velocity: float
    pacing_regime: LocomotorPacingRegime
    step_scale: float
    obstacle_distance: float
    friction_coefficient: float
    deceleration_reason: str

    def to_causal_predicate(self) -> CausalPredicate:
        """Convert locomotor command into an HCIR CausalPredicate."""
        return CausalPredicate(
            variable="ppn_locomotor_drive",
            operator="==",
            value=f"regime:{self.pacing_regime.value}:vel:{self.velocity:.2f}:dist:{self.obstacle_distance:.1f}:fric:{self.friction_coefficient:.2f}",
            relation="ppn_locomotor_velocity_modulation",
            target_entity="pedunculopontine_nucleus",
        )


class PedunculopontineLocomotorEngine:
    """Orchestrator for pedunculopontine locomotor pacing and anticipatory deceleration."""

    def __init__(
        self,
        nominal_velocity: float = 1.0,
        min_velocity: float = 0.20,
        braking_distance: float = 2.5,
        deceleration_exponent: float = 1.20,
    ) -> None:
        """Initialize pedunculopontine locomotor engine.

        Args:
            nominal_velocity: Standard cruise velocity in [0.1, 1.5].
            min_velocity: Floor velocity for cautious approach (never 0 unless stopped).
            braking_distance: Spatial distance threshold under which deceleration begins.
            deceleration_exponent: Exponent shaping the deceleration curve.
        """
        self.nominal_velocity: float = max(0.1, float(nominal_velocity))
        self.min_velocity: float = max(0.05, float(min_velocity))
        self.braking_distance: float = max(0.5, float(braking_distance))
        self.deceleration_exponent: float = max(0.5, float(deceleration_exponent))

    def modulate_locomotor_velocity(
        self,
        distance_to_obstacle: float,
        surface_friction: float = 1.0,
        urgency_score: float = 0.0,
        is_narrow_aperture: bool = False,
    ) -> LocomotorDriveCommand:
        """Compute graded locomotor velocity and pacing regime.

        Args:
            distance_to_obstacle: Metric distance to the closest barrier or target in direction of travel.
            surface_friction: Haptic friction coefficient mu in [0.05, 1.0].
            urgency_score: Cognitive urgency or threat escape drive in [0.0, 1.0].
            is_narrow_aperture: Whether entering a 1-cell corridor or tight opening.

        Returns:
            LocomotorDriveCommand detailing recommended velocity, scale, and regime.
        """
        dist = max(0.0, float(distance_to_obstacle))
        friction = max(0.05, min(1.0, float(surface_friction)))
        urgency = max(0.0, min(1.0, float(urgency_score)))

        # 1. Base Target Velocity (Urgency can push into accelerated burst)
        base_v = self.nominal_velocity * (1.0 + 0.50 * urgency)

        # 2. Friction Clamping: low friction (ice) limits max safe velocity
        safe_friction_ceiling = math.sqrt(friction) * 1.20
        v_clamped = min(base_v, safe_friction_ceiling)

        # 3. Proximity Deceleration Curve
        reason = ""
        regime = LocomotorPacingRegime.NORMAL_CRUISE

        if dist <= self.braking_distance:
            # Smooth deceleration: (dist / braking_distance)^exponent
            ratio = dist / self.braking_distance
            attenuation = math.pow(ratio, self.deceleration_exponent)
            v_effective = max(self.min_velocity, v_clamped * attenuation)
            regime = LocomotorPacingRegime.DECELERATING_BRAKE
            reason = f"proximity_braking:dist={dist:.2f}"
        elif is_narrow_aperture:
            v_effective = max(self.min_velocity, v_clamped * 0.60)
            regime = LocomotorPacingRegime.CREEP_CAUTIOUS
            reason = "narrow_aperture_navigation"
        elif friction < 0.25:
            v_effective = max(self.min_velocity, v_clamped * 0.50)
            regime = LocomotorPacingRegime.CREEP_CAUTIOUS
            reason = f"low_friction_slip_hazard:mu={friction:.2f}"
        elif urgency > 0.60:
            v_effective = v_clamped
            regime = LocomotorPacingRegime.ACCELERATED_BURST
            reason = "high_urgency_burst"
        else:
            v_effective = v_clamped
            regime = LocomotorPacingRegime.NORMAL_CRUISE

        step_scale = max(0.1, min(1.5, v_effective / self.nominal_velocity))

        return LocomotorDriveCommand(
            velocity=float(v_effective),
            pacing_regime=regime,
            step_scale=float(step_scale),
            obstacle_distance=float(dist),
            friction_coefficient=float(friction),
            deceleration_reason=reason,
        )
