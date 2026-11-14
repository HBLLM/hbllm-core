"""Hippocampal-Entorhinal Boundary-to-Grid Phase Alignment & Error Innovation.

Biological Foundations:
- Bush, Barry, Manson, Burgess (2015): "Using grid cells to put object-related vector
  coding on the map" (Front Neural Circuits 9: 36).
- Giocomo, Moser, Moser (2011): "Computational models of grid cells" (Neuron 71: 589-603).
- Hardcastle, Ganguli, Giocomo (2015): "Environmental boundaries as an error correction
  mechanism for grid cells" (Neuron 86: 827-839).

Key Cognitive Capabilities:
1. Boundary-Guided Grid Phase Reset:
   - Anchors multi-scale grid firing peaks (r_grid, c_grid) against detected subicular boundary
     vectors, eliminating accumulated path integration / odometric drift.
2. Kalman Error Innovation Vector:
   - Computes sensory innovation residual: innovation = observed_boundary_distance - expected_distance.
   - Applies optimal Kalman gain K to correct allocentric position and grid phase coordinates.
3. Boundary Recalibration Confidence:
   - Quantifies boundary proximity and certainty; recalibration is strongest when adjacent to
     rigid allocentric perimeters, decaying gracefully in open arena centers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hbllm.hcir.world.causal_discovery import CausalPredicate


@dataclass(frozen=True)
class PhaseInnovationVector:
    """Kalman innovation residual and phase correction vector."""

    boundary_axis: str  # 'north', 'south', 'east', 'west'
    observed_distance: float
    expected_distance: float
    innovation_residual: float
    kalman_gain: float
    phase_correction: float


@dataclass(frozen=True)
class BoundaryPhaseAlignmentState:
    """Consolidated boundary-to-grid phase alignment snapshot."""

    raw_position: tuple[float, float]
    corrected_position: tuple[float, float]
    total_drift_corrected: float
    alignment_confidence: float
    innovations: tuple[PhaseInnovationVector, ...]

    def to_causal_predicate(self) -> CausalPredicate:
        """Convert alignment state into an HCIR CausalPredicate."""
        return CausalPredicate(
            variable="boundary_grid_phase_alignment",
            operator="==",
            value=f"corr_r:{self.corrected_position[0]:.2f}:corr_c:{self.corrected_position[1]:.2f}:drift:{self.total_drift_corrected:.3f}:conf:{self.alignment_confidence:.2f}",
            relation="entorhinal_boundary_recalibration",
            target_entity="entorhinal_grid_cortex",
        )


class BoundaryGridPhaseAligner:
    """Orchestrator for boundary-to-grid phase alignment and odometric drift reset."""

    def __init__(
        self,
        base_kalman_gain: float = 0.40,
        max_correction_step: float = 2.0,
        proximity_threshold: float = 4.0,
    ) -> None:
        """Initialize boundary-grid phase aligner.

        Args:
            base_kalman_gain: Baseline Kalman gain applied to innovation residuals.
            max_correction_step: Maximum single-step coordinate correction clamping.
            proximity_threshold: Distance threshold from boundary for valid recalibration.
        """
        self.base_kalman_gain: float = max(0.05, min(1.0, float(base_kalman_gain)))
        self.max_correction_step: float = max(0.1, float(max_correction_step))
        self.proximity_threshold: float = max(1.0, float(proximity_threshold))

    def align_grid_phase_with_boundaries(
        self,
        estimated_pos: tuple[float, float],
        grid_dimensions: tuple[int, int],
        observed_boundary_distances: dict[str, float],
    ) -> BoundaryPhaseAlignmentState:
        """Align estimated avatar / grid coordinates against observed environmental boundaries.

        Args:
            estimated_pos: Current path-integrated dead-reckoning position (row, col).
            grid_dimensions: Arena dimensions (height_rows, width_cols).
            observed_boundary_distances: Map of detected wall distances:
                {'north': dist_r0, 'south': dist_rMax, 'west': dist_c0, 'east': dist_cMax}.

        Returns:
            BoundaryPhaseAlignmentState with Kalman-corrected coordinates and innovation metrics.
        """
        r, c = float(estimated_pos[0]), float(estimated_pos[1])
        h, w = int(grid_dimensions[0]), int(grid_dimensions[1])

        # Compute expected boundary distances given current estimated position
        expected_distances = {
            "north": max(0.0, r),
            "south": max(0.0, (h - 1) - r),
            "west": max(0.0, c),
            "east": max(0.0, (w - 1) - c),
        }

        innovations: list[PhaseInnovationVector] = []
        delta_r: float = 0.0
        delta_c: float = 0.0
        active_weights: float = 0.0

        for axis in ["north", "south", "west", "east"]:
            if axis not in observed_boundary_distances:
                continue

            obs = float(observed_boundary_distances[axis])
            if obs > self.proximity_threshold:
                continue  # Beyond sensory confidence threshold

            exp = expected_distances[axis]
            residual = obs - exp

            # Distance-dependent Kalman gain (stronger closer to boundary)
            proximity_weight = max(0.0, 1.0 - (obs / self.proximity_threshold))
            kg = self.base_kalman_gain * (1.0 + proximity_weight)

            # Map residual to coordinate correction
            if axis == "north":
                # obs > exp implies actual r is larger than estimated
                corr = kg * residual
                delta_r += corr
            elif axis == "south":
                # obs > exp implies actual r is smaller than estimated
                corr = -kg * residual
                delta_r += corr
            elif axis == "west":
                corr = kg * residual
                delta_c += corr
            elif axis == "east":
                corr = -kg * residual
                delta_c += corr
            else:
                corr = 0.0

            innovations.append(
                PhaseInnovationVector(
                    boundary_axis=axis,
                    observed_distance=obs,
                    expected_distance=exp,
                    innovation_residual=residual,
                    kalman_gain=kg,
                    phase_correction=corr,
                )
            )
            active_weights += proximity_weight

        # Clamp single-step correction
        clamped_dr = max(-self.max_correction_step, min(self.max_correction_step, delta_r))
        clamped_dc = max(-self.max_correction_step, min(self.max_correction_step, delta_c))

        new_r = max(0.0, min(float(h - 1), r + clamped_dr))
        new_c = max(0.0, min(float(w - 1), c + clamped_dc))

        total_drift = math.hypot(clamped_dr, clamped_dc)
        confidence = float(min(1.0, 0.40 + 0.60 * (active_weights / 2.0))) if innovations else 0.20

        return BoundaryPhaseAlignmentState(
            raw_position=(r, c),
            corrected_position=(new_r, new_c),
            total_drift_corrected=total_drift,
            alignment_confidence=confidence,
            innovations=tuple(innovations),
        )
