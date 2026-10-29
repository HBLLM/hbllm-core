"""Entorhinal Border Cells & Geometry-Guided Grid Module Rescaling.

Biologically grounded in medial entorhinal cortex (MEC) border cells
(Solstad et al., 2008; Barry et al., 2007; Giocomo et al., 2011):
1. Boundary-Specific Border Firing:
   - Border cells fire specifically along geometrical boundaries (North, South, East, West walls),
     with activity falling off sharply with distance from the boundary:
     f(d) = exp(-d^2 / (2 * sigma^2)).
2. Grid Module Wavelength Rescaling:
   - When an environmental enclosure is deformed, stretched, or compressed, border cells
     anchor the grid cell spatial phase and dynamically rescale the grid module
     spatial wavelengths lambda_i along principal axes:
     s_r = H_curr / H_ref, s_c = W_curr / W_ref.
3. Path-Integration Drift Reset:
   - Contact with a detected border immediately clamps and zeroes accumulated dead-reckoning
     odometric phase errors along the normal vector to the wall.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class BorderCell:
    """A medial entorhinal cortex neuron tuned to a specific environmental boundary orientation."""

    cell_id: str
    wall_orientation: str  # "north", "south", "east", "west"
    normal_vector: tuple[float, float]  # Unit vector pointing away from wall into room
    distance_threshold: float = 1.5  # Max distance in grid units for active firing
    spatial_sigma: float = 0.80

    def compute_response(self, distance_to_wall: float) -> float:
        """Compute Gaussian border firing response as a function of wall distance."""
        if distance_to_wall > self.distance_threshold:
            return 0.0
        dist_diff = max(0.0, float(distance_to_wall))
        return float(
            math.exp(-(dist_diff * dist_diff) / (2.0 * self.spatial_sigma * self.spatial_sigma))
        )


@dataclass
class GridModuleRescalingResult:
    """Outcome of geometry-guided entorhinal grid module wavelength adaptation."""

    scale_r: float  # Vertical grid wavelength scaling factor
    scale_c: float  # Horizontal grid wavelength scaling factor
    active_borders: list[str]
    phase_drift_reset: bool
    rescaled_wavelengths: list[float] = field(default_factory=list)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export grid module rescaling state as a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable="grid_module_rescaling",
            operator="==",
            value=f"scale_r:{self.scale_r:.2f}:scale_c:{self.scale_c:.2f}:borders:{len(self.active_borders)}",
            relation="entorhinal_border_anchoring",
            target_entity="entorhinal_cortex",
        )


class EntorhinalBorderCellEngine:
    """MEC border cell engine anchoring spatial phases and dynamically rescaling grid modules."""

    def __init__(
        self,
        reference_dimensions: tuple[float, float] = (10.0, 10.0),
        base_wavelengths: list[float] | None = None,
        border_threshold: float = 1.5,
    ) -> None:
        """Initialize entorhinal border cell engine.

        Args:
            reference_dimensions: Standard baseline room dimensions (H_ref, W_ref).
            base_wavelengths: Baseline grid module spatial periodicities in grid units.
            border_threshold: Proximity distance triggering border cell firing.
        """
        self.ref_h = max(2.0, float(reference_dimensions[0]))
        self.ref_w = max(2.0, float(reference_dimensions[1]))
        self.base_wavelengths = base_wavelengths or [2.5, 4.0, 6.5, 10.0]
        self.border_threshold = float(border_threshold)

        # Initialize canonical 4-wall border cell population
        self.border_cells: dict[str, BorderCell] = {
            "north": BorderCell("bc_north", "north", (1.0, 0.0), self.border_threshold),
            "south": BorderCell("bc_south", "south", (-1.0, 0.0), self.border_threshold),
            "east": BorderCell("bc_east", "east", (0.0, -1.0), self.border_threshold),
            "west": BorderCell("bc_west", "west", (0.0, 1.0), self.border_threshold),
        }
        self.accumulated_drift: tuple[float, float] = (0.0, 0.0)
        self.total_rescalings: int = 0

    def evaluate_borders_and_rescaling(
        self,
        boundary_grid: np.ndarray,
        avatar_pos: tuple[int, int],
    ) -> GridModuleRescalingResult:
        """Evaluate border cell activations and compute geometric grid module rescaling.

        Args:
            boundary_grid: 2D binary grid where 1 = wall/boundary, 0 = traversable.
            avatar_pos: Current agent coordinates (r, c).

        Returns:
            GridModuleRescalingResult containing axis scale factors and phase resets.
        """
        self.total_rescalings += 1
        h, w = boundary_grid.shape
        ar, ac = avatar_pos

        # Measure ray distances to nearest perimeter walls in 4 cardinal directions
        # North (up): decreasing r
        d_north = float(ar)
        for r in range(ar - 1, -1, -1):
            if boundary_grid[r, ac] > 0:
                d_north = float(ar - r - 1)
                break

        # South (down): increasing r
        d_south = float(h - 1 - ar)
        for r in range(ar + 1, h):
            if boundary_grid[r, ac] > 0:
                d_south = float(r - ar - 1)
                break

        # West (left): decreasing c
        d_west = float(ac)
        for c in range(ac - 1, -1, -1):
            if boundary_grid[ar, c] > 0:
                d_west = float(ac - c - 1)
                break

        # East (right): increasing c
        d_east = float(w - 1 - ac)
        for c in range(ac + 1, w):
            if boundary_grid[ar, c] > 0:
                d_east = float(c - ac - 1)
                break

        wall_dists = {
            "north": d_north,
            "south": d_south,
            "west": d_west,
            "east": d_east,
        }

        # Check firing of border cells
        active_borders: list[str] = []
        phase_reset = False

        for b_name, cell in self.border_cells.items():
            dist = wall_dists[b_name]
            resp = cell.compute_response(dist)
            if resp > 0.35:
                active_borders.append(b_name)
                phase_reset = True

        if phase_reset:
            # Border contact zeroes out accumulated dead-reckoning drift
            self.accumulated_drift = (0.0, 0.0)

        # Compute dynamic wavelength rescaling relative to reference room dimensions
        scale_r = float(h) / self.ref_h
        scale_c = float(w) / self.ref_w

        # Rescale grid periodicities along principal geometric axes
        rescaled_wavelengths = [
            float(w0 * math.sqrt(scale_r * scale_c)) for w0 in self.base_wavelengths
        ]

        return GridModuleRescalingResult(
            scale_r=scale_r,
            scale_c=scale_c,
            active_borders=active_borders,
            phase_drift_reset=phase_reset,
            rescaled_wavelengths=rescaled_wavelengths,
        )

    def record_odometric_step(self, step_disp: tuple[float, float], noise: float = 0.05) -> None:
        """Accumulate dead reckoning drift during ungrounded navigation."""
        dr = self.accumulated_drift[0] + step_disp[0] * noise
        dc = self.accumulated_drift[1] + step_disp[1] * noise
        self.accumulated_drift = (dr, dc)

    def reset(self) -> None:
        """Reset internal drift and counters."""
        self.accumulated_drift = (0.0, 0.0)
        self.total_rescalings = 0
