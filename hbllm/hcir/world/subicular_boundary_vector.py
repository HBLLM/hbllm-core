"""Subicular Egocentric-to-Allocentric Boundary Vector Cell (BVC) Map.

Biologically grounded in subiculum boundary vector cell (BVC) tuning properties
(Lever, Wills, Cacucci, & Burgess, 2009; Barry et al., 2006):
1. Topographic Boundary Vector Tuning:
   - Individual BVCs fire whenever an environmental barrier or room boundary
     lies at a preferred egocentric distance d_i and allocentric orientation theta_i:
     g_i(r, theta) = exp(-(r - d_i)^2 / (2*sigma_rad^2)) * exp(-(theta - theta_i)^2 / (2*sigma_ang^2)).
   - Radial receptive field width scales with distance: sigma_rad(d_i) = sigma_0 * (1 + alpha * d_i).
2. Population Boundary Vector Map:
   - A distributed population of BVCs (multi-distance, multi-orientation) synthesizes
     a continuous 2D boundary proximity map around the agent.
3. Zero-Shot Place Field Remapping:
   - BVC firing provides the foundational sensory inputs that drive hippocampal place cells.
   - When room boundaries deform, translate, or expand, BVC population tuning
     predicts place field deformations without sensory disorientation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class BoundaryVectorCell:
    """A subicular neuron tuned to an environmental boundary at distance d and angle theta."""

    cell_id: str
    preferred_distance: float  # Preferred distance in grid units
    preferred_angle_rad: float  # Allocentric orientation [0, 2*pi)
    radial_tuning: float = 1.0  # Base radial tuning width sigma_0
    angular_tuning: float = math.pi / 4.0  # Angular tuning width sigma_ang

    def response(self, distance: float, angle_rad: float) -> float:
        """Compute Gaussian tuning response for a detected boundary point."""
        # Scale radial width with preferred distance
        sigma_r = self.radial_tuning * (1.0 + 0.15 * self.preferred_distance)
        dist_diff = distance - self.preferred_distance
        radial_term = math.exp(-(dist_diff * dist_diff) / (2.0 * sigma_r * sigma_r))

        # Circular angular difference
        ang_diff = (angle_rad - self.preferred_angle_rad + math.pi) % (2.0 * math.pi) - math.pi
        angular_term = math.exp(
            -(ang_diff * ang_diff) / (2.0 * self.angular_tuning * self.angular_tuning)
        )

        return float(radial_term * angular_term)


@dataclass
class BoundaryVectorMap:
    """The unified 2D boundary proximity and activation landscape produced by BVCs."""

    activation_grid: np.ndarray
    active_bvc_count: int
    nearest_boundary_distance: float
    dominant_orientation_rad: float
    bvc_activations: dict[str, float] = field(default_factory=dict)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export boundary vector map state as a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable="subicular_boundary_vector",
            operator="==",
            value=f"near_dist:{self.nearest_boundary_distance:.2f}:angle:{math.degrees(self.dominant_orientation_rad):.0f}",
            relation="bvc_boundary_anchoring",
            target_entity="subiculum",
        )


class SubicularBoundaryVectorEngine:
    """Subicular boundary vector cell engine computing spatial proximity maps and place field gating."""

    def __init__(
        self,
        distance_bins: list[float] | None = None,
        num_orientations: int = 8,
    ) -> None:
        """Initialize subicular boundary vector engine.

        Args:
            distance_bins: List of preferred boundary distances in grid units.
            num_orientations: Number of discrete allocentric angular directions (default 8).
        """
        self.distance_bins = distance_bins or [1.0, 2.5, 5.0, 8.0]
        self.num_orientations = max(4, int(num_orientations))

        # Build BVC population covering distance and orientation space
        self.cells: list[BoundaryVectorCell] = []
        angle_step = (2.0 * math.pi) / self.num_orientations

        for d in self.distance_bins:
            for o in range(self.num_orientations):
                angle = o * angle_step
                cid = f"bvc_d{int(d * 10)}_a{int(math.degrees(angle))}"
                cell = BoundaryVectorCell(
                    cell_id=cid,
                    preferred_distance=float(d),
                    preferred_angle_rad=float(angle),
                    radial_tuning=1.2,
                    angular_tuning=math.pi / 4.0,
                )
                self.cells.append(cell)

    def compute_bvc_activations(
        self,
        avatar_pos: tuple[int, int],
        boundary_grid: np.ndarray,
    ) -> BoundaryVectorMap:
        """Compute population BVC firing based on surrounding obstacles and walls.

        Args:
            avatar_pos: Current agent coordinates (r, c).
            boundary_grid: 2D binary grid where 1 = boundary/wall/obstacle, 0 = traversable.

        Returns:
            BoundaryVectorMap representing spatial boundary anchoring.
        """
        h, w = boundary_grid.shape
        ar, ac = avatar_pos

        # Extract boundary points relative to avatar
        boundary_points: list[tuple[float, float]] = []
        min_dist = float("inf")

        for r in range(h):
            for c in range(w):
                if boundary_grid[r, c] > 0:
                    dr = float(r - ar)
                    dc = float(c - ac)
                    dist = math.sqrt(dr * dr + dc * dc)
                    if dist > 0.0:
                        # Angle in [0, 2*pi) relative to north (r-axis)
                        angle = (math.atan2(dc, -dr)) % (2.0 * math.pi)
                        boundary_points.append((dist, angle))
                        if dist < min_dist:
                            min_dist = dist

        if min_dist == float("inf"):
            min_dist = 100.0

        # Calculate firing of each BVC across all detected boundary points
        activations: dict[str, float] = {}
        weighted_angles: list[float] = []
        total_weights: float = 0.0

        for cell in self.cells:
            # Cell activation is maximum or sum response over visible boundary points
            cell_act = 0.0
            for dist, angle in boundary_points:
                resp = cell.response(dist, angle)
                if resp > cell_act:
                    cell_act = resp

            activations[cell.cell_id] = float(cell_act)
            if cell_act > 0.1:
                weighted_angles.append(cell.preferred_angle_rad * cell_act)
                total_weights += cell_act

        dom_angle = (sum(weighted_angles) / total_weights if total_weights > 0 else 0.0) % (
            2.0 * math.pi
        )

        # Synthesize 2D activation field
        act_grid = np.zeros((h, w), dtype=np.float64)
        for r in range(h):
            for c in range(w):
                if 0 <= r < h and 0 <= c < w:
                    dr = float(r - ar)
                    dc = float(c - ac)
                    d = math.sqrt(dr * dr + dc * dc)
                    ang = (math.atan2(dc, -dr)) % (2.0 * math.pi)
                    # Sum responses from all cells at position (r, c)
                    for cell in self.cells:
                        act_grid[r, c] += cell.response(d, ang) * activations[cell.cell_id]

        max_val = act_grid.max()
        if max_val > 0:
            act_grid /= max_val

        active_count = sum(1 for v in activations.values() if v > 0.25)

        return BoundaryVectorMap(
            activation_grid=act_grid,
            active_bvc_count=active_count,
            nearest_boundary_distance=float(min_dist),
            dominant_orientation_rad=float(dom_angle),
            bvc_activations=activations,
        )

    def predict_place_field_deformation(
        self,
        bvc_map_before: BoundaryVectorMap,
        bvc_map_after: BoundaryVectorMap,
    ) -> float:
        """Measure topological similarity between two BVC maps under room deformation."""
        g1 = bvc_map_before.activation_grid
        g2 = bvc_map_after.activation_grid
        if g1.shape != g2.shape:
            return 0.0

        norm1 = np.linalg.norm(g1)
        norm2 = np.linalg.norm(g2)
        if norm1 <= 0 or norm2 <= 0:
            return 1.0 if norm1 == norm2 else 0.0

        # Cosine correlation between boundary activation fields
        correlation = float(np.sum(g1 * g2) / (norm1 * norm2))
        return float(np.clip(correlation, 0.0, 1.0))
