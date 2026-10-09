from __future__ import annotations

"""Posterior Parietal Cortex (PPC) Egocentric-to-Allocentric Spatial Coordinate Transformer.

Biologically modeled on mammalian posterior parietal cortex and hippocampal place/grid cell networks:
1. Dual Reference Frames:
   - Egocentric: Avatar-centric coordinates relative to current position and gaze heading.
   - Allocentric: World-fixed absolute coordinates in environmental canvas space.
2. D4 Dihedral Symmetry Invariance:
   - Evaluates reflectional (horizontal, vertical, diagonal) and rotational (90°, 180°, 270°) symmetries.
   - Projects validated subgoals and motor trajectories across symmetry planes.
3. Spatial Bearing & Relative Displacement:
   - Computes cardinal bearings, relative distances, and target approach vectors.
"""

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpatialEgoVector:
    """Egocentric displacement vector relative to avatar position."""

    delta_r: int
    delta_c: int
    manhattan_dist: int
    bearing: str  # 'N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW', 'SELF'


class ParietalCoordinateTransformer:
    """Posterior Parietal Cortex coordinate conversion, reference frame mapping, and spatial symmetry projector."""

    @staticmethod
    def allo_to_ego(
        allo_coord: tuple[int, int],
        avatar_pos: tuple[int, int],
    ) -> SpatialEgoVector:
        """Transform world-fixed allocentric coordinate to avatar-centric egocentric vector."""
        ar, ac = avatar_pos
        tr, tc = allo_coord
        dr = tr - ar
        dc = tc - ac
        dist = abs(dr) + abs(dc)

        # Infer cardinal bearing
        if dr == 0 and dc == 0:
            bearing = "SELF"
        elif dr < 0 and dc == 0:
            bearing = "N"
        elif dr > 0 and dc == 0:
            bearing = "S"
        elif dr == 0 and dc > 0:
            bearing = "E"
        elif dr == 0 and dc < 0:
            bearing = "W"
        elif dr < 0 and dc > 0:
            bearing = "NE"
        elif dr < 0 and dc < 0:
            bearing = "NW"
        elif dr > 0 and dc > 0:
            bearing = "SE"
        else:
            bearing = "SW"

        return SpatialEgoVector(
            delta_r=dr,
            delta_c=dc,
            manhattan_dist=dist,
            bearing=bearing,
        )

    @staticmethod
    def ego_to_allo(
        ego_vector: SpatialEgoVector | tuple[int, int],
        avatar_pos: tuple[int, int],
    ) -> tuple[int, int]:
        """Transform avatar-relative egocentric displacement to world-fixed allocentric coordinate."""
        ar, ac = avatar_pos
        if isinstance(ego_vector, SpatialEgoVector):
            dr, dc = ego_vector.delta_r, ego_vector.delta_c
        else:
            dr, dc = ego_vector
        return (ar + dr, ac + dc)

    @staticmethod
    def project_dihedral_transform(
        pos: tuple[int, int],
        grid_shape: tuple[int, int],
        transform: str,
    ) -> tuple[int, int]:
        """Project a 2D coordinate under D4 dihedral group transformations."""
        H, W = grid_shape
        r, c = pos

        if transform == "reflect_horizontal":
            return (r, W - 1 - c)
        elif transform == "reflect_vertical":
            return (H - 1 - r, c)
        elif transform == "reflect_main_diag":
            return (c, r)
        elif transform == "rotate_180":
            return (H - 1 - r, W - 1 - c)
        elif transform == "rotate_90_cw":
            return (c, H - 1 - r)
        elif transform == "rotate_90_ccw":
            return (W - 1 - c, r)
        return (r, c)

    @staticmethod
    def infer_symmetry_planes(
        grid: np.ndarray,
        background_feature: int = 0,
    ) -> dict[str, float]:
        """Infer symmetry concordance scores in [0.0, 1.0] across horizontal, vertical, and 180° rotational axes."""
        H, W = grid.shape
        non_bg_mask = grid != background_feature
        n_pixels = int(np.count_nonzero(non_bg_mask))
        if n_pixels == 0:
            return {"horizontal": 1.0, "vertical": 1.0, "rotational_180": 1.0}

        # Horizontal reflection symmetry (left vs right)
        flipped_lr = np.fliplr(grid)
        match_h = int(np.count_nonzero((grid == flipped_lr) & non_bg_mask))
        score_h = match_h / float(n_pixels)

        # Vertical reflection symmetry (top vs bottom)
        flipped_ud = np.flipud(grid)
        match_v = int(np.count_nonzero((grid == flipped_ud) & non_bg_mask))
        score_v = match_v / float(n_pixels)

        # 180° Rotational symmetry
        rot_180 = np.rot90(grid, 2)
        match_r = int(np.count_nonzero((grid == rot_180) & non_bg_mask))
        score_r = match_r / float(n_pixels)

        return {
            "horizontal": float(score_h),
            "vertical": float(score_v),
            "rotational_180": float(score_r),
        }

    @staticmethod
    def transfer_subgoal_across_symmetry(
        subgoal_pos: tuple[int, int],
        grid_shape: tuple[int, int],
        grid: np.ndarray,
        background_feature: int = 0,
    ) -> tuple[int, int] | None:
        """Transfer candidate subgoal to its symmetric counterpart if dominant symmetry exists (>0.75)."""
        planes = ParietalCoordinateTransformer.infer_symmetry_planes(grid, background_feature)
        best_sym = max(planes.items(), key=lambda kv: kv[1])
        if best_sym[1] < 0.75:
            return None

        sym_name = best_sym[0]
        if sym_name == "horizontal":
            return ParietalCoordinateTransformer.project_dihedral_transform(
                subgoal_pos, grid_shape, "reflect_horizontal"
            )
        elif sym_name == "vertical":
            return ParietalCoordinateTransformer.project_dihedral_transform(
                subgoal_pos, grid_shape, "reflect_vertical"
            )
        elif sym_name == "rotational_180":
            return ParietalCoordinateTransformer.project_dihedral_transform(
                subgoal_pos, grid_shape, "rotate_180"
            )
        return None
