"""Parietal and Retrosplenial Egocentric-Allocentric Coordinate Transformers."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from hbllm.hcir.skills.declarative import SpatialActionIntent, SymbolicSubgoal
from hbllm.hcir.world.spatial.models import (
    DihedralTransform,
    EgocentricCoordinate,
    EgocentricEntityRepresentation,
    EgocentricPolarCoord,
    GeometricMappingResult,
    HeadingOrientation,
    SpatialBearing,
    SpatialEgoVector,
)
from hbllm.hcir.world.spatial.transforms import GeometricModelSelector

logger = logging.getLogger(__name__)


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
            bearing = SpatialBearing.SELF
        elif dr < 0 and dc == 0:
            bearing = SpatialBearing.NORTH
        elif dr > 0 and dc == 0:
            bearing = SpatialBearing.SOUTH
        elif dr == 0 and dc > 0:
            bearing = SpatialBearing.EAST
        elif dr == 0 and dc < 0:
            bearing = SpatialBearing.WEST
        elif dr < 0 and dc > 0:
            bearing = SpatialBearing.NORTH_EAST
        elif dr < 0 and dc < 0:
            bearing = SpatialBearing.NORTH_WEST
        elif dr > 0 and dc > 0:
            bearing = SpatialBearing.SOUTH_EAST
        else:
            bearing = SpatialBearing.SOUTH_WEST

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
    def allo_to_ego_polar(
        allo_coord: tuple[int, int],
        avatar_pos: tuple[int, int],
        heading: HeadingOrientation | str = HeadingOrientation.NORTH,
    ) -> EgocentricPolarCoord:
        """Transform allocentric coordinate to viewpoint-invariant polar coordinate relative to heading."""
        ego_vec = ParietalCoordinateTransformer.allo_to_ego(allo_coord, avatar_pos)
        return ego_vec.to_polar(heading=heading)

    @staticmethod
    def ego_polar_to_allo(
        polar: EgocentricPolarCoord,
        avatar_pos: tuple[int, int],
        heading: HeadingOrientation | str = HeadingOrientation.NORTH,
    ) -> tuple[int, int]:
        """Transform viewpoint-invariant polar coordinate back to world-fixed allocentric coordinate."""
        if isinstance(heading, str):
            heading = HeadingOrientation(heading.upper())
        hr, hc = heading.unit_vector
        rr, rc = heading.right_vector
        dr = int(round(polar.forward_depth * hr + polar.lateral_offset * rr))
        dc = int(round(polar.forward_depth * hc + polar.lateral_offset * rc))
        return (avatar_pos[0] + dr, avatar_pos[1] + dc)

    @staticmethod
    def lift_entities_to_ego_frame(
        entities: Sequence[Any],
        avatar_pos: tuple[int, int],
        heading: HeadingOrientation | str = HeadingOrientation.NORTH,
    ) -> list[EgocentricEntityRepresentation]:
        """Lift a collection of spatial entities into canonical viewpoint-invariant egocentric representations."""
        lifted: list[EgocentricEntityRepresentation] = []
        for ent in entities:
            ent_id = getattr(ent, "id", str(getattr(ent, "feature_id", "entity")))
            feat_id = getattr(ent, "feature_id", 0)
            if hasattr(ent, "grid_pos"):
                pos = ent.grid_pos
            elif hasattr(ent, "centroid"):
                pos = (int(round(ent.centroid[0])), int(round(ent.centroid[1])))
            elif hasattr(ent, "cells") and ent.cells:
                first = list(ent.cells)[0]
                pos = (first[0], first[1])
            else:
                pos = avatar_pos

            ego_vec = ParietalCoordinateTransformer.allo_to_ego(pos, avatar_pos)
            polar = ego_vec.to_polar(heading=heading)
            rot_disp = ego_vec.to_rotated_ego(heading=heading)
            lifted.append(
                EgocentricEntityRepresentation(
                    entity_id=ent_id,
                    feature_id=feat_id,
                    allocentric_pos=pos,
                    polar_coord=polar,
                    egocentric_displacement=rot_disp,
                )
            )
        return lifted

    @staticmethod
    def project_dihedral_transform(
        pos: tuple[int, int],
        grid_shape: tuple[int, int],
        transform: DihedralTransform | str,
    ) -> tuple[int, int]:
        """Project a 2D coordinate under D4 dihedral group transformations."""
        H, W = grid_shape
        r, c = pos
        t_str = str(transform)

        if t_str == DihedralTransform.REFLECT_HORIZONTAL:
            return (r, W - 1 - c)
        elif t_str == DihedralTransform.REFLECT_VERTICAL:
            return (H - 1 - r, c)
        elif t_str == DihedralTransform.REFLECT_MAIN_DIAG:
            return (c, r)
        elif t_str == DihedralTransform.ROTATE_180:
            return (H - 1 - r, W - 1 - c)
        elif t_str == DihedralTransform.ROTATE_90_CW:
            return (c, H - 1 - r)
        elif t_str == DihedralTransform.ROTATE_90_CCW:
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
                subgoal_pos, grid_shape, DihedralTransform.REFLECT_HORIZONTAL
            )
        elif sym_name == "vertical":
            return ParietalCoordinateTransformer.project_dihedral_transform(
                subgoal_pos, grid_shape, DihedralTransform.REFLECT_VERTICAL
            )
        elif sym_name == "rotational_180":
            return ParietalCoordinateTransformer.project_dihedral_transform(
                subgoal_pos, grid_shape, DihedralTransform.ROTATE_180
            )
        return None

    @staticmethod
    def transfer_subgoal_to_symbolic(
        subgoal_pos: tuple[int, int],
        grid_shape: tuple[int, int],
        grid: np.ndarray,
        background_feature: int = 0,
        intent: SpatialActionIntent = SpatialActionIntent.NAVIGATE,
    ) -> SymbolicSubgoal | None:
        """Transfer candidate subgoal across symmetry into a formal declarative SymbolicSubgoal."""
        transferred = ParietalCoordinateTransformer.transfer_subgoal_across_symmetry(
            subgoal_pos=subgoal_pos,
            grid_shape=grid_shape,
            grid=grid,
            background_feature=background_feature,
        )
        if transferred is None:
            return None
        return SymbolicSubgoal(
            intent=intent,
            target_query={"target_position": transferred},
        )

    @staticmethod
    def infer_geometric_mapping(
        src_points: np.ndarray,
        dst_points: np.ndarray,
        residual_tol: float = 0.05,
    ) -> GeometricMappingResult:
        """Infer best geometric coordinate model (Affine, Isometric, Projective) between point sets (W040)."""
        return GeometricModelSelector.select_best_model(
            src_points, dst_points, residual_tol=residual_tol
        )


class RetrosplenialCoordinateTransformer:
    """Retrosplenial Cortex (RSC) egocentric-to-allocentric reference frame transformer.

    Biologically modeled on Burgess (2008) and Taube (1995) Head-Direction / RSC network:
    Translates between:
    - Egocentric frame: oriented around the agent's current gaze / heading direction.
      (Forward, Right, Left, Backward)
    - Allocentric frame: world-fixed grid coordinates (row, col).
    """

    @staticmethod
    def infer_heading_from_displacement(
        prev_pos: tuple[int, int] | None,
        curr_pos: tuple[int, int],
        fallback: HeadingOrientation = HeadingOrientation.NORTH,
    ) -> HeadingOrientation:
        """Infer current heading from consecutive positions."""
        if prev_pos is None:
            return fallback
        dr = curr_pos[0] - prev_pos[0]
        dc = curr_pos[1] - prev_pos[1]
        if dr < 0 and abs(dr) >= abs(dc):
            return HeadingOrientation.NORTH
        elif dr > 0 and abs(dr) >= abs(dc):
            return HeadingOrientation.SOUTH
        elif dc > 0 and abs(dc) > abs(dr):
            return HeadingOrientation.EAST
        elif dc < 0 and abs(dc) > abs(dr):
            return HeadingOrientation.WEST
        return fallback

    @staticmethod
    def ego_to_allo(
        ego_coord: EgocentricCoordinate | tuple[int, int],
        agent_pos: tuple[int, int],
        heading: HeadingOrientation = HeadingOrientation.NORTH,
    ) -> tuple[int, int]:
        """Convert egocentric (forward, lateral) offset to world allocentric (row, col)."""
        if isinstance(ego_coord, EgocentricCoordinate):
            fwd, lat = ego_coord.forward, ego_coord.lateral
        else:
            fwd, lat = ego_coord

        fr, fc = heading.unit_vector
        rr, rc = heading.right_vector

        ar, ac = agent_pos
        target_r = ar + fwd * fr + lat * rr
        target_c = ac + fwd * fc + lat * rc
        return (target_r, target_c)

    @staticmethod
    def allo_to_ego(
        target_pos: tuple[int, int],
        agent_pos: tuple[int, int],
        heading: HeadingOrientation = HeadingOrientation.NORTH,
    ) -> EgocentricCoordinate:
        """Convert world allocentric (row, col) target to egocentric (forward, lateral) offset."""
        dr = target_pos[0] - agent_pos[0]
        dc = target_pos[1] - agent_pos[1]

        fr, fc = heading.unit_vector
        rr, rc = heading.right_vector

        # Dot product with forward and right unit vectors
        fwd = dr * fr + dc * fc
        lat = dr * rr + dc * rc
        return EgocentricCoordinate(forward=fwd, lateral=lat)

    @staticmethod
    def extract_egocentric_view(
        grid: np.ndarray,
        agent_pos: tuple[int, int],
        heading: HeadingOrientation = HeadingOrientation.NORTH,
        view_radius: int = 3,
        pad_value: int = 0,
    ) -> np.ndarray:
        """Extract egocentric oriented visual patch centered on agent with forward pointing up."""
        H, W = grid.shape
        size = 2 * view_radius + 1
        raw_patch = np.full((size, size), pad_value, dtype=grid.dtype)

        ar, ac = agent_pos
        for r_off in range(-view_radius, view_radius + 1):
            for c_off in range(-view_radius, view_radius + 1):
                gr, gc = ar + r_off, ac + c_off
                if 0 <= gr < H and 0 <= gc < W:
                    raw_patch[r_off + view_radius, c_off + view_radius] = grid[gr, gc]

        if heading == HeadingOrientation.NORTH:
            return raw_patch
        elif heading == HeadingOrientation.EAST:
            return np.rot90(raw_patch, k=1)
        elif heading == HeadingOrientation.SOUTH:
            return np.rot90(raw_patch, k=2)
        elif heading == HeadingOrientation.WEST:
            return np.rot90(raw_patch, k=3)
        return raw_patch
