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
4. Formal Geometric & Perspective Homography:
   - Evaluates transformation model fits across Affine, Isometric, and Projective models with uncertainty (W040).
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from hbllm.hcir.skills.declarative import SpatialActionIntent, SymbolicSubgoal
from hbllm.hcir.world.cortex_causal import CausalPredicate

logger = logging.getLogger(__name__)


class SpatialBearing(StrEnum):
    """Cardinal, intercardinal, and self bearings in egocentric space."""

    NORTH = "N"
    SOUTH = "S"
    EAST = "E"
    WEST = "W"
    NORTH_EAST = "NE"
    NORTH_WEST = "NW"
    SOUTH_EAST = "SE"
    SOUTH_WEST = "SW"
    SELF = "SELF"


class HeadingOrientation(StrEnum):
    """Gaze and heading orientations in 2D coordinate space."""

    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"

    @property
    def unit_vector(self) -> tuple[int, int]:
        """Directional unit vector in (delta_r, delta_c)."""
        if self == HeadingOrientation.NORTH:
            return (-1, 0)
        elif self == HeadingOrientation.SOUTH:
            return (1, 0)
        elif self == HeadingOrientation.EAST:
            return (0, 1)
        elif self == HeadingOrientation.WEST:
            return (0, -1)
        return (-1, 0)

    @property
    def right_vector(self) -> tuple[int, int]:
        """Right-hand orthogonal lateral unit vector in (delta_r, delta_c)."""
        hr, hc = self.unit_vector
        return (hc, -hr)


class DihedralTransform(StrEnum):
    """Canonical D4 dihedral group transformations."""

    REFLECT_HORIZONTAL = "reflect_horizontal"
    REFLECT_VERTICAL = "reflect_vertical"
    REFLECT_MAIN_DIAG = "reflect_main_diag"
    ROTATE_180 = "rotate_180"
    ROTATE_90_CW = "rotate_90_cw"
    ROTATE_90_CCW = "rotate_90_ccw"
    IDENTITY = "identity"


class GeometricModelType(StrEnum):
    """Geometric transformation family classifications (W040)."""

    ISOMETRIC = "isometric"
    AFFINE = "affine"
    PROJECTIVE = "projective"
    DEGENERATE = "degenerate"


@dataclass(frozen=True)
class EgocentricPolarCoord:
    """Viewpoint-invariant polar coordinate and forward/lateral projections."""

    radius: float
    azimuth_rad: float
    relative_bearing: str
    forward_depth: float
    lateral_offset: float

    def to_causal_predicate(self, entity_id: str = "entity") -> CausalPredicate:
        """Export egocentric polar coordinates as a formal causal predicate."""
        return CausalPredicate(
            variable="egocentric_polar",
            operator="==",
            value=f"r:{self.radius:.2f}:az:{self.azimuth_rad:.2f}:{self.relative_bearing}",
            relation=f"fwd:{self.forward_depth:.2f}:lat:{self.lateral_offset:.2f}",
            target_entity=entity_id,
        )


@dataclass(frozen=True)
class EgocentricEntityRepresentation:
    """High-level lifted spatial entity in avatar-centric egocentric frame."""

    entity_id: str
    feature_id: int
    allocentric_pos: tuple[int, int]
    polar_coord: EgocentricPolarCoord
    egocentric_displacement: tuple[int, int]

    def to_causal_predicate(self) -> CausalPredicate:
        """Export entity egocentric representation as a causal predicate."""
        return CausalPredicate(
            variable=f"ego_entity:{self.entity_id}",
            operator="==",
            value=f"feat:{self.feature_id}:bearing:{self.polar_coord.relative_bearing}",
            relation=f"dist:{self.polar_coord.radius:.2f}",
            target_entity=self.entity_id,
        )


@dataclass(frozen=True)
class SpatialEgoVector:
    """Egocentric displacement vector relative to avatar position."""

    delta_r: int
    delta_c: int
    manhattan_dist: int
    bearing: SpatialBearing | str

    def is_adjacent(self, chebyshev: bool = False) -> bool:
        """True if the target position is within immediate 1-step reach."""
        if chebyshev:
            return max(abs(self.delta_r), abs(self.delta_c)) <= 1
        return self.manhattan_dist <= 1

    def unit_step(self) -> tuple[int, int]:
        """Normalize vector to single cardinal or diagonal unit step."""
        step_r = 0 if self.delta_r == 0 else (1 if self.delta_r > 0 else -1)
        step_c = 0 if self.delta_c == 0 else (1 if self.delta_c > 0 else -1)
        return step_r, step_c

    def to_rotated_ego(
        self,
        heading: HeadingOrientation | str = HeadingOrientation.NORTH,
    ) -> tuple[int, int]:
        """Rotate displacement vector to align with avatar heading (forward_depth, lateral_offset)."""
        if isinstance(heading, str):
            heading = HeadingOrientation(heading.upper())
        hr, hc = heading.unit_vector
        rr, rc = heading.right_vector
        fwd = self.delta_r * hr + self.delta_c * hc
        lat = self.delta_r * rr + self.delta_c * rc
        return int(round(fwd)), int(round(lat))

    def to_polar(
        self,
        heading: HeadingOrientation | str = HeadingOrientation.NORTH,
    ) -> EgocentricPolarCoord:
        """Transform displacement into viewpoint-invariant polar coordinate relative to heading."""
        if isinstance(heading, str):
            heading = HeadingOrientation(heading.upper())
        hr, hc = heading.unit_vector
        rr, rc = heading.right_vector

        fwd = float(self.delta_r * hr + self.delta_c * hc)
        lat = float(self.delta_r * rr + self.delta_c * rc)
        radius = float(np.hypot(self.delta_r, self.delta_c))

        if radius < 1e-6:
            return EgocentricPolarCoord(
                radius=0.0,
                azimuth_rad=0.0,
                relative_bearing="SELF",
                forward_depth=0.0,
                lateral_offset=0.0,
            )

        azimuth = float(np.arctan2(lat, fwd))
        pi = float(np.pi)

        # Discretize relative bearing into 8 octants
        if abs(azimuth) < pi / 8.0:
            rel_bearing = "FORWARD"
        elif pi / 8.0 <= azimuth < 3.0 * pi / 8.0:
            rel_bearing = "FORWARD_RIGHT"
        elif 3.0 * pi / 8.0 <= azimuth < 5.0 * pi / 8.0:
            rel_bearing = "RIGHT"
        elif 5.0 * pi / 8.0 <= azimuth < 7.0 * pi / 8.0:
            rel_bearing = "BACKWARD_RIGHT"
        elif abs(azimuth) >= 7.0 * pi / 8.0:
            rel_bearing = "BACKWARD"
        elif -7.0 * pi / 8.0 <= azimuth < -5.0 * pi / 8.0:
            rel_bearing = "BACKWARD_LEFT"
        elif -5.0 * pi / 8.0 <= azimuth < -3.0 * pi / 8.0:
            rel_bearing = "LEFT"
        else:
            rel_bearing = "FORWARD_LEFT"

        return EgocentricPolarCoord(
            radius=radius,
            azimuth_rad=azimuth,
            relative_bearing=rel_bearing,
            forward_depth=fwd,
            lateral_offset=lat,
        )

    def to_causal_predicate(
        self, agent_id: str = "avatar", target_id: str = "target"
    ) -> CausalPredicate:
        """Export egocentric relational bearing into a formal declarative causal predicate."""
        return CausalPredicate(
            variable="spatial_bearing",
            operator="==",
            value=str(self.bearing),
            relation=f"dist_{self.manhattan_dist}",
            target_entity=target_id,
        )


@dataclass
class GeometricMappingResult:
    """Epistemic result of geometric mapping selection between coordinate frames (W040)."""

    best_model_type: GeometricModelType | str
    best_model: AffineTransform2D | ProjectiveHomography2D | None
    residual: float
    residuals: dict[str, float]
    ambiguity_score: float
    parallelism_deviation: float
    is_isometric: bool
    perspective_distortion: float

    def to_dict(self) -> dict[str, Any]:
        """Convert result to standard dictionary representation."""
        return {
            "best_model_type": str(self.best_model_type),
            "best_model": self.best_model,
            "residual": self.residual,
            "residuals": dict(self.residuals),
            "ambiguity_score": self.ambiguity_score,
            "parallelism_deviation": self.parallelism_deviation,
            "is_isometric": self.is_isometric,
            "perspective_distortion": self.perspective_distortion,
        }

    def __getitem__(self, key: str) -> Any:
        """Dictionary access emulation for backward compatibility."""
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Safe dictionary getter emulation."""
        return self.to_dict().get(key, default)

    def to_causal_predicate(
        self, source_frame: str = "source", target_frame: str = "target"
    ) -> CausalPredicate:
        """Export geometric mapping invariance as a formal causal predicate."""
        return CausalPredicate(
            variable="geometric_mapping",
            operator="==",
            value=str(self.best_model_type),
            relation=str(round(self.residual, 4)),
            target_entity=target_frame,
        )


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


# ── W040: Formal Geometric & Perspective Transformations ─────────────────────


@dataclass
class AffineTransform2D:
    """2D Affine transformation supporting rotation, translation, scaling, and shear (W040)."""

    matrix: np.ndarray  # Shape: (2, 3)

    def forward(self, points: np.ndarray) -> np.ndarray:
        """Apply affine transformation to points of shape (N, 2)."""
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        # pts: (N, 2) -> (N, 3) with homogeneous 1
        homog = np.hstack([pts, np.ones((len(pts), 1), dtype=float)])
        return homog @ self.matrix.T

    def inverse(self) -> AffineTransform2D:
        """Compute analytical inverse affine transformation."""
        A = self.matrix[:, :2]
        b = self.matrix[:, 2]
        A_inv = np.linalg.inv(A)
        b_inv = -A_inv @ b
        inv_mat = np.hstack([A_inv, b_inv.reshape(-1, 1)])
        return AffineTransform2D(matrix=inv_mat)

    @classmethod
    def estimate(
        cls, src_points: np.ndarray, dst_points: np.ndarray
    ) -> tuple[AffineTransform2D, float]:
        """Estimate 2D affine transformation from point correspondences using least squares."""
        src = np.asarray(src_points, dtype=float)
        dst = np.asarray(dst_points, dtype=float)
        if len(src) < 3:
            raise ValueError(
                "At least 3 point correspondences required for affine transformation estimation."
            )

        # Solve [x, y, 1] @ M.T = dst
        A = np.hstack([src, np.ones((len(src), 1), dtype=float)])
        params_x, _, _, _ = np.linalg.lstsq(A, dst[:, 0], rcond=None)
        params_y, _, _, _ = np.linalg.lstsq(A, dst[:, 1], rcond=None)

        matrix = np.vstack([params_x, params_y])
        model = cls(matrix=matrix)
        pred = model.forward(src)
        residual = float(np.mean(np.linalg.norm(pred - dst, axis=1)))
        return model, residual


@dataclass
class AxonometricIsometricTransform:
    """Axonometric parallel projection and 3D isometric coordinate transformation (W040)."""

    alpha_deg: float = 30.0  # Isometric baseline angle (30°)
    scale: float = 1.0

    def project_3d_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        """Project (x, y, z) 3D coordinates into (u, v) 2D screen/grid space."""
        pts = np.asarray(points_3d, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        rad = np.radians(self.alpha_deg)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        u = (x - y) * np.cos(rad) * self.scale
        v = ((x + y) * np.sin(rad) - z) * self.scale
        return np.column_stack([u, v])

    def unproject_2d_to_3d(self, points_2d: np.ndarray, z_plane: float = 0.0) -> np.ndarray:
        """Unproject 2D screen coordinates back to 3D given an assumed ground plane or height prior."""
        pts = np.asarray(points_2d, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        rad = np.radians(self.alpha_deg)
        u = pts[:, 0] / self.scale
        v = pts[:, 1] / self.scale

        diff = u / np.cos(rad)
        sum_xy = (v + z_plane) / np.sin(rad)
        x = 0.5 * (sum_xy + diff)
        y = 0.5 * (sum_xy - diff)
        z = np.full_like(x, z_plane)
        return np.column_stack([x, y, z])


@dataclass
class ProjectiveHomography2D:
    """Projective geometry and 2D planar homography under perspective division (W040)."""

    homography_matrix: np.ndarray  # Shape: (3, 3)

    def forward(self, points: np.ndarray) -> np.ndarray:
        """Apply non-linear projective homography with perspective division."""
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        homog = np.hstack([pts, np.ones((len(pts), 1), dtype=float)])  # (N, 3)
        projected = homog @ self.homography_matrix.T  # (N, 3)
        # Perspective division
        w = projected[:, 2:3]
        w_safe = np.where(np.abs(w) < 1e-8, 1e-8 * np.sign(w + 1e-12), w)
        return projected[:, :2] / w_safe

    def inverse(self) -> ProjectiveHomography2D:
        """Compute inverse projective homography."""
        inv_h = np.linalg.inv(self.homography_matrix)
        return ProjectiveHomography2D(
            homography_matrix=inv_h / (inv_h[2, 2] if inv_h[2, 2] != 0 else 1.0)
        )

    @classmethod
    def estimate(
        cls, src_points: np.ndarray, dst_points: np.ndarray
    ) -> tuple[ProjectiveHomography2D, float]:
        """Estimate 3x3 projective homography via Direct Linear Transformation (DLT) using SVD."""
        src = np.asarray(src_points, dtype=float)
        dst = np.asarray(dst_points, dtype=float)
        N = len(src)
        if N < 4:
            raise ValueError(
                "At least 4 point correspondences required for projective homography estimation."
            )

        A_rows = []
        for i in range(N):
            x, y = src[i, 0], src[i, 1]
            u, v = dst[i, 0], dst[i, 1]
            A_rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
            A_rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])

        A = np.array(A_rows, dtype=float)
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        if abs(H[2, 2]) > 1e-8:
            H = H / H[2, 2]

        model = cls(homography_matrix=H)
        pred = model.forward(src)
        residual = float(np.mean(np.linalg.norm(pred - dst, axis=1)))
        return model, residual


class GeometricModelSelector:
    """Evaluates transformation model fits across Affine, Isometric, and Projective models with uncertainty (W040).

    Uses geometric invariants (parallelism preservation, axonometric axes, vanishing points)
    and hierarchical Occam's selection rather than naive residual minimization.
    """

    @staticmethod
    def check_collinearity(points: np.ndarray) -> float:
        """Measure collinearity of 2D points in [0.0, 1.0]. 1.0 indicates perfectly collinear."""
        pts = np.asarray(points, dtype=float)
        if len(pts) < 3:
            return 1.0
        centered = pts - np.mean(pts, axis=0)
        _, s, _ = np.linalg.svd(centered)
        if len(s) < 2 or s[0] < 1e-12:
            return 1.0
        return float(1.0 - (s[1] / s[0]))

    @staticmethod
    def check_parallelism_preservation(
        src_points: np.ndarray,
        dst_points: np.ndarray,
        parallel_tol: float = 1e-3,
    ) -> tuple[float, int]:
        """Test parallelism preservation between pairs of segments in src vs dst.

        Returns (max_parallelism_deviation, num_pairs_evaluated).
        A deviation of 0.0 indicates perfect parallelism preservation (characteristic of Affine maps).
        """
        src = np.asarray(src_points, dtype=float)
        dst = np.asarray(dst_points, dtype=float)
        N = len(src)
        if N < 4:
            return 0.0, 0

        max_dev = 0.0
        evaluated = 0

        # Find pairs of line segments (i->j) and (k->l) that are parallel in src
        for i in range(N):
            for j in range(i + 1, N):
                u_src = src[j] - src[i]
                u_norm = np.linalg.norm(u_src)
                if u_norm < 1e-6:
                    continue

                for k in range(N):
                    for l in range(k + 1, N):
                        if (i, j) == (k, l) or (i == k and j == l):
                            continue
                        v_src = src[l] - src[k]
                        v_norm = np.linalg.norm(v_src)
                        if v_norm < 1e-6:
                            continue

                        # Check if u_src and v_src are parallel
                        cross_src = abs(u_src[0] * v_src[1] - u_src[1] * v_src[0]) / (
                            u_norm * v_norm
                        )
                        if cross_src < parallel_tol:
                            # Evaluate in dst
                            u_dst = dst[j] - dst[i]
                            v_dst = dst[l] - dst[k]
                            ud_norm = np.linalg.norm(u_dst)
                            vd_norm = np.linalg.norm(v_dst)
                            if ud_norm > 1e-6 and vd_norm > 1e-6:
                                cross_dst = abs(u_dst[0] * v_dst[1] - u_dst[1] * v_dst[0]) / (
                                    ud_norm * vd_norm
                                )
                                dev = abs(cross_dst - cross_src)
                                if dev > max_dev:
                                    max_dev = dev
                                evaluated += 1

        return float(max_dev), evaluated

    @staticmethod
    def check_isometry(affine_matrix: np.ndarray) -> tuple[bool, float]:
        """Test if 2x3 affine matrix is an isometric transformation (rigid rotation/reflection + translation).

        Returns (is_isometric, condition_deviation).
        """
        A = affine_matrix[:, :2]
        _, s, _ = np.linalg.svd(A)
        if len(s) < 2 or s[1] < 1e-12:
            return False, 1.0
        cond_dev = abs(s[0] - s[1])
        scale_dev = abs(s[0] - 1.0)
        is_iso = bool(cond_dev < 0.05 and scale_dev < 0.05)
        return is_iso, float(cond_dev)

    @staticmethod
    def select_best_model(
        src_points: np.ndarray,
        dst_points: np.ndarray,
        residual_tol: float = 0.05,
    ) -> GeometricMappingResult:
        """Select best geometric model under invariant Occam selection and report epistemic model ambiguity.

        Hierarchical Occam's Selection:
        1. Collinear or underconstrained inputs return 'degenerate' with maximal epistemic ambiguity (1.0).
        2. Affine strictly preferred over Projective when parallelism is preserved (deviation < 1e-3)
           and residual is within tolerance, preventing overfitting from 8-param homography.
        3. Isometric specialized when scale is isotropic and canonical axes are preserved.
        4. Projective homography chosen only when genuine perspective foreshortening / vanishing points exist.
        """
        src = np.asarray(src_points, dtype=float)
        dst = np.asarray(dst_points, dtype=float)
        N = len(src)

        collinearity = GeometricModelSelector.check_collinearity(src)
        if N < 3 or collinearity > 0.999:
            return GeometricMappingResult(
                best_model_type=GeometricModelType.DEGENERATE,
                best_model=None,
                residual=999.0,
                residuals={},
                ambiguity_score=1.0,
                parallelism_deviation=1.0,
                is_isometric=False,
                perspective_distortion=0.0,
            )

        results: dict[str, Any] = {}
        residuals: dict[str, float] = {}

        # 1. Fit Affine (6 parameters)
        try:
            affine_model, aff_res = AffineTransform2D.estimate(src, dst)
            results["affine"] = affine_model
            residuals["affine"] = aff_res
        except Exception:
            residuals["affine"] = 999.0

        # 2. Fit Projective Homography (8 parameters)
        proj_model: ProjectiveHomography2D | None = None
        proj_res = 999.0
        persp_dist = 0.0
        if N >= 4:
            try:
                proj_model, proj_res = ProjectiveHomography2D.estimate(src, dst)
                results["projective"] = proj_model
                residuals["projective"] = proj_res
                # Perspective coefficients [h31, h32]
                H = proj_model.homography_matrix
                persp_dist = float(np.linalg.norm(H[2, :2]))
            except Exception:
                residuals["projective"] = 999.0

        par_dev, _ = GeometricModelSelector.check_parallelism_preservation(src, dst)

        # 3. Occam's Invariant Selection
        aff_model = results.get("affine")
        is_iso = False
        if aff_model is not None:
            is_iso, _ = GeometricModelSelector.check_isometry(aff_model.matrix)

        # Decision logic:
        # If affine residual is within tolerance or projective perspective is negligible:
        # Affine wins.
        if residuals["affine"] <= residual_tol or (persp_dist < 1e-4 and residuals["affine"] < 0.2):
            best_type = GeometricModelType.ISOMETRIC if is_iso else GeometricModelType.AFFINE
            best_model = aff_model
            best_res = residuals["affine"]
            # Low ambiguity if clean fit
            ambiguity = max(0.0, min(1.0, best_res / (residual_tol + 1e-6) * 0.2))
        elif (
            proj_model is not None
            and proj_res < residuals["affine"] - 0.05
            and proj_res <= residual_tol
            and (par_dev > 1e-3 or persp_dist > 1e-4)
        ):
            # True non-affine perspective convergence
            best_type = GeometricModelType.PROJECTIVE
            best_model = proj_model
            best_res = proj_res
            ambiguity = max(0.0, min(1.0, proj_res / (residual_tol + 1e-6) * 0.2))
        else:
            # Ambiguous or high residual regime
            best_type = (
                GeometricModelType.AFFINE
                if residuals["affine"] <= proj_res
                else GeometricModelType.PROJECTIVE
            )
            best_model = results.get(str(best_type))
            best_res = residuals[str(best_type)]
            ambiguity = 0.85

        return GeometricMappingResult(
            best_model_type=best_type,
            best_model=best_model,
            residual=best_res,
            residuals=residuals,
            ambiguity_score=round(ambiguity, 4),
            parallelism_deviation=round(par_dev, 6),
            is_isometric=is_iso,
            perspective_distortion=round(persp_dist, 6),
        )
