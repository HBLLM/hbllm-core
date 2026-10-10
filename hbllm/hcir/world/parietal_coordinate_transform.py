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
from typing import Any

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

    @staticmethod
    def infer_geometric_mapping(
        src_points: np.ndarray,
        dst_points: np.ndarray,
        residual_tol: float = 0.05,
    ) -> dict[str, Any]:
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

        # From: u = (x - y) * cos(rad) and v = (x + y) * sin(rad) - z_plane
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
    ) -> dict[str, Any]:
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
            return {
                "best_model_type": "degenerate",
                "best_model": None,
                "residual": 999.0,
                "residuals": {},
                "ambiguity_score": 1.0,
                "parallelism_deviation": 1.0,
                "is_isometric": False,
                "perspective_distortion": 0.0,
            }

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

        par_dev, n_pairs = GeometricModelSelector.check_parallelism_preservation(src, dst)

        # 3. Occam's Invariant Selection
        aff_model = results.get("affine")
        is_iso = False
        if aff_model is not None:
            is_iso, _ = GeometricModelSelector.check_isometry(aff_model.matrix)

        # Decision logic:
        # If affine residual is within tolerance or projective perspective is negligible:
        # Affine wins.
        if residuals["affine"] <= residual_tol or (persp_dist < 1e-4 and residuals["affine"] < 0.2):
            best_type = "isometric" if is_iso else "affine"
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
            best_type = "projective"
            best_model = proj_model
            best_res = proj_res
            ambiguity = max(0.0, min(1.0, proj_res / (residual_tol + 1e-6) * 0.2))
        else:
            # Ambiguous or high residual regime
            best_type = "affine" if residuals["affine"] <= proj_res else "projective"
            best_model = results.get(best_type)
            best_res = residuals[best_type]
            ambiguity = 0.85

        return {
            "best_model_type": best_type,
            "best_model": best_model,
            "residual": best_res,
            "residuals": residuals,
            "ambiguity_score": round(ambiguity, 4),
            "parallelism_deviation": round(par_dev, 6),
            "is_isometric": is_iso,
            "perspective_distortion": round(persp_dist, 6),
        }
