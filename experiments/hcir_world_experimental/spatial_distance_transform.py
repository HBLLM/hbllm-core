from __future__ import annotations

"""Native Fast 2D Euclidean Distance Transform (EDT) & Obstacle Margins.

Biologically & mathematically grounded in:
1. M.S.A. Graziano & D.F. Cooke (2006) - Parietal peri-personal space and the
   protective margin around the body and obstacles.
2. P.F. Felzenszwalb & D.P. Huttenlocher (2012) - "Distance Transforms of Sampled
   Functions" (Theory of Computing). Exact O(H × W) separable parabolic lower-envelope
   Euclidean distance transform.

In HBLLM, this module computes continuous obstacle distance fields, spatial safety
margins, and repulsive potential gradient vectors (∇D) to guide collision-free
spatial navigation and clearance buffering without external C/library dependencies.
"""

import math
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate


@dataclass
class EuclideanDistanceTransformResult:
    """Result of 2D Euclidean Distance Transform computation."""

    grid_shape: tuple[int, int]
    distance_matrix: list[list[float]]
    gradient_r: list[list[float]]
    gradient_c: list[list[float]]
    safest_cell: tuple[int, int]
    max_distance: float
    obstacle_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_distance(self, r: int, c: int) -> float:
        """Get clearance distance at coordinate (r, c)."""
        if 0 <= r < self.grid_shape[0] and 0 <= c < self.grid_shape[1]:
            return self.distance_matrix[r][c]
        return 0.0

    def is_safe(self, r: int, c: int, min_margin: float = 1.0) -> bool:
        """Check if cell clearance exceeds required safety margin."""
        return self.get_distance(r, c) >= min_margin

    def to_causal_predicate(self) -> CausalPredicate:
        """Export distance field summary as an HCIR causal predicate."""
        return CausalPredicate(
            variable="spatial:distance_transform",
            operator="==",
            value=f"max_clearance:{self.max_distance:.2f}",
            relation=f"safest:({self.safest_cell[0]},{self.safest_cell[1]}):obstacles:{self.obstacle_count}",
            target_entity="spatial_distance_transform",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording distance field properties."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="spatial_clearance_field",
                node_data={
                    "grid_shape": [self.grid_shape[0], self.grid_shape[1]],
                    "safest_cell": [self.safest_cell[0], self.safest_cell[1]],
                    "max_distance": float(self.max_distance),
                    "obstacle_count": int(self.obstacle_count),
                },
            )
        ]


class SpatialDistanceTransformEngine:
    """O(H × W) separable 2D Euclidean distance transform and safety margin engine."""

    INF: float = 1e9

    def _edt_1d(self, f: list[float], n: int) -> list[float]:
        """Felzenszwalb-Huttenlocher 1D squared distance transform along array f."""
        d: list[float] = [0.0] * n
        v: list[int] = [0] * n
        z: list[float] = [0.0] * (n + 1)
        k = 0
        v[0] = 0
        z[0] = -self.INF
        z[1] = self.INF

        for q in range(1, n):
            # Intersection of parabola from q and v[k]
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
            while s <= z[k]:
                k -= 1
                s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
            k += 1
            v[k] = q
            z[k] = s
            z[k + 1] = self.INF

        k = 0
        for q in range(n):
            while z[k + 1] < q:
                k += 1
            d[q] = (q - v[k]) * (q - v[k]) + f[v[k]]

        return d

    def compute_edt(
        self,
        grid_shape: tuple[int, int],
        obstacles: set[tuple[int, int]],
    ) -> EuclideanDistanceTransformResult:
        """Compute exact 2D Euclidean Distance Transform in O(H × W) time."""
        rows, cols = grid_shape
        if rows <= 0 or cols <= 0:
            return EuclideanDistanceTransformResult(
                grid_shape=grid_shape,
                distance_matrix=[],
                gradient_r=[],
                gradient_c=[],
                safest_cell=(0, 0),
                max_distance=0.0,
                obstacle_count=0,
            )

        # If no obstacles, distance is distance to boundary or large constant
        if not obstacles:
            diag = math.hypot(rows, cols)
            empty_dist_mat = [[diag for _ in range(cols)] for _ in range(rows)]
            empty_grad_r = [[0.0 for _ in range(cols)] for _ in range(rows)]
            empty_grad_c = [[0.0 for _ in range(cols)] for _ in range(rows)]
            center = (rows // 2, cols // 2)
            return EuclideanDistanceTransformResult(
                grid_shape=grid_shape,
                distance_matrix=empty_dist_mat,
                gradient_r=empty_grad_r,
                gradient_c=empty_grad_c,
                safest_cell=center,
                max_distance=diag,
                obstacle_count=0,
            )

        # Initialize grid: 0 at obstacles, INF elsewhere
        grid: list[list[float]] = [
            [0.0 if (r, c) in obstacles else self.INF for c in range(cols)] for r in range(rows)
        ]

        # Step 1: Transform columns
        col_transformed: list[list[float]] = [[0.0] * cols for _ in range(rows)]
        for c in range(cols):
            col_f = [grid[r][c] for r in range(rows)]
            col_dt = self._edt_1d(col_f, rows)
            for r in range(rows):
                col_transformed[r][c] = col_dt[r]

        # Step 2: Transform rows
        sq_dist_mat: list[list[float]] = [[0.0] * cols for _ in range(rows)]
        for r in range(rows):
            row_dt = self._edt_1d(col_transformed[r], cols)
            for c in range(cols):
                sq_dist_mat[r][c] = row_dt[c]

        # Step 3: Compute Euclidean distances and find safest cell
        dist_mat: list[list[float]] = [[0.0] * cols for _ in range(rows)]
        max_dist = -1.0
        safest = (0, 0)
        for r in range(rows):
            for c in range(cols):
                d = math.sqrt(max(0.0, sq_dist_mat[r][c]))
                dist_mat[r][c] = d
                if d > max_dist:
                    max_dist = d
                    safest = (r, c)

        # Step 4: Compute continuous spatial gradient field (∇D)
        # Gradient points in the direction of increasing distance (away from obstacles)
        grad_r: list[list[float]] = [[0.0] * cols for _ in range(rows)]
        grad_c: list[list[float]] = [[0.0] * cols for _ in range(rows)]

        for r in range(rows):
            for c in range(cols):
                # Row gradient
                if rows == 1:
                    dr = 0.0
                elif r == 0:
                    dr = dist_mat[1][c] - dist_mat[0][c]
                elif r == rows - 1:
                    dr = dist_mat[r][c] - dist_mat[r - 1][c]
                else:
                    dr = (dist_mat[r + 1][c] - dist_mat[r - 1][c]) / 2.0
                grad_r[r][c] = dr

                # Col gradient
                if cols == 1:
                    dc = 0.0
                elif c == 0:
                    dc = dist_mat[r][1] - dist_mat[r][0]
                elif c == cols - 1:
                    dc = dist_mat[r][c] - dist_mat[r][c - 1]
                else:
                    dc = (dist_mat[r][c + 1] - dist_mat[r][c - 1]) / 2.0
                grad_c[r][c] = dc

        return EuclideanDistanceTransformResult(
            grid_shape=grid_shape,
            distance_matrix=dist_mat,
            gradient_r=grad_r,
            gradient_c=grad_c,
            safest_cell=safest,
            max_distance=max(0.0, max_dist),
            obstacle_count=len(obstacles),
        )

    def get_repulsive_force(
        self,
        r: int,
        c: int,
        result: EuclideanDistanceTransformResult,
        buffer_radius: float = 2.0,
    ) -> tuple[float, float]:
        """Compute repulsive force vector pushing away from obstacles if within safety buffer.

        Returns (force_r, force_c) normalized to point towards open clearance.
        """
        dist = result.get_distance(r, c)
        if dist >= buffer_radius or dist <= 0.0:
            return (0.0, 0.0)

        # Magnitude increases inversely with obstacle distance
        magnitude = (1.0 / max(0.1, dist)) - (1.0 / buffer_radius)
        # Force is along the gradient (direction of increasing clearance)
        gr = result.gradient_r[r][c]
        gc = result.gradient_c[r][c]
        norm = math.hypot(gr, gc)
        if norm > 1e-6:
            return (gr / norm * magnitude, gc / norm * magnitude)
        return (0.0, 0.0)
