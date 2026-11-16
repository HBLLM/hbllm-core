from __future__ import annotations

"""Ablated / Experimental ARC-AGI Morphology Solvers (WP-C4).

These heuristics (fractional scaling, pattern stamping, and crystallographic wallpaper tiling)
were specialized for 2D grid benchmark puzzles and are ablated from the core general intelligence loop.
Preserved here for benchmark compatibility and experimental evaluations.
"""

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy import ndimage

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate


@dataclass
class ScaleInferenceResult:
    """Estimated scale factors and residual fidelity between shapes/grids (W054)."""

    scale_r: float
    scale_c: float
    is_isotropic: bool
    residual_error: float

    def to_causal_predicate(self) -> CausalPredicate:
        """Export scale inference result as a declarative causal predicate."""
        return CausalPredicate(
            variable="scale_inference_result",
            operator="==",
            value=f"{self.scale_r}x{self.scale_c}",
            relation=f"iso:{self.is_isotropic}",
            target_entity=f"err:{self.residual_error:.4f}",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording scale inference result."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="scale_inference_result",
                node_data={
                    "scale_r": self.scale_r,
                    "scale_c": self.scale_c,
                    "is_isotropic": self.is_isotropic,
                    "residual_error": self.residual_error,
                },
            )
        ]


class FractionalScaleTransformer:
    """Fractional and integer spatial scaling, anisotropic resizing, and scale factor inference (W054)."""

    @staticmethod
    def scale_grid(
        grid: np.ndarray,
        scale_r: float,
        scale_c: float,
        order: int = 0,  # 0: Nearest-neighbor (preserves discrete tokens), 1: Bilinear
    ) -> np.ndarray:
        """Rescale a 2D array by arbitrary positive float scale factors."""
        if scale_r <= 0.0 or scale_c <= 0.0:
            raise ValueError(
                f"Scale factors must be positive: got scale_r={scale_r}, scale_c={scale_c}"
            )

        H, W = grid.shape
        target_h = max(1, int(round(H * scale_r)))
        target_w = max(1, int(round(W * scale_c)))

        zoom_factors = (target_h / float(H), target_w / float(W))
        # ndimage.zoom with order=0 preserves discrete symbolic token IDs without blurring
        scaled = ndimage.zoom(grid, zoom=zoom_factors, order=order, mode="nearest")
        return np.asarray(scaled, dtype=grid.dtype)

    @classmethod
    def infer_scale_factors(
        cls,
        src: np.ndarray,
        dst: np.ndarray,
    ) -> ScaleInferenceResult:
        """Infer the best-fitting anisotropic scale factors (scale_r, scale_c) between two patterns."""
        src_h, src_w = src.shape
        dst_h, dst_w = dst.shape

        scale_r = dst_h / float(max(1, src_h))
        scale_c = dst_w / float(max(1, src_w))
        is_iso = abs(scale_r - scale_c) < 1e-4

        # Verify by forward simulating scaled src
        simulated = cls.scale_grid(src, scale_r, scale_c, order=0)
        # Resize to match exactly if rounding differs
        if simulated.shape != dst.shape:
            pad_h = max(0, dst_h - simulated.shape[0])
            pad_w = max(0, dst_w - simulated.shape[1])
            simulated = np.pad(simulated, ((0, pad_h), (0, pad_w)), mode="edge")[:dst_h, :dst_w]

        mismatches = int(np.count_nonzero(simulated != dst))
        total_px = max(1, dst_h * dst_w)
        residual = mismatches / float(total_px)

        return ScaleInferenceResult(
            scale_r=round(scale_r, 4),
            scale_c=round(scale_c, 4),
            is_isotropic=is_iso,
            residual_error=round(residual, 4),
        )


@dataclass
class StampInstance:
    """An identified placement of a repeated stamp kernel in a canvas (W056)."""

    row: int
    col: int
    rotation_k: int = 0  # 90° clockwise rotations
    fidelity: float = 1.0

    def to_causal_predicate(self) -> CausalPredicate:
        """Export stamp instance placement as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"stamp_instance:{self.row}_{self.col}",
            operator="==",
            value=f"rot:{self.rotation_k}",
            relation=f"fidelity:{self.fidelity:.2f}",
            target_entity=f"({self.row},{self.col})",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording stamp placement."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"stamp_{self.row}_{self.col}",
                node_data={
                    "row": self.row,
                    "col": self.col,
                    "rotation_k": self.rotation_k,
                    "fidelity": self.fidelity,
                },
            )
        ]


class GenerativePatternStamper:
    """Generative pattern replication, kernel stamping, and motif duplication inference (W056)."""

    @staticmethod
    def find_stamp_placements(
        canvas: np.ndarray,
        stamp_kernel: np.ndarray,
        bg: int = 0,
        min_fidelity: float = 0.85,
    ) -> list[StampInstance]:
        """Discover spatial locations where the stamp kernel is duplicated across the canvas."""
        H, W = canvas.shape
        kh, kw = stamp_kernel.shape
        if kh > H or kw > W or kh == 0 or kw == 0:
            return []

        kernel_non_bg = stamp_kernel != bg
        kernel_cells_count = int(np.count_nonzero(kernel_non_bg))
        if kernel_cells_count == 0:
            return []

        placements: list[StampInstance] = []

        # Test across 4 canonical rotations
        for rot_k in range(4):
            k_rot = np.rot90(stamp_kernel, -rot_k)
            k_h, k_w = k_rot.shape
            mask_non_bg = k_rot != bg
            k_cells = int(np.count_nonzero(mask_non_bg))
            if k_cells == 0 or k_h > H or k_w > W:
                continue

            for r in range(H - k_h + 1):
                for c in range(W - k_w + 1):
                    patch = canvas[r : r + k_h, c : c + k_w]
                    matches = int(np.count_nonzero((patch == k_rot) & mask_non_bg))
                    fidelity = matches / float(k_cells)

                    if fidelity >= min_fidelity:
                        # Avoid duplicates from redundant rotations on symmetric kernels
                        already_found = any(
                            abs(p.row - r) <= 1 and abs(p.col - c) <= 1 and p.fidelity >= fidelity
                            for p in placements
                        )
                        if not already_found:
                            placements.append(
                                StampInstance(
                                    row=r,
                                    col=c,
                                    rotation_k=rot_k,
                                    fidelity=round(fidelity, 4),
                                )
                            )

        return placements

    @staticmethod
    def synthesize_stamped_canvas(
        canvas_shape: tuple[int, int],
        stamp_kernel: np.ndarray,
        placements: list[StampInstance],
        bg: int = 0,
    ) -> np.ndarray:
        """Render a synthesized canvas by stamping kernel instances at target coordinates."""
        canvas = np.full(canvas_shape, bg, dtype=stamp_kernel.dtype)
        H, W = canvas_shape

        for p in placements:
            k_rot = np.rot90(stamp_kernel, -p.rotation_k)
            kh, kw = k_rot.shape
            mask = k_rot != bg

            r_end = min(H, p.row + kh)
            c_end = min(W, p.col + kw)
            kr_end = r_end - p.row
            kc_end = c_end - p.col

            if kr_end > 0 and kc_end > 0:
                canvas_slice = canvas[p.row : r_end, p.col : c_end]
                k_slice = k_rot[:kr_end, :kc_end]
                m_slice = mask[:kr_end, :kc_end]
                canvas_slice[m_slice] = k_slice[m_slice]

        return canvas


class WallpaperSymmetryGroup(StrEnum):
    """The crystallographic planar wallpaper groups describing 2D repetitive tiling (W036, W106)."""

    P1 = "p1"  # Pure translation lattice
    P2 = "p2"  # Translations + 180° rotations
    PM = "pm"  # Translations + reflections along one axis
    P4 = "p4"  # Translations + 90° rotations
    P4M = "p4m"  # Translations + 90° rotations + reflections
    UNKNOWN = "unknown"


@dataclass
class WallpaperTilingModel:
    """Fitted 2D wallpaper group model for repeated motifs and infinite canvas tilings (W036, W106)."""

    group: WallpaperSymmetryGroup
    period_r: int
    period_c: int
    unit_cell: np.ndarray
    concordance_score: float
    coverage_ratio: float

    def to_causal_predicate(self) -> CausalPredicate:
        """Export wallpaper tiling model as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"wallpaper_tiling:{self.group.value}",
            operator="==",
            value=f"period:{self.period_r}x{self.period_c}",
            relation=f"score:{self.concordance_score:.2f}",
            target_entity=self.group.value,
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording wallpaper tiling model."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"wallpaper_{self.group.value}_{self.period_r}_{self.period_c}",
                node_data={
                    "group": self.group.value,
                    "period_r": self.period_r,
                    "period_c": self.period_c,
                    "concordance_score": self.concordance_score,
                    "coverage_ratio": self.coverage_ratio,
                },
            )
        ]


class WallpaperTilingInducer:
    """Discovers repeated visual motifs, 2D translation lattices, and wallpaper group symmetries (W036, W106)."""

    @staticmethod
    def discover_2d_lattice(canvas: np.ndarray, bg: int = 0) -> tuple[int, int]:
        """Discover dominant 2D spatial translation periods (T_r, T_c) via spatial autocorrelation."""
        H, W = canvas.shape
        if H < 2 or W < 2:
            return (H, W)

        non_bg = (canvas != bg).astype(float)
        if np.sum(non_bg) == 0:
            return (1, 1)

        best_pr, best_pc = H, W
        best_score_r, best_score_c = 0.0, 0.0

        for pr in range(1, H // 2 + 1):
            valid_rows = H - pr
            matches = np.count_nonzero(
                (canvas[:valid_rows, :] == canvas[pr:, :]) & (canvas[:valid_rows, :] != bg)
            )
            total = int(np.count_nonzero(canvas[:valid_rows, :] != bg))
            score = float(matches) / float(max(1, total))
            if score > 0.70 and score > best_score_r:
                best_score_r = score
                best_pr = pr

        for pc in range(1, W // 2 + 1):
            valid_cols = W - pc
            matches = np.count_nonzero(
                (canvas[:, :valid_cols] == canvas[:, pc:]) & (canvas[:, :valid_cols] != bg)
            )
            total = int(np.count_nonzero(canvas[:, :valid_cols] != bg))
            score = float(matches) / float(max(1, total))
            if score > 0.70 and score > best_score_c:
                best_score_c = score
                best_pc = pc

        return (best_pr, best_pc)

    @classmethod
    def fit_wallpaper_group(cls, canvas: np.ndarray, bg: int = 0) -> WallpaperTilingModel:
        """Fit the best planar wallpaper symmetry group and extract the minimal fundamental unit cell."""
        pr, pc = cls.discover_2d_lattice(canvas, bg=bg)
        H, W = canvas.shape
        unit_cell = canvas[:pr, :pc].copy()

        tiled = np.tile(unit_cell, (int(math.ceil(H / float(pr))), int(math.ceil(W / float(pc)))))[
            :H, :W
        ]

        non_bg_mask = canvas != bg
        total_non_bg = int(np.count_nonzero(non_bg_mask))
        if total_non_bg == 0:
            return WallpaperTilingModel(
                group=WallpaperSymmetryGroup.P1,
                period_r=pr,
                period_c=pc,
                unit_cell=unit_cell,
                concordance_score=1.0,
                coverage_ratio=0.0,
            )

        match_count = int(np.count_nonzero((canvas == tiled) & non_bg_mask))
        concordance = match_count / float(total_non_bg)

        group = WallpaperSymmetryGroup.P1
        if pr == pc and pr > 1:
            rot90 = np.rot90(unit_cell, 1)
            rot180 = np.rot90(unit_cell, 2)
            if np.array_equal(unit_cell, rot90):
                if np.array_equal(unit_cell, np.fliplr(unit_cell)):
                    group = WallpaperSymmetryGroup.P4M
                else:
                    group = WallpaperSymmetryGroup.P4
            elif np.array_equal(unit_cell, rot180):
                group = WallpaperSymmetryGroup.P2
        elif pr > 1 or pc > 1:
            if np.array_equal(unit_cell, np.fliplr(unit_cell)) or np.array_equal(
                unit_cell, np.flipud(unit_cell)
            ):
                group = WallpaperSymmetryGroup.PM

        coverage = total_non_bg / float(H * W)
        return WallpaperTilingModel(
            group=group,
            period_r=pr,
            period_c=pc,
            unit_cell=unit_cell,
            concordance_score=round(concordance, 4),
            coverage_ratio=round(coverage, 4),
        )

    @staticmethod
    def tile_canvas(
        unit_cell: np.ndarray,
        target_shape: tuple[int, int],
        group: WallpaperSymmetryGroup = WallpaperSymmetryGroup.P1,
    ) -> np.ndarray:
        """Tile the fundamental unit cell across a target canvas under the specified wallpaper symmetry group."""
        th, tw = target_shape
        uh, uw = unit_cell.shape
        if uh == 0 or uw == 0:
            return np.zeros(target_shape, dtype=int)

        reps_r = int(math.ceil(th / float(uh)))
        reps_c = int(math.ceil(tw / float(uw)))

        if group == WallpaperSymmetryGroup.PM:
            blocks = []
            for r in range(reps_r):
                row_blocks = []
                for c in range(reps_c):
                    cell = unit_cell.copy()
                    if c % 2 == 1:
                        cell = np.fliplr(cell)
                    row_blocks.append(cell)
                blocks.append(np.hstack(row_blocks))
            tiled = np.vstack(blocks)
        else:
            tiled = np.tile(unit_cell, (reps_r, reps_c))

        return tiled[:th, :tw]
