from __future__ import annotations

"""Lateral Occipital Complex (LOC) & Ventral Stream Morphological Filtering Faculty.

Biologically modeled on mammalian visual cortex (V1-V4, LOC) intermediate shape processing:
1. Saliency Map Synthesis:
   - Computes bottom-up visual saliency combining feature rarity, morphological contrast gradients,
     and spatial compactness (Itti & Koch saliency architecture).
2. Morphological Primitive Classification:
   - Classifies connected visual components into geometric primitives:
     - SOLID_BLOB: Filled compact entities (agents, boulders, obstacles).
     - HOLLOW_CONTAINER: Entities enclosing interior cavities (receptacles, rooms, goal bays).
     - LINEAR_STRUCTURE: 1-cell wide beams, corridors, or barriers.
     - SINGLETON: Isolated single-cell elements (keys, gems, triggers).
3. Enclosure & Containment Verification:
   - Evaluates whether spatial coordinates are enclosed within morphological boundaries.
4. Morphological Noise Filtering:
   - Removes transient salt-and-pepper noise while preserving semantic structure.
"""

import logging
import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


class MorphologyPrimitiveType(StrEnum):
    """Categorization of geometric morphological primitives."""

    SOLID_BLOB = "SOLID_BLOB"
    HOLLOW_CONTAINER = "HOLLOW_CONTAINER"
    LINEAR_STRUCTURE = "LINEAR_STRUCTURE"
    SINGLETON = "SINGLETON"


@dataclass
class MorphologicalEntity:
    """A segmented visual entity with rich morphological and topological features."""

    entity_id: str
    primitive_type: MorphologyPrimitiveType
    feature_id: int
    area: int
    bounding_box: tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c)
    centroid: tuple[int, int]
    holes_count: int = 0
    interior_coords: list[tuple[int, int]] = field(default_factory=list)
    contour_coords: list[tuple[int, int]] = field(default_factory=list)
    aspect_ratio: float = 1.0


class MorphologicalSaliencyEngine:
    """Intermediate ventral stream morphological filtering and bottom-up saliency faculty."""

    @staticmethod
    def compute_saliency_map(
        grid: np.ndarray,
        background_feature: int = 0,
    ) -> np.ndarray:
        """Compute a normalized [0.0, 1.0] bottom-up visual saliency map for the grid.

        Combines:
        1. Feature color rarity / pop-out (uncommon colors receive higher salience).
        2. Morphological gradient energy (edges between distinct entities).
        3. Compactness and isolation bonus.
        """
        H, W = grid.shape
        saliency = np.zeros((H, W), dtype=float)

        # 1. Color Frequency / Rarity Energy
        vals, counts = np.unique(grid, return_counts=True)
        total_pixels = float(grid.size)
        rarity_lookup: dict[int, float] = {}

        for val, count in zip(vals, counts, strict=False):
            feat = int(val)
            if feat == background_feature:
                rarity_lookup[feat] = 0.05
            else:
                freq = float(count) / total_pixels
                rarity_lookup[feat] = 1.0 / math.log2(2.0 + freq * 100.0)

        for r in range(H):
            for c in range(W):
                saliency[r, c] += rarity_lookup.get(int(grid[r, c]), 0.1)

        # 2. Morphological Gradient (Edges & Boundaries)
        struct = ndimage.generate_binary_structure(2, 1)
        dilated = ndimage.grey_dilation(grid, footprint=struct)
        eroded = ndimage.grey_erosion(grid, footprint=struct)
        gradient = (dilated != eroded).astype(float) * 0.4
        saliency += gradient

        # 3. Normalization to [0.0, 1.0]
        max_s = float(np.max(saliency))
        min_s = float(np.min(saliency))
        if max_s > min_s:
            saliency = (saliency - min_s) / (max_s - min_s)
        else:
            saliency = np.zeros((H, W), dtype=float)

        return saliency

    @staticmethod
    def extract_morphological_primitives(
        grid: np.ndarray,
        background_feature: int = 0,
    ) -> list[MorphologicalEntity]:
        """Segment and classify all visual entities into structural morphological primitives."""
        H, W = grid.shape
        unique_feats = [int(f) for f in np.unique(grid) if int(f) != background_feature]
        entities: list[MorphologicalEntity] = []

        struct = ndimage.generate_binary_structure(2, 2)  # 8-connectivity

        for feat in unique_feats:
            mask = grid == feat
            labeled, num_components = ndimage.label(mask, structure=struct)

            for comp_idx in range(1, num_components + 1):
                comp_mask = labeled == comp_idx
                coords = np.argwhere(comp_mask)
                area = int(len(coords))
                if area == 0:
                    continue

                min_r, min_c = int(np.min(coords[:, 0])), int(np.min(coords[:, 1]))
                max_r, max_c = int(np.max(coords[:, 0])), int(np.max(coords[:, 1]))
                bbox = (min_r, min_c, max_r, max_c)
                height = max_r - min_r + 1
                width = max_c - min_c + 1
                aspect_ratio = float(width) / float(max(1, height))

                centroid = (
                    int(round(float(np.mean(coords[:, 0])))),
                    int(round(float(np.mean(coords[:, 1])))),
                )

                # Check for hollow cavity / holes using binary fill
                filled = ndimage.binary_fill_holes(comp_mask)
                filled_area = int(np.count_nonzero(filled))
                holes_count = filled_area - area

                # Interior coords (points inside holes)
                hole_mask = filled & (~comp_mask)
                interior_pts = [(int(r), int(c)) for r, c in np.argwhere(hole_mask)]

                # Contour coords (erosion diff)
                eroded = ndimage.binary_erosion(comp_mask)
                contour_pts = [(int(r), int(c)) for r, c in np.argwhere(comp_mask & (~eroded))]

                # Classification
                if area == 1:
                    prim_type = MorphologyPrimitiveType.SINGLETON
                elif holes_count > 0 and height >= 3 and width >= 3:
                    prim_type = MorphologyPrimitiveType.HOLLOW_CONTAINER
                elif (height == 1 or width == 1) and max(height, width) >= 3:
                    prim_type = MorphologyPrimitiveType.LINEAR_STRUCTURE
                else:
                    prim_type = MorphologyPrimitiveType.SOLID_BLOB

                eid = f"morph_{feat}_{prim_type.value}_{min_r}_{min_c}"
                entities.append(
                    MorphologicalEntity(
                        entity_id=eid,
                        primitive_type=prim_type,
                        feature_id=feat,
                        area=area,
                        bounding_box=bbox,
                        centroid=centroid,
                        holes_count=holes_count,
                        interior_coords=interior_pts,
                        contour_coords=contour_pts,
                        aspect_ratio=aspect_ratio,
                    )
                )

        return entities

    @staticmethod
    def detect_enclosure(
        grid: np.ndarray,
        point: tuple[int, int],
        barrier_features: set[int],
    ) -> bool:
        """Verify whether a spatial coordinate is topologically enclosed inside barrier walls."""
        H, W = grid.shape
        pr, pc = point
        if not (0 <= pr < H and 0 <= pc < W):
            return False

        # Create binary wall mask
        is_wall = np.isin(grid, list(barrier_features))
        if is_wall[pr, pc]:
            return True

        # Flood fill passable area from point
        passable = ~is_wall
        labeled, _ = ndimage.label(passable)
        point_comp = labeled[pr, pc]
        if point_comp == 0:
            return False

        comp_mask = labeled == point_comp
        # If component touches any boundary of the grid, it is NOT enclosed
        touches_top = np.any(comp_mask[0, :])
        touches_bottom = np.any(comp_mask[-1, :])
        touches_left = np.any(comp_mask[:, 0])
        touches_right = np.any(comp_mask[:, -1])

        return not (touches_top or touches_bottom or touches_left or touches_right)

    @staticmethod
    def filter_salt_pepper_noise(
        grid: np.ndarray,
        background_feature: int = 0,
        min_component_area: int = 2,
    ) -> np.ndarray:
        """Remove isolated single-cell visual noise while preserving multi-cell entities."""
        H, W = grid.shape
        cleaned = np.copy(grid)
        unique_feats = [int(f) for f in np.unique(grid) if int(f) != background_feature]

        struct = ndimage.generate_binary_structure(2, 1)  # 4-connectivity

        for feat in unique_feats:
            mask = grid == feat
            labeled, num_components = ndimage.label(mask, structure=struct)
            for idx in range(1, num_components + 1):
                comp_size = int(np.count_nonzero(labeled == idx))
                if comp_size < min_component_area:
                    cleaned[labeled == idx] = background_feature

        return cleaned


# ── W054: Fractional & Generalized Spatial Scaling Engine ────────────────────


@dataclass
class ScaleInferenceResult:
    """Estimated scale factors and residual fidelity between shapes/grids (W054)."""

    scale_r: float
    scale_c: float
    is_isotropic: bool
    residual_error: float


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


# ── W056: Generative Pattern Copying, Duplication & Stamping ─────────────────


@dataclass
class StampInstance:
    """An identified placement of a repeated stamp kernel in a canvas (W056)."""

    row: int
    col: int
    rotation_k: int = 0  # 90° clockwise rotations
    fidelity: float = 1.0


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


# ── W036, W106: Repeated Motif Detection & Wallpaper Tiling Group Induction ──


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


class WallpaperTilingInducer:
    """Discovers repeated visual motifs, 2D translation lattices, and wallpaper group symmetries (W036, W106)."""

    @staticmethod
    def discover_2d_lattice(canvas: np.ndarray, bg: int = 0) -> tuple[int, int]:
        """Discover dominant 2D spatial translation periods (T_r, T_c) via spatial autocorrelation."""
        H, W = canvas.shape
        if H < 2 or W < 2:
            return (H, W)

        # Non-background binary presence
        non_bg = (canvas != bg).astype(float)
        if np.sum(non_bg) == 0:
            return (1, 1)

        best_pr, best_pc = H, W
        best_score_r, best_score_c = 0.0, 0.0

        # Row period sweep
        for pr in range(1, H // 2 + 1):
            valid_rows = H - pr
            matches = np.count_nonzero(
                (canvas[:valid_rows, :] == canvas[pr:, :]) & (canvas[:valid_rows, :] != bg)
            )
            total = np.count_nonzero(canvas[:valid_rows, :] != bg)
            score = matches / float(max(1, total))
            if score > 0.70 and score > best_score_r:
                best_score_r = score
                best_pr = pr

        # Col period sweep
        for pc in range(1, W // 2 + 1):
            valid_cols = W - pc
            matches = np.count_nonzero(
                (canvas[:, :valid_cols] == canvas[:, pc:]) & (canvas[:, :valid_cols] != bg)
            )
            total = np.count_nonzero(canvas[:, :valid_cols] != bg)
            score = matches / float(max(1, total))
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

        # Reconstruct canvas via periodic tiling
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

        # Detect internal symmetries in the unit cell
        group = WallpaperSymmetryGroup.P1
        if pr == pc and pr > 1:
            rot90 = np.rot90(unit_cell, 1)
            rot180 = np.rot90(unit_cell, 2)
            if np.array_equal(unit_cell, rot90):
                # Check reflection for p4m
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
            # Alternating reflection rows/columns
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
