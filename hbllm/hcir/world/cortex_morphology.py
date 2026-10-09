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
