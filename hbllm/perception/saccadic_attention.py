"""Saccadic Visual Attention & Foveal Saliency System.

Modeled on biological primate visual pathways:
1. Retina & Superior Colliculus: Center-surround contrast (retinal ganglion receptive fields).
2. Early Visual Cortex (V1/V2): Gestalt object segmentation, morphological centroids, boundary junctions.
3. Parieto-Occipital & Pulvinar Attention Networks: Symmetry breaking, rare visual feature entropy,
   and temporal motion saccades.

Reduces continuous or combinatorial 2D sensory-spatial interaction spaces (e.g. click coordinates,
focal points, touch affordances) to a ranked set of discrete, salient foveal fixations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class FovealFixation:
    """A salient point of foveal interest in the visual sensory field."""

    r: int
    c: int
    salience: float
    reason: str
    feature_id: int
    area: int = 1
    bounding_box: tuple[int, int, int, int] = (0, 0, 0, 0)
    metadata: dict[str, float] = field(default_factory=dict)


class SaccadicAttentionSystem:
    """Biologically-inspired visual saliency and saccadic fixation selector.

    Operates across any 2D visual sensory matrix (games, UI, documents, diagrams, scenes)
    to identify candidate affordance hotspots without domain-specific puzzle rules.
    """

    def __init__(self, fovea_radius: int = 2) -> None:
        self.fovea_radius = fovea_radius

    def compute_saliency_map(
        self,
        grid: np.ndarray,
        prev_grid: np.ndarray | None = None,
        background_feature: int | None = None,
    ) -> np.ndarray:
        """Compute continuous 2D saliency density [0.0, 1.0] across the visual field."""
        H, W = grid.shape
        saliency = np.zeros((H, W), dtype=np.float32)

        if background_feature is None:
            # Estimate dominant background via modal border/global frequency
            border_vals = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
            vals, counts = np.unique(border_vals, return_counts=True)
            background_feature = int(vals[np.argmax(counts)])

        # 1. Feature rarity / Information Entropy
        # Rare color/feature values represent high surprise (epistemic value)
        unique_vals, counts = np.unique(grid, return_counts=True)
        total_pixels = H * W
        rarity_weights = {
            int(val): float(-math.log2(count / total_pixels + 1e-9))
            for val, count in zip(unique_vals, counts, strict=False)
        }
        max_rarity = max(rarity_weights.values()) if rarity_weights else 1.0

        for val, r_weight in rarity_weights.items():
            if val == background_feature:
                continue
            mask = grid == val
            saliency[mask] += (r_weight / (max_rarity + 1e-6)) * 0.35

        # 2. Retinal Center-Surround Contrast
        # High contrast difference between receptive center and surrounding ring
        padded = np.pad(grid, pad_width=self.fovea_radius, mode="edge")
        for r in range(H):
            for c in range(W):
                pr, pc = r + self.fovea_radius, c + self.fovea_radius
                center_val = padded[pr, pc]
                surround = padded[
                    pr - self.fovea_radius : pr + self.fovea_radius + 1,
                    pc - self.fovea_radius : pc + self.fovea_radius + 1,
                ]
                surround_diff = np.count_nonzero(surround != center_val)
                norm_contrast = surround_diff / float(surround.size - 1)
                saliency[r, c] += norm_contrast * 0.25

        # 3. Dynamic Temporal Motion Saccade (Visual Transient)
        if prev_grid is not None and prev_grid.shape == grid.shape:
            motion_mask = grid != prev_grid
            saliency[motion_mask] += 0.40

        # Normalize saliency map to [0.0, 1.0]
        max_val = float(np.max(saliency))
        if max_val > 0.0:
            saliency /= max_val

        return saliency

    def extract_fixations(
        self,
        grid: np.ndarray,
        prev_grid: np.ndarray | None = None,
        background_feature: int | None = None,
        top_k: int = 12,
        suppression_radius: int = 2,
    ) -> list[FovealFixation]:
        """Extract discrete, ranked foveal fixation candidates via non-maximum suppression."""
        H, W = grid.shape
        if background_feature is None:
            border_vals = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
            vals, counts = np.unique(border_vals, return_counts=True)
            background_feature = int(vals[np.argmax(counts)])

        saliency_map = self.compute_saliency_map(
            grid, prev_grid=prev_grid, background_feature=background_feature
        )

        # Segment discrete visual entities via Gestalt 4-connectivity
        visited = np.zeros((H, W), dtype=bool)
        entity_fixations: list[FovealFixation] = []

        for r in range(H):
            for c in range(W):
                val = int(grid[r, c])
                if visited[r, c] or val == background_feature:
                    continue

                # BFS connected component
                component_pixels: list[tuple[int, int]] = []
                queue: list[tuple[int, int]] = [(r, c)]
                visited[r, c] = True

                min_r, max_r = r, r
                min_c, max_c = c, c

                while queue:
                    curr_r, curr_c = queue.pop(0)
                    component_pixels.append((curr_r, curr_c))
                    min_r = min(min_r, curr_r)
                    max_r = max(max_r, curr_r)
                    min_c = min(min_c, curr_c)
                    max_c = max(max_c, curr_c)

                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if not visited[nr, nc] and int(grid[nr, nc]) == val:
                                visited[nr, nc] = True
                                queue.append((nr, nc))

                area = len(component_pixels)
                # Compute centroid
                mean_r = sum(p[0] for p in component_pixels) / float(area)
                mean_c = sum(p[1] for p in component_pixels) / float(area)
                centroid_r = int(round(mean_r))
                centroid_c = int(round(mean_c))

                # Ensure centroid pixel is inside the component or select closest valid pixel
                if (centroid_r, centroid_c) not in component_pixels:
                    closest = min(
                        component_pixels,
                        key=lambda p: (p[0] - mean_r) ** 2 + (p[1] - mean_c) ** 2,
                    )
                    centroid_r, centroid_c = closest

                # Entity salience is peak salience within its footprint plus rarity bonus
                comp_saliences = [saliency_map[pr, pc] for pr, pc in component_pixels]
                peak_salience = float(np.max(comp_saliences)) if comp_saliences else 0.0
                mean_salience = float(np.mean(comp_saliences)) if comp_saliences else 0.0

                # Compactness and small focal object bonus (foveal bias toward discrete objects)
                compactness_bonus = 1.0 / (1.0 + math.log2(max(1, area)))
                final_salience = peak_salience * 0.6 + mean_salience * 0.2 + compactness_bonus * 0.2

                reason = "gestalt_object"
                if area <= 4:
                    reason = "punctate_focal_entity"
                elif (
                    prev_grid is not None
                    and prev_grid.shape == grid.shape
                    and any(grid[pr, pc] != prev_grid[pr, pc] for pr, pc in component_pixels)
                ):
                    reason = "dynamic_motion_entity"

                entity_fixations.append(
                    FovealFixation(
                        r=centroid_r,
                        c=centroid_c,
                        salience=final_salience,
                        reason=reason,
                        feature_id=val,
                        area=area,
                        bounding_box=(min_r, max_r, min_c, max_c),
                    )
                )

        # Also detect structural symmetry breaks and grid junction anomalies
        symmetry_fixations = self._detect_symmetry_and_junction_fixations(
            grid, saliency_map, background_feature
        )
        entity_fixations.extend(symmetry_fixations)

        # Sort by salience descending
        entity_fixations.sort(key=lambda f: f.salience, reverse=True)

        # Non-Maximum Suppression (Inhibition of Return / Foveal Spacing)
        filtered_fixations: list[FovealFixation] = []
        for cand in entity_fixations:
            too_close = False
            for accepted in filtered_fixations:
                dist = abs(cand.r - accepted.r) + abs(cand.c - accepted.c)
                if dist < suppression_radius:
                    too_close = True
                    break
            if not too_close:
                filtered_fixations.append(cand)
                if len(filtered_fixations) >= top_k:
                    break

        return filtered_fixations

    def _detect_symmetry_and_junction_fixations(
        self,
        grid: np.ndarray,
        saliency_map: np.ndarray,
        background_feature: int,
    ) -> list[FovealFixation]:
        """Detect foveal fixation targets caused by symmetry discrepancies or structural junctions."""
        H, W = grid.shape
        fixations: list[FovealFixation] = []

        # Check horizontal and vertical reflection discrepancies
        # Human visual system rapidly attends to asymmetries in bilaterally symmetric contexts
        h_sym_diff = np.abs(grid.astype(float) - np.flip(grid, axis=1).astype(float))
        if 0 < np.count_nonzero(h_sym_diff) < (H * W * 0.15):
            # Sparse asymmetry points detected!
            diff_indices = np.argwhere(h_sym_diff > 0)
            for r, c in diff_indices[:6]:
                fixations.append(
                    FovealFixation(
                        r=int(r),
                        c=int(c),
                        salience=float(saliency_map[r, c] + 0.35),
                        reason="bilateral_symmetry_break",
                        feature_id=int(grid[r, c]),
                    )
                )

        v_sym_diff = np.abs(grid.astype(float) - np.flip(grid, axis=0).astype(float))
        if 0 < np.count_nonzero(v_sym_diff) < (H * W * 0.15):
            diff_indices = np.argwhere(v_sym_diff > 0)
            for r, c in diff_indices[:6]:
                fixations.append(
                    FovealFixation(
                        r=int(r),
                        c=int(c),
                        salience=float(saliency_map[r, c] + 0.35),
                        reason="vertical_symmetry_break",
                        feature_id=int(grid[r, c]),
                    )
                )

        return fixations
