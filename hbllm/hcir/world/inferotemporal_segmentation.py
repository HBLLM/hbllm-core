"""Inferotemporal Cortex (IT / Ventral Stream) Affordance Centroid Segmentation.

Modeled on primate ventral visual stream (V1 -> V2 -> V4 -> TEO -> IT):
1. Invariant Object Manifolds: Groups non-background sensory stimuli into discrete
   connected-component representations (object files / Kahneman's object tokens).
2. Medial Axis & Centroid Anchoring: Computes true geometric centroids and medial
   interior anchor coordinates, preventing out-of-boundary clicks on concave objects.
3. Affordance Salience Filtering: Filters out massive static frames/enclosures and
   pure background sheets, isolating interactive manipulanda (buttons, tiles, switches, tokens).
4. Saccadic Effector Pacing: Drastically constrains spatial search space from 4,096+
   arbitrary raster cells down to the K discrete object candidates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import label

logger = logging.getLogger(__name__)


@dataclass
class VentralObjectToken:
    """Discrete perceptual object representation derived from ventral stream processing."""

    object_id: int
    feature_id: int
    area: int
    centroid: tuple[int, int]
    anchor_coord: tuple[int, int]  # Guaranteed interior cell closest to centroid
    bounding_box: tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c)
    aspect_ratio: float
    is_compact: bool
    cells: list[tuple[int, int]] = field(default_factory=list)
    salience_score: float = 0.0


class InferotemporalSegmentationEngine:
    """Primate area IT ventral stream object parser and affordance centroid extractor."""

    @staticmethod
    def segment_objects(
        grid: np.ndarray,
        background_feature: int = 0,
        avatar_features: set[int] | None = None,
        min_area: int = 1,
        max_area: int = 256,
        connectivity: int = 4,
    ) -> list[VentralObjectToken]:
        """Segment 2D scene into discrete ventral object tokens.

        Args:
            grid: 2D numpy array representing the sensory visual frame.
            background_feature: Color index representing empty background.
            avatar_features: Features identified as the player's avatar (excluded).
            min_area: Minimum connected pixel count for valid object token.
            max_area: Maximum pixel count (to filter massive enclosing background walls).
            connectivity: 4 or 8 neighbor connectivity.

        Returns:
            List of parsed VentralObjectTokens sorted by perceptual salience.
        """
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            return []

        H, W = grid.shape
        av_set = avatar_features or set()
        total_cells = H * W

        tokens: list[VentralObjectToken] = []
        token_counter = 0

        # Unique foreground features
        unique_feats = np.unique(grid)

        # 4-connectivity or 8-connectivity structure
        struct = (
            np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]) if connectivity == 4 else np.ones((3, 3))
        )

        for feat in unique_feats:
            feat_val = int(feat)
            if feat_val == background_feature or feat_val in av_set:
                continue

            binary_mask = grid == feat_val
            labeled_mask, num_components = label(binary_mask, structure=struct)

            for comp_idx in range(1, num_components + 1):
                coords = np.argwhere(labeled_mask == comp_idx)
                area = len(coords)

                # Filter out microscopic noise or giant enclosing frames
                if area < min_area or area > max_area:
                    continue
                if area >= int(0.70 * total_cells):
                    continue

                min_r = int(np.min(coords[:, 0]))
                max_r = int(np.max(coords[:, 0]))
                min_c = int(np.min(coords[:, 1]))
                max_c = int(np.max(coords[:, 1]))

                h_box = max_r - min_r + 1
                w_box = max_c - min_c + 1
                aspect = float(h_box) / max(1.0, float(w_box))

                # Compute continuous center of mass
                mean_r = float(np.mean(coords[:, 0]))
                mean_c = float(np.mean(coords[:, 1]))
                centroid = (int(round(mean_r)), int(round(mean_c)))

                # Find the guaranteed interior coordinate closest to the continuous centroid
                cell_tuples = [(int(r), int(c)) for r, c in coords]
                anchor = min(
                    cell_tuples,
                    key=lambda p: (p[0] - mean_r) ** 2 + (p[1] - mean_c) ** 2,
                )

                # Compactness: ratio of area to bounding box area
                box_area = h_box * w_box
                compactness = area / max(1.0, box_area)
                is_compact = 0.5 <= compactness <= 1.0

                # Primate salience: compact, moderately sized manipulanda receive highest visual attention
                # S = 100 * compactness / (1 + log2(area))
                salience = 100.0 * compactness / (1.0 + np.log2(max(2.0, float(area))))

                token_counter += 1
                tokens.append(
                    VentralObjectToken(
                        object_id=token_counter,
                        feature_id=feat_val,
                        area=area,
                        centroid=centroid,
                        anchor_coord=anchor,
                        bounding_box=(min_r, min_c, max_r, max_c),
                        aspect_ratio=aspect,
                        is_compact=is_compact,
                        cells=cell_tuples,
                        salience_score=salience,
                    )
                )

        tokens.sort(key=lambda t: t.salience_score, reverse=True)
        return tokens

    @staticmethod
    def extract_affordance_anchors(
        grid: np.ndarray,
        background_feature: int = 0,
        avatar_features: set[int] | None = None,
        quiescent_coords: set[tuple[int, int]] | None = None,
        effective_coords: set[tuple[int, int]] | None = None,
        quiescent_features: set[int] | None = None,
        effective_features: set[int] | None = None,
        visit_counts: dict[str, int] | None = None,
    ) -> list[tuple[int, int, float]]:
        """Extract prioritized (row, col, score) click targets based on ventral stream object tokens.

        Args:
            grid: 2D scene grid.
            background_feature: Empty background index.
            avatar_features: Self-avatar feature set.
            quiescent_coords: Coordinates confirmed to cause zero environmental changes when clicked.
            effective_coords: Coordinates confirmed to induce state mutations when clicked.
            quiescent_features: Feature values confirmed ineffective.
            effective_features: Feature values confirmed effective.
            visit_counts: Visitation history for habituation / inhibition of return.

        Returns:
            List of (row, col, score) tuples sorted in descending priority.
        """
        tokens = InferotemporalSegmentationEngine.segment_objects(
            grid,
            background_feature=background_feature,
            avatar_features=avatar_features,
        )

        quiescent_c = quiescent_coords or set()
        effective_c = effective_coords or set()
        quiescent_f = quiescent_features or set()
        effective_f = effective_features or set()
        visits = visit_counts or {}

        candidates: list[tuple[int, int, float]] = []

        for token in tokens:
            ar, ac = token.anchor_coord
            if (ar, ac) in quiescent_c:
                continue

            score = token.salience_score

            # Boost if feature or coordinate has shown causal efficacy
            if (ar, ac) in effective_c:
                score += 150.0
            if token.feature_id in effective_f:
                score += 50.0
            if token.feature_id in quiescent_f:
                score -= 80.0

            # Primate Inhibition of Return (IOR) based on past interaction attempts
            v_count = visits.get(f"click_{ar}_{ac}", 0)
            ior_penalty = float(v_count) * 25.0 + (float(v_count) ** 2) * 10.0
            score -= ior_penalty

            candidates.append((ar, ac, score))

        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates
