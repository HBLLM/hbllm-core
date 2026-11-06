from __future__ import annotations

"""Parietal Egocentric Foveation & Spatial Saliency Field Compression.

Biologically & mathematically grounded in:
1. L. Itti & C. Koch (2001) - "A model of saliency-based visual attention for
   rapid scene analysis" (IEEE TPAMI).
2. E.L. Schwartz (1980) - Primate visual cortex non-uniform spatial mapping;
   high-acuity foveal representation with peripheral logarithmic pooling.

In HBLLM, this module foveates the sensory lattice around the agent avatar and
salient anchor targets, applying multi-scale spatial pooling (Fovea 1x1, Parafovea 2x2,
Periphery 4x4) to compress working memory while preserving 100% fine acuity in
the active manipulation zone.
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate


class FoveationResolution(StrEnum):
    """Spatial acuity tier for a region of the sensory lattice."""

    FOVEA = "FOVEA"  # 1x1 full resolution (active manipulation zone)
    PARAFOVEA = "PARAFOVEA"  # 2x2 pooled resolution (intermediate surroundings)
    PERIPHERY = "PERIPHERY"  # 4x4 pooled resolution (distant backdrop)


@dataclass
class FoveatedSaliencyMap:
    """Multi-resolution foveated representation of the sensory environment."""

    gaze_center: tuple[int, int]
    grid_shape: tuple[int, int]
    resolution_grid: list[list[FoveationResolution]]
    compressed_cells: dict[tuple[int, int], int]
    compression_ratio: float
    total_original_cells: int
    total_compressed_cells: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_feature_at(self, r: int, c: int) -> int:
        """Lookup perceived feature at coordinate (r, c), respecting foveation pooling."""
        if (r, c) in self.compressed_cells:
            return self.compressed_cells[(r, c)]
        # For pooled parafoveal or peripheral blocks, lookup block anchor
        res = self.resolution_grid[r][c]
        if res == FoveationResolution.PARAFOVEA:
            block_anchor = (r - (r % 2), c - (c % 2))
            return self.compressed_cells.get(block_anchor, 0)
        elif res == FoveationResolution.PERIPHERY:
            block_anchor = (r - (r % 4), c - (c % 4))
            return self.compressed_cells.get(block_anchor, 0)
        return 0

    def to_causal_predicate(self) -> CausalPredicate:
        """Export foveated representation summary as an HCIR causal predicate."""
        return CausalPredicate(
            variable="parietal:foveated_attention",
            operator="==",
            value=f"gaze:({self.gaze_center[0]},{self.gaze_center[1]})",
            relation=f"compression:{self.compression_ratio:.2f}:tokens:{self.total_compressed_cells}/{self.total_original_cells}",
            target_entity="parietal_foveated_attention",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording foveated saliency state."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="foveated_saliency_state",
                node_data={
                    "gaze_center": [self.gaze_center[0], self.gaze_center[1]],
                    "grid_shape": [self.grid_shape[0], self.grid_shape[1]],
                    "compression_ratio": float(self.compression_ratio),
                    "total_original_cells": int(self.total_original_cells),
                    "total_compressed_cells": int(self.total_compressed_cells),
                },
            )
        ]


class ParietalFoveatedAttentionEngine:
    """Parietal cortex multi-scale foveated attention and spatial compression engine."""

    def __init__(
        self,
        fovea_radius: int = 2,
        parafovea_radius: int = 5,
    ) -> None:
        self.fovea_radius = fovea_radius
        self.parafovea_radius = parafovea_radius

    def foveate_grid(
        self,
        grid: np.ndarray,
        gaze_center: tuple[int, int],
        secondary_anchors: list[tuple[int, int]] | None = None,
    ) -> FoveatedSaliencyMap:
        """Apply non-uniform foveation to a 2D sensory lattice."""
        rows, cols = grid.shape
        anchors = [gaze_center] + (secondary_anchors or [])

        # Step 1: Assign resolution tier to each cell based on proximity to nearest anchor
        res_grid: list[list[FoveationResolution]] = [
            [FoveationResolution.PERIPHERY for _ in range(cols)] for _ in range(rows)
        ]

        for r in range(rows):
            for c in range(cols):
                min_dist = min(math.hypot(r - ar, c - ac) for ar, ac in anchors)
                if min_dist <= float(self.fovea_radius):
                    res_grid[r][c] = FoveationResolution.FOVEA
                elif min_dist <= float(self.parafovea_radius):
                    res_grid[r][c] = FoveationResolution.PARAFOVEA
                else:
                    res_grid[r][c] = FoveationResolution.PERIPHERY

        # Step 2: Pool cells into compressed representation
        compressed_cells: dict[tuple[int, int], int] = {}
        processed_blocks: set[tuple[int, int]] = set()

        for r in range(rows):
            for c in range(cols):
                res = res_grid[r][c]

                if res == FoveationResolution.FOVEA:
                    # 1x1 full resolution
                    compressed_cells[(r, c)] = int(grid[r, c])

                elif res == FoveationResolution.PARAFOVEA:
                    # 2x2 pooled block
                    block_anchor = (r - (r % 2), c - (c % 2))
                    if block_anchor not in processed_blocks:
                        processed_blocks.add(block_anchor)
                        # Extract 2x2 block
                        block_vals: list[int] = []
                        for dr in range(2):
                            for dc in range(2):
                                br, bc = block_anchor[0] + dr, block_anchor[1] + dc
                                if 0 <= br < rows and 0 <= bc < cols:
                                    block_vals.append(int(grid[br, bc]))
                        # Priority pooling: non-zero mode or highest non-zero feature
                        non_zero = [v for v in block_vals if v != 0]
                        pooled_val = Counter(non_zero).most_common(1)[0][0] if non_zero else 0
                        compressed_cells[block_anchor] = pooled_val

                else:
                    # 4x4 peripheral block
                    block_anchor = (r - (r % 4), c - (c % 4))
                    if block_anchor not in processed_blocks:
                        processed_blocks.add(block_anchor)
                        block_vals = []
                        for dr in range(4):
                            for dc in range(4):
                                br, bc = block_anchor[0] + dr, block_anchor[1] + dc
                                if 0 <= br < rows and 0 <= bc < cols:
                                    block_vals.append(int(grid[br, bc]))
                        non_zero = [v for v in block_vals if v != 0]
                        pooled_val = Counter(non_zero).most_common(1)[0][0] if non_zero else 0
                        compressed_cells[block_anchor] = pooled_val

        total_original = rows * cols
        total_compressed = len(compressed_cells)
        ratio = max(0.0, 1.0 - (float(total_compressed) / float(max(1, total_original))))

        return FoveatedSaliencyMap(
            gaze_center=gaze_center,
            grid_shape=(rows, cols),
            resolution_grid=res_grid,
            compressed_cells=compressed_cells,
            compression_ratio=ratio,
            total_original_cells=total_original,
            total_compressed_cells=total_compressed,
        )
