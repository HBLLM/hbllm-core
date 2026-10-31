from __future__ import annotations

"""Foveal Saliency & Active Visual Saccading Faculty.

Modeled on mammalian Frontal Eye Fields (FEF), Superior Colliculus (SC), and Parietal Area LIP:
1. Active Saccade Generation:
   - When consecutive observation frames arrive, computes the visual saliency delta mask
     D = |grid_t - grid_{t-1}| > 0.
   - Computes the foveated saccadic bounding box [min_r, max_r, min_c, max_c] with attentional margin.
2. Gaze Centroid & Saccade Velocity:
   - Tracks the dynamic locus of visual attention and gaze shift velocity.
3. Selective Foveal Segmentation:
   - Rather than re-running full connected-component flood fills across the entire 64x64 sensory lattice,
     foveated attention only segments the actively mutating bounding box.
   - Preserves persistent IDs for unaffected static structures (walls, distant goals).
"""

import logging
from dataclasses import dataclass

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.cortex_perception import SpatialEntity

logger = logging.getLogger(__name__)


@dataclass
class FovealSaccadeFocus:
    """Active gaze focus and foveal bounding box for localized visual processing."""

    fovea_bounding_box: tuple[int, int, int, int]  # (min_r, max_r, min_c, max_c)
    gaze_centroid: tuple[float, float]
    saccade_velocity: tuple[float, float]
    changed_pixel_count: int
    is_foveated_update: bool = False

    def to_causal_predicate(self) -> CausalPredicate:
        """Export foveal saccade focus as a declarative Pearl causal predicate."""
        return CausalPredicate(
            variable="foveal_saccade_focus",
            operator="==",
            value=f"gaze:({self.gaze_centroid[0]:.1f},{self.gaze_centroid[1]:.1f})",
            relation=f"bbox:{self.fovea_bounding_box}:pixels:{self.changed_pixel_count}",
            target_entity="foveal_attention",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording visual gaze focus."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="foveal_gaze_focus",
                node_data={
                    "gaze_centroid": list(self.gaze_centroid),
                    "fovea_bbox": list(self.fovea_bounding_box),
                    "saccade_velocity": list(self.saccade_velocity),
                    "changed_pixels": self.changed_pixel_count,
                    "is_foveated": self.is_foveated_update,
                },
            )
        ]


class FovealSaccadeAttention:
    """Parietal Frontal Eye Field (FEF) & Superior Colliculus active visual saccading engine."""

    def __init__(self, fovea_padding: int = 2, max_fovea_ratio: float = 0.35) -> None:
        self.fovea_padding = fovea_padding
        self.max_fovea_ratio = max_fovea_ratio
        self.last_gaze_centroid: tuple[float, float] = (0.0, 0.0)
        self.cached_static_entities: list[SpatialEntity] = []
        self.last_grid: np.ndarray | None = None

    def reset(self) -> None:
        """Reset gaze focus upon scene transition or episode boundary."""
        self.last_gaze_centroid = (0.0, 0.0)
        self.cached_static_entities.clear()
        self.last_grid = None

    def compute_saccade(
        self,
        curr_grid: np.ndarray,
        prev_grid: np.ndarray | None = None,
    ) -> FovealSaccadeFocus:
        """Compute the active foveal bounding box and gaze shift vector from visual sensory diffs."""
        H, W = curr_grid.shape
        if prev_grid is None or prev_grid.shape != curr_grid.shape:
            # Full field view
            bbox = (0, H - 1, 0, W - 1)
            gaze = (float(H) / 2.0, float(W) / 2.0)
            self.last_gaze_centroid = gaze
            self.last_grid = curr_grid.copy()
            return FovealSaccadeFocus(
                fovea_bounding_box=bbox,
                gaze_centroid=gaze,
                saccade_velocity=(0.0, 0.0),
                changed_pixel_count=H * W,
                is_foveated_update=False,
            )

        diff_mask = curr_grid != prev_grid
        changed_pixels = int(np.sum(diff_mask))

        if changed_pixels == 0:
            # Zero sensory change: maintain previous gaze focus
            return FovealSaccadeFocus(
                fovea_bounding_box=(0, 0, 0, 0),
                gaze_centroid=self.last_gaze_centroid,
                saccade_velocity=(0.0, 0.0),
                changed_pixel_count=0,
                is_foveated_update=True,
            )

        coords = np.argwhere(diff_mask)
        min_r = max(0, int(np.min(coords[:, 0])) - self.fovea_padding)
        max_r = min(H - 1, int(np.max(coords[:, 0])) + self.fovea_padding)
        min_c = max(0, int(np.min(coords[:, 1])) - self.fovea_padding)
        max_c = min(W - 1, int(np.max(coords[:, 1])) + self.fovea_padding)

        mean_r = float(np.mean(coords[:, 0]))
        mean_c = float(np.mean(coords[:, 1]))
        current_gaze = (mean_r, mean_c)

        saccade_velocity = (
            current_gaze[0] - self.last_gaze_centroid[0],
            current_gaze[1] - self.last_gaze_centroid[1],
        )
        self.last_gaze_centroid = current_gaze
        self.last_grid = curr_grid.copy()

        fovea_area = (max_r - min_r + 1) * (max_c - min_c + 1)
        total_area = H * W
        is_foveated = (float(fovea_area) / float(total_area)) <= self.max_fovea_ratio

        return FovealSaccadeFocus(
            fovea_bounding_box=(min_r, max_r, min_c, max_c),
            gaze_centroid=current_gaze,
            saccade_velocity=saccade_velocity,
            changed_pixel_count=changed_pixels,
            is_foveated_update=is_foveated,
        )
