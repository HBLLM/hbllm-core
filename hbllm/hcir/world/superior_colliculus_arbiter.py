"""Superior Colliculus Multisensory Alignment & Saccadic Foveation Target Arbitration.

Biologically grounded in mammalian superior colliculus (SC) deep-layer topographic
sensorimotor integration and saccadic eye movement generation
(Stein & Meredith, 1993; Krauzlis et al., 2013):
1. Multisensory Topographic Priority Map:
   - Deep SC layers synthesize visual feature saliency, goal vector fields, and
     obstacle clearance margins into an aligned, 2D retinotopic priority landscape.
2. Center-Surround Lateral Inhibition (Mexican-Hat Tuning):
   - Enhances local peak salience while suppressing adjacent visual distractors
     within an inhibitory surround radius:
     W(d) = A_exc * exp(-d^2 / 2*sigma_exc^2) - A_inh * exp(-d^2 / 2*sigma_inh^2).
3. Discrete Saccadic Target Selection:
   - Arbitrates overt foveal gaze shifts and covert active inference attention to
     the single highest-priority spatial coordinate (r, c).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class SaccadeTarget:
    """A discrete spatial target chosen for active visual saccade or foveation."""

    position: tuple[int, int]
    saccade_priority: float  # [0, 1] composite priority score
    dominant_modality: str  # "visual_saliency", "goal_proximity", "obstacle_clearance"
    distance_from_avatar: float

    def to_causal_predicate(self) -> CausalPredicate:
        """Export saccade target as a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable="saccade_target",
            operator="==",
            value=f"pos:({self.position[0]},{self.position[1]}):prio:{self.saccade_priority:.2f}",
            relation="superior_colliculus_targeting",
            target_entity="avatar_gaze",
        )


@dataclass
class MultisensoryPriorityMap:
    """The unified 2D topographic sensorimotor map produced by SC deep layers."""

    priority_grid: np.ndarray
    primary_target: SaccadeTarget | None
    secondary_targets: list[SaccadeTarget] = field(default_factory=list)
    inhibition_radius: int = 2


class SuperiorColliculusTargetArbiter:
    """Sensorimotor arbiter generating topographic priority maps and selecting saccadic targets."""

    def __init__(
        self,
        visual_weight: float = 0.35,
        goal_weight: float = 0.45,
        clearance_weight: float = 0.20,
        inhibition_radius: int = 2,
    ) -> None:
        """Initialize superior colliculus target arbiter.

        Args:
            visual_weight: Weight allocated to raw perceptual visual saliency.
            goal_weight: Weight allocated to goal proximity / affordances.
            clearance_weight: Weight allocated to safe clearance pathways.
            inhibition_radius: Grid cell radius for center-surround suppression.
        """
        self.visual_weight = max(0.0, float(visual_weight))
        self.goal_weight = max(0.0, float(goal_weight))
        self.clearance_weight = max(0.0, float(clearance_weight))
        self.inhibition_radius = max(1, int(inhibition_radius))

    def build_priority_map(
        self,
        visual_saliency: np.ndarray,
        goal_positions: list[tuple[int, int]],
        obstacle_clearance: np.ndarray | None = None,
        avatar_pos: tuple[int, int] | None = None,
    ) -> MultisensoryPriorityMap:
        """Synthesize visual, goal, and clearance fields into an integrated SC priority map.

        Args:
            visual_saliency: 2D array of visual feature salience in [0, 1].
            goal_positions: List of known or candidate goal coordinates.
            obstacle_clearance: Optional 2D array of distance transform clearance.
            avatar_pos: Current avatar position (r, c).

        Returns:
            MultisensoryPriorityMap with primary and secondary saccadic targets.
        """
        h, w = visual_saliency.shape
        priority = np.zeros((h, w), dtype=np.float64)

        # 1. Visual Saliency Layer
        norm_vis = np.nan_to_num(visual_saliency, nan=0.0)
        if norm_vis.max() > 0:
            norm_vis = norm_vis / norm_vis.max()
        priority += self.visual_weight * norm_vis

        # 2. Goal Proximity Field (Gaussian bell around goal targets)
        goal_grid = np.zeros((h, w), dtype=np.float64)
        for gr, gc in goal_positions:
            if 0 <= gr < h and 0 <= gc < w:
                for r in range(h):
                    for c in range(w):
                        dist_sq = (r - gr) ** 2 + (c - gc) ** 2
                        goal_grid[r, c] += math.exp(-dist_sq / 18.0)
        if goal_grid.max() > 0:
            goal_grid = goal_grid / goal_grid.max()
        priority += self.goal_weight * goal_grid

        # 3. Obstacle Clearance Layer (prefer open safe zones over obstacle edges)
        if obstacle_clearance is not None and obstacle_clearance.shape == (h, w):
            norm_clr = np.nan_to_num(obstacle_clearance, nan=0.0)
            if norm_clr.max() > 0:
                norm_clr = norm_clr / norm_clr.max()
            priority += self.clearance_weight * norm_clr

        # Normalize total priority into [0, 1]
        max_p = priority.max()
        if max_p > 0:
            priority = priority / max_p

        # Extract discrete saccade peaks using non-maximum suppression
        candidates: list[tuple[float, tuple[int, int]]] = []
        for r in range(h):
            for c in range(w):
                val = float(priority[r, c])
                if val > 0.15:
                    candidates.append((val, (r, c)))

        candidates.sort(key=lambda x: x[0], reverse=True)

        selected_targets: list[SaccadeTarget] = []
        suppressed_mask = np.zeros((h, w), dtype=bool)

        av_r, av_c = avatar_pos if avatar_pos is not None else (h // 2, w // 2)

        for val, (r, c) in candidates:
            if suppressed_mask[r, c]:
                continue

            dr = r - av_r
            dc = c - av_c
            dist = math.sqrt(dr * dr + dc * dc)

            # Determine dominant modality
            vis_c = float(norm_vis[r, c]) * self.visual_weight
            goal_c = float(goal_grid[r, c]) * self.goal_weight
            dominant = "goal_proximity" if goal_c >= vis_c else "visual_saliency"

            target = SaccadeTarget(
                position=(r, c),
                saccade_priority=float(val),
                dominant_modality=dominant,
                distance_from_avatar=float(dist),
            )
            selected_targets.append(target)

            # Apply center-surround suppression mask around this peak
            r_min = max(0, r - self.inhibition_radius)
            r_max = min(h, r + self.inhibition_radius + 1)
            c_min = max(0, c - self.inhibition_radius)
            c_max = min(w, c + self.inhibition_radius + 1)
            suppressed_mask[r_min:r_max, c_min:c_max] = True

            if len(selected_targets) >= 5:
                break

        primary = selected_targets[0] if selected_targets else None
        secondaries = selected_targets[1:] if len(selected_targets) > 1 else []

        return MultisensoryPriorityMap(
            priority_grid=priority,
            primary_target=primary,
            secondary_targets=secondaries,
            inhibition_radius=self.inhibition_radius,
        )

    def select_primary_saccade_target(
        self,
        priority_map: MultisensoryPriorityMap,
    ) -> SaccadeTarget | None:
        """Select top-ranked saccade target from the multisensory priority map."""
        return priority_map.primary_target
