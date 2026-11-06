from __future__ import annotations

"""Posterior Parietal Cortex (PPC) Potential Field Navigation & Affordance Routing.

Biologically & mathematically grounded in:
1. R.A. Andersen & C.A. Buneo (2002) - "Intentional maps in posterior parietal
   cortex" (Annu Rev Neurosci). Parietal LIP/7a coordinates spatial vector fields
   synthesizing goals, obstacles, and sensory motor affordances into intention vectors.
2. O. Khatib (1986) - "Real-time obstacle avoidance for manipulators and mobile
   robots" (IJRR). Artificial Potential Fields (APF) combining attractive goal
   gradients and repulsive obstacle potential fields.

In HBLLM, this module converts Euclidean Distance Transform fields and goal
coordinates into continuous potential navigation vectors, with automatic
orthogonal escape perturbations when local minima/saddle points are encountered.
"""

import math
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.spatial_distance_transform import EuclideanDistanceTransformResult


@dataclass
class PotentialFieldVector:
    """Navigational intention vector synthesized by posterior parietal cortex."""

    force_r: float
    force_c: float
    magnitude: float
    heading_direction: str
    is_stuck_in_local_minima: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_discrete_displacement(self) -> tuple[int, int]:
        """Discretize continuous force vector into unit grid step (dr, dc)."""
        if self.magnitude < 1e-4:
            return (0, 0)
        # Compare relative components
        abs_r, abs_c = abs(self.force_r), abs(self.force_c)
        if abs_r > 1.5 * abs_c:
            return (1 if self.force_r > 0 else -1, 0)
        elif abs_c > 1.5 * abs_r:
            return (0, 1 if self.force_c > 0 else -1)
        else:
            dr = 1 if self.force_r > 0 else (-1 if self.force_r < 0 else 0)
            dc = 1 if self.force_c > 0 else (-1 if self.force_c < 0 else 0)
            return (dr, dc)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export potential field vector as an HCIR causal predicate."""
        return CausalPredicate(
            variable="parietal:potential_field_vector",
            operator="==",
            value=self.heading_direction,
            relation=f"fr:{self.force_r:.2f}:fc:{self.force_c:.2f}:mag:{self.magnitude:.2f}:stuck:{self.is_stuck_in_local_minima}",
            target_entity="parietal_potential_field",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording navigational vector."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="parietal_navigation_vector",
                node_data={
                    "force_r": float(self.force_r),
                    "force_c": float(self.force_c),
                    "magnitude": float(self.magnitude),
                    "heading_direction": self.heading_direction,
                    "is_stuck_in_local_minima": self.is_stuck_in_local_minima,
                },
            )
        ]


class ParietalPotentialFieldNavigator:
    """Posterior Parietal Cortex continuous potential field vector synthesizer."""

    def __init__(
        self,
        attractive_gain: float = 1.0,
        repulsive_gain: float = 2.0,
        obstacle_buffer_radius: float = 2.0,
        minima_threshold: float = 0.08,
    ) -> None:
        self.k_att = attractive_gain
        self.k_rep = repulsive_gain
        self.buffer_radius = obstacle_buffer_radius
        self.minima_threshold = minima_threshold

    def compute_navigation_vector(
        self,
        agent_pos: tuple[int, int],
        goal_pos: tuple[int, int],
        edt_result: EuclideanDistanceTransformResult,
    ) -> PotentialFieldVector:
        """Synthesize attractive and repulsive potential forces into a single navigation vector."""
        ar, ac = agent_pos
        gr, gc = goal_pos

        # 1. Attractive force pointing directly towards goal
        delta_r = gr - ar
        delta_c = gc - ac
        goal_dist = math.hypot(delta_r, delta_c)

        if goal_dist < 1e-4:
            return PotentialFieldVector(
                force_r=0.0,
                force_c=0.0,
                magnitude=0.0,
                heading_direction="AT_GOAL",
                is_stuck_in_local_minima=False,
            )

        f_att_r = self.k_att * (delta_r / goal_dist)
        f_att_c = self.k_att * (delta_c / goal_dist)

        # 2. Repulsive force pushing away from nearest obstacles
        f_rep_r = 0.0
        f_rep_c = 0.0

        obs_dist = edt_result.get_distance(ar, ac)
        if 0.0 < obs_dist < self.buffer_radius:
            # Repulsive magnitude increases inversely with clearance distance
            rep_mag = self.k_rep * ((1.0 / max(0.1, obs_dist)) - (1.0 / self.buffer_radius))
            # Direction is along EDT gradient (pointing toward open space)
            grad_r = edt_result.gradient_r[ar][ac]
            grad_c = edt_result.gradient_c[ar][ac]
            grad_norm = math.hypot(grad_r, grad_c)
            if grad_norm > 1e-5:
                f_rep_r = rep_mag * (grad_r / grad_norm)
                f_rep_c = rep_mag * (grad_c / grad_norm)

        # 3. Superposition of attractive and repulsive fields
        tot_r = f_att_r + f_rep_r
        tot_c = f_att_c + f_rep_c
        magnitude = math.hypot(tot_r, tot_c)

        # 4. Local Minima / Saddle Point Detection & Orthogonal Escape
        is_stuck = False
        if magnitude < self.minima_threshold and goal_dist > 0.5:
            is_stuck = True
            # Inject orthogonal perturbation perpendicular to attractive direction
            perp_r = -f_att_c
            perp_c = f_att_r
            tot_r += perp_r * 0.8
            tot_c += perp_c * 0.8
            magnitude = math.hypot(tot_r, tot_c)

        # Determine semantic heading string
        heading = self._classify_heading(tot_r, tot_c)

        return PotentialFieldVector(
            force_r=tot_r,
            force_c=tot_c,
            magnitude=magnitude,
            heading_direction=heading,
            is_stuck_in_local_minima=is_stuck,
            metadata={
                "goal_dist": float(goal_dist),
                "obs_dist": float(obs_dist),
                "att_force": [float(f_att_r), float(f_att_c)],
                "rep_force": [float(f_rep_r), float(f_rep_c)],
            },
        )

    def _classify_heading(self, fr: float, fc: float) -> str:
        """Classify 2D force vector into cardinal/intercardinal direction."""
        if math.hypot(fr, fc) < 1e-4:
            return "STATIONARY"
        angle = math.degrees(
            math.atan2(fr, fc)
        )  # -180 to 180 (fr down positive, fc right positive)
        # Adjust compass:
        # fr > 0 is SOUTH (down in grid), fr < 0 is NORTH (up)
        # fc > 0 is EAST (right), fc < 0 is WEST (left)
        if -22.5 <= angle < 22.5:
            return "EAST"
        elif 22.5 <= angle < 67.5:
            return "SOUTH_EAST"
        elif 67.5 <= angle < 112.5:
            return "SOUTH"
        elif 112.5 <= angle < 157.5:
            return "SOUTH_WEST"
        elif angle >= 157.5 or angle < -157.5:
            return "WEST"
        elif -157.5 <= angle < -112.5:
            return "NORTH_WEST"
        elif -112.5 <= angle < -67.5:
            return "NORTH"
        else:
            return "NORTH_EAST"
