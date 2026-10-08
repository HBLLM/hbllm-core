"""Premotor Cortex (PMv / pre-SMA) Spatiotemporal Collision Cones & Trajectory Extrapolation.

Modeled on premotor dynamic trajectory simulation and Spelke core knowledge:
1. Dynamic Trajectory Simulation: Takes kinetic entities identified by Area MT/V5
   (KineticStream) and extrapolates forward paths:
   x(t + dt) = x(t) + v * dt.
2. Obstacle & Boundary Reflection: Models physical collision rebounds off static
   boundaries and walls (v' = -v upon barrier contact).
3. Spatiotemporal Collision Cones: Projects hazard volumes into forward space-time (r, c, t).
   Allows the A* forward mental simulator and reactive motor arbiter to avoid walking into
   oncoming lethal threats and moving patrollers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hbllm.hcir.world.kinetic_stream import KineticEntity

logger = logging.getLogger(__name__)


@dataclass
class ProjectedTrajectory:
    """Predicted future positions of a kinetic entity over time."""

    entity_id: str
    feature_id: int
    initial_pos: tuple[int, int]
    velocity: tuple[int, int]
    positions: list[tuple[int, int]]  # positions[t - 1] is position at step t


class SpatiotemporalCollisionCones:
    """Predicts forward space-time collision cones for kinetic non-ego entities."""

    def __init__(self, default_horizon: int = 12) -> None:
        self.default_horizon = default_horizon
        self.projected_trajectories: list[ProjectedTrajectory] = []
        # (time_step) -> set of occupied positions
        self.step_occupied_cells: dict[int, set[tuple[int, int]]] = {}

    def reset_episode(self) -> None:
        """Clear all active trajectory projections for a new trial."""
        self.projected_trajectories.clear()
        self.step_occupied_cells.clear()

    def update_trajectories(
        self,
        kinetic_entities: list[KineticEntity],
        static_barriers: set[tuple[int, int]],
        grid_shape: tuple[int, int],
        horizon: int | None = None,
    ) -> None:
        """Extrapolate forward trajectories for all non-stationary entities."""
        H, W = grid_shape
        T = horizon or self.default_horizon
        self.projected_trajectories.clear()
        self.step_occupied_cells.clear()

        for t in range(1, T + 1):
            self.step_occupied_cells[t] = set()

        for idx, entity in enumerate(kinetic_entities):
            vr = int(round(entity.velocity[0]))
            vc = int(round(entity.velocity[1]))
            if vr == 0 and vc == 0:
                continue

            curr_r = int(round(entity.centroid[0]))
            curr_c = int(round(entity.centroid[1]))
            cur_vr, cur_vc = vr, vc
            simulated_positions: list[tuple[int, int]] = []

            for step in range(1, T + 1):
                next_r = curr_r + cur_vr
                next_c = curr_c + cur_vc

                # Check boundary or barrier rebound
                rebound_r = False
                rebound_c = False

                if next_r < 0 or next_r >= H or (next_r, curr_c) in static_barriers:
                    cur_vr = -cur_vr
                    rebound_r = True

                if next_c < 0 or next_c >= W or (curr_r, next_c) in static_barriers:
                    cur_vc = -cur_vc
                    rebound_c = True

                if rebound_r or rebound_c:
                    next_r = curr_r + cur_vr
                    next_c = curr_c + cur_vc

                # Clamp within boundaries
                next_r = max(0, min(H - 1, next_r))
                next_c = max(0, min(W - 1, next_c))

                simulated_positions.append((next_r, next_c))
                self.step_occupied_cells[step].add((next_r, next_c))

                curr_r, curr_c = next_r, next_c

            feat_id = next(iter(entity.features)) if entity.features else 0
            proj = ProjectedTrajectory(
                entity_id=f"kinetic_{idx}",
                feature_id=feat_id,
                initial_pos=(int(round(entity.centroid[0])), int(round(entity.centroid[1]))),
                velocity=(vr, vc),
                positions=simulated_positions,
            )
            self.projected_trajectories.append(proj)

    def is_collision_hazard(self, r: int, c: int, time_step: int) -> bool:
        """Check whether (r, c) is projected to be occupied by a kinetic entity at time_step."""
        return (r, c) in self.step_occupied_cells.get(time_step, set())

    def get_collision_cone_at_step(self, time_step: int) -> set[tuple[int, int]]:
        """Return all space-time positions projected to be occupied at future time_step."""
        return self.step_occupied_cells.get(time_step, set())

    def get_swept_hazard_volume(self, start_step: int, end_step: int) -> set[tuple[int, int]]:
        """Return all spatial cells swept by moving entities across a time window."""
        swept: set[tuple[int, int]] = set()
        for t in range(start_step, end_step + 1):
            swept.update(self.step_occupied_cells.get(t, set()))
        return swept
