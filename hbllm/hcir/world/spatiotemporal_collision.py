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
4. Declarative Neuro-Symbolic Integration: Synthesizes formal CausalPredicates and
   atomic TransactionOperations for state-space hazard tracking.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.cortex_causal import CausalPredicate

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

    def to_causal_predicate(self) -> CausalPredicate:
        """Export projected trajectory as a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable="projected_trajectory",
            operator="==",
            value=self.velocity,
            relation=f"len_{len(self.positions)}",
            target_entity=self.entity_id,
        )


class SpatiotemporalCollisionCones:
    """Predicts forward space-time collision cones for kinetic non-ego entities."""

    def __init__(self, default_horizon: int = 12) -> None:
        self.default_horizon = default_horizon
        self.projected_trajectories: list[ProjectedTrajectory] = []
        # (time_step) -> set of occupied positions
        self.step_occupied_cells: dict[int, set[tuple[int, int]]] = {}
        # (feature_id, time_step) -> set of occupied positions
        self.step_occupied_by_feat: dict[int, dict[int, set[tuple[int, int]]]] = {}
        # Persistent memory of entity velocities across frames (entity_id or feature_id -> velocity)
        self.persistent_velocities: dict[int, tuple[int, int]] = {}

    def reset_episode(self) -> None:
        """Clear all active trajectory projections for a new trial."""
        self.projected_trajectories.clear()
        self.step_occupied_cells.clear()
        self.step_occupied_by_feat.clear()
        self.persistent_velocities.clear()

    def update_trajectories(
        self,
        kinetic_entities: list[KineticEntity],
        static_barriers: set[tuple[int, int]],
        grid_shape: tuple[int, int],
        horizon: int | None = None,
        oriented_threats: Sequence[Any] | None = None,
        mobile_threat_features: set[int] | None = None,
        stride: int = 1,
    ) -> None:
        """Extrapolate forward space-time trajectories for all moving entities.

        Accurately models:
        1. Multi-cell physical footprint extent when detecting boundary/barrier rebounds.
        2. Persistent kinematic momentum across frames when velocity is temporarily unobserved.
        3. Immediate projection from perceived oriented patrollers even before frame diffs.
        """
        H, W = grid_shape
        T = horizon or self.default_horizon
        self.projected_trajectories.clear()
        self.step_occupied_cells.clear()
        self.step_occupied_by_feat.clear()

        for t in range(1, T + 1):
            self.step_occupied_cells[t] = set()

        # Update persistent velocities from kinetic entities
        for entity in kinetic_entities:
            feat_id = next(iter(entity.features)) if entity.features else None
            if feat_id is None or feat_id == 0:
                continue
            vr = int(round(entity.velocity[0]))
            vc = int(round(entity.velocity[1]))
            if vr != 0 or vc != 0:
                self.persistent_velocities[feat_id] = (vr, vc)

        # Track which features have been projected
        projected_feats: set[int] = set()

        for idx, entity in enumerate(kinetic_entities):
            feat_id = next(iter(entity.features)) if entity.features else None
            if feat_id is None or feat_id == 0:
                continue
            vr = int(round(entity.velocity[0]))
            vc = int(round(entity.velocity[1]))

            # Fall back to persistent velocity if entity temporarily stopped or uncalibrated
            if vr == 0 and vc == 0 and feat_id in self.persistent_velocities:
                vr, vc = self.persistent_velocities[feat_id]

            if vr == 0 and vc == 0:
                continue

            curr_r = int(round(entity.centroid[0]))
            curr_c = int(round(entity.centroid[1]))
            cur_vr, cur_vc = vr, vc
            simulated_positions: list[tuple[int, int]] = []

            # Compute relative offsets of all cells comprising the entity footprint
            rel_offsets: list[tuple[int, int]] = []
            if entity.cells:
                for cr, cc in entity.cells:
                    rel_offsets.append((cr - curr_r, cc - curr_c))
            else:
                rel_offsets = [(0, 0)]

            for step in range(1, T + 1):
                next_r = curr_r + cur_vr
                next_c = curr_c + cur_vc

                # Multi-cell extent barrier and boundary collision check
                rebound_r = any(
                    (next_r + dr_o < 0)
                    or (next_r + dr_o >= H)
                    or ((next_r + dr_o, curr_c + dc_o) in static_barriers)
                    for dr_o, dc_o in rel_offsets
                )
                if rebound_r:
                    cur_vr = -cur_vr

                rebound_c = any(
                    (curr_r + dr_o < 0)
                    or (curr_r + dr_o >= H)
                    or (next_c + dc_o < 0)
                    or (next_c + dc_o >= W)
                    or ((curr_r + dr_o, next_c + dc_o) in static_barriers)
                    for dr_o, dc_o in rel_offsets
                )
                if rebound_c:
                    cur_vc = -cur_vc

                if rebound_r or rebound_c:
                    next_r = curr_r + cur_vr
                    next_c = curr_c + cur_vc
                    # Clamp against secondary barrier collisions
                    if any(
                        (next_r + dr_o < 0)
                        or (next_r + dr_o >= H)
                        or ((next_r + dr_o, curr_c + dc_o) in static_barriers)
                        for dr_o, dc_o in rel_offsets
                    ):
                        next_r = curr_r
                    if any(
                        (curr_r + dr_o < 0)
                        or (curr_r + dr_o >= H)
                        or (next_c + dc_o < 0)
                        or (next_c + dc_o >= W)
                        or ((curr_r + dr_o, next_c + dc_o) in static_barriers)
                        for dr_o, dc_o in rel_offsets
                    ):
                        next_c = curr_c

                # Clamp within boundaries
                next_r = max(0, min(H - 1, next_r))
                next_c = max(0, min(W - 1, next_c))

                simulated_positions.append((next_r, next_c))

                # Add all cells in the simulated physical footprint
                for dr_o, dc_o in rel_offsets:
                    cell_r = next_r + dr_o
                    cell_c = next_c + dc_o
                    if 0 <= cell_r < H and 0 <= cell_c < W:
                        self.step_occupied_cells[step].add((cell_r, cell_c))
                        self.step_occupied_by_feat.setdefault(feat_id, {}).setdefault(
                            step, set()
                        ).add((cell_r, cell_c))

                curr_r, curr_c = next_r, next_c

            proj = ProjectedTrajectory(
                entity_id=f"kinetic_{idx}",
                feature_id=feat_id,
                initial_pos=(int(round(entity.centroid[0])), int(round(entity.centroid[1]))),
                velocity=(vr, vc),
                positions=simulated_positions,
            )
            self.projected_trajectories.append(proj)
            projected_feats.add(feat_id)

        # Incorporate oriented threats (e.g., directional patrollers) not captured by diffs
        if oriented_threats and mobile_threat_features:
            mob_feats = mobile_threat_features
            for o_idx, threat in enumerate(oriented_threats):
                t_feat = getattr(threat, "feature_id", 0)
                if t_feat not in mob_feats:
                    continue
                if t_feat in projected_feats:
                    continue
                facing = getattr(threat, "facing", (0, 0))
                if facing == (0, 0):
                    continue
                step_mult = max(1, stride)
                vr = facing[0] * step_mult
                vc = facing[1] * step_mult
                if vr == 0 and vc == 0 and t_feat in self.persistent_velocities:
                    vr, vc = self.persistent_velocities[t_feat]
                if vr == 0 and vc == 0:
                    continue

                t_pos = getattr(threat, "pos", (0, 0))
                curr_r, curr_c = t_pos
                cur_vr, cur_vc = vr, vc
                sim_pos: list[tuple[int, int]] = []
                # Oriented threat 3x3 footprint approximation centered on pos
                rel_offsets = [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]

                for step in range(1, T + 1):
                    next_r = curr_r + cur_vr
                    next_c = curr_c + cur_vc

                    rebound_r = any(
                        (next_r + dr_o < 0)
                        or (next_r + dr_o >= H)
                        or ((next_r + dr_o, curr_c + dc_o) in static_barriers)
                        for dr_o, dc_o in rel_offsets
                    )
                    if rebound_r:
                        cur_vr = -cur_vr
                    rebound_c = any(
                        (curr_r + dr_o < 0)
                        or (curr_r + dr_o >= H)
                        or (next_c + dc_o < 0)
                        or (next_c + dc_o >= W)
                        or ((curr_r + dr_o, next_c + dc_o) in static_barriers)
                        for dr_o, dc_o in rel_offsets
                    )
                    if rebound_c:
                        cur_vc = -cur_vc

                    if rebound_r or rebound_c:
                        next_r = curr_r + cur_vr
                        next_c = curr_c + cur_vc

                    next_r = max(0, min(H - 1, next_r))
                    next_c = max(0, min(W - 1, next_c))
                    sim_pos.append((next_r, next_c))

                    for dr_o, dc_o in rel_offsets:
                        cell_r = next_r + dr_o
                        cell_c = next_c + dc_o
                        if 0 <= cell_r < H and 0 <= cell_c < W:
                            self.step_occupied_cells[step].add((cell_r, cell_c))
                            self.step_occupied_by_feat.setdefault(t_feat, {}).setdefault(
                                step, set()
                            ).add((cell_r, cell_c))

                    # Dynamic Sentry Gaze Sweeping: project future hazard gaze cone along facing heading
                    if cur_vr != 0 or cur_vc != 0:
                        f_dr = 1 if cur_vr > 0 else (-1 if cur_vr < 0 else 0)
                        f_dc = 1 if cur_vc > 0 else (-1 if cur_vc < 0 else 0)
                        max_gaze_dist = min(2, max(1, step_mult))
                        for d in range(1, max_gaze_dist + 1):
                            gr = next_r + f_dr * d
                            gc = next_c + f_dc * d
                            if 0 <= gr < H and 0 <= gc < W and (gr, gc) not in static_barriers:
                                self.step_occupied_cells[step].add((gr, gc))
                                self.step_occupied_by_feat.setdefault(t_feat, {}).setdefault(
                                    step, set()
                                ).add((gr, gc))
                            else:
                                break

                    curr_r, curr_c = next_r, next_c

                self.projected_trajectories.append(
                    ProjectedTrajectory(
                        entity_id=f"oriented_{o_idx}",
                        feature_id=t_feat,
                        initial_pos=t_pos,
                        velocity=(vr, vc),
                        positions=sim_pos,
                    )
                )
                self.persistent_velocities[t_feat] = (vr, vc)
                projected_feats.add(t_feat)

    def is_collision_hazard(
        self,
        r: int,
        c: int,
        time_step: int,
        clearance: int = 0,
        ignored_features: set[int] | None = None,
    ) -> bool:
        """Check whether (r, c) or its clearance neighborhood is projected to be occupied by a kinetic entity at time_step."""
        if ignored_features:
            occupied: set[tuple[int, int]] = set()
            for feat, steps in self.step_occupied_by_feat.items():
                if feat not in ignored_features:
                    occupied.update(steps.get(time_step, set()))
        else:
            occupied = self.step_occupied_cells.get(time_step, set())

        if not occupied:
            return False
        if clearance == 0:
            return (r, c) in occupied
        for dr in range(-clearance, clearance + 1):
            for dc in range(-clearance, clearance + 1):
                if (r + dr, c + dc) in occupied:
                    return True
        return False

    def is_looming_threat(
        self,
        r: int,
        c: int,
        max_horizon: int = 3,
        clearance: int = 1,
        ignored_features: set[int] | None = None,
    ) -> bool:
        """Check if an oncoming kinetic hazard is projected to enter within clearance distance of (r, c) within max_horizon."""
        for step in range(1, max_horizon + 1):
            if self.is_collision_hazard(
                r, c, time_step=step, clearance=clearance, ignored_features=ignored_features
            ):
                return True
        return False

    def predict_intercept_step(
        self,
        target_pos: tuple[int, int],
        max_horizon: int = 12,
        clearance: int = 0,
    ) -> int | None:
        """Find the earliest step index when a projected hazard occupies target_pos."""
        for step in range(1, max_horizon + 1):
            if self.is_collision_hazard(
                target_pos[0], target_pos[1], time_step=step, clearance=clearance
            ):
                return step
        return None

    def get_collision_cone_at_step(self, time_step: int) -> set[tuple[int, int]]:
        """Return all space-time positions projected to be occupied at future time_step."""
        return self.step_occupied_cells.get(time_step, set())

    def get_swept_hazard_volume(self, start_step: int, end_step: int) -> set[tuple[int, int]]:
        """Return all spatial cells swept by moving entities across a time window."""
        swept: set[tuple[int, int]] = set()
        for t in range(start_step, end_step + 1):
            swept.update(self.step_occupied_cells.get(t, set()))
        return swept

    def export_causal_predicates(self) -> list[CausalPredicate]:
        """Export all projected kinetic hazard trajectories into declarative causal predicates."""
        return [proj.to_causal_predicate() for proj in self.projected_trajectories]

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR state updates for active hazard volumes."""
        ops: list[TransactionOperation] = []
        for proj in self.projected_trajectories:
            ops.append(
                TransactionOperation(
                    op=TransactionOp.MODIFY_NODE,
                    node_id=proj.entity_id,
                    changes={
                        "velocity": proj.velocity,
                        "projected_steps": len(proj.positions),
                        "initial_pos": proj.initial_pos,
                    },
                )
            )
        return ops
