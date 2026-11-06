"""Elastic Particle Dynamics & Angle-of-Incidence Deflection Engine (Milestone 3 Phase 3).

Inductively discovers ballistic particle kinematics and surface deflection mechanisms:
1. ParticleKinematicsSimulator: Simulates discrete 2D ray trajectories and elastic reflections.
2. ElasticParticleDeflectionOperator: Parameterized operator modeling ray propagation and bounce.
3. InductiveParticleDeflectionSynthesizer: Induces kinematics parameters directly from demonstration residuals.
4. Declarative Neuro-Symbolic Integration: Synthesizes formal CausalPredicates and SymbolicSubgoals.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

import numpy as np

from hbllm.hcir.skills.declarative import SpatialActionIntent, SymbolicSubgoal
from hbllm.hcir.world.cortex_causal import CausalPredicate
from hbllm.hcir.world.grid_operator import GridOperator, OperatorBinding

logger = logging.getLogger(__name__)


class DeflectionMode(StrEnum):
    """Kinematic deflection reflection modes upon striking barriers or boundaries."""

    HORIZONTAL_FLIP = "horizontal_flip"
    VERTICAL_FLIP = "vertical_flip"
    SPECULAR_NORMAL = "specular_normal"
    BOUNDARY_BOUNCE = "boundary_bounce"
    MULTI_REBOUND = "multi_rebound"


class ParticleKinematicsSimulator:
    """Simulates 2D ballistic discrete particle rays and surface reflections."""

    @staticmethod
    def simulate_trajectory(
        grid: np.ndarray,
        start_coord: tuple[int, int],
        initial_velocity: tuple[int, int],
        barrier_colors: set[int],
        deflection_mode: DeflectionMode | str = DeflectionMode.HORIZONTAL_FLIP,
        absorbing_colors: set[int] | None = None,
        max_steps: int = 150,
    ) -> list[tuple[int, int]]:
        """Simulate discrete particle trajectory until boundary exit, absorption, or cycle."""
        H, W = grid.shape
        absorbing = absorbing_colors or set()
        r, c = start_coord
        dr, dc = initial_velocity
        d_mode = str(deflection_mode)

        trajectory: list[tuple[int, int]] = []
        visited_states: set[tuple[int, int, int, int]] = set()

        for _ in range(max_steps):
            state = (r, c, dr, dc)
            if state in visited_states:
                break
            visited_states.add(state)

            nr, nc = r + dr, c + dc

            # Check boundary conditions
            if not (0 <= nr < H and 0 <= nc < W):
                if d_mode in (DeflectionMode.BOUNDARY_BOUNCE, DeflectionMode.MULTI_REBOUND):
                    if nr < 0 or nr >= H:
                        dr = -dr
                    if nc < 0 or nc >= W:
                        dc = -dc
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        break
                else:
                    break

            cell_val = int(grid[nr, nc])

            # Check absorption / target collision
            if cell_val in absorbing:
                break

            # Check barrier collision
            if cell_val in barrier_colors:
                if d_mode == DeflectionMode.HORIZONTAL_FLIP:
                    dc = -dc
                elif d_mode == DeflectionMode.VERTICAL_FLIP:
                    dr = -dr
                elif d_mode in (
                    DeflectionMode.SPECULAR_NORMAL,
                    DeflectionMode.MULTI_REBOUND,
                    DeflectionMode.BOUNDARY_BOUNCE,
                ):
                    vert_hit = (
                        0 <= r + dr < H and 0 <= c < W and int(grid[r + dr, c]) in barrier_colors
                    )
                    horiz_hit = (
                        0 <= r < H and 0 <= c + dc < W and int(grid[r, c + dc]) in barrier_colors
                    )
                    if vert_hit and not horiz_hit:
                        dr = -dr
                    elif horiz_hit and not vert_hit:
                        dc = -dc
                    else:
                        dc = -dc
                else:
                    dc = -dc

                # Advance after deflection
                refl_r, refl_c = r + dr, c + dc
                if (
                    0 <= refl_r < H
                    and 0 <= refl_c < W
                    and int(grid[refl_r, refl_c]) not in barrier_colors
                ):
                    r, c = refl_r, refl_c
                    if int(grid[r, c]) in absorbing:
                        break
                    trajectory.append((r, c))
                else:
                    break
            else:
                r, c = nr, nc
                trajectory.append((r, c))

        return trajectory


class ElasticParticleDeflectionOperator(GridOperator):
    """Operator that simulates ballistic particle ray emission and elastic deflection."""

    def __init__(
        self,
        name: str,
        emitter_color: int,
        beam_color: int,
        barrier_colors: tuple[int, ...],
        initial_velocity: tuple[int, int],
        deflection_mode: DeflectionMode | str = DeflectionMode.HORIZONTAL_FLIP,
        absorbing_colors: tuple[int, ...] = (),
        complexity: float = 2.5,
    ) -> None:
        self.name = name
        self.emitter_color = emitter_color
        self.beam_color = beam_color
        self.barrier_colors = barrier_colors
        self.initial_velocity = initial_velocity
        self.deflection_mode = deflection_mode
        self.absorbing_colors = absorbing_colors
        self.complexity = complexity

    def propose(self, scene: dict[str, Any], context: dict[str, Any]) -> list[OperatorBinding]:
        return [
            OperatorBinding(
                operator_name=self.name,
                params={
                    "emitter_color": self.emitter_color,
                    "beam_color": self.beam_color,
                    "barrier_colors": list(self.barrier_colors),
                    "initial_velocity": list(self.initial_velocity),
                    "deflection_mode": str(self.deflection_mode),
                    "absorbing_colors": list(self.absorbing_colors),
                },
                description=f"{self.name}(emitter={self.emitter_color}, beam={self.beam_color}, v={self.initial_velocity})",
                complexity=self.complexity,
            )
        ]

    def apply(self, grid: np.ndarray, binding: OperatorBinding | dict[str, Any]) -> np.ndarray:
        out = np.copy(grid)
        emitters = np.argwhere(grid == self.emitter_color)
        if len(emitters) == 0:
            return out

        start_r, start_c = int(emitters[0][0]), int(emitters[0][1])

        trajectory = ParticleKinematicsSimulator.simulate_trajectory(
            grid=grid,
            start_coord=(start_r, start_c),
            initial_velocity=self.initial_velocity,
            barrier_colors=set(self.barrier_colors),
            deflection_mode=self.deflection_mode,
            absorbing_colors=set(self.absorbing_colors),
        )

        for pr, pc in trajectory:
            if (pr, pc) != (start_r, start_c) and int(grid[pr, pc]) not in set(self.barrier_colors):
                out[pr, pc] = self.beam_color

        return out

    def to_causal_predicate(self) -> CausalPredicate:
        """Export operator kinematics as a formal HCIR CausalPredicate."""
        return CausalPredicate(
            variable="elastic_deflection",
            operator="==",
            value=self.initial_velocity,
            relation=str(self.deflection_mode),
            target_entity=f"emitter_{self.emitter_color}",
        )

    def to_subgoals(self) -> list[SymbolicSubgoal]:
        """Lift deflection operator intent into declarative planning subgoals."""
        return [
            SymbolicSubgoal(
                intent=SpatialActionIntent.NAVIGATE,
                target_query={"emitter_color": self.emitter_color, "beam_color": self.beam_color},
            )
        ]


class InductiveParticleDeflectionSynthesizer:
    """Induces particle ray kinematics and surface deflection parameters from demonstration residuals."""

    @classmethod
    def synthesize_from_residuals(
        cls, train_pairs: list[tuple[np.ndarray, np.ndarray]]
    ) -> ElasticParticleDeflectionOperator | None:
        """Deduce emitter, initial velocity vector, barrier colors, and deflection mode."""
        if not train_pairs:
            return None

        x0, y0 = train_pairs[0]
        diff_mask = x0 != y0
        residual_coords = [tuple(int(c) for c in pt) for pt in np.argwhere(diff_mask)]

        if len(residual_coords) == 0:
            return None

        # Residual beam must share a uniform beam color in y0
        beam_colors = np.unique(y0[diff_mask])
        if len(beam_colors) != 1:
            return None
        beam_color = int(beam_colors[0])

        H, W = x0.shape
        residual_set = set(residual_coords)

        # Identify candidate emitters: non-zero cells in x0 adjacent to residual path endpoints
        endpoints = [
            (r, c)
            for (r, c) in residual_coords
            if sum(
                1
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
                if (r + dr, c + dc) in residual_set
            )
            <= 1
        ]

        candidate_emitters: list[tuple[int, tuple[int, int]]] = []  # (emitter_color, velocity)

        for er, ec in endpoints:
            for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                nr, nc = er + dr, ec + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in residual_set:
                    col = int(x0[nr, nc])
                    if col != 0:
                        vr = 1 if er > nr else (-1 if er < nr else 0)
                        vc = 1 if ec > nc else (-1 if ec < nc else 0)
                        candidate_emitters.append((col, (vr, vc)))

        # Fallback candidate emitters: any non-zero color in x0
        all_colors = set(int(c) for c in np.unique(x0)) - {0}
        for col in all_colors:
            for vr in (-1, 0, 1):
                for vc in (-1, 0, 1):
                    if (vr, vc) != (0, 0):
                        candidate_emitters.append((col, (vr, vc)))

        deflection_modes = [
            DeflectionMode.HORIZONTAL_FLIP,
            DeflectionMode.VERTICAL_FLIP,
            DeflectionMode.SPECULAR_NORMAL,
            DeflectionMode.BOUNDARY_BOUNCE,
        ]

        for emitter_col, init_vel in candidate_emitters:
            remaining_colors = all_colors - {emitter_col}

            # Test obstacle sets
            for barrier_col in remaining_colors:
                absorbing_candidates = remaining_colors - {barrier_col}
                candidate_absorbing_sets: list[tuple[int, ...]] = [()]
                for abs_col in absorbing_candidates:
                    candidate_absorbing_sets.append((abs_col,))

                for absorbing_set in candidate_absorbing_sets:
                    for mode in deflection_modes:
                        candidate_op = ElasticParticleDeflectionOperator(
                            name="synthesized_elastic_particle_deflection",
                            emitter_color=emitter_col,
                            beam_color=beam_color,
                            barrier_colors=(barrier_col,),
                            initial_velocity=init_vel,
                            deflection_mode=mode,
                            absorbing_colors=absorbing_set,
                        )

                        # Test on Demo 0
                        b0 = candidate_op.propose({}, {"train_pairs": train_pairs})[0]
                        pred0 = candidate_op.apply(x0, b0)
                        if not np.array_equal(pred0, y0):
                            continue

                        # Out-of-construction verification across Demo 1..k
                        verified = True
                        for xi, yi in train_pairs[1:]:
                            pred_i = candidate_op.apply(xi, b0)
                            if not np.array_equal(pred_i, yi):
                                verified = False
                                break

                        if verified:
                            logger.info(
                                "Successfully induced elastic particle deflection operator: emitter=%d, v=%s, barrier=%d, beam=%d, mode=%s, absorbing=%s",
                                emitter_col,
                                init_vel,
                                barrier_col,
                                beam_color,
                                mode,
                                absorbing_set,
                            )
                            return candidate_op

        return None
