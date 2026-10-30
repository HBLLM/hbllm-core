"""Intuitive Physics & Ballistics Engine for HCIR World Kernel.

Modeled on mammalian cerebellar internal models and parietal VIP/LIP neural circuits:
1. Environmental Gravity & Passive Drift: Detects uncommanded downward acceleration
   and environmental force fields (g = [1, 0]).
2. Ground Support & Surface Affordance: Differentiates solid landing platforms from
   empty air / chasms.
3. Ballistic Arc & Projectile Simulation: Projects forward parabolic trajectories for
   jumping, leaping, falling, and projectile launches.
4. Declarative Neuro-Symbolic Integration: Converts ballistic landing forecasts into
   SymbolicSubgoal targets, formal CausalPredicates, and HCIRSymbolicWorldTheory rules.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from hbllm.hcir.skills.declarative import SpatialActionIntent, SymbolicSubgoal
from hbllm.hcir.world.cortex_causal import CausalPredicate, HCIRSymbolicWorldTheory

logger = logging.getLogger(__name__)


class GravityDirection(StrEnum):
    """Direction of the environmental gravitational force vector."""

    DOWN = "down"
    UP = "up"
    LEFT = "left"
    RIGHT = "right"
    NONE = "none"


class PhysicalState(StrEnum):
    """Physical state of an entity undergoing kinematic simulation."""

    GROUNDED = "grounded"
    AIRBORNE_ASCENT = "airborne_ascent"
    AIRBORNE_FALL = "airborne_fall"
    LETHAL_TERMINAL = "lethal_terminal"


@dataclass
class BallisticTrajectory:
    """A projected trajectory under physical forces (gravity, momentum)."""

    path: list[tuple[int, int]]
    landing_pos: tuple[int, int]
    is_grounded: bool
    is_terminal_hazard: bool = False
    physical_state: PhysicalState = PhysicalState.GROUNDED

    def __post_init__(self) -> None:
        if self.is_terminal_hazard:
            self.physical_state = PhysicalState.LETHAL_TERMINAL
        elif self.is_grounded:
            self.physical_state = PhysicalState.GROUNDED
        else:
            self.physical_state = PhysicalState.AIRBORNE_FALL

    def to_subgoal(
        self, intent: SpatialActionIntent = SpatialActionIntent.NAVIGATE
    ) -> SymbolicSubgoal:
        """Lift projected ballistic landing target into a formal declarative SymbolicSubgoal."""
        return SymbolicSubgoal(
            intent=intent,
            target_query={"target_position": self.landing_pos},
        )

    def to_causal_predicate(self) -> CausalPredicate:
        """Export ballistic trajectory endpoint as a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable="ballistic_landing",
            operator="==",
            value=self.landing_pos,
            relation="terminal_hazard" if self.is_terminal_hazard else "grounded",
            target_entity="avatar",
        )


class IntuitivePhysicsEngine:
    """Domain-agnostic intuitive physics, momentum, and ballistics engine."""

    def __init__(self, history_len: int = 16) -> None:
        self.history_len = history_len
        self.observed_transitions: deque[tuple[tuple[int, int], tuple[int, int], bool]] = deque(
            maxlen=history_len
        )
        self.gravity_evidence: int = 0
        self.has_gravity: bool = False
        self.gravity_vector: tuple[int, int] = (0, 0)
        self.field_evidence: dict[tuple[int, int], int] = {}
        self.known_platforms: set[tuple[int, int]] = set()

    def reset_episode(self) -> None:
        """Reset transient episode observations while preserving calibrated physics laws."""
        self.observed_transitions.clear()
        self.known_platforms.clear()

    def record_transition(
        self,
        prev_pos: tuple[int, int] | None,
        curr_pos: tuple[int, int] | None,
        is_displacement_action: bool,
        commanded_delta: tuple[int, int] = (0, 0),
    ) -> None:
        """Observe sensorimotor transition to detect uncommanded environmental forces."""
        if prev_pos is None or curr_pos is None:
            return

        actual_dr = curr_pos[0] - prev_pos[0]
        actual_dc = curr_pos[1] - prev_pos[1]

        # Calculate uncommanded environmental force vector
        # (what the world added beyond the agent's motor command)
        uncommanded_r = actual_dr - commanded_delta[0]
        uncommanded_c = actual_dc - commanded_delta[1]

        if uncommanded_r != 0 or uncommanded_c != 0:
            unit_r = 0 if uncommanded_r == 0 else (1 if uncommanded_r > 0 else -1)
            unit_c = 0 if uncommanded_c == 0 else (1 if uncommanded_c > 0 else -1)
            uncommanded_dir = (unit_r, unit_c)

            self.field_evidence[uncommanded_dir] = self.field_evidence.get(uncommanded_dir, 0) + 1
            if self.field_evidence[uncommanded_dir] >= 2:
                self.has_gravity = True
                self.gravity_vector = uncommanded_dir
                self.gravity_evidence = self.field_evidence[uncommanded_dir]
                logger.info(
                    "IntuitivePhysicsEngine: Confirmed environmental force field g=%s (evidence=%d).",
                    self.gravity_vector,
                    self.gravity_evidence,
                )
        elif not is_displacement_action and actual_dr == 0 and actual_dc == 0:
            # Stationary on a surface -> decay unconfirmed evidence
            if not self.has_gravity:
                for k in list(self.field_evidence.keys()):
                    if self.field_evidence[k] > 0:
                        self.field_evidence[k] -= 1

    def record_passive_entity_motion(
        self,
        prev_pos: tuple[int, int],
        curr_pos: tuple[int, int],
    ) -> None:
        """Observe unprompted motion of third-party / passive entities to induce environmental fields."""
        dr = curr_pos[0] - prev_pos[0]
        dc = curr_pos[1] - prev_pos[1]
        if dr != 0 or dc != 0:
            unit_r = 0 if dr == 0 else (1 if dr > 0 else -1)
            unit_c = 0 if dc == 0 else (1 if dc > 0 else -1)
            motion_dir = (unit_r, unit_c)
            self.field_evidence[motion_dir] = self.field_evidence.get(motion_dir, 0) + 1
            if self.field_evidence[motion_dir] >= 2:
                self.has_gravity = True
                self.gravity_vector = motion_dir
                self.gravity_evidence = self.field_evidence[motion_dir]

    def is_supported(
        self,
        r: int,
        c: int,
        grid: np.ndarray,
        barrier_features: set[int],
        static_barriers: set[tuple[int, int]] | None = None,
    ) -> bool:
        """Check if an entity at (r, c) rests on a solid surface / floor along gravity direction."""
        H, W = grid.shape
        gr, gc = self.gravity_vector if self.has_gravity else (1, 0)

        # Support is at the adjacent cell in the direction of the force
        supp_r = r + gr
        supp_c = c + gc

        # Boundary contact acts as physical support
        if supp_r >= H or supp_r < 0 or supp_c >= W or supp_c < 0:
            return True

        if static_barriers and (supp_r, supp_c) in static_barriers:
            return True

        val = int(grid[supp_r, supp_c])
        return val in barrier_features

    def project_fall(
        self,
        start_r: int,
        start_c: int,
        grid: np.ndarray,
        barrier_features: set[int],
        static_barriers: set[tuple[int, int]] | None = None,
        lethal_features: set[int] | None = None,
        lethal_coords: set[tuple[int, int]] | None = None,
        max_fall: int = 16,
    ) -> tuple[tuple[int, int], list[tuple[int, int]], bool]:
        """Project the passive trajectory under environmental force field until hitting support or hazard.

        Returns:
            (landing_pos, trajectory_path, is_lethal)
        """
        H, W = grid.shape
        cur_r = start_r
        cur_c = start_c
        trajectory: list[tuple[int, int]] = []
        gr, gc = self.gravity_vector if self.has_gravity else (1, 0)

        for _ in range(max_fall):
            if self.is_supported(cur_r, cur_c, grid, barrier_features, static_barriers):
                return (cur_r, cur_c), trajectory, False

            next_r = cur_r + gr
            next_c = cur_c + gc
            if not (0 <= next_r < H and 0 <= next_c < W):
                return (cur_r, cur_c), trajectory, False

            if static_barriers and (next_r, next_c) in static_barriers:
                return (cur_r, cur_c), trajectory, False

            cur_r = next_r
            cur_c = next_c
            trajectory.append((cur_r, cur_c))

            if lethal_features and int(grid[cur_r, cur_c]) in lethal_features:
                return (cur_r, cur_c), trajectory, True

            if lethal_coords and (cur_r, cur_c) in lethal_coords:
                return (cur_r, cur_c), trajectory, True

        return (cur_r, cur_c), trajectory, False

    def simulate_ballistic_jump(
        self,
        start_pos: tuple[int, int],
        horizontal_dir: int,
        jump_height: int,
        grid: np.ndarray,
        barrier_features: set[int],
        static_barriers: set[tuple[int, int]] | None = None,
    ) -> BallisticTrajectory:
        """Simulate a parabolic jump arc: ascent -> apex -> descent -> landing."""
        H, W = grid.shape
        r, c = start_pos
        path: list[tuple[int, int]] = []

        # 1. Ascent phase (upward impulse)
        for _ in range(jump_height):
            nr = r - 1
            nc = c + horizontal_dir
            if not (0 <= nr < H and 0 <= nc < W):
                break
            if static_barriers and (nr, nc) in static_barriers:
                break
            r, c = nr, nc
            path.append((r, c))

        # 2. Descent phase under gravity
        landing_pos, fall_path, is_lethal = self.project_fall(
            r, c, grid, barrier_features, static_barriers
        )
        path.extend(fall_path)

        return BallisticTrajectory(
            path=path,
            landing_pos=landing_pos,
            is_grounded=True,
            is_terminal_hazard=is_lethal,
        )

    def infer_gravity_direction(self) -> GravityDirection:
        """Infer canonical direction of the detected environmental gravity field."""
        if not self.has_gravity:
            return GravityDirection.NONE
        dr, dc = self.gravity_vector
        if dr > 0 and dc == 0:
            return GravityDirection.DOWN
        elif dr < 0 and dc == 0:
            return GravityDirection.UP
        elif dr == 0 and dc > 0:
            return GravityDirection.RIGHT
        elif dr == 0 and dc < 0:
            return GravityDirection.LEFT
        return GravityDirection.NONE

    def export_causal_predicates(self) -> list[CausalPredicate]:
        """Export calibrated physical laws into declarative Pearl causal predicates."""
        predicates: list[CausalPredicate] = []
        if self.has_gravity:
            predicates.append(
                CausalPredicate(
                    variable="environmental_gravity",
                    operator="==",
                    value=self.gravity_vector,
                    relation=str(self.infer_gravity_direction()),
                    target_entity="environment",
                )
            )
        return predicates

    def to_world_theory(self, barrier_features: set[int] | None = None) -> HCIRSymbolicWorldTheory:
        """Synthesize an empirical HCIRSymbolicWorldTheory capturing physical dynamics."""
        theory = HCIRSymbolicWorldTheory()
        if barrier_features:
            for bf in barrier_features:
                theory.induce_barrier(bf)
        theory.barrier_predicates.extend(self.export_causal_predicates())
        return theory
