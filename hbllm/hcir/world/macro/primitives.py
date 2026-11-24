"""Domain-Agnostic Spatial and Kinematic Macro Action Primitives."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.spatial_planner import SpatialActionIntent
from hbllm.hcir.world.engine_state import get_engine_state_attr
from hbllm.hcir.world.frontopolar_subgoal_stack import SubgoalType
from hbllm.hcir.world.macro.models import (
    MacroAction,
    MacroActionStatus,
    _spatial_effector_payload,
)

if TYPE_CHECKING:
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

logger = logging.getLogger(__name__)


class NavigateTo(MacroAction):
    """Geodesic shortest-path navigation from current avatar locus to destination."""

    def __init__(
        self,
        destination: tuple[int, int],
        radius: int = 0,
        description: str = "",
    ) -> None:
        super().__init__(name=f"NavigateTo({destination})", intent=SpatialActionIntent.NAVIGATE)
        self.destination = (int(destination[0]), int(destination[1]))
        self.radius = max(0, radius)
        self.description = description or f"Navigate to {self.destination}"
        self.cached_path: list[tuple[Any, tuple[int, int]]] = []
        self.max_steps: int = 40
        self.consecutive_stuck: int = 0
        self.last_av_pos: tuple[int, int] | None = None

    def is_satisfied(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
    ) -> bool:
        if engine.avatar_pos is None:
            return False
        ar, ac = engine.avatar_pos
        dr = abs(ar - self.destination[0])
        dc = abs(ac - self.destination[1])
        if self.radius == 0:
            return dr == 0 and dc == 0
        return max(dr, dc) <= self.radius or (dr + dc) <= self.radius

    def is_feasible(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> bool:
        has_disp = any(engine.is_displacement_action(a) for a in available_actions)
        return has_disp and engine.avatar_pos is not None

    def step(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> tuple[Any, dict[str, Any] | None] | None:
        self.consecutive_steps += 1
        av_pos = engine.avatar_pos
        if av_pos is not None:
            if self.last_av_pos == av_pos:
                self.consecutive_stuck += 1
                if self.consecutive_stuck >= 3:
                    self.status = MacroActionStatus.FAILED
                    if engine.last_action is not None and engine.last_action in available_actions:
                        engine.failed_transitions.add((av_pos, int(engine.last_action)))
                    return None
            else:
                self.consecutive_stuck = 0
                self.last_av_pos = av_pos

        if self.consecutive_steps > self.max_steps:
            self.status = MacroActionStatus.FAILED
            logger.warning(
                "MacroAction %s timed out after %d steps", self.name, self.consecutive_steps
            )
            return None

        from hbllm.hcir.world.topological_graph_induction import GeodesicManifoldRouter

        av_pos = engine.avatar_pos
        if av_pos is None:
            self.status = MacroActionStatus.BLOCKED
            return available_actions[0], None

        failed_tx = get_engine_state_attr(engine, "failed_transitions", set())
        movable_actions = [
            (a, engine.action_dynamics[a].delta_r, engine.action_dynamics[a].delta_c)
            for a in available_actions
            if engine.is_displacement_action(a)
            and a in engine.action_dynamics
            and (av_pos, a) not in failed_tx
        ]

        av_feats = engine.avatar_features or (
            {engine.avatar_feature} if engine.avatar_feature is not None else set()
        )

        from hbllm.hcir.world.cortex_perception import PerceptionEngine

        threats = get_engine_state_attr(engine, "_prev_oriented_threats", [])
        if not threats:
            threats = PerceptionEngine.detect_oriented_threats(
                curr_grid,
                get_engine_state_attr(engine, "previous_entities", []),
                engine.bg_feature or 0,
                avatar_pos=av_pos,
            )

        H, W = curr_grid.shape
        bg_val = engine.bg_feature or 0
        wall_feats = set(engine.symbolic_theory.barrier_features)
        walkable_and_safe = set(engine.symbolic_theory.walkable_features) | set(
            get_engine_state_attr(engine, "verified_safe_features", set()) or set()
        )
        bg_is_impassable_void = len(walkable_and_safe) > 0 and bg_val not in walkable_and_safe
        if not bg_is_impassable_void:
            bg_count = int(np.count_nonzero(curr_grid == bg_val))
            ar, ac = (
                (av_pos[0], av_pos[1])
                if (av_pos and 0 <= av_pos[0] < H and 0 <= av_pos[1] < W)
                else (-1, -1)
            )
            avatar_on_bg = ar >= 0 and (
                int(curr_grid[ar, ac]) == bg_val
                or any(
                    0 <= ar + dr_a < H
                    and 0 <= ac + dc_a < W
                    and int(curr_grid[ar + dr_a, ac + dc_a]) == bg_val
                    for dr_a, dc_a in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                )
            )
            if bg_count / max(1, H * W) >= 0.20 and not avatar_on_bg:
                bg_is_impassable_void = True

        grid_walls: set[tuple[int, int]] = set(engine.learned_barriers)
        for rr in range(H):
            for cc in range(W):
                c_val = int(curr_grid[rr, cc])
                if c_val in wall_feats or (
                    bg_is_impassable_void and c_val == bg_val and (rr, cc) != self.destination
                ):
                    grid_walls.add((rr, cc))

        # Dead-end corridor reduction across dynamic action strides
        max_stride = 1
        for a in available_actions:
            if a in engine.action_dynamics:
                dyn = engine.action_dynamics[a]
                max_stride = max(max_stride, abs(dyn.delta_r), abs(dyn.delta_c))

        passable_grid = np.ones((H, W), dtype=bool)
        for w in grid_walls:
            if 0 <= w[0] < H and 0 <= w[1] < W:
                passable_grid[w[0], w[1]] = False
        pruned_topology = PerceptionEngine.prune_topological_dead_ends(
            passable=passable_grid,
            goals={self.destination},
            avatar_pos=av_pos,
            step_size=max_stride,
        )
        effective_walls = set(grid_walls)
        for rr in range(H):
            for cc in range(W):
                if not pruned_topology[rr, cc]:
                    effective_walls.add((rr, cc))

        wall_feats = set(engine.symbolic_theory.barrier_features)
        if not wall_feats and grid_walls:
            for wr, wc in list(grid_walls)[:20]:
                wall_feats.add(int(curr_grid[wr, wc]))

        # Plan next geodesic step
        plan = GeodesicManifoldRouter.route_geodesic(
            curr_grid=curr_grid,
            start_pos=av_pos,
            goals=[self.destination],
            static_barriers=engine.learned_barriers,
            grid_walls=effective_walls,
            unknown_entity_cells=set(),
            oriented_threats=threats,
            movable_actions=movable_actions,
            bg=bg_val,
            barrier_features=wall_feats,
            avatar_features=av_feats,
            max_candidate_goals=1,
            transition_kb=engine.symbolic_theory.transition_kb,
        )

        if plan and len(plan) > 0:
            next_act, next_pos = plan[0]
            if next_act in available_actions and (av_pos, next_act) not in failed_tx:
                self.status = MacroActionStatus.RUNNING
                self.next_predicted_pos = next_pos
                from collections import deque

                from hbllm.hcir.world.cortex_planner import MentalSimulationStep

                _plan_deque = deque(
                    [
                        MentalSimulationStep(
                            action=act,
                            predicted_avatar_pos=pos,
                            expected_mutation="macro_geodesic_step",
                            is_deterministic_geodesic=True,
                        )
                        for act, pos in plan[1:]
                    ]
                )
                from hbllm.hcir.world.engine_state import StateDelta

                _d = StateDelta()
                _d.set("mental_plan", _plan_deque)
                engine.apply_delta(_d, author="macro_actions")
                return next_act, None

        self.status = MacroActionStatus.BLOCKED
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. InteractWith Operator (Actuate / Targeted Effector)
# ─────────────────────────────────────────────────────────────────────────────


class InteractWith(MacroAction):
    """Allocentric spatial effector command (e.g. actuate widget, toggle optical mirror)."""

    def __init__(
        self,
        target_coord: tuple[int, int],
        effector_action: Any | None = None,
        description: str = "",
        contact_radius: int = 1,
    ) -> None:
        super().__init__(name=f"InteractWith({target_coord})", intent=SpatialActionIntent.INTERACT)
        self.target_coord = (int(target_coord[0]), int(target_coord[1]))
        self.effector_action = effector_action
        self.description = description or f"Interact with coordinate {self.target_coord}"
        self.has_executed: bool = False
        self.contact_radius = contact_radius
        self.nav_op: NavigateTo | None = None

    def is_satisfied(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
    ) -> bool:
        # Satisfied once executed and target was registered as visited/quiescent
        return self.has_executed

    def is_feasible(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> bool:
        if self.effector_action is not None and self.effector_action in available_actions:
            return True
        if any(engine.is_spatial_effector(a) for a in available_actions):
            return True
        # Fallback to discrete interact / transform affordances
        return any(
            not engine.is_displacement_action(a) and not engine.is_spatial_effector(a)
            for a in available_actions
        )

    def step(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> tuple[Any, dict[str, Any] | None] | None:
        # Embodied Contact Prior:
        # If displacement actions exist and an avatar is localized, physical interaction
        # requires approaching the contact manifold unless already proven to act distally at range.
        has_disp = engine.has_displacement_capabilities(available_actions)
        if has_disp and engine.avatar_pos is not None:
            dist = abs(engine.avatar_pos[0] - self.target_coord[0]) + abs(
                engine.avatar_pos[1] - self.target_coord[1]
            )
            is_proven_remote = self.target_coord in engine.effective_actuation_targets
            if dist > self.contact_radius and not is_proven_remote:
                if self.nav_op is None:
                    self.nav_op = NavigateTo(
                        destination=self.target_coord,
                        radius=self.contact_radius,
                        description=f"Approach {self.target_coord} for interaction",
                    )
                if not self.nav_op.is_satisfied(engine, curr_grid):
                    nav_step = self.nav_op.step(engine, curr_grid, available_actions)
                    if nav_step is not None:
                        return nav_step

        eff = self.effector_action
        if eff is None or eff not in available_actions:
            eff_candidates = [a for a in available_actions if engine.is_spatial_effector(a)]
            if eff_candidates:
                eff = eff_candidates[0]
            else:
                discrete_interacts = [
                    a
                    for a in available_actions
                    if not engine.is_displacement_action(a) and not engine.is_spatial_effector(a)
                ]
                if discrete_interacts:
                    eff = discrete_interacts[0]
                    self.has_executed = True
                    self.status = MacroActionStatus.SATISFIED
                    return eff, None
                self.status = MacroActionStatus.FAILED
                return available_actions[0], None

        data = _spatial_effector_payload(engine, eff, self.target_coord)
        self.has_executed = True
        self.status = MacroActionStatus.SATISFIED
        return eff, data


# ─────────────────────────────────────────────────────────────────────────────
# 3. SwitchFocus Operator (Multi-Agent / Multi-Body Handover)
# ─────────────────────────────────────────────────────────────────────────────


class SwitchFocus(MacroAction):
    """Executive handover of control to the partner agent or slider."""

    def __init__(
        self,
        switch_action: Any | None = None,
        target_feature: int | None = None,
        description: str = "",
    ) -> None:
        super().__init__(name="SwitchFocus", intent=SubgoalType.SWITCH_FOCUS)
        self.switch_action = switch_action
        self.target_feature = target_feature
        self.description = description or "Switch active agent focus"
        self.executed: bool = False
        self.initial_avatar_pos: tuple[int, int] | None = None

    def is_satisfied(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
    ) -> bool:
        if not self.executed:
            return False
        # Satisfied if locus changed or feature changed
        if self.target_feature is not None:
            return engine.avatar_feature == self.target_feature
        return engine.avatar_pos != self.initial_avatar_pos or self.executed

    def is_feasible(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> bool:
        if self.switch_action is not None:
            return self.switch_action in available_actions
        return any(
            engine.is_focus_switch_action(a) or a in engine.focus_switch_actions
            for a in available_actions
        )

    def step(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> tuple[Any, dict[str, Any] | None]:
        act = self.switch_action
        if act is None or act not in available_actions:
            candidates = [
                a
                for a in available_actions
                if engine.is_focus_switch_action(a) or a in engine.focus_switch_actions
            ]
            if candidates:
                act = candidates[0]
            else:
                act = available_actions[0]

        self.initial_avatar_pos = engine.avatar_pos
        self.executed = True
        self.status = MacroActionStatus.SATISFIED
        # Clear transient single-agent failed transitions upon intentional focus switch
        engine.failed_transitions.clear()
        return act, None


# ─────────────────────────────────────────────────────────────────────────────
# 4. PickUpCargo Operator (Dock with Movable Item)
# ─────────────────────────────────────────────────────────────────────────────


class PickUpCargo(MacroAction):
    """Navigate adjacent to a manipulandum/cargo and bind it to the body schema."""

    def __init__(
        self,
        cargo_pos: tuple[int, int],
        cargo_feature: int | None = None,
        approach_radius: int = 1,
        description: str = "",
    ) -> None:
        super().__init__(name=f"PickUpCargo({cargo_pos})", intent=SpatialActionIntent.PICKUP)
        self.cargo_pos = (int(cargo_pos[0]), int(cargo_pos[1]))
        self.cargo_feature = cargo_feature
        self.approach_radius = approach_radius
        self.description = description or f"Pick up cargo at {self.cargo_pos}"
        self.nav_op = NavigateTo(destination=self.cargo_pos, radius=approach_radius)

    def is_satisfied(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
    ) -> bool:
        if self.cargo_feature is not None and engine.working_memory.body_schema.is_holding(
            self.cargo_feature
        ):
            return True
        if engine.working_memory.body_schema.has_manipulandum():
            return True
        if engine.avatar_pos is None:
            return False
        # Also satisfied if avatar is touching cargo
        dr = abs(engine.avatar_pos[0] - self.cargo_pos[0])
        dc = abs(engine.avatar_pos[1] - self.cargo_pos[1])
        return dr + dc <= self.approach_radius and (
            engine.avatar_pos == self.cargo_pos or self.consecutive_steps > 0
        )

    def is_feasible(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> bool:
        return self.nav_op.is_feasible(engine, curr_grid, available_actions)

    def step(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> tuple[Any, dict[str, Any] | None] | None:
        if not self.nav_op.is_satisfied(engine, curr_grid):
            return self.nav_op.step(engine, curr_grid, available_actions)

        # Avatar is adjacent to cargo!
        av_pos = engine.avatar_pos
        if av_pos is not None:
            interact_candidates = [
                a
                for a in available_actions
                if not engine.is_displacement_action(a) and not engine.is_spatial_effector(a)
            ]
            if interact_candidates:
                chosen = 5 if 5 in interact_candidates else interact_candidates[0]
                self.status = MacroActionStatus.SATISFIED
                if self.cargo_feature is not None:
                    engine.working_memory.body_schema.acquire_tool(
                        feature_id=self.cargo_feature,
                        step=engine.step_counter,
                    )
                return chosen, None

            dr = int(np.sign(self.cargo_pos[0] - av_pos[0]))
            dc = int(np.sign(self.cargo_pos[1] - av_pos[1]))
            for a, dyn in engine.action_dynamics.items():
                if a in available_actions and dyn.is_displacement_action():
                    if dyn.get_displacement() == (dr, dc):
                        self.status = MacroActionStatus.SATISFIED
                        return a, None

        return available_actions[0], None


# ─────────────────────────────────────────────────────────────────────────────
# 5. DepositCargo Operator (Deliver Cargo to Receptacle)
# ─────────────────────────────────────────────────────────────────────────────


class DepositCargo(MacroAction):
    """Transport docked cargo to a target receptacle or drop zone."""

    def __init__(
        self,
        receptacle_pos: tuple[int, int],
        cargo_feature: int | None = None,
        barrier_feature: int | None = None,
        approach_radius: int = 1,
        description: str = "",
    ) -> None:
        super().__init__(name=f"DepositCargo({receptacle_pos})", intent=SpatialActionIntent.DROP)
        self.receptacle_pos = (int(receptacle_pos[0]), int(receptacle_pos[1]))
        self.cargo_feature = cargo_feature
        self.barrier_feature = barrier_feature
        self.approach_radius = approach_radius
        self.description = description or f"Deposit cargo at {self.receptacle_pos}"
        self.nav_op = NavigateTo(destination=self.receptacle_pos, radius=approach_radius)

    def is_satisfied(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
    ) -> bool:
        if self.status == MacroActionStatus.SATISFIED:
            return True
        # 1. If barrier at receptacle was cleared
        r, c = self.receptacle_pos
        if 0 <= r < curr_grid.shape[0] and 0 <= c < curr_grid.shape[1]:
            if self.barrier_feature is not None and curr_grid[r, c] != self.barrier_feature:
                return True
        # 2. Satisfied if cargo is at or adjacent to receptacle
        active_manip = engine.working_memory.body_schema.active_manipulandum
        if active_manip is not None:
            mr, mc = active_manip.pos
            if (
                abs(mr - self.receptacle_pos[0]) + abs(mc - self.receptacle_pos[1])
                <= self.approach_radius
            ):
                return True
        return self.nav_op.is_satisfied(engine, curr_grid)

    def is_feasible(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> bool:
        return self.nav_op.is_feasible(engine, curr_grid, available_actions)

    def step(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> tuple[Any, dict[str, Any] | None] | None:
        if not self.nav_op.is_satisfied(engine, curr_grid):
            return self.nav_op.step(engine, curr_grid, available_actions)

        # In stance at receptacle: trigger deposit action if interact available
        interact_candidates = [
            a
            for a in available_actions
            if not engine.is_displacement_action(a) and not engine.is_spatial_effector(a)
        ]
        if interact_candidates:
            chosen = 5 if 5 in interact_candidates else interact_candidates[0]
            self.status = MacroActionStatus.SATISFIED
            if self.cargo_feature is not None:
                engine.working_memory.body_schema.expend_tool(self.cargo_feature)
            elif engine.working_memory.body_schema.held_tools:
                engine.working_memory.body_schema.held_tools.pop()
            return chosen, None

        self.status = MacroActionStatus.SATISFIED
        return available_actions[0], None


class UseToolOnBarrier(DepositCargo):
    """Macro-action targeting using a tool/key on an obstructing barrier."""

    def __init__(
        self,
        barrier_pos: tuple[int, int],
        tool_feature: int | None = None,
        barrier_feature: int | None = None,
        approach_radius: int = 1,
        description: str = "",
    ) -> None:
        super().__init__(
            receptacle_pos=barrier_pos,
            cargo_feature=tool_feature,
            barrier_feature=barrier_feature,
            approach_radius=approach_radius,
            description=description or f"Use tool {tool_feature} on barrier at {barrier_pos}",
        )
        self.name = f"UseToolOnBarrier({barrier_pos})"
        self.intent = SpatialActionIntent.INTERACT


# ─────────────────────────────────────────────────────────────────────────────
# 5B. PushObject Operator (Adjacent Stance + Directional Displacement)
# ─────────────────────────────────────────────────────────────────────────────


class PushObject(MacroAction):
    """Adjacent stance navigation followed by directional displacement to push an object.

    Used when an object cannot be docked into body schema (e.g., heavy box/crate/sokoban block),
    requiring the agent to position itself on the opposite side and push into the object.
    """

    def __init__(
        self,
        target_pos: tuple[int, int],
        direction: tuple[int, int] = (0, 1),
        description: str = "",
    ) -> None:
        super().__init__(
            name=f"PushObject({target_pos}, dir={direction})", intent=SpatialActionIntent.INTERACT
        )
        self.target_pos = (int(target_pos[0]), int(target_pos[1]))
        self.direction = (int(direction[0]), int(direction[1]))
        self.stance_pos = (
            self.target_pos[0] - self.direction[0],
            self.target_pos[1] - self.direction[1],
        )
        self.description = (
            description or f"Push object at {self.target_pos} in direction {self.direction}"
        )
        self.nav_op = NavigateTo(destination=self.stance_pos, radius=0)
        self.has_pushed: bool = False

    def is_satisfied(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
    ) -> bool:
        return self.has_pushed

    def is_feasible(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> bool:
        return self.nav_op.is_feasible(engine, curr_grid, available_actions)

    def step(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> tuple[Any, dict[str, Any] | None] | None:
        self.consecutive_steps += 1
        if not self.nav_op.is_satisfied(engine, curr_grid):
            return self.nav_op.step(engine, curr_grid, available_actions)

        # In stance: push directly in direction
        av_pos = engine.avatar_pos
        if av_pos == self.stance_pos:
            for a, dyn in engine.action_dynamics.items():
                if a in available_actions and dyn.is_displacement_action():
                    if dyn.get_displacement() == self.direction:
                        self.has_pushed = True
                        self.status = MacroActionStatus.SATISFIED
                        return a, None

        self.status = MacroActionStatus.FAILED
        return available_actions[0], None


# ─────────────────────────────────────────────────────────────────────────────
# 5C. ExploreObject Operator (Affordance Probing on Unknown Entities)
# ─────────────────────────────────────────────────────────────────────────────


class ExploreObject(MacroAction):
    """Active epistemic probing of an unfamiliar or unprobed entity to test affordances."""

    def __init__(
        self,
        target_pos: tuple[int, int],
        target_feature: int | None = None,
        description: str = "",
    ) -> None:
        super().__init__(name=f"ExploreObject({target_pos})", intent=SpatialActionIntent.NAVIGATE)
        self.target_pos = (int(target_pos[0]), int(target_pos[1]))
        self.target_feature = target_feature
        self.description = description or f"Explore entity at {self.target_pos}"
        self.nav_op = NavigateTo(destination=self.target_pos, radius=1)
        self.probed: bool = False

    def is_satisfied(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
    ) -> bool:
        return self.probed or self.consecutive_steps >= 15

    def is_feasible(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> bool:
        return len(available_actions) > 0

    def step(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> tuple[Any, dict[str, Any] | None] | None:
        self.consecutive_steps += 1
        # Allocentric targeted effector only if agent lacks displacement capabilities
        if not engine.has_displacement_capabilities(available_actions) and any(
            engine.is_spatial_effector(a) for a in available_actions
        ):
            eff_candidates = [a for a in available_actions if engine.is_spatial_effector(a)]
            if eff_candidates:
                self.probed = True
                self.status = MacroActionStatus.SATISFIED
                return eff_candidates[0], _spatial_effector_payload(
                    engine, eff_candidates[0], self.target_pos
                )

        # Embodied: navigate adjacent to target
        if engine.avatar_pos is not None:
            if not self.nav_op.is_satisfied(engine, curr_grid):
                res = self.nav_op.step(engine, curr_grid, available_actions)
                if res is not None:
                    return res

        # Adjacent or non-embodied: try interact action or uninhibited probe actions
        self.probed = True
        self.status = MacroActionStatus.SATISFIED
        interact_candidates = [
            a
            for a in available_actions
            if not engine.is_displacement_action(a) and engine.inhibited_actions.get(a, 0) == 0
        ]
        if interact_candidates:
            chosen = interact_candidates[0]
            if engine.is_spatial_effector(chosen):
                return chosen, _spatial_effector_payload(engine, chosen, self.target_pos)
            return chosen, None

        if engine.avatar_pos is not None:
            dr = int(np.sign(self.target_pos[0] - engine.avatar_pos[0]))
            dc = int(np.sign(self.target_pos[1] - engine.avatar_pos[1]))
            for a, dyn in engine.action_dynamics.items():
                if a in available_actions and dyn.is_displacement_action():
                    if dyn.get_displacement() == (dr, dc):
                        return a, None

        for a in available_actions:
            if engine.inhibited_actions.get(a, 0) == 0:
                return a, None

        return available_actions[0], None


class TranslateObject(MacroAction):
    """Domain-agnostic macro-action: Translates selected/focused object to target coordinate (x, y)."""

    def __init__(
        self,
        target_pos: tuple[int, int],
        get_current_pos: Callable[[AutonomousEpistemicEngine, np.ndarray], tuple[int, int] | None]
        | None = None,
        action_map: dict[str, Any] | None = None,
        description: str = "",
    ) -> None:
        super().__init__(name=f"TranslateObject({target_pos})", intent=SpatialActionIntent.NAVIGATE)
        self.target_pos = (int(target_pos[0]), int(target_pos[1]))
        self.get_current_pos = get_current_pos
        self.action_map = action_map or {"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4}
        self.description = description or f"Translate object to {self.target_pos}"

    def _resolve_current_pos(
        self, engine: AutonomousEpistemicEngine, curr_grid: np.ndarray
    ) -> tuple[int, int] | None:
        if self.get_current_pos is not None:
            return self.get_current_pos(engine, curr_grid)
        if hasattr(engine, "current_actor_pos") and engine.current_actor_pos is not None:
            return (
                int(round(engine.current_actor_pos[1])),
                int(round(engine.current_actor_pos[0])),
            )
        if engine.avatar_pos is not None:
            return (int(engine.avatar_pos[1]), int(engine.avatar_pos[0]))
        return None

    def is_satisfied(self, engine: AutonomousEpistemicEngine, curr_grid: np.ndarray) -> bool:
        cur = self._resolve_current_pos(engine, curr_grid)
        if cur is None:
            return False
        return cur[0] == self.target_pos[0] and cur[1] == self.target_pos[1]

    def is_feasible(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> bool:
        return any(a in available_actions for a in self.action_map.values())

    def step(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> tuple[Any, dict[str, Any] | None]:
        self.consecutive_steps += 1
        cur = self._resolve_current_pos(engine, curr_grid)
        if cur is None:
            self.status = MacroActionStatus.BLOCKED
            return available_actions[0], None

        cur_x, cur_y = cur
        tgt_x, tgt_y = self.target_pos

        act = None
        if cur_x > tgt_x and self.action_map.get("LEFT") in available_actions:
            act = self.action_map["LEFT"]
        elif cur_x < tgt_x and self.action_map.get("RIGHT") in available_actions:
            act = self.action_map["RIGHT"]
        elif cur_y > tgt_y and self.action_map.get("UP") in available_actions:
            act = self.action_map["UP"]
        elif cur_y < tgt_y and self.action_map.get("DOWN") in available_actions:
            act = self.action_map["DOWN"]

        if act is None:
            self.status = MacroActionStatus.SATISFIED
            return available_actions[0], None
        return act, None
