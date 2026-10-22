"""Macro-Action Operators and Hierarchical Option Scheduling.

Decomposes monolithic procedural cognitive loops into discrete, composable macro-actions:
1. NavigateTo(from -> to): Geodesic A* navigation across traversable manifold.
2. InteractWith(target): Allocentric effector actuation (clicks, toggles, widget switches).
3. PickUpCargo(target): Locomotion adjacent to cargo followed by body-schema manipulandum docking.
4. DepositCargo(target): Transport and release of carried cargo to destination receptacle.
5. SwitchFocus(): Clean multi-agent / multi-slider control handover.
6. ExploreFrontier(): Epistemic curiosity search toward nearest unvisited spatial frontier.

Conforms strictly to HBLLM Cognitive OS invariants (LLM-free, deterministic, state owned by HCIR).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole, SpatialActionIntent
from hbllm.hcir.world.frontopolar_subgoal_stack import SubgoalType

if TYPE_CHECKING:
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

logger = logging.getLogger(__name__)


class MacroActionStatus(StrEnum):
    """Execution status of an active macro action."""

    PENDING = "pending"
    RUNNING = "running"
    SATISFIED = "satisfied"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class ObjectSpec:
    """Formal specification of an entity and its properties passed to the cognitive core."""

    id: str
    pos: tuple[int, int]
    feature_id: int = 0
    role: str = "unknown"  # "goal", "cargo", "key", "receptacle", "switch", "barrier", "hazard", "manipulandum"
    properties: dict[str, Any] = field(default_factory=dict)
    macro_tasks: list[MacroAction] = field(default_factory=list)


@dataclass
class MicroActionSpec:
    """Formal specification of an environment primitive micro-action and its physical dynamics."""

    action_id: Any
    name: str = ""
    delta_r: int = 0
    delta_c: int = 0
    is_spatial_effector: bool = False
    is_focus_switch: bool = False
    parameters: dict[str, Any] = field(default_factory=dict)


class MacroAction(ABC):
    """Abstract base operator for discrete neuro-symbolic macro-actions."""

    def __init__(self, name: str, intent: SpatialActionIntent | SubgoalType) -> None:
        self.name = name
        self.intent = intent
        self.status: MacroActionStatus = MacroActionStatus.PENDING
        self.consecutive_steps: int = 0
        self.max_steps: int = 150

    @abstractmethod
    def is_satisfied(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
    ) -> bool:
        """Evaluate if postconditions have been achieved."""
        ...

    @abstractmethod
    def is_feasible(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> bool:
        """Evaluate if preconditions and affordances are met."""
        ...

    @abstractmethod
    def step(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> tuple[Any, dict[str, Any] | None] | None:
        """Produce the next concrete environment action."""
        ...

    def reset(self) -> None:
        """Reset execution counters."""
        self.status = MacroActionStatus.PENDING
        self.consecutive_steps = 0


# ─────────────────────────────────────────────────────────────────────────────
# 1. NavigateTo Operator (Move A -> B)
# ─────────────────────────────────────────────────────────────────────────────


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

        failed_tx = getattr(engine, "failed_transitions", set())
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

        threats = getattr(engine, "_prev_oriented_threats", [])
        if not threats:
            from hbllm.hcir.world.cortex_perception import PerceptionEngine

            threats = PerceptionEngine.detect_oriented_threats(
                curr_grid,
                getattr(engine, "previous_entities", []),
                engine.bg_feature or 0,
                avatar_pos=av_pos,
            )

        # Plan next geodesic step
        plan = GeodesicManifoldRouter.route_geodesic(
            curr_grid=curr_grid,
            start_pos=av_pos,
            goals=[self.destination],
            static_barriers=engine.learned_barriers,
            grid_walls=set(),
            unknown_entity_cells=set(),
            oriented_threats=threats,
            movable_actions=movable_actions,
            bg=engine.bg_feature or 0,
            barrier_features=engine.symbolic_theory.barrier_features,
            avatar_features=av_feats,
            max_candidate_goals=1,
        )

        if plan and len(plan) > 0:
            next_act, next_pos = plan[0]
            if next_act in available_actions and (av_pos, next_act) not in failed_tx:
                self.status = MacroActionStatus.RUNNING
                from collections import deque

                from hbllm.hcir.world.cortex_planner import MentalSimulationStep

                engine.mental_plan = deque(
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
                return next_act, None

        # Fallback: exploratory probe toward destination using unfailed displacement actions
        valid_disp_acts = [
            a
            for a in available_actions
            if engine.is_displacement_action(a) and (av_pos, a) not in failed_tx
        ]
        if valid_disp_acts:
            best_a = min(
                valid_disp_acts,
                key=lambda a: (
                    (av_pos[0] + engine.action_dynamics[a].delta_r - self.destination[0]) ** 2
                    + (av_pos[1] + engine.action_dynamics[a].delta_c - self.destination[1]) ** 2
                    if a in engine.action_dynamics
                    else 9999
                ),
            )
            return best_a, None

        self.status = MacroActionStatus.BLOCKED
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. InteractWith Operator (Actuate / Click Target)
# ─────────────────────────────────────────────────────────────────────────────


class InteractWith(MacroAction):
    """Allocentric spatial effector command (e.g. click widget, toggle optical mirror)."""

    def __init__(
        self,
        target_coord: tuple[int, int],
        effector_action: Any | None = None,
        description: str = "",
    ) -> None:
        super().__init__(name=f"InteractWith({target_coord})", intent=SpatialActionIntent.INTERACT)
        self.target_coord = (int(target_coord[0]), int(target_coord[1]))
        self.effector_action = effector_action
        self.description = description or f"Interact with coordinate {self.target_coord}"
        self.has_executed: bool = False

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
        if self.effector_action is not None:
            return self.effector_action in available_actions
        return any(engine.is_spatial_effector(a) for a in available_actions)

    def step(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> tuple[Any, dict[str, Any] | None]:
        eff = self.effector_action
        if eff is None or eff not in available_actions:
            eff_candidates = [a for a in available_actions if engine.is_spatial_effector(a)]
            if eff_candidates:
                eff = eff_candidates[0]
            else:
                self.status = MacroActionStatus.FAILED
                return available_actions[0], None

        data = {"x": int(self.target_coord[1]), "y": int(self.target_coord[0])}
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

        # Avatar is adjacent to cargo! Step directly toward cargo to dock/bind
        av_pos = engine.avatar_pos
        if av_pos is not None:
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
        return self.nav_op.step(engine, curr_grid, available_actions)


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
        # Allocentric click effector
        if any(engine.is_spatial_effector(a) or a == 6 for a in available_actions):
            eff_candidates = [
                a for a in available_actions if engine.is_spatial_effector(a) or a == 6
            ]
            if eff_candidates:
                self.probed = True
                self.status = MacroActionStatus.SATISFIED
                return eff_candidates[0], {
                    "x": int(self.target_pos[1]),
                    "y": int(self.target_pos[0]),
                }

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
            return interact_candidates[0], None

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


# ─────────────────────────────────────────────────────────────────────────────
# 6. ExploreFrontier Operator (Epistemic Curiosity)
# ─────────────────────────────────────────────────────────────────────────────


class ExploreFrontier(MacroAction):
    """Epistemic exploration toward the nearest unvisited traversable coordinate."""

    def __init__(self, description: str = "") -> None:
        super().__init__(name="ExploreFrontier", intent=SpatialActionIntent.NAVIGATE)
        self.description = description or "Explore unvisited frontier cells"

    def is_satisfied(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
    ) -> bool:
        return self.consecutive_steps >= 12 or engine.phase != engine.phase.EPISTEMIC_EXPLORATION

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
    ) -> tuple[Any, dict[str, Any] | None]:
        self.consecutive_steps += 1
        return engine.plan_epistemic_probe(curr_grid, available_actions)


# ─────────────────────────────────────────────────────────────────────────────
# 7. MacroActionDriverManager (Domain Adapter & Plugin Registry)
# ─────────────────────────────────────────────────────────────────────────────


class MacroActionDriverManager:
    """Central registry and dispatch manager for pluggable domain-specific macro-actions and drivers.

    Decouples core neuro-symbolic reasoning from any specific domain, game, benchmark,
    or peripheral (e.g. ARC-AGI, AI2-THOR, Crafter, MineDojo, ROS2).
    Domain drivers or runner scripts register their custom macro-action operators
    and synthesis hooks here.
    """

    _registry: dict[str, type[MacroAction]] = {}
    _synthesis_hooks: list[
        Callable[
            [AutonomousEpistemicEngine, np.ndarray, Sequence[Any], Sequence[Any] | None],
            MacroActionPlan | None,
        ]
    ] = []

    @classmethod
    def register_macro_action(cls, name: str, op_class: type[MacroAction]) -> None:
        """Register a domain-specific macro-action operator class."""
        cls._registry[name] = op_class
        logger.info("MacroActionDriverManager: Registered operator '%s'", name)

    @classmethod
    def register_synthesis_hook(
        cls,
        hook: Callable[
            [AutonomousEpistemicEngine, np.ndarray, Sequence[Any], Sequence[Any] | None],
            MacroActionPlan | None,
        ],
    ) -> None:
        if hook not in cls._synthesis_hooks:
            cls._synthesis_hooks.append(hook)

    @classmethod
    def get_registered_actions(cls) -> dict[str, type[MacroAction]]:
        return dict(cls._registry)

    @classmethod
    def get_registered_action(cls, name: str) -> type[MacroAction] | None:
        return cls._registry.get(name)

    @classmethod
    def run_synthesis_hooks(
        cls,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
        objects: Sequence[Any] | None = None,
    ) -> MacroActionPlan | None:
        """Query registered domain drivers for specialized plan synthesis."""
        for hook in cls._synthesis_hooks:
            try:
                plan = hook(engine, curr_grid, available_actions, objects)
                if plan is not None and not plan.is_empty():
                    return plan
            except Exception as e:
                logger.debug("MacroActionDriverManager: Hook error: %s", e)
        return None

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()
        cls._synthesis_hooks.clear()


# ─────────────────────────────────────────────────────────────────────────────
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


class CustomMacroAction(MacroAction):
    """Declarative custom macro action parameterized by arbitrary user/runner predicates."""

    def __init__(
        self,
        name: str,
        satisfaction_fn: Any,
        feasibility_fn: Any,
        step_fn: Any,
        intent: SpatialActionIntent | SubgoalType = SpatialActionIntent.NAVIGATE,
        description: str = "",
    ) -> None:
        super().__init__(name=name, intent=intent)
        self._satisfaction_fn = satisfaction_fn
        self._feasibility_fn = feasibility_fn
        self._step_fn = step_fn
        self.description = description or name

    def is_satisfied(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
    ) -> bool:
        return bool(self._satisfaction_fn(engine, curr_grid))

    def is_feasible(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> bool:
        return bool(self._feasibility_fn(engine, curr_grid, available_actions))

    def step(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> tuple[Any, dict[str, Any] | None]:
        self.consecutive_steps += 1
        return self._step_fn(engine, curr_grid, available_actions)


# ─────────────────────────────────────────────────────────────────────────────
# 9. MacroActionPlan Scheduler & Causal Attribution
# ─────────────────────────────────────────────────────────────────────────────


def attribute_macro_action_outcome(
    engine: AutonomousEpistemicEngine,
    macro_action: MacroAction,
    prev_grid: np.ndarray | None,
    curr_grid: np.ndarray,
    is_success: bool,
    action_used: Any | None = None,
    action_data: dict[str, Any] | None = None,
) -> None:
    """Attribute macro-action completion or failure into Pearl's do(a) causal theory and body schema."""
    if prev_grid is None and hasattr(engine, "grid_history") and len(engine.grid_history) >= 2:
        prev_grid = engine.grid_history[-2]

    if is_success:
        macro_action.status = MacroActionStatus.SATISFIED
        if isinstance(macro_action, PickUpCargo):
            feat = macro_action.cargo_feature
            if feat is None and prev_grid is not None:
                r, c = macro_action.cargo_pos
                if 0 <= r < prev_grid.shape[0] and 0 <= c < prev_grid.shape[1]:
                    feat = int(prev_grid[r, c])
            if feat is not None and feat != 0:
                step_cnt = getattr(engine, "step_count", 0)
                engine.working_memory.body_schema.acquire_tool(feat, step=step_cnt, role="key")
                engine.symbolic_theory.cargo_features.add(feat)
                if hasattr(engine, "learned_cargo_features"):
                    engine.learned_cargo_features.add(feat)
                logger.info(
                    "ExtendedBodySchema: Acquired tool/cargo feature %d into body schema at %s",
                    feat,
                    macro_action.cargo_pos,
                )
            if engine.subgoal_stack.has_pending_goals:
                cur_sg = engine.subgoal_stack.current_subgoal
                if cur_sg and cur_sg.subgoal_type in (
                    SubgoalType.ACQUIRE_TOOL,
                    SubgoalType.COLLECT_ITEM,
                ):
                    if cur_sg.target_entity_pos == macro_action.cargo_pos or (
                        feat is not None and cur_sg.required_feature == feat
                    ):
                        engine.subgoal_stack.pop_subgoal()
        elif isinstance(macro_action, DepositCargo):
            tool_feat = macro_action.cargo_feature
            if tool_feat is None:
                held = engine.working_memory.body_schema.held_tools
                if held:
                    tool_feat = held[-1].feature_id
            r, c = macro_action.receptacle_pos
            barrier_feat = macro_action.barrier_feature
            if barrier_feat is None and prev_grid is not None:
                if 0 <= r < prev_grid.shape[0] and 0 <= c < prev_grid.shape[1]:
                    bg = engine.estimate_background(prev_grid)
                    val = int(prev_grid[r, c])
                    if val != bg:
                        barrier_feat = val

            cell_cleared = False
            if (
                prev_grid is not None
                and 0 <= r < curr_grid.shape[0]
                and 0 <= c < curr_grid.shape[1]
            ):
                bg = engine.estimate_background(curr_grid)
                cell_cleared = (prev_grid[r, c] != curr_grid[r, c]) or (curr_grid[r, c] == bg)

            if (
                tool_feat is not None
                and barrier_feat is not None
                and (cell_cleared or barrier_feat in engine.symbolic_theory.barrier_features)
            ):
                engine.symbolic_theory.record_tool_barrier_affinity(tool_feat, barrier_feat)
                engine.working_memory.body_schema.register_resonance(tool_feat, barrier_feat)
                engine.working_memory.body_schema.expend_tool(tool_feat)
                engine.learned_barriers.discard((r, c))
                if hasattr(engine, "causal_cortex"):
                    engine.causal_cortex.attribute_macro_outcome(
                        SpatialActionIntent.DROP,
                        macro_action.receptacle_pos,
                        tool_feat,
                        barrier_feat,
                        is_success=True,
                    )
                logger.info(
                    "Pearl do(a) causal attribution: tool %d cleared barrier %d at %s",
                    tool_feat,
                    barrier_feat,
                    macro_action.receptacle_pos,
                )
            elif cell_cleared:
                engine.learned_barriers.discard((r, c))

            if engine.subgoal_stack.has_pending_goals:
                cur_sg = engine.subgoal_stack.current_subgoal
                if cur_sg and cur_sg.target_destination == macro_action.receptacle_pos:
                    engine.subgoal_stack.pop_subgoal()
        elif isinstance(macro_action, InteractWith):
            if prev_grid is not None:
                bg = engine.estimate_background(curr_grid)
                av_feats = engine.avatar_features or (
                    {engine.avatar_feature} if engine.avatar_feature is not None else None
                )
                attributor = getattr(
                    engine,
                    "remote_causal",
                    getattr(getattr(engine, "causal_cortex", None), "causal_attributor", None),
                )
                if attributor is not None:
                    attributor.record_transition(
                        prev_grid=prev_grid,
                        curr_grid=curr_grid,
                        action_pos=macro_action.target_coord,
                        background_feature=bg,
                        avatar_features=av_feats,
                    )
                # Active Causal Induction for Distal Actuations
                diff_coords = np.argwhere(prev_grid != curr_grid)
                trig_coord = macro_action.target_coord
                trig_feat = (
                    int(prev_grid[trig_coord])
                    if 0 <= trig_coord[0] < prev_grid.shape[0]
                    and 0 <= trig_coord[1] < prev_grid.shape[1]
                    else 0
                )
                for d_coord in diff_coords:
                    dr, dc = int(d_coord[0]), int(d_coord[1])
                    if (dr, dc) == trig_coord or (av_feats and int(curr_grid[dr, dc]) in av_feats):
                        continue
                    orig_feat = int(prev_grid[dr, dc])
                    new_feat = int(curr_grid[dr, dc])
                    if (
                        (dr, dc) in engine.learned_barriers
                        or orig_feat in engine.symbolic_theory.barrier_features
                        or new_feat == bg
                    ):
                        engine.learned_barriers.discard((dr, dc))
                        engine.symbolic_theory.record_distal_actuation(
                            trig_feat, orig_feat, affected_pos=(dr, dc)
                        )
                        logger.info(
                            "Distal actuation unblocked barrier at (%d, %d) via trigger %d",
                            dr,
                            dc,
                            trig_feat,
                        )
                        if engine.subgoal_stack.has_pending_goals:
                            top_sg = engine.subgoal_stack.current_subgoal
                            if top_sg and top_sg.target_destination in ((dr, dc), trig_coord):
                                engine.subgoal_stack.pop_subgoal()
        elif isinstance(macro_action, NavigateTo):
            if macro_action.destination in engine.learned_goal_positions:
                r, c = macro_action.destination
                if 0 <= r < curr_grid.shape[0] and 0 <= c < curr_grid.shape[1]:
                    dest_feat = int(curr_grid[r, c])
                    bg = engine.estimate_background(curr_grid)
                    if dest_feat != bg:
                        engine.symbolic_theory.goal_features.add(dest_feat)
                        engine.learned_goal_features.add(dest_feat)
            if engine.subgoal_stack.has_pending_goals:
                cur_sg = engine.subgoal_stack.current_subgoal
                if cur_sg and cur_sg.target_destination == macro_action.destination:
                    engine.subgoal_stack.pop_subgoal()
    else:
        macro_action.status = MacroActionStatus.FAILED
        if isinstance(macro_action, DepositCargo):
            tool_feat = macro_action.cargo_feature
            if tool_feat is None:
                held = engine.working_memory.body_schema.held_tools
                if held:
                    tool_feat = held[-1].feature_id
            barrier_feat = macro_action.barrier_feature
            if barrier_feat is None and prev_grid is not None:
                r, c = macro_action.receptacle_pos
                if 0 <= r < prev_grid.shape[0] and 0 <= c < prev_grid.shape[1]:
                    bg = engine.estimate_background(prev_grid)
                    val = int(prev_grid[r, c])
                    if val != bg:
                        barrier_feat = val
            if tool_feat is not None and barrier_feat is not None:
                engine.symbolic_theory.record_tool_barrier_incompatibility(tool_feat, barrier_feat)
                logger.info(
                    "Causal Incompatibility: Tool %d failed to clear barrier %d at %s",
                    tool_feat,
                    barrier_feat,
                    macro_action.receptacle_pos,
                )
        elif isinstance(macro_action, NavigateTo):
            engine.learned_barriers.add(macro_action.destination)
            if prev_grid is not None:
                r, c = macro_action.destination
                if 0 <= r < prev_grid.shape[0] and 0 <= c < prev_grid.shape[1]:
                    feat = int(prev_grid[r, c])
                    bg = engine.estimate_background(prev_grid)
                    if feat != bg:
                        engine.symbolic_theory.barrier_features.add(feat)
            if engine.avatar_pos is not None and action_used is not None:
                engine.failed_transitions.add((engine.avatar_pos, action_used))


@dataclass
class MacroActionPlan:
    """Sequential queue of macro-action operators managed by PrefrontalWorkingMemory."""

    actions: deque[MacroAction] = field(default_factory=deque)
    plan_id: str = "macro_plan_default"
    is_strict: bool = False

    def is_empty(self) -> bool:
        return len(self.actions) == 0

    def current(self) -> MacroAction | None:
        return self.actions[0] if self.actions else None

    def current_action(self) -> MacroAction | None:
        return self.current()

    def append(self, op: MacroAction) -> None:
        self.actions.append(op)

    def extend(self, ops: Sequence[MacroAction]) -> None:
        self.actions.extend(ops)

    def clear(self) -> None:
        self.actions.clear()

    def advance_if_satisfied(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        prev_grid: np.ndarray | None = None,
    ) -> bool:
        """Advance the plan if the active operator's postconditions are met."""
        while self.actions:
            cur = self.actions[0]
            if cur.is_satisfied(engine, curr_grid):
                popped = self.actions.popleft()
                popped.status = MacroActionStatus.SATISFIED
                logger.info(
                    "MacroActionPlan: Completed and popped %s (%d remaining)",
                    popped.name,
                    len(self.actions),
                )
                attribute_macro_action_outcome(
                    engine=engine,
                    macro_action=popped,
                    prev_grid=prev_grid,
                    curr_grid=curr_grid,
                    is_success=True,
                )
            else:
                break

        # Check and advance BA10 frontopolar subgoal stack
        if hasattr(engine, "subgoal_stack") and engine.subgoal_stack.has_pending_goals:
            top_sg = engine.subgoal_stack.current_subgoal
            if top_sg is not None:
                held_feats = {t.feature_id for t in engine.working_memory.body_schema.held_tools}
                bg = engine.estimate_background(curr_grid)
                if engine.subgoal_stack.is_subgoal_satisfied(
                    top_sg,
                    current_entity_positions=set(),
                    avatar_pos=engine.avatar_pos,
                    held_features=held_feats,
                    grid=curr_grid,
                    background_feature=bg,
                ):
                    engine.subgoal_stack.pop_subgoal()

        return len(self.actions) > 0

    def step(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
        prev_grid: np.ndarray | None = None,
    ) -> tuple[Any, dict[str, Any] | None] | None:
        """Execute the current macro-action."""
        self.advance_if_satisfied(engine, curr_grid, prev_grid=prev_grid)
        while self.actions:
            cur = self.actions[0]
            if cur.consecutive_steps >= cur.max_steps:
                logger.warning(
                    "MacroAction %s exceeded max steps (%d); aborting operator",
                    cur.name,
                    cur.max_steps,
                )
                popped = self.actions.popleft()
                popped.status = MacroActionStatus.FAILED
                attribute_macro_action_outcome(
                    engine=engine,
                    macro_action=popped,
                    prev_grid=prev_grid,
                    curr_grid=curr_grid,
                    is_success=False,
                )
                continue
            if not cur.is_feasible(engine, curr_grid, available_actions):
                logger.warning(
                    "MacroAction %s is infeasible in current state; dropping operator", cur.name
                )
                popped = self.actions.popleft()
                popped.status = MacroActionStatus.FAILED
                attribute_macro_action_outcome(
                    engine=engine,
                    macro_action=popped,
                    prev_grid=prev_grid,
                    curr_grid=curr_grid,
                    is_success=False,
                )
                continue
            res = cur.step(engine, curr_grid, available_actions)
            if res is None or cur.status in (MacroActionStatus.BLOCKED, MacroActionStatus.FAILED):
                popped = self.actions.popleft()
                popped.status = MacroActionStatus.FAILED
                attribute_macro_action_outcome(
                    engine=engine,
                    macro_action=popped,
                    prev_grid=prev_grid,
                    curr_grid=curr_grid,
                    is_success=False,
                )
                continue
            return res
        return None

    @classmethod
    def from_skill(
        cls,
        skill: Any,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> MacroActionPlan | None:
        """Compile a MacroActionPlan directly from a declarative skill."""
        if hasattr(skill, "compile_to_macro_plan"):
            return skill.compile_to_macro_plan(engine, curr_grid, available_actions)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 10. MacroPlanSynthesizer (Core Planner Bridge)
# ─────────────────────────────────────────────────────────────────────────────


class MacroPlanSynthesizer:
    """Synthesizes high-level MacroActionPlans from Knowledge Base state and external specs."""

    @classmethod
    def ingest_knowledge(
        cls,
        engine: AutonomousEpistemicEngine,
        objects: Sequence[Any] | None = None,
        micro_actions: Sequence[Any] | None = None,
    ) -> None:
        """Ingest external objects and micro-action specifications into Core Knowledge Base."""
        if micro_actions:
            from hbllm.hcir.world.cortex_causal import ActionAffordance
            from hbllm.hcir.world.motor_calibration import ActionDynamicsModel

            for act_spec in micro_actions:
                if isinstance(act_spec, MicroActionSpec):
                    aid = act_spec.action_id
                    aff = engine.action_affordances.get(aid)
                    if aff is None:
                        aff = ActionAffordance(action_id=aid, name=act_spec.name)
                        engine.action_affordances[aid] = aff
                    if act_spec.delta_r != 0 or act_spec.delta_c != 0:
                        cur_dyn = engine.action_dynamics.get(aid)
                        if cur_dyn is None or getattr(cur_dyn, "probes_tested", 0) == 0:
                            engine.action_dynamics[aid] = ActionDynamicsModel(
                                action_id=aid,
                                delta_r=act_spec.delta_r,
                                delta_c=act_spec.delta_c,
                                confidence=1.0,
                            )
                            engine.action_deltas[aid] = (act_spec.delta_r, act_spec.delta_c)
                        aff.is_displacement = True
                        aff.delta = (act_spec.delta_r, act_spec.delta_c)
                    if act_spec.is_spatial_effector:
                        aff.requires_spatial_target = True
                    if act_spec.is_focus_switch:
                        aff.is_focus_switch = True
                        engine.focus_switch_actions.add(aid)
                        engine.working_memory.focus_switch_actions.add(aid)
                elif isinstance(act_spec, dict):
                    aid = act_spec.get("action_id", act_spec.get("id"))
                    if aid is not None:
                        aff = engine.action_affordances.get(aid)
                        if aff is None:
                            aff = ActionAffordance(
                                action_id=aid, name=str(act_spec.get("name", ""))
                            )
                            engine.action_affordances[aid] = aff
                        dr = act_spec.get("delta_r", 0)
                        dc = act_spec.get("delta_c", 0)
                        if dr != 0 or dc != 0:
                            cur_dyn = engine.action_dynamics.get(aid)
                            if cur_dyn is None or getattr(cur_dyn, "probes_tested", 0) == 0:
                                engine.action_dynamics[aid] = ActionDynamicsModel(
                                    action_id=aid, delta_r=dr, delta_c=dc, confidence=1.0
                                )
                                engine.action_deltas[aid] = (dr, dc)
                            aff.is_displacement = True
                            aff.delta = (dr, dc)
                        if act_spec.get("is_spatial_effector", False) or "CLICK" in str(
                            act_spec.get("name", "")
                        ):
                            aff.requires_spatial_target = True
                        if act_spec.get("is_focus_switch", False):
                            aff.is_focus_switch = True
                            engine.focus_switch_actions.add(aid)
                            engine.working_memory.focus_switch_actions.add(aid)

        if objects:
            for obj in objects:
                role = "unknown"
                pos = None
                feat = 0
                props = {}
                obj_id = ""
                m_tasks = []
                if isinstance(obj, ObjectSpec):
                    obj_id = obj.id
                    role = obj.role.lower()
                    pos = obj.pos
                    feat = obj.feature_id
                    props = obj.properties
                    m_tasks = getattr(obj, "macro_tasks", [])
                elif hasattr(obj, "role") and hasattr(obj, "grid_pos"):
                    role = str(obj.role.value if hasattr(obj.role, "value") else obj.role).lower()
                    pos = obj.grid_pos
                    feat = getattr(obj, "feature_id", 0) or 0
                    props = getattr(obj, "properties", {})
                    obj_id = getattr(obj, "id", f"{role}_{pos[0]}_{pos[1]}")
                elif isinstance(obj, dict):
                    role = str(obj.get("role", "unknown")).lower()
                    pos = obj.get("pos", obj.get("position", obj.get("grid_pos")))
                    feat = obj.get("feature_id", obj.get("feature", 0)) or 0
                    props = obj.get("properties", {})
                    obj_id = obj.get("id", f"{role}_{pos[0]}_{pos[1]}")

                if pos is not None:
                    pos = (int(pos[0]), int(pos[1]))
                    if not obj_id:
                        obj_id = f"{role}_{pos[0]}_{pos[1]}"

                    # Store in working memory knowledge base (Rule 1.8)
                    if hasattr(engine, "working_memory"):
                        spec = (
                            obj
                            if isinstance(obj, ObjectSpec)
                            else ObjectSpec(
                                id=obj_id,
                                pos=pos,
                                feature_id=feat,
                                role=role,
                                properties=props,
                                macro_tasks=m_tasks,
                            )
                        )
                        engine.working_memory.known_objects[obj_id] = spec
                        if m_tasks:
                            engine.working_memory.object_macro_tasks[obj_id] = list(m_tasks)
                        engine.working_memory.object_trial_counts.setdefault(obj_id, 0)

                    if role == "goal" or "exit" in role:
                        if props.get("source") == "learned_goal" or props.get("confirmed", False):
                            engine.learned_goal_positions.add(pos)
                            if hasattr(engine, "goals") and pos not in engine.goals:
                                engine.goals.append(pos)
                        if feat:
                            engine.symbolic_theory.goal_features.add(feat)
                            engine.learned_goal_features.add(feat)
                    elif "cargo" in role or "key" in role or "box" in role or "manipulable" in role:
                        if feat:
                            engine.symbolic_theory.cargo_features.add(feat)
                            engine.learned_cargo_features.add(feat)
                    elif (
                        "receptacle" in role
                        or "door" in role
                        or "container" in role
                        or "silhouette" in role
                    ):
                        engine.learned_receptacle_positions.add(pos)
                        if feat:
                            engine.symbolic_theory.receptacle_features.add(feat)
                    elif (
                        "barrier" in role
                        or "wall" in role
                        or "obstacle" in role
                        or props.get("is_solid")
                    ):
                        engine.learned_barriers.add(pos)
                        if feat:
                            engine.symbolic_theory.barrier_features.add(feat)
                    elif "switch" in role or "actuator" in role or "button" in role:
                        if feat:
                            engine.symbolic_theory.actuator_features.add(feat)

    @classmethod
    def synthesize_plan(
        cls,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
        objects: Sequence[Any] | None = None,
        micro_actions: Sequence[Any] | None = None,
    ) -> MacroActionPlan | None:
        """Synthesize high-level macro plans from object affordances and causal theory."""
        if objects or micro_actions:
            cls.ingest_knowledge(engine, objects=objects, micro_actions=micro_actions)

        # 0. Query registered domain drivers & plugins
        driver_plan = MacroActionDriverManager.run_synthesis_hooks(
            engine, curr_grid, available_actions, objects=objects
        )
        if driver_plan is not None and not driver_plan.is_empty():
            return driver_plan

        # 0B. Query declarative skills registered on engine
        skills = getattr(engine, "declarative_skills", [])
        for skill in skills:
            if hasattr(skill, "can_handle") and skill.can_handle(
                curr_grid, list(available_actions), metadata={"engine": engine}
            ):
                skill_plan = MacroActionPlan.from_skill(skill, engine, curr_grid, available_actions)
                if skill_plan is not None and not skill_plan.is_empty():
                    return skill_plan

        plan = MacroActionPlan()

        # 0C. Check active executive subgoal stack (Brodmann Area 10)
        if engine.subgoal_stack.has_pending_goals:
            top_sg = engine.subgoal_stack.current_subgoal
            if top_sg is not None:
                if top_sg.subgoal_type in (SubgoalType.ACQUIRE_TOOL, SubgoalType.COLLECT_ITEM):
                    plan.append(
                        PickUpCargo(
                            cargo_pos=top_sg.target_entity_pos,
                            cargo_feature=top_sg.required_feature,
                            description=f"Acquire tool at {top_sg.target_entity_pos}",
                        )
                    )
                    return plan
                elif top_sg.subgoal_type in (
                    SubgoalType.CLEAR_OBSTACLE,
                    SubgoalType.UNLOCK_REMOTE_MECHANISM,
                ):
                    held_feats = {
                        t.feature_id for t in engine.working_memory.body_schema.held_tools
                    }
                    if top_sg.required_feature in held_feats or not top_sg.prerequisite_features:
                        plan.append(
                            UseToolOnBarrier(
                                barrier_pos=top_sg.target_destination,
                                tool_feature=top_sg.required_feature,
                                description=f"Clear obstacle at {top_sg.target_destination}",
                            )
                        )
                    else:
                        plan.append(
                            InteractWith(
                                target_coord=top_sg.target_destination,
                                description=f"Actuate mechanism at {top_sg.target_destination}",
                            )
                        )
                    return plan
                elif top_sg.subgoal_type in (SubgoalType.DELIVER_TO_GOAL, SubgoalType.NAVIGATE_TO):
                    plan.append(
                        NavigateTo(
                            destination=top_sg.target_destination, description="Navigate to goal"
                        )
                    )
                    return plan
                elif top_sg.subgoal_type == SubgoalType.SWITCH_FOCUS:
                    sw_act = top_sg.metadata.get("switch_action")
                    if sw_act is None and engine.focus_switch_actions:
                        sw_act = next(iter(engine.focus_switch_actions))
                    if sw_act is not None:
                        plan.append(SwitchFocus(switch_action=sw_act, description="Handover focus"))
                        return plan

        bg = engine.estimate_background(curr_grid)
        entities = engine.extract_entities(curr_grid, bg)
        known_objs = list(getattr(engine.working_memory, "known_objects", {}).values())
        trial_counts = getattr(engine.working_memory, "object_trial_counts", {})

        # Collect candidate goals
        goals: list[tuple[int, int]] = []
        for obj in known_objs:
            if "goal" in obj.role or "exit" in obj.role:
                if obj.pos not in goals:
                    goals.append(obj.pos)
        for e in entities:
            if (
                e.role == EntityRole.GOAL
                or e.feature_id in engine.learned_goal_features
                or e.grid_pos in engine.learned_goal_positions
            ):
                if e.grid_pos not in goals:
                    goals.append(e.grid_pos)
        for g_pos in engine.learned_goal_positions:
            if g_pos not in goals:
                goals.append(g_pos)

        # Multi-Agent focus switch verification
        has_multi_agent = (
            len(engine.working_memory.focus_switch_actions) > 0
            or len(engine.focus_switch_actions) > 0
            or len(getattr(engine.working_memory, "controllable_avatar_features", set())) > 1
        )
        focus_switch = next(
            (
                a
                for a in available_actions
                if (engine.is_focus_switch_action(a) or a in engine.focus_switch_actions)
                and engine.inhibited_actions.get(a, 0) == 0
            ),
            None,
        )

        # Collect candidate cargo / boxes
        cargo_objs: list[ObjectSpec] = [
            obj
            for obj in known_objs
            if "cargo" in obj.role
            or "box" in obj.role
            or "key" in obj.role
            or "crate" in obj.role
            or "item" in obj.role
            or obj.feature_id in engine.learned_cargo_features
        ]
        for e in entities:
            if (
                e.role in (EntityRole.MANIPULABLE, EntityRole.RESOURCE)
                or e.feature_id in engine.learned_cargo_features
            ):
                if not any(c.pos == e.grid_pos for c in cargo_objs):
                    cargo_objs.append(
                        ObjectSpec(
                            id=f"cargo_{e.grid_pos[0]}_{e.grid_pos[1]}",
                            pos=e.grid_pos,
                            feature_id=e.feature_id,
                            role="cargo",
                        )
                    )

        # Collect candidate receptacles / sockets / gates
        receptacles: list[tuple[int, int]] = []
        for obj in known_objs:
            if (
                "receptacle" in obj.role
                or "door" in obj.role
                or "silhouette" in obj.role
                or "container" in obj.role
            ):
                if obj.pos not in receptacles:
                    receptacles.append(obj.pos)
        for e in entities:
            if (
                e.role in (EntityRole.RECEPTACLE, EntityRole.PORTAL)
                or e.grid_pos in engine.learned_receptacle_positions
            ):
                if e.grid_pos not in receptacles:
                    receptacles.append(e.grid_pos)
        for r_pos in engine.learned_receptacle_positions:
            if r_pos not in receptacles:
                receptacles.append(r_pos)

        # Collect candidate switches / buttons / actuators
        switch_objs: list[ObjectSpec] = [
            obj
            for obj in known_objs
            if "switch" in obj.role
            or "button" in obj.role
            or "actuator" in obj.role
            or "lever" in obj.role
            or "valve" in obj.role
            or obj.feature_id in engine.symbolic_theory.actuator_features
        ]
        for e in entities:
            if (
                e.role == EntityRole.ACTUATOR
                or e.feature_id in engine.symbolic_theory.actuator_features
            ):
                if not any(s.pos == e.grid_pos for s in switch_objs):
                    switch_objs.append(
                        ObjectSpec(
                            id=f"switch_{e.grid_pos[0]}_{e.grid_pos[1]}",
                            pos=e.grid_pos,
                            feature_id=e.feature_id,
                            role="switch",
                        )
                    )

        # ── Autonomous First-Principles Plan Synthesis ────────────────────────
        # 1. HELD CARGO / EXTENDED BODY SCHEMA:
        # "if box can pick, pick will be macro task for object called box.
        #  also if it need to move it need to be picked and then travel. like that."
        held_tools = engine.working_memory.body_schema.held_tools
        has_held = bool(held_tools) or engine.working_memory.body_schema.has_manipulandum()
        if has_held and engine.avatar_pos is not None:
            held_feat = held_tools[-1].feature_id if held_tools else None
            # If a known barrier blocks the goal:
            barriers_to_clear = []
            for b in engine.learned_barriers:
                b_feat = (
                    int(curr_grid[b[0], b[1]])
                    if 0 <= b[0] < curr_grid.shape[0] and 0 <= b[1] < curr_grid.shape[1]
                    else None
                )
                if hasattr(
                    engine.symbolic_theory, "is_tool_barrier_incompatible"
                ) and engine.symbolic_theory.is_tool_barrier_incompatible(held_feat, b_feat):
                    continue
                if (
                    hasattr(engine.symbolic_theory, "has_tool_barrier_affinity")
                    and engine.symbolic_theory.has_tool_barrier_affinity(held_feat, b_feat)
                ) or held_feat is not None:
                    barriers_to_clear.append(b)
            if barriers_to_clear:
                closest_barrier = min(
                    barriers_to_clear,
                    key=lambda b: (
                        abs(b[0] - engine.avatar_pos[0]) + abs(b[1] - engine.avatar_pos[1])
                    ),
                )
                plan.append(UseToolOnBarrier(barrier_pos=closest_barrier, tool_feature=held_feat))
                if goals:
                    plan.append(NavigateTo(destination=goals[0]))
                return plan

            if receptacles:
                closest_rec = min(
                    receptacles,
                    key=lambda r: (
                        abs(r[0] - engine.avatar_pos[0]) + abs(r[1] - engine.avatar_pos[1])
                    ),
                )
                plan.append(DepositCargo(receptacle_pos=closest_rec, cargo_feature=held_feat))
                if goals:
                    plan.append(NavigateTo(destination=goals[0]))
                return plan

            if goals:
                plan.append(DepositCargo(receptacle_pos=goals[0], cargo_feature=held_feat))
                return plan

        # 2. PREREQUISITE UNHELD CARGO / BOX ACQUISITION & MOVEMENT:
        # If goals are blocked or receptacles are unsatisfied, acquire cargo first
        if cargo_objs and engine.avatar_pos is not None:
            av_pos = engine.avatar_pos
            unpicked_cargo = [
                c
                for c in cargo_objs
                if c.pos != av_pos
                and not engine.working_memory.body_schema.is_holding(c.feature_id)
            ]
            if unpicked_cargo:
                best_cargo = min(
                    unpicked_cargo,
                    key=lambda c: (
                        trial_counts.get(c.id, 0),
                        abs(c.pos[0] - av_pos[0]) + abs(c.pos[1] - av_pos[1]),
                    ),
                )
                target_dest = None
                if receptacles:
                    target_dest = min(
                        receptacles,
                        key=lambda r: abs(r[0] - best_cargo.pos[0]) + abs(r[1] - best_cargo.pos[1]),
                    )
                elif goals:
                    target_dest = goals[0]

                plan.append(
                    PickUpCargo(
                        cargo_pos=best_cargo.pos,
                        cargo_feature=best_cargo.feature_id,
                        description=f"Pick up cargo {best_cargo.id} at {best_cargo.pos}",
                    )
                )
                if target_dest is not None:
                    plan.append(
                        DepositCargo(
                            receptacle_pos=target_dest,
                            cargo_feature=best_cargo.feature_id,
                            description=f"Travel with cargo to {target_dest}",
                        )
                    )
                if goals:
                    plan.append(
                        NavigateTo(destination=goals[0], description="Navigate to exit goal")
                    )

                trial_counts[best_cargo.id] = trial_counts.get(best_cargo.id, 0) + 1
                return plan

        # 3. INTERACTIVE SWITCHES / BUTTONS (Distal Actuators):
        if switch_objs:
            av_pos = engine.avatar_pos or (0, 0)
            sorted_switches = sorted(
                switch_objs,
                key=lambda s: (
                    trial_counts.get(s.id, 0),
                    abs(s.pos[0] - av_pos[0]) + abs(s.pos[1] - av_pos[1]),
                ),
            )
            best_switch = sorted_switches[0]
            has_click = any(engine.is_spatial_effector(a) or a == 6 for a in available_actions)
            if has_click and (engine.avatar_pos is None or set(available_actions).issubset({6})):
                try:
                    from plugins.arc_agi_adapter.arc_macro_actions import ArcClickCell

                    plan.append(
                        ArcClickCell(
                            cell_coord=best_switch.pos,
                            description=f"Click switch at {best_switch.pos}",
                        )
                    )
                except Exception:
                    plan.append(
                        InteractWith(
                            target_coord=best_switch.pos,
                            description=f"Click switch at {best_switch.pos}",
                        )
                    )
            else:
                plan.append(
                    InteractWith(
                        target_coord=best_switch.pos,
                        description=f"Interact with switch at {best_switch.pos}",
                    )
                )

            if goals:
                plan.append(
                    NavigateTo(destination=goals[0], description="Reach goal unblocked by switch")
                )

            trial_counts[best_switch.id] = trial_counts.get(best_switch.id, 0) + 1
            return plan

        # 4. TERMINAL GOAL REACHABILITY:
        if goals and engine.avatar_pos is not None:
            sorted_goals = sorted(
                goals,
                key=lambda g: abs(g[0] - engine.avatar_pos[0]) + abs(g[1] - engine.avatar_pos[1]),
            )
            target_goal = sorted_goals[0]
            plan.append(NavigateTo(destination=target_goal, description="Reach exit goal"))
            if has_multi_agent and focus_switch is not None:
                plan.append(
                    SwitchFocus(
                        switch_action=focus_switch, description="Handover after reaching goal"
                    )
                )
            return plan

        # 5. GENERAL EPISTEMIC EXPLORATION: "trying everything on available objects"
        if known_objs:
            av_pos = engine.avatar_pos or (0, 0)
            explorable = [
                obj
                for obj in known_objs
                if "barrier" not in obj.role and "hazard" not in obj.role and obj.pos != av_pos
            ]
            if explorable:
                best_obj = min(
                    explorable,
                    key=lambda o: (
                        trial_counts.get(o.id, 0),
                        abs(o.pos[0] - av_pos[0]) + abs(o.pos[1] - av_pos[1]),
                    ),
                )
                trial_counts[best_obj.id] = trial_counts.get(best_obj.id, 0) + 1

                if best_obj.macro_tasks:
                    for task in best_obj.macro_tasks:
                        plan.append(task)
                    return plan

                if any(engine.is_spatial_effector(a) or a == 6 for a in available_actions):
                    try:
                        from plugins.arc_agi_adapter.arc_macro_actions import ArcClickCell

                        plan.append(
                            ArcClickCell(
                                cell_coord=best_obj.pos,
                                description=f"Try probing object {best_obj.id} at {best_obj.pos}",
                            )
                        )
                    except Exception:
                        plan.append(
                            InteractWith(
                                target_coord=best_obj.pos,
                                description=f"Try probing object {best_obj.id} at {best_obj.pos}",
                            )
                        )
                else:
                    plan.append(
                        ExploreObject(target_pos=best_obj.pos, target_feature=best_obj.feature_id)
                    )
                return plan

        return None
