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
    role: str = "unknown"  # "goal", "cargo", "key", "receptacle", "switch", "barrier", "hazard"
    properties: dict[str, Any] = field(default_factory=dict)


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
    ) -> tuple[Any, dict[str, Any] | None]:
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
    ) -> tuple[Any, dict[str, Any] | None]:
        self.consecutive_steps += 1
        if self.consecutive_steps > self.max_steps:
            self.status = MacroActionStatus.FAILED
            logger.warning(
                "MacroAction %s timed out after %d steps", self.name, self.consecutive_steps
            )
            return available_actions[0], None

        from hbllm.hcir.world.topological_graph_induction import GeodesicManifoldRouter

        av_pos = engine.avatar_pos
        if av_pos is None:
            self.status = MacroActionStatus.BLOCKED
            return available_actions[0], None

        movable_actions = [
            (a, engine.action_dynamics[a].delta_r, engine.action_dynamics[a].delta_c)
            for a in available_actions
            if engine.is_displacement_action(a) and a in engine.action_dynamics
        ]

        av_feats = engine.avatar_features or (
            {engine.avatar_feature} if engine.avatar_feature is not None else set()
        )

        # Plan next geodesic step
        plan = GeodesicManifoldRouter.route_geodesic(
            curr_grid=curr_grid,
            start_pos=av_pos,
            goals=[self.destination],
            static_barriers=engine.learned_barriers,
            grid_walls=set(),
            unknown_entity_cells=set(),
            oriented_threats=[],
            movable_actions=movable_actions,
            bg=engine.bg_feature or 0,
            barrier_features=engine.symbolic_theory.barrier_features,
            avatar_features=av_feats,
            max_candidate_goals=1,
            failed_transitions=engine.failed_transitions,
        )

        if plan and len(plan) > 0:
            next_act, next_pos = plan[0]
            if next_act in available_actions:
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

        # Fallback: exploratory probe toward destination
        disp_acts = [a for a in available_actions if engine.is_displacement_action(a)]
        if disp_acts:
            best_a = min(
                disp_acts,
                key=lambda a: (
                    (av_pos[0] + engine.action_dynamics[a].delta_r - self.destination[0]) ** 2
                    + (av_pos[1] + engine.action_dynamics[a].delta_c - self.destination[1]) ** 2
                    if a in engine.action_dynamics
                    else 9999
                ),
            )
            return best_a, None

        return available_actions[0], None


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
        description: str = "",
    ) -> None:
        super().__init__(name=f"PickUpCargo({cargo_pos})", intent=SpatialActionIntent.PICKUP)
        self.cargo_pos = (int(cargo_pos[0]), int(cargo_pos[1]))
        self.cargo_feature = cargo_feature
        self.description = description or f"Pick up cargo at {self.cargo_pos}"
        self.nav_op = NavigateTo(destination=self.cargo_pos, radius=1)

    def is_satisfied(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
    ) -> bool:
        if engine.working_memory.body_schema.has_manipulandum():
            return True
        if engine.avatar_pos is None:
            return False
        # Also satisfied if avatar is touching cargo
        dr = abs(engine.avatar_pos[0] - self.cargo_pos[0])
        dc = abs(engine.avatar_pos[1] - self.cargo_pos[1])
        return dr + dc <= 1 and (
            engine.avatar_pos == self.cargo_pos or self.nav_op.consecutive_steps > 0
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
    ) -> tuple[Any, dict[str, Any] | None]:
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
        description: str = "",
    ) -> None:
        super().__init__(name=f"DepositCargo({receptacle_pos})", intent=SpatialActionIntent.DROP)
        self.receptacle_pos = (int(receptacle_pos[0]), int(receptacle_pos[1]))
        self.description = description or f"Deposit cargo at {self.receptacle_pos}"
        self.nav_op = NavigateTo(destination=self.receptacle_pos, radius=1)

    def is_satisfied(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
    ) -> bool:
        # Satisfied if cargo is at or adjacent to receptacle
        active_manip = engine.working_memory.body_schema.active_manipulandum
        if active_manip is not None:
            mr, mc = active_manip.pos
            if abs(mr - self.receptacle_pos[0]) + abs(mc - self.receptacle_pos[1]) <= 1:
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
    ) -> tuple[Any, dict[str, Any] | None]:
        return self.nav_op.step(engine, curr_grid, available_actions)


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
        """Register a domain-specific plan synthesis hook."""
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
# 8. CustomMacroAction (Runner / Extensible Operator)
# ─────────────────────────────────────────────────────────────────────────────


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
# 9. MacroActionPlan Scheduler
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MacroActionPlan:
    """Sequential queue of macro-action operators managed by PrefrontalWorkingMemory."""

    actions: deque[MacroAction] = field(default_factory=deque)
    plan_id: str = "macro_plan_default"

    def is_empty(self) -> bool:
        return len(self.actions) == 0

    def current(self) -> MacroAction | None:
        return self.actions[0] if self.actions else None

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
    ) -> bool:
        """Advance the plan if the active operator's postconditions are met."""
        while self.actions:
            cur = self.actions[0]
            if cur.is_satisfied(engine, curr_grid):
                popped = self.actions.popleft()
                logger.info(
                    "MacroActionPlan: Completed and popped %s (%d remaining)",
                    popped.name,
                    len(self.actions),
                )
            else:
                break
        return len(self.actions) > 0

    def step(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> tuple[Any, dict[str, Any] | None] | None:
        """Execute the current macro-action."""
        self.advance_if_satisfied(engine, curr_grid)
        if not self.actions:
            return None
        cur = self.actions[0]
        if not cur.is_feasible(engine, curr_grid, available_actions):
            logger.warning(
                "MacroAction %s is infeasible in current state; dropping operator", cur.name
            )
            self.actions.popleft()
            return None
        return cur.step(engine, curr_grid, available_actions)


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
                if isinstance(obj, ObjectSpec):
                    role = obj.role.lower()
                    pos = obj.pos
                    feat = obj.feature_id
                    props = obj.properties
                elif hasattr(obj, "role") and hasattr(obj, "grid_pos"):
                    role = str(obj.role.value if hasattr(obj.role, "value") else obj.role).lower()
                    pos = obj.grid_pos
                    feat = getattr(obj, "feature_id", 0) or 0
                    props = getattr(obj, "properties", {})
                elif isinstance(obj, dict):
                    role = str(obj.get("role", "unknown")).lower()
                    pos = obj.get("pos", obj.get("position", obj.get("grid_pos")))
                    feat = obj.get("feature_id", obj.get("feature", 0)) or 0
                    props = obj.get("properties", {})

                if pos is not None:
                    pos = (int(pos[0]), int(pos[1]))
                    if "goal" in role or "exit" in role:
                        engine.learned_goal_positions.add(pos)
                        if pos not in engine.goals:
                            engine.goals.append(pos)
                        if feat:
                            engine.symbolic_theory.goal_features.add(feat)
                            engine.learned_goal_features.add(feat)
                    elif "cargo" in role or "key" in role or "manipulable" in role:
                        if feat:
                            engine.symbolic_theory.cargo_features.add(feat)
                            engine.learned_cargo_features.add(feat)
                    elif "receptacle" in role or "door" in role or "container" in role:
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
        """Query ObjectStateGraph, driver hooks, and SymbolicWorldTheory to synthesize a plan."""
        if objects or micro_actions:
            cls.ingest_knowledge(engine, objects=objects, micro_actions=micro_actions)

        # 0. Query registered domain drivers & plugins (e.g. ARC-AGI, AI2-THOR, Crafter)
        driver_plan = MacroActionDriverManager.run_synthesis_hooks(
            engine, curr_grid, available_actions, objects=objects
        )
        if driver_plan is not None and not driver_plan.is_empty():
            return driver_plan

        plan = MacroActionPlan()
        bg = engine.estimate_background(curr_grid)
        entities = engine.extract_entities(curr_grid, bg)

        goals = [
            e.grid_pos
            for e in entities
            if e.role == EntityRole.GOAL
            or e.feature_id in engine.learned_goal_features
            or e.grid_pos in engine.learned_goal_positions
        ]
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

        # 1. Multi-Stage Subgoals: Key / Cargo Delivery to Receptacle
        cargo_entities = [
            e
            for e in entities
            if e.role == EntityRole.MANIPULABLE or e.feature_id in engine.learned_cargo_features
        ]
        receptacles = [
            e.grid_pos
            for e in entities
            if e.role == EntityRole.RECEPTACLE or e.grid_pos in engine.learned_receptacle_positions
        ]
        for r_pos in engine.learned_receptacle_positions:
            if r_pos not in receptacles:
                receptacles.append(r_pos)

        if cargo_entities and receptacles and engine.avatar_pos is not None:
            cargo = cargo_entities[0]
            receptacle = receptacles[0]
            plan.append(PickUpCargo(cargo_pos=cargo.grid_pos, cargo_feature=cargo.feature_id))
            plan.append(DepositCargo(receptacle_pos=receptacle))
            if goals:
                plan.append(NavigateTo(destination=goals[0]))
            return plan

        # 2. Terminal Goal: Direct Geodesic Navigation to Verified Exit
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

        # 3. Exploratory Frontier Fallback
        if engine.avatar_pos is not None:
            plan.append(ExploreFrontier())
            return plan

        return None
