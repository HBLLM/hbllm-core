"""ARC-AGI Domain Macro-Actions and Plan Synthesis Driver.

Decoupled peripheral adapter implementing ARC-AGI-specific macro-actions:
1. ToggleSwitch: Allocentric spatial click or geodesic navigation to actuate switches,
   levers, optical mirrors, or toggle widgets.
2. ArcClickCell: Coordinate-level direct cell clicking.
3. arc_macro_plan_synthesis_hook: Domain-specific hook registered with
   MacroActionDriverManager to synthesize ARC-AGI puzzle macro plans.

Conforms strictly to HBLLM General Intelligence Invariant (Rule 1.7):
Task-specific macro-actions and benchmark heuristics live in plugins, NEVER in core.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole, SpatialActionIntent
from hbllm.hcir.world.macro_actions import (
    MacroAction,
    MacroActionDriverManager,
    MacroActionPlan,
    MacroActionStatus,
    NavigateTo,
    ObjectSpec,
)

if TYPE_CHECKING:
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

logger = logging.getLogger(__name__)


class ToggleSwitch(MacroAction):
    """ARC-AGI Domain Macro-Action: Actuates an environmental switch, lever, button, or optical mirror.

    If a spatial effector (e.g. CLICK action 6 with {x, y}) is available in available_actions,
    dispatches an allocentric coordinate interaction at the switch locus.
    Otherwise, computes geodesic navigation adjacent to the switch and triggers physical interaction.
    """

    def __init__(
        self,
        switch_pos: tuple[int, int],
        switch_id: str = "",
        switch_feature: int = 0,
        description: str = "",
    ) -> None:
        super().__init__(name=f"ToggleSwitch({switch_pos})", intent=SpatialActionIntent.INTERACT)
        self.switch_pos = (int(switch_pos[0]), int(switch_pos[1]))
        self.switch_id = switch_id
        self.switch_feature = switch_feature
        self.description = description or f"Toggle switch at {self.switch_pos}"
        self.has_executed: bool = False
        self._nav_helper: NavigateTo | None = None

    def is_satisfied(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
    ) -> bool:
        return self.has_executed

    def is_feasible(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> bool:
        # Feasible if any spatial effector exists, or if movement actions are available
        has_effector = any(engine.is_spatial_effector(a) for a in available_actions)
        if has_effector:
            return True
        return len(available_actions) > 0

    def step(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> tuple[Any, dict[str, Any] | None] | None:
        self.consecutive_steps += 1

        # 1. Direct Allocentric Spatial Click Effector (e.g. ARC Action 6)
        effector_candidates = [a for a in available_actions if engine.is_spatial_effector(a)]
        if effector_candidates:
            eff_action = effector_candidates[0]
            data = {"x": int(self.switch_pos[1]), "y": int(self.switch_pos[0])}
            self.has_executed = True
            self.status = MacroActionStatus.SATISFIED
            logger.info(
                "ToggleSwitch: Actuated switch at %s via spatial effector %s",
                self.switch_pos,
                eff_action,
            )
            return eff_action, data

        # 2. Embodied Locomotion: Navigate adjacent to or onto switch
        if engine.avatar_pos is not None:
            ar, ac = engine.avatar_pos
            dr = abs(ar - self.switch_pos[0])
            dc = abs(ac - self.switch_pos[1])
            if dr + dc <= 1:
                # Adjacent or on top of switch: trigger discrete interact action (e.g. 5) if available
                interact_candidates = [
                    a for a in available_actions if not engine.is_displacement_action(a)
                ]
                chosen = interact_candidates[0] if interact_candidates else available_actions[0]
                self.has_executed = True
                self.status = MacroActionStatus.SATISFIED
                return chosen, None

            if self._nav_helper is None:
                self._nav_helper = NavigateTo(destination=self.switch_pos, radius=1)

            if not self._nav_helper.is_satisfied(engine, curr_grid):
                return self._nav_helper.step(engine, curr_grid, available_actions)

        # Fallback
        self.has_executed = True
        self.status = MacroActionStatus.SATISFIED
        return available_actions[0], None


class ArcClickCell(MacroAction):
    """ARC-AGI Domain Macro-Action: Direct cell click at (row, col) coordinates."""

    def __init__(
        self,
        cell_coord: tuple[int, int],
        description: str = "",
    ) -> None:
        super().__init__(name=f"ArcClickCell({cell_coord})", intent=SpatialActionIntent.INTERACT)
        self.cell_coord = (int(cell_coord[0]), int(cell_coord[1]))
        self.description = description or f"Click cell at {self.cell_coord}"
        self.has_executed: bool = False

    def is_satisfied(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
    ) -> bool:
        return self.has_executed

    def is_feasible(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> bool:
        return any(engine.is_spatial_effector(a) for a in available_actions)

    def step(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> tuple[Any, dict[str, Any] | None] | None:
        self.consecutive_steps += 1
        effector_candidates = [a for a in available_actions if engine.is_spatial_effector(a)]
        if not effector_candidates:
            self.status = MacroActionStatus.FAILED
            return available_actions[0], None

        eff = effector_candidates[0]
        data = {"x": int(self.cell_coord[1]), "y": int(self.cell_coord[0])}
        self.has_executed = True
        self.status = MacroActionStatus.SATISFIED
        return eff, data


def arc_macro_plan_synthesis_hook(
    engine: AutonomousEpistemicEngine,
    curr_grid: np.ndarray,
    available_actions: Sequence[Any],
    objects: Sequence[Any] | None = None,
) -> MacroActionPlan | None:
    """Domain-specific plan synthesis hook for ARC-AGI games.

    Detects switches, buttons, optical mirrors, and actuators, synthesizing:
    [ToggleSwitch(pos)] -> [NavigateTo(goal)]
    """
    bg = engine.estimate_background(curr_grid)
    entities = engine.extract_entities(curr_grid, bg)

    # 1. Identify switch / actuator entities
    switch_positions: list[tuple[int, int]] = []
    if objects:
        for obj in objects:
            role = ""
            pos = None
            if isinstance(obj, ObjectSpec):
                role = obj.role.lower()
                pos = obj.pos
            elif hasattr(obj, "role") and hasattr(obj, "grid_pos"):
                role = str(obj.role.value if hasattr(obj.role, "value") else obj.role).lower()
                pos = obj.grid_pos
            elif isinstance(obj, dict):
                role = str(obj.get("role", "")).lower()
                pos = obj.get("pos", obj.get("position", obj.get("grid_pos")))

            if pos is not None and (
                "switch" in role
                or "actuator" in role
                or "lever" in role
                or "button" in role
                or "mirror" in role
            ):
                switch_positions.append((int(pos[0]), int(pos[1])))

    for e in entities:
        if (
            e.role == EntityRole.ACTUATOR
            or e.feature_id in engine.symbolic_theory.actuator_features
        ):
            if e.grid_pos not in switch_positions:
                switch_positions.append(e.grid_pos)

    if not switch_positions:
        return None

    # 2. Identify exit goals
    goals: list[tuple[int, int]] = []
    if objects:
        for obj in objects:
            role = ""
            pos = None
            if isinstance(obj, ObjectSpec):
                role = obj.role.lower()
                pos = obj.pos
            elif hasattr(obj, "role") and hasattr(obj, "grid_pos"):
                role = str(obj.role.value if hasattr(obj.role, "value") else obj.role).lower()
                pos = obj.grid_pos
            elif isinstance(obj, dict):
                role = str(obj.get("role", "")).lower()
                pos = obj.get("pos", obj.get("position", obj.get("grid_pos")))

            if pos is not None and ("goal" in role or "exit" in role):
                goals.append((int(pos[0]), int(pos[1])))

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

    # 3. Synthesize MacroActionPlan: [ToggleSwitch] -> [NavigateTo]
    plan = MacroActionPlan(plan_id="arc_switch_plan")
    for s_pos in switch_positions:
        plan.append(ToggleSwitch(switch_pos=s_pos))

    if goals:
        if engine.avatar_pos is not None:
            sorted_goals = sorted(
                goals,
                key=lambda g: abs(g[0] - engine.avatar_pos[0]) + abs(g[1] - engine.avatar_pos[1]),
            )
            plan.append(NavigateTo(destination=sorted_goals[0], description="Reach exit goal"))
        else:
            plan.append(NavigateTo(destination=goals[0], description="Reach exit goal"))

    return plan


def register_arc_macro_actions() -> None:
    """Register ARC-AGI domain macro-actions and synthesis hooks into the Core MacroActionDriverManager."""
    MacroActionDriverManager.register_macro_action("ToggleSwitch", ToggleSwitch)
    MacroActionDriverManager.register_macro_action("ArcClickCell", ArcClickCell)
    MacroActionDriverManager.register_synthesis_hook(arc_macro_plan_synthesis_hook)
    logger.info("Registered ARC-AGI macro actions and synthesis hooks in MacroActionDriverManager")
