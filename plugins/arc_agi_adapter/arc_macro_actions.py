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

from hbllm.hcir.spatial_planner import SpatialActionIntent
from hbllm.hcir.world.macro_actions import (
    MacroAction,
    MacroActionDriverManager,
    MacroActionPlan,
    MacroActionStatus,
    NavigateTo,
    ObjectSpec,
    SwitchFocus,
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
        # Only actuate remotely if the agent is unembodied or the environment is click-only
        is_click_only = set(available_actions).issubset({6})
        effector_candidates = [a for a in available_actions if engine.is_spatial_effector(a)]
        if effector_candidates and (engine.avatar_pos is None or is_click_only):
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
        return any(engine.is_spatial_effector(a) or a == 6 for a in available_actions)

    def step(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> tuple[Any, dict[str, Any] | None] | None:
        self.consecutive_steps += 1
        effector_candidates = [
            a for a in available_actions if engine.is_spatial_effector(a) or a == 6
        ]
        if not effector_candidates:
            self.status = MacroActionStatus.FAILED
            return available_actions[0], None

        eff = effector_candidates[0]
        data = {"x": int(self.cell_coord[1]), "y": int(self.cell_coord[0])}
        # Update visit count for this click coordinate so exploration alternates when quiescent
        engine.entity_visit_counts[f"actuate_{self.cell_coord[0]}_{self.cell_coord[1]}"] = (
            engine.entity_visit_counts.get(f"actuate_{self.cell_coord[0]}_{self.cell_coord[1]}", 0)
            + 1
        )
        self.has_executed = True
        self.status = MacroActionStatus.SATISFIED
        return eff, data


class ArcPrimitiveAction(MacroAction):
    """ARC-AGI Domain Macro-Action: Dispatches a specific primitive action (e.g. directional move or click)."""

    def __init__(
        self,
        action_id: Any,
        data: dict[str, Any] | None = None,
        description: str = "",
    ) -> None:
        super().__init__(
            name=f"ArcPrimitiveAction({action_id})", intent=SpatialActionIntent.NAVIGATE
        )
        self.action_id = action_id
        self.data = data
        self.description = description or f"Execute action {action_id}"
        self.has_executed: bool = False
        self.is_strict: bool = True

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
        return True

    def step(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> tuple[Any, dict[str, Any] | None] | None:
        self.consecutive_steps += 1
        self.has_executed = True
        self.status = MacroActionStatus.SATISFIED
        chosen_action = self.action_id
        for a in available_actions:
            if getattr(a, "value", None) == self.action_id or a == self.action_id:
                chosen_action = a
                break
        return chosen_action, self.data


def arc_macro_plan_synthesis_hook(
    engine: AutonomousEpistemicEngine,
    curr_grid: np.ndarray,
    available_actions: Sequence[Any],
    objects: Sequence[Any] | None = None,
) -> MacroActionPlan | None:
    """Domain-specific plan synthesis hook for ARC-AGI games.

    Provides fast allocentric button and multi-shape silhouette macros.
    If none apply, returns None so that the cognitive core (MacroPlanSynthesizer)
    autonomously plans from first principles.
    """
    # 1. Allocentric Click Games with Interactive Buttons / Switches
    if objects and any(engine.is_spatial_effector(a) or a == 6 for a in available_actions):
        is_click_only = not any(engine.is_displacement_action(a) for a in available_actions)
        if is_click_only:
            btn_coords: list[tuple[int, int]] = []
            for obj in objects:
                if isinstance(obj, ObjectSpec) and ("switch" in obj.role or "button" in obj.role):
                    click_coord = obj.properties.get("click_coord") or obj.pos
                    if click_coord:
                        btn_coords.append((int(click_coord[0]), int(click_coord[1])))
            if btn_coords:
                sorted_btns = sorted(
                    btn_coords,
                    key=lambda b: engine.entity_visit_counts.get(f"actuate_{b[0]}_{b[1]}", 0),
                )
                plan = MacroActionPlan(plan_id="arc_click_button_plan")
                plan.append(
                    ArcClickCell(
                        cell_coord=sorted_btns[0], description=f"Actuate button at {sorted_btns[0]}"
                    )
                )
                return plan

    # 2. Multi-Agent / Multi-Shape Silhouette Puzzles with Focus Switch (Action 5)
    if objects and 5 in available_actions:
        shapes = [
            obj for obj in objects if isinstance(obj, ObjectSpec) and obj.role == "manipulandum"
        ]
        silhouettes = [
            obj
            for obj in objects
            if isinstance(obj, ObjectSpec) and obj.role in ("receptacle", "goal")
        ]
        if shapes and silhouettes:
            active_shape = next(
                (sh for sh in shapes if sh.properties.get("is_active")),
                shapes[0],
            )
            engine.avatar_pos = active_shape.pos
            engine.avatar_feature = active_shape.feature_id

            matching_sil = next(
                (sil for sil in silhouettes if sil.feature_id == active_shape.feature_id),
                silhouettes[0],
            )
            dr = abs(active_shape.pos[0] - matching_sil.pos[0])
            dc = abs(active_shape.pos[1] - matching_sil.pos[1])
            plan = MacroActionPlan(plan_id="arc_shape_silhouette_plan")
            if dr + dc > 0:
                plan.append(
                    NavigateTo(
                        destination=matching_sil.pos,
                        radius=0,
                        description=f"Fit shape {active_shape.id} to silhouette at {matching_sil.pos}",
                    )
                )
            plan.append(SwitchFocus(switch_action=5, description="Switch active shape"))
            return plan

    return None


def register_arc_macro_actions() -> None:
    """Register ARC-AGI domain macro-actions and synthesis hooks into the Core MacroActionDriverManager."""
    MacroActionDriverManager.register_macro_action("ToggleSwitch", ToggleSwitch)
    MacroActionDriverManager.register_macro_action("ArcClickCell", ArcClickCell)
    MacroActionDriverManager.register_macro_action("ArcPrimitiveAction", ArcPrimitiveAction)
    MacroActionDriverManager.register_synthesis_hook(arc_macro_plan_synthesis_hook)
    logger.info("Registered ARC-AGI macro actions and synthesis hooks in MacroActionDriverManager")
