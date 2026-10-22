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
        engine.entity_visit_counts[f"click_{self.cell_coord[0]}_{self.cell_coord[1]}"] = (
            engine.entity_visit_counts.get(f"click_{self.cell_coord[0]}_{self.cell_coord[1]}", 0)
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


def _recall_procedural_skill_plan(
    game_id: str,
    level_index: int,
    game: Any = None,
    engine: Any = None,
) -> MacroActionPlan | None:
    """Recall a compiled procedural skill from core PrefrontalWorkingMemory or ProceduralMemory."""
    # Ensure current game identity matches the target skill game_id to avoid cross-game pollution
    cur_gid = (
        getattr(engine, "current_game_id", None)
        or getattr(game, "game_id", None)
        or getattr(getattr(engine, "_arc_game", None), "game_id", None)
    )
    if cur_gid:
        clean_cur = str(cur_gid).lower().split("-")[0].strip()
        if clean_cur and clean_cur != game_id.lower():
            return None

    skill_name = f"{game_id}_lvl_{level_index}"

    # 1. Check working memory procedural library if engine is present
    if engine is not None and hasattr(engine, "working_memory"):
        skill = engine.working_memory.query_procedural_skill(skill_name)
        if skill is not None and hasattr(skill, "compile_macro_plan"):
            return skill.compile_macro_plan()

    # 2. Query core ProceduralMemory SQLite store
    import os
    from pathlib import Path

    from hbllm.memory.procedural import ProceduralMemory

    db_paths = [
        Path("core/procedural_memory.db"),
        Path("procedural_memory.db"),
        Path(__file__).resolve().parent.parent.parent.parent / "core" / "procedural_memory.db",
        Path(__file__).resolve().parent.parent.parent / "procedural_memory.db",
    ]
    if os.environ.get("HBLLM_PROCEDURAL_DB"):
        db_paths.insert(0, Path(os.environ["HBLLM_PROCEDURAL_DB"]))

    skill_record = None
    for p in db_paths:
        if p.exists():
            pm = ProceduralMemory(db_path=p)
            skill_record = pm.get_skill_sync("arc_agi", skill_name)
            if skill_record:
                break

    if not skill_record:
        return None

    steps = skill_record.get("steps", [])
    if not steps:
        return None

    plan = MacroActionPlan(plan_id=f"{game_id}_procedural_lvl_{level_index}", is_strict=True)

    for s in steps:
        stype = s.get("type")
        if stype == "click_cell":
            r, c = s["cell_coord"]
            plan.append(
                ArcClickCell(
                    cell_coord=(int(r), int(c)), description=f"Procedural click ({r}, {c})"
                )
            )
        elif stype == "primitive":
            plan.append(
                ArcPrimitiveAction(
                    action_id=int(s["action_id"]), description=f"Procedural action {s['action_id']}"
                )
            )
        elif stype == "primitive_with_data":
            plan.append(
                ArcPrimitiveAction(
                    action_id=int(s["action_id"]),
                    data=s.get("data"),
                    description=f"Procedural action {s['action_id']}",
                )
            )
        elif stype == "click_xy":
            plan.append(
                ArcPrimitiveAction(
                    action_id=6,
                    data={"x": int(s["x"]), "y": int(s["y"])},
                    description=f"Procedural click ({s['x']}, {s['y']})",
                )
            )

    return plan if plan.actions else None


class DynamicARCArchetypeClassifier:
    """Lightweight compatibility descriptor for ARC game environments."""

    @classmethod
    def classify(
        cls,
        game: Any = None,
        curr_grid: np.ndarray | None = None,
        available_actions: Sequence[Any] | None = None,
        objects: Sequence[Any] | None = None,
    ) -> Any | None:
        return None


def arc_macro_plan_synthesis_hook(
    engine: AutonomousEpistemicEngine,
    curr_grid: np.ndarray,
    available_actions: Sequence[Any],
    objects: Sequence[Any] | None = None,
) -> MacroActionPlan | None:
    """Domain-specific plan synthesis hook for ARC-AGI games.

    Recalls previously consolidated declarative skills from core ProceduralMemory.
    If no skill is recorded, returns None so that the cognitive core
    (MacroPlanSynthesizer) autonomously plans the action sequence from first principles
    over the individual objects and atomic macro-tasks stored in the Knowledge Base.
    """
    game = getattr(engine, "_arc_game", None)
    if game is not None:
        raw_gid = (
            getattr(engine, "current_game_id", None)
            or getattr(game, "game_id", None)
            or getattr(game, "_game_id", "")
        )
        if raw_gid:
            clean_gid = str(raw_gid).split("-")[0].strip().lower()
            lvl_idx = int(getattr(game, "level_index", getattr(game, "_current_level_index", 0)))
            plan = _recall_procedural_skill_plan(clean_gid, lvl_idx, game=game, engine=engine)
            if plan is not None and len(plan.actions) > 0:
                return plan

    # 1. Allocentric Click Games (Action 6) with Interactive Buttons / Switches
    if objects and any(engine.is_spatial_effector(a) or a == 6 for a in available_actions):
        is_click_only = set(available_actions).issubset({6})
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
                    key=lambda b: engine.entity_visit_counts.get(f"click_{b[0]}_{b[1]}", 0),
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
