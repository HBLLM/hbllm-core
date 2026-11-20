"""Macro Action Plan Execution, Synthesis, and Cortical Stage."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole
from hbllm.hcir.transactions import TransactionOperation
from hbllm.hcir.world.cortical_stage import StageResult
from hbllm.hcir.world.engine_state import CognitiveContext, get_engine_state_attr
from hbllm.hcir.world.frontopolar_subgoal_stack import SubgoalType
from hbllm.hcir.world.macro.custom import (
    attribute_macro_action_outcome,
)
from hbllm.hcir.world.macro.driver import MacroActionDriverManager
from hbllm.hcir.world.macro.models import (
    MacroAction,
    MacroActionStatus,
    MicroActionSpec,
    ObjectSpec,
    get_sub_attr,
)
from hbllm.hcir.world.macro.primitives import (
    DepositCargo,
    ExploreObject,
    InteractWith,
    NavigateTo,
    PickUpCargo,
    SwitchFocus,
    UseToolOnBarrier,
)
from hbllm.hcir.world.percept import TerminalSignal
from hbllm.hcir.world.topological_graph_induction import GeodesicManifoldRouter

if TYPE_CHECKING:
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

logger = logging.getLogger(__name__)


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
            plan = skill.compile_to_macro_plan(engine, curr_grid, available_actions)
            if plan is not None and not plan.is_empty():
                return plan

        program = getattr(skill, "program", None)
        subgoals = getattr(program, "subgoals", None)
        if not subgoals:
            return None

        cursor_pos = getattr(engine, "avatar_pos", None)
        if cursor_pos is None:
            return None

        bg = engine.estimate_background(curr_grid)
        entities = engine.extract_entities(curr_grid, bg)
        theory = getattr(engine, "symbolic_theory", None)
        H, W = curr_grid.shape

        def _find_target(
            intent_str: str, tq: dict[str, Any], cur_p: tuple[int, int]
        ) -> tuple[tuple[int, int] | None, int | None]:
            pos_q = tq.get("pos") or tq.get("position")
            if isinstance(pos_q, (tuple, list)) and len(pos_q) == 2:
                rr, cc = int(pos_q[0]), int(pos_q[1])
                if 0 <= rr < H and 0 <= cc < W:
                    return (rr, cc), int(curr_grid[rr, cc])

            req_feat = tq.get("feature", tq.get("feature_id"))
            role_q = str(tq.get("role", "")).lower()
            cand_feats: set[int] = set()
            if req_feat is not None:
                cand_feats.add(int(req_feat))
            elif theory is not None:
                if (
                    "pickup" in intent_str
                    or "cargo" in role_q
                    or "key" in role_q
                    or "resource" in role_q
                ):
                    cand_feats.update(getattr(theory, "cargo_features", set()))
                elif (
                    "interact" in intent_str
                    or "actuate" in intent_str
                    or "switch" in role_q
                    or "actuator" in role_q
                ):
                    cand_feats.update(getattr(theory, "actuator_features", set()))
                elif (
                    "drop" in intent_str
                    or "barrier" in role_q
                    or "door" in role_q
                    or "portal" in role_q
                ):
                    cand_feats.update(getattr(theory, "conditional_affordances", {}).keys())
                elif "nav" in intent_str or "goal" in role_q:
                    for gp in getattr(engine, "learned_goal_positions", ()):
                        if 0 <= gp[0] < H and 0 <= gp[1] < W:
                            return gp, int(curr_grid[gp[0], gp[1]])
                    cand_feats.update(getattr(theory, "goal_features", set()))

            best_coord: tuple[int, int] | None = None
            best_feat: int | None = None
            best_dist = 10**9
            for ent in entities:
                ent_role = str(getattr(ent.role, "value", ent.role)).lower()
                feat_match = bool(cand_feats and ent.feature_id in cand_feats)
                role_match = bool(role_q and role_q in ent_role)
                if not feat_match and not role_match:
                    continue
                ent_cells = (
                    ent.properties.get("cells")
                    if isinstance(getattr(ent, "properties", None), dict)
                    else None
                ) or ([ent.grid_pos] if getattr(ent, "grid_pos", None) is not None else [])
                for pt in ent_cells:
                    r_p, c_p = int(pt[0]), int(pt[1])
                    if (r_p, c_p) == cur_p:
                        continue
                    d = abs(r_p - cur_p[0]) + abs(c_p - cur_p[1])
                    if d < best_dist:
                        best_dist = d
                        best_coord = (r_p, c_p)
                        best_feat = int(ent.feature_id)
            return best_coord, best_feat

        plan = cls()
        for sg in subgoals:
            intent_raw = getattr(sg, "intent", "")
            intent_val = str(getattr(intent_raw, "value", intent_raw)).upper()
            tq = dict(getattr(sg, "target_query", {}) or {})
            target_pos, target_feat = _find_target(intent_val.lower(), tq, cursor_pos)
            if target_pos is None:
                return None

            if "PICKUP" in intent_val and target_feat is not None:
                plan.append(
                    PickUpCargo(
                        cargo_pos=target_pos,
                        cargo_feature=target_feat,
                        description=f"Skill subgoal PICKUP at {target_pos}",
                    )
                )
            elif "INTERACT" in intent_val or "ACTUATE" in intent_val or "ACTIVATE" in intent_val:
                plan.append(
                    InteractWith(
                        target_coord=target_pos,
                        description=f"Skill subgoal INTERACT at {target_pos}",
                    )
                )
            elif "DROP" in intent_val:
                held = getattr(engine.working_memory.body_schema, "held_tools", ())
                tool_f = held[0].feature_id if held else (target_feat or 0)
                plan.append(
                    UseToolOnBarrier(
                        barrier_pos=target_pos,
                        tool_feature=tool_f,
                        description=f"Skill subgoal DROP/USE at {target_pos}",
                    )
                )
            elif "NAV" in intent_val:
                plan.append(
                    NavigateTo(
                        destination=target_pos,
                        description=f"Skill subgoal NAVIGATE to {target_pos}",
                    )
                )
            else:
                return None
            cursor_pos = target_pos

        return plan if not plan.is_empty() else None


# ─────────────────────────────────────────────────────────────────────────────
# 10. MacroPlanSynthesizer (Core Planner Bridge)
# ─────────────────────────────────────────────────────────────────────────────


class MacroPlanSynthesizer:
    """Synthesizes high-level MacroActionPlans from Knowledge Base state and external specs."""

    @classmethod
    def ingest_knowledge(
        cls,
        engine: AutonomousEpistemicEngine | Any,
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
                        _params = act_spec.get("parameters")
                        if act_spec.get("is_spatial_effector", False) or (
                            isinstance(_params, dict)
                            and any(
                                k in _params
                                for k in (
                                    "x",
                                    "y",
                                    "col",
                                    "row",
                                    "lat",
                                    "lon",
                                    "azimuth",
                                    "elevation",
                                    "distance",
                                    "c",
                                    "r",
                                )
                            )
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
                        sw_acts = props.get("focus_switch_actions")
                        if sw_acts:
                            for sw_a in sw_acts:
                                engine.focus_switch_actions.add(sw_a)
                                engine.working_memory.focus_switch_actions.add(sw_a)

                    av_feats = set(get_engine_state_attr(engine, "avatar_features", set()) or set())
                    if getattr(engine, "avatar_feature", None) is not None:
                        av_feats.add(engine.avatar_feature)

                    if role == "goal" or "exit" in role or role == "candidate_goal":
                        rec_pos = get_engine_state_attr(engine, "recent_positions", set())
                        av_f = av_feats
                        av_p = get_engine_state_attr(engine, "avatar_pos", None)
                        if (
                            (av_p is None or pos != av_p)
                            and pos not in rec_pos
                            and (not feat or feat not in av_f)
                        ):
                            is_confirmed = bool(props.get("confirmed", False))
                            has_reward = props.get("has_terminal_reward") is True or props.get(
                                "provenance"
                            ) in ("terminal_reward", str(TerminalSignal.SUCCESS))
                            if is_confirmed and has_reward:
                                engine.learned_goal_positions.add(pos)
                                if hasattr(engine, "goals") and pos not in engine.goals:
                                    engine.goals.append(pos)
                                if feat:
                                    engine.symbolic_theory.induce_goal(
                                        feat, evidence="sync_known_objects:confirmed_terminal"
                                    )
                                engine.learned_barriers.discard(pos)
                            else:
                                if feat and (
                                    feat not in engine.hazard_tracker.mobile_entities
                                    or feat in engine.symbolic_theory.cargo_features
                                ):
                                    engine.symbolic_theory.induce_candidate_goal(
                                        feat, evidence="sync_known_objects:candidate"
                                    )
                    elif "cargo" in role or "key" in role or "box" in role or "manipulable" in role:
                        if feat:
                            engine.symbolic_theory.induce_cargo(feat, evidence="sync_known_objects")
                    elif (
                        "receptacle" in role
                        or "door" in role
                        or "container" in role
                        or "silhouette" in role
                    ):
                        engine.learned_receptacle_positions.add(pos)
                        if feat:
                            engine.symbolic_theory.induce_receptacle(
                                feat, evidence="sync_known_objects"
                            )
                    elif (
                        "barrier" in role
                        or "wall" in role
                        or "obstacle" in role
                        or props.get("is_solid")
                    ):
                        if (
                            feat not in engine.symbolic_theory.goal_features
                            and feat not in engine.symbolic_theory.candidate_goal_features
                            and pos not in engine.learned_goal_positions
                        ):
                            engine.learned_barriers.add(pos)
                            if feat:
                                engine.symbolic_theory.induce_barrier(
                                    feat, evidence="sync_known_objects:solid"
                                )
                    elif "switch" in role or "actuator" in role or "button" in role:
                        if (
                            feat
                            and feat not in av_feats
                            and (not engine.avatar_features or feat not in engine.avatar_features)
                            and feat != engine.avatar_feature
                            and feat
                            not in getattr(engine.symbolic_theory, "walkable_features", set())
                            and feat not in getattr(engine, "verified_safe_features", set())
                        ):
                            engine.symbolic_theory.induce_actuator(
                                feat, evidence="sync_known_objects"
                            )

    @classmethod
    def synthesize_plan(
        cls,
        engine: AutonomousEpistemicEngine | Any,
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
        skills = get_engine_state_attr(engine, "declarative_skills", []) or []
        for skill in skills:
            if hasattr(skill, "can_handle") and skill.can_handle(
                curr_grid, list(available_actions), metadata={"engine": engine}
            ):
                skill_plan = MacroActionPlan.from_skill(skill, engine, curr_grid, available_actions)
                if skill_plan is not None and not skill_plan.is_empty():
                    return skill_plan

        # Embodied Grounding Invariant:
        # If the domain possesses physical displacement capabilities, an embodied agent
        # cannot plan macroscopic bodily actions (navigating to switches, moving cargo, reaching goals)
        # before discovering its avatar body.
        if engine.has_displacement_capabilities(available_actions) and engine.avatar_pos is None:
            return None

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
        known_objs = list(get_sub_attr(engine.working_memory, "known_objects", {}).values())
        trial_counts = get_sub_attr(engine.working_memory, "object_trial_counts", {})

        # Collect candidate goals
        av_pos = engine.avatar_pos
        recent_pos = set(get_engine_state_attr(engine, "recent_positions", set()))
        av_feats = set(engine.avatar_features or set())
        if engine.avatar_feature is not None:
            av_feats.add(engine.avatar_feature)

        # Collect confirmed goals and candidate goals separately
        confirmed_goals: list[tuple[int, int]] = []
        candidate_goals: list[tuple[int, int]] = []
        cand_feature_map: dict[tuple[int, int], int] = {}
        cargo_feats = set(engine.symbolic_theory.cargo_features) | set(
            engine.learned_cargo_features
        )

        for obj in known_objs:
            if (
                obj.pos == av_pos
                or obj.pos in recent_pos
                or obj.feature_id in av_feats
                or obj.feature_id in cargo_feats
            ):
                continue
            is_conf = (
                bool(obj.properties.get("confirmed", False))
                or obj.role in ("goal", "exit")
                or "exit" in obj.role
            )
            if is_conf and obj.role != "candidate_goal":
                if obj.pos not in confirmed_goals:
                    confirmed_goals.append(obj.pos)
            elif obj.role == "candidate_goal" or "goal" in obj.role:
                if obj.pos not in candidate_goals:
                    candidate_goals.append(obj.pos)
                    cand_feature_map[obj.pos] = obj.feature_id

        for e in entities:
            if (
                e.grid_pos == av_pos
                or e.grid_pos in recent_pos
                or e.feature_id in av_feats
                or e.feature_id in cargo_feats
            ):
                continue
            if (
                e.role == EntityRole.GOAL
                or e.feature_id in engine.learned_goal_features
                or e.grid_pos in engine.learned_goal_positions
            ):
                if e.grid_pos not in confirmed_goals:
                    confirmed_goals.append(e.grid_pos)
            elif (
                bool(engine.symbolic_theory)
                and e.feature_id in engine.symbolic_theory.candidate_goal_features
            ):
                if e.grid_pos not in candidate_goals:
                    candidate_goals.append(e.grid_pos)
                    cand_feature_map[e.grid_pos] = e.feature_id

        for g_pos in engine.learned_goal_positions:
            if g_pos not in confirmed_goals and g_pos != av_pos and g_pos not in recent_pos:
                confirmed_goals.append(g_pos)

        if confirmed_goals:
            goals = confirmed_goals
        else:
            # When only candidate goals exist:
            # Check how many distinct non-background, non-avatar candidate goal features exist.
            cand_feats = (
                {
                    cand_feature_map.get(pos, int(curr_grid[pos[0], pos[1]]))
                    for pos in candidate_goals
                    if 0 <= pos[0] < curr_grid.shape[0] and 0 <= pos[1] < curr_grid.shape[1]
                }
                - {0, bg}
                - av_feats
            )
            if len(cand_feats) <= 1:
                # Unambiguous: exactly one candidate goal feature exists (e.g. death_reset, stride2, blocked_first)
                goals = candidate_goals
            else:
                # Ambiguous: multiple candidate features (e.g. push block vs target in push_block).
                # Defer terminal navigation to mental simulation / epistemic probing to avoid
                # prematurely walking to the target and bypassing intermediate object manipulation.
                goals = []

        # Multi-Agent focus switch verification
        has_multi_agent = (
            len(engine.working_memory.focus_switch_actions) > 0
            or len(engine.focus_switch_actions) > 0
            or len(get_sub_attr(engine.working_memory, "controllable_avatar_features", set())) > 1
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
            if (
                "cargo" in obj.role
                or "box" in obj.role
                or "key" in obj.role
                or "crate" in obj.role
                or "item" in obj.role
                or obj.feature_id in engine.learned_cargo_features
            )
            and obj.pos != av_pos
            and obj.pos not in recent_pos
            and obj.feature_id not in av_feats
        ]
        for e in entities:
            if (
                (
                    e.role in (EntityRole.MANIPULABLE, EntityRole.RESOURCE)
                    or e.feature_id in engine.learned_cargo_features
                )
                and e.grid_pos != av_pos
                and e.grid_pos not in recent_pos
                and e.feature_id not in av_feats
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

        movable_actions = [
            (a, engine.action_dynamics[a].delta_r, engine.action_dynamics[a].delta_c)
            for a in available_actions
            if engine.is_displacement_action(a) and a in engine.action_dynamics
        ]

        # Static reachability inputs are invariant for the whole synthesize_plan
        # call: curr_grid, learned_barriers and barrier_features are not mutated
        # between _can_navigate_to calls — compute walls once, lazily, and
        # memoize reachability per destination.
        _nav_static: tuple[set[int], set[tuple[int, int]]] | None = None
        _nav_cache: dict[tuple[int, int], bool] = {}

        def _can_navigate_to(dest: tuple[int, int]) -> bool:
            if av_pos is None:
                return False
            if dest == av_pos:
                return True
            dr = abs(dest[0] - av_pos[0])
            dc = abs(dest[1] - av_pos[1])
            if dr <= 1 and dc <= 1:
                return True
            if not movable_actions:
                return False
            if dest in _nav_cache:
                return _nav_cache[dest]
            nonlocal _nav_static
            if _nav_static is None:
                wall_f_new = set(engine.symbolic_theory.barrier_features)
                grid_w_new = set(engine.learned_barriers)
                if wall_f_new:
                    grid_w_new |= {
                        (int(rr), int(cc))
                        for rr, cc in np.argwhere(np.isin(curr_grid, list(wall_f_new)))
                    }
                _nav_static = (wall_f_new, grid_w_new)
            wall_f, grid_w = _nav_static
            p = GeodesicManifoldRouter.route_geodesic(
                curr_grid=curr_grid,
                start_pos=av_pos,
                goals=[dest],
                static_barriers=engine.learned_barriers,
                grid_walls=grid_w,
                unknown_entity_cells=set(),
                oriented_threats=[],
                movable_actions=movable_actions,
                bg=bg,
                barrier_features=wall_f,
                avatar_features=av_feats,
                max_candidate_goals=1,
            )
            reached = p is not None and len(p) > 0
            _nav_cache[dest] = reached
            return reached

        # Collect candidate switches / buttons / actuators
        switch_objs: list[ObjectSpec] = [
            obj
            for obj in known_objs
            if (
                "switch" in obj.role
                or "button" in obj.role
                or "actuator" in obj.role
                or "lever" in obj.role
                or "valve" in obj.role
                or obj.feature_id in engine.symbolic_theory.actuator_features
            )
            and obj.pos != av_pos
            and obj.feature_id not in av_feats
        ]
        for e in entities:
            if (
                (
                    e.role == EntityRole.ACTUATOR
                    or e.feature_id in engine.symbolic_theory.actuator_features
                )
                and e.grid_pos != av_pos
                and e.feature_id not in av_feats
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

        # Check if goals are already directly reachable
        has_directly_reachable_goal = bool(goals and any(_can_navigate_to(g) for g in goals))

        # 3. INTERACTIVE SWITCHES / BUTTONS (Distal Actuators):
        has_effector = any(
            engine.is_spatial_effector(a) or not engine.is_displacement_action(a)
            for a in available_actions
        )
        if switch_objs and has_effector and not has_directly_reachable_goal:
            av_pos = engine.avatar_pos or (0, 0)
            sorted_switches = sorted(
                switch_objs,
                key=lambda s: (
                    trial_counts.get(s.id, 0),
                    abs(s.pos[0] - av_pos[0]) + abs(s.pos[1] - av_pos[1]),
                ),
            )
            reachable_switch = next((s for s in sorted_switches if _can_navigate_to(s.pos)), None)
            if (
                reachable_switch is None
                and sorted_switches
                and has_multi_agent
                and focus_switch is not None
            ):
                plan.append(
                    SwitchFocus(
                        switch_action=focus_switch,
                        description="Switch to partner agent to reach switch",
                    )
                )
                return plan
            if reachable_switch is not None:
                best_switch = reachable_switch
                if (
                    engine.has_displacement_capabilities(available_actions)
                    and engine.avatar_pos is not None
                ):
                    is_bar = (
                        best_switch.pos in engine.learned_barriers
                    ) or engine.symbolic_theory.is_barrier(best_switch.feature_id)
                    switch_rad = 1 if is_bar else 0
                    plan.append(
                        NavigateTo(
                            destination=best_switch.pos,
                            radius=switch_rad,
                            description=f"Navigate to switch at {best_switch.pos}",
                        )
                    )
                plan.append(
                    InteractWith(
                        target_coord=best_switch.pos,
                        description=f"Interact with switch at {best_switch.pos}",
                    )
                )

                if goals:
                    plan.append(
                        NavigateTo(
                            destination=goals[0], description="Reach goal unblocked by switch"
                        )
                    )

                trial_counts[best_switch.id] = trial_counts.get(best_switch.id, 0) + 1
                return plan

        # 4. TERMINAL GOAL REACHABILITY:
        if goals and engine.avatar_pos is not None:
            sorted_goals = sorted(
                goals,
                key=lambda g: abs(g[0] - engine.avatar_pos[0]) + abs(g[1] - engine.avatar_pos[1]),
            )
            reachable_goal = next((g for g in sorted_goals if _can_navigate_to(g)), None)
            if (
                reachable_goal is None
                and sorted_goals
                and has_multi_agent
                and focus_switch is not None
            ):
                plan.append(
                    SwitchFocus(
                        switch_action=focus_switch,
                        description="Switch to partner agent to reach goal",
                    )
                )
                return plan
            if reachable_goal is not None:
                target_goal = reachable_goal
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
                if "barrier" not in obj.role
                and "hazard" not in obj.role
                and "floor" not in obj.role
                and "background" not in obj.role
                and "passable" not in obj.role
                and "void" not in obj.role
                and "empty" not in obj.role
                and "avatar" not in obj.role
                and obj.pos != av_pos
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

                if any(engine.is_spatial_effector(a) for a in available_actions):
                    if (
                        engine.has_displacement_capabilities(available_actions)
                        and engine.avatar_pos is not None
                    ):
                        is_bar = (
                            best_obj.pos in engine.learned_barriers
                        ) or engine.symbolic_theory.is_barrier(best_obj.feature_id)
                        obj_rad = 1 if is_bar else 0
                        plan.append(
                            NavigateTo(
                                destination=best_obj.pos,
                                radius=obj_rad,
                                description=f"Navigate to object {best_obj.id} at {best_obj.pos}",
                            )
                        )
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

    @classmethod
    def synthesize_stage_result(
        cls,
        ctx: CognitiveContext | Any,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
        objects: Sequence[Any] | None = None,
        micro_actions: Sequence[Any] | None = None,
    ) -> StageResult:
        """Pure generation of StageResult from immutable CognitiveContext (WP1)."""
        ops: list[TransactionOperation] = []
        signals: dict[str, Any] = {}
        plan = cls.synthesize_plan(
            ctx,
            curr_grid,
            available_actions,
            objects=objects,
            micro_actions=micro_actions,
        )
        if plan is not None:
            signals["macro_plan"] = plan
            signals["has_macro_plan"] = True
            if not plan.is_empty():
                cur = plan.current()
                if cur:
                    signals["current_macro_action"] = cur.name
        else:
            signals["has_macro_plan"] = False

        return StageResult(
            author="macro_action_planner",
            operations=ops,
            signals=signals,
            proposed_action=None,
        )


class MacroActionStage:
    """Pure CorticalStage for macro-action planning.

    In accordance with Invariant 1 and WP1, this stage observes an immutable CognitiveContext
    and emits a pure StageResult containing proposed macro-plans and transactional intent.
    """

    name: str = "macro_action_planner"

    def bind_engine(self, engine: Any) -> None:
        self.engine = engine

    def observe(self, ctx: CognitiveContext) -> StageResult:
        grid = ctx.prev_grid if ctx.prev_grid is not None else np.zeros((1, 1), dtype=np.int32)
        actions = (
            list(ctx.action_dynamics.keys())
            or list(ctx.signals.get("available_actions", []))
            or [1, 2, 3, 4]
        )
        return MacroPlanSynthesizer.synthesize_stage_result(ctx, grid, actions)
