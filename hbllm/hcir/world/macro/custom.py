"""Custom Macro Actions, Procedural Replay, and Causal Credit Attribution."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.spatial_planner import SpatialActionIntent
from hbllm.hcir.world.engine_state import get_engine_state_attr
from hbllm.hcir.world.frontopolar_subgoal_stack import SubgoalType
from hbllm.hcir.world.macro.models import (
    MacroAction,
    MacroActionStatus,
    get_sub_attr,
)
from hbllm.hcir.world.macro.primitives import (
    DepositCargo,
    InteractWith,
    NavigateTo,
    PickUpCargo,
)

if TYPE_CHECKING:
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 8b. ProceduralReplayAction (Replay Consolidated Macro)
# ─────────────────────────────────────────────────────────────────────────────


class ProceduralReplayAction(MacroAction):
    """Replay a learned action sequence from a consolidated ProceduralMacro.

    This is the execution arm of the procedural consolidation pipeline (Phase A).
    When the deliberation engine finds an applicable macro from episodic memory,
    it wraps it in this macro action to replay the proven action sequence.

    The replay tracks progress through the sequence and records outcome back
    to the source macro for Bayesian confidence updating.
    """

    def __init__(
        self,
        macro: Any,  # ProceduralMacro (avoid circular import)
        description: str = "",
    ) -> None:
        super().__init__(
            name=f"ProceduralReplay({getattr(macro, 'macro_id', 'unknown')})",
            intent=SpatialActionIntent.NAVIGATE,
        )
        self._macro = macro
        get_seq_fn = get_sub_attr(macro, "get_execution_sequence", None)
        if callable(get_seq_fn):
            act_seq, data_seq = get_seq_fn()
            self._action_seq = list(act_seq)
            self._data_seq = list(data_seq)
        else:
            self._action_seq = list(getattr(macro, "action_sequence", []))
            self._data_seq = list(getattr(macro, "action_data_sequence", []))
        self._cursor: int = 0
        self.max_steps = len(self._action_seq) + 5  # Small buffer for timing
        self.is_strict = True  # Trust the proven sequence
        self.description = description or f"Replaying {len(self._action_seq)}-step macro"

    def is_satisfied(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
    ) -> bool:
        """Satisfied when we've dispatched all actions in the sequence."""
        return self._cursor >= len(self._action_seq)

    def is_feasible(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> bool:
        """Feasible if the next action is in the available action set."""
        if self._cursor >= len(self._action_seq):
            return False
        next_action = self._action_seq[self._cursor]
        return next_action in available_actions

    def step(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> tuple[Any, dict[str, Any] | None] | None:
        """Dispatch the next action from the learned sequence."""
        if self._cursor >= len(self._action_seq):
            return None
        action = self._action_seq[self._cursor]
        data = self._data_seq[self._cursor] if self._cursor < len(self._data_seq) else None
        self._cursor += 1
        self.consecutive_steps += 1

        if action not in available_actions:
            logger.warning(
                "ProceduralReplay: action %s not available at step %d, aborting",
                action,
                self._cursor,
            )
            self.status = MacroActionStatus.FAILED
            rec_fn = get_sub_attr(self._macro, "record_replay_outcome", None)
            if callable(rec_fn):
                rec_fn(False)
            return None

        return (action, data)

    def record_success(self) -> None:
        """Record successful replay completion back to the source macro."""
        rec_fn = get_sub_attr(self._macro, "record_replay_outcome", None)
        if callable(rec_fn):
            rec_fn(True)

    def record_failure(self) -> None:
        """Record replay failure back to the source macro."""
        rec_fn = get_sub_attr(self._macro, "record_replay_outcome", None)
        if callable(rec_fn):
            rec_fn(False)


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
    grid_hist = get_sub_attr(engine, "grid_history", None)
    if prev_grid is None and grid_hist is not None and len(grid_hist) >= 2:
        prev_grid = grid_hist[-2]

    if is_success:
        macro_action.status = MacroActionStatus.SATISFIED
        if isinstance(macro_action, PickUpCargo):
            feat = macro_action.cargo_feature
            if feat is None and prev_grid is not None:
                r, c = macro_action.cargo_pos
                if 0 <= r < prev_grid.shape[0] and 0 <= c < prev_grid.shape[1]:
                    feat = int(prev_grid[r, c])
            if feat is not None and feat != 0:
                step_cnt = get_engine_state_attr(engine, "step_count", 0)
                engine.working_memory.body_schema.acquire_tool(feat, step=step_cnt, role="key")
                engine.symbolic_theory.induce_cargo(feat, evidence="macro_acquire_tool")
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
                from hbllm.hcir.world.causal.transition_kb import TransitionOutcome

                engine.symbolic_theory.transition_kb.retract_outcome(
                    (r, c), TransitionOutcome.BLOCKED, scope="locus"
                )
                logger.info(
                    "Pearl do(a) causal attribution: tool %d cleared barrier %d at %s",
                    tool_feat,
                    barrier_feat,
                    macro_action.receptacle_pos,
                )
            elif cell_cleared:
                engine.learned_barriers.discard((r, c))
                from hbllm.hcir.world.causal.transition_kb import TransitionOutcome

                engine.symbolic_theory.transition_kb.retract_outcome(
                    (r, c), TransitionOutcome.BLOCKED, scope="locus"
                )

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
                attributor = getattr(engine, "remote_causal", None)
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
                        from hbllm.hcir.world.causal.transition_kb import TransitionOutcome

                        engine.symbolic_theory.transition_kb.retract_outcome(
                            (dr, dc), TransitionOutcome.BLOCKED, scope="locus"
                        )
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
                        engine.symbolic_theory.induce_candidate_goal(
                            dest_feat, evidence="macro_navigate_to"
                        )
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
            dest = macro_action.destination
            is_goal_dest = (
                dest in engine.learned_goal_positions or getattr(engine, "goal_pos", None) == dest
            )
            if not is_goal_dest:
                engine.learned_barriers.add(dest)
                if prev_grid is not None:
                    r, c = dest
                    if 0 <= r < prev_grid.shape[0] and 0 <= c < prev_grid.shape[1]:
                        feat = int(prev_grid[r, c])
                        bg = engine.estimate_background(prev_grid)
                        if (
                            feat != bg
                            and feat not in engine.symbolic_theory.goal_features
                            and feat not in engine.symbolic_theory.candidate_goal_features
                        ):
                            engine.symbolic_theory.induce_barrier(
                                feat, evidence="macro_navigate_failure"
                            )
            if engine.avatar_pos is not None and action_used is not None:
                engine.failed_transitions.add((engine.avatar_pos, action_used))
