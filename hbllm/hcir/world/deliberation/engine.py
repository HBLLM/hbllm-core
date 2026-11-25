"""Prefrontal Cortex Deliberation Engine & Action Proposal."""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.cortex_assimilator import EpistemicPhase
from hbllm.hcir.world.cortex_planner import EpistemicGoalScheduler, MentalSimulationStep
from hbllm.hcir.world.cortical_stage import WorkingMemoryDelta
from hbllm.hcir.world.deliberation.basal_ganglia import (
    BasalGangliaDecision,
    BasalGangliaGatingCircuit,
)
from hbllm.hcir.world.deliberation.neuromodulation import (
    AdaptiveNeuromodulationGovernor,
    NeuromodulatoryState,
)
from hbllm.hcir.world.deliberation.safety_predicates import (
    PlanSafetyEvaluator,
)
from hbllm.hcir.world.engine_state import get_engine_state_attr

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class PrefrontalDeliberationEngine:
    """Executive prefrontal arbitration view coordinating perception, mental simulation,
    affordance chunking, and motor output.
    """

    def __init__(
        self,
        safety_evaluator: PlanSafetyEvaluator | None = None,
        basal_ganglia: BasalGangliaGatingCircuit | None = None,
        neuromodulation: AdaptiveNeuromodulationGovernor | None = None,
    ) -> None:
        self.safety_evaluator = safety_evaluator or PlanSafetyEvaluator()
        self.basal_ganglia = basal_ganglia or BasalGangliaGatingCircuit()
        self.neuromodulation = neuromodulation or AdaptiveNeuromodulationGovernor()

    def arbitrate_basal_ganglia_action(
        self,
        view: Any,
        candidate_utilities: dict[Any, float],
        acc_conflict: float = 0.0,
        threat_looming: float = 0.0,
        surprise_entropy: float = 0.0,
        risk_penalties: dict[Any, float] | None = None,
    ) -> tuple[BasalGangliaDecision | None, list[BasalGangliaDecision]]:
        """Arbitrate candidate actions using the Basal Ganglia tri-pathway circuit."""
        dopamine = 1.0

        habenular_dict: dict[Any, float] = {}
        for act, cnt in view.inhibited_actions.items():
            if cnt > 0:
                habenular_dict[act] = min(1.0, 0.2 * cnt)

        return self.basal_ganglia.arbitrate_candidates(
            candidate_utilities=candidate_utilities,
            dopamine_level=dopamine,
            acc_conflict=acc_conflict,
            threat_looming=threat_looming,
            surprise_entropy=surprise_entropy,
            habenular_inhibitions=habenular_dict,
            risk_penalties=risk_penalties,
        )

    def step_neuromodulatory_dynamics(
        self,
        view: Any,
        reward: float = 0.0,
        predicted_value: float = 0.0,
        actual_value: float = 0.0,
        conflict_level: float = 0.0,
    ) -> NeuromodulatoryState:
        """Update VTA-LC dual neuromodulatory state using view tracking context."""
        stuck = get_engine_state_attr(view, "consecutive_stuck_steps", 0)
        return self.neuromodulation.step_dynamics(
            reward=reward,
            predicted_value=predicted_value,
            actual_value=actual_value,
            conflict_level=conflict_level,
            stuck_steps=stuck,
        )

    def validate_plan_safety(
        self,
        view: Any,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
        next_step: MentalSimulationStep,
    ) -> tuple[bool, bool, int]:
        """Validate safety of next plan step with multi-faculty reactive checks:
        A) Static lethal entities along planned path
        A2) Spatiotemporal looming collision cones
        B) Dynamic patrol trajectory forward simulation
        C) Habenular episodic action inhibition
        D) Cerebellar phase gating

        Returns:
            (plan_safe, should_wait, wait_steps)
        """
        res = self.safety_evaluator.evaluate(
            view=view,
            curr_grid=curr_grid,
            next_step=next_step,
            available_actions=available_actions,
        )
        return res.is_safe, res.should_wait, res.wait_steps

    def arbitrate_affordances(
        self,
        view: Any,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
        known_av_feats: set[int],
    ) -> tuple[Any, dict[str, Any] | None] | None:
        """Arbitrate interactive control clusters and target affordances via policies or fallbacks."""
        # 1. Query external registered deliberation policies first
        for policy in get_engine_state_attr(view, "deliberation_policies", []):
            arb_fn = getattr(policy, "arbitrate_affordances", None)
            if callable(arb_fn):
                res = arb_fn(view, curr_grid, available_actions, known_av_feats)
                if res is not None:
                    return res

        # 2. Domain-agnostic fallback for discrete transform / focus switch actions
        spatial_effector_actions = [a for a in available_actions if view.is_spatial_effector(a)]
        discrete_transform_actions = [
            a
            for a in available_actions
            if not view.is_displacement_action(a) and not view.is_spatial_effector(a)
        ]
        focus_switches = [
            a
            for a in discrete_transform_actions
            if view.is_focus_switch_action(a) and view.inhibited_actions.get(a, 0) == 0
        ]
        if focus_switches:
            view.inhibited_actions[focus_switches[0]] = max(
                view.inhibited_actions.get(focus_switches[0], 0), 8
            )
            view.state.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
            return focus_switches[0], None

        # Faculty: Dial / State-Cycler Interleaved Deliberation (Norman Parameter Space Navigation)
        index_selectors = [
            a
            for a in available_actions
            if a in view.index_selector_actions and view.inhibited_actions.get(a, 0) == 0
        ]
        value_mutators = [
            a
            for a in available_actions
            if a in view.value_mutator_actions and view.inhibited_actions.get(a, 0) == 0
        ]
        if index_selectors and value_mutators:
            mut_streak = view.working_memory.dial_mutation_streak
            if mut_streak < 2:
                chosen = value_mutators[view.step_counter % len(value_mutators)]
                view.working_memory.dial_mutation_streak = mut_streak + 1
                view.state.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
                return chosen, None
            else:
                chosen = index_selectors[view.step_counter % len(index_selectors)]
                view.working_memory.dial_mutation_streak = 0
                view.state.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
                return chosen, None

        # Modal Select-Then-Direct Gating: In pure effector domains without displacement, reach for interactive entities
        if (
            not view.has_displacement_capabilities(available_actions)
            and view.avatar_pos is None
            and not view.working_memory.body_schema.has_manipulandum()
            and spatial_effector_actions
        ):
            bg = view.estimate_background(curr_grid)
            entities = view.extract_entities(curr_grid, bg)
            has_interactive = bool(
                entities
                or get_engine_state_attr(view, "learned_goal_positions", None)
                or get_engine_state_attr(view, "target_scorers", None)
            )
            if has_interactive:
                chosen_eff = spatial_effector_actions[0]
                eff_data = view.ground_effector_action(curr_grid, chosen_eff)
                if eff_data is not None:
                    view.state.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
                    return chosen_eff, eff_data

        is_stuck_or_looping = (
            view.consecutive_stuck_steps >= 2
            or (view.avatar_pos is not None and view.recent_positions.count(view.avatar_pos) >= 2)
            or (view.avatar_pos is None and view.step_counter > 4)
        )
        has_displacement_actions = any(
            view.is_displacement_action(a) for a in available_actions
        ) or any(
            not view.is_spatial_effector(a) and a not in view.action_dynamics
            for a in available_actions
        )

        if not has_displacement_actions or is_stuck_or_looping:
            if spatial_effector_actions:
                # When spatial effectors are present without an interactive control target,
                # arbitrate_affordances must return None instead of blind discrete execution
                return None

            undo_acts = get_engine_state_attr(view, "undo_actions", set())
            uninhibited_discrete = [
                a
                for a in discrete_transform_actions
                if view.inhibited_actions.get(a, 0) == 0
                and a not in undo_acts
                and (
                    view.avatar_pos is None
                    or (a, view.avatar_pos) not in view.tested_action_positions
                )
            ]
            # Only fire discrete fallback when genuinely stuck/oscillating (>=3 visits or
            # >=2 blocked steps) or when no displacement actions exist; single backtrack (count==2)
            # defers to displacement exploration/simulation.
            truly_stuck = (
                not has_displacement_actions
                or view.consecutive_stuck_steps >= 2
                or (
                    view.avatar_pos is not None
                    and view.recent_positions.count(view.avatar_pos) >= 3
                )
            )
            if uninhibited_discrete and truly_stuck:
                # WP0.3: prefer the action least dominated by no_op/blocked
                # samples and never repeat a null-effect action >2x in a row
                # (unless every action is null-dominated).
                def _null_score(a: Any) -> tuple[int, float, int]:
                    post_fn = getattr(view, "motor_posterior", None)
                    post = post_fn(a) if callable(post_fn) else {}
                    null_p = post.get("no_op", 0.0) + post.get("blocked", 0.0)
                    consec_fn = getattr(view, "consecutive_null_count", None)
                    consec = consec_fn(a) if callable(consec_fn) else 0
                    return (consec >= 2, null_p, a)

                viable = [a for a in uninhibited_discrete if not _null_score(a)[0]]
                pool = viable or uninhibited_discrete
                chosen_action = min(pool, key=_null_score)
                if view.avatar_pos is not None:
                    view.tested_action_positions.add((chosen_action, view.avatar_pos))
                view.inhibited_actions[chosen_action] = max(
                    view.inhibited_actions.get(chosen_action, 0), 4
                )
                view.state.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
                return chosen_action, None

        return None

    def _execute_temporal_pacing(
        self,
        view: Any,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
        next_step: MentalSimulationStep,
        wait_steps: int,
    ) -> tuple[Any, dict[str, Any] | None, tuple[int, int] | None]:
        """Cerebellar / Premotor oscillatory temporal pacing:
        In environments with strictly directional actions (no explicit wait),
        synthesize an oscillatory burn step (e.g. step back / lateral dodge -> return)
        to burn phase steps until dynamic hazards clear.
        """
        view.consecutive_pacing_steps += 1
        if view.consecutive_pacing_steps > 2:
            view.state.consecutive_pacing_steps = 0
            view.mental_plan.clear()
            view.transition_to_phase(
                EpistemicPhase.REPLANNING,
                reason="pacing_window_exhausted",
            )
            act, data = view.plan_epistemic_probe(curr_grid, available_actions)
            return act, data, None

        view.mental_plan.appendleft(next_step)
        view.state.pending_phase_wait_steps = wait_steps
        undo_acts = get_engine_state_attr(view, "undo_actions", set())
        non_disp = [
            a
            for a in available_actions
            if not view.is_displacement_action(a)
            and not view.is_spatial_effector(a)
            and a not in undo_acts
        ]
        curr_pos = view.avatar_pos or get_engine_state_attr(view, "current_actor_pos", None)
        if non_disp:
            return non_disp[0], None, curr_pos

        H, W = curr_grid.shape
        reverse_map: dict[Any, Any] = {}
        for a1, d1 in view.action_deltas.items():
            for a2, d2 in view.action_deltas.items():
                if d1[0] == -d2[0] and d1[1] == -d2[1] and (d1[0] != 0 or d1[1] != 0):
                    reverse_map[a1] = a2
        if not reverse_map:
            reverse_map = {1: 2, 2: 1, 3: 4, 4: 3}
        pacing_candidates: list[tuple[Any, tuple[int, int]]] = []

        if curr_pos is not None:
            bg = view.estimate_background(curr_grid)
            walkable_and_safe = view.symbolic_theory.walkable_features
            bg_is_impassable_void = len(walkable_and_safe) > 0 and bg not in walkable_and_safe
            if not bg_is_impassable_void:
                bg_count = int(np.count_nonzero(curr_grid == bg))
                bg_ratio = bg_count / max(1, H * W)
                ar, ac = curr_pos
                avatar_on_bg = (
                    0 <= ar < H
                    and 0 <= ac < W
                    and (
                        int(curr_grid[ar, ac]) == bg
                        or any(
                            0 <= ar + dr_a < H
                            and 0 <= ac + dc_a < W
                            and int(curr_grid[ar + dr_a, ac + dc_a]) == bg
                            for dr_a, dc_a in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                        )
                    )
                )
                if bg_ratio >= 0.20 and not avatar_on_bg:
                    bg_is_impassable_void = True

            action_deltas = view.action_deltas
            for act in available_actions:
                if not view.is_displacement_action(act) or act == next_step.action:
                    continue
                delta = action_deltas.get(act)
                if delta is None:
                    continue
                dr, dc = delta
                nr, nc = curr_pos[0] + dr, curr_pos[1] + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if (nr, nc) in view.current_context_hazards:
                    continue
                val = int(curr_grid[nr, nc])
                ev = view.symbolic_theory.evaluate_transition(val, locus=(nr, nc))
                locus_b = view.symbolic_theory.transition_kb.get_belief((nr, nc), scope="locus")
                in_learned_b = (nr, nc) in view.learned_barriers and not (
                    locus_b is not None and ev.traversable and ev.confidence > 0.0
                )
                if (
                    in_learned_b
                    or not ev.traversable
                    or ev.risk_cost >= 40.0
                    or val in view.hazard_tracker.known_lethal_features
                    or (bg_is_impassable_void and val == bg)
                ):
                    continue
                cand_step = MentalSimulationStep(
                    action=act,
                    predicted_avatar_pos=(nr, nc),
                    expected_mutation="oscillatory_pacing_step",
                )
                cand_res = self.safety_evaluator.evaluate(
                    view=view,
                    curr_grid=curr_grid,
                    next_step=cand_step,
                    available_actions=available_actions,
                    future_steps=[cand_step],
                )
                if cand_res.is_safe and not cand_res.should_wait:
                    pacing_candidates.append((act, (nr, nc)))

        if pacing_candidates:
            # Prefer reverse of recent action to oscillate along existing safe path
            chosen_pacing_act, chosen_pacing_pos = pacing_candidates[0]
            if len(view.recent_actions) > 0:
                last_act = view.recent_actions[-1]
                rev_last = reverse_map.get(last_act)
                for act, pos in pacing_candidates:
                    if act == rev_last:
                        chosen_pacing_act, chosen_pacing_pos = act, pos
                        break

            rev_act = reverse_map.get(chosen_pacing_act)
            # Enqueue return step to restore spatial trajectory once hazard passes
            if rev_act is not None and rev_act in available_actions and curr_pos is not None:
                view.mental_plan.appendleft(
                    MentalSimulationStep(
                        action=rev_act,
                        predicted_avatar_pos=curr_pos,
                        expected_mutation="oscillatory_return_step",
                    )
                )
            return chosen_pacing_act, None, chosen_pacing_pos

        # Invariant fallback: Invalidate plan and replan if no safe pacing candidate exists
        view.mental_plan.clear()
        view.state.phase = EpistemicPhase.REPLANNING
        act, data = view.plan_epistemic_probe(curr_grid, available_actions)
        return act, data, None

    @staticmethod
    def has_confirmed_goal(view: Any) -> bool:
        """Returns True if the engine possesses verified causal or positional evidence of a goal."""
        theory = getattr(view, "symbolic_theory", None)
        has_theory_goal = bool(theory and theory.goal_features)
        has_pos_goal = bool(getattr(view, "learned_goal_positions", None))
        return has_theory_goal or has_pos_goal

    @classmethod
    def _target_phase(cls, view: Any) -> EpistemicPhase:
        """Determine target cognitive phase: exploitation requires verified reward evidence."""
        return (
            EpistemicPhase.EXPLOITATION
            if cls.has_confirmed_goal(view)
            else EpistemicPhase.MENTAL_SIMULATION
        )

    def prepare(
        self,
        view: Any,
        curr_grid: Any,
        available_actions: Sequence[Any],
        terminal: Any | None = None,
        is_success: bool = False,
        is_failure: bool = False,
        action_schemas: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """Pre-deliberation preamble: terminal resolution, step counter,
        normalization, bg estimate, and sensory assimilation (WP1)."""
        from hbllm.hcir.world.percept import TerminalSignal

        if terminal is None:
            if is_success:
                terminal = TerminalSignal.SUCCESS
            elif is_failure:
                terminal = TerminalSignal.FAILURE

        view.step_counter += 1
        if action_schemas:
            view.register_action_space(action_schemas)

        curr_grid = view.normalize_sensory_input(curr_grid)

        # Metacognitive Urgency Modulation:
        if view.is_under_strict_step_budget():
            view.apply_wm_delta(WorkingMemoryDelta("exploration_cooldown", 0))
            view.apply_wm_delta(WorkingMemoryDelta("simulation_cooldown", 0))

        # 1. Update background & assimilate sensory feedback from previous action
        view.state.bg_feature = view.estimate_background(curr_grid)
        if view.prev_grid is not None and not view.state.feedback_assimilated:
            view.assimilate_feedback(
                curr_grid,
                available_actions,
                terminal=terminal,
                is_success=is_success,
                is_failure=is_failure,
                action=view.last_action,
                action_data=view.last_action_data,
                **kwargs,
            )
        view.state.feedback_assimilated = False
        return curr_grid, terminal

    def decide(
        self,
        view: Any,
        curr_grid: Any,
        available_actions: Sequence[Any],
        terminal: Any | None = None,
        is_success: bool = False,
        is_failure: bool = False,
        action_schemas: Sequence[Any] | None = None,
        objects: Sequence[Any] | None = None,
        micro_actions: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any] | None]:
        """Unified cognitive decision (prepare + deliberate; kept for direct callers)."""
        g, term = self.prepare(
            view,
            curr_grid,
            available_actions,
            terminal=terminal,
            is_success=is_success,
            is_failure=is_failure,
            action_schemas=action_schemas,
            **kwargs,
        )
        proposal = self.deliberate(
            view,
            g,
            available_actions,
            terminal=term,
            is_success=is_success,
            is_failure=is_failure,
            action_schemas=action_schemas,
            objects=objects,
            micro_actions=micro_actions,
            **kwargs,
        )
        if isinstance(proposal, ActionProposal):
            if proposal.finalized:
                return proposal.action, proposal.action_data
            return self._finalize_step(
                view,
                proposal.grid,
                proposal.action,
                proposal.action_data,
                predicted_pos=proposal.predicted_pos,
            )
        return proposal

    def deliberate(
        self,
        view: Any,
        curr_grid: Any,
        available_actions: Sequence[Any],
        terminal: Any | None = None,
        is_success: bool = False,
        is_failure: bool = False,
        action_schemas: Sequence[Any] | None = None,
        objects: Sequence[Any] | None = None,
        micro_actions: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> ActionProposal | tuple[Any, dict[str, Any] | None]:
        """Post-assimilation deliberation body."""
        # When confirmed goals are known, exploit them directly
        if self.has_confirmed_goal(view):
            view.state.phase = EpistemicPhase.EXPLOITATION

        # VTA-LC Coupled Neuromodulatory Dynamics: Step dopamine & noradrenaline dynamics
        try:
            rew = getattr(view, "last_reward", 0.0) or 0.0
            acc_mon = view.acc_conflict_monitor
            acc_conf = float(acc_mon.current_conflict or 0.0) if acc_mon is not None else 0.0
            self.step_neuromodulatory_dynamics(
                view,
                reward=rew,
                predicted_value=0.0,
                actual_value=rew,
                conflict_level=acc_conf,
            )
        except Exception as exc:
            logger.debug("Neuromodulatory dynamics step skipped: %s", exc)

        # Track recent avatar position on every decision step (before macro or micro action planning)
        if view.avatar_pos is not None:
            view.recent_positions.append(view.avatar_pos)

        # Metacognitive Fail-Fast & Entropy Monitoring (Phase I)
        if getattr(view, "entropy_monitor", None):
            cur_h = float("inf")
            if view.avatar_pos is not None and view.learned_goal_positions:
                cur_h = min(
                    abs(view.avatar_pos[0] - g[0]) + abs(view.avatar_pos[1] - g[1])
                    for g in view.learned_goal_positions
                )
            if view.entropy_monitor.update(view.avatar_pos, cur_h):
                view.entropy_monitor.trigger_fail_fast(view)

        # Faculty: Executive Macro-Action Plan Execution & Synthesis
        def _is_macro_step_safe(plan: Any, act: Any, data: dict[str, Any] | None) -> bool:
            # Executive prefrontal macro plans supersede subcortical motor refractory inhibition (inhibited_actions)
            # and exploratory sensory foraging IOR (habenular_ior).
            # If the macro plan is marked is_strict, trust executive dispatch completely (e.g. verified evasion/solution sequence).
            if getattr(plan, "is_strict", False):
                return True
            cur_op = getattr(plan, "current", lambda: None)()
            if cur_op is not None and getattr(cur_op, "is_strict", False):
                return True

            # Safety gating aborts if displacement moves into collision cone, solid obstacle, or ping-pongs in limit cycle.
            if view.avatar_pos is not None and act in view.action_dynamics:
                dyn = view.action_dynamics[act]
                if dyn.is_displacement_action():
                    dr, dc = dyn.delta_r, dyn.delta_c
                    nr, nc = view.avatar_pos[0] + dr, view.avatar_pos[1] + dc
                    if isinstance(curr_grid, np.ndarray) and curr_grid.ndim >= 2:
                        H, W = curr_grid.shape[0], curr_grid.shape[1]
                        if not (0 <= nr < H and 0 <= nc < W):
                            return False
                        cell_v = int(curr_grid[nr, nc])
                        bg_v = getattr(view, "bg_feature", 0)
                        theory = getattr(view, "symbolic_theory", None)
                        if theory is not None:
                            ev = theory.evaluate_transition(cell_v, locus=(nr, nc))
                            locus_b = (
                                theory.transition_kb.get_belief((nr, nc), scope="locus")
                                if theory.transition_kb is not None
                                else None
                            )
                            in_learned_b = (nr, nc) in getattr(
                                view, "learned_barriers", set()
                            ) and not (
                                locus_b is not None and ev.traversable and ev.confidence > 0.0
                            )
                            if in_learned_b or not ev.traversable or ev.risk_cost >= 40.0:
                                return False
                            if ev.confidence == 0.0 and cell_v != bg_v:
                                return False
                    # Abort if displacement moves into limit cycle or oscillation
                    rec = list(get_engine_state_attr(view, "recent_positions", []))
                    if rec:
                        # Immediate 2-step reversal
                        if len(rec) >= 2 and (nr, nc) == rec[-2]:
                            return False
                        # 4-step limit cycle
                        if len(rec) >= 4 and (nr, nc) == rec[-4]:
                            return False
                        # Repeated local trapping in small window
                        if len(rec) >= 6 and rec[-6:].count((nr, nc)) >= 2:
                            return False
            return True

        macro_plan = view.working_memory.macro_plan
        if macro_plan is not None and not macro_plan.is_empty():
            macro_step = macro_plan.step(
                view, curr_grid, available_actions, prev_grid=view.prev_grid
            )
            if macro_step is not None:
                m_act, m_data = macro_step
                if m_act in available_actions:
                    if _is_macro_step_safe(macro_plan, m_act, m_data):
                        return self._build_proposal(
                            view, m_act, m_data, branch="macro_option", grid=curr_grid
                        )
                    else:
                        macro_plan.clear()

        from hbllm.hcir.world.macro_actions import (
            MacroActionPlan,
            MacroPlanSynthesizer,
            ProceduralReplayAction,
        )

        # ── Procedural Memory Recall ───────────────────────────────────────
        # Before synthesizing a new plan from scratch, check if episodic memory
        # has a consolidated procedural macro applicable to the current observation.
        # When motor dynamics and causal goal/cargo ontology are grounded, prefer
        # model-based shortest-path planning over replaying noisy exploration tapes.
        _has_grounded_goal_model = view.is_motor_grounded() and bool(
            view.learned_goal_positions
            or (
                view.symbolic_theory is not None
                and (view.symbolic_theory.goal_features or view.symbolic_theory.cargo_features)
            )
        )
        if (
            not _has_grounded_goal_model
            and (macro_plan is None or macro_plan.is_empty())
            and not view.mental_plan
            and view.episodic_cortex.consolidated_macros
        ):
            curr_entities = view.extract_entities(curr_grid, view.bg_feature)
            applicable_macros = view.episodic_cortex.query_applicable_macros(
                grid=curr_grid,
                available_actions=set(available_actions),
                top_k=1,
                min_confidence=0.5,
                entities=curr_entities,
                min_match_score=0.9,
            )
            if applicable_macros:
                best_macro = applicable_macros[0]
                logger.info(
                    "Procedural Recall: Replaying consolidated macro '%s' "
                    "(%d actions, confidence=%.2f)",
                    best_macro.macro_id,
                    len(best_macro.action_sequence),
                    best_macro.confidence,
                )
                replay_action = ProceduralReplayAction(macro=best_macro)
                from hbllm.hcir.world.macro_actions import MacroActionPlan

                replay_plan = MacroActionPlan(
                    actions=deque([replay_action]),
                    plan_id=f"procedural_replay_{best_macro.macro_id}",
                    is_strict=True,
                )
                view.working_memory.macro_plan = replay_plan
                first_step = replay_plan.step(
                    view, curr_grid, available_actions, prev_grid=view.prev_grid
                )
                if first_step is not None:
                    m_act, m_data = first_step
                    if m_act in available_actions:
                        return self._build_proposal(
                            view, m_act, m_data, branch="macro_recall", grid=curr_grid
                        )

        # ── Relational Subgoal Macro Recall (Phase E) ──────────────────────
        # When exact concrete macro does not match, query coordinate-invariant
        # relational macros and plan dynamically through intermediate waypoints.
        if (
            (macro_plan is None or macro_plan.is_empty())
            and not view.mental_plan
            and view.episodic_cortex.relational_macros
            and view.avatar_pos is not None
            and view.is_motor_grounded()
        ):
            applicable_rel = view.episodic_cortex.query_applicable_relational_macros(
                grid=curr_grid,
                available_actions=set(available_actions),
                top_k=1,
            )
            if applicable_rel:
                rel_macro = applicable_rel[0]
                from hbllm.hcir.world.autonomous_learning import CausalTheoryPlanner

                ctx = CausalTheoryPlanner.extract_planning_context(view, curr_grid)
                rel_path = CausalTheoryPlanner.plan_relational_macro(
                    view.avatar_pos,
                    curr_grid,
                    ctx,
                    rel_macro,
                    visited_cells=view.working_memory.visited_cells,
                    working_memory=view.working_memory,
                    bg=view.bg_feature,
                )
                if rel_path:
                    from hbllm.hcir.world.cortex_planner import MentalSimulationStep

                    rel_plan: list[MentalSimulationStep] = []
                    sim_pos = view.avatar_pos
                    for act_id in rel_path:
                        dr, dc = ctx.action_displacements.get(act_id, (0, 0))
                        next_pos = (sim_pos[0] + dr, sim_pos[1] + dc)
                        rel_plan.append(
                            MentalSimulationStep(
                                action=act_id,
                                predicted_avatar_pos=next_pos,
                            )
                        )
                        sim_pos = next_pos
                    view.state.phase = self._target_phase(view)
                    view.apply_wm_delta(WorkingMemoryDelta("mental_plan", deque(rel_plan)))
                    next_step = view.mental_plan.popleft()
                    logger.info(
                        "Relational Macro Recall: Dynamically planned %d-step path for '%s'",
                        len(rel_path),
                        rel_macro.macro_id,
                    )
                    return self._build_proposal(
                        view,
                        next_step.action,
                        next_step.action_data,
                        branch="relational_macro",
                        grid=curr_grid,
                    )

        has_uncalibrated_cardinal = False
        if (
            curr_grid.ndim >= 2
            and curr_grid.shape[0] > 1
            and curr_grid.shape[1] > 1
            and view.avatar_pos is not None
        ):
            cal_cardinal_dirs = {
                (
                    (1 if d.delta_r > 0 else (-1 if d.delta_r < 0 else 0)),
                    (1 if d.delta_c > 0 else (-1 if d.delta_c < 0 else 0)),
                )
                for d in view.action_dynamics.values()
                if d.is_displacement_action()
            }
            cal_cardinal_dirs.discard((0, 0))
            if 0 < len(cal_cardinal_dirs) < 4 and any(
                not view.is_spatial_effector(a)
                and not view.is_action_calibrated(a)
                and a not in view.immobile_entity_actions
                and view.inhibited_actions.get(a, 0) == 0
                and (a, view.avatar_pos) not in view.tested_action_positions
                and a not in view.undo_actions
                for a in available_actions
            ):
                ar_u, ac_u = view.avatar_pos
                H_u, W_u = curr_grid.shape[0], curr_grid.shape[1]
                stride_u = max(1, int(getattr(view.state, "spatial_lattice_stride", 1) or 1))
                bg_u = (
                    view.bg_feature
                    if view.bg_feature is not None
                    else view.estimate_background(curr_grid)
                )

                def _is_open_cardinal(nr_c: int, nc_c: int) -> bool:
                    v_u = int(curr_grid[nr_c, nc_c])
                    if v_u == bg_u:
                        return True
                    ev_u = view.symbolic_theory.evaluate_transition(v_u, locus=(nr_c, nc_c))
                    return bool(ev_u.traversable and ev_u.confidence > 0.0)

                has_uncalibrated_cardinal = any(
                    0 <= ar_u + dr_u * stride_u < H_u
                    and 0 <= ac_u + dc_u * stride_u < W_u
                    and _is_open_cardinal(ar_u + dr_u * stride_u, ac_u + dc_u * stride_u)
                    for dr_u, dc_u in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    if (dr_u, dc_u) not in cal_cardinal_dirs
                )

        can_synthesize = (
            view.is_motor_grounded() and not has_uncalibrated_cardinal
        ) or not view.has_displacement_capabilities(available_actions)

        if (
            (objects or micro_actions)
            and (macro_plan is None or macro_plan.is_empty())
            and not view.mental_plan
            and can_synthesize
        ):
            synthesized = MacroPlanSynthesizer.synthesize_plan(
                view,
                curr_grid,
                available_actions,
                objects=objects,
                micro_actions=micro_actions,
            )
            if synthesized is not None and not synthesized.is_empty():
                view.working_memory.macro_plan = synthesized
                macro_step = synthesized.step(
                    view, curr_grid, available_actions, prev_grid=view.prev_grid
                )
                if macro_step is not None:
                    m_act, m_data = macro_step
                    if m_act in available_actions:
                        if _is_macro_step_safe(synthesized, m_act, m_data):
                            if (
                                view.phase == EpistemicPhase.MOTOR_GROUNDING
                                and view.is_motor_grounded()
                            ):
                                view.state.phase = self._target_phase(view)
                            pred_pos = getattr(
                                synthesized.current_action, "next_predicted_pos", None
                            )
                            if pred_pos is None and view.mental_plan:
                                pred_pos = view.mental_plan[0].predicted_avatar_pos
                            return self._build_proposal(
                                view,
                                m_act,
                                m_data,
                                branch="synthesized_schema",
                                predicted_pos=pred_pos,
                                grid=curr_grid,
                            )
                        else:
                            view.working_memory.macro_plan.clear()

        # Phasic Salience Reset: Environmental discovery, dynamic threat emergence, or high surprise
        # Query prefrontal working memory for salience shift to awaken deliberate forward search
        if view.working_memory.has_salience_shift(view):
            view.apply_wm_delta(WorkingMemoryDelta("simulation_cooldown", 0))
            view.apply_wm_delta(WorkingMemoryDelta("consecutive_simulation_failures", 0))
            view.mental_plan.clear()
            view.state.phase = (
                EpistemicPhase.MENTAL_SIMULATION
                if view.is_motor_grounded()
                else EpistemicPhase.EPISTEMIC_EXPLORATION
            )

        has_temporal = bool(view.hazard_tracker.periodic_cells) or bool(view.mobile_threat_features)

        # Metacognitive Refractory Inhibition: decay refractory timers
        in_simulation_cooldown = view.simulation_cooldown > 0
        if view.simulation_cooldown > 0:
            view.simulation_cooldown -= 1
        view.working_memory.decay_inhibited_actions()

        # Anterior Mid-Cingulate & Lateral Habenula: Limit Cycle / Oscillation Detection
        osc_acts = view.working_memory.habenular_ior.detect_cyclic_actions()
        if osc_acts:
            cyclic_positions = set(list(view.recent_positions)[-4:])
            if (
                len(cyclic_positions) <= 2 and len(view.recent_positions) >= 4
            ) or view.consecutive_stuck_steps >= 2:
                has_progress = False
            else:
                has_progress = (
                    view.phase in (EpistemicPhase.EXPLOITATION, EpistemicPhase.MENTAL_SIMULATION)
                    and bool(view.mental_plan)
                    and any(
                        s.predicted_avatar_pos is not None
                        and s.predicted_avatar_pos != view.avatar_pos
                        for s in view.mental_plan
                    )
                )
            if not has_progress:
                for osc_act in osc_acts:
                    view.inhibited_actions[osc_act] = min(
                        view.inhibited_actions.get(osc_act, 0) + 2, 3
                    )

        # Faculty: Anterior Cingulate Cortex (ACC) Limit Cycle & Conflict Suppression
        forced_acc_perturbation: Any | None = None
        eff_pos: Any = view.avatar_pos
        cont_pos = getattr(getattr(view, "working_memory", None), "continuous_agent_pos", None)
        if cont_pos is not None:
            eff_pos = (round(float(cont_pos[0]), 3), round(float(cont_pos[1]), 3))
        view.acc_conflict_monitor.record_step(view.last_action, eff_pos)
        is_osc, cyclic_acts = view.acc_conflict_monitor.detect_oscillation()
        if is_osc:
            # Purposeful Active Inference & Hesitation Invariant:
            # An action oscillation is NOT a pathological stereotypy if the organism is engaged
            # in intentional temporal pacing/evasion around dynamic threats, or if an active
            # mental plan exhibits future forward progress once the looming hazard clears.
            has_progress = (
                view.phase in (EpistemicPhase.EXPLOITATION, EpistemicPhase.MENTAL_SIMULATION)
                and bool(view.mental_plan)
                and any(
                    s.predicted_avatar_pos is not None and s.predicted_avatar_pos != view.avatar_pos
                    for s in view.mental_plan
                )
            ) or getattr(getattr(view, "working_memory", None), "last_continuous_moved", False)
            is_deliberate_pacing = has_temporal and view.consecutive_pacing_steps > 0
            if not has_progress and not is_deliberate_pacing:
                wm_ref = getattr(view, "working_memory", None)
                inhibit_fn = getattr(wm_ref, "inhibit_action", None) if wm_ref is not None else None
                for cyc_a in cyclic_acts:
                    if callable(inhibit_fn):
                        inhibit_fn(cyc_a, duration=2)
                    else:
                        view.inhibited_actions[cyc_a] = min(
                            view.inhibited_actions.get(cyc_a, 0) + 2, 3
                        )
                view.mental_plan.clear()
                from hbllm.hcir.world.cortex_motor import ACCConflictMonitor

                ortho_acts = ACCConflictMonitor.get_orthogonal_actions(
                    available_actions=available_actions,
                    cyclic_actions=cyclic_acts,
                    action_dynamics=view.action_dynamics,
                )
                H, W = curr_grid.shape
                unverified_fallback: Any | None = None
                bg_feat = getattr(view, "bg_feature", getattr(view, "default_bg_feature", 0))
                for cand_a in ortho_acts:
                    if view.inhibited_actions.get(cand_a, 0) > 0:
                        continue
                    if (view.avatar_pos, cand_a) in view.failed_transitions:
                        continue
                    dyn = view.action_dynamics.get(cand_a)
                    if dyn is not None and dyn.is_displacement_action():
                        dr, dc = dyn.get_displacement()
                        if view.avatar_pos is not None:
                            nr, nc = view.avatar_pos[0] + dr, view.avatar_pos[1] + dc
                            if not (0 <= nr < H and 0 <= nc < W):
                                continue
                            if (nr, nc) in view.hazard_tracker.static_lethal_positions:
                                continue
                            feat_at_dest = int(curr_grid[nr, nc])
                            ev = view.symbolic_theory.evaluate_transition(
                                feat_at_dest, locus=(nr, nc)
                            )
                            locus_b = view.symbolic_theory.transition_kb.get_belief(
                                (nr, nc), scope="locus"
                            )
                            in_learned_b = (nr, nc) in view.learned_barriers and not (
                                locus_b is not None and ev.traversable and ev.confidence > 0.0
                            )
                            if (
                                in_learned_b
                                or feat_at_dest in view.hazard_tracker.known_lethal_features
                                or not ev.traversable
                                or ev.risk_cost >= 40.0
                            ):
                                continue
                            is_known_safe = (
                                feat_at_dest == bg_feat
                                or (ev.traversable and ev.confidence > 0.0)
                                or ev.goal_value > 0.0
                                or feat_at_dest in (view.avatar_features or ())
                            )
                            if not is_known_safe:
                                if unverified_fallback is None:
                                    unverified_fallback = cand_a
                                continue
                            forced_acc_perturbation = cand_a
                            break
                    else:
                        forced_acc_perturbation = cand_a
                        break
                if forced_acc_perturbation is None and unverified_fallback is not None:
                    forced_acc_perturbation = unverified_fallback

        # Faculty: ACC Recurring Catastrophe Conflict Resolution (Metacognitive Executive Shift)
        acc_death_conflict = view.working_memory.acc_conflict
        if acc_death_conflict is not None and getattr(acc_death_conflict, "active_conflict", False):
            c_radius = float(getattr(acc_death_conflict, "cluster_radius", 3.0))
            # Invalidate any active plans routing into failure clusters
            if view.mental_plan:
                for cl in acc_death_conflict.clusters:
                    if cl.count >= 2:
                        for step_s in view.mental_plan:
                            if step_s.predicted_avatar_pos is not None:
                                pr, pc = step_s.predicted_avatar_pos
                                if math.hypot(pr - cl.center_r, pc - cl.center_c) <= c_radius:
                                    view.mental_plan.clear()
                                    view.state.current_simulated_goal = None
                                    break
                            if not view.mental_plan:
                                break

            # Mark death cluster cells as static lethal hazards to force topological detours
            for cl in acc_death_conflict.clusters:
                if cl.count >= 2:
                    cr_int, cc_int = int(round(cl.center_r)), int(round(cl.center_c))
                    rad_int = int(math.ceil(c_radius))
                    for dr_c in range(-rad_int, rad_int + 1):
                        for dc_c in range(-rad_int, rad_int + 1):
                            if math.hypot(dr_c, dc_c) <= c_radius:
                                lr, lc = cr_int + dr_c, cc_int + dc_c
                                if 0 <= lr < curr_grid.shape[0] and 0 <= lc < curr_grid.shape[1]:
                                    view.hazard_tracker.static_lethal_positions.add((lr, lc))
                                    cell_f = int(curr_grid[lr, lc])
                                    if cell_f in view.symbolic_theory.goal_features:
                                        from hbllm.hcir.world.cortex_causal import (
                                            invalidate_stale_context_ontology,
                                        )

                                        was_unverified_prior = (
                                            view.symbolic_theory.has_prior_context_transition
                                            and not view.symbolic_theory.is_verified_in_context(
                                                "goal", cell_f
                                            )
                                        )
                                        view.symbolic_theory.retract_goal(
                                            cell_f, reason="death_cluster"
                                        )
                                        view.symbolic_theory.retract_walkable(
                                            cell_f, reason="death_cluster"
                                        )
                                        view.symbolic_theory.retract_conditional_affordance(
                                            cell_f, reason="death_cluster"
                                        )
                                        view.symbolic_theory.induce_barrier(
                                            cell_f, evidence="death_cluster"
                                        )
                                        view.symbolic_theory.mark_verified_in_context(
                                            "lethal", cell_f
                                        )
                                        if (
                                            0 <= lr < curr_grid.shape[0]
                                            and 0 <= lc < curr_grid.shape[1]
                                        ):
                                            from hbllm.hcir.world.causal.transition_kb import (
                                                TransitionOutcome,
                                            )

                                            view.symbolic_theory.transition_kb.record_outcome(
                                                (lr, lc),
                                                TransitionOutcome.TERMINAL_FAIL,
                                                scope="locus",
                                                evidence="death_cluster",
                                            )
                                        view.hazard_tracker.known_lethal_features.add(cell_f)
                                        if was_unverified_prior:
                                            invalidate_stale_context_ontology(
                                                view.symbolic_theory,
                                                hazard_tracker=view.hazard_tracker,
                                                episodic_cortex=getattr(
                                                    view, "episodic_cortex", None
                                                ),
                                                working_memory=view.working_memory,
                                                engine_or_view=view,
                                                bg_feature=getattr(view, "bg_feature", 0),
                                                preserve_current_context=True,
                                                reason=f"acc_goal_to_lethal:{cell_f}",
                                            )
                                        logger.warning(
                                            "ACC Metacognitive Shift: Inverted hypothesis for feature %d (GOAL -> LETHAL) due to death cluster at (%d, %d)",
                                            cell_f,
                                            lr,
                                            lc,
                                        )

            # Invalidate deadlocked subgoals in FrontopolarSubgoalStack if they target conflict zones
            if view.subgoal_stack.has_pending_goals:
                cur_sg = view.subgoal_stack.current_subgoal
                if cur_sg is not None:
                    target = cur_sg.target_entity_pos or cur_sg.target_destination
                    if target and acc_death_conflict.is_in_conflict_zone(target):
                        logger.warning(
                            "ACC Metacognitive Gating: Aborting deadlocked subgoal %s targeting conflict zone %s",
                            cur_sg.subgoal_id,
                            target,
                        )
                        view.subgoal_stack.clear_subgoals()
                        view.transition_to_phase(
                            EpistemicPhase.REPLANNING,
                            reason="acc_subgoal_conflict_abort",
                        )

        # Visual avatar localization
        known_av_feats = set(view.avatar_features) if view.avatar_features else set()
        if view.avatar_feature is not None:
            known_av_feats.add(view.avatar_feature)
        elif view.prior_avatar_feature is not None:
            known_av_feats.add(view.prior_avatar_feature)
        if known_av_feats:
            view._update_avatar_position_from_grid(curr_grid, known_av_feats)
            if (
                view.avatar_pos is not None
                and view.avatar_feature is None
                and view.prior_avatar_feature is not None
            ):
                view.state.avatar_feature = view.prior_avatar_feature
                view.avatar_features.add(view.prior_avatar_feature)

        # Faculty: Active Manipulandum Localization
        if view.avatar_pos is None and view.working_memory.body_schema.has_manipulandum():
            manip = view.working_memory.body_schema.active_manipulandum
            if manip is not None and getattr(manip, "pos", None) is not None:
                view.state.avatar_pos = manip.pos

        # Observe creature motion
        view._observe_oriented_threat_motion(curr_grid, available_actions, known_av_feats)

        # Faculty: Epistemic & Perceptual Goal Scheduling (Stage 1 Active Inference)
        # Identify structural and candidate goals prior to forward mental simulation once avatar is recognized.
        if not view.learned_goal_positions and (
            view.avatar_pos is not None or bool(view.avatar_features)
        ):
            bg_val = (
                view.bg_feature
                if view.bg_feature is not None
                else view.estimate_background(curr_grid)
            )
            entities = view.extract_entities(curr_grid, bg_val)
            av_feats_cand = view.avatar_features or (
                {view.avatar_feature} if view.avatar_feature is not None else set()
            )
            proposal = EpistemicGoalScheduler.schedule_goals(
                engine=view,
                curr_grid=curr_grid,
                entities=entities,
                bg=bg_val,
                av_feats=av_feats_cand,
                target_role=EntityRole.GOAL,
                available_actions=available_actions,
            )
            for gp in proposal.new_learned_goal_positions:
                view.record_learned_goal(gp)
            for feat in proposal.new_candidate_features:
                if (
                    feat not in view.hazard_tracker.mobile_entities
                    or feat in view.symbolic_theory.cargo_features
                ):
                    view.symbolic_theory.induce_candidate_goal(
                        feat, evidence="deliberation:proposal"
                    )

        # Faculty E: Basal Ganglia Motor Gating & Phase Entrainment
        if view.pending_phase_wait_steps > 0:
            view.pending_phase_wait_steps -= 1
            undo_acts = get_engine_state_attr(view, "undo_actions", set())
            non_disp = [
                a
                for a in available_actions
                if not view.is_displacement_action(a)
                and not view.is_spatial_effector(a)
                and a not in undo_acts
            ]
            if non_disp:
                return self._build_proposal(
                    view, non_disp[0], None, branch="non_displacement_fallback", grid=curr_grid
                )

        # Faculty: Brodmann Area 10 (aPFC) Hierarchical Subgoal Stack & Cognitive Branching
        if view.subgoal_stack.has_pending_goals:
            cur_sg = view.subgoal_stack.current_subgoal
            if cur_sg is not None and view.subgoal_stack.is_subgoal_satisfied(
                cur_sg, set(), avatar_pos=view.avatar_pos
            ):
                view.subgoal_stack.pop_subgoal()

        chosen_action: Any
        chosen_data: dict[str, Any] | None = None
        predicted_pos: tuple[int, int] | None = None
        branch: str = "default_arbitration"

        # Resolve active in-flight plan step before arbitration
        active_step = getattr(view.working_memory, "active_plan_step", None)
        if active_step is not None:
            tgt = active_step.predicted_avatar_pos
            if tgt is None or tgt == (0, 0) or view.avatar_pos == tgt:
                view.working_memory.active_plan_step = None
                active_step = None
            elif (
                getattr(view.working_memory, "last_continuous_moved", False)
                and view.consecutive_stuck_steps == 0
            ):
                pass
            else:
                view.mental_plan.clear()
                view.working_memory.active_plan_step = None
                active_step = None

        # 0. ACC Forced Perturbation: break limit-cycle oscillation immediately
        if forced_acc_perturbation is not None:
            logger.info(
                "Anterior Cingulate Cortex (ACC): Breaking limit cycle oscillation with orthogonal perturbation action %s",
                forced_acc_perturbation,
            )
            view.state.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
            chosen_action = forced_acc_perturbation
            chosen_data = None
            branch = "acc_perturbation"

        # 1. Prefrontal Affordance Panel & Optical Symmetry Gating (exploration / discrete manipulation only)
        elif (
            not self.has_confirmed_goal(view)
            and view.phase not in (EpistemicPhase.EXPLOITATION, EpistemicPhase.MENTAL_SIMULATION)
            and (
                affordance_choice := self.arbitrate_affordances(
                    view, curr_grid, available_actions, known_av_feats
                )
            )
            is not None
        ):
            chosen_action, chosen_data = affordance_choice
            branch = "affordance_choice"

        # Continuous Metric Kinematic Servoing (Local Proportional Guidance into Goal Capture Basin)
        elif (
            not has_uncalibrated_cardinal
            and (c_goal := getattr(view.working_memory, "continuous_goal_pos", None)) is not None
            and (c_agent := getattr(view.working_memory, "continuous_agent_pos", None)) is not None
            and float(np.linalg.norm(c_goal - c_agent)) <= 1.5
            and any(
                (dyn := view.action_dynamics.get(a)) is not None and dyn.is_displacement_action()
                for a in available_actions
            )
        ):
            rf = getattr(view.working_memory, "reference_frame", None)
            origin_desc = (
                getattr(rf, "origin_description", "bottom_left_metric")
                if rf
                else "bottom_left_metric"
            )
            y_sign = 1 if "bottom_left" in origin_desc else -1
            dx = float(c_goal[0] - c_agent[0])
            dy = float(c_goal[1] - c_agent[1])

            best_act = None
            best_dot = 0.0
            for act in available_actions:
                dyn = view.action_dynamics.get(act)
                if dyn is not None and dyn.is_displacement_action():
                    if dyn.continuous_deltas:
                        vx, vy = dyn.get_continuous_velocity()
                    else:
                        vx = float(dyn.delta_c)
                        vy = float(-dyn.delta_r * y_sign)
                    dot = dx * vx + dy * vy
                    if dot > best_dot and view.inhibited_actions.get(act, 0) == 0:
                        best_dot = dot
                        best_act = act
            if best_act is not None:
                chosen_action = best_act
                chosen_data = None
                branch = "continuous_servoing"
            else:
                # Fall through to mental plan if servoing action is inhibited
                chosen_action = None

        # 2. Check active mental plan during EXPLOITATION or MENTAL_SIMULATION phase
        elif (
            not has_uncalibrated_cardinal
            and view.phase in (EpistemicPhase.EXPLOITATION, EpistemicPhase.MENTAL_SIMULATION)
            and (view.mental_plan or active_step is not None)
        ):
            # Active Inference & Efference Copy Divergence:
            # If the organism is physically stuck or the previous step produced zero movement,
            # the counterfactual mental rollout is falsified. Clear stale plan immediately and replan
            # via forward mental simulation over updated causal dynamics and Habenular IOR.
            if view.consecutive_stuck_steps > 0:
                view.mental_plan.clear()
                view.working_memory.active_plan_step = None
                view.state.phase = EpistemicPhase.REPLANNING
                simulated_plan = view.simulate_in_mind(curr_grid, available_actions)
                if simulated_plan:
                    view.apply_wm_delta(WorkingMemoryDelta("consecutive_simulation_failures", 0))
                    view.apply_wm_delta(WorkingMemoryDelta("simulation_cooldown", 0))
                    view.apply_wm_delta(WorkingMemoryDelta("phase", self._target_phase(view)))
                    view.apply_wm_delta(WorkingMemoryDelta("mental_plan", deque(simulated_plan)))
                    next_step = view.mental_plan.popleft()
                    view.working_memory.active_plan_step = next_step
                    plan_safe, should_wait, wait_steps = self.validate_plan_safety(
                        view, curr_grid, available_actions, next_step
                    )
                    if should_wait:
                        chosen_action, chosen_data, predicted_pos = self._execute_temporal_pacing(
                            view, curr_grid, available_actions, next_step, wait_steps
                        )
                        branch = "temporal_pacing"
                    elif plan_safe and not view.working_memory.habenular_ior.is_action_inhibited(
                        view.avatar_pos, next_step.action, next_step.action_data
                    ):
                        chosen_action = next_step.action
                        chosen_data = next_step.action_data
                        predicted_pos = next_step.predicted_avatar_pos
                        branch = "mental_simulation"
                    else:
                        view.mental_plan.clear()
                        view.working_memory.active_plan_step = None
                        chosen_action, chosen_data = view.plan_epistemic_probe(
                            curr_grid, available_actions
                        )
                        branch = "epistemic_probe"
                    chosen_action, chosen_data = self.ensure_effector_grounded(
                        view, curr_grid, available_actions, chosen_action, chosen_data
                    )
                    return self._build_proposal(
                        view,
                        chosen_action,
                        chosen_data,
                        branch=branch,
                        predicted_pos=predicted_pos,
                        grid=curr_grid,
                    )
                else:
                    chosen_action, chosen_data = view.plan_epistemic_probe(
                        curr_grid, available_actions
                    )
                    chosen_action, chosen_data = self.ensure_effector_grounded(
                        view, curr_grid, available_actions, chosen_action, chosen_data
                    )
                    return self._build_proposal(
                        view,
                        chosen_action,
                        chosen_data,
                        branch="epistemic_probe",
                        grid=curr_grid,
                    )
            else:
                if active_step is not None:
                    next_step = active_step
                else:
                    next_step = view.mental_plan.popleft()
                    view.working_memory.active_plan_step = next_step

            if not view.mental_plan and view.working_memory.active_plan_step is None:
                view.state.phase = EpistemicPhase.REPLANNING
                if view.object_planner.active_target_object_id:
                    view.object_planner.ledger.record_interaction(
                        view.object_planner.active_target_object_id,
                        next_step.predicted_avatar_pos or view.avatar_pos or (0, 0),
                        feature_id=view.object_planner.active_target_feature_id,
                    )
                    view.object_planner.active_target_object_id = None
                    view.object_planner.active_target_feature_id = None

            plan_safe, should_wait, wait_steps = self.validate_plan_safety(
                view, curr_grid, available_actions, next_step
            )

            if should_wait:
                chosen_action, chosen_data, predicted_pos = self._execute_temporal_pacing(
                    view, curr_grid, available_actions, next_step, wait_steps
                )
                branch = "temporal_pacing"
            elif not plan_safe:
                view.state.consecutive_pacing_steps = 0
                view.mental_plan.clear()
                view.working_memory.active_plan_step = None
                view.state.phase = EpistemicPhase.REPLANNING

                danger_cells = (
                    {next_step.predicted_avatar_pos}
                    if next_step.predicted_avatar_pos is not None and not has_temporal
                    else set()
                )
                simulated_plan = view.simulate_in_mind(
                    curr_grid, available_actions, blocked_cells=danger_cells
                )
                if simulated_plan:
                    view.apply_wm_delta(WorkingMemoryDelta("consecutive_simulation_failures", 0))
                    view.apply_wm_delta(WorkingMemoryDelta("simulation_cooldown", 0))
                    view.apply_wm_delta(WorkingMemoryDelta("phase", self._target_phase(view)))
                    view.apply_wm_delta(WorkingMemoryDelta("mental_plan", deque(simulated_plan)))
                    next_step = view.mental_plan.popleft()
                    view.working_memory.active_plan_step = next_step
                    chosen_action = next_step.action
                    chosen_data = next_step.action_data
                    predicted_pos = next_step.predicted_avatar_pos
                    branch = "mental_simulation"
                else:
                    chosen_action, chosen_data = view.plan_epistemic_probe(
                        curr_grid, available_actions
                    )
                    branch = "epistemic_probe"
            else:
                is_action_trigger = getattr(next_step, "expected_mutation", "") in (
                    "ACTION_TRIGGER",
                    "INTERACT_TRIGGER",
                )
                is_failed_step = (view.avatar_pos, next_step.action) in view.failed_transitions
                if (
                    next_step.action in available_actions
                    and (
                        has_temporal
                        or is_action_trigger
                        or not is_failed_step
                        or view.inhibited_actions.get(next_step.action, 0) == 0
                        or all(view.inhibited_actions.get(a, 0) > 0 for a in available_actions)
                    )
                    and (
                        not is_failed_step
                        or not view.working_memory.habenular_ior.is_action_inhibited(
                            view.avatar_pos, next_step.action, next_step.action_data
                        )
                    )
                ):
                    if is_action_trigger:
                        view.inhibited_actions[next_step.action] = 0
                    view.state.consecutive_pacing_steps = 0
                    chosen_action = next_step.action
                    chosen_data = next_step.action_data
                    predicted_pos = next_step.predicted_avatar_pos
                    branch = "mental_plan"

                else:
                    view.mental_plan.clear()
                    view.working_memory.active_plan_step = None
                    view.state.phase = EpistemicPhase.REPLANNING
                    simulated_plan = view.simulate_in_mind(curr_grid, available_actions)
                    if simulated_plan:
                        view.apply_wm_delta(
                            WorkingMemoryDelta("consecutive_simulation_failures", 0)
                        )
                        view.apply_wm_delta(WorkingMemoryDelta("simulation_cooldown", 0))
                        view.apply_wm_delta(WorkingMemoryDelta("phase", self._target_phase(view)))
                        view.apply_wm_delta(
                            WorkingMemoryDelta("mental_plan", deque(simulated_plan))
                        )
                        next_step = view.mental_plan.popleft()
                        view.working_memory.active_plan_step = next_step
                        chosen_action = next_step.action
                        chosen_data = next_step.action_data
                        predicted_pos = next_step.predicted_avatar_pos
                        branch = "mental_simulation"
                    else:
                        chosen_action, chosen_data = view.plan_epistemic_probe(
                            curr_grid, available_actions
                        )
                        branch = "epistemic_probe"

        # 3. Master Planning Branch (when no active plan is in flight)
        else:
            chosen_action = None
            chosen_data = None
            predicted_pos = None

            if has_uncalibrated_cardinal:
                view.mental_plan.clear()
                view.working_memory.active_plan_step = None

            # 3A. Primary: Forward Mental Simulation over learned dynamics & verified exits
            # (Executive Directive 1 & 2: Check if exit is open, if open go through it)
            if (
                view.is_motor_grounded()
                and not has_uncalibrated_cardinal
                and not in_simulation_cooldown
                and view.simulation_cooldown == 0
            ):
                simulated_plan = view.simulate_in_mind(curr_grid, available_actions)
                if simulated_plan:
                    view.apply_wm_delta(WorkingMemoryDelta("consecutive_simulation_failures", 0))
                    view.apply_wm_delta(WorkingMemoryDelta("simulation_cooldown", 0))
                    view.apply_wm_delta(WorkingMemoryDelta("phase", self._target_phase(view)))
                    view.apply_wm_delta(WorkingMemoryDelta("mental_plan", deque(simulated_plan)))
                    next_step = view.mental_plan.popleft()
                    view.working_memory.active_plan_step = next_step
                    plan_safe, should_wait, wait_steps = self.validate_plan_safety(
                        view, curr_grid, available_actions, next_step
                    )
                    if should_wait:
                        chosen_action, chosen_data, predicted_pos = self._execute_temporal_pacing(
                            view, curr_grid, available_actions, next_step, wait_steps
                        )
                        branch = "temporal_pacing"
                    else:
                        is_action_trigger = getattr(next_step, "expected_mutation", "") in (
                            "ACTION_TRIGGER",
                            "INTERACT_TRIGGER",
                        )
                        is_failed_step = (
                            view.avatar_pos,
                            next_step.action,
                        ) in view.failed_transitions
                        if (
                            plan_safe
                            and next_step.action in available_actions
                            and (
                                has_temporal
                                or is_action_trigger
                                or not is_failed_step
                                or view.inhibited_actions.get(next_step.action, 0) == 0
                                or all(
                                    view.inhibited_actions.get(a, 0) > 0 for a in available_actions
                                )
                            )
                            and (
                                not is_failed_step
                                or not view.working_memory.habenular_ior.is_action_inhibited(
                                    view.avatar_pos, next_step.action, next_step.action_data
                                )
                            )
                        ):
                            if is_action_trigger:
                                view.inhibited_actions[next_step.action] = 0
                            chosen_action = next_step.action
                            chosen_data = next_step.action_data
                            predicted_pos = next_step.predicted_avatar_pos
                            branch = "mental_simulation"
                        else:
                            view.mental_plan.clear()
                            view.working_memory.active_plan_step = None
                            view.state.phase = EpistemicPhase.REPLANNING
                            chosen_action, chosen_data = view.plan_epistemic_probe(
                                curr_grid, available_actions
                            )
                            branch = "epistemic_probe"

                else:
                    view.consecutive_simulation_failures += 1
                    view.apply_wm_delta(
                        WorkingMemoryDelta(
                            "simulation_cooldown",
                            min(2, 2**view.consecutive_simulation_failures),
                        )
                    )
                    view.working_memory.update_salience_baseline(view)

            # 3B. Secondary: Object-Centric Macro Action Planning (Directives 3, 4, 5)
            # If exit is not open/reachable or in search backoff, interact with intermediate objects.
            # Must not preempt initial motor calibration when uncalibrated actions remain.
            has_uncalibrated = any(
                not view.is_action_calibrated(a)
                and a not in view.immobile_entity_actions
                and view.inhibited_actions.get(a, 0) == 0
                and (a, view.avatar_pos) not in view.tested_action_positions
                for a in available_actions
                if not view.is_spatial_effector(a)
            )
            if chosen_action is None and not has_uncalibrated:
                macro_plan = view.object_planner.plan_macro_option(
                    view, curr_grid, available_actions
                )
                if (
                    macro_plan
                    and macro_plan[0].action in available_actions
                    and view.inhibited_actions.get(macro_plan[0].action, 0) == 0
                    and not view.working_memory.habenular_ior.is_action_inhibited(
                        view.avatar_pos, macro_plan[0].action, macro_plan[0].action_data
                    )
                ):
                    view.apply_wm_delta(WorkingMemoryDelta("phase", self._target_phase(view)))
                    view.apply_wm_delta(WorkingMemoryDelta("mental_plan", deque(macro_plan)))
                    next_step = view.mental_plan.popleft()
                    chosen_action = next_step.action
                    chosen_data = next_step.action_data
                    predicted_pos = next_step.predicted_avatar_pos
                    branch = "macro_plan"
                    if not view.mental_plan:
                        view.state.phase = EpistemicPhase.REPLANNING
                        if view.object_planner.active_target_object_id:
                            view.object_planner.ledger.record_interaction(
                                view.object_planner.active_target_object_id,
                                predicted_pos or view.avatar_pos or (0, 0),
                                feature_id=view.object_planner.active_target_feature_id,
                            )
                            view.object_planner.active_target_object_id = None
                            view.object_planner.active_target_feature_id = None
            if chosen_action is None:
                if view.exploration_cooldown > 0:
                    view.exploration_cooldown -= 1
                view.state.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
                chosen_action, chosen_data = view.plan_epistemic_probe(curr_grid, available_actions)
                branch = "epistemic_probe"

        # Effector Grounding Invariant
        chosen_action, chosen_data = self.ensure_effector_grounded(
            view, curr_grid, available_actions, chosen_action, chosen_data
        )

        return self._build_proposal(
            view,
            chosen_action,
            chosen_data,
            branch=branch,
            predicted_pos=predicted_pos,
            grid=curr_grid,
        )

    @classmethod
    def ensure_effector_grounded(
        cls,
        view: Any,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
        chosen_action: Any,
        chosen_data: dict[str, Any] | None,
    ) -> tuple[Any, dict[str, Any] | None]:
        """Effector Grounding Invariant: Ensure any spatial effector has valid spatial targets."""
        eff_fn = getattr(view, "is_spatial_effector", None)
        if callable(eff_fn) and eff_fn(chosen_action):
            aff = view.action_affordances.get(chosen_action)
            req_keys = aff.target_param_keys if aff else ("x", "y")
            if (
                chosen_data is None
                or not isinstance(chosen_data, dict)
                or not all(k in chosen_data for k in req_keys)
            ):
                chosen_data = view.ground_effector_action(curr_grid, chosen_action)
                if chosen_data is None:
                    view.inhibited_actions[chosen_action] = max(
                        view.inhibited_actions.get(chosen_action, 0), 6
                    )
                    non_eff = [
                        a
                        for a in available_actions
                        if not view.is_spatial_effector(a) and view.inhibited_actions.get(a, 0) == 0
                    ]
                    chosen_action = (
                        non_eff[0]
                        if non_eff
                        else (available_actions[0] if available_actions else 1)
                    )
                    chosen_data = None
        return chosen_action, chosen_data

    @classmethod
    def _build_proposal(
        cls,
        view: Any,
        action: Any,
        action_data: dict[str, Any] | None,
        branch: str,
        predicted_pos: tuple[int, int] | None = None,
        grid: np.ndarray | None = None,
        candidates: dict[int, float] | None = None,
        veto_reasons: list[str] | None = None,
    ) -> ActionProposal:
        # Determine goal target and goal source
        goal_target: tuple[int, int] | None = None
        goal_source: str = "none"

        # Check active mental plan target first
        mental_plan = view.mental_plan
        if mental_plan:
            last_step = mental_plan[-1]
            last_pos = last_step.predicted_avatar_pos
            if last_pos is not None:
                goal_target = last_pos
                goal_source = (
                    "learned_position"
                    if goal_target in view.learned_goal_positions
                    else "epistemic"
                )

        # If no target from active plan, check confirmed reward goal features in visible grid
        if goal_target is None and grid is not None and len(grid.shape) >= 2:
            confirmed_feats = view.symbolic_theory.goal_features
            if confirmed_feats:
                for r in range(grid.shape[0]):
                    for c in range(grid.shape[1]):
                        if int(
                            grid[r, c]
                        ) in confirmed_feats and not view.working_memory.is_exogenous_locus(
                            (r, c), lattice=grid, bg=view.bg_feature, include_outer_edge=True
                        ):
                            goal_target = (r, c)
                            goal_source = "reward"
                            break
                    if goal_target is not None:
                        break

        # Check learned goal positions
        if goal_target is None and view.learned_goal_positions:
            av_p = view.avatar_pos
            if av_p is not None:
                goal_target = min(
                    view.learned_goal_positions,
                    key=lambda g: abs(g[0] - av_p[0]) + abs(g[1] - av_p[1]),
                )
            else:
                goal_target = next(iter(view.learned_goal_positions))
            goal_source = "learned_position"

        # Check subgoal stack
        if goal_target is None:
            curr_sg = view.subgoal_stack.current_subgoal
            if curr_sg and curr_sg.target_destination:
                goal_target = curr_sg.target_destination
                goal_source = "epistemic"

        # Check current simulated target or committed active epistemic probe target
        if goal_target is None:
            sim_goal = view.state.current_simulated_goal
            if sim_goal is not None:
                goal_target = sim_goal
                goal_source = "epistemic"
            elif view.active_probe_target is not None:
                goal_target = view.active_probe_target
                goal_source = "epistemic"

        # Check nearest reachable unvisited frontier cell when localized
        if (
            goal_target is None
            and grid is not None
            and len(grid.shape) >= 2
            and view.avatar_pos is not None
        ):
            bg_val = (
                view.bg_feature if view.bg_feature is not None else view.estimate_background(grid)
            )
            av_feats = set(view.avatar_features)
            if view.avatar_feature is not None:
                av_feats.add(view.avatar_feature)
            frontier = EpistemicGoalScheduler.compute_frontier_candidates(
                engine=view,
                curr_grid=grid,
                bg=bg_val,
                av_feats=av_feats,
            )
            if frontier:
                goal_target = frontier[0]
                goal_source = "epistemic"

        # Subgoal stack representation (top-of-stack first)
        subgoal_stack: list[Any] = [s.subgoal_id for s in reversed(view.subgoal_stack.stack)]

        # Plan trajectory representation
        plan: list[tuple[int, int]] | None = None
        if mental_plan:
            plan_positions = [
                s.predicted_avatar_pos for s in mental_plan if s.predicted_avatar_pos is not None
            ]
            if plan_positions:
                plan = plan_positions

        # Plan source classification
        if branch in ("mental_simulation", "mental_plan", "causal_path", "astar"):
            plan_source = "astar"
        elif branch in (
            "macro_option",
            "macro_recall",
            "relational_macro",
            "synthesized_schema",
            "macro_plan",
        ):
            plan_source = "macro"
        elif branch in ("escape", "evasion", "temporal_pacing"):
            plan_source = "escape"
        else:
            plan_source = "arbitration"

        # Candidate action scoring
        if candidates is None:
            avail = getattr(view, "available_actions", [1, 2, 3, 4]) or [1, 2, 3, 4]
            candidates = {int(a): (1.0 if a == action else 0.0) for a in avail}

        # Veto reasons
        if veto_reasons is None:
            veto_reasons = []
            for act_id, count in view.inhibited_actions.items():
                if count > 0:
                    veto_reasons.append(f"inhibited_action_{act_id}_cooldown_{count}")
            hab_ior = view.working_memory.habenular_ior
            av_p = view.avatar_pos
            if av_p:
                for a in getattr(view, "available_actions", []):
                    if hab_ior.is_action_inhibited(av_p, a, None):
                        veto_reasons.append(f"habenula_ior_veto_{a}")
            if view.consecutive_stuck_steps > 0:
                veto_reasons.append(f"consecutive_stuck_steps_{view.consecutive_stuck_steps}")
            if view.simulation_cooldown > 0:
                veto_reasons.append(f"simulation_cooldown_{view.simulation_cooldown}")

        # Veto candidates that deviate from an unblocked plan during exploitation
        if (
            view.phase == EpistemicPhase.EXPLOITATION
            and action is not None
            and plan is not None
            and candidates is not None
        ):
            for cand_act in candidates:
                if (
                    cand_act != action
                    and f"deviates_from_verified_plan_{cand_act}" not in veto_reasons
                ):
                    veto_reasons.append(f"deviates_from_verified_plan_{cand_act}")

        # Cognitive phase
        phase_str = str(view.state.phase)

        # Basal Ganglia Tri-Pathway Channel Scoring & Neuromodulatory State Provenance
        bg_channel_scores: dict[Any, dict[str, Any]] = {}
        neuromodulators: dict[str, Any] = {}
        try:
            delib_inst = view.deliberation_engine
            if delib_inst is not None:
                acc_conf = 0.0
                if view.acc_conflict_monitor is not None:
                    acc_conf = float(view.acc_conflict_monitor.current_conflict or 0.0)
                _, bg_decisions = delib_inst.arbitrate_basal_ganglia_action(
                    view, candidates or {}, acc_conflict=acc_conf
                )
                for d in bg_decisions:
                    bg_channel_scores[d.action] = {
                        "go": d.d1_go_drive,
                        "nogo": d.d2_nogo_drive,
                        "hyperdirect": d.stn_brake_signal,
                        "net": d.net_drive,
                        "pathway": d.pathway.value,
                        "suppressed": d.is_suppressed,
                    }
                nm = delib_inst.neuromodulation.current_state
                neuromodulators = {
                    "dopamine_tonic": nm.dopamine_tonic,
                    "dopamine_phasic_rpe": nm.dopamine_phasic_rpe,
                    "noradrenaline_tonic": nm.noradrenaline_tonic,
                    "noradrenaline_phasic": nm.noradrenaline_phasic,
                    "exploration_gain": nm.exploration_gain,
                    "exploitation_gain": nm.exploitation_gain,
                    "regime": nm.regime,
                    "is_disengaged": nm.is_disengaged,
                }
        except Exception as exc:
            logger.debug("Arbitration diagnostics provenance skipped: %s", exc)

        meta = {
            "goal_target": goal_target,
            "goal_source": goal_source,
            "subgoal_stack": subgoal_stack,
            "plan": plan,
            "plan_source": plan_source,
            "candidates": candidates,
            "bg_channel_scores": bg_channel_scores,
            "neuromodulators": neuromodulators,
            "chosen_action": action,
            "veto_reasons": veto_reasons,
            "phase": phase_str,
            "branch": branch,
            "action_data": action_data,
        }

        return ActionProposal(
            action=action,
            action_data=action_data,
            branch=branch,
            predicted_pos=predicted_pos,
            grid=grid,
            meta=meta,
        )

    @classmethod
    def _finalize_step(
        cls,
        view: Any,
        curr_grid: np.ndarray,
        chosen_action: Any,
        chosen_data: dict[str, Any] | None,
        predicted_pos: tuple[int, int] | None = None,
    ) -> tuple[Any, dict[str, Any] | None]:
        chosen_action, chosen_data = cls.ensure_effector_grounded(
            view,
            curr_grid,
            getattr(view, "available_actions", []) or [1, 2, 3, 4],
            chosen_action,
            chosen_data,
        )
        view.state.prev_grid = curr_grid.copy()
        view.grid_history.append(curr_grid.copy())
        view.state.last_action = chosen_action
        view.state.last_action_data = chosen_data
        view.state.last_predicted_pos = predicted_pos
        return chosen_action, chosen_data

    def to_causal_predicate(self: Any = None) -> CausalPredicate:
        """Export prefrontal deliberation view capability and runtime neuromodulatory state as a declarative causal predicate."""
        regime = "ACTIVE"
        relation = "dlpfc_vmpfc_arbitration"
        if self is not None:
            nm = self.neuromodulation.current_state
            relation = f"dlpfc_vmpfc_arbitration:regime:{nm.regime}:da:{nm.dopamine_tonic:.2f}:na:{nm.noradrenaline_tonic:.2f}"
        return CausalPredicate(
            variable="prefrontal_deliberation_engine",
            operator="==",
            value=regime,
            relation=relation,
            target_entity="cortex_deliberation",
        )

    def to_transaction_operations(self: Any = None) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction registering deliberation view and runtime state."""
        node_data: dict[str, Any] = {
            "faculty": "prefrontal_deliberation_engine",
            "status": "ACTIVE",
        }
        if self is not None:
            nm = self.neuromodulation.current_state
            node_data["regime"] = nm.regime
            node_data["dopamine_tonic"] = float(nm.dopamine_tonic)
            node_data["noradrenaline_tonic"] = float(nm.noradrenaline_tonic)
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="deliberation_engine",
                node_data=node_data,
            )
        ]


@dataclass(frozen=True)
class ActionProposal:
    """Deliberation's decision intent — finalised by MotorStage."""

    action: Any
    action_data: Any = None
    branch: str = "unknown"
    predicted_pos: Any = None
    grid: Any = None
    finalized: bool = False
    meta: dict[str, Any] = field(default_factory=dict)
