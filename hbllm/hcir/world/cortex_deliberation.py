from __future__ import annotations

"""Prefrontal Cortex Deliberation & Executive Arbitration Faculty.

Biologically modeled on mammalian dorsolateral / ventromedial prefrontal cortex (dlPFC / vmPFC)
and striatal gating:
1. Multi-Faculty Cognitive Arbitration:
   - Evaluates active mental plans against static lethal entities, spatiotemporal looming cones,
     dynamic patroller forward trajectories, and Habenular episodic action inhibition.
2. Cerebellar Phase-Locked Motor Gating:
   - Synchronizes kinetic movements to rhythmic oscillation windows.
3. Affordance Panel & Optical Symmetry Arbitration:
   - Directs spatial effectors toward interactive control panels and relational reflection midpoints.
4. Deliberate-to-Habitual Search Backoff (Daw, Niv, & Dayan; Dolan & Dayan):
   - Transitions between deliberate forward mental simulation and habitual macro-action exploration.
"""

import logging
from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.world.cortex_assimilator import EpistemicPhase
from hbllm.hcir.world.cortex_perception import PerceptionEngine
from hbllm.hcir.world.cortex_planner import MentalSimulationStep

if TYPE_CHECKING:
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

logger = logging.getLogger(__name__)


class PrefrontalDeliberationEngine:
    """Executive prefrontal arbitration engine coordinating perception, mental simulation,
    affordance chunking, and motor output.
    """

    def __init__(self) -> None:
        pass

    def validate_plan_safety(
        self,
        engine: AutonomousEpistemicEngine,
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
        H, W = curr_grid.shape
        bg = engine.estimate_background(curr_grid)
        all_future_steps = [next_step] + list(engine.mental_plan)

        # (A) Static feature check along the planned path
        for future_step in all_future_steps:
            if future_step.predicted_avatar_pos is None:
                continue
            fr, fc = future_step.predicted_avatar_pos
            if 0 <= fr < H and 0 <= fc < W:
                feat = int(curr_grid[fr, fc])
                if feat in engine.mobile_threat_features:
                    continue
                if feat in engine.hazard_tracker.known_lethal_features:
                    return False, False, 0

        # (A2) Spatiotemporal Collision Cones check
        if engine.avatar_pos is not None:
            ar, ac = engine.avatar_pos
            is_stationary = (
                next_step.predicted_avatar_pos is None or next_step.predicted_avatar_pos == (ar, ac)
            )
            if is_stationary and any(
                engine.collision_cones.is_collision_hazard(
                    ar, ac, t_f, ignored_features=engine.mobile_threat_features
                )
                for t_f in (1, 2)
            ):
                return False, False, 0
            elif next_step.predicted_avatar_pos is not None:
                nr, nc = next_step.predicted_avatar_pos
                if engine.collision_cones.is_collision_hazard(
                    nr, nc, 1, ignored_features=engine.mobile_threat_features
                ):
                    return False, False, 0

        # (B) Dynamic patrol trajectory simulation
        if engine.mobile_threat_features:
            try:
                step_size = engine.infer_motor_step_size(available_actions)
                entities = engine.extract_entities(curr_grid, bg)
                av_feats = engine.avatar_features or (
                    {engine.avatar_feature} if engine.avatar_feature is not None else set()
                )
                live_threats = PerceptionEngine.detect_oriented_threats(
                    curr_grid,
                    entities,
                    bg=bg,
                    step_size=step_size,
                    avatar_pos=engine.avatar_pos,
                    avatar_features=av_feats,
                )
                live_patrols: set[tuple[tuple[int, int], tuple[int, int]]] = set()
                for t in live_threats:
                    if t.feature_id in engine.mobile_threat_features:
                        live_patrols.add((t.pos, t.facing))

                if live_patrols:
                    half_step = max(1, step_size // 2)
                    barrier_feats = engine.symbolic_theory.barrier_features
                    bg_is_void = (
                        len(engine.symbolic_theory.walkable_features) > 0
                        and bg not in engine.symbolic_theory.walkable_features
                    )
                    patrol_walls: set[tuple[int, int]] = set(engine.learned_barriers)
                    for rr in range(H):
                        for cc in range(W):
                            v = int(curr_grid[rr, cc])
                            if (
                                engine.symbolic_theory.is_barrier(v)
                                or v in barrier_feats
                                or (bg_is_void and v == bg)
                            ):
                                patrol_walls.add((rr, cc))

                    def _blocked(p: tuple[int, int], f: tuple[int, int]) -> bool:
                        probe = (p[0] + f[0] * half_step, p[1] + f[1] * half_step)
                        dest = (p[0] + f[0] * step_size, p[1] + f[1] * step_size)
                        for q in (probe, dest):
                            if not (0 <= q[0] < H and 0 <= q[1] < W) or q in patrol_walls:
                                return True
                        return False

                    sim_patrols = frozenset(live_patrols)
                    for future_step in all_future_steps:
                        avatar_dest = future_step.predicted_avatar_pos
                        new_patrol_set: set[tuple[tuple[int, int], tuple[int, int]]] = set()
                        collision = False
                        for pat_pos, pat_facing in sim_patrols:
                            if pat_pos == avatar_dest:
                                collision = True
                                break
                            dest = (
                                pat_pos[0] + pat_facing[0] * step_size,
                                pat_pos[1] + pat_facing[1] * step_size,
                            )
                            if not (0 <= dest[0] < H and 0 <= dest[1] < W) or dest in patrol_walls:
                                new_f = (-pat_facing[0], -pat_facing[1])
                                new_patrol_set.add((pat_pos, new_f))
                                continue
                            if dest == avatar_dest:
                                collision = True
                                break
                            new_f = pat_facing
                            if _blocked(dest, pat_facing):
                                new_f = (-pat_facing[0], -pat_facing[1])
                            new_patrol_set.add((dest, new_f))
                        if collision:
                            return False, False, 0
                        sim_patrols = frozenset(new_patrol_set)
            except Exception:
                pass

        # (C) Habenular Episodic Gating
        if engine.working_memory.habenular_ior.is_action_inhibited(
            engine.avatar_pos, next_step.action
        ):
            return False, False, 0

        # (D) Cerebellar Phase Gating & Rhythm-Locked Safe Windows
        if next_step.predicted_avatar_pos is not None:
            phase_gate = engine.cerebellar_clock.evaluate_motion_hazard_gate(
                current_step=engine.step_counter,
                avatar_pos=engine.avatar_pos,
                target_pos=next_step.predicted_avatar_pos,
                hazard_tracker=engine.hazard_tracker,
                background_feature=engine.bg_feature,
                avatar_features=engine.avatar_features,
                walkable_features=engine.symbolic_theory.walkable_features
                | engine.verified_safe_features,
            )
            if phase_gate.should_wait:
                return True, True, phase_gate.wait_steps_recommended

        return True, False, 0

    def arbitrate_affordances(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
        known_av_feats: set[int],
    ) -> tuple[Any, dict[str, Any] | None] | None:
        """Arbitrate control panel affordances and optical symmetry targets."""
        has_displacement_actions = any(
            engine.is_displacement_action(a) for a in available_actions
        ) or any(
            not engine.is_spatial_effector(a) and a not in engine.action_dynamics
            for a in available_actions
        )
        spatial_effector_actions = [a for a in available_actions if engine.is_spatial_effector(a)]
        has_immobile_failures = bool(
            getattr(engine, "immobile_entity_actions", None)
            and len(engine.recent_actions) >= 2
            and any(a in engine.immobile_entity_actions for a in list(engine.recent_actions)[-2:])
        )
        is_stuck_or_looping = (
            engine.consecutive_stuck_steps >= 2
            or (
                engine.avatar_pos is not None
                and engine.recent_positions.count(engine.avatar_pos) >= 2
            )
            or (engine.avatar_pos is None and engine.step_counter > 4)
            or has_immobile_failures
        )
        has_aligned_optical_subgoal = False
        if engine.avatar_pos is not None and spatial_effector_actions:
            bg_eval = engine.estimate_background(curr_grid)
            sgs = PerceptionEngine.detect_structural_goals(
                curr_grid,
                bg=bg_eval,
                avatar_features=known_av_feats,
                avatar_feature=engine.avatar_feature,
            )
            for sg in sgs:
                if (
                    sg.get("type")
                    in ("relational_alignment", "optical_mirror_target", "reflection_target")
                    and sg.get("is_midpoint")
                    and (
                        engine.avatar_feature is None
                        or sg.get("feature") == engine.avatar_feature
                        or sg.get("feature") in engine.avatar_features
                    )
                ):
                    mp = sg["position"]
                    if (
                        abs(mp[1] - engine.avatar_pos[1]) <= 1
                        and abs(mp[0] - engine.avatar_pos[0]) <= 2
                    ):
                        has_aligned_optical_subgoal = True
                        break

        discrete_transform_actions = [
            a
            for a in available_actions
            if not engine.is_displacement_action(a) and not engine.is_spatial_effector(a)
        ]

        if (
            (spatial_effector_actions or discrete_transform_actions)
            and (
                not has_displacement_actions
                or is_stuck_or_looping
                or (has_aligned_optical_subgoal and not engine.mental_plan)
            )
            and engine.step_counter > 1
        ):
            bg = engine.estimate_background(curr_grid)
            entities = engine.extract_entities(curr_grid, bg)
            panels = PerceptionEngine.detect_affordance_panels(entities, curr_grid, bg=bg)
            active_panel_target: tuple[int, int] | None = None
            for p in panels:
                unvisited_minority = [
                    m.grid_pos
                    for m in p["minority_items"]
                    if engine.entity_visit_counts.get(f"click_{m.grid_pos[0]}_{m.grid_pos[1]}", 0)
                    == 0
                    and m.grid_pos not in engine.quiescent_click_targets
                ]
                if unvisited_minority:
                    active_panel_target = unvisited_minority[0]
                    break

            if active_panel_target is None:
                optical_goals = [
                    g
                    for g in PerceptionEngine.detect_structural_goals(
                        curr_grid,
                        bg=bg,
                        avatar_features=known_av_feats,
                        avatar_feature=engine.avatar_feature,
                    )
                    if g.get("type")
                    in ("relational_alignment", "optical_mirror_target", "reflection_target")
                    and g.get("position") is not None
                    and not g.get("is_midpoint")
                ]
                optical_goals.sort(key=lambda g: g.get("confidence", 0.5), reverse=True)
                for og in optical_goals:
                    mp = og["position"]
                    if engine.avatar_pos is not None:
                        dist_l1 = abs(mp[0] - engine.avatar_pos[0]) + abs(
                            mp[1] - engine.avatar_pos[1]
                        )
                        same_col = abs(mp[1] - engine.avatar_pos[1]) <= 3
                        same_row = abs(mp[0] - engine.avatar_pos[0]) <= 3
                        if dist_l1 < 8 or same_col or same_row:
                            continue
                    if engine.entity_visit_counts.get(f"click_{mp[0]}_{mp[1]}", 0) < 4:
                        active_panel_target = mp
                        break

            if active_panel_target is not None and spatial_effector_actions:
                chosen_action = spatial_effector_actions[0]
                tr, tc = active_panel_target
                engine.entity_visit_counts[f"click_{tr}_{tc}"] = (
                    engine.entity_visit_counts.get(f"click_{tr}_{tc}", 0) + 1
                )
                aff = engine.action_affordances.get(chosen_action)
                param_keys = aff.target_param_keys if aff else ("x", "y")
                chosen_data: dict[str, Any] = {}
                for k in param_keys:
                    if k in ("x", "col", "c", "column", "azimuth"):
                        chosen_data[k] = tc
                    elif k in ("y", "row", "r", "elevation", "distance"):
                        chosen_data[k] = tr
                    else:
                        chosen_data[k] = 0
                engine.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
                return chosen_action, chosen_data
            elif (
                (not spatial_effector_actions or active_panel_target is None)
                and (
                    is_stuck_or_looping or (has_aligned_optical_subgoal and not engine.mental_plan)
                )
                and discrete_transform_actions
            ):
                focus_switches = [
                    a for a in discrete_transform_actions if engine.is_focus_switch_action(a)
                ]
                if focus_switches:
                    chosen_action = focus_switches[0]
                    engine.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
                    return chosen_action, None
                uninhibited_discrete = [
                    a for a in discrete_transform_actions if engine.inhibited_actions.get(a, 0) == 0
                ]
                chosen_action = (
                    uninhibited_discrete[0]
                    if uninhibited_discrete
                    else discrete_transform_actions[0]
                )
                engine.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
                return chosen_action, None

        return None

    def decide(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: Any,
        available_actions: Sequence[Any],
        is_win: bool = False,
        is_lost: bool = False,
        action_schemas: Sequence[Any] | None = None,
    ) -> tuple[Any, dict[str, Any] | None]:
        """Unified cognitive decision function:
        Perceive -> Assimilate -> Simulate in Mind -> Exploit / Epistemically Probe.
        """
        engine.step_counter += 1
        if action_schemas:
            engine.register_action_space(action_schemas)

        curr_grid = engine.normalize_sensory_input(curr_grid)

        # 1. Update background & assimilate sensory feedback from previous action
        engine.bg_feature = engine.estimate_background(curr_grid)
        if engine.prev_grid is not None and not engine._feedback_assimilated:
            engine.assimilate_feedback(curr_grid, available_actions, is_win=is_win, is_lost=is_lost)
        engine._feedback_assimilated = False

        # Phasic Salience Reset: Environmental discovery or high surprise awakens deliberate forward search
        n_calibrated_actions = len(engine.action_dynamics)
        if (
            len(engine.learned_goal_positions) > engine._last_sim_goal_count
            or len(engine.learned_barriers) != engine._last_sim_barrier_count
            or n_calibrated_actions > getattr(engine, "_last_sim_action_count", 0)
            or engine.last_surprise >= 0.4
        ):
            engine.simulation_cooldown = 0
            engine.consecutive_simulation_failures = 0
            if n_calibrated_actions > getattr(engine, "_last_sim_action_count", 0):
                engine.mental_plan.clear()
                engine.phase = (
                    EpistemicPhase.MENTAL_SIMULATION
                    if engine.is_motor_grounded()
                    else EpistemicPhase.EPISTEMIC_EXPLORATION
                )
                engine._last_sim_action_count = n_calibrated_actions
            engine._last_sim_goal_count = len(engine.learned_goal_positions)
            engine._last_sim_barrier_count = len(engine.learned_barriers)

        # Metacognitive Refractory Inhibition: decay refractory timers
        if engine.simulation_cooldown > 0:
            engine.simulation_cooldown -= 1
        to_uninhibited = [act for act, timer in engine.inhibited_actions.items() if timer <= 1]
        for act in engine.inhibited_actions:
            engine.inhibited_actions[act] -= 1
        for act in to_uninhibited:
            engine.inhibited_actions.pop(act, None)

        # Anterior Mid-Cingulate & Lateral Habenula: Limit Cycle / Oscillation Detection
        osc_act = engine.working_memory.habenular_ior.detect_action_oscillation()
        if osc_act is not None:
            engine.inhibited_actions[osc_act] = max(engine.inhibited_actions.get(osc_act, 0), 4)

        # Faculty: Anterior Cingulate Cortex (ACC) Limit Cycle & Conflict Suppression
        if engine.is_motor_grounded() and hasattr(engine, "acc_conflict_monitor"):
            engine.acc_conflict_monitor.record_step(engine.last_action, engine.avatar_pos)
            is_osc, cyclic_acts = engine.acc_conflict_monitor.detect_oscillation()
            if is_osc:
                for cyc_a in cyclic_acts:
                    engine.inhibited_actions[cyc_a] = max(engine.inhibited_actions.get(cyc_a, 0), 4)

        # Visual avatar localization
        known_av_feats = set(engine.avatar_features) if engine.avatar_features else set()
        if engine.avatar_feature is not None:
            known_av_feats.add(engine.avatar_feature)
        elif engine.prior_avatar_feature is not None:
            known_av_feats.add(engine.prior_avatar_feature)
        if known_av_feats:
            engine._update_avatar_position_from_grid(curr_grid, known_av_feats)
            if (
                engine.avatar_pos is not None
                and engine.avatar_feature is None
                and engine.prior_avatar_feature is not None
            ):
                engine.avatar_feature = engine.prior_avatar_feature
                engine.avatar_features.add(engine.prior_avatar_feature)

        # Observe creature motion
        engine._observe_oriented_threat_motion(curr_grid, available_actions, known_av_feats)

        # Faculty E: Basal Ganglia Motor Gating & Phase Entrainment
        if engine.pending_phase_wait_steps > 0:
            engine.pending_phase_wait_steps -= 1
            non_disp = [
                a
                for a in available_actions
                if not engine.is_displacement_action(a) and not engine.is_spatial_effector(a)
            ]
            if non_disp:
                return non_disp[0], None

        # Faculty: Brodmann Area 10 (aPFC) Hierarchical Subgoal Stack & Cognitive Branching
        if engine.subgoal_stack.has_pending_goals:
            cur_sg = engine.subgoal_stack.current_subgoal
            if cur_sg is not None and engine.subgoal_stack.is_subgoal_satisfied(
                cur_sg, set(), avatar_pos=engine.avatar_pos
            ):
                engine.subgoal_stack.pop_subgoal()

        chosen_action: Any
        chosen_data: dict[str, Any] | None = None
        predicted_pos: tuple[int, int] | None = None

        # Prefrontal Affordance Panel & Optical Symmetry Gating
        affordance_choice = self.arbitrate_affordances(
            engine, curr_grid, available_actions, known_av_feats
        )
        if affordance_choice is not None:
            chosen_action, chosen_data = affordance_choice

        # 2. Check exploration cooldown
        elif engine.exploration_cooldown > 0:
            engine.exploration_cooldown -= 1
            engine.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
            chosen_action, chosen_data = engine.plan_epistemic_probe(curr_grid, available_actions)

        # 3. Check active mental plan during EXPLOITATION phase
        elif engine.phase == EpistemicPhase.EXPLOITATION and engine.mental_plan:
            next_step = engine.mental_plan.popleft()
            if (
                not engine.mental_plan
                and hasattr(engine, "object_planner")
                and engine.object_planner.active_target_object_id
            ):
                engine.object_planner.ledger.record_interaction(
                    engine.object_planner.active_target_object_id,
                    next_step.predicted_avatar_pos or engine.avatar_pos or (0, 0),
                )
                engine.object_planner.active_target_object_id = None

            plan_safe, should_wait, wait_steps = self.validate_plan_safety(
                engine, curr_grid, available_actions, next_step
            )

            if should_wait:
                engine.mental_plan.appendleft(next_step)
                engine.pending_phase_wait_steps = wait_steps
                non_disp = [
                    a
                    for a in available_actions
                    if not engine.is_displacement_action(a) and not engine.is_spatial_effector(a)
                ]
                if non_disp:
                    return non_disp[0], None

            if not plan_safe:
                engine.mental_plan.clear()
                engine.phase = EpistemicPhase.REPLANNING
                simulated_plan = engine.simulate_in_mind(curr_grid, available_actions)
                if not simulated_plan:
                    danger_cells = (
                        {next_step.predicted_avatar_pos}
                        if next_step.predicted_avatar_pos is not None
                        else set()
                    )
                    simulated_plan = engine.simulate_in_mind(
                        curr_grid, available_actions, blocked_cells=danger_cells
                    )
                if simulated_plan:
                    engine.consecutive_simulation_failures = 0
                    engine.simulation_cooldown = 0
                    engine.phase = EpistemicPhase.EXPLOITATION
                    engine.mental_plan = deque(simulated_plan)
                    next_step = engine.mental_plan.popleft()
                    chosen_action = next_step.action
                    chosen_data = next_step.action_data
                    predicted_pos = next_step.predicted_avatar_pos
                else:
                    engine.consecutive_simulation_failures += 1
                    engine.simulation_cooldown = min(
                        12, 2 ** min(engine.consecutive_simulation_failures, 4)
                    )
                    macro_plan = (
                        engine.object_planner.plan_macro_option(
                            engine, curr_grid, available_actions
                        )
                        if hasattr(engine, "object_planner")
                        else None
                    )
                    if macro_plan:
                        engine.phase = EpistemicPhase.EXPLOITATION
                        engine.mental_plan = deque(macro_plan)
                        next_step = engine.mental_plan.popleft()
                        chosen_action = next_step.action
                        chosen_data = next_step.action_data
                        predicted_pos = next_step.predicted_avatar_pos
                    else:
                        chosen_action, chosen_data = engine.plan_epistemic_probe(
                            curr_grid, available_actions
                        )
            elif next_step.action in available_actions:
                chosen_action = next_step.action
                chosen_data = next_step.action_data
                predicted_pos = next_step.predicted_avatar_pos
            else:
                engine.mental_plan.clear()
                engine.phase = EpistemicPhase.REPLANNING
                chosen_action, chosen_data = engine.plan_epistemic_probe(
                    curr_grid, available_actions
                )

        # 4. If motor grounded, attempt Forward Mental Simulation or Habitual Exploration
        elif engine.is_motor_grounded():
            can_simulate = engine.simulation_cooldown == 0
            simulated_plan = None
            if can_simulate:
                simulated_plan = engine.simulate_in_mind(curr_grid, available_actions)
                if simulated_plan:
                    engine.consecutive_simulation_failures = 0
                    engine.simulation_cooldown = 0
                    engine.phase = EpistemicPhase.EXPLOITATION
                    engine.mental_plan = deque(simulated_plan)
                    next_step = engine.mental_plan.popleft()
                    chosen_action = next_step.action
                    chosen_data = next_step.action_data
                    predicted_pos = next_step.predicted_avatar_pos
                else:
                    engine.consecutive_simulation_failures += 1
                    engine.simulation_cooldown = min(
                        12, 2 ** min(engine.consecutive_simulation_failures, 4)
                    )
                    engine._last_sim_goal_count = len(engine.learned_goal_positions)
                    engine._last_sim_barrier_count = len(engine.learned_barriers)

            if simulated_plan is None:
                focus_switches = [a for a in available_actions if engine.is_focus_switch_action(a)]
                if focus_switches and engine.consecutive_simulation_failures >= 1:
                    engine.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
                    return focus_switches[0], None

                macro_plan = (
                    engine.object_planner.plan_macro_option(engine, curr_grid, available_actions)
                    if hasattr(engine, "object_planner")
                    else None
                )
                if macro_plan:
                    engine.phase = EpistemicPhase.EXPLOITATION
                    engine.mental_plan = deque(macro_plan)
                    next_step = engine.mental_plan.popleft()
                    chosen_action = next_step.action
                    chosen_data = next_step.action_data
                    predicted_pos = next_step.predicted_avatar_pos
                    if (
                        not engine.mental_plan
                        and hasattr(engine, "object_planner")
                        and engine.object_planner.active_target_object_id
                    ):
                        engine.object_planner.ledger.record_interaction(
                            engine.object_planner.active_target_object_id,
                            predicted_pos or engine.avatar_pos or (0, 0),
                        )
                        engine.object_planner.active_target_object_id = None
                else:
                    engine.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
                    chosen_action, chosen_data = engine.plan_epistemic_probe(
                        curr_grid, available_actions
                    )

        # 5. Fallback to Epistemic Curiosity Probing
        else:
            engine.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
            chosen_action, chosen_data = engine.plan_epistemic_probe(curr_grid, available_actions)

        # Effector Grounding Invariant
        if engine.is_spatial_effector(chosen_action):
            aff = engine.action_affordances.get(chosen_action)
            req_keys = aff.target_param_keys if aff else ("x", "y")
            if (
                chosen_data is None
                or not isinstance(chosen_data, dict)
                or not all(k in chosen_data for k in req_keys)
            ):
                chosen_data = engine.ground_effector_action(curr_grid, chosen_action)

        engine.prev_grid = curr_grid.copy()
        engine.last_action = chosen_action
        engine.last_action_data = chosen_data
        engine.last_predicted_pos = predicted_pos
        return chosen_action, chosen_data
