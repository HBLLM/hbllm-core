from __future__ import annotations

"""Basal Ganglia & Anterior Cingulate Cortex (ACC) Motor Arbitration Faculty.

Biologically modeled on mammalian striatal action selection, dopamine RPE gating,
and ACC conflict monitoring:
1. Basal Ganglia Active Inference:
   - Evaluates candidate motor commands by trading off pragmatic utility (goal approach)
     and epistemic value (information gain / uncertainty reduction).
2. Motor Grounding & Agency Discovery:
   - Motor babbling correlates intentional action efference copies with sensory visual feedback.
   - Distinguishes self-avatar displacements from external environmental dynamics.
3. ACC Conflict Monitor & Limit-Cycle Breaker:
   - Detects motor deadlocks (e.g. 2-cycle or 3-cycle oscillations) and forces orthogonal exploration.
4. Epistemic Curiosity Explorer:
   - Systematically probes uncalibrated actions, unvisited spatial frontiers, and affordance panels.
"""

import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.spatial_planner import SpatialEntity
from hbllm.hcir.world.active_inference import ActionNode
from hbllm.hcir.world.cortex_perception import PerceptionEngine
from hbllm.hcir.world.counterfactual_simulation import CounterfactualDeadlockDetector
from hbllm.hcir.world.spatial_containment import RoomDoor

if TYPE_CHECKING:
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

logger = logging.getLogger(__name__)


@dataclass
class ACCConflictRecord:
    """Record of cognitive conflict or motor deadlock detected by the ACC."""

    step: int
    cycle_period: int
    cyclic_actions: list[Any]
    suppressed_actions: list[Any] = field(default_factory=list)


class ACCConflictMonitor:
    """Anterior Cingulate Cortex conflict detector and limit-cycle oscillation breaker."""

    def __init__(self, history_window: int = 16, oscillation_threshold: int = 3) -> None:
        self.history_window = history_window
        self.oscillation_threshold = oscillation_threshold
        self.action_history: deque[Any] = deque(maxlen=history_window)
        self.position_history: deque[tuple[int, int]] = deque(maxlen=history_window)
        self.conflict_records: list[ACCConflictRecord] = []

    def record_step(self, action: Any, avatar_pos: tuple[int, int] | None) -> None:
        """Record an executed action and resulting position."""
        self.action_history.append(action)
        if avatar_pos is not None:
            self.position_history.append(avatar_pos)

    def detect_oscillation(self) -> tuple[bool, list[Any]]:
        """Detect 2-cycle or 3-cycle limit cycles in recent actions or positions.

        Returns:
            (is_oscillating, cyclic_actions_to_suppress)
        """
        acts = list(self.action_history)
        if len(acts) >= 4:
            # Check 2-cycle: e.g. [A, B, A, B]
            if acts[-1] == acts[-3] and acts[-2] == acts[-4] and acts[-1] != acts[-2]:
                return True, [acts[-1], acts[-2]]

        if len(acts) >= 6:
            # Check 3-cycle: e.g. [A, B, C, A, B, C]
            if acts[-1] == acts[-4] and acts[-2] == acts[-5] and acts[-3] == acts[-6]:
                return True, [acts[-1], acts[-2], acts[-3]]

        # Check position bounce: avatar stuck between two coordinates
        pos_list = list(self.position_history)
        if len(pos_list) >= 4:
            if (
                pos_list[-1] == pos_list[-3]
                and pos_list[-2] == pos_list[-4]
                and pos_list[-1] != pos_list[-2]
            ):
                return True, list(acts[-2:])

        return False, []

    def reset(self) -> None:
        """Clear conflict history."""
        self.action_history.clear()
        self.position_history.clear()
        self.conflict_records.clear()


class EpistemicCuriosityExplorer:
    """Active curiosity-driven hypothesis testing and information-gain exploration."""

    @staticmethod
    def _is_ray_passable(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        r0: int,
        c0: int,
        dr: int,
        dc: int,
        bg: int,
        bg_is_impassable_void: bool,
    ) -> bool:
        """Check whether the full displacement ray from (r0, c0) to (r0+dr, c0+dc) is passable."""
        H, W = curr_grid.shape
        step_r = int(np.sign(dr))
        step_c = int(np.sign(dc))
        dist = max(abs(dr), abs(dc))
        for k in range(1, dist + 1):
            kr = r0 + step_r * k
            kc = c0 + step_c * k
            if not (0 <= kr < H and 0 <= kc < W):
                return False
            feat = int(curr_grid[kr, kc])
            if (kr, kc) in engine.learned_barriers or engine.symbolic_theory.is_barrier(feat):
                if not engine.working_memory.body_schema.is_barrier_permeable(feat):
                    return False
            if (
                bg_is_impassable_void
                and feat == bg
                and (kr, kc) not in engine.learned_goal_positions
            ):
                return False
            if feat in engine.hazard_tracker.known_lethal_features:
                return False
        return True

    @staticmethod
    def plan_epistemic_probe(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
    ) -> tuple[int, dict[str, Any] | None]:
        """Select an exploratory probe action to resolve epistemic uncertainty."""
        H, W = curr_grid.shape
        bg = engine.estimate_background(curr_grid)
        bg_is_impassable_void = (
            len(engine.symbolic_theory.walkable_features) > 0
            and bg not in engine.symbolic_theory.walkable_features
        )
        entities = engine.extract_entities(curr_grid, bg)
        engine.level_epistemic_probes += 1
        engine.total_epistemic_probes += 1

        # 1. Uncalibrated actions take absolute priority for motor grounding
        uncalibrated = [a for a in available_actions if not engine.is_action_calibrated(a)]
        untested_at_pos = [
            a for a in uncalibrated if (a, engine.avatar_pos) not in engine.tested_action_positions
        ]
        if untested_at_pos:
            chosen_action = untested_at_pos[0]
            engine.tested_action_positions.add((chosen_action, engine.avatar_pos))
            engine.tested_actions.add(chosen_action)

            action_data = None
            if engine.is_spatial_effector(chosen_action):
                action_data = engine.ground_effector_action(curr_grid, chosen_action)
            return chosen_action, action_data

        # 2. Dynamic Affordance Partitioning (Modality-Agnostic, No Hardcoded Action IDs)
        displacement_actions = [a for a in available_actions if engine.is_displacement_action(a)]
        spatial_effector_actions = [a for a in available_actions if engine.is_spatial_effector(a)]
        discrete_transform_actions = [
            a
            for a in available_actions
            if a not in displacement_actions and a not in spatial_effector_actions
        ]

        # 2b. Motor babbling for self-identification. Agency is established by
        # observing which entity co-varies with DIFFERENT self-generated actions,
        # so vary actions round-robin until the controllability test commits.
        if displacement_actions and not engine.avatar_features and engine.avatar_feature is None:
            babble_idx = engine.level_epistemic_probes % len(displacement_actions)
            return displacement_actions[babble_idx], None

        # A. Non-displacement or Spatial Effector Environments
        if spatial_effector_actions:
            # Parietal Affordance Competition (Cisek's Hypothesis):
            # When manual reach effectors (e.g. CLICK_CELL) are available, evaluate whether manual reach
            # should gate over locomotion.
            structural_goals = PerceptionEngine.detect_structural_goals(
                curr_grid,
                bg=engine.estimate_background(curr_grid),
                avatar_features=engine.avatar_features,
                avatar_feature=engine.avatar_feature,
            )
            has_optical_targets = any(
                g.get("type")
                in ("relational_alignment", "optical_mirror_target", "reflection_target")
                for g in structural_goals
            )
            is_stuck_or_looping = engine.consecutive_stuck_steps >= 2 or (
                engine.avatar_pos is not None
                and engine.recent_positions.count(engine.avatar_pos) >= 2
            )
            has_effective_target = any(
                p not in engine.quiescent_click_targets
                and engine.entity_visit_counts.get(f"click_{p[0]}_{p[1]}", 0) < 3
                for p in engine.effective_click_targets
            )
            interleaved_reach = engine.level_epistemic_probes % 4 == 0

            should_probe_effector = (
                not displacement_actions
                or (engine.avatar_pos is None and engine.level_epistemic_probes > 4)
                or is_stuck_or_looping
                or has_effective_target
                or interleaved_reach
                or has_optical_targets
            )
            if should_probe_effector:
                chosen_eff = spatial_effector_actions[0]
                action_data = engine.ground_effector_action(curr_grid, chosen_eff)
                if action_data is not None:
                    return chosen_eff, action_data

        # 3. Spatial Movement Curiosity with Target Commitment & Loop Breaking
        if engine.avatar_pos is not None:
            engine.recent_positions.append(engine.avatar_pos)
            engine.position_visit_counts[engine.avatar_pos] = (
                engine.position_visit_counts.get(engine.avatar_pos, 0) + 1
            )

            for e in entities:
                if (
                    abs(e.grid_pos[0] - engine.avatar_pos[0])
                    + abs(e.grid_pos[1] - engine.avatar_pos[1])
                    <= 1
                ):
                    engine.probed_entity_ids.add(e.id)

            if engine.active_probe_target is not None:
                dist_to_target = abs(engine.active_probe_target[0] - engine.avatar_pos[0]) + abs(
                    engine.active_probe_target[1] - engine.avatar_pos[1]
                )
                engine.probe_target_steps += 1
                if dist_to_target <= 1 or engine.probe_target_steps > 12:
                    if engine.active_probe_id:
                        engine.probed_entity_ids.add(engine.active_probe_id)
                    engine.active_probe_target = None
                    engine.active_probe_id = None
                    engine.probe_target_steps = 0

            # Oscillation & Attractor Loop Breaking
            is_stuck = engine.consecutive_stuck_steps >= 2
            is_oscillating = engine.recent_positions.count(engine.avatar_pos) >= 3 or is_stuck
            if is_oscillating:
                logger.debug(
                    "AutonomousEpistemicEngine: Attractor loop/stuck state detected at %s! Breaking loop...",
                    engine.avatar_pos,
                )
                engine.active_probe_target = None
                engine.active_probe_id = None
                uncalibrated = [
                    a
                    for a in available_actions
                    if a not in engine.action_dynamics
                    or not engine.action_dynamics[a].is_displacement_action()
                ]
                untested_here = [
                    a
                    for a in uncalibrated
                    if (a, engine.avatar_pos) not in engine.tested_action_positions
                ]
                if untested_here:
                    act = untested_here[0]
                    engine.tested_action_positions.add((act, engine.avatar_pos))
                    return act, None
                if discrete_transform_actions and (is_stuck or is_oscillating):
                    disc_act = discrete_transform_actions[
                        engine.level_epistemic_probes % len(discrete_transform_actions)
                    ]
                    return disc_act, None
                best_act = None
                min_visits = float("inf")
                disp_actions = [
                    (act, engine.action_dynamics[act].get_displacement())
                    for act in available_actions
                    if act in engine.action_dynamics
                    and engine.action_dynamics[act].is_displacement_action()
                ]
                for act, (dr_cal, dc_cal) in disp_actions:
                    if engine.inhibited_actions.get(act, 0) > 0:
                        continue
                    if (engine.avatar_pos, act) in engine.failed_transitions:
                        continue
                    nr = engine.avatar_pos[0] + dr_cal
                    nc = engine.avatar_pos[1] + dc_cal
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if (
                        (nr, nc) in engine.hazard_tracker.static_lethal_positions
                        or (nr, nc) in engine.learned_barriers
                        or engine.symbolic_theory.is_barrier(int(curr_grid[nr, nc]))
                    ):
                        continue
                    # Known-lethal features are hard-blocked; unverified features
                    # carry an uncertainty cost (caution, not prohibition).
                    _osc_feat = int(curr_grid[nr, nc])
                    if _osc_feat in engine.hazard_tracker.known_lethal_features:
                        continue
                    uncertainty_pen = engine._epistemic_uncertainty_cost(_osc_feat, bg)
                    visits = engine.position_visit_counts.get((nr, nc), 0)
                    recency_pen = 20.0 if (nr, nc) in list(engine.recent_positions)[-6:] else 0.0
                    total_score = visits + recency_pen + uncertainty_pen
                    if total_score < min_visits:
                        min_visits = total_score
                        best_act = act
                if best_act is not None:
                    return best_act, None
                if spatial_effector_actions and min_visits == float("inf"):
                    chosen_eff = spatial_effector_actions[0]
                    return chosen_eff, engine.ground_effector_action(curr_grid, chosen_eff)
                return available_actions[0], None

            # 3a. Update room topology and extract topological subgoals
            engine.update_room_topology(curr_grid)

            candidate_entities = [
                e
                for e in entities
                if e.id not in engine.probed_entity_ids
                and not engine.symbolic_theory.is_barrier(e.feature_id)
                and e.feature_id not in engine.hazard_tracker.known_lethal_features
                and (
                    abs(e.grid_pos[0] - engine.avatar_pos[0])
                    + abs(e.grid_pos[1] - engine.avatar_pos[1])
                )
                > 1
            ]

            if engine.active_probe_target is None and candidate_entities:
                scored: list[tuple[SpatialEntity, float]] = []
                current_room_cells = (
                    set(engine.topology_rooms.get(engine.current_room_id, []))
                    if engine.current_room_id is not None
                    else set()
                )
                for e in candidate_entities:
                    dist = abs(e.grid_pos[0] - engine.avatar_pos[0]) + abs(
                        e.grid_pos[1] - engine.avatar_pos[1]
                    )
                    visits = engine.entity_visit_counts.get(e.id, 0)
                    info_val = 10.0 / (visits + 1.0)
                    if current_room_cells and e.grid_pos in current_room_cells:
                        info_val *= 1.5  # Room-locality preference
                    cost = dist + 1.0
                    scored.append((e, info_val / cost))
                scored.sort(key=lambda x: x[1], reverse=True)
                chosen = scored[0][0]
                engine.active_probe_target = chosen.grid_pos
                engine.active_probe_id = chosen.id
                engine.probe_target_steps = 0
                engine.entity_visit_counts[chosen.id] = (
                    engine.entity_visit_counts.get(chosen.id, 0) + 1
                )

            # 3b. Room Topology Doorway Macro-Subgoal Selection:
            # If no probe target is active, and the avatar is in a partitioned room,
            # select connecting doorway leading to the least-explored adjacent chamber!
            if (
                engine.active_probe_target is None
                and engine.topology_doors
                and engine.current_room_id is not None
            ):
                connecting_doors: list[tuple[RoomDoor, int, float]] = []
                for door in engine.topology_doors:
                    if engine.current_room_id in door.connects_rooms:
                        nbr_rooms = [r for r in door.connects_rooms if r != engine.current_room_id]
                        if not nbr_rooms:
                            continue
                        nbr_room = nbr_rooms[0]
                        nbr_cells = engine.topology_rooms.get(nbr_room, [])
                        unvisited_count = sum(
                            1
                            for cell in nbr_cells
                            if engine.position_visit_counts.get(cell, 0) == 0
                        )
                        dist_to_door = abs(door.door_coord[0] - engine.avatar_pos[0]) + abs(
                            door.door_coord[1] - engine.avatar_pos[1]
                        )
                        door_score = float(unvisited_count) / (dist_to_door + 1.0)
                        connecting_doors.append((door, nbr_room, door_score))

                if connecting_doors:
                    connecting_doors.sort(key=lambda item: item[2], reverse=True)
                    best_door, target_room, score = connecting_doors[0]
                    if score > 0.1:
                        engine.active_probe_target = best_door.door_coord
                        engine.active_probe_id = (
                            f"door_{best_door.door_coord[0]}_{best_door.door_coord[1]}"
                        )
                        engine.probe_target_steps = 0
                        engine.working_memory.register_topological_doorway(
                            best_door.door_coord, target_room
                        )

            target_pos = engine.active_probe_target or (H // 2, W // 2)
            tr, tc = target_pos

            # 3c. Free Energy Active Inference Action Selection
            disp_actions = [
                (act, engine.action_dynamics[act].get_displacement())
                for act in displacement_actions
                if act in engine.action_dynamics
                and hasattr(engine.action_dynamics[act], "get_displacement")
            ]
            if not disp_actions and displacement_actions:
                uncalibrated = [
                    act for act in displacement_actions if act not in engine.action_dynamics
                ]
                if uncalibrated:
                    return uncalibrated[0], None

            # Premotor Looming Hazard Reflex: If the avatar's current tile is on an impending collision course
            if engine.avatar_pos is not None:
                ar, ac = engine.avatar_pos
                is_looming_threat = any(
                    engine.collision_cones.is_collision_hazard(ar, ac, t_future)
                    for t_future in (1, 2)
                )
                if is_looming_threat:
                    for act, (dr_cal, dc_cal) in disp_actions:
                        nr, nc = ar + dr_cal, ac + dc_cal
                        if not EpistemicCuriosityExplorer._is_ray_passable(
                            engine, curr_grid, ar, ac, dr_cal, dc_cal, bg, bg_is_impassable_void
                        ):
                            continue
                        if any(
                            engine.collision_cones.is_collision_hazard(nr, nc, t_f)
                            for t_f in (1, 2)
                        ):
                            continue
                        if (engine.avatar_pos, act) in engine.failed_transitions:
                            continue
                        # Safe evasion found!
                        return act, None

            # 3c. Free Energy Active Inference Action Selection
            best_action = None
            candidate_action_nodes: list[ActionNode] = []
            info_gain_map: dict[str, float] = {}
            node_to_act_map: dict[str, int] = {}

            for act, (dr_cal, dc_cal) in disp_actions:
                if (engine.avatar_pos, act) in engine.failed_transitions:
                    continue
                if engine.inhibited_actions.get(act, 0) > 0:
                    continue
                ar, ac = engine.avatar_pos
                nr = ar + dr_cal
                nc = ac + dc_cal
                if not EpistemicCuriosityExplorer._is_ray_passable(
                    engine, curr_grid, ar, ac, dr_cal, dc_cal, bg, bg_is_impassable_void
                ):
                    continue

                # Premotor Collision Cones: Avoid stepping into oncoming kinetic hazards
                if any(engine.collision_cones.is_collision_hazard(nr, nc, t_f) for t_f in (1, 2)):
                    continue

                cell_feat = int(curr_grid[nr, nc])

                # OFC Deadlock Pruning: If stepping onto a cargo block pushes it into an irreversible deadlock, prune!
                if cell_feat in engine.learned_cargo_features:
                    pushed_r, pushed_c = nr + dr_cal, nc + dc_cal
                    dl_res = CounterfactualDeadlockDetector.evaluate_deadlock(
                        (pushed_r, pushed_c),
                        set(engine.learned_goal_positions),
                        engine.learned_barriers,
                        {(pushed_r, pushed_c)},
                        (H, W),
                    )
                    if dl_res.is_deadlock:
                        continue

                # Epistemic information gain
                visit_count = engine.position_visit_counts.get((nr, nc), 0)
                info_gain = 1.0 / (1.0 + float(visit_count))
                if (nr, nc) == (tr, tc):
                    info_gain += 0.50

                # Risk factor (uncertainty & danger)
                uncertainty_pen = (
                    0.0
                    if (nr, nc) == (tr, tc)
                    else engine._epistemic_uncertainty_cost(cell_feat, bg)
                )
                risk_factor = min(1.0, uncertainty_pen / 40.0)

                # Estimated cost (progress to probe target + recency penalty)
                dist_to_target = abs(tr - nr) + abs(tc - nc)
                recency_penalty = 10.0 if (nr, nc) in list(engine.recent_positions)[-4:] else 0.0
                estimated_cost = float(dist_to_target) + recency_penalty

                act_id = f"act_{act}_{nr}_{nc}"
                node = ActionNode(
                    id=act_id,
                    intent=f"move_{dr_cal}_{dc_cal}",
                    risk_factor=risk_factor,
                    estimated_cost=int(round(estimated_cost)),
                )
                candidate_action_nodes.append(node)
                info_gain_map[act_id] = info_gain
                node_to_act_map[act_id] = act

            if candidate_action_nodes:
                eval_results = engine.active_inference.evaluate_candidates(
                    candidate_action_nodes,
                    information_gain_map=info_gain_map,
                )
                if eval_results:
                    best_action = node_to_act_map[eval_results[0].action.id]

            if best_action is not None:
                return best_action, None

            # Metacognitive Refractory Relaxation: if uninhibited actions were all blocked, try candidate with lowest inhibition
            if best_action is None:
                sorted_disp = sorted(
                    disp_actions, key=lambda x: engine.inhibited_actions.get(x[0], 0)
                )
                ar, ac = engine.avatar_pos
                for act, (dr_cal, dc_cal) in sorted_disp:
                    if (engine.avatar_pos, act) in engine.failed_transitions:
                        continue
                    nr = ar + dr_cal
                    nc = ac + dc_cal
                    if not EpistemicCuriosityExplorer._is_ray_passable(
                        engine, curr_grid, ar, ac, dr_cal, dc_cal, bg, bg_is_impassable_void
                    ):
                        continue
                    if any(
                        engine.collision_cones.is_collision_hazard(nr, nc, t_f) for t_f in (1, 2)
                    ):
                        continue
                    best_action = act
                    break

            if best_action is not None:
                return best_action, None

            # Frontier Backtracking: find shortest path to nearest unexhausted walkable cell / junction
            if disp_actions and engine.avatar_pos is not None:
                bfs_q: deque[tuple[tuple[int, int], list[Any]]] = deque([(engine.avatar_pos, [])])
                bfs_visited = {engine.avatar_pos}
                best_frontier_path: list[Any] | None = None
                least_visited_path: list[Any] | None = None
                least_visits = float("inf")

                while bfs_q and len(bfs_visited) < 300:
                    p, path = bfs_q.popleft()
                    p_visits = engine.position_visit_counts.get(p, 0)
                    if p != engine.avatar_pos and p_visits < least_visits:
                        least_visits = p_visits
                        least_visited_path = path

                    has_unvisited_neighbor = False
                    for act, (dr_cal, dc_cal) in disp_actions:
                        nbr = (p[0] + dr_cal, p[1] + dc_cal)
                        if EpistemicCuriosityExplorer._is_ray_passable(
                            engine, curr_grid, p[0], p[1], dr_cal, dc_cal, bg, bg_is_impassable_void
                        ):
                            if engine.position_visit_counts.get(nbr, 0) == 0:
                                has_unvisited_neighbor = True
                                break
                    if has_unvisited_neighbor and path:
                        best_frontier_path = path
                        break

                    for act, (dr_cal, dc_cal) in disp_actions:
                        nr = p[0] + dr_cal
                        nc = p[1] + dc_cal
                        if (nr, nc) in bfs_visited:
                            continue
                        if not EpistemicCuriosityExplorer._is_ray_passable(
                            engine, curr_grid, p[0], p[1], dr_cal, dc_cal, bg, bg_is_impassable_void
                        ):
                            continue
                        step_idx = len(path) + 1
                        if engine.collision_cones.is_collision_hazard(nr, nc, step_idx):
                            continue
                        if p == engine.avatar_pos and (p, act) in engine.failed_transitions:
                            continue
                        bfs_visited.add((nr, nc))
                        bfs_q.append(((nr, nc), path + [act]))

                chosen_escape = best_frontier_path or least_visited_path
                if chosen_escape:
                    return chosen_escape[0], None

            if spatial_effector_actions:
                chosen_eff = spatial_effector_actions[0]
                return chosen_eff, engine.ground_effector_action(curr_grid, chosen_eff)
            elif discrete_transform_actions:
                return discrete_transform_actions[
                    engine.step_counter % len(discrete_transform_actions)
                ], None
            uncalibrated = [a for a in available_actions if not engine.is_displacement_action(a)]
            untested_here = [
                a
                for a in uncalibrated
                if (a, engine.avatar_pos) not in engine.tested_action_positions
            ]
            if untested_here:
                act = untested_here[0]
                engine.tested_action_positions.add((act, engine.avatar_pos))
                return act, None
            return available_actions[engine.step_counter % len(available_actions)], None

        # 4. Fallback probe
        if spatial_effector_actions:
            chosen_eff = spatial_effector_actions[0]
            return chosen_eff, engine.ground_effector_action(curr_grid, chosen_eff)
        return available_actions[0], None
