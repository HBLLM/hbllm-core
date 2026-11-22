"""Object-Centric Macro-Action State Graph Planner."""

from __future__ import annotations

import heapq
import logging
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole
from hbllm.hcir.world.engine_state import get_engine_state_attr
from hbllm.hcir.world.objects.ledger import ObjectStateLedger
from hbllm.hcir.world.objects.models import (
    MacroOption,
    ObjectCategory,
    SalientObject,
)
from hbllm.hcir.world.objects.relational import RelationalGraphMatcher
from hbllm.hcir.world.objects.risk import estimate_entity_risk
from hbllm.hcir.world.percept import PerceptionFrame, PerceptionRegistry

logger = logging.getLogger(__name__)


class ObjectStateGraphPlanner:
    """Domain-general Object-Centric Macro-Action State Graph Planner.

    Operates at two distinct levels of abstraction:
    1. Discrete Macro-Option Selection: chooses among K salient objects based on
       the 5 executive cognitive directives.
    2. Continuous Path Controller: finds optimal collision-free primitive action
       sequences (distance D) to reach the chosen object's interaction stance.
    """

    def __init__(self) -> None:
        self.ledger: ObjectStateLedger = ObjectStateLedger()
        self.current_macro_option: MacroOption | None = None
        self.active_target_object_id: str | None = None
        self.active_target_feature_id: int | None = None
        self.relational_matcher: RelationalGraphMatcher = RelationalGraphMatcher()

    def on_boundary(self, boundary) -> None:
        """Episode lifecycle boundary handler (WP0.5). Delegates to the
        deprecated reset_episode for backward-compatible semantics."""
        self.reset_episode()

    @property
    def boundary_retention(self):
        """Declared retention policy: default episode-local semantics."""
        from hbllm.hcir.world.percept import RetentionPolicy

        return RetentionPolicy.EPISODE_LOCAL

    def reset_episode(
        self, is_new_context: bool | None = None, *, is_new_level: bool | None = None
    ) -> None:
        """Reset planner state for a new context or trial."""
        new_ctx = (
            is_new_context
            if is_new_context is not None
            else (is_new_level if is_new_level is not None else False)
        )
        if new_ctx:
            self.ledger.reset_context_epoch()
        else:
            # On same-context retry: decay interaction counts so heavily-probed objects
            # have their IOR decayed, but unprobed candidates take priority over repeating initial failures.
            for obj_id in list(self.ledger.interaction_counts.keys()):
                self.ledger.interaction_counts[obj_id] = self.ledger.interaction_counts[obj_id] // 2
        self.current_macro_option = None
        self.active_target_object_id = None
        self.active_target_feature_id = None

    def lift_salient_objects(
        self,
        percept: PerceptionFrame,
        engine: Any,
    ) -> list[SalientObject]:
        """Lift K salient objects directly from an abstract PerceptionFrame without raw pixel inspection."""
        H, W = percept.shape
        av_pos = percept.avatar_pos or engine.avatar_pos
        av_feats = (
            percept.avatar_features
            or engine.avatar_features
            or ({engine.avatar_feature} if engine.avatar_feature is not None else set())
        )
        barrier_features = engine.symbolic_theory.barrier_features
        lethal_features = set(engine.hazard_tracker.known_lethal_features)
        known_goal_positions = set(percept.candidate_goals)
        known_goal_features = set(engine.symbolic_theory.goal_features)

        salient_objects: list[SalientObject] = []
        grid_arr = percept.metadata.get("grid")
        bg_val = int(percept.metadata.get("bg", engine.bg_feature or 0))
        for e in percept.entities:
            # Skip avatar (rely on EntityRole.AGENT lifted by perception so disjoint
            # non-avatar objects sharing a palette color with the avatar remain visible)
            if e.role == EntityRole.AGENT or (av_pos is None and e.feature_id in av_feats):
                continue
            if grid_arr is not None and engine.working_memory.is_exogenous_locus(
                e.grid_pos, lattice=grid_arr, bg=bg_val, include_outer_edge=False
            ):
                continue
            # Skip massive background or bulk perimeter walls
            if e.area > H * W * 0.30:
                continue
            # Skip pure lethal hazards unless small interactive entity
            if (
                e.feature_id in lethal_features or e.role == EntityRole.DYNAMIC_HAZARD
            ) and e.area > 9:
                continue
            # Skip known walkable floor substrates unless they carry an interactive/goal role
            if (
                engine.symbolic_theory.is_walkable(e.feature_id)
                and e.feature_id not in known_goal_features
                and not engine.symbolic_theory.is_goal(e)
                and e.feature_id not in engine.symbolic_theory.actuator_features
                and e.feature_id not in engine.symbolic_theory.cargo_features
                and e.role not in (EntityRole.GOAL, EntityRole.ACTUATOR, EntityRole.MANIPULABLE)
            ):
                continue

            entity_cells = set(e.properties.get("cells", [e.grid_pos]))

            # Categorize object based on role, symbolic theory, and known goals
            is_barrier = (
                e.role == EntityRole.OBSTACLE
                or engine.symbolic_theory.is_barrier(e.feature_id)
                or e.feature_id in barrier_features
            )
            is_goal = not is_barrier and (
                e.role == EntityRole.GOAL
                or engine.symbolic_theory.is_goal(e)
                or e.feature_id in known_goal_features
                or any(cp in known_goal_positions for cp in entity_cells)
            )
            if not is_goal and all(
                engine.symbolic_theory.is_locus_explored(cp)
                or cp in engine.exhausted_candidate_goals
                for cp in entity_cells
            ):
                continue
            is_switch = (
                (
                    bool(engine.symbolic_theory)
                    and e.feature_id in engine.symbolic_theory.actuator_features
                )
                or e.role == EntityRole.ACTUATOR
                or any(
                    m.trigger_pos in entity_cells
                    or (m.trigger_feature is not None and m.trigger_feature == e.feature_id)
                    for m in engine.state_mutations
                )
            )
            is_cargo = e.role == EntityRole.MANIPULABLE or (
                bool(engine.symbolic_theory)
                and (
                    e.feature_id in engine.symbolic_theory.cargo_features
                    or engine.symbolic_theory.is_cargo(e.feature_id)
                )
            )

            if is_goal:
                category = ObjectCategory.EXIT
                is_solid = False
            elif is_switch:
                category = ObjectCategory.SWITCH
                is_solid = is_barrier
            elif is_barrier:
                category = ObjectCategory.BARRIER
                is_solid = True
            elif is_cargo:
                category = ObjectCategory.MANIPULABLE
                is_solid = True
            elif e.area <= 16:
                category = ObjectCategory.KEY
                is_solid = is_barrier
            else:
                category = ObjectCategory.UNKNOWN
                is_solid = e.role == EntityRole.OBSTACLE

            # Compute interaction stances using percept.is_passable
            confirmed_goal = not is_barrier and (
                e.feature_id in known_goal_features
                or any(cp in engine.learned_goal_positions for cp in entity_cells)
                or (
                    is_goal
                    and not engine.is_feature_unverified(e.feature_id or 0, engine.bg_feature)
                )
            )
            stances: set[tuple[int, int]] = set()
            adj_stances: set[tuple[int, int]] = set()
            if category == ObjectCategory.EXIT or not is_solid:
                for cr, cc in entity_cells:
                    if percept.is_passable(cr, cc):
                        stances.add((cr, cc))

            for cr, cc in entity_cells:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ar, ac = cr + dr, cc + dc
                    if percept.is_passable(ar, ac) and (ar, ac) not in entity_cells:
                        adj_stances.add((ar, ac))
            if is_solid or not stances:
                stances.update(adj_stances)

            if not stances:
                stances = {e.grid_pos}

            centroid = (int(round(e.centroid[0])), int(round(e.centroid[1])))
            anchor = (
                min(
                    entity_cells,
                    key=lambda p: (p[0] - centroid[0]) ** 2 + (p[1] - centroid[1]) ** 2,
                )
                if entity_cells
                else centroid
            )
            salience = float(
                e.properties.get(
                    "salience_score",
                    100.0 / math.log2(2 + max(1, e.area)),
                )
            )
            obj = SalientObject(
                object_id=e.id,
                category=category,
                feature_id=e.feature_id if e.feature_id is not None else 0,
                centroid=centroid,
                cells=entity_cells,
                area=e.area,
                is_solid=is_solid,
                interaction_stances=sorted(list(stances)),
                anchor_coord=anchor,
                salience_score=salience,
                properties={
                    "role": str(e.role),
                    "adjacent_stances": sorted(adj_stances),
                    "confirmed_goal": confirmed_goal,
                    **dict(e.properties),
                },
            )
            salient_objects.append(obj)

        return salient_objects

    def extract_salient_objects(
        self,
        engine: Any,
        curr_grid: np.ndarray,
        bg: int,
    ) -> list[SalientObject]:
        """Extract and categorize K salient objects via domain perception adapter."""
        if isinstance(curr_grid, PerceptionFrame):
            percept = curr_grid
        elif hasattr(engine, "perceive") and callable(engine.perceive):
            percept = engine.perceive(curr_grid)
        else:
            adapter = PerceptionRegistry.get_default()
            if adapter is not None:
                percept = adapter.lift_percept(curr_grid, context=engine)
            else:
                if not (hasattr(curr_grid, "shape") and len(curr_grid.shape) >= 2):
                    raise ValueError("lattice extent unknown")
                H, W = curr_grid.shape[0], curr_grid.shape[1]
                percept = PerceptionFrame(shape=(H, W))
        return self.lift_salient_objects(percept, engine)

    def find_shortest_path(
        self,
        start_pos: tuple[int, int],
        target_stances: set[tuple[int, int]],
        static_barriers: set[tuple[int, int]],
        lethal_features: set[int],
        movable_actions: Sequence[tuple[Any, int, int]],
        grid_shape: tuple[int, int],
        curr_grid: np.ndarray | None = None,
        patroller_cells: set[tuple[int, int]] | None = None,
        failed_transitions: set[tuple[tuple[int, int], Any]] | None = None,
        percept: PerceptionFrame | None = None,
        engine: Any | None = None,
    ) -> list[Any] | None:
        """Find the lowest-risk deterministic path of primitive actions to any target stance.

        Strongly prefers routing around intermediate unverified or risky features whenever
        a safe walkable/background detour exists, while retaining completeness if crossing
        an unverified tile is unavoidable.
        """
        if start_pos in target_stances:
            return []

        H, W = grid_shape
        from hbllm.hcir.world.autonomous_epistemic_engine import MentalSimulationStep

        bg_val = int(engine.bg_feature) if engine is not None else 0
        max_step_span = max((abs(dr) + abs(dc) for _, dr, dc in movable_actions), default=1) or 1
        target_list = list(target_stances)
        seq = 0
        h0 = (
            min(abs(start_pos[0] - tr) + abs(start_pos[1] - tc) for tr, tc in target_list)
            / max_step_span
        )
        heap: list[tuple[float, float, int, tuple[int, int]]] = [(h0, 0.0, seq, start_pos)]
        best_cost: dict[tuple[int, int], float] = {start_pos: 0.0}
        came_from: dict[tuple[int, int], tuple[tuple[int, int], Any]] = {}
        verified_cache: dict[int, bool] = {bg_val: True}
        unverified_penalty_cache: dict[int, float] = {}

        _budget = engine.compute_budget if engine is not None else None
        while heap:
            if _budget is not None and not _budget.allows("cheap"):
                return None  # anytime: expansion budget exhausted
            if _budget is not None:
                _budget.charge(1)
            _, cur_cost, _, cur_pos = heapq.heappop(heap)
            if cur_cost > best_cost.get(cur_pos, float("inf")):
                continue
            if cur_pos in target_stances:
                win_path: list[MentalSimulationStep] = []
                node_cur = cur_pos
                while node_cur in came_from:
                    prev_p, prev_act = came_from[node_cur]
                    win_path.append(
                        MentalSimulationStep(action=prev_act, predicted_avatar_pos=node_cur)
                    )
                    node_cur = prev_p
                win_path.reverse()
                return win_path

            cr, cc = cur_pos
            for act, dr, dc in movable_actions:
                if failed_transitions and (cur_pos, act) in failed_transitions:
                    continue
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue

                step_r = 1 if dr > 0 else (-1 if dr < 0 else 0)
                step_c = 1 if dc > 0 else (-1 if dc < 0 else 0)
                dist = max(abs(dr), abs(dc))
                blocked = False
                step_cost = 1.0

                for k in range(1, dist + 1):
                    kr = cr + step_r * k
                    kc = cc + step_c * k
                    k_pos = (kr, kc)
                    is_target = k_pos in target_stances and (nr, nc) in target_stances
                    if not is_target:
                        if k_pos in static_barriers:
                            blocked = True
                            break
                        if percept is not None and not percept.is_passable(kr, kc):
                            blocked = True
                            break
                    if curr_grid is not None:
                        val = int(curr_grid[kr, kc])
                        if not is_target and val in lethal_features:
                            blocked = True
                            break
                        if not is_target and engine is not None:
                            is_ver = verified_cache.get(val)
                            if is_ver is None:
                                is_ver = not engine.is_feature_unverified(val, bg_val)
                                verified_cache[val] = is_ver
                            if not is_ver:
                                cached_pen = unverified_penalty_cache.get(val)
                                if cached_pen is None:
                                    risk = estimate_entity_risk(
                                        engine, val, {k_pos}, curr_grid, bg_val, ledger=self.ledger
                                    )
                                    cached_pen = engine.UNVERIFIED_FEATURE_COST * 4.0 + risk * 50.0
                                    if len(self.ledger.object_causal_effects) == 0:
                                        unverified_penalty_cache[val] = cached_pen
                                step_cost += cached_pen

                if blocked:
                    continue
                nxt = (nr, nc)
                if patroller_cells and nxt in patroller_cells:
                    continue

                next_cost = cur_cost + step_cost
                if next_cost < best_cost.get(nxt, float("inf")):
                    best_cost[nxt] = next_cost
                    came_from[nxt] = (cur_pos, act)
                    h_nxt = (
                        min(abs(nr - tr) + abs(nc - tc) for tr, tc in target_list) / max_step_span
                    )
                    seq += 1
                    heapq.heappush(heap, (next_cost + h_nxt, next_cost, seq, nxt))

        return None

    def plan_macro_option(
        self,
        engine: Any,
        curr_grid: np.ndarray | PerceptionFrame,
        available_actions: Sequence[Any],
    ) -> list[Any] | None:
        """Synthesize a complete macro-option plan adhering to the 5 executive cognitive directives.
        Supports continuous spatial displacement paths (BFS/A*) and allocentric spatial effector (point actuation) actions.
        """
        if not available_actions:
            return None
        if curr_grid is None:
            return None  # lattice extent unknown — no plan on a guessed shape

        # 1. Resolve PerceptionFrame from domain sensory adapter
        percept: PerceptionFrame
        grid_arr: np.ndarray | None
        if isinstance(curr_grid, PerceptionFrame):
            percept = curr_grid
            grid_arr = percept.metadata.get("grid")
        elif hasattr(engine, "perceive") and callable(engine.perceive):
            percept = engine.perceive(curr_grid)
            grid_arr = curr_grid
        else:
            adapter = PerceptionRegistry.get_default()
            if adapter is not None:
                percept = adapter.lift_percept(curr_grid, context=engine)
            else:
                if not (hasattr(curr_grid, "shape") and len(curr_grid.shape) >= 2):
                    return None  # lattice extent unknown — no plan on a guessed shape
                H, W = curr_grid.shape[0], curr_grid.shape[1]
                percept = PerceptionFrame(shape=(H, W))
            grid_arr = curr_grid

        H, W = percept.shape
        av_pos = percept.avatar_pos or engine.avatar_pos

        theory = engine.symbolic_theory
        bg_val = int(engine.bg_feature)
        self.ledger.observe_grid_connectivity(
            grid_arr,
            av_pos,
            bg_val,
            set(theory.walkable_features),
            set(engine.avatar_features)
            | ({engine.avatar_feature} if engine.avatar_feature is not None else set()),
        )

        # Extract calibrated movable actions and spatial effector actions
        movable_actions: list[tuple[Any, int, int]] = []
        spatial_effector_actions: list[Any] = []
        action_deltas = engine.action_deltas
        for act in available_actions:
            if (
                get_engine_state_attr(engine, "inhibited_actions", None)
                and engine.inhibited_actions.get(act, 0) > 0
            ):
                continue
            if engine.is_spatial_effector(act):
                spatial_effector_actions.append(act)
                continue
            if (
                get_engine_state_attr(engine, "immobile_entity_actions", None)
                and act in engine.immobile_entity_actions
            ):
                continue
            if (
                get_engine_state_attr(engine, "inhibited_actions", None)
                and engine.inhibited_actions.get(act, 0) > 0
            ):
                continue
            if act in action_deltas:
                dr, dc = action_deltas[act]
                movable_actions.append((act, dr, dc))

        if not movable_actions and not spatial_effector_actions:
            return None

        # Static barriers and hazards directly from percept
        static_barriers = set(percept.static_barriers)
        barrier_features = engine.symbolic_theory.barrier_features
        lethal_features = set(engine.hazard_tracker.known_lethal_features)
        failed_transitions = set(get_engine_state_attr(engine, "failed_transitions", set()))

        # Detect patroller / dynamic threat positions from abstract perceptual entities
        patroller_cells: set[tuple[int, int]] = set()
        if engine.mobile_threat_features:
            for e in percept.entities:
                if e.role == EntityRole.DYNAMIC_HAZARD or (
                    e.feature_id is not None and e.feature_id in engine.mobile_threat_features
                ):
                    patroller_cells.update(e.properties.get("cells", [e.grid_pos]))

        # Extract all salient objects from the lifted percept
        salient_objects = self.lift_salient_objects(percept, engine)

        # ── DIRECTIVE 1 & 2: Check if exit is open. If open, go through it. ───
        # Suppress bare-avatar exit walk when unconfirmed cargo must still be pushed onto a receptacle
        # (unless an exit feature is already confirmed in symbolic_theory.goal_features).
        known_cargo = set(theory.cargo_features) | set(engine.learned_cargo_features)
        known_goal_feats = set(theory.goal_features)
        has_cargo_on_grid = bool(known_cargo) and any(
            e.feature_id in known_cargo for e in percept.entities
        )
        has_real_play_history = bool(engine.tested_actions)
        discrete_interact_act = next(
            (
                a
                for a in available_actions
                if not engine.is_displacement_action(a)
                and not engine.is_spatial_effector(a)
                and engine.inhibited_actions.get(a, 0) <= 5
                and (
                    a in engine.action_dynamics
                    or a in engine.tested_actions
                    or has_real_play_history
                )
            ),
            None,
        )
        exit_objects = [o for o in salient_objects if o.category == ObjectCategory.EXIT]
        confirmed_exit_objects = [o for o in exit_objects if o.properties.get("confirmed_goal")]
        tentative_exit_objects = [o for o in exit_objects if not o.properties.get("confirmed_goal")]
        non_exit_base = [
            o
            for o in salient_objects
            if o.category != ObjectCategory.EXIT
            and o.category != ObjectCategory.BARRIER
            and not theory.is_barrier(o.feature_id)
            and o.feature_id not in barrier_features
        ]
        allow_unconfirmed_exit_shortcut = bool(confirmed_exit_objects) or (
            discrete_interact_act is None and len(tentative_exit_objects) <= 1 and not non_exit_base
        )
        has_confirmed_walk_exit = any(o.feature_id in known_goal_feats for o in exit_objects)
        exit_stances: set[tuple[int, int]] = set()
        if confirmed_exit_objects:
            for eo in confirmed_exit_objects:
                exit_stances.update(eo.interaction_stances)
        elif allow_unconfirmed_exit_shortcut:
            exit_stances.update(percept.candidate_goals)
            for eo in exit_objects:
                exit_stances.update(eo.interaction_stances)
        for cg in percept.candidate_goals:
            if (
                0 <= cg[0] < H
                and 0 <= cg[1] < W
                and grid_arr is not None
                and not engine.is_feature_unverified(int(grid_arr[cg[0], cg[1]]), bg_val)
            ):
                exit_stances.add(cg)

        def _filter_reversible(
            candidate_cells: set[tuple[int, int]],
            stances: set[tuple[int, int]],
            barriers: set[tuple[int, int]],
        ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
            if grid_arr is None or theory is None or av_pos is None:
                return stances, barriers
            from hbllm.hcir.world.cortex_planner import EpistemicGoalScheduler

            bg_val = getattr(engine, "bg_feature", 0) or 0
            eff_stances = set(stances)
            eff_barriers = set(barriers)
            for ep in candidate_cells:
                if not (0 <= ep[0] < H and 0 <= ep[1] < W):
                    continue
                e_feat = int(grid_arr[ep[0], ep[1]])
                e_area = next(
                    (
                        int(e.area)
                        for e in percept.entities
                        if e.grid_pos == ep or ep in e.properties.get("cells", ())
                    ),
                    1,
                )
                safe_stances = EpistemicGoalScheduler.compute_reversible_approach_stances(
                    engine=engine,
                    curr_grid=grid_arr,
                    H=H,
                    W=W,
                    bg=bg_val,
                    avatar_pos=av_pos,
                    entity_pos=ep,
                    feature_id=e_feat,
                    entity_area=e_area,
                    has_cargo_context=has_cargo_on_grid,
                )
                if (
                    safe_stances
                    and any(not theory.is_locus_explored(sp) for sp in safe_stances)
                    and not self.ledger.is_in_newly_reachable({ep})
                ):
                    eff_stances.discard(ep)
                    for dr_x, dc_x in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        adj_x = (ep[0] + dr_x, ep[1] + dc_x)
                        if adj_x not in safe_stances:
                            eff_stances.discard(adj_x)
                    eff_stances.update(safe_stances)
                    eff_barriers.add(ep)
            return eff_stances, eff_barriers

        # 1A. Displacement Navigation to Exit
        if (
            exit_stances
            and movable_actions
            and av_pos is not None
            and (not has_cargo_on_grid or has_confirmed_walk_exit)
        ):
            eff_exit_stances, eff_exit_barriers = _filter_reversible(
                exit_stances, exit_stances, static_barriers
            )
            path_to_exit = self.find_shortest_path(
                start_pos=av_pos,
                target_stances=eff_exit_stances,
                static_barriers=eff_exit_barriers,
                lethal_features=lethal_features,
                movable_actions=movable_actions,
                grid_shape=(H, W),
                engine=engine,
                curr_grid=grid_arr,
                patroller_cells=patroller_cells,
                failed_transitions=failed_transitions,
                percept=percept,
            )
            if path_to_exit is not None and len(path_to_exit) > 0:
                logger.info(
                    "ObjectStateGraphPlanner: Directive 1 & 2 SATISFIED! Exit is OPEN. "
                    "Synthesized %d-step path directly to exit.",
                    len(path_to_exit),
                )
                self.active_target_object_id = (
                    exit_objects[0].object_id if exit_objects else "learned_exit"
                )
                self.active_target_feature_id = exit_objects[0].feature_id if exit_objects else None
                return path_to_exit

        # 1B. Spatial Effector Interaction with Exit
        if exit_objects and spatial_effector_actions and not movable_actions:
            eff_act = spatial_effector_actions[0]
            quiescent_coords = get_engine_state_attr(engine, "quiescent_actuation_targets", set())
            for target_obj in exit_objects:
                unprobed_exit_cells = [c for c in target_obj.cells if c not in quiescent_coords]
                if not unprobed_exit_cells:
                    continue
                target_cell = (
                    target_obj.anchor_coord
                    if target_obj.anchor_coord in unprobed_exit_cells
                    else unprobed_exit_cells[0]
                )
                tr, tc = target_cell
                from hbllm.hcir.world.cortex_planner import MentalSimulationStep

                self.active_target_object_id = target_obj.object_id
                return [
                    MentalSimulationStep(
                        action=eff_act,
                        action_data={"x": tc, "y": tr},
                        predicted_avatar_pos=av_pos or (tr, tc),
                        expected_mutation="ACTUATE_EXIT",
                    )
                ]

        # ── DIRECTIVE 3: If not, explore other objects. ──────────────────────
        non_exit_objects = list(non_exit_base)
        if not allow_unconfirmed_exit_shortcut:
            for eo in tentative_exit_objects:
                eo.properties["tentative_exit"] = True
                non_exit_objects.append(eo)

        # ── DIRECTIVE 4B: Active Causal Actuator Contingency Pursuit ──────────
        if spatial_effector_actions:
            eff_act = spatial_effector_actions[0]
            quiescent_coords = get_engine_state_attr(engine, "quiescent_actuation_targets", set())
            actuator_coord = get_engine_state_attr(engine, "active_causal_actuator_coord", None)
            streak = get_engine_state_attr(engine, "consecutive_causal_probes", 0)
            streak_limit = get_engine_state_attr(engine, "causal_actuator_streak_limit", 4)
            if (
                actuator_coord is not None
                and streak < streak_limit
                and actuator_coord not in quiescent_coords
            ):
                tr, tc = actuator_coord
                target_obj_id = f"actuator_{tr}_{tc}"
                for obj in non_exit_objects:
                    if actuator_coord in obj.cells:
                        target_obj_id = obj.object_id
                        break
                self.active_target_object_id = target_obj_id
                from hbllm.hcir.world.cortex_planner import MentalSimulationStep

                return [
                    MentalSimulationStep(
                        action=eff_act,
                        action_data={"x": tc, "y": tr},
                        predicted_avatar_pos=av_pos or (tr, tc),
                        expected_mutation="PURSUIT_CAUSAL_ACTUATOR",
                    )
                ]

        if not non_exit_objects:
            return None

        # 2A. Displacement Navigation to Objects
        if movable_actions and av_pos is not None:
            reachable_objects: list[tuple[SalientObject, list[Any]]] = []
            reversible_probe_ids: set[str] = set()
            reversibly_probed_ids: set[str] = set()
            obj_adj_map: dict[str, list[tuple[int, int]]] = {}
            for obj in non_exit_objects:
                if not obj.interaction_stances:
                    continue
                obj_stances, obj_barriers = _filter_reversible(
                    set(obj.cells) if len(obj.cells) <= 4 else set(),
                    set(obj.interaction_stances),
                    static_barriers,
                )
                adj_stances = [
                    tuple(c)
                    for c in obj.properties.get("adjacent_stances", ())
                    if tuple(c) not in static_barriers
                ]
                obj_adj_map[obj.object_id] = adj_stances
                is_reversibly_probed = self.ledger.is_interacted(obj.object_id) or any(
                    theory.is_locus_actuator_probed(adj) for adj in adj_stances
                )
                if is_reversibly_probed and obj.object_id not in self.ledger.object_causal_effects:
                    reversibly_probed_ids.add(obj.object_id)
                is_known_contact_trigger = any(
                    m.trigger_type == "CONTACT"
                    and (m.trigger_pos in obj.cells or m.trigger_feature == obj.feature_id)
                    for m in engine.state_mutations
                )
                is_unverified = engine.is_feature_unverified(obj.feature_id, bg_val)
                prefer_reversible_phase = (
                    discrete_interact_act is not None
                    and is_unverified
                    and not is_reversibly_probed
                    and not is_known_contact_trigger
                    and bool(adj_stances)
                )
                if prefer_reversible_phase:
                    path = self.find_shortest_path(
                        start_pos=av_pos,
                        target_stances=set(adj_stances),
                        static_barriers=set(obj_barriers) | set(obj.cells),
                        lethal_features=lethal_features,
                        movable_actions=movable_actions,
                        grid_shape=(H, W),
                        curr_grid=grid_arr,
                        patroller_cells=patroller_cells,
                        failed_transitions=failed_transitions,
                        percept=percept,
                        engine=engine,
                    )
                    if path is not None:
                        reversible_probe_ids.add(obj.object_id)
                        reachable_objects.append((obj, path))
                        continue
                path = self.find_shortest_path(
                    start_pos=av_pos,
                    target_stances=obj_stances,
                    static_barriers=obj_barriers,
                    lethal_features=lethal_features,
                    movable_actions=movable_actions,
                    grid_shape=(H, W),
                    curr_grid=grid_arr,
                    patroller_cells=patroller_cells,
                    failed_transitions=failed_transitions,
                    percept=percept,
                    engine=engine,
                )
                if path is not None:
                    reachable_objects.append((obj, path))

            if reachable_objects:
                # ── DIRECTIVE 4: If you know what they already do and those can unlock the door, use it. ──
                unlocking_candidates: list[tuple[SalientObject, list[Any]]] = []
                for obj, path in reachable_objects:
                    is_known_trigger = obj.object_id in self.ledger.unlocked_by_object or any(
                        m.trigger_pos in obj.cells
                        or (m.trigger_feature is not None and m.trigger_feature == obj.feature_id)
                        for m in engine.state_mutations
                    )
                    is_tool_affinity = (
                        obj.feature_id in engine.working_memory.tool_barrier_affinities
                        or engine.working_memory.is_holding(obj.feature_id)
                    )
                    if is_known_trigger or is_tool_affinity:
                        unlocking_candidates.append((obj, path))

                if unlocking_candidates:
                    unlocking_candidates.sort(key=lambda x: len(x[1]))
                    chosen_obj, chosen_path = unlocking_candidates[0]
                    logger.info(
                        "ObjectStateGraphPlanner: Directive 4 SATISFIED! Actuating known unlocking object "
                        "[%s, category=%s] (%d steps).",
                        chosen_obj.object_id,
                        chosen_obj.category.value,
                        len(chosen_path),
                    )
                    self.active_target_object_id = chosen_obj.object_id
                    self.active_target_feature_id = chosen_obj.feature_id
                    return self._finalize_option_path(
                        engine, chosen_obj, chosen_path, available_actions, movable_actions
                    )

                # Identify features of barriers blocking access or adjacent to exit via entity adjacency
                barrier_features = engine.symbolic_theory.barrier_features
                blocking_barrier_feats: set[int] = set()
                exit_adjacent_coords: set[tuple[int, int]] = {
                    (er + dr, ec + dc)
                    for eo in exit_objects
                    for er, ec in eo.cells
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                }
                for o in salient_objects:
                    if (
                        o.is_solid
                        or o.category == ObjectCategory.BARRIER
                        or o.feature_id in barrier_features
                        or engine.symbolic_theory.is_barrier(o.feature_id)
                    ):
                        if not o.cells.isdisjoint(exit_adjacent_coords):
                            blocking_barrier_feats.add(o.feature_id)

                # Step-budget and temporal horizon awareness
                remaining_budget = (
                    engine.get_remaining_step_budget()
                    if hasattr(engine, "get_remaining_step_budget")
                    else 1000
                )
                is_strict_budget = (
                    engine.is_under_strict_step_budget()
                    if hasattr(engine, "is_under_strict_step_budget")
                    else False
                )

                # ── DIRECTIVE 5: If not, play with it (interact) and see what is the outcome. ───
                scored_candidates: list[tuple[SalientObject, list[Any], float]] = []
                for obj, path in reachable_objects:
                    path_len = len(path)
                    # Hard-prune exploration candidates that exceed remaining attempt budget
                    if remaining_budget is not None and path_len >= remaining_budget:
                        continue

                    visit_count = self.ledger.get_interaction_count(obj.object_id)
                    untouched_bonus = 100.0 if visit_count == 0 else 0.0
                    recency_penalty = 15.0 * float(visit_count)
                    # Under strict step budget, prioritize nearby objects to unlock exits before timeout
                    distance_penalty = float(path_len) * (2.0 if is_strict_budget else 0.5)
                    small_item_bonus = 25.0 if obj.category == ObjectCategory.KEY else 0.0
                    color_affinity_bonus = 80.0 if obj.feature_id in blocking_barrier_feats else 0.0
                    adj_set = set(obj_adj_map.get(obj.object_id, ()))
                    in_unlocked = self.ledger.is_in_newly_reachable(
                        obj.cells
                    ) or self.ledger.is_in_newly_reachable(adj_set)
                    newly_unlocked_bonus = 250.0 if in_unlocked else 0.0
                    reversible_probe_bonus = 180.0 if obj.object_id in reversible_probe_ids else 0.0
                    tentative_exit_bonus = 35.0 if obj.properties.get("tentative_exit") else 0.0
                    is_inert = (
                        self.ledger.is_inert(obj.object_id)
                        or obj.object_id in reversibly_probed_ids
                    ) and not in_unlocked
                    inert_penalty = 120.0 if is_inert else 0.0
                    risk = estimate_entity_risk(
                        engine, obj.feature_id, obj.cells, grid_arr, bg_val, ledger=self.ledger
                    )
                    risk_penalty = risk * 220.0

                    score = (
                        untouched_bonus
                        + newly_unlocked_bonus
                        + reversible_probe_bonus
                        + tentative_exit_bonus
                        + small_item_bonus
                        + color_affinity_bonus
                        + obj.salience_score * 0.2
                        - recency_penalty
                        - distance_penalty
                        - inert_penalty
                        - risk_penalty
                    )
                    scored_candidates.append((obj, path, score))

                if not scored_candidates and reachable_objects:
                    # Fallback if all candidates exceeded remaining budget
                    scored_candidates = [
                        (obj, path, -float(len(path))) for obj, path in reachable_objects
                    ]

                scored_candidates.sort(key=lambda x: x[2], reverse=True)
                best_obj, best_path, best_score = scored_candidates[0]

                logger.info(
                    "ObjectStateGraphPlanner: Directive 5 SATISFIED! Epistemically probing object "
                    "[%s, category=%s, centroid=%s] (%d steps, score=%.1f).",
                    best_obj.object_id,
                    best_obj.category.value,
                    best_obj.centroid,
                    len(best_path),
                    best_score,
                )
                self.active_target_object_id = best_obj.object_id
                self.active_target_feature_id = best_obj.feature_id
                return self._finalize_option_path(
                    engine, best_obj, best_path, available_actions, movable_actions
                )

            elif non_exit_objects:
                # ── TARGET-DIRECTED HEADING & MISSING AXIS DISCOVERY ──────────────
                # When complete A* path cannot be computed (e.g. uncalibrated actions or obstacles),
                # advance towards the nearest safe stance of the lowest-risk highest-saliency object.
                best_candidate_obj = max(
                    non_exit_objects,
                    key=lambda o: (
                        (100.0 if self.ledger.get_interaction_count(o.object_id) == 0 else 0.0)
                        + (250.0 if self.ledger.is_in_newly_reachable(o.cells) else 0.0)
                        + (o.salience_score or 10.0)
                        - 15.0 * float(self.ledger.get_interaction_count(o.object_id))
                        - 220.0
                        * estimate_entity_risk(
                            engine, o.feature_id, o.cells, grid_arr, bg_val, ledger=self.ledger
                        )
                    ),
                )
                adj_fallback = [
                    tuple(c) for c in best_candidate_obj.properties.get("adjacent_stances", ())
                ]
                if (
                    discrete_interact_act is not None
                    and engine.is_feature_unverified(best_candidate_obj.feature_id, bg_val)
                    and not self.ledger.is_interacted(best_candidate_obj.object_id)
                    and adj_fallback
                ):
                    candidate_stances = adj_fallback
                else:
                    candidate_stances = (
                        best_candidate_obj.interaction_stances or best_candidate_obj.cells
                    )
                if candidate_stances:
                    target_stance = min(
                        candidate_stances,
                        key=lambda s: abs(av_pos[0] - s[0]) + abs(av_pos[1] - s[1]),
                    )
                    dr_target = target_stance[0] - av_pos[0]
                    dc_target = target_stance[1] - av_pos[1]

                    # 1. Active Inference Missing Axis Calibration
                    calibrated_dr = {dr for _, dr, _ in movable_actions}
                    calibrated_dc = {dc for _, _, dc in movable_actions}
                    uncalibrated_acts = [
                        a
                        for a in available_actions
                        if a not in engine.action_dynamics
                        and engine.inhibited_actions.get(a, 0) == 0
                        and not engine.is_spatial_effector(a)
                    ]
                    needs_horiz = dc_target != 0 and not any(dc != 0 for dc in calibrated_dc)
                    needs_vert = dr_target != 0 and not any(dr != 0 for dr in calibrated_dr)
                    if (needs_horiz or needs_vert) and uncalibrated_acts:
                        from hbllm.hcir.world.cortex_planner import MentalSimulationStep

                        self.active_target_object_id = best_candidate_obj.object_id
                        logger.info(
                            "ObjectStateGraphPlanner: Missing axis heading probe [%s, action=%s] towards %s.",
                            best_candidate_obj.object_id,
                            uncalibrated_acts[0],
                            target_stance,
                        )
                        return [
                            MentalSimulationStep(
                                action=uncalibrated_acts[0],
                                predicted_avatar_pos=av_pos,
                                expected_mutation="CALIBRATE_OBJECT_HEADING",
                            )
                        ]

        # 2B. Spatial Effector (Point Actuation) Object Interaction
        if spatial_effector_actions:
            eff_act = spatial_effector_actions[0]
            if movable_actions and engine.inhibited_actions.get(eff_act, 0) > 0:
                return None
            from hbllm.hcir.world.cortex_planner import MentalSimulationStep

            quiescent_coords = get_engine_state_attr(engine, "quiescent_actuation_targets", set())
            quiescent_feats = get_engine_state_attr(engine, "quiescent_features", set())

            # Check if this effector action defines discrete valid targets
            eff_aff = get_engine_state_attr(engine, "action_affordances", {}).get(eff_act)
            raw_valid_targets = (
                eff_aff.metadata.get("valid_targets")
                if (eff_aff and hasattr(eff_aff, "metadata") and isinstance(eff_aff.metadata, dict))
                else None
            )

            valid_targets_list: list[tuple[int, int, dict[str, Any]]] = []
            if isinstance(raw_valid_targets, (list, tuple)) and raw_valid_targets:
                for vt in raw_valid_targets:
                    if isinstance(vt, dict):
                        raw_vx = (
                            vt.get("x")
                            if vt.get("x") is not None
                            else vt.get("c", vt.get("col", 0))
                        )
                        raw_vy = (
                            vt.get("y")
                            if vt.get("y") is not None
                            else vt.get("r", vt.get("row", 0))
                        )
                        vx = int(raw_vx) if raw_vx is not None else 0
                        vy = int(raw_vy) if raw_vy is not None else 0
                        valid_targets_list.append((vy, vx, dict(vt)))

            def _find_best_valid_target(
                target_r: int, target_c: int
            ) -> tuple[int, int, dict[str, Any]] | None:
                """Find the closest un-quiescent valid target to (target_r, target_c)."""
                best_vt_tuple = None
                best_d = float("inf")
                for vy, vx, vt in valid_targets_list:
                    if (vy, vx) in quiescent_coords:
                        continue
                    if hasattr(engine, "working_memory") and hasattr(
                        engine.working_memory, "habenular_ior"
                    ):
                        if engine.working_memory.habenular_ior.is_action_inhibited(
                            engine.avatar_pos, eff_act, action_data={"x": vx, "y": vy}
                        ):
                            continue
                    vis = get_engine_state_attr(engine, "entity_visit_counts", {}).get(
                        f"actuate_{vy}_{vx}", 0
                    )
                    d = float(abs(vy - target_r) + abs(vx - target_c)) + float(vis) * 20.0
                    if d < best_d:
                        best_d = d
                        best_vt_tuple = (vy, vx, vt)
                return best_vt_tuple

            # DIRECTIVE 4 for effectors: Known triggers
            for obj in non_exit_objects:
                unprobed_cells = [c for c in obj.cells if c not in quiescent_coords]
                if not unprobed_cells:
                    continue
                target_cell = (
                    obj.anchor_coord if obj.anchor_coord in unprobed_cells else unprobed_cells[0]
                )
                tr, tc = target_cell
                is_known_trigger = obj.object_id in self.ledger.unlocked_by_object or any(
                    m.trigger_pos in obj.cells
                    or (m.trigger_feature is not None and m.trigger_feature == obj.feature_id)
                    for m in engine.state_mutations
                )
                if is_known_trigger:
                    if valid_targets_list:
                        snapped = _find_best_valid_target(tr, tc)
                        if snapped is not None:
                            svy, svx, svt = snapped
                            self.active_target_object_id = obj.object_id
                            return [
                                MentalSimulationStep(
                                    action=eff_act,
                                    action_data=dict(svt),
                                    predicted_avatar_pos=av_pos or (svy, svx),
                                    expected_mutation="ACTUATE_UNLOCK",
                                )
                            ]
                    else:
                        self.active_target_object_id = obj.object_id
                        return [
                            MentalSimulationStep(
                                action=eff_act,
                                action_data={"x": tc, "y": tr},
                                predicted_avatar_pos=av_pos or (tr, tc),
                                expected_mutation="ACTUATE_UNLOCK",
                            )
                        ]

            # DIRECTIVE 5 for effectors: Probe highest-saliency object with unprobed cell selection & quiescence avoidance
            from hbllm.hcir.world.percept import ExogenousSignalTracker

            actuatable_candidates: list[
                tuple[SalientObject, tuple[int, int], dict[str, Any], float]
            ] = []
            for obj in non_exit_objects:
                unprobed_cells = [c for c in obj.cells if c not in quiescent_coords]
                if not unprobed_cells:
                    continue
                # Prioritize interior non-boundary unprobed cells if available
                non_frame_unprobed = [
                    c
                    for c in unprobed_cells
                    if not (
                        grid_arr is not None
                        and ExogenousSignalTracker.is_partitioned_or_enclosing(
                            c, lattice=grid_arr, include_outer_edge=True
                        )
                    )
                ]
                target_cell = (
                    non_frame_unprobed[0]
                    if non_frame_unprobed
                    else (
                        obj.anchor_coord
                        if obj.anchor_coord in unprobed_cells
                        else unprobed_cells[0]
                    )
                )
                tr, tc = target_cell

                action_data: dict[str, Any]
                target_r, target_c = tr, tc
                if valid_targets_list:
                    snapped = _find_best_valid_target(tr, tc)
                    if snapped is None:
                        continue
                    target_r, target_c, action_data = snapped
                else:
                    action_data = {"x": tc, "y": tr}

                visit_count = self.ledger.get_interaction_count(obj.object_id)
                untouched_bonus = 100.0 if visit_count == 0 else 0.0
                ior_penalty = 20.0 * float(visit_count) + (visit_count**2) * 5.0
                quiescent_pen = 80.0 if obj.feature_id in quiescent_feats else 0.0
                inert_pen = 100.0 if self.ledger.is_inert(obj.object_id) else 0.0
                is_frame_cell = (
                    ExogenousSignalTracker.is_partitioned_or_enclosing(
                        (target_r, target_c), lattice=grid_arr, include_outer_edge=True
                    )
                    if grid_arr is not None
                    else False
                )
                frame_pen = 150.0 if is_frame_cell else 0.0
                saliency = obj.salience_score or (100.0 / math.log2(2 + max(1, obj.area)))
                score = (
                    untouched_bonus + saliency - ior_penalty - quiescent_pen - frame_pen - inert_pen
                )
                actuatable_candidates.append((obj, (target_r, target_c), action_data, score))

            if actuatable_candidates:
                actuatable_candidates.sort(key=lambda x: x[3], reverse=True)
                chosen_obj, chosen_target_cell, chosen_act_data, chosen_score = (
                    actuatable_candidates[0]
                )
                tr, tc = chosen_target_cell
                self.active_target_object_id = chosen_obj.object_id
                logger.info(
                    "ObjectStateGraphPlanner: Directive 5 SATISFIED (Effector)! Probing object "
                    "[%s, target_cell=(%d,%d), score=%.1f, action_data=%s].",
                    chosen_obj.object_id,
                    tr,
                    tc,
                    chosen_score,
                    chosen_act_data,
                )
                if movable_actions:
                    engine.inhibited_actions[eff_act] = max(
                        engine.inhibited_actions.get(eff_act, 0), 4
                    )
                return [
                    MentalSimulationStep(
                        action=eff_act,
                        action_data=chosen_act_data,
                        predicted_avatar_pos=av_pos or (tr, tc),
                        expected_mutation="ACTUATE_OBJECT",
                    )
                ]

            # DIRECTIVE 5B: If no salient object produced an un-quiescent target, but discrete valid_targets exist
            if valid_targets_list:
                unquiescent_vts = [
                    (vy, vx, vt)
                    for (vy, vx, vt) in valid_targets_list
                    if (vy, vx) not in quiescent_coords
                ]
                if unquiescent_vts:
                    unquiescent_vts.sort(
                        key=lambda item: get_engine_state_attr(
                            engine, "entity_visit_counts", {}
                        ).get(f"actuate_{item[0]}_{item[1]}", 0)
                    )
                    best_vy, best_vx, best_vt = unquiescent_vts[0]
                    logger.info(
                        "ObjectStateGraphPlanner: Directive 5B (Direct Valid Target)! Probing discrete target "
                        "[(%d,%d), action_data=%s].",
                        best_vy,
                        best_vx,
                        best_vt,
                    )
                    if movable_actions:
                        engine.inhibited_actions[eff_act] = max(
                            engine.inhibited_actions.get(eff_act, 0), 4
                        )
                    return [
                        MentalSimulationStep(
                            action=eff_act,
                            action_data=dict(best_vt),
                            predicted_avatar_pos=av_pos or (best_vy, best_vx),
                            expected_mutation="ACTUATE_OBJECT",
                        )
                    ]

        return None

    def _finalize_option_path(
        self,
        engine: Any,
        obj: SalientObject,
        path: list[Any],
        available_actions: Sequence[Any],
        movable_actions: Sequence[tuple[Any, int, int]],
    ) -> list[Any]:
        """Finalize the primitive action sequence, attaching terminal interaction if necessary."""
        from hbllm.hcir.world.autonomous_epistemic_engine import MentalSimulationStep

        interact_candidates = [
            a
            for a in available_actions
            if engine.is_spatial_effector(a)
            or (not engine.is_displacement_action(a) and engine.inhibited_actions.get(a, 0) <= 5)
        ]
        interact_act = interact_candidates[0] if interact_candidates else None
        has_real_play_history = bool(engine.tested_actions)
        confirmed_discrete_interact = next(
            (
                a
                for a in available_actions
                if not engine.is_displacement_action(a)
                and not engine.is_spatial_effector(a)
                and engine.inhibited_actions.get(a, 0) <= 5
                and (
                    a in engine.action_dynamics
                    or a in engine.tested_actions
                    or has_real_play_history
                )
            ),
            None,
        )
        is_known_contact_trigger = any(
            m.trigger_type == "CONTACT"
            and (m.trigger_pos in obj.cells or m.trigger_feature == obj.feature_id)
            for m in engine.state_mutations
        )
        bg_val = int(engine.bg_feature)
        unverified_obj = (
            engine.is_feature_unverified(obj.feature_id, bg_val)
            and obj.feature_id not in engine.symbolic_theory.goal_features
            and not obj.properties.get("confirmed_goal")
        )

        def _prefer_reversible_interact(stance: tuple[int, int] | None) -> bool:
            if (
                confirmed_discrete_interact is None
                or not unverified_obj
                or is_known_contact_trigger
            ):
                return False
            if self.ledger.is_interacted(obj.object_id):
                return False
            if stance is not None and engine.symbolic_theory.is_locus_actuator_probed(stance):
                return False
            return True

        if not path:
            # Avatar is already at or adjacent to the object stance!
            if obj.is_solid and interact_act is not None:
                return [
                    MentalSimulationStep(
                        action=interact_act,
                        predicted_avatar_pos=engine.avatar_pos,
                        expected_mutation="INTERACT",
                    )
                ]
            if (
                engine.avatar_pos is not None
                and engine.avatar_pos not in obj.cells
                and _prefer_reversible_interact(engine.avatar_pos)
            ):
                return [
                    MentalSimulationStep(
                        action=confirmed_discrete_interact,
                        predicted_avatar_pos=engine.avatar_pos,
                        expected_mutation="INTERACT",
                    )
                ]
            if engine.avatar_pos is not None and engine.avatar_pos not in obj.cells:
                for act, dr, dc in movable_actions:
                    bump_pos = (engine.avatar_pos[0] + dr, engine.avatar_pos[1] + dc)
                    if bump_pos in obj.cells:
                        return [
                            MentalSimulationStep(
                                action=act,
                                predicted_avatar_pos=bump_pos
                                if not obj.is_solid
                                else engine.avatar_pos,
                                expected_mutation="BUMP",
                            )
                        ]
            if interact_act is not None:
                return [
                    MentalSimulationStep(
                        action=interact_act,
                        predicted_avatar_pos=engine.avatar_pos,
                        expected_mutation="INTERACT",
                    )
                ]
            return []

        target_stance = path[-1].predicted_avatar_pos

        # If target stance is adjacent to the object (not onto it), execute terminal interaction or bump
        if target_stance not in obj.cells:
            if (obj.is_solid and interact_act is not None) or _prefer_reversible_interact(
                target_stance
            ):
                chosen_interact = (
                    confirmed_discrete_interact
                    if _prefer_reversible_interact(target_stance)
                    else interact_act
                )
                interact_step = MentalSimulationStep(
                    action=chosen_interact,
                    predicted_avatar_pos=target_stance,
                    expected_mutation="INTERACT",
                )
                return path + [interact_step]
            else:
                # Find bump/contact move directed towards the object
                for act, dr, dc in movable_actions:
                    bump_pos = (target_stance[0] + dr, target_stance[1] + dc)
                    if bump_pos in obj.cells:
                        bump_step = MentalSimulationStep(
                            action=act,
                            predicted_avatar_pos=bump_pos if not obj.is_solid else target_stance,
                            expected_mutation="BUMP",
                        )
                        return path + [bump_step]

        return path
