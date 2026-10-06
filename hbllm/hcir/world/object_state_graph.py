"""Object-Centric Macro-Action State Graph Planner (Options Framework).

Implements the 5 executive cognitive directives:
1. Check if exit is open.
2. If open, go through it.
3. If not, explore other objects.
4. If you know what they already do and those can unlock the door, use it.
5. If not, play with it (interact) and see what is the outcome.

Decouples high-level discrete choice across K salient objects (2^K abstract state space)
from low-level physical distance D (handled by deterministic BFS/A* path controller).
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity

logger = logging.getLogger(__name__)


class ObjectCategory(str, Enum):
    """Semantic category for lifted salient objects."""

    EXIT = "EXIT"
    KEY = "KEY"
    SWITCH = "SWITCH"
    BARRIER = "BARRIER"
    CONTAINER = "CONTAINER"
    MANIPULABLE = "MANIPULABLE"
    UNKNOWN = "UNKNOWN"


@dataclass
class SalientObject:
    """A distinct interactive object in the lifted object-centric graph."""

    object_id: str
    category: ObjectCategory
    feature_id: int
    centroid: tuple[int, int]
    cells: set[tuple[int, int]]
    area: int
    is_solid: bool = False
    interaction_stances: list[tuple[int, int]] = field(default_factory=list)
    interaction_action: int | None = None
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class MacroOption:
    """A high-level macro-action representing an option to interact with a salient object."""

    option_id: str
    directive: int
    target_object: SalientObject | None
    target_stance: tuple[int, int]
    primitive_steps: list[Any]  # list[MentalSimulationStep]
    score: float = 0.0
    description: str = ""


class ObjectStateLedger:
    """Maintains the discrete state configuration across the K salient objects."""

    def __init__(self) -> None:
        self.interaction_counts: dict[str, int] = {}
        self.interacted_positions: set[tuple[int, int]] = set()
        self.interacted_features: set[int] = set()
        self.observed_mutations: list[Any] = []
        self.unlocked_barriers: set[tuple[int, int]] = set()

    def reset_level(self) -> None:
        """Reset transient level-specific interaction state."""
        self.interaction_counts.clear()
        self.interacted_positions.clear()
        self.interacted_features.clear()
        self.observed_mutations.clear()
        self.unlocked_barriers.clear()

    def record_interaction(
        self,
        object_id: str,
        pos: tuple[int, int],
        feature_id: int | None = None,
    ) -> None:
        """Record an interaction event with an object."""
        self.interaction_counts[object_id] = self.interaction_counts.get(object_id, 0) + 1
        self.interacted_positions.add(pos)
        if feature_id is not None:
            self.interacted_features.add(feature_id)

    def get_interaction_count(self, object_id: str) -> int:
        """Return the number of times an object has been interacted with."""
        return self.interaction_counts.get(object_id, 0)


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

    def reset_episode(self, is_new_level: bool = False) -> None:
        """Reset planner state for a new level or trial."""
        self.ledger.reset_level()
        self.current_macro_option = None
        self.active_target_object_id = None

    def extract_salient_objects(
        self,
        engine: Any,
        curr_grid: np.ndarray,
        bg: int,
    ) -> list[SalientObject]:
        """Extract and categorize K salient objects from the sensory grid."""
        H, W = curr_grid.shape
        av_pos = engine.avatar_pos
        av_feats = engine.avatar_features or (
            {engine.avatar_feature} if engine.avatar_feature is not None else set()
        )
        barrier_features = engine.symbolic_theory.barrier_features
        static_barriers = set(engine.learned_barriers)
        lethal_features = set(engine.hazard_tracker.known_lethal_features)

        # Supplement static_barriers with known barrier features
        for r in range(H):
            for c in range(W):
                val = int(curr_grid[r, c])
                if engine.symbolic_theory.is_barrier(val) or val in barrier_features:
                    static_barriers.add((r, c))

        raw_entities: list[SpatialEntity] = engine.extract_entities(curr_grid, bg)
        salient_objects: list[SalientObject] = []

        # Track exit goals
        known_goal_positions = set(engine.learned_goal_positions)
        known_goal_features = set(engine.symbolic_theory.goal_features) | set(
            engine.learned_goal_features
        )

        # Check structural goals
        try:
            from hbllm.hcir.world.autonomous_epistemic_engine import PerceptionEngine

            structural_goals = PerceptionEngine.detect_structural_goals(
                curr_grid,
                bg=bg,
                avatar_features=av_feats,
                avatar_feature=engine.avatar_feature,
                barrier_features=barrier_features,
                known_lethal_features=lethal_features,
            )
            for sg in structural_goals:
                cells = sg.get("cells", [])
                if cells:
                    for cp in cells:
                        known_goal_positions.add(cp)
                pos = sg.get("position")
                if pos:
                    known_goal_positions.add(pos)
        except Exception:
            pass

        for e in raw_entities:
            # Skip avatar
            if e.feature_id in av_feats or (av_pos is not None and e.grid_pos == av_pos):
                continue
            # Skip massive background or bulk perimeter walls
            if e.area > H * W * 0.30:
                continue
            # Skip pure lethal hazards unless small interactive entity
            if e.feature_id in lethal_features and e.area > 9:
                continue

            entity_cells = set(e.properties.get("cells", [e.grid_pos]))

            # Categorize object
            is_goal = (
                engine.symbolic_theory.is_goal(e)
                or e.feature_id in known_goal_features
                or any(cp in known_goal_positions for cp in entity_cells)
                or e.role == EntityRole.GOAL
            )

            is_switch = any(
                m.trigger_pos in entity_cells
                or (m.trigger_feature is not None and m.trigger_feature == e.feature_id)
                for m in engine.state_mutations
            )

            is_cargo = (
                engine.symbolic_theory.is_cargo(e)
                or e.feature_id in engine.learned_cargo_features
                or e.role == EntityRole.MANIPULABLE
            )

            if is_goal:
                category = ObjectCategory.EXIT
                is_solid = False
            elif is_switch:
                category = ObjectCategory.SWITCH
                is_solid = (
                    engine.symbolic_theory.is_barrier(e.feature_id)
                    or e.feature_id in barrier_features
                )
            elif is_cargo:
                category = ObjectCategory.MANIPULABLE
                is_solid = True
            elif e.area <= 16:
                category = ObjectCategory.KEY
                is_solid = (
                    engine.symbolic_theory.is_barrier(e.feature_id)
                    or e.feature_id in barrier_features
                )
            elif (
                engine.symbolic_theory.is_barrier(e.feature_id) or e.feature_id in barrier_features
            ):
                category = ObjectCategory.BARRIER
                is_solid = True
            else:
                category = ObjectCategory.UNKNOWN
                is_solid = e.role == EntityRole.OBSTACLE

            # Compute interaction stances
            stances: set[tuple[int, int]] = set()

            if category == ObjectCategory.EXIT or not is_solid:
                # Walkable object / Exit: avatar walks onto the object cells
                for cr, cc in entity_cells:
                    if (
                        0 <= cr < H
                        and 0 <= cc < W
                        and (cr, cc) not in static_barriers
                        and int(curr_grid[cr, cc]) not in lethal_features
                    ):
                        stances.add((cr, cc))

            if is_solid or not stances:
                # Solid object (or fallback): avatar stands in orthogonal adjacent neighbor cell
                for cr, cc in entity_cells:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ar, ac = cr + dr, cc + dc
                        if (
                            0 <= ar < H
                            and 0 <= ac < W
                            and (ar, ac) not in static_barriers
                            and int(curr_grid[ar, ac]) not in lethal_features
                            and (ar, ac) not in entity_cells
                        ):
                            stances.add((ar, ac))

            obj = SalientObject(
                object_id=e.id,
                category=category,
                feature_id=e.feature_id,
                centroid=(int(round(e.centroid[0])), int(round(e.centroid[1]))),
                cells=entity_cells,
                area=e.area,
                is_solid=is_solid,
                interaction_stances=sorted(list(stances)),
                properties={"role": str(e.role)},
            )
            salient_objects.append(obj)

        return salient_objects

    def find_shortest_path(
        self,
        start_pos: tuple[int, int],
        target_stances: set[tuple[int, int]],
        static_barriers: set[tuple[int, int]],
        lethal_features: set[int],
        movable_actions: list[tuple[Any, int, int]],
        grid_shape: tuple[int, int],
        curr_grid: np.ndarray,
        patroller_cells: set[tuple[int, int]] | None = None,
        failed_transitions: set[tuple[tuple[int, int], Any]] | None = None,
    ) -> list[Any] | None:
        """Find the shortest deterministic path of primitive motor actions to any target stance."""
        if start_pos in target_stances:
            return []

        H, W = grid_shape
        from hbllm.hcir.world.autonomous_epistemic_engine import MentalSimulationStep

        q: deque[tuple[tuple[int, int], list[MentalSimulationStep]]] = deque([(start_pos, [])])
        visited: set[tuple[int, int]] = {start_pos}

        while q:
            cur_pos, path = q.popleft()
            if cur_pos in target_stances:
                return list(path)

            for act, dr, dc in movable_actions:
                if failed_transitions and (cur_pos, act) in failed_transitions:
                    continue
                nr, nc = cur_pos[0] + dr, cur_pos[1] + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if (nr, nc) in visited:
                    continue

                step_r = int(np.sign(dr))
                step_c = int(np.sign(dc))
                dist = max(abs(dr), abs(dc))
                blocked = False

                for k in range(1, dist + 1):
                    kr = cur_pos[0] + step_r * k
                    kc = cur_pos[1] + step_c * k
                    if (kr, kc) in static_barriers:
                        blocked = True
                        break
                    if 0 <= kr < H and 0 <= kc < W:
                        val = int(curr_grid[kr, kc])
                        if val in lethal_features:
                            blocked = True
                            break

                if blocked:
                    continue
                if patroller_cells and (nr, nc) in patroller_cells:
                    continue

                visited.add((nr, nc))
                new_step = MentalSimulationStep(action=act, predicted_avatar_pos=(nr, nc))
                q.append(((nr, nc), path + [new_step]))

        return None

    def plan_macro_option(
        self,
        engine: Any,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> list[Any] | None:
        """Synthesize a complete macro-option plan adhering to the 5 executive directives."""
        if not engine.is_motor_grounded() or engine.avatar_pos is None:
            return None

        H, W = curr_grid.shape
        bg = engine.estimate_background(curr_grid)
        av_pos = engine.avatar_pos

        # Extract calibrated movable actions
        movable_actions: list[tuple[Any, int, int]] = []
        for act in available_actions:
            if engine.is_spatial_effector(act):
                continue
            dyn = engine.action_dynamics.get(act)
            if dyn is not None and dyn.is_displacement_action():
                dr, dc = dyn.get_displacement()
                movable_actions.append((act, dr, dc))

        if not movable_actions:
            return None

        # Build barrier and lethal sets
        barrier_features = engine.symbolic_theory.barrier_features
        static_barriers = set(engine.learned_barriers)
        for r in range(H):
            for c in range(W):
                val = int(curr_grid[r, c])
                if engine.symbolic_theory.is_barrier(val) or val in barrier_features:
                    static_barriers.add((r, c))

        lethal_features = set(engine.hazard_tracker.known_lethal_features)
        failed_transitions = set(getattr(engine, "failed_transitions", set()))

        # Detect patroller / dynamic threat positions
        patroller_cells: set[tuple[int, int]] = set()
        if engine.mobile_threat_features:
            for r in range(H):
                for c in range(W):
                    if int(curr_grid[r, c]) in engine.mobile_threat_features:
                        patroller_cells.add((r, c))

        # Extract all salient objects
        salient_objects = self.extract_salient_objects(engine, curr_grid, bg)

        # ── DIRECTIVE 1 & 2: Check if exit is open. If open, go through it. ───
        exit_objects = [o for o in salient_objects if o.category == ObjectCategory.EXIT]
        exit_stances: set[tuple[int, int]] = set()
        for eo in exit_objects:
            exit_stances.update(eo.interaction_stances)

        # Supplement with learned goal positions
        for gp in engine.learned_goal_positions:
            if 0 <= gp[0] < H and 0 <= gp[1] < W and gp not in static_barriers:
                exit_stances.add(gp)

        if exit_stances:
            path_to_exit = self.find_shortest_path(
                start_pos=av_pos,
                target_stances=exit_stances,
                static_barriers=static_barriers,
                lethal_features=lethal_features,
                movable_actions=movable_actions,
                grid_shape=(H, W),
                curr_grid=curr_grid,
                patroller_cells=patroller_cells,
                failed_transitions=failed_transitions,
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
                return path_to_exit

        # ── DIRECTIVE 3: If not, explore other objects. ──────────────────────
        non_exit_objects = [
            o
            for o in salient_objects
            if o.category != ObjectCategory.EXIT and o.interaction_stances
        ]

        if not non_exit_objects:
            return None

        # Check reachability for all candidate objects
        reachable_objects: list[tuple[SalientObject, list[Any]]] = []
        for obj in non_exit_objects:
            obj_stances = set(obj.interaction_stances)
            path = self.find_shortest_path(
                start_pos=av_pos,
                target_stances=obj_stances,
                static_barriers=static_barriers,
                lethal_features=lethal_features,
                movable_actions=movable_actions,
                grid_shape=(H, W),
                curr_grid=curr_grid,
                patroller_cells=patroller_cells,
                failed_transitions=failed_transitions,
            )
            if path is not None:
                reachable_objects.append((obj, path))

        if not reachable_objects:
            logger.debug(
                "ObjectStateGraphPlanner: No interactive objects currently reachable from %s.",
                av_pos,
            )
            return None

        # ── DIRECTIVE 4: If you know what they already do and those can unlock the door, use it. ──
        unlocking_candidates: list[tuple[SalientObject, list[Any]]] = []
        for obj, path in reachable_objects:
            is_known_trigger = any(
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
            # Sort by path length (prefer closest unlocking trigger)
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
            return self._finalize_option_path(
                engine, chosen_obj, chosen_path, available_actions, movable_actions
            )

        # ── DIRECTIVE 5: If not, play with it (interact) and see what is the outcome. ───
        scored_candidates: list[tuple[SalientObject, list[Any], float]] = []
        for obj, path in reachable_objects:
            visit_count = self.ledger.get_interaction_count(obj.object_id)
            untouched_bonus = 100.0 if visit_count == 0 else 0.0
            recency_penalty = 15.0 * float(visit_count)
            distance_penalty = float(len(path)) * 0.5
            small_item_bonus = 20.0 if obj.category == ObjectCategory.KEY else 0.0
            score = untouched_bonus + small_item_bonus - recency_penalty - distance_penalty
            scored_candidates.append((obj, path, score))

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
        return self._finalize_option_path(
            engine, best_obj, best_path, available_actions, movable_actions
        )

    def _finalize_option_path(
        self,
        engine: Any,
        obj: SalientObject,
        path: list[Any],
        available_actions: Sequence[Any],
        movable_actions: list[tuple[Any, int, int]],
    ) -> list[Any]:
        """Finalize the primitive action sequence, attaching terminal interaction if necessary."""
        from hbllm.hcir.world.autonomous_epistemic_engine import MentalSimulationStep

        if not path:
            # Avatar is already at or adjacent to the object stance!
            if 5 in available_actions:
                return [
                    MentalSimulationStep(
                        action=5,
                        predicted_avatar_pos=engine.avatar_pos,
                        expected_mutation="INTERACT",
                    )
                ]
            return []

        target_stance = path[-1].predicted_avatar_pos

        # If the object is solid and the stance is adjacent to it, execute terminal interaction
        if obj.is_solid and target_stance not in obj.cells:
            if 5 in available_actions:
                interact_step = MentalSimulationStep(
                    action=5,
                    predicted_avatar_pos=target_stance,
                    expected_mutation="INTERACT",
                )
                return path + [interact_step]
            else:
                # Find bump move directed towards the object
                for act, dr, dc in movable_actions:
                    bump_pos = (target_stance[0] + dr, target_stance[1] + dc)
                    if bump_pos in obj.cells:
                        bump_step = MentalSimulationStep(
                            action=act,
                            predicted_avatar_pos=target_stance,
                            expected_mutation="BUMP",
                        )
                        return path + [bump_step]

        return path
