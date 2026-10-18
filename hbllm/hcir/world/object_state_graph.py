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
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.percept import PerceptionFrame, PerceptionRegistry

logger = logging.getLogger(__name__)


class ExecutiveDirective(IntEnum):
    """The 5 executive directives guiding object-centric option selection."""

    CHECK_EXIT_OPEN = 1
    GO_TO_EXIT = 2
    EXPLORE_OBJECTS = 3
    USE_UNLOCKING_OBJECT = 4
    INTERACT_UNKNOWN_OBJECT = 5


class ObjectCategory(StrEnum):
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
    anchor_coord: tuple[int, int] = (0, 0)
    salience_score: float = 0.0
    properties: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_spatial_entity(
        cls,
        entity: SpatialEntity,
        category: ObjectCategory = ObjectCategory.UNKNOWN,
    ) -> SalientObject:
        """Construct a SalientObject directly from an HCIR SpatialEntity component."""
        cells = set(entity.properties.get("cells", [entity.grid_pos]))
        centroid = (int(round(entity.centroid[0])), int(round(entity.centroid[1])))
        anchor = (
            min(cells, key=lambda p: (p[0] - centroid[0]) ** 2 + (p[1] - centroid[1]) ** 2)
            if cells
            else centroid
        )
        salience = float(
            entity.properties.get(
                "salience_score",
                100.0 / math.log2(2 + max(1, entity.area)),
            )
        )
        return cls(
            object_id=entity.id,
            category=category,
            feature_id=entity.feature_id if entity.feature_id is not None else 0,
            centroid=centroid,
            cells=cells,
            area=entity.area,
            is_solid=(entity.role == EntityRole.OBSTACLE),
            interaction_stances=[entity.grid_pos],
            anchor_coord=anchor,
            salience_score=salience,
            properties={"role": str(entity.role), **dict(entity.properties)},
        )

    def to_spatial_entity(self, role: EntityRole | None = None) -> SpatialEntity:
        """Convert salient object back into a formal HCIR SpatialEntity."""
        eff_role = role or (
            EntityRole.GOAL
            if self.category == ObjectCategory.EXIT
            else EntityRole.MANIPULABLE
            if self.category in (ObjectCategory.KEY, ObjectCategory.MANIPULABLE)
            else EntityRole.OBSTACLE
            if self.is_solid
            else EntityRole.UNKNOWN
        )
        min_r = min(r for r, c in self.cells) if self.cells else self.centroid[0]
        max_r = max(r for r, c in self.cells) if self.cells else self.centroid[0]
        min_c = min(c for r, c in self.cells) if self.cells else self.centroid[1]
        max_c = max(c for r, c in self.cells) if self.cells else self.centroid[1]
        return SpatialEntity(
            id=self.object_id,
            role=eff_role,
            centroid=(float(self.centroid[0]), float(self.centroid[1])),
            grid_pos=self.interaction_stances[0] if self.interaction_stances else self.centroid,
            area=self.area,
            bounding_box=(min_r, max_r, min_c, max_c),
            feature_id=self.feature_id,
            properties=dict(self.properties),
        )

    def to_causal_predicate(self) -> CausalPredicate:
        """Export object identity as a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable=f"object_category::{self.category.value}",
            operator="==",
            value=self.feature_id,
            target_entity=self.object_id,
        )


@dataclass
class MacroOption:
    """A high-level macro-action representing an option to interact with a salient object."""

    option_id: str
    directive: ExecutiveDirective | int
    target_object: SalientObject | None
    target_stance: tuple[int, int]
    primitive_steps: list[Any]  # list[MentalSimulationStep]
    score: float = 0.0
    description: str = ""

    def to_subgoal(self) -> dict[str, Any]:
        """Convert macro-option into a structured declarative subgoal dictionary."""
        return {
            "option_id": self.option_id,
            "directive": int(self.directive),
            "target_object_id": self.target_object.object_id if self.target_object else None,
            "target_stance": self.target_stance,
            "step_count": len(self.primitive_steps),
            "score": self.score,
            "description": self.description,
        }


class ObjectStateLedger:
    """Maintains the discrete state configuration across the K salient objects."""

    def __init__(self) -> None:
        self.interaction_counts: dict[str, int] = {}
        self.last_interacted_object_id: str | None = None
        self.consecutive_counts: dict[str, int] = {}
        self.interacted_positions: set[tuple[int, int]] = set()
        self.interacted_features: set[int] = set()
        self.observed_mutations: list[Any] = []
        self.unlocked_barriers: set[tuple[int, int]] = set()
        self.object_causal_effects: dict[str, list[str]] = {}
        self.unlocked_by_object: dict[str, set[tuple[int, int]]] = {}

    def reset_level(self) -> None:
        """Reset transient level-specific interaction state."""
        self.interaction_counts.clear()
        self.last_interacted_object_id = None
        self.consecutive_counts.clear()
        self.interacted_positions.clear()
        self.interacted_features.clear()
        self.observed_mutations.clear()
        self.unlocked_barriers.clear()
        self.object_causal_effects.clear()
        self.unlocked_by_object.clear()

    def record_interaction(
        self,
        object_id: str,
        pos: tuple[int, int],
        feature_id: int | None = None,
    ) -> None:
        """Record an interaction event with an object."""
        self.interaction_counts[object_id] = self.interaction_counts.get(object_id, 0) + 1
        if object_id == self.last_interacted_object_id:
            self.consecutive_counts[object_id] = self.consecutive_counts.get(object_id, 0) + 1
        else:
            self.consecutive_counts[object_id] = 1
            self.last_interacted_object_id = object_id
        self.interacted_positions.add(pos)
        if feature_id is not None:
            self.interacted_features.add(feature_id)

    def record_mutation(self, mutation: Any) -> None:
        """Record an environmental state mutation triggered by interaction."""
        self.observed_mutations.append(mutation)

    def record_causal_effect(
        self,
        object_id: str,
        effect_type: str,
        affected_coords: set[tuple[int, int]] | None = None,
    ) -> None:
        """Record an empirical causal effect attributed to an object interaction."""
        self.object_causal_effects.setdefault(object_id, []).append(effect_type)
        if affected_coords:
            self.unlocked_barriers.update(affected_coords)
            self.unlocked_by_object.setdefault(object_id, set()).update(affected_coords)

    def get_interaction_count(self, object_id: str) -> int:
        """Return the number of times an object has been interacted with."""
        return self.interaction_counts.get(object_id, 0)

    def get_consecutive_interaction_count(self, object_id: str) -> int:
        """Return the number of consecutive times an object has been interacted with."""
        if object_id == self.last_interacted_object_id:
            return self.consecutive_counts.get(object_id, 0)
        return 0

    def is_interacted(self, object_id: str) -> bool:
        """True if the object has been probed at least once."""
        return self.get_interaction_count(object_id) > 0

    def is_inert(self, object_id: str) -> bool:
        """True if the object was interacted with at least once and produced no causal mutations."""
        return (
            self.get_interaction_count(object_id) >= 1
            and object_id not in self.object_causal_effects
        )

    def get_unlocked_barriers(self) -> set[tuple[int, int]]:
        """Return all barrier coordinates unlocked via object interactions."""
        return set(self.unlocked_barriers)


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
        self.relational_matcher: RelationalGraphMatcher = RelationalGraphMatcher()

    def reset_episode(self, is_new_level: bool = False) -> None:
        """Reset planner state for a new level or trial."""
        if is_new_level:
            self.ledger.reset_level()
        else:
            # On same-level retry: decay interaction counts so heavily-probed objects
            # have their IOR decayed, but unprobed candidates take priority over repeating initial failures.
            for obj_id in list(self.ledger.interaction_counts.keys()):
                self.ledger.interaction_counts[obj_id] = self.ledger.interaction_counts[obj_id] // 2
        self.current_macro_option = None
        self.active_target_object_id = None

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
        for e in percept.entities:
            # Skip avatar
            if e.feature_id in av_feats or (
                bool(av_feats) and av_pos is not None and e.grid_pos == av_pos
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
            is_switch = any(
                m.trigger_pos in entity_cells
                or (m.trigger_feature is not None and m.trigger_feature == e.feature_id)
                for m in engine.state_mutations
            )
            is_cargo = (
                e.role == EntityRole.MANIPULABLE
                or engine.symbolic_theory.is_cargo(e)
                or e.feature_id in engine.symbolic_theory.cargo_features
            )

            if is_goal:
                category = ObjectCategory.EXIT
                is_solid = False
            elif is_switch:
                category = ObjectCategory.SWITCH
                is_solid = is_barrier
            elif is_cargo:
                category = ObjectCategory.MANIPULABLE
                is_solid = True
            elif is_barrier:
                category = ObjectCategory.BARRIER
                is_solid = True
            elif e.area <= 16:
                category = ObjectCategory.KEY
                is_solid = False
            else:
                category = ObjectCategory.UNKNOWN
                is_solid = e.role == EntityRole.OBSTACLE

            # Compute interaction stances using percept.is_passable
            stances: set[tuple[int, int]] = set()
            if category == ObjectCategory.EXIT or not is_solid:
                for cr, cc in entity_cells:
                    if percept.is_passable(cr, cc):
                        stances.add((cr, cc))

            if is_solid or not stances:
                for cr, cc in entity_cells:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ar, ac = cr + dr, cc + dc
                        if percept.is_passable(ar, ac) and (ar, ac) not in entity_cells:
                            stances.add((ar, ac))

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
                properties={"role": str(e.role), **dict(e.properties)},
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
                H, W = (
                    (curr_grid.shape[0], curr_grid.shape[1])
                    if hasattr(curr_grid, "shape") and len(curr_grid.shape) >= 2
                    else (16, 16)
                )
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
                    is_target = k == dist and (kr, kc) in target_stances
                    if percept is not None:
                        if not is_target and not percept.is_passable(kr, kc):
                            blocked = True
                            break
                    else:
                        if (kr, kc) in static_barriers and not is_target:
                            blocked = True
                            break
                        if 0 <= kr < H and 0 <= kc < W and curr_grid is not None:
                            val = int(curr_grid[kr, kc])
                            if val in lethal_features and not is_target:
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
        curr_grid: np.ndarray | PerceptionFrame,
        available_actions: Sequence[Any],
    ) -> list[Any] | None:
        """Synthesize a complete macro-option plan adhering to the 5 executive cognitive directives.
        Supports continuous spatial displacement paths (BFS/A*) and allocentric spatial effector (Click) actions.
        """
        if not available_actions:
            return None

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
                H, W = (
                    (curr_grid.shape[0], curr_grid.shape[1])
                    if hasattr(curr_grid, "shape") and len(curr_grid.shape) >= 2
                    else (16, 16)
                )
                percept = PerceptionFrame(shape=(H, W))
            grid_arr = curr_grid

        H, W = percept.shape
        av_pos = percept.avatar_pos or engine.avatar_pos

        # Extract calibrated movable actions and spatial effector actions
        movable_actions: list[tuple[Any, int, int]] = []
        spatial_effector_actions: list[Any] = []
        for act in available_actions:
            if (
                getattr(engine, "inhibited_actions", None)
                and engine.inhibited_actions.get(act, 0) > 0
            ):
                continue
            if engine.is_spatial_effector(act):
                spatial_effector_actions.append(act)
                continue
            if (
                getattr(engine, "immobile_entity_actions", None)
                and act in engine.immobile_entity_actions
            ):
                continue
            if (
                getattr(engine, "inhibited_actions", None)
                and engine.inhibited_actions.get(act, 0) > 0
            ):
                continue
            dyn = engine.action_dynamics.get(act)
            if dyn is not None and dyn.is_displacement_action():
                dr, dc = dyn.get_displacement()
                movable_actions.append((act, dr, dc))

        if not movable_actions and not spatial_effector_actions:
            return None

        # Static barriers and hazards directly from percept
        static_barriers = set(percept.static_barriers)
        lethal_features = set(engine.hazard_tracker.known_lethal_features)
        failed_transitions = set(getattr(engine, "failed_transitions", set()))

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
        exit_objects = [o for o in salient_objects if o.category == ObjectCategory.EXIT]
        exit_stances: set[tuple[int, int]] = set(percept.candidate_goals)
        for eo in exit_objects:
            exit_stances.update(eo.interaction_stances)

        # 1A. Displacement Navigation to Exit
        if exit_stances and movable_actions and av_pos is not None:
            path_to_exit = self.find_shortest_path(
                start_pos=av_pos,
                target_stances=exit_stances,
                static_barriers=static_barriers,
                lethal_features=lethal_features,
                movable_actions=movable_actions,
                grid_shape=(H, W),
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
                return path_to_exit

        # 1B. Spatial Effector Interaction with Exit
        if exit_objects and spatial_effector_actions and not movable_actions:
            eff_act = spatial_effector_actions[0]
            quiescent_coords = getattr(engine, "quiescent_click_targets", set())
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
                        expected_mutation="CLICK_EXIT",
                    )
                ]

        # ── DIRECTIVE 3: If not, explore other objects. ──────────────────────
        non_exit_objects = [o for o in salient_objects if o.category != ObjectCategory.EXIT]

        # ── DIRECTIVE 4B: Active Causal Actuator Contingency Pursuit ──────────
        if spatial_effector_actions:
            eff_act = spatial_effector_actions[0]
            quiescent_coords = getattr(engine, "quiescent_click_targets", set())
            actuator_coord = getattr(engine, "active_causal_actuator_coord", None)
            streak = getattr(engine, "consecutive_causal_probes", 0)
            streak_limit = getattr(engine, "causal_actuator_streak_limit", 4)
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
            for obj in non_exit_objects:
                if not obj.interaction_stances:
                    continue
                obj_stances = set(obj.interaction_stances)
                path = self.find_shortest_path(
                    start_pos=av_pos,
                    target_stances=obj_stances,
                    static_barriers=static_barriers,
                    lethal_features=lethal_features,
                    movable_actions=movable_actions,
                    grid_shape=(H, W),
                    curr_grid=grid_arr,
                    patroller_cells=patroller_cells,
                    failed_transitions=failed_transitions,
                    percept=percept,
                )
                if path is not None:
                    reachable_objects.append((obj, path))

            if reachable_objects:
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

                # Identify barriers along the geodesic corridor separating avatar and exit
                if not blocking_barrier_feats and exit_objects and av_pos is not None:
                    target_stances = [
                        st for eo in exit_objects for st in (eo.interaction_stances or eo.cells)
                    ]
                    if target_stances:
                        closest_stance = min(
                            target_stances,
                            key=lambda s: abs(av_pos[0] - s[0]) + abs(av_pos[1] - s[1]),
                        )
                        r0, c0 = av_pos
                        r1, c1 = closest_stance
                        for r in range(min(r0, r1), max(r0, r1) + 1):
                            for c in range(min(c0, c1), max(c0, c1) + 1):
                                feat = int(grid_arr[r, c])
                                if feat in barrier_features or engine.symbolic_theory.is_barrier(
                                    feat
                                ):
                                    blocking_barrier_feats.add(feat)

                # ── DIRECTIVE 4: If you know what they already do and those can unlock the door, use it. ──
                unlocking_candidates: list[tuple[SalientObject, list[Any]]] = []
                for obj, path in reachable_objects:
                    # Skip already-actuated static objects if the door remains locked
                    if self.ledger.get_interaction_count(
                        obj.object_id
                    ) >= 1 and not engine.working_memory.is_holding(obj.feature_id):
                        continue

                    # Check tool-barrier causal affinity
                    is_relevant_tool = False
                    if hasattr(engine.symbolic_theory, "get_causal_affinity"):
                        is_relevant_tool = (
                            engine.symbolic_theory.get_causal_affinity(
                                obj.feature_id, blocking_barrier_feats
                            )
                            > 0.0
                        )
                    if not is_relevant_tool:
                        tool_affinities: set[int] = set()
                        if hasattr(engine.working_memory, "tool_barrier_affinities"):
                            tool_affinities.update(
                                engine.working_memory.tool_barrier_affinities.get(
                                    obj.feature_id, set()
                                )
                            )
                        if hasattr(engine.symbolic_theory, "tool_barrier_affinities"):
                            tool_affinities.update(
                                getattr(engine.symbolic_theory, "tool_barrier_affinities", {}).get(
                                    obj.feature_id, set()
                                )
                            )
                        is_relevant_tool = bool(tool_affinities & blocking_barrier_feats) or (
                            not blocking_barrier_feats and bool(tool_affinities)
                        )
                    is_held_tool = engine.working_memory.is_holding(obj.feature_id)

                    # Check remote switch/trigger records
                    is_known_trigger = obj.object_id in self.ledger.unlocked_by_object or any(
                        (
                            m.trigger_pos in obj.cells
                            or (
                                m.trigger_feature is not None
                                and m.trigger_feature == obj.feature_id
                            )
                        )
                        and (
                            not blocking_barrier_feats
                            or (
                                m.prior_value is not None
                                and m.prior_value in blocking_barrier_feats
                            )
                        )
                        for m in engine.state_mutations
                    )

                    if is_relevant_tool or is_held_tool or is_known_trigger:
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
                    return self._finalize_option_path(
                        engine, chosen_obj, chosen_path, available_actions, movable_actions
                    )

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
                    consec_count = self.ledger.get_consecutive_interaction_count(obj.object_id)
                    untouched_bonus = 100.0 if visit_count == 0 else 0.0
                    recency_penalty = 15.0 * float(visit_count)
                    # Habituation penalty for repetitive manipulation of the same object
                    habituation_penalty = (
                        200.0 if consec_count >= 2 else (60.0 * float(consec_count))
                    )
                    if path_len == 0 and consec_count >= 1:
                        # Avatar is already at stance and just manipulated this object
                        habituation_penalty += 300.0

                    # Under strict step budget, prioritize nearby objects to unlock exits before timeout
                    distance_penalty = float(path_len) * (2.0 if is_strict_budget else 0.5)
                    small_item_bonus = 25.0 if obj.category == ObjectCategory.KEY else 0.0
                    color_affinity_bonus = (
                        80.0
                        if (
                            obj.category != ObjectCategory.BARRIER
                            and obj.feature_id in blocking_barrier_feats
                        )
                        else 0.0
                    )
                    barrier_penalty = 150.0 if obj.category == ObjectCategory.BARRIER else 0.0
                    inert_penalty = 120.0 if self.ledger.is_inert(obj.object_id) else 0.0

                    # Penalize objects known to trigger effects unrelated to blocking barriers
                    unrelated_mutation_penalty = 0.0
                    known_affs = getattr(engine.symbolic_theory, "tool_barrier_affinities", {}).get(
                        obj.feature_id, set()
                    )
                    if (
                        blocking_barrier_feats
                        and known_affs
                        and known_affs.isdisjoint(blocking_barrier_feats)
                    ):
                        unrelated_mutation_penalty = 80.0
                    elif any(
                        (m.trigger_feature == obj.feature_id or m.trigger_pos in obj.cells)
                        and blocking_barrier_feats
                        and (
                            m.prior_value is not None
                            and m.prior_value not in blocking_barrier_feats
                        )
                        for m in engine.state_mutations
                    ):
                        unrelated_mutation_penalty = 80.0

                    score = (
                        untouched_bonus
                        + small_item_bonus
                        + color_affinity_bonus
                        + obj.salience_score * 0.2
                        - recency_penalty
                        - distance_penalty
                        - inert_penalty
                        - habituation_penalty
                        - unrelated_mutation_penalty
                        - barrier_penalty
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
                return self._finalize_option_path(
                    engine, best_obj, best_path, available_actions, movable_actions
                )

            elif non_exit_objects:
                # ── TARGET-DIRECTED HEADING & MISSING AXIS DISCOVERY ──────────────
                # When complete A* path cannot be computed (e.g. uncalibrated actions or obstacles),
                # advance towards the nearest stance of the highest-saliency candidate object.
                best_candidate_obj = max(
                    non_exit_objects,
                    key=lambda o: (
                        (100.0 if self.ledger.get_interaction_count(o.object_id) == 0 else 0.0)
                        + (o.salience_score or 10.0)
                        - 15.0 * float(self.ledger.get_interaction_count(o.object_id))
                    ),
                )
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

                    # 2. Greedy displacement step towards target stance
                    best_step_act = None
                    min_dist = abs(dr_target) + abs(dc_target)
                    for act, dr, dc in movable_actions:
                        if failed_transitions and (av_pos, act) in failed_transitions:
                            continue
                        if engine.inhibited_actions.get(act, 0) > 0:
                            continue
                        nr, nc = av_pos[0] + dr, av_pos[1] + dc
                        if not (0 <= nr < H and 0 <= nc < W):
                            continue
                        if not percept.is_passable(nr, nc):
                            continue
                        if patroller_cells and (nr, nc) in patroller_cells:
                            continue
                        cand_dist = abs(target_stance[0] - nr) + abs(target_stance[1] - nc)
                        if cand_dist < min_dist:
                            min_dist = cand_dist
                            best_step_act = (act, nr, nc)

                    if best_step_act is not None:
                        from hbllm.hcir.world.cortex_planner import MentalSimulationStep

                        self.active_target_object_id = best_candidate_obj.object_id
                        return [
                            MentalSimulationStep(
                                action=best_step_act[0],
                                predicted_avatar_pos=(best_step_act[1], best_step_act[2]),
                                expected_mutation="APPROACH_OBJECT",
                            )
                        ]

        # 2B. Spatial Effector (Click) Object Interaction
        if spatial_effector_actions:
            eff_act = spatial_effector_actions[0]
            from hbllm.hcir.world.cortex_planner import MentalSimulationStep

            quiescent_coords = getattr(engine, "quiescent_click_targets", set())
            quiescent_feats = getattr(engine, "quiescent_features", set())

            # Check if this effector action defines discrete valid targets
            eff_aff = getattr(engine, "action_affordances", {}).get(eff_act)
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
                        if 0 <= vx <= 63 and 0 <= vy <= 63:
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
                    vis = getattr(engine, "entity_visit_counts", {}).get(f"click_{vy}_{vx}", 0)
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
                                    expected_mutation="CLICK_UNLOCK",
                                )
                            ]
                    else:
                        self.active_target_object_id = obj.object_id
                        return [
                            MentalSimulationStep(
                                action=eff_act,
                                action_data={"x": tc, "y": tr},
                                predicted_avatar_pos=av_pos or (tr, tc),
                                expected_mutation="CLICK_UNLOCK",
                            )
                        ]

            # DIRECTIVE 5 for effectors: Probe highest-saliency object with unprobed cell selection & quiescence avoidance
            from hbllm.hcir.world.cortex_motor import MotorCortexEffector

            clickable_candidates: list[
                tuple[SalientObject, tuple[int, int], dict[str, Any], float]
            ] = []
            for obj in non_exit_objects:
                unprobed_cells = [c for c in obj.cells if c not in quiescent_coords]
                if not unprobed_cells:
                    continue
                # Prioritize non-frame unprobed cells if available
                non_frame_unprobed = [
                    c
                    for c in unprobed_cells
                    if not (
                        grid_arr is not None
                        and MotorCortexEffector.is_peripheral_frame_cell(c[0], c[1], H, W, grid_arr)
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
                    MotorCortexEffector.is_peripheral_frame_cell(target_r, target_c, H, W, grid_arr)
                    if grid_arr is not None
                    else False
                )
                frame_pen = 150.0 if is_frame_cell else 0.0
                saliency = obj.salience_score or (100.0 / math.log2(2 + max(1, obj.area)))
                score = (
                    untouched_bonus + saliency - ior_penalty - quiescent_pen - frame_pen - inert_pen
                )
                clickable_candidates.append((obj, (target_r, target_c), action_data, score))

            if clickable_candidates:
                clickable_candidates.sort(key=lambda x: x[3], reverse=True)
                chosen_obj, chosen_target_cell, chosen_act_data, chosen_score = (
                    clickable_candidates[0]
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
                return [
                    MentalSimulationStep(
                        action=eff_act,
                        action_data=chosen_act_data,
                        predicted_avatar_pos=av_pos or (tr, tc),
                        expected_mutation="CLICK_OBJECT",
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
                        key=lambda item: getattr(engine, "entity_visit_counts", {}).get(
                            f"click_{item[0]}_{item[1]}", 0
                        )
                    )
                    best_vy, best_vx, best_vt = unquiescent_vts[0]
                    logger.info(
                        "ObjectStateGraphPlanner: Directive 5B (Direct Valid Target)! Probing discrete target "
                        "[(%d,%d), action_data=%s].",
                        best_vy,
                        best_vx,
                        best_vt,
                    )
                    return [
                        MentalSimulationStep(
                            action=eff_act,
                            action_data=dict(best_vt),
                            predicted_avatar_pos=av_pos or (best_vy, best_vx),
                            expected_mutation="CLICK_OBJECT",
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

        interact_act = (
            engine.get_interaction_action(available_actions)
            if hasattr(engine, "get_interaction_action")
            else None
        )

        if not path:
            # Avatar is already at or adjacent to the object stance!
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

        # If the object is solid and the stance is adjacent to it, execute terminal interaction
        if obj.is_solid and target_stance not in obj.cells:
            if interact_act is not None:
                interact_step = MentalSimulationStep(
                    action=interact_act,
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


# ── W038: Constraint-Aware Relational Graph Matcher ──────────────────────────


@dataclass
class RelationalNode:
    """Attributed spatial entity in relational scene graph."""

    node_id: str
    feature_id: int = 0
    area: int = 0
    centroid: tuple[float, float] = (0.0, 0.0)
    bounding_box: tuple[int, int, int, int] = (0, 0, 0, 0)
    role: EntityRole = EntityRole.UNKNOWN
    properties: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        node_id: str,
        feature_id: int = 0,
        area: int = 0,
        centroid: tuple[float, float] = (0.0, 0.0),
        bounding_box: tuple[int, int, int, int] = (0, 0, 0, 0),
        role: EntityRole = EntityRole.UNKNOWN,
        properties: dict[str, Any] | None = None,
        color: int | None = None,
    ) -> None:
        self.node_id = node_id
        self.feature_id = color if color is not None else feature_id
        self.area = area
        self.centroid = centroid
        self.bounding_box = bounding_box
        self.role = role
        self.properties = properties or {}

    @property
    def color(self) -> int:
        """Backward-compatibility alias for feature_id."""
        return self.feature_id

    @classmethod
    def from_spatial_entity(cls, entity: SpatialEntity) -> RelationalNode:
        """Lift an HCIR SpatialEntity component into a RelationalNode."""
        min_r, max_r, min_c, max_c = entity.bounding_box
        return cls(
            node_id=entity.id,
            feature_id=entity.feature_id if entity.feature_id is not None else 0,
            area=entity.area,
            centroid=entity.centroid,
            bounding_box=(min_r, min_c, max_r, max_c),
            role=entity.role,
            properties=dict(entity.properties),
        )


@dataclass
class RelationalEdge:
    """Attributed relation between two entities."""

    source_id: str
    target_id: str
    relation: str  # "adjacent", "contains", "left_of", "right_of", "above", "below"


class RelationalGraphMatcher:
    """Constraint-aware attributed subgraph matcher with exponential worst-case bounding (W038).

    Addresses NP-complete subgraph isomorphism by:
    1. Pruning domain candidates using semantic compatibility (feature_id, role, area, degree).
    2. Enforcing structural relational consistency along attributed edges.
    3. Enforcing a hard search step budget (max_steps) with backtracking to prevent freeze.
    """

    def __init__(self, max_steps: int = 5000) -> None:
        self.max_steps = max_steps

    def find_mapping(
        self,
        source_nodes: Sequence[RelationalNode],
        source_edges: Sequence[RelationalEdge],
        target_nodes: Sequence[RelationalNode],
        target_edges: Sequence[RelationalEdge],
    ) -> dict[str, str] | None:
        """Find bijective mapping f: source_id -> target_id satisfying all constraints."""
        if len(source_nodes) > len(target_nodes):
            return None

        # Build edge relation lookup tables: (u, v) -> set(relations)
        src_adj: dict[tuple[str, str], set[str]] = {}
        for e in source_edges:
            src_adj.setdefault((e.source_id, e.target_id), set()).add(e.relation)

        tgt_adj: dict[tuple[str, str], set[str]] = {}
        for e in target_edges:
            tgt_adj.setdefault((e.source_id, e.target_id), set()).add(e.relation)

        # 1. Candidate domains: for each u in source, find compatible v in target
        domains: dict[str, list[str]] = {}
        for u in source_nodes:
            candidates: list[str] = []
            for v in target_nodes:
                # Feature constraint
                if u.feature_id != v.feature_id:
                    continue
                # Role constraint
                if (
                    u.role != EntityRole.UNKNOWN
                    and v.role != EntityRole.UNKNOWN
                    and u.role != v.role
                ):
                    continue
                # Area constraint (within 20% or equal)
                if abs(u.area - v.area) > max(1, 0.2 * u.area):
                    continue
                candidates.append(v.node_id)
            if not candidates:
                return None
            domains[u.node_id] = candidates

        # Sort source nodes by MRV (Minimum Remaining Values heuristic)
        sorted_u_ids = sorted(domains.keys(), key=lambda uid: len(domains[uid]))

        mapping: dict[str, str] = {}
        used_target_ids: set[str] = set()
        steps = [0]

        def backtrack(idx: int) -> bool:
            if idx == len(sorted_u_ids):
                return True
            steps[0] += 1
            if steps[0] > self.max_steps:
                logger.warning(
                    "RelationalGraphMatcher: search budget (%d steps) exceeded", self.max_steps
                )
                return False

            u = sorted_u_ids[idx]
            for v in domains[u]:
                if v in used_target_ids:
                    continue

                # Relational consistency check with already assigned neighbors
                consistent = True
                for prev_u, prev_v in mapping.items():
                    # Check edge (u, prev_u)
                    if (u, prev_u) in src_adj:
                        required_rels = src_adj[(u, prev_u)]
                        if (v, prev_v) not in tgt_adj or not required_rels.issubset(
                            tgt_adj[(v, prev_v)]
                        ):
                            consistent = False
                            break
                    # Check edge (prev_u, u)
                    if (prev_u, u) in src_adj:
                        required_rels = src_adj[(prev_u, u)]
                        if (prev_v, v) not in tgt_adj or not required_rels.issubset(
                            tgt_adj[(prev_v, v)]
                        ):
                            consistent = False
                            break

                if not consistent:
                    continue

                mapping[u] = v
                used_target_ids.add(v)

                if backtrack(idx + 1):
                    return True

                del mapping[u]
                used_target_ids.remove(v)

            return False

        success = backtrack(0)
        return mapping if success else None
