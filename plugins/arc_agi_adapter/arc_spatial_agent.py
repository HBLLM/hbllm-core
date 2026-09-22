"""ARC-3 Spatial Cognitive Agent — Autonomous, General Spatial Intelligence via HCIR.

Delegates scene lifting, topological cut-set discovery, state-space sequence planning,
and constraint induction to HCIRSpatialEntityPlanner, PhysicsPredictor, and native HCIR memory.
"""

from __future__ import annotations

import logging
import math
import sys
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict

_core_dir = str(Path(__file__).resolve().parent.parent.parent)
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

import numpy as np

from hbllm.hcir.graph import (
    ActionNode,
    EpisodeNode,
    GoalNode,
    NodeLifecycle,
    PhysicalEntityNode,
    Provenance,
    Scope,
    WorldVariableNode,
)
from hbllm.hcir.spatial_planner import (
    EntityGraph,
    EntityRole,
    HCIRSpatialEntityPlanner,
    SequencePlanStep,
    SpatialEntity,
)
from hbllm.hcir.subgoal_decomposer import HierarchicalGoalDecomposer
from hbllm.hcir.topological_cut_set import TopologicalCutSetAnalyzer
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.workspace_tiers import InterruptionCheckpoint
from hbllm.hcir.world.predictors.physics import PhysicsPredictor
from plugins.arc_agi_adapter.arc_agi_runner import (
    ARCGrid,
    GridTopologyExtractor,
)
from plugins.arc_agi_adapter.control_mode import ControlContext

logger = logging.getLogger(__name__)

# Graceful import of official Arcade & arcengine
try:
    from arc_agi import Arcade
except ImportError:
    Arcade = None

try:
    from arcengine import GameAction as ARCGameAction
    from arcengine import GameState as ARCGameState
except ImportError:

    class _FallbackARCGameAction(Enum):
        RESET = 0
        ACTION1 = 1
        ACTION2 = 2
        ACTION3 = 3
        ACTION4 = 4
        ACTION5 = 5
        ACTION6 = 6
        ACTION7 = 7

    class _FallbackARCGameState(Enum):
        NOT_FINISHED = "NOT_FINISHED"
        WIN = "WIN"
        GAME_OVER = "GAME_OVER"

    ARCGameAction = _FallbackARCGameAction  # type: ignore
    ARCGameState = _FallbackARCGameState  # type: ignore


from hbllm.hcir.world.morphology import MorphologicalConcept, ShapeArchetype
from hbllm.hcir.world.motor_calibration import ActionDynamicsModel, StateMutationModel
from hbllm.hcir.world.predictors.physics import PhysicsPredictor


class CircuitInfo(TypedDict):
    """Spatial circuit connecting two endpoints with a specific color identifier."""

    endpoints: list[tuple[int, int]]
    color: int
    is_latching: bool


class TopologicalPathPlanner:
    """Computes obstacle-clearing shortest paths over 2D visual grids using core HCIR PhysicsPredictor."""

    @staticmethod
    def find_shortest_path(
        start: tuple[int, int],
        goal: tuple[int, int],
        grid_shape: tuple[int, int],
        barrier_mask: np.ndarray,
        step_size: int = 1,
        footprint_offsets: list[tuple[int, int]] | set[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int]]:
        """Breadth-first search for shortest path avoiding barrier cells via native HCIR PhysicsPredictor."""
        barrier_cells = set(zip(*np.where(barrier_mask)))
        return PhysicsPredictor.compute_geodesic_path(
            start=start,
            goal=goal,
            barrier_cells=barrier_cells,
            grid_shape=grid_shape,
            step_size=step_size,
            footprint_offsets=footprint_offsets,
        )


class CornerDeadlockDetector:
    """Detects irreversible corner deadlocks for pushable objects via core HCIR PhysicsPredictor."""

    @staticmethod
    def is_corner_deadlock(
        box_pos: tuple[int, int],
        barrier_mask: np.ndarray,
        target_positions: set[tuple[int, int]],
        grid_shape: tuple[int, int],
        step_size: int = 1,
    ) -> bool:
        """Returns True if box_pos is in a corner of barriers and not on a target."""
        barrier_cells = set(zip(*np.where(barrier_mask)))
        return PhysicsPredictor.is_corner_deadlock(
            box_pos=box_pos,
            barrier_cells=barrier_cells,
            target_positions=target_positions,
            grid_shape=grid_shape,
            step_size=step_size,
        )


from plugins.arc_agi_adapter.arc_memory import AgentPhase, HCIRCrossGameMemory
from plugins.arc_agi_adapter.arc_perception import ARCPerceptualLifter


class ARC3SpatialCognitiveAgent:
    """Universal Spatial Cognitive Agent powered by HCIRSpatialEntityPlanner."""

    # Shared global memory across game instances in the process
    global_memory: HCIRCrossGameMemory = HCIRCrossGameMemory()

    def __init__(
        self,
        step_size: int = 1,
        enable_soft_restart: bool = False,
        shared_memory: HCIRCrossGameMemory | None = None,
    ) -> None:
        self.step_size: int = step_size
        self.enable_soft_restart: bool = enable_soft_restart
        self.should_soft_restart: bool = False
        self.phase: AgentPhase = AgentPhase.EPISTEMIC_LEARNING
        self.is_first_level_learning: bool = True

        self.avatar_color: int | None = None
        self.avatar_centroid: tuple[float, float] | None = None
        self.action_models: dict[int, ActionDynamicsModel] = {}
        self.available_actions: list[int] = []

        # Cognitive World & Planning Engine
        self.spatial_planner = HCIRSpatialEntityPlanner(step_size=step_size)
        self.workspace = HCIRWorkspaceState()

        # Physical Interaction State
        self.holding_item: bool = False
        self.carried_offset: tuple[float, float] = (0.0, 0.0)
        self.current_facing: tuple[int, int] = (0, 0)
        self.delivered_positions: set[tuple[int, int]] = set()

        # Cumulative Memory across Episodes/Levels
        self.learned_barrier_colors: set[int] = set()
        self.learned_walkable_colors: set[int] = set()
        self.learned_item_colors: set[int] = set()
        self.learned_receptacle_colors: set[int] = set()
        self.target_zone_bounds: tuple[int, int, int, int] | None = None
        self.is_cooperative_handoff: bool = False

        # Navigation & Probing State
        self.probe_step_counter: int = 0
        self.level_step_counter: int = 0
        self.visited_positions: list[tuple[int, int]] = []
        self.blocked_actions: set[int] = set()
        self.stuck_counter: int = 0
        self.last_action: int | None = None
        self.current_plan: list[SequencePlanStep] = []
        self.optimal_task_plan: list[SequencePlanStep] = []
        self.known_barriers: np.ndarray | None = None

        # Child-Like Exploratory State
        self.action_5_affordance: str = "UNKNOWN"
        self.actuator_to_portals: dict[str, set[str]] = {}
        self.latching_portals: set[str] = set()
        self.action_queue: list[int] = []
        self.start_pos: tuple[int, int] | None = None
        self.raw_avatar_centroid: tuple[float, float] | None = None
        self.raw_start_pos: tuple[float, float] | None = None
        self.initial_grid: np.ndarray | None = None
        self.learned_walkable_cells: set[tuple[int, int]] = set()
        self.visited_cells: set[tuple[int, int]] = set()
        self.lattice_offset: tuple[int, int] = (0, 0)
        self.probe_reset_done: bool = False
        self.shape_concepts: dict[tuple[tuple[int, int], ...], MorphologicalConcept] = {}
        self.last_action_data: dict[str, int] | None = None
        self.goal_centroid: tuple[float, float] | None = None
        self.primary_goal_node: Any = None
        self.state_mutations: list[StateMutationModel] = []
        self.pushable_colors: set[int] = set()
        self.control_context: ControlContext = ControlContext()
        self.target_zones: set[tuple[int, int]] = set()
        self.walkable_colors: set[int] = self.learned_walkable_colors
        self.decomposer: HierarchicalGoalDecomposer = HierarchicalGoalDecomposer()
        self.interruption_stack: list[InterruptionCheckpoint] = []
        self.active_goal_node: Any = None
        self.gate_target_cell: tuple[int, int] | None = None
        self.gate_approach_cell: tuple[int, int] | None = None
        self.target_zone_base_colors: set[int] = set()
        self.attempted_pickup_item_id: str | None = None
        self.picked_up_source_position: tuple[int, int] | None = None

        # Persistent Cross-Game Memory
        self.cross_game_memory: HCIRCrossGameMemory = (
            shared_memory if shared_memory is not None else ARC3SpatialCognitiveAgent.global_memory
        )
        self.cross_game_memory.transfer_to_agent(self)

    @classmethod
    def is_spatial_candidate(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Domain-agnostic check if environment possesses 2D movement actions."""
        return any(a in available_actions for a in [1, 2, 3, 4])

    @classmethod
    def is_cooperative_candidate(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Backward-compatible alias for is_spatial_candidate."""
        return cls.is_spatial_candidate(grid, available_actions)

    def reset_episode(self, retain_dynamics: bool = False, is_retry: bool = False) -> None:
        """Reset internal agent state for a new level/episode."""
        if not retain_dynamics:
            self.action_models.clear()
            self.avatar_color = None
            self.step_size = 1
            self.spatial_planner = HCIRSpatialEntityPlanner(step_size=1)
            self.learned_barrier_colors.clear()
            self.learned_walkable_colors.clear()
            self.learned_item_colors.clear()
            self.learned_receptacle_colors.clear()
            self.action_5_affordance = "UNKNOWN"
            self.actuator_to_portals.clear()
            self.latching_portals.clear()
            self.phase = AgentPhase.EPISTEMIC_LEARNING
            self.is_first_level_learning = True
            self.optimal_task_plan.clear()
            # Seed from persistent cross-game memory
            self.cross_game_memory.transfer_to_agent(self)
        else:
            self.spatial_planner = HCIRSpatialEntityPlanner(step_size=self.step_size)
            self.is_first_level_learning = False
            self.phase = AgentPhase.OPTIMAL_EXECUTION

        self.avatar_centroid = None
        self.holding_item = False
        self.carried_offset = (0.0, 0.0)
        self.current_facing = (0, 0)
        if not is_retry:
            self.delivered_positions.clear()
            self.known_barriers = None
        self.visited_positions.clear()
        self.blocked_actions.clear()
        self.stuck_counter = 0
        self.last_action = None
        self.current_plan.clear()
        self.probe_step_counter = 0
        self.level_step_counter = 0
        self.should_soft_restart = False
        self.is_cooperative_handoff = False
        self.target_zone_bounds = None
        self.workspace = HCIRWorkspaceState()
        self.spatial_planner.reset()
        self.active_goal_node = None
        self.gate_target_cell = None
        self.gate_approach_cell = None
        self.attempted_pickup_item_id = None
        self.picked_up_source_position = None

        self.action_queue.clear()
        self.start_pos = None
        self.raw_avatar_centroid = None
        self.raw_start_pos = None
        self.initial_grid = None
        self.learned_walkable_cells.clear()
        self.visited_cells.clear()
        self.probe_reset_done = False

    def soft_restart(self) -> None:
        """Execute a soft restart: reset runtime state while keeping verified optimal task plan."""
        self.avatar_centroid = None
        self.holding_item = False
        self.carried_offset = (0.0, 0.0)
        self.current_facing = (0, 0)
        self.delivered_positions.clear()
        self.visited_positions.clear()
        self.blocked_actions.clear()
        self.stuck_counter = 0
        self.last_action = None
        self.probe_step_counter = 0
        self.level_step_counter = 0
        self.should_soft_restart = False
        self.phase = AgentPhase.OPTIMAL_EXECUTION

        if self.optimal_task_plan:
            self.current_plan = list(self.optimal_task_plan)
        else:
            self.current_plan.clear()

        logger.info(
            "ARC3SpatialCognitiveAgent soft restarted: initialized OPTIMAL_EXECUTION with %d subgoals.",
            len(self.current_plan),
        )

    def record_episode_outcome(self, completed: bool, reason: str = "") -> None:
        """Record trial outcome in HCIR memory and update persistent cross-game knowledge."""
        if completed:
            self.phase = AgentPhase.COMPLETED
            tasks = [s.target_entity_id for s in self.optimal_task_plan]
            self.cross_game_memory.record_success(
                session_id=f"lvl_{self.level_step_counter}",
                tasks=tasks,
                score=1.0,
            )
            self.cross_game_memory.update_from_agent(self)
            if self.workspace:
                ep_node = EpisodeNode(
                    id=f"ep_win_{uuid.uuid4().hex[:6]}",
                    summary="Level completed successfully via HCIR spatial sequence",
                    outcome="WIN",
                    reward=1.0,
                    lifecycle=NodeLifecycle.ACTIVE,
                    provenance=Provenance(created_by="ARC3SpatialCognitiveAgent"),
                    scope=Scope(tenant_id="default"),
                    properties={
                        "steps": self.level_step_counter,
                        "optimal_tasks": tasks,
                    },
                    tags=["episode_success", "win"],
                )
                self.workspace.upsert_node(ep_node)
        else:
            fail_pos = (
                (int(round(self.avatar_centroid[0])), int(round(self.avatar_centroid[1])))
                if self.avatar_centroid
                else (0, 0)
            )
            self.cross_game_memory.record_failure(
                session_id=f"lvl_{self.level_step_counter}",
                reason=reason or "incomplete",
                failed_action=self.last_action or 0,
                failure_pos=fail_pos,
            )
            self.cross_game_memory.update_from_agent(self)

    def export_knowledge(self) -> dict[str, Any]:
        """Export serialized cross-game knowledge dictionary."""
        self.cross_game_memory.update_from_agent(self)
        return self.cross_game_memory.export_dict()

    def import_knowledge(self, data: dict[str, Any]) -> None:
        """Import cross-game knowledge into memory and update current agent."""
        self.cross_game_memory.import_dict(data)
        self.cross_game_memory.transfer_to_agent(self)

    def _get_or_create_shape_concept(
        self,
        archetype: ShapeArchetype,
        color: int,
        role: EntityRole = EntityRole.UNKNOWN,
        name_prefix: str = "shape",
    ) -> MorphologicalConcept:
        """Retrieve existing shape concept or instantiate a new disentangled archetype concept."""
        cid = archetype.canonical_id
        if cid not in self.shape_concepts:
            name = f"{name_prefix}_{archetype.area}p_{archetype.height}x{archetype.width}"
            self.shape_concepts[cid] = MorphologicalConcept(
                canonical_id=cid,
                archetype=archetype,
                canonical_name=name,
                inferred_role=role,
                observed_colors={color},
            )
        concept = self.shape_concepts[cid]
        concept.observed_colors.add(color)
        if role != EntityRole.UNKNOWN and concept.inferred_role == EntityRole.UNKNOWN:
            concept.inferred_role = role
        return concept

    def _analyze_morphological_transitions(
        self,
        action_id: int,
        prev_grid: np.ndarray,
        curr_grid: np.ndarray,
    ) -> None:
        """Detect rotating doors, color transitions, and retracting obstacles online via HCIR."""
        if self.avatar_color is None or self.holding_item:
            return

        diff_mask = prev_grid != curr_grid
        if not np.any(diff_mask):
            return

        # Mask out avatar pixels if known
        non_avatar_diff = diff_mask.copy()
        non_avatar_diff[prev_grid == self.avatar_color] = False
        non_avatar_diff[curr_grid == self.avatar_color] = False

        if not np.any(non_avatar_diff):
            return

        import scipy.ndimage as ndi

        # ── 1. Color State Transitions & Retracting Doors ──
        # Find static pixel clusters that changed color in place
        for p_col in np.unique(prev_grid[non_avatar_diff]):
            if (
                p_col == 0
                or p_col == self.avatar_color
                or p_col in self.learned_walkable_colors
                or p_col in self.learned_item_colors
            ):
                continue
            p_mask = (prev_grid == p_col) & non_avatar_diff
            p_lbls, p_num = ndi.label(p_mask)
            for p_idx in range(1, p_num + 1):
                pts_prev = np.where(p_lbls == p_idx)
                prev_coords = set(zip(pts_prev[0], pts_prev[1]))

                if self.target_zone_bounds:
                    tz = self.target_zone_bounds
                    if any(tz[0] <= r <= tz[1] and tz[2] <= c <= tz[3] for r, c in prev_coords):
                        continue

                # Must be a known barrier or obstacle
                is_known_barrier = (
                    int(p_col) in self.learned_barrier_colors
                    or (
                        self.known_barriers is not None
                        and any(self.known_barriers[r, c] for r, c in prev_coords)
                    )
                    or (
                        self.spatial_planner is not None
                        and any(
                            pos in getattr(self.spatial_planner, "_learned_barriers", set())
                            for pos in prev_coords
                        )
                    )
                )
                if not is_known_barrier:
                    continue

                curr_vals = [int(curr_grid[r, c]) for r, c in prev_coords]
                if not curr_vals:
                    continue
                most_common_curr = max(set(curr_vals), key=curr_vals.count)

                arch_prev = ShapeArchetype.from_coords(prev_coords)
                concept = self._get_or_create_shape_concept(
                    arch_prev, color=int(p_col), role=EntityRole.PORTAL, name_prefix="obstacle"
                )

                # Case A: Transitioned to floor / background 0 (Door opened or retracted)
                if most_common_curr == 0 or most_common_curr in self.learned_walkable_colors:
                    if self.known_barriers is not None:
                        for r, c in prev_coords:
                            self.known_barriers[r, c] = False
                    for r, c in prev_coords:
                        self.spatial_planner.remove_collision_barrier((r, c))
                    concept.is_color_switch = True
                    concept.color_transitions[(action_id, int(p_col))] = int(most_common_curr)
                    concept.passable_colors.add(int(most_common_curr))
                    concept.barrier_colors.discard(int(most_common_curr))
                    concept.inferred_role = EntityRole.PORTAL
                    self.current_plan.clear()
                    self.action_queue.clear()
                    logger.info(
                        "Door/barrier %s unblocked (color %d -> %d) via action %d",
                        concept.canonical_name,
                        p_col,
                        most_common_curr,
                        action_id,
                    )

                # Case B: Mutated color (State switch)
                elif most_common_curr != p_col:
                    concept.is_color_switch = True
                    concept.color_transitions[(action_id, int(p_col))] = int(most_common_curr)
                    concept.observed_colors.add(int(most_common_curr))
                    if most_common_curr in self.learned_walkable_colors:
                        if self.known_barriers is not None:
                            for r, c in prev_coords:
                                self.known_barriers[r, c] = False
                        for r, c in prev_coords:
                            self.spatial_planner.remove_collision_barrier((r, c))
                        self.current_plan.clear()
                        self.action_queue.clear()

        # ── 2. Rotating Door Discovery (Orthogonal 90°/180°/270° Transformation) ──
        for col in np.unique(curr_grid[non_avatar_diff]):
            if (
                col == 0
                or col == self.avatar_color
                or col in self.learned_walkable_colors
                or col in self.learned_item_colors
            ):
                continue
            vanished_mask = (prev_grid == col) & non_avatar_diff & (curr_grid != col)
            appeared_mask = (curr_grid == col) & non_avatar_diff & (prev_grid != col)
            if not np.any(vanished_mask) or not np.any(appeared_mask):
                continue

            v_lbls, v_n = ndi.label(vanished_mask)
            a_lbls, a_n = ndi.label(appeared_mask)

            for vi in range(1, v_n + 1):
                v_pts = np.where(v_lbls == vi)
                v_coords = set(zip(v_pts[0], v_pts[1]))

                if self.target_zone_bounds:
                    tz = self.target_zone_bounds
                    if any(tz[0] <= r <= tz[1] and tz[2] <= c <= tz[3] for r, c in v_coords):
                        continue

                # Check if this vanished cluster was known as a barrier
                is_barrier = (
                    int(col) in self.learned_barrier_colors
                    or (
                        self.known_barriers is not None
                        and any(self.known_barriers[r, c] for r, c in v_coords)
                    )
                    or (
                        self.spatial_planner is not None
                        and any(
                            pos in getattr(self.spatial_planner, "_learned_barriers", set())
                            for pos in v_coords
                        )
                    )
                )
                if not is_barrier:
                    continue

                v_arch = ShapeArchetype.from_coords(v_coords)

                for ai in range(1, a_n + 1):
                    a_pts = np.where(a_lbls == ai)
                    a_coords = set(zip(a_pts[0], a_pts[1]))
                    a_arch = ShapeArchetype.from_coords(a_coords)

                    if v_arch.is_rotation_of(a_arch):
                        v_cr, v_cc = float(np.mean(v_pts[0])), float(np.mean(v_pts[1]))
                        a_cr, a_cc = float(np.mean(a_pts[0])), float(np.mean(a_pts[1]))
                        pivot_dist = math.hypot(v_cr - a_cr, v_cc - a_cc)
                        max_pivot = max(v_arch.height, v_arch.width) * 1.75

                        if pivot_dist <= max_pivot:
                            concept = self._get_or_create_shape_concept(
                                v_arch,
                                color=int(col),
                                role=EntityRole.PORTAL,
                                name_prefix="rotating_door",
                            )
                            concept.is_rotatable = True
                            concept.rotation_trigger = action_id

                            vacated = v_coords - a_coords
                            newly_occupied = a_coords - v_coords

                            if self.known_barriers is None:
                                self.known_barriers = np.zeros(curr_grid.shape, dtype=bool)

                            # Free the vacated passage cells so A* will route through
                            for vr, vc in vacated:
                                self.known_barriers[vr, vc] = False
                                self.spatial_planner.remove_collision_barrier((vr, vc))

                            # Block the newly occupied cells
                            for nr, nc in newly_occupied:
                                self.known_barriers[nr, nc] = True

                            self.current_plan.clear()
                            self.action_queue.clear()
                            logger.info(
                                "Discovered rotating door %s (action %d): %d passage cells unblocked",
                                concept.canonical_name,
                                action_id,
                                len(vacated),
                            )

    def _active_probe_action(self, available_actions: list[int]) -> int:
        """Systematically probe available actions to discover motor displacements."""
        uncalibrated = [
            a
            for a in available_actions
            if a in [1, 2, 3, 4]
            and (a not in self.action_models or self.action_models[a].confidence < 0.8)
        ]
        if uncalibrated:
            return uncalibrated[self.probe_step_counter % len(uncalibrated)]
        directional = [a for a in available_actions if a in [1, 2, 3, 4]]
        if directional:
            return directional[self.probe_step_counter % len(directional)]
        return available_actions[0]

    active_probe_action = _active_probe_action

    def _snap_coord(self, raw_val: float, axis: int | None = None) -> int:
        """Snap float coordinate to discrete lattice coordinate using sprite offset."""
        if self.step_size > 1:
            offset = (self.step_size - 1) * 0.5
            return int(round((raw_val - offset) / self.step_size)) * self.step_size
        return int(round(raw_val))

    def _stand_clear_action(
        self,
        curr_r: float,
        curr_c: float,
        eg: Any,
        available_actions: list[int],
    ) -> tuple[int, float]:
        """Stand clear of active delivery zones so companion bots or environment can complete delivery."""
        doorways = [
            e for e in eg.entities.values() if e.role in (EntityRole.PORTAL, EntityRole.PORTAL)
        ]
        zones_to_avoid: list[tuple[int, int]] = []
        if doorways:
            d = doorways[0]
            appr = d.properties.get("approach_cell", d.grid_pos)
            zones_to_avoid.append(appr)
        if self.target_zone_bounds:
            tz = self.target_zone_bounds
            zones_to_avoid.append(((tz[0] + tz[1]) // 2, (tz[2] + tz[3]) // 2))
        for dp in self.delivered_positions:
            zones_to_avoid.append(dp)

        safe_dist = self.step_size * 2.5
        agent_pos = (int(round(curr_r)), int(round(curr_c)))

        # If already safely away from all active zones, idle safely
        if zones_to_avoid and all(
            math.hypot(curr_r - zr, curr_c - zc) >= safe_dist for zr, zc in zones_to_avoid
        ):
            if 5 in available_actions:
                return 5, 0.95
            return available_actions[0], 0.90

        comp = TopologicalCutSetAnalyzer.get_reachable_component(
            agent_pos, eg.barriers, eg.grid_shape, self.step_size
        )
        safe_cands = [
            p
            for p in comp
            if zones_to_avoid
            and all(math.hypot(p[0] - zr, p[1] - zc) >= safe_dist for zr, zc in zones_to_avoid)
        ]
        if safe_cands:
            safe_cands.sort(key=lambda p: math.hypot(p[0] - curr_r, p[1] - curr_c))
            target_safe = safe_cands[0]
            path = self.spatial_planner.compute_safe_path(
                agent_pos, target_safe, eg.barriers, eg.grid_shape, self.step_size
            )
            if path and len(path) > 1:
                dr = path[1][0] - curr_r
                dc = path[1][1] - curr_c
                act = self.spatial_planner.get_action_for_delta(
                    int(round(dr)), int(round(dc)), self.action_models
                )
                if act is not None and act in available_actions:
                    return act, 0.95
        if 5 in available_actions:
            return 5, 0.95
        return available_actions[0], 0.90

    def _discover_and_plan_temporal_clone(
        self, curr_grid: np.ndarray, available_actions: list[int]
    ) -> list[int] | None:
        """Domain-agnostic discovery and planning for multi-timeline temporal clone puzzles (e.g. g50t)."""
        if self.avatar_color is None or self.step_size <= 1:
            return None

        H, W = curr_grid.shape
        step = self.step_size
        planner = self.spatial_planner

        playable_mask = np.zeros_like(curr_grid, dtype=bool)
        playable_mask[6 : H - 3, 6 : W - 3] = True

        raw_r = (
            self.raw_avatar_centroid[0]
            if getattr(self, "raw_avatar_centroid", None) is not None
            else (self.avatar_centroid[0] if self.avatar_centroid else 0)
        )
        raw_c = (
            self.raw_avatar_centroid[1]
            if getattr(self, "raw_avatar_centroid", None) is not None
            else (self.avatar_centroid[1] if self.avatar_centroid else 0)
        )
        off_r = int(round(raw_r)) % step
        off_c = int(round(raw_c)) % step

        def snap_maze(v: float, off: int) -> int:
            return int(round((v - off) / step)) * step + off

        raw_start_r = (
            self.raw_start_pos[0]
            if getattr(self, "raw_start_pos", None) is not None
            else (self.start_pos[0] if self.start_pos else raw_r)
        )
        raw_start_c = (
            self.raw_start_pos[1]
            if getattr(self, "raw_start_pos", None) is not None
            else (self.start_pos[1] if self.start_pos else raw_c)
        )
        maze_start = (snap_maze(raw_start_r, off_r), snap_maze(raw_start_c, off_c))

        # 1. Discover Goal
        goal_pos = None
        import scipy.ndimage as ndi

        for col in [self.avatar_color] + [
            int(c) for c in np.unique(curr_grid) if c not in (0, 5, self.avatar_color)
        ]:
            target_mask = (curr_grid == col) & playable_mask
            lbls, n = ndi.label(target_mask)
            for i in range(1, n + 1):
                pts = np.where(lbls == i)
                cr, cc = float(np.mean(pts[0])), float(np.mean(pts[1]))
                sr = snap_maze(cr, off_r)
                sc = snap_maze(cc, off_c)
                if math.hypot(sr - maze_start[0], sc - maze_start[1]) >= step * 1.5:
                    goal_pos = (sr, sc)
                    break
            if goal_pos:
                break

        if goal_pos is None:
            return None

        # 2. Discover Circuits
        lattice_all = [(r, c) for r in range(off_r, H, step) for c in range(off_c, W, step)]
        lattice = [
            (r, c) for (r, c) in lattice_all if step <= r < H - step and step <= c < W - step
        ]
        circuits: list[CircuitInfo] = []
        for col in np.unique(curr_grid):
            if col in (0, 5, self.avatar_color):
                continue
            mask = (curr_grid == col) & playable_mask
            c_lbls, c_n = ndi.label(mask)
            for i in range(1, c_n + 1):
                comp = c_lbls == i
                ends = [
                    p
                    for p in lattice
                    if np.sum(
                        comp[
                            max(0, p[0] - 1) : min(H, p[0] + 2),
                            max(0, p[1] - 1) : min(W, p[1] + 2),
                        ]
                    )
                    >= 7
                ]
                if len(ends) == 2:
                    circuits.append(
                        {
                            "endpoints": ends,
                            "color": int(col),
                            "is_latching": (col == 11),
                        }
                    )

        if not circuits:
            return None

        # 3. Walkable lattice & barriers
        walkable_lattice: set[tuple[int, int]] = set()
        for r, c in lattice_all:
            if curr_grid[r, c] == 5 or (r, c) == maze_start or (r, c) == goal_pos:
                walkable_lattice.add((r, c))
        for c in circuits:
            walkable_lattice.update(c["endpoints"])
        base_barriers = set(lattice_all) - walkable_lattice

        # 4. Disambiguate Switch vs Gate
        actuator_entities = []
        portal_entities = []
        act_to_port = {}
        latching_portals = set()

        for c in circuits:
            e1, e2 = c["endpoints"]
            col = c["color"]
            cnt1 = np.sum(curr_grid[e1[0] - 2 : e1[0] + 3, e1[1] - 2 : e1[1] + 3] == col)
            cnt2 = np.sum(curr_grid[e2[0] - 2 : e2[0] + 3, e2[1] - 2 : e2[1] + 3] == col)
            sw, gate = (e1, e2) if cnt1 < cnt2 else (e2, e1)

            act_id = f"act_{sw[0]}_{sw[1]}"
            port_id = f"portal_{gate[0]}_{gate[1]}"
            actuator_entities.append(
                SpatialEntity(
                    id=act_id,
                    role=EntityRole.ACTUATOR,
                    centroid=(float(sw[0]), float(sw[1])),
                    grid_pos=sw,
                    area=9,
                    bounding_box=(sw[0] - 1, sw[0] + 1, sw[1] - 1, sw[1] + 1),
                )
            )
            portal_entities.append(
                SpatialEntity(
                    id=port_id,
                    role=EntityRole.PORTAL,
                    centroid=(float(gate[0]), float(gate[1])),
                    grid_pos=gate,
                    area=9,
                    bounding_box=(gate[0] - 1, gate[0] + 1, gate[1] - 1, gate[1] + 1),
                )
            )
            act_to_port[act_id] = {port_id}
            if c["is_latching"]:
                latching_portals.add(port_id)

        curr_av = (
            (snap_maze(self.avatar_centroid[0], off_r), snap_maze(self.avatar_centroid[1], off_c))
            if self.avatar_centroid
            else maze_start
        )
        plan = planner.plan_temporal_clone_sequence(
            start_pos=maze_start,
            goal_pos=goal_pos,
            actuators=actuator_entities,
            portals=portal_entities,
            actuator_to_portals=act_to_port,
            latching_portals=latching_portals,
            barriers=base_barriers,
            grid_shape=(H, W),
            step_size=step,
            action_models=self.action_models,
            max_clones=3,
            rewind_action=5,
            current_pos=curr_av,
        )
        if plan:
            acts = []
            for ts in plan:
                acts.extend(ts.actions)
            return acts
        return None

    def _plan_geodesic_navigation_step(
        self,
        curr_grid: np.ndarray,
        available_actions: list[int],
        tags: list[str] | None = None,
    ) -> tuple[int, float]:
        """Plans navigation step using geodesic shortest paths, topological cut-sets, and deadlock avoidance."""
        import math

        H, W = curr_grid.shape
        if not hasattr(self, "walkable_colors"):
            self.walkable_colors = self.learned_walkable_colors

        if self.avatar_color is not None:
            avatar_mask = np.where(curr_grid == self.avatar_color)
            if len(avatar_mask[0]) > 0:
                curr_r = float(np.mean(avatar_mask[0]))
                curr_c = float(np.mean(avatar_mask[1]))
                self.avatar_centroid = (curr_r, curr_c)
            elif self.avatar_centroid is not None:
                curr_r, curr_c = self.avatar_centroid
            else:
                curr_r, curr_c = 0.0, 0.0
        elif self.avatar_centroid is not None:
            curr_r, curr_c = self.avatar_centroid
        else:
            curr_r, curr_c = 0.0, 0.0
        H, W = curr_grid.shape

        arc_grid = ARCGrid.from_list(curr_grid.tolist())
        objs = GridTopologyExtractor.extract_objects(arc_grid)

        # Detect level-transition UI overlay frames via general background uniformity
        bg_counts = np.bincount(curr_grid.flatten())
        dom_color = int(bg_counts.argmax())
        dom_frac = float(bg_counts[dom_color]) / curr_grid.size
        self._is_transition_frame = bool(dom_frac > 0.95 and len(objs) <= 1)

        candidate_goals = [
            o
            for o in objs
            if o.color != self.avatar_color
            and o.color != 0
            and (o.color not in self.walkable_colors or o.color in self.learned_receptacle_colors)
            and np.count_nonzero(curr_grid == o.color) < (curr_grid.size * 0.30)
            and (H <= 20 or (2 <= o.centroid[0] < (H - 3) and 2 <= o.centroid[1] < (W - 3)))
            and not (H > 30 and o.centroid[0] > (H - 14) and o.centroid[1] < 15)
            and not (H > 30 and o.centroid[0] > (H - 14) and o.centroid[1] > (W - 15))
        ]

        if not candidate_goals:
            unblocked = [a for a in available_actions if a not in self.blocked_actions]
            act = unblocked[0] if unblocked else available_actions[0]
            if act == 6:
                self.last_action_data = {"x": W // 2, "y": H // 2}
            else:
                self.last_action_data = None
            return act, 0.40

        if self.avatar_color is not None and self.avatar_centroid is None:
            p_avatar = np.where(curr_grid == self.avatar_color)
            if len(p_avatar[0]) > 0:
                self.avatar_centroid = (float(np.mean(p_avatar[0])), float(np.mean(p_avatar[1])))

        if self.known_barriers is None:
            self.known_barriers = np.zeros(curr_grid.shape, dtype=bool)
            if self.learned_barrier_colors:
                for b_col in self.learned_barrier_colors:
                    if (
                        b_col != 0
                        and b_col != self.avatar_color
                        and b_col not in self.walkable_colors
                    ):
                        self.known_barriers[curr_grid == b_col] = True

        barrier_cells: set[tuple[int, int]] = set()
        if self.known_barriers is not None:
            # Macro linear partition detection: thin structures spanning >= 40% of grid dimension
            for o in objs:
                if o.color in (0, self.avatar_color) or o.color in self.walkable_colors:
                    continue
                span_r = o.max_r - o.min_r
                span_c = o.max_c - o.min_c
                if (span_r >= int(H * 0.4) and span_c <= max(4, self.step_size * 2)) or (
                    span_c >= int(W * 0.4) and span_r <= max(4, self.step_size * 2)
                ):
                    for cr, cc in o.coords:
                        self.known_barriers[cr, cc] = True

            if self.target_zone_bounds and not self.holding_item:
                tz_min_r, tz_max_r, tz_min_c, tz_max_c = self.target_zone_bounds
                self.known_barriers[
                    max(0, tz_min_r - 1) : min(H, tz_max_r + 2),
                    max(0, tz_min_c - 1) : min(W, tz_max_c + 2),
                ] = False
            barrier_cells = set(zip(*np.where(self.known_barriers)))

        # Augment barrier_cells with physical objects that obstruct spatial traversal
        tz_base = getattr(self, "target_zone_base_colors", set()) | self.learned_receptacle_colors
        for o in objs:
            if (
                o.color != self.avatar_color
                and o.color != 0
                and o.color not in self.walkable_colors
                and (self.holding_item or o.color not in tz_base)
            ):
                if self.holding_item:
                    cur_off = getattr(self, "carried_offset", (0.0, 0.0))
                    c_ir = curr_r + cur_off[0]
                    c_ic = curr_c + cur_off[1]
                    item_is_delivered = any(
                        math.hypot(o.centroid[0] - dp[0], o.centroid[1] - dp[1])
                        < max(2.5, self.step_size * 1.5)
                        for dp in self.delivered_positions
                    )
                    if not item_is_delivered and math.hypot(
                        o.centroid[0] - c_ir, o.centroid[1] - c_ic
                    ) < max(self.step_size * 1.0, 3.0):
                        continue
                    if getattr(self, "picked_up_source_position", None):
                        p_pos = self.picked_up_source_position
                        if math.hypot(o.centroid[0] - p_pos[0], o.centroid[1] - p_pos[1]) < max(
                            self.step_size * 1.0, 3.0
                        ):
                            continue
                # Exempt goal targets (primary exit or active subgoals) from barrier_cells
                if self.active_goal_node:
                    t_pos = self.active_goal_node.properties.get("target_position")
                    if t_pos and math.hypot(
                        o.centroid[0] - t_pos[0], o.centroid[1] - t_pos[1]
                    ) < max(2.5, self.step_size * 1.5):
                        continue
                    i_pos = self.active_goal_node.properties.get("item_position")
                    if i_pos and math.hypot(
                        o.centroid[0] - i_pos[0], o.centroid[1] - i_pos[1]
                    ) < max(2.5, self.step_size * 1.5):
                        continue
                if self.primary_goal_node:
                    pt_pos = self.primary_goal_node.properties.get("target_position")
                    if pt_pos and math.hypot(
                        o.centroid[0] - pt_pos[0], o.centroid[1] - pt_pos[1]
                    ) < max(2.5, self.step_size * 1.5):
                        continue
                for cr, cc in o.coords:
                    barrier_cells.add((cr, cc))

        # Add delivered item footprints as solid obstacles
        for dp in self.delivered_positions:
            d_ir, d_ic = int(round(dp[0])), int(round(dp[1]))
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if 0 <= d_ir + dr < H and 0 <= d_ic + dc < W:
                        barrier_cells.add((d_ir + dr, d_ic + dc))

        # Identify primary destination: prioritized by learned receptacle colors, or distinct target zone
        target_zones = []
        if self.learned_receptacle_colors:
            receptacle_objs = [
                o
                for o in candidate_goals
                if o.color in self.learned_receptacle_colors
                and not (
                    self.known_barriers is not None
                    and self.known_barriers[int(round(o.centroid[0])), int(round(o.centroid[1]))]
                )
            ]
            if receptacle_objs:
                target_zones = receptacle_objs

        if not target_zones:
            target_zones = [
                o
                for o in candidate_goals
                if (getattr(o, "is_frame", False) and o.area > 16)
                or (
                    16 <= o.area <= 160
                    and (o.max_r - o.min_r >= 4)
                    and (o.max_c - o.min_c >= 4)
                    and o.color not in (self.avatar_color, 0)
                    and (
                        o.color not in self.walkable_colors
                        or o.color in self.learned_receptacle_colors
                    )
                )
            ]
        target_pos: tuple[int, int]
        if target_zones:
            # Pick the best distinct target zone entity (prefer learned receptacle colors, then enclosed frames, then area)
            best_tz = max(
                target_zones,
                key=lambda o: (
                    2
                    if o.color in self.learned_receptacle_colors
                    else (1 if getattr(o, "is_frame", False) else 0),
                    o.area,
                ),
            )
            fresh_tz_min_r = best_tz.min_r
            fresh_tz_max_r = best_tz.max_r
            fresh_tz_min_c = best_tz.min_c
            fresh_tz_max_c = best_tz.max_c
            fresh_bounds = (fresh_tz_min_r, fresh_tz_max_r, fresh_tz_min_c, fresh_tz_max_c)
            fresh_area = (fresh_tz_max_r - fresh_tz_min_r + 1) * (
                fresh_tz_max_c - fresh_tz_min_c + 1
            )

            # Update target_zone_bounds if:
            # - Not yet set
            # - Previous bounds were set during a transition frame (stale)
            # - A genuine target zone is found that is significantly different/larger
            should_update = False
            if self.target_zone_bounds is None:
                should_update = True
            elif not getattr(self, "_is_transition_frame", False):
                old_area = (self.target_zone_bounds[1] - self.target_zone_bounds[0] + 1) * (
                    self.target_zone_bounds[3] - self.target_zone_bounds[2] + 1
                )
                # If a target zone is found that is significantly different or relocated, re-detect
                has_receptacle = any(
                    (
                        o.color in self.learned_receptacle_colors
                        or getattr(o, "is_frame", False)
                        or o.area >= 24
                    )
                    for o in target_zones
                )
                bounds_differ = (
                    abs(fresh_bounds[0] - self.target_zone_bounds[0]) > self.step_size * 2
                    or abs(fresh_bounds[2] - self.target_zone_bounds[2]) > self.step_size * 2
                )
                if (has_receptacle or self.learned_receptacle_colors) and (
                    bounds_differ or fresh_area > old_area * 1.5
                ):
                    should_update = True

            if should_update and not getattr(self, "_is_transition_frame", False):
                self.target_zone_bounds = fresh_bounds
                tz_patch = curr_grid[
                    fresh_tz_min_r : fresh_tz_max_r + 1, fresh_tz_min_c : fresh_tz_max_c + 1
                ]
                self.target_zone_base_colors = {
                    int(c)
                    for c in np.unique(tz_patch)
                    if c not in (0, self.avatar_color)
                    and c not in self.walkable_colors
                    and c not in self.learned_barrier_colors
                    and np.count_nonzero(tz_patch == c) >= max(16, int(tz_patch.size * 0.20))
                }
                self.learned_receptacle_colors.update(self.target_zone_base_colors)
                self.learned_receptacle_bounds = self.target_zone_bounds

            closest_tz = min(
                target_zones,
                key=lambda o: math.hypot(o.centroid[0] - curr_r, o.centroid[1] - curr_c),
            )
            target_pos = (int(round(closest_tz.centroid[0])), int(round(closest_tz.centroid[1])))
            self.primary_goal_node = GoalNode(
                id="g_primary_exit",
                description="Reach exit or deliver to target zone",
                properties={"target_position": target_pos},
            )
        elif self.primary_goal_node is None:
            furthest_target = max(
                candidate_goals,
                key=lambda o: math.hypot(o.centroid[0] - curr_r, o.centroid[1] - curr_c),
            )
            target_pos = (
                int(round(furthest_target.centroid[0])),
                int(round(furthest_target.centroid[1])),
            )
            self.primary_goal_node = GoalNode(
                id="g_primary_exit",
                description="Reach exit or deliver to target zone",
                properties={"target_position": target_pos},
            )
        else:
            t_prop = self.primary_goal_node.properties.get("target_position")
            target_pos = (
                (int(round(t_prop[0])), int(round(t_prop[1])))
                if t_prop
                else (int(round(curr_r)), int(round(curr_c)))
            )

        # Snapshot current target zone bounds AFTER detection/re-detection
        tz_bounds = self.target_zone_bounds

        def is_in_zone(o: Any) -> bool:
            if tz_bounds:
                return bool(
                    (tz_bounds[0] - 1) <= o.centroid[0] <= (tz_bounds[1] + 1)
                    and (tz_bounds[2] - 1) <= o.centroid[1] <= (tz_bounds[3] + 1)
                )
            return bool(
                math.hypot(o.centroid[0] - target_pos[0], o.centroid[1] - target_pos[1])
                <= max(2.0, self.step_size * 2.2)
            )

        def is_delivered(o: Any) -> bool:
            if is_in_zone(o):
                return True
            return any(
                math.hypot(o.centroid[0] - dp[0], o.centroid[1] - dp[1])
                < max(2.5, self.step_size * 1.1)
                for dp in self.delivered_positions
            )

        tz_colors = getattr(self, "target_zone_base_colors", set())
        if not tz_colors and target_zones:
            tz_colors = {tz.color for tz in target_zones}

        max_item_area = max(36, int(self.step_size * self.step_size * 2.5))

        def is_barrier_obj(o: Any) -> bool:
            span_r = o.max_r - o.min_r
            span_c = o.max_c - o.min_c
            if (span_r >= int(H * 0.4) and span_c <= max(4, self.step_size * 2)) or (
                span_c >= int(W * 0.4) and span_r <= max(4, self.step_size * 2)
            ):
                return True
            if o.area <= max_item_area:
                return False
            if self.known_barriers is None:
                return False
            r, c = int(round(o.centroid[0])), int(round(o.centroid[1]))
            if 0 <= r < H and 0 <= c < W and self.known_barriers[r, c]:
                return True
            return False

        # Filter candidate items: prioritize free uncarried, undelivered, and non-barrier items
        item_excluded = (
            {self.avatar_color} | tz_colors | self.learned_barrier_colors | self.walkable_colors
        )
        uncarried_items = [
            o
            for o in candidate_goals
            if not is_delivered(o)
            and not is_barrier_obj(o)
            and o.area <= max_item_area
            and o.color not in item_excluded
        ]
        # Prioritize objects matching learned item colors from earlier versions/levels
        if self.learned_item_colors:
            known_item_objs = [o for o in uncarried_items if o.color in self.learned_item_colors]
            if known_item_objs:
                uncarried_items = known_item_objs

        if uncarried_items:
            candidate_items = uncarried_items
        else:
            fallback_excluded = (
                {self.avatar_color} | tz_colors | self.learned_barrier_colors | self.walkable_colors
            )
            candidate_items = [
                o
                for o in candidate_goals
                if not is_delivered(o)
                and not is_barrier_obj(o)
                and o.area <= max_item_area
                and o.color not in fallback_excluded
            ]
            if self.learned_item_colors:
                known_fallback = [o for o in candidate_items if o.color in self.learned_item_colors]
                if known_fallback:
                    candidate_items = known_fallback

        # Gestalt grouping: unify concentric / co-located multi-color components into single composite entities
        deduped_items = []
        for o in candidate_items:
            matched = False
            for idx, existing in enumerate(deduped_items):
                if math.hypot(
                    o.centroid[0] - existing.centroid[0], o.centroid[1] - existing.centroid[1]
                ) < max(2.5, self.step_size * 0.6):
                    matched = True
                    if o.area > existing.area:
                        deduped_items[idx] = o
                    break
            if not matched:
                deduped_items.append(o)
        candidate_items = deduped_items

        gate_dir: tuple[int, int] | None = None
        if self.primary_goal_node:
            t_prop = self.primary_goal_node.properties.get("target_position")
            if t_prop:
                cut_res = TopologicalCutSetAnalyzer.analyze_cut_set(
                    (int(round(curr_r)), int(round(curr_c))),
                    (int(t_prop[0]), int(t_prop[1])),
                    barrier_cells,
                    (H, W),
                    self.step_size,
                )
                if cut_res.is_partitioned and cut_res.best_gate_cell and cut_res.approach_cell:
                    dr_g = int(np.sign(cut_res.best_gate_cell[0] - cut_res.approach_cell[0]))
                    dc_g = int(np.sign(cut_res.best_gate_cell[1] - cut_res.approach_cell[1]))
                    if dr_g != 0 or dc_g != 0:
                        gate_dir = (dr_g, dc_g)

        affordance_type = "INTERACTION" if 5 in available_actions else "CONTACT"
        candidate_subgoals = []
        for o in candidate_items:
            o_r, o_c = int(round(o.centroid[0])), int(round(o.centroid[1]))
            # Determine best adjacent approach cell (stand position)
            adj_cells = [
                (o_r - int(self.step_size), o_c),
                (o_r + int(self.step_size), o_c),
                (o_r, o_c - int(self.step_size)),
                (o_r, o_c + int(self.step_size)),
            ]
            valid_adj = [
                (ar, ac)
                for (ar, ac) in adj_cells
                if 0 <= ar < H and 0 <= ac < W and (ar, ac) not in barrier_cells
            ]
            if valid_adj:
                if gate_dir is not None:
                    preferred = (
                        o_r - gate_dir[0] * int(self.step_size),
                        o_c - gate_dir[1] * int(self.step_size),
                    )
                    if preferred in valid_adj:
                        sub_pos = preferred
                    else:
                        sub_pos = min(
                            valid_adj, key=lambda p: math.hypot(p[0] - curr_r, p[1] - curr_c)
                        )
                else:
                    sub_pos = min(valid_adj, key=lambda p: math.hypot(p[0] - curr_r, p[1] - curr_c))
            else:
                sub_pos = (o_r, o_c)

            candidate_subgoals.append(
                {
                    "id": f"sub_{o.color}_{o_r}_{o_c}",
                    "position": sub_pos,
                    "item_position": (o_r, o_c),
                    "color": int(o.color),
                    "area": o.area,
                    "affordance": affordance_type,
                    "description": f"Prerequisite element color={o.color} at ({o_r}, {o_c})",
                }
            )

        # Item holding / delivery state machine
        if self.holding_item:
            offset = getattr(self, "carried_offset", (0.0, 0.0))
            max_off = max(1.0, float(self.step_size * 1.5))
            offset = (
                float(np.clip(offset[0], -max_off, max_off)),
                float(np.clip(offset[1], -max_off, max_off)),
            )
            item_r = curr_r + offset[0]
            item_c = curr_c + offset[1]

            # Find open delivery slot in target zone bounds
            open_target: tuple[int, int] | None = None
            if tz_bounds:
                tz_min_r, tz_max_r, tz_min_c, tz_max_c = tz_bounds
                # Adjust slot boundaries if the zone border is walled with barriers
                start_r = (
                    tz_min_r + 1
                    if any((tz_min_r, c) in barrier_cells for c in range(tz_min_c, tz_max_c + 1))
                    else tz_min_r
                )
                start_c = (
                    tz_min_c + 1
                    if any((r, tz_min_c) in barrier_cells for r in range(tz_min_r, tz_max_r + 1))
                    else tz_min_c
                )
                end_r = (
                    tz_max_r - 1
                    if any((tz_max_r, c) in barrier_cells for c in range(tz_min_c, tz_max_c + 1))
                    else tz_max_r
                )
                end_c = (
                    tz_max_c - 1
                    if any((r, tz_max_c) in barrier_cells for r in range(tz_min_r, tz_max_r + 1))
                    else tz_max_c
                )

                slot_step = max(1, (self.step_size + 1) // 2)
                slots: list[tuple[int, int]] = []
                for sr in range(start_r, end_r + 1, slot_step):
                    for sc in range(start_c, end_c + 1, slot_step):
                        if (sr, sc) not in barrier_cells:
                            slots.append((sr, sc))
                if not slots:
                    slots.append(((tz_min_r + tz_max_r) // 2, (tz_min_c + tz_max_c) // 2))

                def is_slot_occupied(sr: int, sc: int) -> bool:
                    if (sr, sc) in barrier_cells:
                        return True
                    if (
                        self.known_barriers is not None
                        and (0 <= sr < H and 0 <= sc < W)
                        and self.known_barriers[sr, sc]
                    ):
                        return True
                    for dr, dc in self.delivered_positions:
                        if math.hypot(sr - dr, sc - dc) < slot_step * 0.8:
                            return True
                    # Check for non-zone objects blocking the slot.
                    tz_base = getattr(self, "target_zone_base_colors", set())
                    occupancy_colors = [c for c in [4, 5] if c not in tz_base]
                    if occupancy_colors and 0 <= sr < H and 0 <= sc < W:
                        patch = curr_grid[
                            max(0, sr - 1) : min(H, sr + 2), max(0, sc - 1) : min(W, sc + 2)
                        ]
                        if np.any(np.isin(patch, occupancy_colors)):
                            return True
                    return False

                unoccupied_slots: list[tuple[int, int]] = [
                    s for s in slots if not is_slot_occupied(s[0], s[1])
                ]

                # Prefer slots where the avatar's delivery position
                # (slot - offset) is OUTSIDE the zone.  This lets the
                # agent approach from the side without walking through
                # previously delivered packages inside the zone.
                def delivery_pos_outside_zone(slot: tuple[int, int]) -> bool:
                    dr = slot[0] - offset[0]
                    dc = slot[1] - offset[1]
                    return not (tz_min_r <= dr <= tz_max_r and tz_min_c <= dc <= tz_max_c)

                candidates = unoccupied_slots if unoccupied_slots else slots
                outside_slots: list[tuple[int, int]] = [
                    s for s in candidates if delivery_pos_outside_zone(s)
                ]
                pref_delivery_slots: list[tuple[int, int]] = (
                    outside_slots if outside_slots else candidates
                )

                if pref_delivery_slots:

                    def slot_geodesic_dist(s: tuple[int, int]) -> float:
                        deliv_stand_r = max(0, min(H - 1, int(round(s[0] - offset[0]))))
                        deliv_stand_c = max(0, min(W - 1, int(round(s[1] - offset[1]))))
                        if (deliv_stand_r, deliv_stand_c) in barrier_cells:
                            return 99999.0
                        if (
                            tz_bounds is not None
                            and tz_min_r <= deliv_stand_r <= tz_max_r
                            and tz_min_c <= deliv_stand_c <= tz_max_c
                        ):
                            return 99999.0
                        carried_offsets = (
                            [(0, 0), (int(round(offset[0])), int(round(offset[1])))]
                            if self.holding_item and (offset[0] != 0 or offset[1] != 0)
                            else [(0, 0)]
                        )
                        p = PhysicsPredictor.compute_geodesic_path(
                            (int(round(curr_r)), int(round(curr_c))),
                            (deliv_stand_r, deliv_stand_c),
                            barrier_cells,
                            (H, W),
                            self.step_size,
                            footprint_offsets=carried_offsets,
                        )
                        if p and len(p) > 1:
                            return float(len(p))
                        elif p and (deliv_stand_r, deliv_stand_c) == (
                            int(round(curr_r)),
                            int(round(curr_c)),
                        ):
                            return 0.0
                        return 9999.0 + math.hypot(
                            s[0] - offset[0] - curr_r, s[1] - offset[1] - curr_c
                        )

                    reachable_preferred = [
                        s for s in pref_delivery_slots if slot_geodesic_dist(s) < 9000.0
                    ]
                    target_slots = (
                        reachable_preferred if reachable_preferred else pref_delivery_slots
                    )
                    open_target = min(target_slots, key=slot_geodesic_dist)
                else:
                    open_target = ((tz_min_r + tz_max_r) // 2, (tz_min_c + tz_max_c) // 2)
            else:
                open_target = (int(target_pos[0]), int(target_pos[1]))

            # Topologically verify accessibility of delivery destination
            if getattr(self, "gate_target_cell", None) is not None:
                open_target = (
                    self.gate_approach_cell
                    if getattr(self, "gate_approach_cell", None) is not None
                    else self.gate_target_cell
                )
            elif open_target is not None:
                cut_res = TopologicalCutSetAnalyzer.analyze_cut_set(
                    (int(round(curr_r)), int(round(curr_c))),
                    open_target,
                    barrier_cells,
                    (H, W),
                    self.step_size,
                )
                if cut_res.is_partitioned and cut_res.approach_cell:
                    self.gate_target_cell = cut_res.best_gate_cell
                    self.gate_approach_cell = cut_res.approach_cell
                    open_target = cut_res.approach_cell
                else:
                    self.gate_target_cell = None
                    self.gate_approach_cell = None

            in_delivery_zone = False
            if tz_bounds:
                # When target zone bounds are known, always verify item is in the zone.
                # Gate is only a navigation aid, not a delivery trigger.
                tz_min_r, tz_max_r, tz_min_c, tz_max_c = tz_bounds
                # Include zone border (1 cell margin around the detected interior).
                in_delivery_zone = (tz_min_r - 1.0 <= item_r <= tz_max_r + 1.0) and (
                    tz_min_c - 1.0 <= item_c <= tz_max_c + 1.0
                )
            elif self.gate_target_cell:
                item_dist_to_gate = math.hypot(
                    self.gate_target_cell[0] - item_r, self.gate_target_cell[1] - item_c
                )
                player_dist_to_appr = math.hypot(open_target[0] - curr_r, open_target[1] - curr_c)
                in_delivery_zone = (item_dist_to_gate <= max(2.5, self.step_size * 0.9)) or (
                    player_dist_to_appr <= max(1.5, self.step_size * 0.9)
                )
            else:
                deliv_r = int(round(open_target[0] - offset[0]))
                deliv_c = int(round(open_target[1] - offset[1]))
                in_delivery_zone = math.hypot(deliv_r - curr_r, deliv_c - curr_c) <= max(
                    1.5, self.step_size * 0.6
                )

            if in_delivery_zone:
                # If approaching a gate bottleneck, face the gate before releasing / handoff
                if self.gate_target_cell:
                    dr_gate = int(np.sign(self.gate_target_cell[0] - curr_r))
                    dc_gate = int(np.sign(self.gate_target_cell[1] - curr_c))
                    if abs(self.gate_target_cell[0] - curr_r) >= abs(
                        self.gate_target_cell[1] - curr_c
                    ):
                        target_gate_facing = (dr_gate, 0)
                    else:
                        target_gate_facing = (0, dc_gate)

                    facing = getattr(self, "current_facing", (0, 0))
                    if target_gate_facing != (0, 0) and facing != target_gate_facing:
                        for a in available_actions:
                            if a in self.action_models:
                                m = self.action_models[a]
                                if (
                                    target_gate_facing[0] != 0
                                    and np.sign(m.delta_r) == target_gate_facing[0]
                                ) or (
                                    target_gate_facing[1] != 0
                                    and np.sign(m.delta_c) == target_gate_facing[1]
                                ):
                                    self.current_facing = target_gate_facing
                                    return a, 0.98

                if 5 in available_actions:
                    self.holding_item = False
                    self.gate_target_cell = None
                    self.gate_approach_cell = None
                    self.delivered_positions.add((int(round(item_r)), int(round(item_c))))
                    if getattr(self, "picked_up_source_position", None) is not None:
                        p_src = self.picked_up_source_position
                        if p_src is not None:
                            self.delivered_positions.add(
                                (int(round(p_src[0])), int(round(p_src[1])))
                            )
                        self.picked_up_source_position = None
                    if getattr(self, "attempted_pickup_item_id", None) is not None:
                        a_id = self.attempted_pickup_item_id
                        if a_id is not None:
                            self._on_subgoal_resolved(str(a_id))
                        self.attempted_pickup_item_id = None
                    if self.target_zone_base_colors:
                        self.learned_receptacle_colors.update(self.target_zone_base_colors)
                        self.learned_receptacle_bounds = self.target_zone_bounds
                    self.carried_offset = (0.0, 0.0)
                    self.blocked_actions.clear()
                    self.visited_positions.clear()
                    self.last_action_data = None
                    return 5, 0.99

            if self.gate_target_cell:
                # Check if we've reached the gate approach cell
                appr = getattr(self, "gate_approach_cell", self.gate_target_cell)
                dist_to_appr = math.hypot(appr[0] - curr_r, appr[1] - curr_c)
                if dist_to_appr <= max(1.5, self.step_size * 0.6):
                    # Reached the gate — clear it and navigate directly to zone
                    self.gate_target_cell = None
                    self.gate_approach_cell = None
                    deliv_r = max(0, min(H - 1, int(round(open_target[0] - offset[0]))))
                    deliv_c = max(0, min(W - 1, int(round(open_target[1] - offset[1]))))
                    goal_r, goal_c = deliv_r, deliv_c
                else:
                    goal_r, goal_c = int(round(open_target[0])), int(round(open_target[1]))
            else:
                deliv_r = max(0, min(H - 1, int(round(open_target[0] - offset[0]))))
                deliv_c = max(0, min(W - 1, int(round(open_target[1] - offset[1]))))
                goal_r, goal_c = deliv_r, deliv_c
        else:
            unobserved_mask = None

            # Pre-lift scene & candidate entities to HCIR workspace before goal decomposition
            self.lift_to_hcir(curr_grid, target_pos, candidate_items=candidate_items)

            force_subgoals = bool(5 in available_actions and candidate_subgoals)
            self.active_goal_node = self.decomposer.decompose_goal(
                workspace=self.workspace,
                primary_goal=self.primary_goal_node,
                avatar_pos=(int(round(curr_r)), int(round(curr_c))),
                barrier_cells=barrier_cells,
                grid_shape=(H, W),
                candidate_subgoals=candidate_subgoals,
                step_size=self.step_size,
                force_subgoals=force_subgoals,
                unobserved_mask=unobserved_mask,
            )
            t_pos = self.active_goal_node.properties.get("target_position", target_pos)
            item_pos = self.active_goal_node.properties.get("item_position", t_pos)
            goal_r, goal_c = int(t_pos[0]), int(t_pos[1])
            item_r, item_c = int(item_pos[0]), int(item_pos[1])

            # Check if reached approach stand position for prerequisite item with affordance
            is_at_stand = math.hypot(goal_r - curr_r, goal_c - curr_c) <= max(
                1.5, self.step_size * 0.75
            )
            is_item_goal = self.active_goal_node.id.startswith("subgoal_sub_")
            if is_at_stand and is_item_goal:
                dr_dir = int(np.sign(item_r - curr_r))
                dc_dir = int(np.sign(item_c - curr_c))
                if abs(item_r - curr_r) >= abs(item_c - curr_c):
                    target_facing = (dr_dir, 0)
                else:
                    target_facing = (0, dc_dir)

                facing = getattr(self, "current_facing", (0, 0))
                if target_facing != (0, 0) and facing != target_facing:
                    for a in available_actions:
                        if a in self.action_models:
                            facing_m = self.action_models[a]
                            if (
                                target_facing[0] != 0
                                and np.sign(facing_m.delta_r) == target_facing[0]
                            ) or (
                                target_facing[1] != 0
                                and np.sign(facing_m.delta_c) == target_facing[1]
                            ):
                                self.current_facing = target_facing
                                return a, 0.98

                if 5 in available_actions:
                    self.holding_item = True
                    dr_off = float(np.clip(item_r - curr_r, -self.step_size, self.step_size))
                    dc_off = float(np.clip(item_c - curr_c, -self.step_size, self.step_size))
                    self.carried_offset = (dr_off, dc_off)
                    self.picked_up_source_position = (item_r, item_c)
                    self.attempted_pickup_item_id = self.active_goal_node.id
                    i_col = self.active_goal_node.properties.get("item_color")
                    if i_col is not None:
                        self.learned_item_colors[int(i_col)] = {"action": 5}
                    self.last_action_data = None
                    return 5, 0.99

        self.goal_centroid = (float(goal_r), float(goal_c))

        # 5. Native HCIR Scene Lifting & Topological Geodesic Search
        _ws, _goal_node, _candidates = self.lift_to_hcir(
            curr_grid, (goal_r, goal_c), candidate_items=candidate_items
        )

        carried_offsets = (
            [(0, 0), (int(round(self.carried_offset[0])), int(round(self.carried_offset[1])))]
            if self.holding_item and (self.carried_offset[0] != 0 or self.carried_offset[1] != 0)
            else [(0, 0)]
        )
        shortest_path = PhysicsPredictor.compute_geodesic_path(
            start=(int(round(curr_r)), int(round(curr_c))),
            goal=(goal_r, goal_c),
            barrier_cells=barrier_cells,
            grid_shape=(H, W),
            step_size=self.step_size,
            footprint_offsets=carried_offsets,
        )

        dr_des = goal_r - curr_r
        dc_des = goal_c - curr_c
        if len(shortest_path) > 1:
            next_waypoint = shortest_path[1]
            dr_des = next_waypoint[0] - curr_r
            dc_des = next_waypoint[1] - curr_c

        # 6. Action Scoring with Native HCIR Physics Deadlock Avoidance & Loop Prevention
        best_action = available_actions[0]
        best_score = -999999.0

        for a in available_actions:
            m = self.action_models.get(a)
            if not m or (m.delta_r == 0 and m.delta_c == 0):
                continue

            dest_r = curr_r + m.delta_r
            dest_c = curr_c + m.delta_c
            dest_ir = int(round(dest_r))
            dest_ic = int(round(dest_c))

            # Check if this action pushes a box into a corner deadlock via HCIR PhysicsPredictor
            deadlock_penalty = 0.0
            if self.pushable_colors:
                if 0 <= dest_ir < H and 0 <= dest_ic < W:
                    if curr_grid[dest_ir, dest_ic] in self.pushable_colors:
                        box_next_r = int(round(dest_ir + m.delta_r))
                        box_next_c = int(round(dest_ic + m.delta_c))
                        if PhysicsPredictor.is_corner_deadlock(
                            box_pos=(box_next_r, box_next_c),
                            barrier_cells=barrier_cells,
                            target_positions=self.target_zones,
                            grid_shape=(H, W),
                            step_size=self.step_size,
                        ):
                            deadlock_penalty = 50000.0

            # Alignment with BFS shortest path waypoint
            alignment = (m.delta_r * dr_des) + (m.delta_c * dc_des)
            penalty = 10000.0 if a in self.blocked_actions else 0.0

            # Oscillation penalty: penalize stepping into cells visited recently (exempt if following geodesic path)

            recents = self.visited_positions[-8:]
            is_geodesic_step = (
                (dr_des != 0 or dc_des != 0)
                and np.sign(m.delta_r) == np.sign(dr_des)
                and np.sign(m.delta_c) == np.sign(dc_des)
            )
            # Immediate 2-step bounce penalty (stepping back to position from 2 steps ago)
            is_2step_bounce = (
                len(self.visited_positions) >= 2
                and math.hypot(
                    dest_r - self.visited_positions[-2][0], dest_c - self.visited_positions[-2][1]
                )
                < self.step_size * 0.7
            )
            bounce_penalty = 800.0 if is_2step_bounce else 0.0

            loop_penalty = (
                bounce_penalty
                if is_geodesic_step
                else bounce_penalty
                + sum(
                    35.0
                    for vr, vc in recents
                    if math.hypot(dest_r - vr, dest_c - vc) < self.step_size * 0.9
                )
            )

            # Barrier penalty: exempt if destination cell reaches the goal
            is_dest_goal = (dest_ir, dest_ic) == (goal_r, goal_c) or math.hypot(
                dest_r - goal_r, dest_c - goal_c
            ) <= self.step_size * 0.95
            dest_in_barrier = (dest_ir, dest_ic) in barrier_cells
            if self.holding_item and (self.carried_offset[0] != 0 or self.carried_offset[1] != 0):
                carried_ir = int(round(dest_r + self.carried_offset[0]))
                carried_ic = int(round(dest_c + self.carried_offset[1]))
                is_carried_goal = (
                    self.target_zone_bounds is not None
                    and self.target_zone_bounds[0] <= carried_ir <= self.target_zone_bounds[1]
                    and self.target_zone_bounds[2] <= carried_ic <= self.target_zone_bounds[3]
                )
                if not (0 <= carried_ir < H and 0 <= carried_ic < W) or (
                    (carried_ir, carried_ic) in barrier_cells and not is_carried_goal
                ):
                    dest_in_barrier = True
            barrier_penalty = 0.0 if is_dest_goal else (20000.0 if dest_in_barrier else 0.0)

            score = float(alignment - penalty - deadlock_penalty - loop_penalty - barrier_penalty)

            if score > best_score:
                best_score = score
                best_action = a

        # If all actions are blocked or deadlocked, clear blocked set to allow detour
        if best_score < -5000.0:
            if self.active_goal_node:
                self.interruption_stack.append(
                    InterruptionCheckpoint(
                        parent_goal_id=self.active_goal_node.id,
                        parent_frame_id="frame_active",
                        interrupt_goal_id="goal_detour",
                        interrupt_frame_id="frame_detour",
                        in_flight_action=f"ACTION{best_action}",
                        step_index=self.stuck_counter,
                        context_data={
                            "reason": f"arc3_detour_barrier_{self.stuck_counter}",
                            "avatar_pos": (curr_r, curr_c),
                            "holding_item": self.holding_item,
                        },
                    )
                )
            self.blocked_actions.clear()
            for a in available_actions:
                m = self.action_models.get(a)
                if m and (m.delta_r != 0 or m.delta_c != 0):
                    alignment = (m.delta_r * dr_des) + (m.delta_c * dc_des)
                    if alignment > best_score:
                        best_score = float(alignment)
                        best_action = a

        confidence = 0.94 if best_score > 0 else 0.65
        if best_action == 6:
            click_x = int(round(goal_c)) if "goal_c" in locals() else W // 2
            click_y = int(round(goal_r)) if "goal_r" in locals() else H // 2
            self.last_action_data = {
                "x": max(0, min(W - 1, click_x)),
                "y": max(0, min(H - 1, click_y)),
            }
        else:
            self.last_action_data = None
        return best_action, confidence

    def plan_next_action(
        self,
        curr_grid: np.ndarray,
        available_actions: list[int],
        tags: list[str] | None = None,
        allow_soft_restart: bool = False,
    ) -> tuple[int, float]:
        """Synthesize next action by lifting scene to EntityGraph and planning state-space paths."""
        self.available_actions = list(available_actions)
        self.level_step_counter += 1

        # Immediately execute queued actions synthesized by multi-timeline or task planner
        if self.action_queue:
            return self.action_queue.pop(0), 0.99

        if self.initial_grid is None:
            self.initial_grid = curr_grid.copy()

        # Pure 2D navigation / spatial movement game without Action 5 (Pickup/Drop)
        if 5 not in available_actions:
            calibrated_models = [
                m
                for a, m in self.action_models.items()
                if a in available_actions
                and m.confidence >= 0.8
                and (m.delta_r != 0 or m.delta_c != 0)
            ]
            directional_avail = [a for a in available_actions if a in [1, 2, 3, 4]]
            if len(calibrated_models) < min(4, len(directional_avail)):
                self.probe_step_counter += 1
                probe_act = self._active_probe_action(available_actions)
                if probe_act == 6:
                    self.last_action_data = {
                        "x": curr_grid.shape[1] // 2,
                        "y": curr_grid.shape[0] // 2,
                    }
                return probe_act, 0.50
            return self._plan_geodesic_navigation_step(curr_grid, available_actions, tags=tags)

        # Check if agent requested a soft restart
        if self.phase == AgentPhase.SOFT_RESTART_PENDING:
            if 0 in available_actions or allow_soft_restart:
                self.soft_restart()
                return 0, 1.0
            else:
                self.phase = AgentPhase.OPTIMAL_EXECUTION

        H, W = curr_grid.shape

        # Learn dominant floor / background colors dynamically
        counts = np.bincount(curr_grid.ravel())
        bg_color = int(np.argmax(counts))
        self.learned_walkable_colors.add(bg_color)
        self.learned_walkable_colors.add(0)
        for col, count in enumerate(counts):
            if count >= int(H * W * 0.20) and (
                self.avatar_color is None or col != self.avatar_color
            ):
                self.learned_walkable_colors.add(int(col))
        self.learned_barrier_colors.difference_update(self.learned_walkable_colors)
        self.learned_item_colors.difference_update(self.learned_walkable_colors)

        # ── 1. Motor Calibration Probing ──
        calibrated_models = [
            m
            for a, m in self.action_models.items()
            if a in available_actions and m.confidence >= 0.8 and (m.delta_r != 0 or m.delta_c != 0)
        ]
        directional_avail = [a for a in available_actions if a in [1, 2, 3, 4]]
        if len(calibrated_models) < min(4, len(directional_avail)):
            self.probe_step_counter += 1
            probe_act = self._active_probe_action(available_actions)
            if probe_act == 6:
                self.last_action_data = {"x": curr_grid.shape[1] // 2, "y": curr_grid.shape[0] // 2}
            return probe_act, 0.50

        # ── 2. Locate Avatar on Grid Lattice ──
        if self.avatar_color is not None:
            import scipy.ndimage as ndi

            playable_av = (curr_grid == self.avatar_color).copy()
            if H >= 16 and W >= 16:
                playable_av[:5, :] = False
                playable_av[-3:, :] = False
            lbls, n_comp = ndi.label(playable_av)
            if n_comp > 0:
                best_comp_idx = 1
                if self.avatar_centroid is not None:
                    best_dist = 999.0
                    for i in range(1, n_comp + 1):
                        pts = np.where(lbls == i)
                        cr, cc = float(np.mean(pts[0])), float(np.mean(pts[1]))
                        d = math.hypot(cr - self.avatar_centroid[0], cc - self.avatar_centroid[1])
                        if d < best_dist:
                            best_dist = d
                            best_comp_idx = i
                else:
                    comp_areas = [int(np.sum(lbls == i)) for i in range(1, n_comp + 1)]
                    best_comp_idx = int(np.argmax(comp_areas)) + 1

                pts = np.where(lbls == best_comp_idx)
                raw_r = float(np.mean(pts[0]))
                raw_c = float(np.mean(pts[1]))
                self.raw_avatar_centroid = (raw_r, raw_c)
                curr_r = self._snap_coord(raw_r)
                curr_c = self._snap_coord(raw_c)
                self.avatar_centroid = (float(curr_r), float(curr_c))
                if self.start_pos is None:
                    self.start_pos = (curr_r, curr_c)
                    self.raw_start_pos = (raw_r, raw_c)
            else:
                self.probe_step_counter += 1
                return self._active_probe_action(available_actions), 0.40
        else:
            self.probe_step_counter += 1
            return self._active_probe_action(available_actions), 0.40

        # ── 2.5 Multi-Timeline / Circuit Maze Planning (e.g. g50t) ──
        if 5 in available_actions and self.action_5_affordance not in ("PICKUP", "PICKUP_DROP"):
            clone_plan = self._discover_and_plan_temporal_clone(curr_grid, available_actions)
            if clone_plan:
                self.action_queue = clone_plan
                return self.action_queue.pop(0), 0.99

        # ── 3. Extract Objects & Lift Scene to EntityGraph ──
        arc_grid = ARCGrid.from_list(curr_grid.tolist())
        objs = GridTopologyExtractor.extract_objects(arc_grid)

        # Check if target_zone_bounds is still valid on current frame
        if self.target_zone_bounds is not None and self.learned_receptacle_colors:
            tz_r1, tz_r2, tz_c1, tz_c2 = self.target_zone_bounds
            tz_slice = curr_grid[tz_r1 : tz_r2 + 1, tz_c1 : tz_c2 + 1]
            if tz_slice.size == 0 or not any(c in tz_slice for c in self.learned_receptacle_colors):
                self.target_zone_bounds = None
                self.current_plan.clear()

        # Detect Receptacle / Target Zone (only when not yet established for current level)
        if self.target_zone_bounds is None:
            candidate_goals = [
                o
                for o in objs
                if o.color not in (0, self.avatar_color)
                and (
                    o.color in self.learned_receptacle_colors
                    or getattr(o, "is_frame", False)
                    or o.color not in self.learned_walkable_colors
                )
                and np.count_nonzero(curr_grid == o.color) < (curr_grid.size * 0.30)
            ]
            min_frame_receptacle_area = self.step_size * self.step_size
            target_zones = [
                o
                for o in candidate_goals
                if (
                    o.color in self.learned_receptacle_colors and o.area > min_frame_receptacle_area
                )
                or (getattr(o, "is_frame", False) and o.area > min_frame_receptacle_area)
                or (
                    o.area > min_frame_receptacle_area
                    and (o.max_r - o.min_r + 1 >= self.step_size)
                    and (o.max_c - o.min_c + 1 >= self.step_size)
                )
            ]
            if target_zones:
                best_tz = max(
                    target_zones,
                    key=lambda o: (
                        3
                        if (
                            o.color in self.learned_receptacle_colors
                            and getattr(o, "is_frame", False)
                        )
                        else (
                            2
                            if o.color in self.learned_receptacle_colors
                            else (1 if getattr(o, "is_frame", False) else 0)
                        ),
                        o.area,
                    ),
                )
                self.target_zone_bounds = (
                    best_tz.min_r,
                    best_tz.max_r,
                    best_tz.min_c,
                    best_tz.max_c,
                )
                self.learned_receptacle_colors.add(int(best_tz.color))

        # Dynamically learn item colors from objects outside target zone
        from collections import Counter

        candidate_item_colors = Counter()
        max_item_area = max(36, int(self.step_size * self.step_size * 2.5))
        min_item_area = max(3, int(self.step_size * self.step_size * 0.25))
        for o in objs:
            if (
                o.color in (0, self.avatar_color)
                or o.color in self.learned_walkable_colors
                or o.color in self.learned_receptacle_colors
            ):
                continue
            if self.target_zone_bounds:
                tz = self.target_zone_bounds
                if tz[0] <= o.min_r and o.max_r <= tz[1] and tz[2] <= o.min_c and o.max_c <= tz[3]:
                    continue
            span_r = o.max_r - o.min_r
            span_c = o.max_c - o.min_c
            if self.step_size > 1 and (
                span_r < max(1, self.step_size // 2) or span_c < max(1, self.step_size // 2)
            ):
                continue
            if min_item_area <= o.area <= max_item_area:
                if span_r <= self.step_size * 2 and span_c <= self.step_size * 2:
                    candidate_item_colors[o.color] += 1

        for col, count in candidate_item_colors.items():
            if count >= 2 or (col in self.learned_item_colors and count >= 1):
                self.learned_item_colors.add(col)
        self.learned_receptacle_colors.difference_update(self.learned_item_colors)

        # Track delivered items inside receptacle (e.g. delivered by self or cooperative NPC)
        if self.target_zone_bounds:
            tz = self.target_zone_bounds
            for o in objs:
                if tz[0] <= o.min_r and o.max_r <= tz[1] and tz[2] <= o.min_c and o.max_c <= tz[3]:
                    if o.color in self.learned_item_colors:
                        r = self._snap_coord(float(o.centroid[0]))
                        c = self._snap_coord(float(o.centroid[1]))
                        self.delivered_positions.add((r, c))

            # Prune external handoff positions that no longer contain an item (picked up by NPC)
            occupied_slots = {
                (self._snap_coord(float(o.centroid[0])), self._snap_coord(float(o.centroid[1])))
                for o in objs
                if o.color in self.learned_item_colors
            }
            to_remove = {
                dp
                for dp in self.delivered_positions
                if not (tz[0] <= dp[0] <= tz[1] and tz[2] <= dp[1] <= tz[3])
                and dp not in occupied_slots
            }
            self.delivered_positions.difference_update(to_remove)

        known_b: set[tuple[int, int]] = set()
        if self.known_barriers is not None:
            known_b = set(zip(*np.where(self.known_barriers)))

        # Lift scene using ARCPerceptualLifter (adapter domain)
        entities, raw_barriers = ARCPerceptualLifter.lift(
            grid=curr_grid,
            raw_objects=objs,
            avatar_color=self.avatar_color,
            avatar_centroid=self.avatar_centroid,
            learned_item_colors=self.learned_item_colors,
            learned_receptacle_colors=self.learned_receptacle_colors,
            learned_barrier_colors=self.learned_barrier_colors,
            walkable_colors=self.learned_walkable_colors,
            step_size=self.step_size,
            target_zone_bounds=self.target_zone_bounds,
            shape_concepts=self.shape_concepts,
        )

        # Construct topological EntityGraph using HCIRSpatialEntityPlanner (HCIR domain-agnostic)
        eg: EntityGraph = self.spatial_planner.construct_entity_graph(
            entities=entities,
            barriers=raw_barriers,
            grid_shape=(H, W),
            step_size=self.step_size,
            known_barriers=known_b,
        )

        # ── 4. Determine Cooperative Partition State ──
        if self.target_zone_bounds:
            tz_min_r, tz_max_r, tz_min_c, tz_max_c = self.target_zone_bounds
            tz_center = (
                self._snap_coord((tz_min_r + tz_max_r) / 2.0),
                self._snap_coord((tz_min_c + tz_max_c) / 2.0),
            )
            if not self.is_cooperative_handoff:
                p_comp = TopologicalCutSetAnalyzer.get_reachable_component(
                    (curr_r, curr_c),
                    eg.barriers,
                    (H, W),
                    self.step_size,
                )
                self.is_cooperative_handoff = bool(tz_center not in p_comp)
        else:
            tz_center = (H // 2, W // 2)

        # ── 5. Plan High-Level Entity Subgoal Sequence ──
        if not self.current_plan:
            self.current_plan = self.spatial_planner.plan_sequence(
                eg=eg,
                workspace=self.workspace,
                delivered_positions=self.delivered_positions,
                is_carrying=self.holding_item,
                carried_offset=self.carried_offset,
                has_pickup_drop=(5 in available_actions),
                item_colors=self.learned_item_colors or None,
            )
            if self.current_plan and self.phase == AgentPhase.EPISTEMIC_LEARNING:
                self.optimal_task_plan = list(self.current_plan)
                logger.info(
                    "HCIR deduced exact tasks needed (%d subgoals).",
                    len(self.optimal_task_plan),
                )
                if self.enable_soft_restart or (0 in available_actions) or allow_soft_restart:
                    self.phase = AgentPhase.SOFT_RESTART_PENDING
                    self.should_soft_restart = True
                    if 0 in available_actions or allow_soft_restart:
                        self.soft_restart()
                        return 0, 1.0
                else:
                    self.phase = AgentPhase.OPTIMAL_EXECUTION
        # ── 6. State Machine: Execute Active Subgoal Step ──
        # If no plan steps left:
        if not self.current_plan:
            return self._stand_clear_action(curr_r, curr_c, eg, available_actions)
        else:
            active_step = self.current_plan[0]
            # Dynamic re-planning if drop slot was occupied or invalid
            if active_step.action_type == "DROP":
                drop_slot = (
                    int(round(active_step.target_pos[0] + self.carried_offset[0])),
                    int(round(active_step.target_pos[1] + self.carried_offset[1])),
                )
                is_portal_drop = any(
                    e.role in (EntityRole.PORTAL, EntityRole.PORTAL)
                    and (e.grid_pos == drop_slot or e.id == active_step.target_entity_id)
                    for e in eg.entities.values()
                )
                is_blocked = (drop_slot in self.delivered_positions) or (
                    not is_portal_drop and drop_slot in eg.barriers
                )
                if not is_blocked and is_portal_drop:
                    # Portal drops only re-plan if a previously-delivered item is still
                    # physically in the doorway.  The gate cell's own wall entity is
                    # expected and must not trigger a re-plan.
                    if any(
                        e.role in (EntityRole.MANIPULABLE, EntityRole.MANIPULABLE)
                        and math.hypot(e.grid_pos[0] - drop_slot[0], e.grid_pos[1] - drop_slot[1])
                        < self.step_size * 0.75
                        and e.grid_pos in self.delivered_positions
                        for e in eg.entities.values()
                    ):
                        is_blocked = True

                if is_blocked:
                    self.current_plan.clear()
                    self.current_plan = self.spatial_planner.plan_sequence(
                        eg=eg,
                        workspace=self.workspace,
                        delivered_positions=self.delivered_positions,
                        is_carrying=self.holding_item,
                        carried_offset=self.carried_offset,
                        has_pickup_drop=(5 in available_actions),
                        item_colors=self.learned_item_colors or None,
                    )
                    if not self.current_plan:
                        return self._stand_clear_action(curr_r, curr_c, eg, available_actions)
                    active_step = self.current_plan[0]

            step_target = active_step.target_pos
            target_facing = active_step.approach_facing
            action_type = active_step.action_type
            carried_footprint: list[tuple[int, int]] = (
                [
                    (0, 0),
                    (
                        int(round(active_step.carried_offset[0])),
                        int(round(active_step.carried_offset[1])),
                    ),
                ]
                if self.holding_item and (active_step.carried_offset != (0, 0))
                else [(0, 0)]
            )

        # Check if arrived at stand position
        dist_to_stand = math.hypot(step_target[0] - curr_r, step_target[1] - curr_c)
        is_at_stand = dist_to_stand <= max(1.5, self.step_size * 0.75)

        if is_at_stand:
            # Action: PICKUP
            if action_type == "PICKUP":
                # If the pickup was already detected by update_causal_dynamics
                # (e.g. the game auto-picked up on movement), skip sending another
                # act=5 which would accidentally DROP the item.
                if self.holding_item:
                    self.current_plan.pop(0)
                    return self._stand_clear_action(curr_r, curr_c, eg, available_actions)

                # Orientation alignment: ensure avatar faces target_facing before pickup
                if target_facing and self.current_facing != target_facing:
                    for a in available_actions:
                        m = self.action_models.get(a)
                        if m and m.confidence >= 0.8:
                            if (
                                target_facing[0] != 0
                                and np.sign(m.delta_r) == np.sign(target_facing[0])
                            ) or (
                                target_facing[1] != 0
                                and np.sign(m.delta_c) == np.sign(target_facing[1])
                            ):
                                self.current_facing = target_facing
                                return a, 0.95

                item_adj = (
                    step_target[0] + (target_facing[0] if target_facing else 0) * self.step_size,
                    step_target[1] + (target_facing[1] if target_facing else 0) * self.step_size,
                )
                item_exists = any(
                    e.role in (EntityRole.MANIPULABLE, EntityRole.MANIPULABLE, EntityRole.ACTUATOR)
                    and math.hypot(e.grid_pos[0] - item_adj[0], e.grid_pos[1] - item_adj[1])
                    < self.step_size * 0.75
                    and not e.is_delivered
                    for e in eg.entities.values()
                )
                if not item_exists:
                    self.current_plan.clear()
                    return self._stand_clear_action(curr_r, curr_c, eg, available_actions)

                if 5 in available_actions:
                    self.holding_item = True
                    for e in eg.entities.values():
                        if e.role in (
                            EntityRole.MANIPULABLE,
                            EntityRole.MANIPULABLE,
                            EntityRole.ACTUATOR,
                        ):
                            if (
                                math.hypot(e.grid_pos[0] - item_adj[0], e.grid_pos[1] - item_adj[1])
                                < self.step_size * 0.75
                            ):
                                if (
                                    e.color is not None
                                    and e.color != 0
                                    and e.color != self.avatar_color
                                ):
                                    self.learned_item_colors.add(e.color)
                                    self.learned_receptacle_colors.discard(e.color)
                                    self.learned_barrier_colors.discard(e.color)
                                if e.shape_archetype:
                                    self._get_or_create_shape_concept(
                                        e.shape_archetype,
                                        color=e.color,
                                        role=EntityRole.MANIPULABLE,
                                    )
                    if target_facing:
                        self.carried_offset = (
                            float(target_facing[0] * self.step_size),
                            float(target_facing[1] * self.step_size),
                        )
                    self.current_plan.pop(0)
                    return 5, 0.99

            # Action: DROP
            elif action_type == "DROP":
                if 5 in available_actions:
                    self.holding_item = False
                    if target_facing:
                        g_r = step_target[0] + target_facing[0] * self.step_size
                        g_c = step_target[1] + target_facing[1] * self.step_size
                        self.delivered_positions.add((g_r, g_c))
                    else:
                        self.delivered_positions.add(step_target)
                    self.current_plan.clear()
                    return 5, 0.99

            # Action: MOVE (Reached destination)
            elif action_type == "MOVE":
                if self.current_plan:
                    self.current_plan.pop(0)
                return self._stand_clear_action(curr_r, curr_c, eg, available_actions)

        # ── 7. Pathfinding: Navigate Towards Stand Position ──
        # Obstacles include barriers, delivered items, and other uncarried boxes
        carried_pos = (
            (
                int(round(curr_r + self.carried_offset[0])),
                int(round(curr_c + self.carried_offset[1])),
            )
            if self.holding_item
            else None
        )

        item_coords = set()
        for e in eg.entities.values():
            if e.role in (EntityRole.MANIPULABLE, EntityRole.MANIPULABLE) and not e.is_delivered:
                if (
                    carried_pos
                    and math.hypot(e.grid_pos[0] - carried_pos[0], e.grid_pos[1] - carried_pos[1])
                    < self.step_size * 0.75
                ):
                    continue
                item_coords.add(e.grid_pos)
        item_coords.discard(step_target)
        item_coords.discard(step_target)

        effective_barriers = set(eg.barriers)
        if self.target_zone_bounds:
            tz_min_r, tz_max_r, tz_min_c, tz_max_c = self.target_zone_bounds
            for tz_r in range(tz_min_r, tz_max_r + 1):
                for tz_c in range(tz_min_c, tz_max_c + 1):
                    effective_barriers.add((tz_r, tz_c))
        effective_barriers.discard(step_target)
        if self.holding_item:
            goal_payload = (
                int(round(step_target[0] + self.carried_offset[0])),
                int(round(step_target[1] + self.carried_offset[1])),
            )
            effective_barriers.discard(goal_payload)

        for dp in self.delivered_positions:
            if dp != step_target:
                effective_barriers.add(dp)
        effective_barriers.update(item_coords)
        for e in eg.entities.values():
            if e != eg.agent and e.role not in (
                EntityRole.RECEPTACLE,
                EntityRole.GOAL,
                EntityRole.GOAL,
                EntityRole.MANIPULABLE,
                EntityRole.MANIPULABLE,
            ):
                if (
                    math.hypot(e.grid_pos[0] - curr_r, e.grid_pos[1] - curr_c)
                    < self.step_size * 0.75
                ):
                    continue
                if (
                    carried_pos
                    and math.hypot(e.grid_pos[0] - carried_pos[0], e.grid_pos[1] - carried_pos[1])
                    < self.step_size * 0.75
                ):
                    continue
                if e.grid_pos != step_target:
                    effective_barriers.add(e.grid_pos)

        effective_barriers.discard((curr_r, curr_c))
        if carried_pos:
            effective_barriers.discard(carried_pos)

        shortest_path = PhysicsPredictor.compute_geodesic_path(
            start=(curr_r, curr_c),
            goal=step_target,
            barrier_cells=effective_barriers,
            grid_shape=(H, W),
            step_size=self.step_size,
            footprint_offsets=carried_footprint,
        )

        if not shortest_path:
            shortest_path = PhysicsPredictor.compute_geodesic_path(
                start=(curr_r, curr_c),
                goal=step_target,
                barrier_cells=eg.barriers,
                grid_shape=(H, W),
                step_size=self.step_size,
                footprint_offsets=carried_footprint,
            )

        # Fallback: if carrying prevents pathfinding through narrow gaps (e.g. portal
        # openings in a vertical wall), retry with avatar-only footprint.  The game
        # physics will handle the carried item following the avatar through the gap.
        if not shortest_path and self.holding_item and len(carried_footprint) > 1:
            shortest_path = PhysicsPredictor.compute_geodesic_path(
                start=(curr_r, curr_c),
                goal=step_target,
                barrier_cells=effective_barriers,
                grid_shape=(H, W),
                step_size=self.step_size,
                footprint_offsets=[(0, 0)],
            )

        if shortest_path and len(shortest_path) > 1 and shortest_path[-1] == step_target:
            next_pt = shortest_path[1]
            dr_des = next_pt[0] - curr_r
            dc_des = next_pt[1] - curr_c
            for a in available_actions:
                if a in self.blocked_actions:
                    continue
                m = self.action_models.get(a)
                if m and m.confidence >= 0.8:
                    if (dr_des != 0 and np.sign(m.delta_r) == np.sign(dr_des)) or (
                        dc_des != 0 and np.sign(m.delta_c) == np.sign(dc_des)
                    ):
                        self.last_action = a
                        return a, 0.95

        dr_des = step_target[0] - curr_r
        dc_des = step_target[1] - curr_c

        # ── 8. Action Scoring with Collision & Oscillation Avoidance ──
        best_action = available_actions[0]
        best_score = -999999.0

        for a in available_actions:
            m = self.action_models.get(a)
            if not m or (m.delta_r == 0 and m.delta_c == 0):
                continue

            alignment = (m.delta_r * dr_des) + (m.delta_c * dc_des)
            penalty = 10000.0 if a in self.blocked_actions else 0.0

            dest_r = curr_r + m.delta_r
            dest_c = curr_c + m.delta_c
            dest_ir, dest_ic = int(round(dest_r)), int(round(dest_c))

            # Barrier collision check for avatar and carried footprint
            dest_in_barrier = (dest_ir, dest_ic) in effective_barriers or not (
                0 <= dest_ir < H and 0 <= dest_ic < W
            )
            if self.holding_item and self.carried_offset != (0.0, 0.0):
                c_ir = int(round(dest_r + self.carried_offset[0]))
                c_ic = int(round(dest_c + self.carried_offset[1]))
                if not (0 <= c_ir < H and 0 <= c_ic < W) or (c_ir, c_ic) in effective_barriers:
                    dest_in_barrier = True

            barrier_penalty = 20000.0 if dest_in_barrier else 0.0

            # Oscillation penalty
            recents = self.visited_positions[-8:]
            loop_penalty = sum(
                35.0
                for vr, vc in recents
                if math.hypot(dest_r - vr, dest_c - vc) < self.step_size * 0.9
            )

            score = float(alignment - penalty - barrier_penalty - loop_penalty)
            if score > best_score:
                best_score = score
                best_action = a

        # Detour fallback if all actions are blocked
        if best_score < -5000.0:
            self.blocked_actions.clear()
            best_detour_score = -999999.0
            for a in available_actions:
                m = self.action_models.get(a)
                if m and (m.delta_r != 0 or m.delta_c != 0):
                    dest_r = curr_r + m.delta_r
                    dest_c = curr_c + m.delta_c
                    dest_ir, dest_ic = int(round(dest_r)), int(round(dest_c))
                    is_b = (dest_ir, dest_ic) in effective_barriers or not (
                        0 <= dest_ir < H and 0 <= dest_ic < W
                    )
                    b_pen = 20000.0 if is_b else 0.0
                    align = (m.delta_r * dr_des) + (m.delta_c * dc_des)
                    recents = self.visited_positions[-8:]
                    l_pen = sum(
                        35.0
                        for vr, vc in recents
                        if math.hypot(dest_r - vr, dest_c - vc) < self.step_size * 0.9
                    )
                    d_score = float(align - b_pen - l_pen)
                    if d_score > best_detour_score:
                        best_detour_score = d_score
                        best_action = a
            best_score = best_detour_score

        self.last_action = best_action
        confidence = 0.95 if best_score > 0 else 0.60
        if best_action == 6 and (
            not isinstance(self.last_action_data, dict)
            or "x" not in self.last_action_data
            or "y" not in self.last_action_data
        ):
            click_x = int(round(self.avatar_centroid[1])) if self.avatar_centroid else W // 2
            click_y = int(round(self.avatar_centroid[0])) if self.avatar_centroid else H // 2
            self.last_action_data = {
                "x": max(0, min(W - 1, click_x)),
                "y": max(0, min(H - 1, click_y)),
            }
        return best_action, confidence

    async def plan_next_action_counterfactual(
        self, grid: np.ndarray, available_actions: list[int]
    ) -> tuple[int, float]:
        """Backward-compatible async counterfactual planner wrapper."""
        return self.plan_next_action(grid, available_actions)

    def lift_to_hcir(
        self,
        curr_grid: np.ndarray,
        chosen_goal: tuple[int, int] | tuple[float, float] | None = None,
        candidate_items: list[Any] | None = None,
    ) -> tuple[HCIRWorkspaceState, GoalNode, list[ActionNode]]:
        """Lift 2D visual sensory observation into native HCIR CognitiveGraph & Workspace."""
        ws = (
            self.workspace
            if hasattr(self, "workspace") and self.workspace is not None
            else HCIRWorkspaceState()
        )
        H, W = curr_grid.shape

        # 1. Controllable Avatar PhysicalEntityNode
        if self.avatar_centroid is not None:
            ar, ac = int(round(self.avatar_centroid[0])), int(round(self.avatar_centroid[1]))
            active_eid = (
                self.control_context.active_entity.label
                if hasattr(self, "control_context") and self.control_context.active_entity
                else "avatar"
            )
            ws.upsert_node(
                PhysicalEntityNode(
                    id="avatar",
                    entity_name="avatar",
                    entity_type="agent",
                    status="active",
                    properties={
                        "position": (ar, ac),
                        "color": int(self.avatar_color) if self.avatar_color is not None else -1,
                        "is_avatar": True,
                        "movable": True,
                        "passable": False,
                        "controlled_entity_id": active_eid,
                    },
                )
            )

        # 2. Pushable blocks / Movable Objects
        for p_col in self.pushable_colors:
            pts = np.where(curr_grid == p_col)
            for r, c in zip(pts[0], pts[1]):
                box_id = f"box_{r}_{c}"
                ws.upsert_node(
                    PhysicalEntityNode(
                        id=box_id,
                        entity_name="pushable_block",
                        entity_type="movable_object",
                        status="active",
                        properties={
                            "position": (int(r), int(c)),
                            "color": int(p_col),
                            "movable": True,
                            "passable": False,
                            "affordances": ["PUSHABLE"],
                        },
                    )
                )

        # 3. Target Goal Zones
        goal_node = GoalNode(id="goal_arc3", description="Reach target location or deliver object")
        if chosen_goal is not None:
            gr, gc = int(round(chosen_goal[0])), int(round(chosen_goal[1]))
            goal_node.properties = {"target_position": (gr, gc), "target_entity": "goal_primary"}
            ws.upsert_node(
                PhysicalEntityNode(
                    id="goal_primary",
                    entity_name="goal_zone",
                    entity_type="target_zone",
                    status="active",
                    properties={
                        "position": (gr, gc),
                        "is_goal": True,
                        "movable": False,
                        "passable": True,
                    },
                )
            )
        ws.upsert_node(goal_node)

        # 4. Barriers & Environment Variables
        barrier_cells: list[tuple[int, int]] = []
        if self.known_barriers is not None:
            b_coords = np.where(self.known_barriers)
            barrier_cells = [(int(r), int(c)) for r, c in zip(b_coords[0], b_coords[1])]

        target_positions = list(self.target_zones)
        if chosen_goal is not None:
            goal_cell = (int(round(chosen_goal[0])), int(round(chosen_goal[1])))
            if goal_cell not in target_positions:
                target_positions.append(goal_cell)

        ws.upsert_node(
            WorldVariableNode(
                id="var_grid_shape",
                variable_name="grid_shape",
                value=[H, W],
            )
        )
        ws.upsert_node(
            WorldVariableNode(
                id="var_barrier_cells",
                variable_name="barrier_cells",
                value=barrier_cells,
            )
        )
        ws.upsert_node(
            WorldVariableNode(
                id="var_target_positions",
                variable_name="target_positions",
                value=target_positions,
            )
        )
        ws.upsert_node(
            WorldVariableNode(
                id="var_control_context",
                variable_name="control_context",
                value=self.control_context.to_dict(),
            )
        )

        # 5. Build candidate ActionNodes from calibrated motor models
        candidate_actions: list[ActionNode] = []
        for a_id, model in self.action_models.items():
            if model.confidence >= 0.8 and (model.delta_r != 0 or model.delta_c != 0):
                act_node = ActionNode(
                    id=f"act_{a_id}",
                    intent=f"MOVE_A{a_id}",
                    properties={
                        "action_id": a_id,
                        "delta_r": model.delta_r,
                        "delta_c": model.delta_c,
                        "grid_shape": [H, W],
                        "barrier_cells": barrier_cells,
                        "target_positions": target_positions,
                    },
                )
                candidate_actions.append(act_node)

        return ws, goal_node, candidate_actions

    def _on_subgoal_resolved(self, subgoal_id: str) -> None:
        """Handle subgoal resolution: update HCIR, clear local gate barriers, and reset transient blocks."""
        self.decomposer.resolve_subgoal(self.workspace, subgoal_id)
        self.blocked_actions.clear()
        self.visited_positions.clear()
        self.stuck_counter = 0
        if self.known_barriers is not None:
            H, W = self.known_barriers.shape
            if self.primary_goal_node:
                t_pos = self.primary_goal_node.properties.get("target_position")
                if t_pos:
                    tr, tc = int(round(t_pos[0])), int(round(t_pos[1]))
                    rad = max(2, int(self.step_size * 2.5))
                    r_min, r_max = max(0, tr - rad), min(H, tr + rad + 1)
                    c_min, c_max = max(0, tc - rad), min(W, tc + rad + 1)
                    self.known_barriers[r_min:r_max, c_min:c_max] = False
            node = self.workspace.graph.get_node(subgoal_id)
            if isinstance(node, GoalNode):
                s_pos = node.properties.get("target_position")
                if s_pos:
                    sr, sc = int(round(s_pos[0])), int(round(s_pos[1]))
                    rad = max(2, int(self.step_size * 2.0))
                    r_min, r_max = max(0, sr - rad), min(H, sr + rad + 1)
                    c_min, c_max = max(0, sc - rad), min(W, sc + rad + 1)
                    self.known_barriers[r_min:r_max, c_min:c_max] = False

    def update_causal_dynamics(
        self,
        action_id: int,
        prev_grid: np.ndarray,
        curr_grid: np.ndarray,
    ) -> None:
        """Infer avatar displacement, calibrate motor dynamics, and mark barrier collisions."""
        if prev_grid.shape != curr_grid.shape:
            return

        H, W = curr_grid.shape
        diff_mask = prev_grid != curr_grid
        if np.count_nonzero(diff_mask) > max(100, int(H * W * 0.10)):
            return
        diff_mask[0, :] = False
        diff_mask[H - 1, :] = False

        p_prev = (
            np.where(prev_grid == self.avatar_color)
            if self.avatar_color is not None
            else (np.array([]), np.array([]))
        )
        p_curr = (
            np.where(curr_grid == self.avatar_color)
            if self.avatar_color is not None
            else (np.array([]), np.array([]))
        )
        if len(p_prev[0]) > 0 and len(p_curr[0]) == 0 and self.avatar_centroid is not None:
            r_c = int(round(self.avatar_centroid[0]))
            c_c = int(round(self.avatar_centroid[1]))
            if 0 <= r_c < H and 0 <= c_c < W:
                new_c = int(curr_grid[r_c, c_c])
                new_c_count = int(np.count_nonzero(curr_grid == new_c))
                if (
                    new_c != 0
                    and new_c != self.avatar_color
                    and new_c_count <= max(25, int(curr_grid.size * 0.15))
                    and len(np.unique(curr_grid)) > 1
                ):
                    self.state_mutations.append(
                        StateMutationModel(
                            trigger_type="TILE_CONTACT",
                            trigger_pos=(r_c, c_c),
                            trigger_color=self.avatar_color,
                            mutation_type="COLOR_REMAP",
                            prior_value=self.avatar_color,
                            posterior_value=new_c,
                            confidence=0.95,
                        )
                    )
                    self.avatar_color = new_c
                    if self.active_goal_node and not getattr(
                        self.active_goal_node, "resolved", False
                    ):
                        self._on_subgoal_resolved(self.active_goal_node.id)

        # Morphological transition and dynamic obstacle analysis
        if self.avatar_color is not None and not np.array_equal(prev_grid, curr_grid):
            self._analyze_morphological_transitions(action_id, prev_grid, curr_grid)

        # 1. Action 5 (Interact/Rewind) Dynamics
        if action_id == 5:
            if action_id not in self.action_models:
                self.action_models[action_id] = ActionDynamicsModel(
                    action_id=action_id, delta_r=0, delta_c=0, confidence=0.99, probes_tested=1
                )
            if self.avatar_color is not None:
                import scipy.ndimage as ndi

                playable_mask = np.zeros_like(curr_grid, dtype=bool)
                if H >= 16 and W >= 16:
                    playable_mask[5 : H - 2, :] = True
                else:
                    playable_mask[:, :] = True
                lbls, n_comp = ndi.label((curr_grid == self.avatar_color) & playable_mask)
                if n_comp > 0:
                    best_comp = 1
                    if self.start_pos is not None:
                        best_dist = 999.0
                        for i in range(1, n_comp + 1):
                            pts = np.where(lbls == i)
                            cr, cc = float(np.mean(pts[0])), float(np.mean(pts[1]))
                            d = math.hypot(cr - self.start_pos[0], cc - self.start_pos[1])
                            if d < best_dist:
                                best_dist = d
                                best_comp = i
                    pts = np.where(lbls == best_comp)
                    curr_r = self._snap_coord(float(np.mean(pts[0])))
                    curr_c = self._snap_coord(float(np.mean(pts[1])))
                    if self.start_pos is not None:
                        dist_start = math.hypot(
                            curr_r - self.start_pos[0], curr_c - self.start_pos[1]
                        )
                        if dist_start <= max(1.5, self.step_size * 0.75):
                            self.action_5_affordance = "REWIND"
                            self.avatar_centroid = (
                                float(self.start_pos[0]),
                                float(self.start_pos[1]),
                            )
            if self.holding_item:
                self.action_5_affordance = "PICKUP_DROP"
            return

        # 2. Case: Avatar color already known
        if self.avatar_color is not None:
            p_prev = np.where(prev_grid == self.avatar_color)
            p_curr = np.where(curr_grid == self.avatar_color)

            # Check for discrete state mutation: avatar color changed upon stepping on tile
            if len(p_prev[0]) > 0 and len(p_curr[0]) == 0 and self.avatar_centroid is not None:
                r_c, c_c = int(round(self.avatar_centroid[0])), int(round(self.avatar_centroid[1]))
                if 0 <= r_c < H and 0 <= c_c < W:
                    new_c = int(curr_grid[r_c, c_c])
                    new_c_count = int(np.count_nonzero(curr_grid == new_c))
                    if (
                        new_c != 0
                        and new_c != self.avatar_color
                        and new_c_count <= max(25, int(curr_grid.size * 0.15))
                        and len(np.unique(curr_grid)) > 1
                    ):
                        self.state_mutations.append(
                            StateMutationModel(
                                trigger_type="TILE_CONTACT",
                                trigger_pos=(r_c, c_c),
                                mutation_type="COLOR_REMAP",
                                prior_value=self.avatar_color,
                                posterior_value=new_c,
                                confidence=0.95,
                            )
                        )
                        self.avatar_color = new_c
                        if self.active_goal_node and not getattr(
                            self.active_goal_node, "resolved", False
                        ):
                            self._on_subgoal_resolved(self.active_goal_node.id)

            playable_diff = diff_mask.copy()
            if H >= 16 and W >= 16:
                playable_diff[:5, :] = False
                playable_diff[-3:, :] = False
            vanished_mask = (prev_grid == self.avatar_color) & playable_diff
            appeared_mask = (curr_grid == self.avatar_color) & playable_diff

            import scipy.ndimage as ndi

            v_lbls, n_v = ndi.label(vanished_mask)
            a_lbls, n_a = ndi.label(appeared_mask)

            chosen_v_pts = None
            chosen_a_pts = None

            if n_v > 0 and n_a > 0:
                if n_v == 1:
                    chosen_v_pts = np.where(v_lbls == 1)
                elif self.avatar_centroid is not None:
                    best_dist = 999.0
                    for i in range(1, n_v + 1):
                        pts = np.where(v_lbls == i)
                        cr, cc = float(np.mean(pts[0])), float(np.mean(pts[1]))
                        d = math.hypot(cr - self.avatar_centroid[0], cc - self.avatar_centroid[1])
                        if d < best_dist:
                            best_dist = d
                            chosen_v_pts = pts
                else:
                    chosen_v_pts = np.where(v_lbls == 1)

                if chosen_v_pts is not None:
                    v_cr, v_cc = float(np.mean(chosen_v_pts[0])), float(np.mean(chosen_v_pts[1]))
                    if n_a == 1:
                        chosen_a_pts = np.where(a_lbls == 1)
                    else:
                        best_dist = 999.0
                        for i in range(1, n_a + 1):
                            pts = np.where(a_lbls == i)
                            cr, cc = float(np.mean(pts[0])), float(np.mean(pts[1]))
                            d = math.hypot(cr - v_cr, cc - v_cc)
                            if d < best_dist:
                                best_dist = d
                                chosen_a_pts = pts

            if chosen_v_pts is not None and chosen_a_pts is not None:
                old_r = float(np.mean(chosen_v_pts[0]))
                old_c = float(np.mean(chosen_v_pts[1]))
                new_r = float(np.mean(chosen_a_pts[0]))
                new_c = float(np.mean(chosen_a_pts[1]))
                dr = int(round(new_r - old_r))
                dc = int(round(new_c - old_c))
            elif len(p_prev[0]) > 0 and len(p_curr[0]) > 0:
                old_r = float(np.mean(p_prev[0]))
                old_c = float(np.mean(p_prev[1]))
                new_r = float(np.mean(p_curr[0]))
                new_c = float(np.mean(p_curr[1]))
                dr = int(round(new_r - old_r))
                dc = int(round(new_c - old_c))
            else:
                old_r = old_c = new_r = new_c = None
                dr = dc = 0

            if old_r is not None and old_c is not None and new_r is not None and new_c is not None:
                # Only update translation dynamics if displacement is significant (not in-place sprite rotation)
                is_real_translation = max(abs(dr), abs(dc)) >= max(1, int(self.step_size * 0.75))
                if is_real_translation:
                    max_jump = max(8, int(min(H, W) * 0.25))
                    if max(abs(dr), abs(dc)) <= max_jump:
                        prev_step_size = self.step_size
                        self.step_size = max(self.step_size, abs(dr), abs(dc))
                        self.spatial_planner.step_size = self.step_size
                        if self.step_size != prev_step_size:
                            for act_k, act_m in self.action_models.items():
                                if act_k in [1, 2, 3, 4]:
                                    if act_m.delta_r != 0:
                                        act_m.delta_r = int(np.sign(act_m.delta_r)) * self.step_size
                                    if act_m.delta_c != 0:
                                        act_m.delta_c = int(np.sign(act_m.delta_c)) * self.step_size
                        self.current_facing = (int(np.sign(dr)), int(np.sign(dc)))

                        norm_dr = int(np.sign(dr)) * self.step_size if dr != 0 else 0
                        norm_dc = int(np.sign(dc)) * self.step_size if dc != 0 else 0

                        if action_id not in self.action_models:
                            self.action_models[action_id] = ActionDynamicsModel(
                                action_id=action_id,
                                delta_r=norm_dr,
                                delta_c=norm_dc,
                                confidence=0.95,
                                probes_tested=1,
                            )
                        else:
                            m = self.action_models[action_id]
                            m.delta_r = norm_dr
                            m.delta_c = norm_dc
                            m.confidence = min(0.99, m.confidence + 0.1)
                            m.probes_tested += 1

                        if 1 not in self.action_models:
                            self.action_models[1] = ActionDynamicsModel(
                                1, -self.step_size, 0, 0.95, 1
                            )
                        if 2 not in self.action_models:
                            self.action_models[2] = ActionDynamicsModel(
                                2, self.step_size, 0, 0.95, 1
                            )
                        if 3 not in self.action_models:
                            self.action_models[3] = ActionDynamicsModel(
                                3, 0, -self.step_size, 0.95, 1
                            )
                        if 4 not in self.action_models:
                            self.action_models[4] = ActionDynamicsModel(
                                4, 0, self.step_size, 0.95, 1
                            )

                    snap_r = self._snap_coord(new_r)
                    snap_c = self._snap_coord(new_c)
                    self.raw_avatar_centroid = (float(new_r), float(new_c))
                    self.avatar_centroid = (float(snap_r), float(snap_c))
                    self.visited_positions.append((snap_r, snap_c))
                    self.visited_cells.add((snap_r, snap_c))
                    self.learned_walkable_cells.add((snap_r, snap_c))
                    if self.start_pos is None:
                        self.start_pos = (self._snap_coord(old_r), self._snap_coord(old_c))
                        self.raw_start_pos = (float(old_r), float(old_c))
                        self.learned_walkable_cells.add(self.start_pos)

                    # Dynamic empirical discovery: the cell traversed in prev_grid is confirmed walkable
                    ir_new, ic_new = int(round(new_r)), int(round(new_c))
                    if 0 <= ir_new < H and 0 <= ic_new < W:
                        traversed_col = int(prev_grid[ir_new, ic_new])
                        if (
                            traversed_col != 0
                            and traversed_col != self.avatar_color
                            and traversed_col not in self.learned_receptacle_colors
                        ):
                            self.learned_walkable_colors.add(traversed_col)
                            self.learned_barrier_colors.discard(traversed_col)

                    # Dynamic empirical discovery: carried object in motion confirms item color
                    if self.holding_item and self.carried_offset != (0.0, 0.0):
                        c_r = int(round(snap_r + self.carried_offset[0]))
                        c_c = int(round(snap_c + self.carried_offset[1]))
                        if 0 <= c_r < H and 0 <= c_c < W:
                            carried_col = int(curr_grid[c_r, c_c])
                            if (
                                carried_col != 0
                                and carried_col != self.avatar_color
                                and carried_col not in self.learned_walkable_colors
                            ):
                                self.learned_item_colors.add(carried_col)
                                self.learned_receptacle_colors.discard(carried_col)
                                self.learned_barrier_colors.discard(carried_col)

                    if len(self.visited_positions) > 30:
                        self.visited_positions.pop(0)

                    self.blocked_actions.clear()
                    self.stuck_counter = 0

                    if (
                        self.active_goal_node
                        and self.primary_goal_node
                        and self.active_goal_node.id != self.primary_goal_node.id
                        and HierarchicalGoalDecomposer.check_subgoal_completion(
                            self.active_goal_node,
                            (int(round(snap_r)), int(round(snap_c))),
                            tolerance=self.step_size * 1.2,
                        )
                    ):
                        self._on_subgoal_resolved(self.active_goal_node.id)
                        self.active_goal_node = None

                    # Causal Mutability / Affordance Detection
                    non_avatar_diff = diff_mask.copy()
                    prev_av_mask = prev_grid == self.avatar_color
                    curr_av_mask = curr_grid == self.avatar_color
                    non_avatar_diff[prev_av_mask] = False
                    non_avatar_diff[curr_av_mask] = False
                    if np.any(non_avatar_diff):
                        diff_pts = np.where(non_avatar_diff)
                        portal_r = self._snap_coord(float(np.mean(diff_pts[0])))
                        portal_c = self._snap_coord(float(np.mean(diff_pts[1])))
                        act_key = f"act_{snap_r}_{snap_c}"
                        port_key = f"portal_{portal_r}_{portal_c}"
                        self.actuator_to_portals.setdefault(act_key, set()).add(port_key)
                    return
                else:
                    # Blocked by obstacle or wall
                    p_curr = np.where(curr_grid == self.avatar_color)
                    if len(p_curr[0]) > 0:
                        old_r = float(np.mean(p_curr[0]))
                        old_c = float(np.mean(p_curr[1]))
                    elif self.avatar_centroid is not None:
                        old_r, old_c = self.avatar_centroid
                    else:
                        old_r, old_c = (0.0, 0.0)

                    if action_id in self.action_models:
                        m = self.action_models[action_id]
                        m.probes_tested += 1
                        if m.delta_r != 0 or m.delta_c != 0:
                            dest_r = int(round(old_r + m.delta_r))
                            dest_c = int(round(old_c + m.delta_c))
                            self.spatial_planner.record_collision_barrier((dest_r, dest_c))
                            self.spatial_planner.record_failure(
                                workspace=self.workspace,
                                session_id=f"lvl_{self.level_step_counter}",
                                failed_action=action_id,
                                failure_pos=(dest_r, dest_c),
                                reason="collision",
                            )
                            self.cross_game_memory.record_failure(
                                session_id=f"lvl_{self.level_step_counter}",
                                reason="collision",
                                failed_action=action_id,
                                failure_pos=(dest_r, dest_c),
                            )
                            self.visited_positions.append((int(round(old_r)), int(round(old_c))))
                            if len(self.visited_positions) > 30:
                                self.visited_positions.pop(0)
                            if self.known_barriers is None:
                                self.known_barriers = np.zeros(curr_grid.shape, dtype=bool)
                            half_w = max(0, (self.step_size - 1) // 2)
                            for b_dr in range(-half_w, half_w + 1):
                                for b_dc in range(-half_w, half_w + 1):
                                    br, bc = dest_r + b_dr, dest_c + b_dc
                                    if 0 <= br < H and 0 <= bc < W:
                                        self.known_barriers[br, bc] = True
                            if 0 <= dest_r < H and 0 <= dest_c < W:
                                blk_col = int(curr_grid[dest_r, dest_c])
                                if (
                                    blk_col != 0
                                    and blk_col != self.avatar_color
                                    and blk_col not in self.learned_walkable_colors
                                    and blk_col not in self.learned_item_colors
                                ):
                                    self.learned_barrier_colors.add(blk_col)
                                    self.known_barriers[curr_grid == blk_col] = True
                                    if self.target_zone_bounds:
                                        tz_min_r, tz_max_r, tz_min_c, tz_max_c = (
                                            self.target_zone_bounds
                                        )
                                        self.known_barriers[
                                            max(0, tz_min_r - 1) : min(H, tz_max_r + 2),
                                            max(0, tz_min_c - 1) : min(W, tz_max_c + 2),
                                        ] = False

                    self.current_plan.clear()
                    self.action_queue.clear()
                    self.blocked_actions.add(action_id)
                    self.stuck_counter += 1
                    return

        # 3. Case: Discover Avatar by finding rigid moving pixel cluster
        counts = np.bincount(prev_grid.ravel())
        bg_color = int(np.argmax(counts))

        candidates = []
        for col in np.unique(prev_grid):
            if col == 0 or col == bg_color:
                continue
            p_prev = np.where(prev_grid == col)
            p_curr = np.where(curr_grid == col)
            n_prev, n_curr = len(p_prev[0]), len(p_curr[0])
            if 0 < n_prev < 300 and 0 < n_curr < 300 and abs(n_prev - n_curr) <= 2:
                dr_f = float(np.mean(p_curr[0]) - np.mean(p_prev[0]))
                dc_f = float(np.mean(p_curr[1]) - np.mean(p_prev[1]))
                if abs(dr_f) > 0.5 or abs(dc_f) > 0.5:
                    candidates.append(
                        (
                            int(col),
                            dr_f,
                            dc_f,
                            (float(np.mean(p_prev[0])), float(np.mean(p_prev[1]))),
                            (float(np.mean(p_curr[0])), float(np.mean(p_curr[1]))),
                            n_curr,
                        )
                    )

        if candidates:
            candidates.sort(key=lambda x: x[5])
            col, dr_f, dc_f, pr_pos, curr_pos, _ = candidates[0]
            dr = int(round(dr_f))
            dc = int(round(dc_f))
            self.avatar_color = col
            self.step_size = max(1, abs(dr), abs(dc))
            self.spatial_planner.step_size = self.step_size
            norm_dr = int(np.sign(dr)) * self.step_size if abs(dr) > 0.5 else 0
            norm_dc = int(np.sign(dc)) * self.step_size if abs(dc) > 0.5 else 0
            self.action_models[action_id] = ActionDynamicsModel(
                action_id=action_id,
                delta_r=norm_dr,
                delta_c=norm_dc,
                confidence=0.95,
                probes_tested=1,
            )
            if 1 not in self.action_models:
                self.action_models[1] = ActionDynamicsModel(1, -self.step_size, 0, 0.95, 1)
            if 2 not in self.action_models:
                self.action_models[2] = ActionDynamicsModel(2, self.step_size, 0, 0.95, 1)
            if 3 not in self.action_models:
                self.action_models[3] = ActionDynamicsModel(3, 0, -self.step_size, 0.95, 1)
            if 4 not in self.action_models:
                self.action_models[4] = ActionDynamicsModel(4, 0, self.step_size, 0.95, 1)
            self.raw_start_pos = pr_pos
            self.raw_avatar_centroid = curr_pos
            self.start_pos = (self._snap_coord(pr_pos[0]), self._snap_coord(pr_pos[1]))
            curr_snap = (self._snap_coord(curr_pos[0]), self._snap_coord(curr_pos[1]))
            self.avatar_centroid = (float(curr_snap[0]), float(curr_snap[1]))
            self.learned_walkable_cells.add(self.start_pos)
            self.learned_walkable_cells.add((int(round(curr_pos[0])), int(round(curr_pos[1]))))
            self.visited_cells.add((int(round(curr_pos[0])), int(round(curr_pos[1]))))
            self.visited_positions.append((int(round(curr_pos[0])), int(round(curr_pos[1]))))
            if not np.array_equal(prev_grid, curr_grid):
                self._analyze_morphological_transitions(action_id, prev_grid, curr_grid)
            return
        else:
            self.probe_step_counter += 1
            if not np.array_equal(prev_grid, curr_grid):
                self._analyze_morphological_transitions(action_id, prev_grid, curr_grid)


# ── Backward-compatible aliases ──────────────────────────────────────────────
ARC3InteractiveAgent = ARC3SpatialCognitiveAgent

# Re-export benchmark classes from canonical location for backward compat
from plugins.arc_agi_adapter.arc_agi_3_runner import (  # noqa: E402, F401
    ARC3BenchmarkReport,
    ARC3BenchmarkRunner,
    ARC3EnvironmentResult,
    ARC3LevelResult,
)
