"""ARC-3 Spatial Cognitive Agent — Autonomous, General Spatial Intelligence via HCIR.

Delegates scene lifting, topological cut-set discovery, state-space sequence planning,
and constraint induction to HCIRSpatialEntityPlanner, PhysicsPredictor, and native HCIR memory.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from hbllm.hcir.graph import (
    EpisodeNode,
    NodeLifecycle,
    Provenance,
    Scope,
)
from hbllm.hcir.spatial_planner import (
    EntityGraph,
    EntityRole,
    HCIRSpatialEntityPlanner,
    SequencePlanStep,
    SpatialEntity,
)
from hbllm.hcir.topological_cut_set import TopologicalCutSetAnalyzer
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.world.predictors.physics import PhysicsPredictor
from plugins.arc_agi_adapter.arc_agi_3_runner import (
    ActionDynamicsModel,
    ARCGrid,
    GridTopologyExtractor,
)

logger = logging.getLogger(__name__)


class AgentPhase(str, Enum):
    """Cognitive lifecycle phases for autonomous spatial reasoning."""

    EPISTEMIC_LEARNING = "EPISTEMIC_LEARNING"
    SOFT_RESTART_PENDING = "SOFT_RESTART_PENDING"
    OPTIMAL_EXECUTION = "OPTIMAL_EXECUTION"
    COMPLETED = "COMPLETED"


@dataclass
class HCIRCrossGameMemory:
    """Persistent cross-game and cross-level HCIR memory store."""

    action_models: dict[int, ActionDynamicsModel] = field(default_factory=dict)
    avatar_color: int | None = None
    step_size: int = 1
    action_5_affordance: str = "UNKNOWN"
    learned_barrier_colors: set[int] = field(default_factory=set)
    learned_walkable_colors: set[int] = field(default_factory=set)
    learned_item_colors: set[int] = field(default_factory=set)
    learned_receptacle_colors: set[int] = field(default_factory=set)
    negative_constraints: set[tuple[int, int]] = field(default_factory=set)
    successful_episodes: list[dict[str, Any]] = field(default_factory=list)
    failed_episodes: list[dict[str, Any]] = field(default_factory=list)

    def transfer_to_agent(self, agent: Any) -> None:
        """Transfer accumulated knowledge zero-shot to a new or reset agent."""
        if self.avatar_color is not None and agent.avatar_color is None:
            agent.avatar_color = self.avatar_color
        if self.step_size > 1:
            agent.step_size = max(agent.step_size, self.step_size)
            agent.spatial_planner.step_size = agent.step_size
        for a, m in self.action_models.items():
            if a not in agent.action_models or agent.action_models[a].confidence < m.confidence:
                agent.action_models[a] = ActionDynamicsModel(
                    action_id=m.action_id,
                    delta_r=m.delta_r,
                    delta_c=m.delta_c,
                    confidence=m.confidence,
                    probes_tested=m.probes_tested,
                )
        if self.action_5_affordance != "UNKNOWN" and agent.action_5_affordance == "UNKNOWN":
            agent.action_5_affordance = self.action_5_affordance
        agent.learned_barrier_colors.update(self.learned_barrier_colors)
        agent.learned_walkable_colors.update(self.learned_walkable_colors)
        agent.learned_item_colors.update(self.learned_item_colors)
        agent.learned_receptacle_colors.update(self.learned_receptacle_colors)
        for pos in self.negative_constraints:
            agent.spatial_planner.record_collision_barrier(pos)

    def update_from_agent(self, agent: Any) -> None:
        """Harvest discoveries and causal models from agent upon episode or trial completion."""
        if agent.avatar_color is not None:
            self.avatar_color = agent.avatar_color
        if agent.step_size > 1:
            self.step_size = max(self.step_size, agent.step_size)
        for a, m in agent.action_models.items():
            if m.confidence >= 0.8:
                self.action_models[a] = ActionDynamicsModel(
                    action_id=m.action_id,
                    delta_r=m.delta_r,
                    delta_c=m.delta_c,
                    confidence=m.confidence,
                    probes_tested=m.probes_tested,
                )
        if agent.action_5_affordance != "UNKNOWN":
            self.action_5_affordance = agent.action_5_affordance
        self.learned_barrier_colors.update(agent.learned_barrier_colors)
        self.learned_walkable_colors.update(agent.learned_walkable_colors)
        self.learned_item_colors.update(agent.learned_item_colors)
        self.learned_receptacle_colors.update(agent.learned_receptacle_colors)
        self.negative_constraints.update(agent.spatial_planner._learned_barriers)

    def record_success(
        self,
        session_id: str,
        tasks: list[str],
        score: float = 1.0,
    ) -> None:
        """Record corroborated successful trial in persistent memory."""
        self.successful_episodes.append(
            {
                "session_id": session_id,
                "tasks": tasks,
                "score": score,
                "timestamp": math.floor(1e6),
            }
        )

    def record_failure(
        self,
        session_id: str,
        reason: str,
        failed_action: int,
        failure_pos: tuple[int, int],
    ) -> None:
        """Record trial failure and update negative constraints."""
        if failure_pos:
            self.negative_constraints.add(failure_pos)
        self.failed_episodes.append(
            {
                "session_id": session_id,
                "reason": reason,
                "failed_action": failed_action,
                "failure_pos": failure_pos,
            }
        )

    def export_dict(self) -> dict[str, Any]:
        """Export serialized dictionary for cross-game transfer."""
        return {
            "avatar_color": self.avatar_color,
            "step_size": self.step_size,
            "action_5_affordance": self.action_5_affordance,
            "learned_barrier_colors": list(self.learned_barrier_colors),
            "learned_walkable_colors": list(self.learned_walkable_colors),
            "learned_item_colors": list(self.learned_item_colors),
            "learned_receptacle_colors": list(self.learned_receptacle_colors),
            "negative_constraints": [list(p) for p in self.negative_constraints],
            "action_models": {
                a: {
                    "action_id": m.action_id,
                    "delta_r": m.delta_r,
                    "delta_c": m.delta_c,
                    "confidence": m.confidence,
                    "probes_tested": m.probes_tested,
                }
                for a, m in self.action_models.items()
            },
            "successful_episodes": self.successful_episodes,
            "failed_episodes": self.failed_episodes,
        }

    def import_dict(self, data: dict[str, Any]) -> None:
        """Import knowledge from serialized dictionary."""
        self.avatar_color = data.get("avatar_color", self.avatar_color)
        self.step_size = data.get("step_size", self.step_size)
        self.action_5_affordance = data.get("action_5_affordance", self.action_5_affordance)
        self.learned_barrier_colors.update(data.get("learned_barrier_colors", []))
        self.learned_walkable_colors.update(data.get("learned_walkable_colors", []))
        self.learned_item_colors.update(data.get("learned_item_colors", []))
        self.learned_receptacle_colors.update(data.get("learned_receptacle_colors", []))
        self.negative_constraints.update(tuple(p) for p in data.get("negative_constraints", []))
        for a_str, md in data.get("action_models", {}).items():
            a = int(a_str)
            self.action_models[a] = ActionDynamicsModel(
                action_id=md["action_id"],
                delta_r=md["delta_r"],
                delta_c=md["delta_c"],
                confidence=md["confidence"],
                probes_tested=md.get("probes_tested", 1),
            )
        self.successful_episodes.extend(data.get("successful_episodes", []))
        self.failed_episodes.extend(data.get("failed_episodes", []))


class ARCPerceptualLifter:
    """Perceptual front-end for ARC-AGI-3 environments.

    Lifts raw pixel grids and segmented objects into domain-agnostic SpatialEntity instances
    and barrier coordinate sets for HCIRSpatialEntityPlanner.
    """

    @staticmethod
    def lift(
        grid: np.ndarray,
        raw_objects: list[Any],
        avatar_color: int | None = None,
        avatar_centroid: tuple[float, float] | None = None,
        learned_item_colors: set[int] | None = None,
        learned_receptacle_colors: set[int] | None = None,
        learned_barrier_colors: set[int] | None = None,
        walkable_colors: set[int] | None = None,
        step_size: int = 1,
        target_zone_bounds: tuple[int, int, int, int] | None = None,
    ) -> tuple[list[SpatialEntity], set[tuple[int, int]]]:
        H, W = grid.shape
        step = step_size
        counts = np.bincount(grid.ravel())
        bg_color = int(np.argmax(counts))
        walkable = set(walkable_colors or set()) | {0, bg_color}
        for col, count in enumerate(counts):
            if count >= int(H * W * 0.20) and col != avatar_color:
                walkable.add(int(col))
        learned_b = (learned_barrier_colors or set()) - walkable
        learned_i = (learned_item_colors or set()) - walkable
        learned_r = (learned_receptacle_colors or set()) - walkable

        raw_barriers: set[tuple[int, int]] = set()
        for b_col in learned_b:
            if b_col not in walkable and b_col != avatar_color:
                raw_barriers.update(set(zip(*np.where(grid == b_col))))

        # Detect collinear segmented wall blocks
        from collections import defaultdict

        col_blocks = defaultdict(list)
        row_blocks = defaultdict(list)
        for o in raw_objects:
            if (
                o.color in walkable
                or o.color == 0
                or (avatar_color is not None and o.color == avatar_color)
            ):
                continue
            span_r = o.max_r - o.min_r
            span_c = o.max_c - o.min_c
            # Single continuous wall spanning >= 40% of grid
            if span_r >= int(H * 0.4) and span_c <= max(4, step * 2):
                for br in range(o.min_r, o.max_r + 1):
                    for bc in range(o.min_c, o.max_c + 1):
                        raw_barriers.add((br, bc))
            elif span_c >= int(W * 0.4) and span_r <= max(4, step * 2):
                for br in range(o.min_r, o.max_r + 1):
                    for bc in range(o.min_c, o.max_c + 1):
                        raw_barriers.add((br, bc))
            else:
                col_key = int(round(o.centroid[1] / step)) * step
                row_key = int(round(o.centroid[0] / step)) * step
                col_blocks[(o.color, col_key)].append(o)
                row_blocks[(o.color, row_key)].append(o)

        for (b_col, c_pos), b_list in col_blocks.items():
            if len(b_list) >= 3:
                min_r = min(o.min_r for o in b_list)
                max_r = max(o.max_r for o in b_list)
                if (max_r - min_r) >= int(H * 0.4):
                    for o in b_list:
                        for br in range(o.min_r, o.max_r + 1):
                            for bc in range(o.min_c, o.max_c + 1):
                                raw_barriers.add((br, bc))

        for (b_col, r_pos), b_list in row_blocks.items():
            if len(b_list) >= 3:
                min_c = min(o.min_c for o in b_list)
                max_c = max(o.max_c for o in b_list)
                if (max_c - min_c) >= int(W * 0.4):
                    for o in b_list:
                        for br in range(o.min_r, o.max_r + 1):
                            for bc in range(o.min_c, o.max_c + 1):
                                raw_barriers.add((br, bc))

        # Detect receptacle bounds to prevent interior cavity colors from being classified as items
        receptacle_bounds = []
        min_receptacle_span = max(3, step * 2)
        min_receptacle_area = min_receptacle_span * min_receptacle_span
        if target_zone_bounds:
            receptacle_bounds.append(target_zone_bounds)
        else:
            for o in raw_objects:
                if o.color in walkable or o.color == 0 or o.area >= int(H * W * 0.35):
                    continue
                if (
                    o.color in learned_r
                    or getattr(o, "is_frame", False)
                    or (
                        o.color not in (avatar_color, 0)
                        and (o.max_r - o.min_r >= min_receptacle_span)
                        and (o.max_c - o.min_c >= min_receptacle_span)
                        and o.area >= min_receptacle_area
                    )
                ):
                    receptacle_bounds.append((o.min_r, o.max_r, o.min_c, o.max_c))

        entities: list[SpatialEntity] = []
        max_item_area = max(16, int((step * 2) ** 2 * 2.5))

        if target_zone_bounds:
            tz_r = (
                int(round(((target_zone_bounds[0] + target_zone_bounds[1]) * 0.5) / step)) * step
                if step > 1
                else int(round((target_zone_bounds[0] + target_zone_bounds[1]) * 0.5))
            )
            tz_c = (
                int(round(((target_zone_bounds[2] + target_zone_bounds[3]) * 0.5) / step)) * step
                if step > 1
                else int(round((target_zone_bounds[2] + target_zone_bounds[3]) * 0.5))
            )
            receptacle_ent = SpatialEntity(
                id=f"receptacle_{target_zone_bounds[0]}_{target_zone_bounds[2]}",
                role=EntityRole.RECEPTACLE,
                centroid=(
                    (target_zone_bounds[0] + target_zone_bounds[1]) * 0.5,
                    (target_zone_bounds[2] + target_zone_bounds[3]) * 0.5,
                ),
                grid_pos=(tz_r, tz_c),
                area=(target_zone_bounds[1] - target_zone_bounds[0] + 1)
                * (target_zone_bounds[3] - target_zone_bounds[2] + 1),
                bounding_box=target_zone_bounds,
            )
            entities.append(receptacle_ent)

        for o in raw_objects:
            if o.color in walkable or o.color == 0 or o.area >= int(H * W * 0.30):
                continue
            if H >= 16 and W >= 16 and (o.min_r <= 4 or o.max_r >= H - 2):
                is_av_cand = (avatar_color is not None and o.color == avatar_color) or (
                    avatar_centroid is not None
                    and math.hypot(
                        o.centroid[0] - avatar_centroid[0], o.centroid[1] - avatar_centroid[1]
                    )
                    < step * 1.5
                )
                if not is_av_cand:
                    continue

            # 1. Avatar / Goal check
            is_avatar = False
            is_goal = False
            if avatar_color is not None and o.color == avatar_color:
                if avatar_centroid is not None:
                    d_av = math.hypot(
                        o.centroid[0] - avatar_centroid[0], o.centroid[1] - avatar_centroid[1]
                    )
                    if d_av <= step * 1.5:
                        is_avatar = True
                    elif d_av > step * 1.5:
                        is_goal = True
                else:
                    is_avatar = True
            elif (
                avatar_centroid
                and math.hypot(
                    o.centroid[0] - avatar_centroid[0], o.centroid[1] - avatar_centroid[1]
                )
                < 2.0
            ):
                is_avatar = True

            r = int(round(o.centroid[0] / step)) * step if step > 1 else int(round(o.centroid[0]))
            c = int(round(o.centroid[1] / step)) * step if step > 1 else int(round(o.centroid[1]))
            e_id = f"ent_{o.color}_{r}_{c}"

            if is_avatar:
                ent = SpatialEntity(
                    id=e_id,
                    role=EntityRole.AGENT,
                    centroid=(float(o.centroid[0]), float(o.centroid[1])),
                    grid_pos=(r, c),
                    area=int(o.area),
                    bounding_box=(int(o.min_r), int(o.max_r), int(o.min_c), int(o.max_c)),
                    color=int(o.color),
                )
                entities.append(ent)
                continue

            if is_goal:
                ent = SpatialEntity(
                    id=e_id,
                    role=EntityRole.GOAL,
                    centroid=(float(o.centroid[0]), float(o.centroid[1])),
                    grid_pos=(r, c),
                    area=int(o.area),
                    bounding_box=(int(o.min_r), int(o.max_r), int(o.min_c), int(o.max_c)),
                    color=int(o.color),
                )
                entities.append(ent)
                continue

            # 2. If inside receptacle bounds, it is part of receptacle or a delivered item
            is_inside_receptacle = any(
                b[0] <= o.min_r and o.max_r <= b[1] and b[2] <= o.min_c and o.max_c <= b[3]
                for b in receptacle_bounds
            )
            if is_inside_receptacle and target_zone_bounds:
                # Any external object inside receptacle (delivered items, NPC bots) is a physical obstacle
                if o.color not in learned_r and o.color not in walkable and o.color != avatar_color:
                    if hasattr(o, "coords"):
                        raw_barriers.update((int(cr), int(cc)) for cr, cc in o.coords)
                    else:
                        for br in range(o.min_r, o.max_r + 1):
                            for bc in range(o.min_c, o.max_c + 1):
                                raw_barriers.add((br, bc))
                continue

            role = EntityRole.UNKNOWN
            if (
                o.color in learned_b
                or (int(round(o.centroid[0])), int(round(o.centroid[1]))) in raw_barriers
                or (o.min_r, o.min_c) in raw_barriers
            ):
                role = EntityRole.OBSTACLE
                if hasattr(o, "coords"):
                    raw_barriers.update((int(cr), int(cc)) for cr, cc in o.coords)
                else:
                    for br in range(o.min_r, o.max_r + 1):
                        for bc in range(o.min_c, o.max_c + 1):
                            raw_barriers.add((br, bc))
            elif o.color in learned_r or is_inside_receptacle:
                role = EntityRole.RECEPTACLE
            elif (o.color in learned_i) or (
                not learned_i and o.color != avatar_color and o.area <= max_item_area
            ):
                role = EntityRole.MANIPULABLE
            else:
                role = EntityRole.ACTUATOR

            ent = SpatialEntity(
                id=e_id,
                role=role,
                centroid=(float(o.centroid[0]), float(o.centroid[1])),
                grid_pos=(r, c),
                area=int(o.area),
                bounding_box=(int(o.min_r), int(o.max_r), int(o.min_c), int(o.max_c)),
                color=int(o.color),
            )
            entities.append(ent)

        return entities, raw_barriers


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

        # Persistent Cross-Game Memory
        self.cross_game_memory: HCIRCrossGameMemory = (
            shared_memory if shared_memory is not None else ARC3SpatialCognitiveAgent.global_memory
        )
        self.cross_game_memory.transfer_to_agent(self)

    @classmethod
    def is_spatial_candidate(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Domain-agnostic check if environment possesses 2D movement and affordance action (Action 5)."""
        return 5 in available_actions and any(a in available_actions for a in [1, 2, 3, 4])

    @classmethod
    def is_cooperative_candidate(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Backward-compatible alias for is_spatial_candidate."""
        return cls.is_spatial_candidate(grid, available_actions)

    def reset_episode(self, retain_dynamics: bool = False) -> None:
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
        self.delivered_positions.clear()
        self.visited_positions.clear()
        self.blocked_actions.clear()
        self.stuck_counter = 0
        self.last_action = None
        self.current_plan.clear()
        self.probe_step_counter = 0
        self.level_step_counter = 0
        self.should_soft_restart = False
        self.is_cooperative_handoff = False
        self.known_barriers = None
        self.target_zone_bounds = None
        self.workspace = HCIRWorkspaceState()
        self.spatial_planner.reset()

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

    def _snap_coord(self, raw_val: float, axis: int | None = None) -> int:
        """Snap float coordinate to discrete lattice coordinate."""
        if self.step_size > 1:
            return int(round(raw_val / self.step_size)) * self.step_size
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
            e for e in eg.entities.values() if e.role in (EntityRole.PORTAL, EntityRole.DOORWAY)
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
        circuits = []
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
        walkable_lattice = set()
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
            return self._active_probe_action(available_actions), 0.50

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
        if 5 in available_actions and self.action_5_affordance != "PICKUP":
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
                    e.role in (EntityRole.PORTAL, EntityRole.DOORWAY)
                    and (e.grid_pos == drop_slot or e.id == active_step.target_entity_id)
                    for e in eg.entities.values()
                )
                is_blocked = (drop_slot in self.delivered_positions) or (
                    not is_portal_drop and drop_slot in eg.barriers
                )
                if not is_blocked and is_portal_drop:
                    # Portal drops only re-plan if another movable item is physically in the doorway
                    if any(
                        e.role in (EntityRole.MANIPULABLE, EntityRole.MOVABLE_ITEM)
                        and math.hypot(e.grid_pos[0] - drop_slot[0], e.grid_pos[1] - drop_slot[1])
                        < self.step_size * 0.75
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
                    e.role in (EntityRole.MANIPULABLE, EntityRole.MOVABLE_ITEM, EntityRole.ACTUATOR)
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
                            EntityRole.MOVABLE_ITEM,
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
            if e.role in (EntityRole.MANIPULABLE, EntityRole.MOVABLE_ITEM) and not e.is_delivered:
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
                EntityRole.EXIT,
                EntityRole.MANIPULABLE,
                EntityRole.MOVABLE_ITEM,
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
        return best_action, confidence

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
        diff_mask[0, :] = False
        diff_mask[H - 1, :] = False

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

                # Only update translation dynamics if displacement is significant (not in-place sprite rotation)
                is_real_translation = max(abs(dr), abs(dc)) >= max(1, int(self.step_size * 0.75))
                if is_real_translation:
                    max_jump = max(8, int(min(H, W) * 0.25))
                    if max(abs(dr), abs(dc)) <= max_jump:
                        self.step_size = max(self.step_size, abs(dr), abs(dc))
                        self.spatial_planner.step_size = self.step_size
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
                        elif self.action_models[action_id].confidence < 0.9:
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
                            ):
                                self.learned_barrier_colors.add(blk_col)

                self.current_plan.clear()
                self.action_queue.clear()
                self.blocked_actions.add(action_id)
                self.stuck_counter += 1
                return

        # 3. Case: Discover Avatar by finding rigid moving pixel cluster
        counts = np.bincount(prev_grid.ravel())
        bg_color = int(np.argmax(counts))

        playable_diff = diff_mask.copy()
        if H >= 16 and W >= 16:
            playable_diff[:5, :] = False
            playable_diff[-3:, :] = False

        best_cand = None
        best_ratio = 0.0

        for col in np.unique(prev_grid):
            if col == 0 or col == bg_color:
                continue
            vanished = (prev_grid == col) & playable_diff
            appeared = (curr_grid == col) & playable_diff
            nv = int(np.sum(vanished))
            na = int(np.sum(appeared))
            total_count = int(np.sum(prev_grid == col))
            if nv > 0 and abs(nv - na) <= max(2, nv // 4):
                ratio = nv / max(1, total_count)
                if ratio >= 0.15 and ratio > best_ratio:
                    dr_f = float(np.mean(np.where(appeared)[0]) - np.mean(np.where(vanished)[0]))
                    dc_f = float(np.mean(np.where(appeared)[1]) - np.mean(np.where(vanished)[1]))
                    max_jump = max(8, int(min(H, W) * 0.25))
                    if 0.5 < max(abs(dr_f), abs(dc_f)) <= max_jump:
                        best_cand = (int(col), dr_f, dc_f, vanished, appeared)
                        best_ratio = ratio

        if best_cand is not None:
            col, dr_f, dc_f, vanished, appeared = best_cand
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
            pr_p = float(np.mean(np.where(vanished)[0]))
            pc_p = float(np.mean(np.where(vanished)[1]))
            pr_c = float(np.mean(np.where(appeared)[0]))
            pc_c = float(np.mean(np.where(appeared)[1]))
            self.raw_start_pos = (pr_p, pc_p)
            self.raw_avatar_centroid = (pr_c, pc_c)
            self.start_pos = (self._snap_coord(pr_p), self._snap_coord(pc_p))
            curr_pos = (self._snap_coord(pr_c), self._snap_coord(pc_c))
            self.avatar_centroid = (float(curr_pos[0]), float(curr_pos[1]))
            self.learned_walkable_cells.add(self.start_pos)
            self.learned_walkable_cells.add(curr_pos)
            self.visited_cells.add(curr_pos)
            self.visited_positions.append(curr_pos)
            return
        else:
            self.probe_step_counter += 1
