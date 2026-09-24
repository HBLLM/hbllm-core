"""HCIR Spatial Entity Planner — General Cognitive Path & Subgoal Sequence Reasoning.

Provides a general, level-agnostic intelligence feature that:
1. Lifts raw pixel grids and segmented objects into an Entity Graph (Avatar, Barriers, Items, Refills, Switches, Gates, Goals).
2. Decomposes environments into topological connected components and cut-sets (discovering partition doorways zero-shot).
3. Searches multi-goal state-space sequences (A* over (position, energy, inventory, world_state)) without hardcoded action sequences.
4. Synthesizes collision-free geodesic paths accounting for carried-object footprints.
5. Interfaces directly with HCIR's native memory architecture (EpisodeNode, BeliefNode, CognitiveGraph) to learn negative constraints from failure and soft-restart with accumulated knowledge.
"""

from __future__ import annotations

import heapq
import logging
import math
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from hbllm.hcir.graph import (
    BeliefNode,
    EpisodeNode,
    FalsificationStatus,
    HCIREdge,
    HCIREdgeType,
    HCIRNodeType,
    NodeLifecycle,
)
from hbllm.hcir.topological_cut_set import CutSetResult, TopologicalCutSetAnalyzer
from hbllm.hcir.types import Provenance, Scope
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.world.predictors.physics import PhysicsPredictor

logger = logging.getLogger(__name__)


class EntityRole(StrEnum):
    """Functional role of an extracted spatial entity in cognitive space."""

    AGENT = "agent"  # Controllable agent / avatar
    OBSTACLE = "obstacle"  # Static impassable barrier
    MANIPULABLE = "manipulable"  # Movable / portable item
    RECEPTACLE = "receptacle"  # Container, deposit slot, or drop zone
    PORTAL = "portal"  # Bottleneck transit point, doorway, or partition gate
    ACTUATOR = "actuator"  # Switch, lever, trigger
    RESOURCE = "resource"  # Energy refill, key, consumable
    GOAL = "goal"  # Terminal target / exit
    DYNAMIC_HAZARD = "dynamic_hazard"  # Moving hostile / obstacle
    COMPANION = "companion"  # Autonomous cooperative agent / companion
    UNKNOWN = "unknown"


class SpatialActionIntent(StrEnum):
    """Semantic action intents for spatial planning subgoals."""

    NAVIGATE = "NAVIGATE"
    INTERACT = "INTERACT"
    ACTUATE = "ACTIVATE"
    PICKUP = "PICKUP"
    DROP = "DROP"


# ── Backward Compatibility Aliases (deprecated, use canonical names above) ──
AVATAR = EntityRole.AGENT
BARRIER = EntityRole.OBSTACLE
MOVABLE_ITEM = EntityRole.MANIPULABLE
DOORWAY = EntityRole.PORTAL
SWITCH = EntityRole.ACTUATOR
REFILL = EntityRole.RESOURCE
EXIT = EntityRole.GOAL


@dataclass
class SpatialEntity:
    """A lifted entity in the cognitive spatial graph."""

    id: str
    role: EntityRole
    centroid: tuple[float, float]
    grid_pos: tuple[int, int]
    area: int
    bounding_box: tuple[int, int, int, int]  # (min_r, max_r, min_c, max_c)
    component_id: int = -1
    is_deliverable: bool = False
    is_delivered: bool = False
    shape_archetype: Any = None
    properties: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        id: str,
        role: EntityRole,
        centroid: tuple[float, float],
        grid_pos: tuple[int, int],
        area: int,
        bounding_box: tuple[int, int, int, int],
        feature_id: Any | None = None,
        color: int | None = None,
        component_id: int = -1,
        is_deliverable: bool = False,
        is_delivered: bool = False,
        shape_archetype: Any = None,
        properties: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.id = id
        self.role = role
        self.centroid = centroid
        self.grid_pos = grid_pos
        self.area = area
        self.bounding_box = bounding_box
        self.component_id = component_id
        self.is_deliverable = is_deliverable
        self.is_delivered = is_delivered
        self.shape_archetype = shape_archetype
        self.properties = dict(properties or {})
        feat = feature_id if feature_id is not None else color
        if feat is not None:
            self.properties["feature_id"] = feat
            self.properties["visual_id"] = feat
        if shape_archetype is not None:
            self.properties["shape_archetype"] = shape_archetype
        self.properties.update(kwargs)

    @property
    def feature_id(self) -> Any:
        """Generic perceptual feature identifier."""
        return self.properties.get("feature_id", self.properties.get("visual_id", 0))

    @feature_id.setter
    def feature_id(self, val: Any) -> None:
        self.properties["feature_id"] = val
        self.properties["visual_id"] = val

    @property
    def color(self) -> int:
        """Backward-compatible access to visual property."""
        return self.properties.get("visual_id", 0)

    @color.setter
    def color(self, val: int) -> None:
        self.properties["visual_id"] = val
        self.properties["feature_id"] = val


@dataclass
class EntityGraph:
    """Graph of spatial entities, topological partitions, and navigable geodesics."""

    entities: dict[str, SpatialEntity] = field(default_factory=dict)
    agent: SpatialEntity | None = None
    barriers: set[tuple[int, int]] = field(default_factory=set)
    components: dict[int, set[tuple[int, int]]] = field(default_factory=dict)
    entity_to_comp: dict[str, int] = field(default_factory=dict)
    cut_sets: list[CutSetResult] = field(default_factory=list)
    grid_shape: tuple[int, int] = (64, 64)
    step_size: int = 1

    @property
    def avatar(self) -> SpatialEntity | None:
        return self.agent

    @avatar.setter
    def avatar(self, value: SpatialEntity | None) -> None:
        self.agent = value


@dataclass
class SequencePlanStep:
    """A high-level subgoal step in the planned entity sequence."""

    target_entity_id: str
    target_pos: tuple[int, int]
    action_type: str = SpatialActionIntent.NAVIGATE  # SpatialActionIntent semantic intent
    approach_facing: tuple[int, int] | None = None
    carried_offset: tuple[int, int] | tuple[float, float] = (0, 0)
    expected_energy_cost: int = 0


@dataclass
class TimelinePlanStep:
    """A multi-timeline plan step for temporal clone cooperative execution."""

    timeline_index: int
    target_pos: tuple[int, int]
    path: list[tuple[int, int]]
    actions: list[int]
    is_rewind: bool


class HCIRSpatialEntityPlanner:
    """Universal cognitive spatial planning and constraint induction engine."""

    def __init__(self, step_size: int = 1) -> None:
        self.step_size: int = step_size
        self._learned_barriers: set[tuple[int, int]] = set()
        self._learned_switch_rules: dict[str, Any] = {}
        self._failed_sequences: set[tuple[str, ...]] = set()
        self._known_doorways: dict[tuple[int, int], tuple[int, int]] = {}

    def reset(self, is_retry: bool = False) -> None:
        """Resets episode-specific spatial memory caches."""
        self._known_doorways.clear()
        self._failed_sequences.clear()
        if not is_retry:
            self._learned_barriers.clear()

    def record_collision_barrier(self, attempted_pos: tuple[int, int]) -> None:
        """Records an empirically discovered impassable barrier from a blocked movement attempt."""
        self._learned_barriers.add(attempted_pos)

    def remove_collision_barrier(self, pos: tuple[int, int]) -> None:
        """Removes a previously recorded collision barrier when dynamic obstacles unblock or rotate."""
        self._learned_barriers.discard(pos)

    # ═══════════════════════════════════════════════════════════════════════
    # 1. TOPOLOGICAL ENTITY GRAPH CONSTRUCTION (DOMAIN-AGNOSTIC)
    # ═══════════════════════════════════════════════════════════════════════

    def construct_entity_graph(
        self,
        entities: list[SpatialEntity],
        barriers: set[tuple[int, int]],
        grid_shape: tuple[int, int],
        step_size: int | None = None,
        known_barriers: set[tuple[int, int]] | None = None,
    ) -> EntityGraph:
        """Constructs a topological EntityGraph from abstract spatial entities and barriers."""
        H, W = grid_shape
        step = step_size or self.step_size
        eg = EntityGraph(grid_shape=(H, W), step_size=step)

        if known_barriers:
            eg.barriers.update(known_barriers)
        eg.barriers.update(self._learned_barriers)
        eg.barriers.update(barriers)

        # Gestalt grouping: deduplicate co-located entities (e.g. concentric composite items)
        deduped_entities: dict[str, SpatialEntity] = {}
        for ent in entities:
            if ent.role == EntityRole.OBSTACLE:
                continue
            co_located = [
                ex_id
                for ex_id, ex in deduped_entities.items()
                if ex.grid_pos == ent.grid_pos
                and (ex.role == ent.role or ex.role == EntityRole.MANIPULABLE)
            ]
            if co_located:
                ex_id = co_located[0]
                if ent.area > deduped_entities[ex_id].area:
                    del deduped_entities[ex_id]
                    deduped_entities[ent.id] = ent
            else:
                deduped_entities[ent.id] = ent

        eg.entities = deduped_entities

        # Identify controllable agent
        for ent in eg.entities.values():
            if ent.role == EntityRole.AGENT:
                eg.agent = ent
                break

        # Physical transit barriers include physical items that occupy space and block transit
        transit_barriers = set(eg.barriers)
        movable_barriers = set()
        for ent in eg.entities.values():
            if ent.role == EntityRole.MANIPULABLE and ent != eg.agent:
                transit_barriers.add(ent.grid_pos)
                movable_barriers.add(ent.grid_pos)
        # Established doorway slots along a cut-set partition are boundary portals, not free transit
        transit_barriers.update(self._known_doorways.keys())

        # Compute connected reachability components based on physical transit barriers
        eg.components = self._compute_connected_components(transit_barriers, (H, W), step)

        # Assign each entity to a connected component
        for e_id, ent in eg.entities.items():
            comp_id = self._find_component_for_pos(ent.grid_pos, eg.components, step)
            ent.component_id = comp_id
            eg.entity_to_comp[e_id] = comp_id

        # Check for topological partitions between agent and goals / receptacles
        if eg.agent:
            receptacles = [
                e
                for e in eg.entities.values()
                if e.role in (EntityRole.RECEPTACLE, EntityRole.GOAL)
            ]

            for rec in receptacles:
                if rec.component_id != -1 and rec.component_id != eg.agent.component_id:
                    cut_res = TopologicalCutSetAnalyzer.analyze_cut_set(
                        eg.agent.grid_pos,
                        rec.grid_pos,
                        transit_barriers,
                        (H, W),
                        step,
                        movable_barriers=movable_barriers,
                    )
                    if cut_res.is_partitioned and cut_res.best_gate_cell and cut_res.approach_cell:
                        eg.cut_sets.append(cut_res)
                        # Partition entity components based on topological reachability across cut-set
                        eg.agent.component_id = 0
                        rec.component_id = 1
                        for e_id, ent in list(eg.entities.items()):
                            if ent.role == EntityRole.MANIPULABLE:
                                if (ent.grid_pos in cut_res.start_component) or any(
                                    math.hypot(ent.grid_pos[0] - r, ent.grid_pos[1] - c)
                                    <= step * 1.2
                                    for r, c in cut_res.start_component
                                ):
                                    ent.component_id = 0
                                else:
                                    ent.component_id = 1

                        created_gates = set()
                        gates_to_create = getattr(cut_res, "all_gates", []) or [
                            (cut_res.best_gate_cell, cut_res.approach_cell)
                        ]
                        for d_gate, d_appr in self._known_doorways.items():
                            if not any(g[0] == d_gate for g in gates_to_create):
                                gates_to_create.append((d_gate, d_appr))

                        for gate_cell, appr_cell in gates_to_create:
                            if gate_cell in movable_barriers or gate_cell in self._known_doorways:
                                self._known_doorways[gate_cell] = appr_cell
                            if (
                                gate_cell in movable_barriers
                                or gate_cell in self._known_doorways
                                or gate_cell == cut_res.best_gate_cell
                            ) and gate_cell not in created_gates:
                                created_gates.add(gate_cell)
                                portal_id = f"portal_{gate_cell[0]}_{gate_cell[1]}"
                                portal_ent = SpatialEntity(
                                    id=portal_id,
                                    role=EntityRole.PORTAL,
                                    centroid=(float(gate_cell[0]), float(gate_cell[1])),
                                    grid_pos=gate_cell,
                                    area=step * step,
                                    bounding_box=(
                                        gate_cell[0],
                                        gate_cell[0] + step,
                                        gate_cell[1],
                                        gate_cell[1] + step,
                                    ),
                                    component_id=eg.agent.component_id,
                                    properties={
                                        "gate_cell": gate_cell,
                                        "approach_cell": appr_cell,
                                    },
                                )
                                eg.entities[portal_id] = portal_ent

        return eg

    # ═══════════════════════════════════════════════════════════════════════
    # 2. SEQUENCE & MULTI-GOAL A* SEARCH OVER STATE TRANSITIONS
    # ═══════════════════════════════════════════════════════════════════════

    def plan_sequence(
        self,
        eg: EntityGraph,
        initial_energy: int | None = None,
        workspace: HCIRWorkspaceState | None = None,
        delivered_positions: set[tuple[int, int]] | None = None,
        is_carrying: bool = False,
        carried_offset: tuple[float, float] = (0.0, 0.0),
        item_colors: set[int] | None = None,
        **kwargs: Any,
    ) -> list[SequencePlanStep]:
        """Plans the sequence of entity interactions from first principles using state-space A*.

        Agent state is read from workspace graph variables when available.
        Explicit parameters override workspace-sourced values for backward compatibility.
        """
        if not eg.avatar:
            return []

        # Explored entities and positions across episodes/retries
        explored_ids: set[str] = set()
        explored_positions: set[tuple[int, int]] = set()

        # ── Resolve agent state from workspace graph (MIMO blackbox reads) ──
        if workspace is not None and hasattr(workspace, "graph"):
            from hbllm.hcir.graph import WorldVariableNode

            def _ws_var(name: str) -> Any:
                n = workspace.graph.get_node(name)
                return n.value if isinstance(n, WorldVariableNode) else None

            # Carrying state
            ws_carry = _ws_var("var_carrying_state")
            if ws_carry and isinstance(ws_carry, dict):
                if not is_carrying:
                    is_carrying = ws_carry.get("holding", False)
                if carried_offset == (0.0, 0.0):
                    ws_off = ws_carry.get("offset", (0.0, 0.0))
                    carried_offset = (float(ws_off[0]), float(ws_off[1]))

            # Delivered positions
            if delivered_positions is None:
                ws_deliv = _ws_var("var_delivered_positions")
                if ws_deliv and isinstance(ws_deliv, (list, set)):
                    delivered_positions = {tuple(p) for p in ws_deliv}

            # Target features (for filtering non-target entities)
            if item_colors is None:
                ws_ic = (
                    _ws_var("var_target_features")
                    or _ws_var("var_target_entity_features")
                    or _ws_var("var_learned_item_colors")
                )
                if ws_ic and isinstance(ws_ic, (list, set)):
                    item_colors = set(ws_ic)

            # Explored entities and positions (to avoid repetition loops across retries)
            ws_ids = _ws_var("var_explored_entity_ids")
            if ws_ids and isinstance(ws_ids, (list, set)):
                explored_ids = set(ws_ids)
            ws_pos = _ws_var("var_explored_entity_positions")
            if ws_pos and isinstance(ws_pos, (list, set)):
                explored_positions = {tuple(p) for p in ws_pos}

        # Fallback if memorized item colors do not match any entity in the current layout
        if item_colors and not any(e.color in item_colors for e in eg.entities.values()):
            item_colors = None

        # Recall active constraints from native HCIR memory
        impassable_gates, energy_records = self._recall_memory_constraints(workspace)

        avatar_comp = eg.avatar.component_id
        is_partitioned = len(eg.cut_sets) > 0

        # Topology 1: Topologically Partitioned Transit & Bottleneck Portal Delivery
        has_manipulable_items = (
            any(e.role == EntityRole.MANIPULABLE for e in eg.entities.values()) or is_carrying
        )
        if is_partitioned and has_manipulable_items:
            doorways = [e for e in eg.entities.values() if e.role == EntityRole.PORTAL]
            if doorways:
                doorway_positions = {d.properties.get("gate_cell", d.grid_pos) for d in doorways}
                occupied_gates = set()
                for ent in eg.entities.values():
                    if ent.role == EntityRole.MANIPULABLE and ent != eg.agent:
                        for gp in doorway_positions:
                            if (
                                math.hypot(ent.grid_pos[0] - gp[0], ent.grid_pos[1] - gp[1])
                                < eg.step_size * 0.75
                            ):
                                occupied_gates.add(gp)
                if delivered_positions:
                    occupied_gates.update(delivered_positions)

                unoccupied_doorways = [
                    d
                    for d in doorways
                    if d.properties.get("gate_cell", d.grid_pos) not in occupied_gates
                ]
                # Only consider portals whose approach cell is reachable from the
                # agent's current partition side.
                avatar_comp_set = eg.cut_sets[0].start_component if eg.cut_sets else set()
                reachable_unoccupied = [
                    d
                    for d in unoccupied_doorways
                    if d.properties.get("approach_cell", d.grid_pos) in avatar_comp_set
                ]
                reachable_occupied = [
                    d
                    for d in doorways
                    if d.properties.get("approach_cell", d.grid_pos) in avatar_comp_set
                    and d not in unoccupied_doorways
                ]
                # Prefer: unoccupied+reachable > occupied+reachable > any
                active_doorways = (
                    reachable_unoccupied or reachable_occupied or unoccupied_doorways or doorways
                )

                # If the agent is ALREADY carrying an item:
                if is_carrying:
                    doorway = active_doorways[0]
                    gate_cell = doorway.properties.get("gate_cell", doorway.grid_pos)
                    appr_cell = doorway.properties.get("approach_cell", doorway.grid_pos)
                    dr_handoff = int(np.sign(gate_cell[0] - appr_cell[0]))
                    dc_handoff = int(np.sign(gate_cell[1] - appr_cell[1]))
                    handoff_facing = (
                        (dr_handoff, 0) if abs(dr_handoff) >= abs(dc_handoff) else (0, dc_handoff)
                    )
                    if handoff_facing == (0, 0):
                        handoff_facing = (0, 1)
                    return [
                        SequencePlanStep(
                            target_entity_id=doorway.id,
                            target_pos=appr_cell,
                            action_type=SpatialActionIntent.DROP,
                            approach_facing=handoff_facing,
                            carried_offset=carried_offset,
                        )
                    ]

                # Collect uncarried, undelivered items in agent's component (excluding items already at gates)
                candidate_items = [
                    e
                    for e in eg.entities.values()
                    if e.role == EntityRole.MANIPULABLE
                    and e.component_id == avatar_comp
                    and not e.is_delivered
                    and (item_colors is None or e.color in item_colors)
                    and not any(
                        math.hypot(e.grid_pos[0] - gp[0], e.grid_pos[1] - gp[1])
                        < eg.step_size * 0.75
                        for gp in doorway_positions
                    )
                    and not (
                        delivered_positions
                        and (
                            e.grid_pos in delivered_positions
                            or any(
                                math.hypot(e.grid_pos[0] - dp[0], e.grid_pos[1] - dp[1])
                                < eg.step_size
                                for dp in delivered_positions
                            )
                        )
                    )
                ]

                # Sort candidate items by distance to agent
                candidate_items.sort(
                    key=lambda e: math.hypot(
                        e.grid_pos[0] - eg.avatar.grid_pos[0],
                        e.grid_pos[1] - eg.avatar.grid_pos[1],
                    )
                )

                plan: list[SequencePlanStep] = []
                pickup_facing = (0, 1)
                for idx, item in enumerate(candidate_items):
                    doorway = active_doorways[idx % len(active_doorways)]
                    gate_cell = doorway.properties.get("gate_cell", doorway.grid_pos)
                    appr_cell = doorway.properties.get("approach_cell", doorway.grid_pos)

                    # Subgoal 1: Approach item so carried payload matches handoff gate orientation
                    dr_handoff = int(np.sign(gate_cell[0] - appr_cell[0]))
                    dc_handoff = int(np.sign(gate_cell[1] - appr_cell[1]))
                    handoff_facing = (
                        (dr_handoff, 0) if abs(dr_handoff) >= abs(dc_handoff) else (0, dc_handoff)
                    )
                    if handoff_facing == (0, 0):
                        handoff_facing = (0, 1)
                    pickup_facing = handoff_facing

                    pickup_stand = (
                        item.grid_pos[0] - pickup_facing[0] * eg.step_size,
                        item.grid_pos[1] - pickup_facing[1] * eg.step_size,
                    )
                    carried_offset = (
                        pickup_facing[0] * eg.step_size,
                        pickup_facing[1] * eg.step_size,
                    )

                    plan.append(
                        SequencePlanStep(
                            target_entity_id=item.id,
                            target_pos=pickup_stand,
                            action_type=SpatialActionIntent.PICKUP,
                            approach_facing=pickup_facing,
                            carried_offset=carried_offset,
                        )
                    )
                    # Subgoal 2: Deliver to doorway approach cell
                    plan.append(
                        SequencePlanStep(
                            target_entity_id=doorway.id,
                            target_pos=appr_cell,
                            action_type=SpatialActionIntent.DROP,
                            approach_facing=pickup_facing,
                            carried_offset=carried_offset,
                        )
                    )

                # If all items are delivered, add a stand-clear step away from the bottleneck
                if not candidate_items:
                    doorway = active_doorways[0]
                    appr_cell = doorway.properties.get("approach_cell", doorway.grid_pos)
                    stand_clear_pos = (
                        appr_cell[0] - pickup_facing[0] * eg.step_size * 3,
                        appr_cell[1] - pickup_facing[1] * eg.step_size * 3,
                    )
                    plan.append(
                        SequencePlanStep(
                            target_entity_id="stand_clear",
                            target_pos=stand_clear_pos,
                            action_type=SpatialActionIntent.NAVIGATE,
                        )
                    )
                return plan

        # Topology 2: Resource-Constrained Multi-Goal Sequence Search
        exits = [
            e for e in eg.entities.values() if e.role in (EntityRole.GOAL, EntityRole.RECEPTACLE)
        ]
        refills = [e for e in eg.entities.values() if e.role == EntityRole.RESOURCE]
        switches = [e for e in eg.entities.values() if e.role == EntityRole.ACTUATOR]

        if exits and (initial_energy is not None or refills):
            exit_ent = exits[0]
            # State-space BFS over entity graph with energy constraints
            seq = self._search_resource_constrained_sequence(
                eg.avatar,
                exit_ent,
                refills,
                switches,
                eg.barriers,
                eg.grid_shape,
                eg.step_size,
                initial_energy=initial_energy or 30,
            )
            if seq:
                return [
                    SequencePlanStep(
                        target_entity_id=ent.id,
                        target_pos=ent.grid_pos,
                        action_type=(
                            SpatialActionIntent.ACTUATE
                            if ent.role
                            in (
                                EntityRole.ACTUATOR,
                                EntityRole.RESOURCE,
                            )
                            else SpatialActionIntent.NAVIGATE
                        ),
                    )
                    for ent in seq
                ]

        # Topology 3: Multi-Entity Receptacle Allocation & Placement
        def is_item_delivered(e: SpatialEntity) -> bool:
            if e.is_delivered:
                return True
            for rec in exits:
                b = rec.bounding_box
                if (b[0] <= e.grid_pos[0] <= b[1] - eg.step_size + 1) and (
                    b[2] <= e.grid_pos[1] <= b[3] - eg.step_size + 1
                ):
                    return True
            if delivered_positions:
                if e.grid_pos in delivered_positions or any(
                    math.hypot(e.grid_pos[0] - dp[0], e.grid_pos[1] - dp[1]) < eg.step_size
                    for dp in delivered_positions
                ):
                    return True
            return False

        items = [
            e
            for e in eg.entities.values()
            if e.role == EntityRole.MANIPULABLE
            and not is_item_delivered(e)
            and (item_colors is None or e.color in item_colors)
        ]
        if (items or is_carrying) and exits:
            exit_ent = max(exits, key=lambda e: (e.role == EntityRole.RECEPTACLE, e.area))
            items.sort(
                key=lambda e: math.hypot(
                    e.grid_pos[0] - eg.avatar.grid_pos[0],
                    e.grid_pos[1] - eg.avatar.grid_pos[1],
                )
            )

            # Compute unoccupied receptacle slots for sequential delivery
            b = exit_ent.bounding_box
            step_s = eg.step_size
            slots = []
            occupied = set()
            if delivered_positions:
                occupied.update(delivered_positions)
            for e in eg.entities.values():
                if e != eg.agent and e.role not in (
                    EntityRole.RECEPTACLE,
                    EntityRole.GOAL,
                ):
                    occupied.add(e.grid_pos)

            snap_min_r = int(math.ceil(b[0] / step_s)) * step_s
            max_valid_r = b[1] - step_s + 1
            snap_min_c = int(math.ceil(b[2] / step_s)) * step_s
            max_valid_c = b[3] - step_s + 1
            for sr in range(snap_min_r, max_valid_r + 1, step_s):
                for sc in range(snap_min_c, max_valid_c + 1, step_s):
                    pos = (sr, sc)
                    if pos not in occupied:
                        slots.append(pos)
            if not slots:
                slots = [exit_ent.grid_pos]

            plan = []
            available_slots = list(slots)

            # If the avatar is already carrying a payload, plan its delivery first
            if is_carrying and carried_offset != (0.0, 0.0):
                best_drop_info = None
                for slot in available_slots:
                    ds = (
                        int(round(slot[0] - carried_offset[0])),
                        int(round(slot[1] - carried_offset[1])),
                    )
                    is_valid_ds = (
                        0 <= ds[0] < eg.grid_shape[0]
                        and 0 <= ds[1] < eg.grid_shape[1]
                        and ds not in eg.barriers
                        and ds not in (delivered_positions or set())
                    )
                    if is_valid_ds:
                        dr = int(np.sign(carried_offset[0]))
                        dc = int(np.sign(carried_offset[1]))
                        best_drop_info = (ds, slot, (dr, dc))
                        break
                if not best_drop_info and available_slots:
                    slot = available_slots[0]
                    ds = (
                        int(round(slot[0] - carried_offset[0])),
                        int(round(slot[1] - carried_offset[1])),
                    )
                    best_drop_info = (
                        ds,
                        slot,
                        (int(np.sign(carried_offset[0])), int(np.sign(carried_offset[1]))),
                    )

                if best_drop_info:
                    ds, slot, facing = best_drop_info
                    plan.append(
                        SequencePlanStep(
                            target_entity_id=exit_ent.id,
                            target_pos=ds,
                            action_type=SpatialActionIntent.DROP,
                            approach_facing=facing,
                            carried_offset=carried_offset,
                        )
                    )
                    if slot in available_slots:
                        available_slots.remove(slot)

                carried_item_pos = (
                    int(round(eg.avatar.grid_pos[0] + carried_offset[0])),
                    int(round(eg.avatar.grid_pos[1] + carried_offset[1])),
                )
                items = [
                    it
                    for it in items
                    if math.hypot(
                        it.grid_pos[0] - carried_item_pos[0], it.grid_pos[1] - carried_item_pos[1]
                    )
                    >= eg.step_size * 0.75
                ]

            for idx, item in enumerate(items):
                # Determine adjacent approach stand position
                adj_cells = [
                    (item.grid_pos[0] - eg.step_size, item.grid_pos[1]),
                    (item.grid_pos[0] + eg.step_size, item.grid_pos[1]),
                    (item.grid_pos[0], item.grid_pos[1] - eg.step_size),
                    (item.grid_pos[0], item.grid_pos[1] + eg.step_size),
                ]
                valid_adj = [
                    p
                    for p in adj_cells
                    if p not in eg.barriers
                    and 0 <= p[0] < eg.grid_shape[0]
                    and 0 <= p[1] < eg.grid_shape[1]
                ]

                # Pick an available slot from the pool
                drop_pos = (
                    available_slots[idx % len(available_slots)]
                    if available_slots
                    else exit_ent.grid_pos
                )
                planned_drops = set(slots)

                # Rigid-body kinematic alignment:
                valid_candidates = []
                for p in valid_adj:
                    dr = int(np.sign(item.grid_pos[0] - p[0]))
                    dc = int(np.sign(item.grid_pos[1] - p[1]))
                    c_off = (float(dr * eg.step_size), float(dc * eg.step_size))
                    ds = (drop_pos[0] - int(c_off[0]), drop_pos[1] - int(c_off[1]))

                    is_valid_ds = (
                        0 <= ds[0] < eg.grid_shape[0]
                        and 0 <= ds[1] < eg.grid_shape[1]
                        and ds not in eg.barriers
                        and ds not in planned_drops
                        and ds not in (delivered_positions or set())
                    )
                    valid_candidates.append((is_valid_ds, p, (dr, dc), c_off, ds))

                valid_candidates.sort(
                    key=lambda x: (
                        not x[0],
                        math.hypot(
                            x[1][0] - eg.avatar.grid_pos[0], x[1][1] - eg.avatar.grid_pos[1]
                        ),
                    )
                )

                if valid_candidates:
                    _, stand_pos, pickup_facing, carried_offset_cand, drop_stand = valid_candidates[
                        0
                    ]
                    drop_facing = pickup_facing
                else:
                    stand_pos = item.grid_pos
                    pickup_facing = (0, 1)
                    carried_offset_cand = (0.0, float(eg.step_size))
                    drop_stand = drop_pos
                    drop_facing = (0, 1)

                plan.append(
                    SequencePlanStep(
                        target_entity_id=item.id,
                        target_pos=stand_pos,
                        action_type=SpatialActionIntent.PICKUP,
                        approach_facing=pickup_facing,
                        carried_offset=carried_offset_cand,
                    )
                )
                plan.append(
                    SequencePlanStep(
                        target_entity_id=exit_ent.id,
                        target_pos=drop_stand,
                        action_type=SpatialActionIntent.DROP,
                        approach_facing=drop_facing,
                        carried_offset=carried_offset_cand,
                    )
                )
            return plan

        nav_goals = [e for e in exits if e.role in (EntityRole.GOAL, EntityRole.RECEPTACLE)]
        if nav_goals and eg.avatar:

            def nav_goal_priority(g: SpatialEntity) -> tuple[bool, float]:
                is_explored = (
                    g.id in explored_ids
                    or g.grid_pos in explored_positions
                    or any(
                        math.hypot(g.grid_pos[0] - p[0], g.grid_pos[1] - p[1]) < eg.step_size * 0.75
                        for p in explored_positions
                    )
                )
                dist = math.hypot(
                    g.grid_pos[0] - eg.avatar.grid_pos[0],
                    g.grid_pos[1] - eg.avatar.grid_pos[1],
                )
                return (is_explored, dist)

            nav_goals.sort(key=nav_goal_priority)
            for g in nav_goals:
                target_pos = g.grid_pos
                plan_action = SpatialActionIntent.NAVIGATE
                if target_pos in eg.barriers:
                    # Inaccessible barrier cell — stand in adjacent non-barrier cell
                    adj_cells = [
                        (target_pos[0] + dr, target_pos[1] + dc)
                        for dr, dc in (
                            (-eg.step_size, 0),
                            (eg.step_size, 0),
                            (0, -eg.step_size),
                            (0, eg.step_size),
                        )
                    ]
                    valid_adj = [
                        p
                        for p in adj_cells
                        if 0 <= p[0] < eg.grid_shape[0]
                        and 0 <= p[1] < eg.grid_shape[1]
                        and p not in eg.barriers
                    ]
                    if valid_adj:
                        valid_adj.sort(
                            key=lambda p: math.hypot(
                                p[0] - eg.avatar.grid_pos[0],
                                p[1] - eg.avatar.grid_pos[1],
                            )
                        )
                        target_pos = valid_adj[0]
                        plan_action = SpatialActionIntent.INTERACT
                    else:
                        continue

                path = self.compute_safe_path(
                    start=eg.avatar.grid_pos,
                    goal=target_pos,
                    barrier_cells=eg.barriers,
                    grid_shape=eg.grid_shape,
                    step_size=eg.step_size,
                )
                if (
                    path
                    and (len(path) > 1 or target_pos == eg.avatar.grid_pos)
                    and math.hypot(path[-1][0] - target_pos[0], path[-1][1] - target_pos[1])
                    <= eg.step_size * 0.95
                ):
                    return [
                        SequencePlanStep(
                            target_entity_id=g.id,
                            target_pos=target_pos,
                            action_type=plan_action,
                        )
                    ]

        # Epistemic candidate exploration: if no reachable goals exist, plan toward
        # nearest reachable interactive actuators/switches or unknown entities.
        # De-prioritize entities explored without success in this or prior attempts.
        candidate_entities = [
            e for e in eg.entities.values() if e != eg.avatar and e.role != EntityRole.AGENT
        ]
        if candidate_entities and eg.avatar:

            def exploration_priority(e: SpatialEntity) -> tuple[int, bool, float]:
                role_prio = 1
                if e.role in (EntityRole.ACTUATOR, EntityRole.PORTAL):
                    role_prio = 0
                elif e.role in (EntityRole.GOAL, EntityRole.RECEPTACLE):
                    role_prio = 0

                is_explored = (
                    e.id in explored_ids
                    or e.grid_pos in explored_positions
                    or any(
                        math.hypot(e.grid_pos[0] - p[0], e.grid_pos[1] - p[1]) < eg.step_size * 0.75
                        for p in explored_positions
                    )
                )
                dist = math.hypot(
                    e.grid_pos[0] - eg.avatar.grid_pos[0],
                    e.grid_pos[1] - eg.avatar.grid_pos[1],
                )
                return (role_prio, is_explored, dist)

            candidate_entities.sort(key=exploration_priority)
            for cand in candidate_entities:
                cand_pos = cand.grid_pos
                cand_action = (
                    SpatialActionIntent.ACTUATE
                    if cand.role == EntityRole.ACTUATOR
                    else SpatialActionIntent.NAVIGATE
                )
                if cand_pos in eg.barriers:
                    adj_cells = [
                        (cand_pos[0] + dr, cand_pos[1] + dc)
                        for dr, dc in (
                            (-eg.step_size, 0),
                            (eg.step_size, 0),
                            (0, -eg.step_size),
                            (0, eg.step_size),
                        )
                    ]
                    valid_adj = [
                        p
                        for p in adj_cells
                        if 0 <= p[0] < eg.grid_shape[0]
                        and 0 <= p[1] < eg.grid_shape[1]
                        and p not in eg.barriers
                    ]
                    if valid_adj:
                        valid_adj.sort(
                            key=lambda p: math.hypot(
                                p[0] - eg.avatar.grid_pos[0],
                                p[1] - eg.avatar.grid_pos[1],
                            )
                        )
                        cand_pos = valid_adj[0]
                        cand_action = SpatialActionIntent.INTERACT
                    else:
                        continue
                elif cand.role in (EntityRole.ACTUATOR, EntityRole.MANIPULABLE):
                    if (
                        math.hypot(
                            cand_pos[0] - eg.avatar.grid_pos[0],
                            cand_pos[1] - eg.avatar.grid_pos[1],
                        )
                        <= eg.step_size * 1.5
                    ):
                        cand_action = SpatialActionIntent.INTERACT

                path = self.compute_safe_path(
                    start=eg.avatar.grid_pos,
                    goal=cand_pos,
                    barrier_cells=eg.barriers,
                    grid_shape=eg.grid_shape,
                    step_size=eg.step_size,
                )
                if (
                    path
                    and len(path) > 1
                    and math.hypot(path[-1][0] - cand_pos[0], path[-1][1] - cand_pos[1])
                    <= eg.step_size * 0.95
                ):
                    return [
                        SequencePlanStep(
                            target_entity_id=cand.id,
                            target_pos=cand_pos,
                            action_type=cand_action,
                        )
                    ]

        # Topological Frontier Exploration: if no entities are reachable, explore
        # the closest reachable unvisited open-space cell to map out unexplored corridors
        if eg.avatar:
            frontier_cell = self._find_nearest_unexplored_frontier(
                start=eg.avatar.grid_pos,
                barrier_cells=eg.barriers,
                grid_shape=eg.grid_shape,
                step_size=eg.step_size,
                visited_cells=explored_positions,
            )
            if frontier_cell is not None:
                path = self.compute_safe_path(
                    start=eg.avatar.grid_pos,
                    goal=frontier_cell,
                    barrier_cells=eg.barriers,
                    grid_shape=eg.grid_shape,
                    step_size=eg.step_size,
                )
                if path and len(path) > 1:
                    return [
                        SequencePlanStep(
                            target_entity_id=f"frontier_{idx}",
                            target_pos=wp,
                            action_type=SpatialActionIntent.NAVIGATE,
                        )
                        for idx, wp in enumerate(path[1:])
                    ]
                return [
                    SequencePlanStep(
                        target_entity_id="frontier",
                        target_pos=frontier_cell,
                        action_type=SpatialActionIntent.NAVIGATE,
                    )
                ]

        return []

    def _find_nearest_unexplored_frontier(
        self,
        start: tuple[int, int],
        barrier_cells: set[tuple[int, int]],
        grid_shape: tuple[int, int],
        step_size: int,
        visited_cells: set[tuple[int, int]],
    ) -> tuple[int, int] | None:
        """Topological frontier search (BFS) for the nearest unvisited open space cell."""
        from collections import deque

        H, W = grid_shape
        step = max(1, step_size)
        queue = deque([start])
        seen = {start}

        while queue:
            cr, cc = queue.popleft()
            for dr, dc in ((-step, 0), (step, 0), (0, -step), (0, step)):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    has_barrier = any(
                        (nr + ro, nc + co) in barrier_cells
                        for ro in range(step)
                        for co in range(step)
                    )
                    if not has_barrier:
                        is_visited = (nr, nc) in visited_cells or any(
                            math.hypot(nr - vr, nc - vc) < step * 0.75 for vr, vc in visited_cells
                        )
                        if not is_visited and (nr, nc) != start:
                            return (nr, nc)
                        queue.append((nr, nc))
        return None

    # ═══════════════════════════════════════════════════════════════════════
    # 3. NATIVE HCIR MEMORY INTEGRATION (EpisodeNode, BeliefNode)
    # ═══════════════════════════════════════════════════════════════════════

    def record_failure(
        self,
        workspace: HCIRWorkspaceState | None,
        session_id: str,
        failed_action: int,
        failure_pos: tuple[int, int],
        reason: str,
        attempted_sequence: list[str] | None = None,
    ) -> None:
        """Records a trial failure into HCIR native memory (EpisodeNode & BeliefNode)."""
        if failure_pos:
            failure_pos = (int(failure_pos[0]), int(failure_pos[1]))

        is_new_barrier = (
            reason in ("collision", "hazard", "trial_failed", "death")
            and failure_pos
            and failure_pos not in self._learned_barriers
        )
        if is_new_barrier or (reason not in ("collision", "hazard", "trial_failed", "death")):
            logger.info(
                f"Recording trial failure in HCIR memory: session={session_id}, "
                f"pos={failure_pos}, reason={reason}"
            )
        else:
            logger.debug(
                f"Barrier {failure_pos} already known in HCIR memory (session={session_id})"
            )

        if attempted_sequence:
            self._failed_sequences.add(tuple(attempted_sequence))

        if reason in ("collision", "hazard", "trial_failed", "death") and failure_pos:
            self._learned_barriers.add(failure_pos)

        if workspace is None:
            return

        # 1. Commit EpisodeNode
        ep_node = EpisodeNode(
            id=f"ep_{session_id}_{uuid.uuid4().hex[:6]}",
            summary=f"Trial failed at {failure_pos}: {reason}",
            outcome=reason,
            reward=-1.0,
            lifecycle=NodeLifecycle.ACTIVE,
            provenance=Provenance(created_by="HCIRSpatialEntityPlanner"),
            scope=Scope(tenant_id="default"),
            properties={
                "failed_action": failed_action,
                "failure_position": failure_pos,
                "reason": reason,
                "sequence": attempted_sequence or [],
            },
            tags=["trial_failure", reason],
        )
        workspace.upsert_node(ep_node)

        # 2. Upsert BeliefNode for constraint retention
        belief_id = (
            f"belief_barrier_{failure_pos[0]}_{failure_pos[1]}"
            if failure_pos
            else f"belief_failure_{reason}"
        )
        belief_node = BeliefNode(
            id=belief_id,
            claim=f"Position {failure_pos} is impassable or causes {reason}",
            statement=f"Position {failure_pos} is impassable or causes {reason}",
            epistemic_confidence=0.95,
            belief_type="causal",
            falsification_status=FalsificationStatus.CORROBORATED,
            properties={
                "position": failure_pos,
                "negative_constraint": True,
                "reason": reason,
            },
            tags=["negative_constraint", reason],
        )
        workspace.upsert_node(belief_node)

        # Link Episode -> Belief
        workspace.add_edge(
            HCIREdge(
                sources=[ep_node.id],
                targets=[belief_node.id],
                edge_type=HCIREdgeType.CAUSES,
            )
        )

    def _recall_memory_constraints(
        self, workspace: HCIRWorkspaceState | None
    ) -> tuple[set[tuple[int, int]], list[dict[str, Any]]]:
        """Retrieves negative constraints from existing HCIR BeliefNodes and EpisodeNodes."""
        impassable_cells = set(self._learned_barriers)
        energy_records: list[dict[str, Any]] = []

        if workspace is None or not hasattr(workspace, "graph"):
            return impassable_cells, energy_records

        for node in workspace.graph.nodes_by_type(HCIRNodeType.BELIEF):
            if isinstance(node, BeliefNode) and node.properties.get("negative_constraint"):
                pos = node.properties.get("position")
                if pos and isinstance(pos, (list, tuple)) and len(pos) == 2:
                    impassable_cells.add((int(pos[0]), int(pos[1])))

        for node in workspace.graph.nodes_by_type(HCIRNodeType.EPISODE):
            if isinstance(node, EpisodeNode) and node.properties.get("reason") == "out_of_energy":
                energy_records.append(node.properties)

        return impassable_cells, energy_records

    # ═══════════════════════════════════════════════════════════════════════
    # 4. LOW-LEVEL GEODESIC SEARCH & ACTION SELECTION
    # ═══════════════════════════════════════════════════════════════════════

    def get_action_for_delta(
        self,
        dr: int,
        dc: int,
        action_models: dict[int, Any],
        condition: str | None = None,
    ) -> int | None:
        """Finds available action whose dynamics produce delta (dr, dc) under given condition."""
        sign_r = int(np.sign(dr))
        sign_c = int(np.sign(dc))

        for act, model in action_models.items():
            if hasattr(model, "get_displacement"):
                m_dr, m_dc = model.get_displacement(condition)
            elif hasattr(model, "delta_r") and hasattr(model, "delta_c"):
                m_dr, m_dc = model.delta_r, model.delta_c
            else:
                continue

            m_r = int(np.sign(m_dr))
            m_c = int(np.sign(m_dc))
            if (sign_r != 0 and m_r == sign_r and m_c == 0) or (
                sign_c != 0 and m_c == sign_c and m_r == 0
            ):
                return int(act) if isinstance(act, str) and act.isdigit() else act

        return None

    def compute_safe_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        barrier_cells: set[tuple[int, int]],
        grid_shape: tuple[int, int],
        step_size: int,
        footprint_offsets: list[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int]]:
        """Computes a collision-free geodesic path accounting for carried object footprint offsets."""
        offsets = footprint_offsets or [(0, 0)]
        return PhysicsPredictor.compute_geodesic_path(
            start=start,
            goal=goal,
            barrier_cells=barrier_cells,
            grid_shape=grid_shape,
            step_size=step_size,
            footprint_offsets=offsets,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # 5. INTERNAL TOPOLOGICAL HELPERS
    # ═══════════════════════════════════════════════════════════════════════

    def _compute_connected_components(
        self,
        barriers: set[tuple[int, int]],
        grid_shape: tuple[int, int],
        step_size: int,
    ) -> dict[int, set[tuple[int, int]]]:
        """Partitions all non-barrier cells into connected reachability components."""
        H, W = grid_shape
        visited: set[tuple[int, int]] = set()
        components: dict[int, set[tuple[int, int]]] = {}
        comp_id = 0
        step = max(1, step_size)

        for r in range(0, H, step):
            for c in range(0, W, step):
                if (r, c) not in barriers and (r, c) not in visited:
                    comp = TopologicalCutSetAnalyzer.get_reachable_component(
                        (r, c), barriers, grid_shape, step
                    )
                    if comp:
                        components[comp_id] = comp
                        visited.update(comp)
                        comp_id += 1

        return components

    def _find_component_for_pos(
        self,
        pos: tuple[int, int],
        components: dict[int, set[tuple[int, int]]],
        step_size: int,
    ) -> int:
        """Finds which component contains pos (or nearest cell)."""
        r, c = pos
        # Exact match
        for cid, comp in components.items():
            if (r, c) in comp:
                return cid

        # Closest match within step tolerance
        best_cid = -1
        min_dist = float("inf")
        for cid, comp in components.items():
            for cr, cc in comp:
                d = math.hypot(cr - r, cc - c)
                if d < min_dist:
                    min_dist = d
                    best_cid = cid
                    if d <= step_size:
                        return cid

        return best_cid if min_dist <= step_size * 2 else -1

    def _search_resource_constrained_sequence(
        self,
        start_ent: SpatialEntity,
        exit_ent: SpatialEntity,
        refills: list[SpatialEntity],
        switches: list[SpatialEntity],
        barriers: set[tuple[int, int]],
        grid_shape: tuple[int, int],
        step_size: int,
        initial_energy: int,
    ) -> list[SpatialEntity] | None:
        """A* search over entity graph with energy replenishment."""
        all_tokens = refills + switches
        # Priority queue: (cost, energy_remaining, current_entity, visited_mask, path)
        queue: list[tuple[float, int, SpatialEntity, int, list[SpatialEntity]]] = []
        heapq.heappush(queue, (0.0, initial_energy, start_ent, 0, []))
        visited: dict[tuple[str, int], int] = {}  # (entity_id, visited_mask) -> max_energy

        while queue:
            cost, energy, curr, mask, path = heapq.heappop(queue)

            # Check if exit is reachable from current
            exit_p = self.compute_safe_path(
                curr.grid_pos, exit_ent.grid_pos, barriers, grid_shape, step_size
            )
            if exit_p and len(exit_p) - 1 <= energy:
                return path + [exit_ent]

            state_key = (curr.id, mask)
            if state_key in visited and visited[state_key] >= energy:
                continue
            visited[state_key] = energy

            for idx, nxt in enumerate(all_tokens):
                if not (mask & (1 << idx)):
                    p = self.compute_safe_path(
                        curr.grid_pos, nxt.grid_pos, barriers, grid_shape, step_size
                    )
                    if p:
                        dist = len(p) - 1
                        if dist <= energy:
                            # Resource restores budget to full, actuator consumes energy
                            nxt_energy = (
                                initial_energy if nxt.role == EntityRole.RESOURCE else energy - dist
                            )
                            nxt_mask = mask | (1 << idx)
                            heapq.heappush(
                                queue,
                                (cost + dist, nxt_energy, nxt, nxt_mask, path + [nxt]),
                            )

        return None

    def find_nearest_unvisited_frontier(
        self,
        start_pos: tuple[int, int],
        known_walkable: set[tuple[int, int]],
        known_barriers: set[tuple[int, int]],
        visited_cells: set[tuple[int, int]],
        grid_shape: tuple[int, int],
        step_size: int,
        unexamined_entities: list[SpatialEntity] | None = None,
    ) -> list[tuple[int, int]] | None:
        """Finds the shortest collision-free geodesic path to the most promising unvisited frontier.

        Prioritizes frontiers near unexamined interactive entities to foster child-like curiosity.
        """
        step = max(1, step_size)
        start_r, start_c = start_pos

        candidates: list[tuple[int, int]] = []
        for r, c in known_walkable:
            if (r, c) not in visited_cells and (r, c) not in known_barriers:
                for dr, dc in [(-step, 0), (step, 0), (0, -step), (0, step)]:
                    adj = (r + dr, c + dc)
                    if adj in visited_cells or adj == start_pos:
                        candidates.append((r, c))
                        break

        if not candidates:
            candidates = [
                c for c in known_walkable if c not in visited_cells and c not in known_barriers
            ]

        if not candidates:
            return None

        scored: list[tuple[float, tuple[int, int]]] = []
        for cr, cc in candidates:
            base_d = math.hypot(cr - start_r, cc - start_c)
            entity_bonus = 0.0
            if unexamined_entities:
                for ent in unexamined_entities:
                    er, ec = ent.grid_pos
                    d_ent = math.hypot(cr - er, cc - ec)
                    if d_ent <= step * 3:
                        entity_bonus += 10.0 / (1.0 + d_ent)
            scored.append((base_d - entity_bonus, (cr, cc)))

        scored.sort(key=lambda item: item[0])

        for _, target in scored[:10]:
            path = self.compute_safe_path(
                start=start_pos,
                goal=target,
                barrier_cells=known_barriers,
                grid_shape=grid_shape,
                step_size=step,
            )
            if path and len(path) > 1:
                return path

        return None

    def plan_temporal_clone_sequence(
        self,
        start_pos: tuple[int, int],
        goal_pos: tuple[int, int],
        actuators: list[SpatialEntity],
        portals: list[SpatialEntity],
        actuator_to_portals: dict[str, set[str]],
        latching_portals: set[str],
        barriers: set[tuple[int, int]],
        grid_shape: tuple[int, int],
        step_size: int,
        action_models: dict[int, Any],
        max_clones: int = 3,
        rewind_action: int = 5,
        current_pos: tuple[int, int] | None = None,
        initial_probe_actions: int = 0,
    ) -> list[TimelinePlanStep] | None:
        """Domain-agnostic multi-timeline temporal clone coordination planner.

        Synthesizes a minimal sequence of switch activations and timeline rewinds
        so that replaying clones hold required portals open for subsequent agents.
        """
        step = max(1, step_size)
        portal_map = {p.id: p for p in portals}
        actuator_map = {a.id: a for a in actuators}

        # Helper to compute effective barriers when a subset of portals are open
        def get_barriers_with_open(open_portal_ids: set[str]) -> set[tuple[int, int]]:
            eff = set(barriers)
            for p_id, p in portal_map.items():
                if p_id not in open_portal_ids:
                    # Portal is closed, add its grid footprint
                    eff.add(p.grid_pos)
                else:
                    # Portal is open, remove its grid footprint
                    eff.discard(p.grid_pos)
            return eff

        def path_reaches(p: list[tuple[int, int]] | None, target: tuple[int, int]) -> bool:
            return bool(
                p
                and len(p) > 0
                and math.hypot(p[-1][0] - target[0], p[-1][1] - target[1]) <= step * 0.9
            )

        def path_to_actions(p: list[tuple[int, int]]) -> list[int]:
            acts = []
            for i in range(len(p) - 1):
                dr = p[i + 1][0] - p[i][0]
                dc = p[i + 1][1] - p[i][1]
                if dr == 0 and dc == 0:
                    continue
                act = self.get_action_for_delta(dr, dc, action_models)
                if act is not None:
                    acts.append(act)
            return acts

        # Search for a sequence of actuators to trigger
        # State: (current_open_portals: frozenset[str], activated_actuators: tuple[str, ...])
        queue: list[tuple[frozenset[str], tuple[str, ...]]] = [(frozenset(), ())]
        visited: set[tuple[frozenset[str], tuple[str, ...]]] = set()

        solution_actuator_seq: tuple[str, ...] | None = None

        while queue:
            open_ports, act_seq = queue.pop(0)

            # Check if goal is reachable with currently open portals
            eff_barriers = get_barriers_with_open(set(open_ports))
            goal_path = self.compute_safe_path(
                start=start_pos,
                goal=goal_pos,
                barrier_cells=eff_barriers,
                grid_shape=grid_shape,
                step_size=step,
            )
            if path_reaches(goal_path, goal_pos):
                solution_actuator_seq = act_seq
                break

            if len(act_seq) >= max_clones:
                continue

            state_key = (open_ports, act_seq)
            if state_key in visited:
                continue
            visited.add(state_key)

            # Try activating any reachable actuator not yet activated
            for a_id, a_ent in actuator_map.items():
                if a_id in act_seq:
                    continue
                t0_start = current_pos if (not act_seq and current_pos is not None) else start_pos
                a_path = self.compute_safe_path(
                    start=t0_start,
                    goal=a_ent.grid_pos,
                    barrier_cells=eff_barriers,
                    grid_shape=grid_shape,
                    step_size=step,
                )
                if path_reaches(a_path, a_ent.grid_pos):
                    # Actuator is reachable!
                    new_open = set(open_ports)
                    opened_by_a = actuator_to_portals.get(a_id, set())
                    new_open.update(opened_by_a)
                    for other_id, other_ent in actuator_map.items():
                        if other_id != a_id and other_ent.grid_pos in a_path:
                            other_opened = actuator_to_portals.get(other_id, set())
                            if other_opened and other_opened.issubset(latching_portals):
                                new_open.update(other_opened)
                    queue.append((frozenset(new_open), act_seq + (a_id,)))

        if solution_actuator_seq is None:
            return None

        # Prune redundant latching actuators if a later timeline traverses them
        pruned_seq = list(solution_actuator_seq)
        for a_id in list(pruned_seq):
            opened = actuator_to_portals.get(a_id, set())
            if opened and opened.issubset(latching_portals):
                a_pos = actuator_map[a_id].grid_pos
                for other_id in list(pruned_seq):
                    if other_id != a_id:
                        eff_b = get_barriers_with_open(set())
                        p = self.compute_safe_path(
                            start_pos, actuator_map[other_id].grid_pos, eff_b, grid_shape, step
                        )
                        if path_reaches(p, actuator_map[other_id].grid_pos) and a_pos in p:
                            pruned_seq.remove(a_id)
                            break
        solution_actuator_seq = tuple(pruned_seq)

        # Reconstruct timeline plans with temporal delay synchronization
        timeline_steps: list[TimelinePlanStep] = []
        accumulated_open: set[str] = set()
        portal_trigger_time: dict[str, int] = {}
        eff_probe = (
            initial_probe_actions
            if initial_probe_actions > 0
            else (1 if (current_pos is not None and current_pos != start_pos) else 0)
        )

        def synchronize_path(raw_path: list[tuple[int, int]]) -> list[tuple[int, int]]:
            if not raw_path:
                return []
            synced = [raw_path[0]]
            step_idx = 0
            for i in range(len(raw_path) - 1):
                curr_cell = synced[-1]
                next_cell = raw_path[i + 1]

                matching_portal_id = None
                for p_ent in portals:
                    if (
                        math.hypot(
                            next_cell[0] - p_ent.grid_pos[0],
                            next_cell[1] - p_ent.grid_pos[1],
                        )
                        <= step * 0.9
                    ):
                        matching_portal_id = p_ent.id
                        break

                if matching_portal_id and matching_portal_id in portal_trigger_time:
                    t_trig = portal_trigger_time[matching_portal_id]
                    if step_idx + 1 <= t_trig:
                        delay = t_trig - (step_idx + 1) + 1
                        num_oscillations = (delay + 1) // 2
                        if len(synced) >= 2:
                            prev_cell = synced[-2]
                        else:
                            prev_cell = curr_cell
                            for dr, dc in [(-step, 0), (step, 0), (0, -step), (0, step)]:
                                cand = (curr_cell[0] + dr, curr_cell[1] + dc)
                                if (
                                    0 <= cand[0] < grid_shape[0]
                                    and 0 <= cand[1] < grid_shape[1]
                                    and cand not in barriers
                                    and cand != next_cell
                                ):
                                    prev_cell = cand
                                    break
                        for _ in range(num_oscillations):
                            synced.append(prev_cell)
                            synced.append(curr_cell)
                            step_idx += 2

                synced.append(next_cell)
                step_idx += 1
            return synced

        for t_idx, a_id in enumerate(solution_actuator_seq):
            eff_barriers = get_barriers_with_open(accumulated_open)
            a_ent = actuator_map[a_id]
            t_start = current_pos if (t_idx == 0 and current_pos is not None) else start_pos
            raw_path = self.compute_safe_path(
                start=t_start,
                goal=a_ent.grid_pos,
                barrier_cells=eff_barriers,
                grid_shape=grid_shape,
                step_size=step,
            )
            if not path_reaches(raw_path, a_ent.grid_pos):
                return None
            path = synchronize_path(raw_path)

            num_clone_moves = len(path) - 1
            if t_idx == 0:
                num_clone_moves += eff_probe

            for p_id in actuator_to_portals.get(a_id, set()):
                portal_trigger_time[p_id] = num_clone_moves

            acts = path_to_actions(path) + [rewind_action]
            timeline_steps.append(
                TimelinePlanStep(
                    timeline_index=t_idx,
                    target_pos=a_ent.grid_pos,
                    path=path,
                    actions=acts,
                    is_rewind=True,
                )
            )
            accumulated_open.update(actuator_to_portals.get(a_id, set()))
            for other_id, other_ent in actuator_map.items():
                if other_ent.grid_pos in path:
                    other_opened = actuator_to_portals.get(other_id, set())
                    if other_opened and other_opened.issubset(latching_portals):
                        accumulated_open.update(other_opened)

        # Final timeline: reach the goal
        final_eff_barriers = get_barriers_with_open(accumulated_open)
        final_raw_path = self.compute_safe_path(
            start=start_pos,
            goal=goal_pos,
            barrier_cells=final_eff_barriers,
            grid_shape=grid_shape,
            step_size=step,
        )
        if not path_reaches(final_raw_path, goal_pos):
            return None

        final_path = synchronize_path(final_raw_path)
        final_acts = path_to_actions(final_path)
        timeline_steps.append(
            TimelinePlanStep(
                timeline_index=len(solution_actuator_seq),
                target_pos=goal_pos,
                path=final_path,
                actions=final_acts,
                is_rewind=False,
            )
        )

        return timeline_steps
