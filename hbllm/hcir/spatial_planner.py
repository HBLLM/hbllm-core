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

    # Backward compatibility aliases
    AVATAR = "agent"
    BARRIER = "obstacle"
    MOVABLE_ITEM = "manipulable"
    DOORWAY = "portal"
    SWITCH = "actuator"
    REFILL = "resource"
    EXIT = "goal"


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
    properties: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        id: str,
        role: EntityRole,
        centroid: tuple[float, float],
        grid_pos: tuple[int, int],
        area: int,
        bounding_box: tuple[int, int, int, int],
        color: int | None = None,
        component_id: int = -1,
        is_deliverable: bool = False,
        is_delivered: bool = False,
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
        self.properties = dict(properties or {})
        if color is not None:
            self.properties["visual_id"] = color
        self.properties.update(kwargs)

    @property
    def color(self) -> int:
        """Backward-compatible access to visual property."""
        return self.properties.get("visual_id", 0)

    @color.setter
    def color(self, val: int) -> None:
        self.properties["visual_id"] = val


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
    action_type: str  # "MOVE", "PICKUP", "DROP", "ACTIVATE"
    approach_facing: tuple[int, int] | None = None
    carried_offset: tuple[int, int] | tuple[float, float] = (0, 0)
    expected_energy_cost: int = 0


class HCIRSpatialEntityPlanner:
    """Universal cognitive spatial planning and constraint induction engine."""

    def __init__(self, step_size: int = 1) -> None:
        self.step_size: int = step_size
        self._learned_barriers: set[tuple[int, int]] = set()
        self._learned_switch_rules: dict[str, Any] = {}
        self._failed_sequences: set[tuple[str, ...]] = set()
        self._known_doorways: dict[tuple[int, int], tuple[int, int]] = {}

    def reset(self) -> None:
        """Resets episode-specific spatial memory caches."""
        self._known_doorways.clear()
        self._failed_sequences.clear()

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
            if ent.role in (EntityRole.OBSTACLE, EntityRole.BARRIER):
                continue
            co_located = [
                ex_id
                for ex_id, ex in deduped_entities.items()
                if ex.grid_pos == ent.grid_pos
                and (
                    ex.role == ent.role
                    or ex.role in (EntityRole.MANIPULABLE, EntityRole.MOVABLE_ITEM)
                )
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
            if ent.role in (EntityRole.AGENT, EntityRole.AVATAR):
                eg.agent = ent
                break

        # Physical transit barriers include physical items that occupy space and block transit
        transit_barriers = set(eg.barriers)
        movable_barriers = set()
        for ent in eg.entities.values():
            if ent.role in (EntityRole.MANIPULABLE, EntityRole.MOVABLE_ITEM) and ent != eg.agent:
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
                if e.role in (EntityRole.RECEPTACLE, EntityRole.GOAL, EntityRole.EXIT)
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
                            if ent.role in (EntityRole.MANIPULABLE, EntityRole.MOVABLE_ITEM):
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
    ) -> list[SequencePlanStep]:
        """Plans the sequence of entity interactions from first principles using state-space A*."""
        if not eg.avatar:
            return []

        # Recall active constraints from native HCIR memory
        impassable_gates, energy_records = self._recall_memory_constraints(workspace)

        avatar_comp = eg.avatar.component_id
        is_partitioned = len(eg.cut_sets) > 0

        # Topology 1: Topologically Partitioned Transit & Bottleneck Portal Delivery
        if is_partitioned:
            doorways = [
                e for e in eg.entities.values() if e.role in (EntityRole.PORTAL, EntityRole.DOORWAY)
            ]
            if doorways:
                doorway_positions = {d.properties.get("gate_cell", d.grid_pos) for d in doorways}
                occupied_gates = set()
                for ent in eg.entities.values():
                    if (
                        ent.role in (EntityRole.MANIPULABLE, EntityRole.MOVABLE_ITEM)
                        and ent != eg.agent
                    ):
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
                active_doorways = unoccupied_doorways if unoccupied_doorways else doorways

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
                            action_type="DROP",
                            approach_facing=handoff_facing,
                            carried_offset=carried_offset,
                        )
                    ]

                # Collect uncarried, undelivered items in agent's component (excluding items already at gates)
                candidate_items = [
                    e
                    for e in eg.entities.values()
                    if e.role in (EntityRole.MANIPULABLE, EntityRole.MOVABLE_ITEM)
                    and e.component_id == avatar_comp
                    and not e.is_delivered
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
                            action_type="PICKUP",
                            approach_facing=pickup_facing,
                            carried_offset=carried_offset,
                        )
                    )
                    # Subgoal 2: Deliver to doorway approach cell
                    plan.append(
                        SequencePlanStep(
                            target_entity_id=doorway.id,
                            target_pos=appr_cell,
                            action_type="DROP",
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
                            action_type="MOVE",
                        )
                    )
                return plan

        # Topology 2: Resource-Constrained Multi-Goal Sequence Search
        exits = [
            e
            for e in eg.entities.values()
            if e.role in (EntityRole.GOAL, EntityRole.EXIT, EntityRole.RECEPTACLE)
        ]
        refills = [
            e for e in eg.entities.values() if e.role in (EntityRole.RESOURCE, EntityRole.REFILL)
        ]
        switches = [
            e for e in eg.entities.values() if e.role in (EntityRole.ACTUATOR, EntityRole.SWITCH)
        ]

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
                            "ACTIVATE"
                            if ent.role
                            in (
                                EntityRole.ACTUATOR,
                                EntityRole.SWITCH,
                                EntityRole.RESOURCE,
                                EntityRole.REFILL,
                            )
                            else "MOVE"
                        ),
                    )
                    for ent in seq
                ]

        # Topology 3: Multi-Entity Receptacle Allocation & Placement
        def is_item_delivered(e: SpatialEntity) -> bool:
            if e.is_delivered:
                return True
            if delivered_positions:
                if e.grid_pos in delivered_positions or any(
                    math.hypot(e.grid_pos[0] - dp[0], e.grid_pos[1] - dp[1]) < eg.step_size
                    for dp in delivered_positions
                ):
                    return True
            for rec in exits:
                b = rec.bounding_box
                if (b[0] - 1 <= e.grid_pos[0] <= b[1] + 1) and (
                    b[2] - 1 <= e.grid_pos[1] <= b[3] + 1
                ):
                    return True
            return False

        items = [
            e
            for e in eg.entities.values()
            if e.role in (EntityRole.MANIPULABLE, EntityRole.MOVABLE_ITEM)
            and not is_item_delivered(e)
        ]
        if items and exits:
            exit_ent = exits[0]
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
            occupied = set(eg.barriers)
            if delivered_positions:
                occupied.update(delivered_positions)
            for e in eg.entities.values():
                if e != eg.agent and e.role not in (
                    EntityRole.RECEPTACLE,
                    EntityRole.GOAL,
                    EntityRole.EXIT,
                ):
                    occupied.add(e.grid_pos)

            snap_min_r = int(round(b[0] / step_s)) * step_s
            snap_max_r = int(round(b[1] / step_s)) * step_s
            snap_min_c = int(round(b[2] / step_s)) * step_s
            snap_max_c = int(round(b[3] / step_s)) * step_s
            for sr in range(snap_min_r, snap_max_r + 1, step_s):
                for sc in range(snap_min_c, snap_max_c + 1, step_s):
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
                            action_type="DROP",
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
                        action_type="PICKUP",
                        approach_facing=pickup_facing,
                        carried_offset=carried_offset_cand,
                    )
                )
                plan.append(
                    SequencePlanStep(
                        target_entity_id=exit_ent.id,
                        target_pos=drop_stand,
                        action_type="DROP",
                        approach_facing=drop_facing,
                        carried_offset=carried_offset_cand,
                    )
                )
            return plan

        nav_goals = [e for e in exits if e.role in (EntityRole.GOAL, EntityRole.EXIT)]
        if nav_goals:
            return [
                SequencePlanStep(
                    target_entity_id=nav_goals[0].id,
                    target_pos=nav_goals[0].grid_pos,
                    action_type="MOVE",
                )
            ]

        return []

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
        logger.info(
            f"Recording trial failure in HCIR memory: session={session_id}, "
            f"pos={failure_pos}, reason={reason}"
        )

        if attempted_sequence:
            self._failed_sequences.add(tuple(attempted_sequence))

        if reason == "collision" and failure_pos:
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

    def get_action_for_delta(self, dr: int, dc: int, action_models: dict[int, Any]) -> int | None:
        """Finds available action whose dynamics produce delta (dr, dc)."""
        sign_r = int(np.sign(dr))
        sign_c = int(np.sign(dc))

        for act, model in action_models.items():
            if not hasattr(model, "delta_r") or not hasattr(model, "delta_c"):
                continue
            m_r = int(np.sign(model.delta_r))
            m_c = int(np.sign(model.delta_c))
            if (sign_r != 0 and m_r == sign_r and m_c == 0) or (
                sign_c != 0 and m_c == sign_c and m_r == 0
            ):
                return act

        # Fallback to standard cardinal mapping
        if dr < 0:
            return 1
        if dr > 0:
            return 2
        if dc < 0:
            return 3
        if dc > 0:
            return 4
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
                                initial_energy
                                if nxt.role in (EntityRole.RESOURCE, EntityRole.REFILL)
                                else energy - dist
                            )
                            nxt_mask = mask | (1 << idx)
                            heapq.heappush(
                                queue,
                                (cost + dist, nxt_energy, nxt, nxt_mask, path + [nxt]),
                            )

        return None
