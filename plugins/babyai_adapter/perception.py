"""
BabyAI Perception Adapter — translates MiniGrid partial grid observations into HCIR CognitiveGraph.

Converts ego-centric 7x7x3 grid slices into allocentric persistent PhysicalEntityNodes,
preserving spatial coordinates, attribute bindings (color, shape, state), and topological relations.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.hcir.graph import (
    CognitiveGraph,
    EntityLifecycle,
    HCIREdge,
    HCIREdgeType,
    PhysicalEntityNode,
)
from hbllm.perception import EpistemicSpatialGrid

from .types import (
    DIR_TO_VEC,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    IDX_TO_STATE,
    MiniGridDirection,
    MiniGridObservation,
)

logger = logging.getLogger(__name__)


class BabyAIPerceptionAdapter:
    """Ingests MiniGrid observations and incrementally maintains an allocentric CognitiveGraph."""

    def __init__(self, graph: CognitiveGraph | None = None) -> None:
        self.graph = graph if graph is not None else CognitiveGraph()
        self.agent_id = "agent_primary"
        self._known_entities: dict[tuple[int, int], str] = {}  # (wx, wy) -> entity_id
        self.grid = EpistemicSpatialGrid(self.graph)
        self.initial_agent_pos: tuple[int, int] | None = None
        self.initial_agent_dir: int | None = None

    @property
    def visited_cells(self) -> set[tuple[int, int]]:
        """Allocentric cells stepped on by the agent."""
        return self.grid.visited_cells

    @property
    def seen_cells(self) -> set[tuple[int, int]]:
        """Allocentric cells observed by the agent."""
        return self.grid.seen_cells

    def ingest_observation(
        self,
        observation: MiniGridObservation | dict[str, Any],
        known_agent_pos: tuple[int, int] | None = None,
        known_carrying: Any = ...,
    ) -> CognitiveGraph:
        """Process observation and update CognitiveGraph nodes and edges."""
        if isinstance(observation, dict):
            image = observation.get("image", [])
            direction = observation.get("direction", 0)
            mission = observation.get("mission", "")
            extra = observation.get("extra", {})
            step_count = observation.get("step_count", 0)
            obs = MiniGridObservation(
                image=image,
                direction=direction,
                mission=mission,
                step_count=step_count,
                extra=extra,
            )
        else:
            obs = observation

        # Determine agent position: from extra metadata, argument, or graph
        agent_pos = known_agent_pos or obs.extra.get("agent_pos", (1, 1))
        direction = obs.direction
        if self.initial_agent_pos is None:
            self.initial_agent_pos = agent_pos
            self.initial_agent_dir = direction
        self.grid.mark_visited(agent_pos)

        # In MiniGrid, the agent's carried item is placed at (view_size // 2, view_size - 1)
        # in the partially observable view (e.g. vx=3, vy=6 for a 7x7 grid)
        view_size = len(obs.image)
        carrying_from_obs: dict[str, Any] | None = None
        if view_size > 0:
            agent_vx = view_size // 2
            if agent_vx < len(obs.image):
                agent_vy = len(obs.image[agent_vx]) - 1
                if agent_vy >= 0:
                    car_tuple = obs.image[agent_vx][agent_vy]
                    car_obj = IDX_TO_OBJECT.get(car_tuple[0], "empty")
                    car_col = IDX_TO_COLOR.get(car_tuple[1], "red")
                    if car_obj not in ("unseen", "empty", "floor"):
                        carrying_from_obs = {"type": car_obj, "color": car_col}

        # 1. Update Agent Node and Carrying State
        carrying_info: dict[str, Any] | None = None
        if known_carrying is not ...:
            if known_carrying is None:
                carrying_info = None
            elif hasattr(known_carrying, "type") and hasattr(known_carrying, "color"):
                carrying_info = {
                    "type": str(known_carrying.type),
                    "color": str(known_carrying.color),
                }
            elif isinstance(known_carrying, tuple):
                car_type = IDX_TO_OBJECT.get(known_carrying[0], "unknown")
                car_col = IDX_TO_COLOR.get(known_carrying[1], "unknown")
                carrying_info = {"type": car_type, "color": car_col}
            elif isinstance(known_carrying, dict):
                carrying_info = known_carrying
        elif "carrying" in obs.extra and obs.extra["carrying"] is not None:
            carrying_tuple = obs.extra.get("carrying")
            if carrying_tuple:
                car_type = IDX_TO_OBJECT.get(carrying_tuple[0], "unknown")
                car_col = IDX_TO_COLOR.get(carrying_tuple[1], "unknown")
                carrying_info = {"type": car_type, "color": car_col}
        elif carrying_from_obs is not None:
            carrying_info = carrying_from_obs
        elif "carrying" in obs.extra and obs.extra["carrying"] is None:
            carrying_info = None
        elif self.graph.has_node(self.agent_id):
            # The observation's carried cell is authoritative
            carrying_info = carrying_from_obs

        agent_node = PhysicalEntityNode(
            id=self.agent_id,
            entity_name="BabyAIAgent",
            entity_type="agent",
            properties={
                "coords": agent_pos,
                "direction": direction,
                "direction_name": MiniGridDirection(direction).name.lower(),
                "initial_pos": self.initial_agent_pos,
                "initial_dir": self.initial_agent_dir,
                "carrying": carrying_info,
                "visited_cells": set(self.visited_cells),
                "seen_cells": set(self.seen_cells),
            },
            entity_lifecycle=EntityLifecycle.TRACKED,
        )
        if self.graph.has_node(self.agent_id):
            existing = self.graph.get_node(self.agent_id)
            if isinstance(existing, PhysicalEntityNode):
                existing.properties.update(agent_node.properties)
        else:
            self.graph.add_node(agent_node)

        # 2. Extract Visible Entities from 7x7 Egocentric View
        fwd_vec = DIR_TO_VEC[MiniGridDirection(direction)]
        right_vec = (-fwd_vec[1], fwd_vec[0])

        current_view_cells: set[tuple[int, int]] = set()

        for vx in range(view_size):
            for vy in range(len(obs.image[vx])):
                fwd_dist = 6 - vy
                right_dist = vx - 3

                # Skip the agent's own position (it represents carried item, not a world grid cell)
                if fwd_dist == 0 and right_dist == 0:
                    continue

                cell_tuple = obs.image[vx][vy]
                obj_idx, col_idx, state_idx = cell_tuple[0], cell_tuple[1], cell_tuple[2]

                wx = agent_pos[0] + fwd_dist * fwd_vec[0] + right_dist * right_vec[0]
                wy = agent_pos[1] + fwd_dist * fwd_vec[1] + right_dist * right_vec[1]

                current_view_cells.add((wx, wy))

                obj_type = IDX_TO_OBJECT.get(obj_idx, "unseen")
                color = IDX_TO_COLOR.get(col_idx, "red")
                state = IDX_TO_STATE.get(state_idx, "open")

                if obj_type != "unseen":
                    self.grid.mark_seen([(wx, wy)])

                if obj_type in ("unseen", "empty", "floor"):
                    # If this cell previously had an object, and now it's empty, mark forgotten
                    if (wx, wy) in self._known_entities:
                        old_eid = self._known_entities.pop((wx, wy))
                        if self.graph.has_node(old_eid):
                            old_node = self.graph.get_node(old_eid)
                            if isinstance(old_node, PhysicalEntityNode):
                                old_node.entity_lifecycle = EntityLifecycle.FORGOTTEN
                    continue

                is_pickupable = obj_type in ("ball", "box", "key")
                is_door = obj_type == "door"
                is_passable = obj_type in ("empty", "floor") or (is_door and state == "open")

                # Entity unique persistent identifier for this spatial instance
                entity_id = f"ent_{obj_type}_{color}_{wx}_{wy}"
                self._known_entities[(wx, wy)] = entity_id

                is_from_init = agent_pos == self.initial_agent_pos

                node = PhysicalEntityNode(
                    id=entity_id,
                    entity_name=f"{color}_{obj_type}",
                    entity_type=obj_type,
                    properties={
                        "color": color,
                        "state": state,
                        "coords": (wx, wy),
                        "passable": is_passable,
                        "pickupable": is_pickupable,
                        "is_door": is_door,
                        "seen_from_initial_pos": is_from_init,
                        "last_observed_step": obs.step_count,
                    },
                    entity_lifecycle=EntityLifecycle.TRACKED,
                )

                if self.graph.has_node(entity_id):
                    ex_node = self.graph.get_node(entity_id)
                    if isinstance(ex_node, PhysicalEntityNode):
                        ex_node.properties.update(node.properties)
                        if is_from_init or ex_node.properties.get("seen_from_initial_pos"):
                            ex_node.properties["seen_from_initial_pos"] = True
                        ex_node.entity_lifecycle = EntityLifecycle.TRACKED
                else:
                    self.graph.add_node(node)

                # Add spatial edge to Agent if directly in front
                front_pos = (agent_pos[0] + fwd_vec[0], agent_pos[1] + fwd_vec[1])
                if (wx, wy) == front_pos:
                    edge_id = f"edge_in_front_{entity_id}"
                    if not self.graph.has_edge(edge_id):
                        self.graph.add_edge(
                            HCIREdge(
                                id=edge_id,
                                edge_type=HCIREdgeType.NEAR,
                                sources=[entity_id],
                                targets=[self.agent_id],
                            )
                        )

        # 3. Transition entities outside current field of view to OCCLUDED (persistent memory)
        for (ex, ey), eid in self._known_entities.items():
            if (ex, ey) not in current_view_cells:
                if self.graph.has_node(eid):
                    n = self.graph.get_node(eid)
                    if (
                        isinstance(n, PhysicalEntityNode)
                        and n.entity_lifecycle == EntityLifecycle.TRACKED
                    ):
                        n.entity_lifecycle = EntityLifecycle.OCCLUDED

        # 4. Maintain pairwise NEAR edges between tracked/occluded physical entities
        active_items = []
        for pos, eid in self._known_entities.items():
            if self.graph.has_node(eid):
                node = self.graph.get_node(eid)
                if isinstance(node, PhysicalEntityNode) and node.entity_lifecycle in (
                    EntityLifecycle.TRACKED,
                    EntityLifecycle.OCCLUDED,
                ):
                    active_items.append((eid, node.properties.get("coords")))

        for i in range(len(active_items)):
            eid1, c1 = active_items[i]
            if not c1:
                continue
            for j in range(i + 1, len(active_items)):
                eid2, c2 = active_items[j]
                if not c2:
                    continue
                d = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])
                pair_id = f"edge_near_{min(eid1, eid2)}_{max(eid1, eid2)}"
                if d == 1:
                    if not self.graph.has_edge(pair_id):
                        self.graph.add_edge(
                            HCIREdge(
                                id=pair_id,
                                edge_type=HCIREdgeType.NEAR,
                                sources=[eid1],
                                targets=[eid2],
                            )
                        )
                else:
                    if self.graph.has_edge(pair_id):
                        self.graph.remove_edge(pair_id)

        return self.graph

    def get_known_entities(self) -> list[PhysicalEntityNode]:
        """Return all active physical entities currently tracked or occluded in the CognitiveGraph."""
        entities: list[PhysicalEntityNode] = []
        for node in self.graph.all_nodes():
            if (
                isinstance(node, PhysicalEntityNode)
                and node.entity_type != "agent"
                and node.entity_lifecycle
                in (
                    EntityLifecycle.TRACKED,
                    EntityLifecycle.OCCLUDED,
                    EntityLifecycle.DISCOVERED,
                )
            ):
                entities.append(node)
        return entities
