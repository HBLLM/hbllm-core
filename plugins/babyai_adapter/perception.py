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

    def ingest_observation(
        self,
        observation: MiniGridObservation | dict[str, Any],
        known_agent_pos: tuple[int, int] | None = None,
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
        carrying_tuple = obs.extra.get("carrying")

        # 1. Update Agent Node
        carrying_info = None
        if carrying_tuple:
            car_type = IDX_TO_OBJECT.get(carrying_tuple[0], "unknown")
            car_col = IDX_TO_COLOR.get(carrying_tuple[1], "unknown")
            carrying_info = {"type": car_type, "color": car_col}

        agent_node = PhysicalEntityNode(
            id=self.agent_id,
            entity_name="BabyAIAgent",
            entity_type="agent",
            properties={
                "coords": agent_pos,
                "direction": direction,
                "direction_name": MiniGridDirection(direction).name.lower(),
                "carrying": carrying_info,
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
        view_size = len(obs.image)
        fwd_vec = DIR_TO_VEC[MiniGridDirection(direction)]
        right_vec = (-fwd_vec[1], fwd_vec[0])

        current_view_cells: set[tuple[int, int]] = set()

        for vx in range(view_size):
            for vy in range(len(obs.image[vx])):
                cell_tuple = obs.image[vx][vy]
                obj_idx, col_idx, state_idx = cell_tuple[0], cell_tuple[1], cell_tuple[2]

                fwd_dist = 6 - vy
                right_dist = vx - 3

                wx = agent_pos[0] + fwd_dist * fwd_vec[0] + right_dist * right_vec[0]
                wy = agent_pos[1] + fwd_dist * fwd_vec[1] + right_dist * right_vec[1]

                current_view_cells.add((wx, wy))

                obj_type = IDX_TO_OBJECT.get(obj_idx, "unseen")
                color = IDX_TO_COLOR.get(col_idx, "red")
                state = IDX_TO_STATE.get(state_idx, "open")

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
                is_passable = obj_type in ("empty", "floor") or (
                    obj_type == "door" and state == "open"
                )

                # Entity unique persistent identifier for this spatial instance
                entity_id = f"ent_{obj_type}_{color}_{wx}_{wy}"
                self._known_entities[(wx, wy)] = entity_id

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
                        "last_observed_step": obs.step_count,
                    },
                    entity_lifecycle=EntityLifecycle.TRACKED,
                )

                if self.graph.has_node(entity_id):
                    ex_node = self.graph.get_node(entity_id)
                    if isinstance(ex_node, PhysicalEntityNode):
                        ex_node.properties.update(node.properties)
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
