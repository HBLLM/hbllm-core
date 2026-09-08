"""
MineDojo Perception Adapter.

Translates 3D voxel chunks and Minecraft inventory dictionaries into
typed HCIR CognitiveGraph and spatial representations.
"""

from __future__ import annotations

import logging

from hbllm.hcir.graph import (
    CognitiveGraph,
    EntityLifecycle,
    PhysicalEntityNode,
)

from .types import (
    MineDojoObservation,
    MineDojoVoxel,
)

logger = logging.getLogger(__name__)


class MineDojoPerceptionAdapter:
    """
    Ingests 3D voxel chunk observations and maintains CognitiveGraph
    with player coordinates, inventory counts, and nearby harvestable resource blocks.
    """

    def __init__(self, graph: CognitiveGraph | None = None) -> None:
        self.graph = graph if graph is not None else CognitiveGraph()

    def ingest_observation(self, obs: MineDojoObservation) -> CognitiveGraph:
        """Update CognitiveGraph with voxel resources and inventory."""
        px, py, pz = obs.player_pos

        # Update Agent Node
        agent_node = PhysicalEntityNode(
            id="agent",
            entity_name="steve_player",
            entity_type="agent",
            properties={
                "x": px,
                "y": py,
                "z": pz,
                "yaw": obs.player_yaw,
                "pitch": obs.player_pitch,
                "inventory": obs.inventory.to_dict(),
                "equipped": obs.equipped,
                "step_count": obs.step_count,
            },
            entity_lifecycle=EntityLifecycle.TRACKED,
        )
        if self.graph.has_node("agent"):
            ex = self.graph.get_node("agent")
            if isinstance(ex, PhysicalEntityNode):
                ex.properties.update(agent_node.properties)
        else:
            self.graph.add_node(agent_node)

        # Ingest visible wood log and stone blocks from chunk
        for z_idx, layer in enumerate(obs.voxels):
            for y_idx, row in enumerate(layer):
                for x_idx, block_id in enumerate(row):
                    if block_id in (
                        MineDojoVoxel.WOOD_LOG,
                        MineDojoVoxel.STONE,
                        MineDojoVoxel.CRAFTING_TABLE,
                    ):
                        # Global coordinates relative to chunk
                        gx = px - 5 + x_idx
                        gy = py - 5 + y_idx
                        gz = pz - 2 + z_idx
                        node_id = f"block_{MineDojoVoxel(block_id).name.lower()}_{gx}_{gy}_{gz}"
                        node = PhysicalEntityNode(
                            id=node_id,
                            entity_name=MineDojoVoxel(block_id).name.lower(),
                            entity_type="voxel_block",
                            properties={"block_id": block_id, "coords": (gx, gy, gz)},
                            entity_lifecycle=EntityLifecycle.TRACKED,
                        )
                        if not self.graph.has_node(node_id):
                            self.graph.add_node(node)

        return self.graph
