"""
Crafter Perception Adapter.

Projects Crafter 2D semantic grid, vitals, and inventory into
HBLLM CognitiveGraph and EpistemicSpatialGrid.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.hcir.graph import (
    CognitiveGraph,
    EntityLifecycle,
    GoalNode,
    PhysicalEntityNode,
)
from hbllm.perception import EpistemicSpatialGrid

from .types import (
    CrafterAchievement,
    CrafterGoal,
    CrafterInventory,
    CrafterObject,
    CrafterObservation,
    CrafterVitals,
)

logger = logging.getLogger(__name__)


class CrafterPerceptionAdapter:
    """
    Translates raw Crafter observations into typed HCIR representations:
    - EpistemicSpatialGrid for 2D spatial pathfinding and obstacle detection.
    - CognitiveGraph for vital tracking, inventory state, and resource entities.
    """

    CRAFTER_NATIVE_TO_OBJECT = {
        0: int(CrafterObject.EMPTY),
        1: int(CrafterObject.WATER),
        2: int(CrafterObject.GRASS),
        3: int(CrafterObject.STONE),
        4: int(CrafterObject.PATH),
        5: int(CrafterObject.SAND),
        6: int(CrafterObject.TREE),
        7: int(CrafterObject.LAVA),
        8: int(CrafterObject.COAL),
        9: int(CrafterObject.IRON),
        10: int(CrafterObject.DIAMOND),
        11: int(CrafterObject.CRAFTING_TABLE),
        12: int(CrafterObject.FURNACE),
        13: int(CrafterObject.PLAYER),
        14: int(CrafterObject.COW),
        15: int(CrafterObject.ZOMBIE),
        16: int(CrafterObject.SKELETON),
        17: int(CrafterObject.ARROW),
        18: int(CrafterObject.PLANT),
    }

    def __init__(
        self,
        graph: CognitiveGraph | None = None,
        width: int = 64,
        height: int = 64,
    ) -> None:
        self.graph = graph if graph is not None else CognitiveGraph()
        self.grid = EpistemicSpatialGrid(self.graph)
        self.width = width
        self.height = height

    def ingest_observation(
        self, observation: CrafterObservation | dict[str, Any]
    ) -> CognitiveGraph:
        """Update spatial grid and cognitive graph from observation."""
        if isinstance(observation, dict):
            sem = observation.get("semantic_grid")
            if sem is None and "semantic" in observation:
                raw_sem = observation["semantic"]
                if hasattr(raw_sem, "T"):
                    raw_sem = raw_sem.T
                raw_list = raw_sem.tolist() if hasattr(raw_sem, "tolist") else list(raw_sem)
                sem = [
                    [
                        self.CRAFTER_NATIVE_TO_OBJECT.get(int(cell), int(CrafterObject.EMPTY))
                        for cell in row
                    ]
                    for row in raw_list
                ]
            elif sem is None:
                sem = []

            raw_pos = observation.get("player_pos", (32, 32))
            player_pos = (int(raw_pos[0]), int(raw_pos[1]))
            raw_facing = observation.get("player_facing", (0, 1))
            player_facing = (int(raw_facing[0]), int(raw_facing[1]))

            raw_inv = observation.get("inventory", {})
            inv_dict = {}
            for k in [
                "wood",
                "stone",
                "coal",
                "iron",
                "diamond",
                "sapling",
                "wood_pickaxe",
                "stone_pickaxe",
                "iron_pickaxe",
                "wood_sword",
                "stone_sword",
                "iron_sword",
            ]:
                if k in raw_inv:
                    inv_dict[k] = int(raw_inv[k])
            inventory = CrafterInventory(**inv_dict)

            vitals_dict = {"health": 9, "food": 9, "drink": 9, "energy": 9}
            raw_vitals = observation.get("vitals", {})
            for v in ["health", "food", "drink", "energy"]:
                if v in raw_vitals:
                    vitals_dict[v] = int(raw_vitals[v])
                elif v in raw_inv:
                    vitals_dict[v] = int(raw_inv[v])
            vitals = CrafterVitals(**vitals_dict)

            raw_achs = observation.get("achievements", set())
            achs = set()
            if isinstance(raw_achs, dict):
                for k, count in raw_achs.items():
                    if count > 0:
                        try:
                            achs.add(CrafterAchievement(k))
                        except ValueError:
                            pass
            elif isinstance(raw_achs, (set, list, tuple)):
                for k in raw_achs:
                    try:
                        achs.add(CrafterAchievement(k))
                    except ValueError:
                        pass

            obs = CrafterObservation(
                semantic_grid=sem,
                player_pos=player_pos,
                player_facing=player_facing,
                inventory=inventory,
                vitals=vitals,
                achievements=achs,
                step_count=int(observation.get("step_count", 0)),
                day_time=float(observation.get("day_time", 0.0)),
                raw_obs=observation.get("raw_obs"),
                info=observation.get("info", {}),
            )
        else:
            obs = observation

        px, py = obs.player_pos

        # Update Agent Node
        agent_node = PhysicalEntityNode(
            id="agent",
            entity_name="player",
            entity_type="agent",
            properties={
                "x": px,
                "y": py,
                "facing": obs.player_facing,
                "health": obs.vitals.health,
                "food": obs.vitals.food,
                "drink": obs.vitals.drink,
                "energy": obs.vitals.energy,
                "inventory": obs.inventory.to_dict(),
                "achievements": [a.value for a in obs.achievements],
                "step_count": obs.step_count,
            },
            entity_lifecycle=EntityLifecycle.TRACKED,
        )
        if self.graph.has_node("agent"):
            existing = self.graph.get_node("agent")
            if isinstance(existing, PhysicalEntityNode):
                existing.properties.update(agent_node.properties)
        else:
            self.graph.add_node(agent_node)

        # Ingest nearby objects into CognitiveGraph
        if not obs.semantic_grid or not obs.semantic_grid[0]:
            return self.graph

        # Clean up previous resource entities so graph remains bounded
        stale_ids = [
            node.id
            for node in self.graph.all_nodes()
            if node.id != "agent" and not node.id.startswith("goal_")
        ]
        for nid in stale_ids:
            self.graph.remove_node(nid)

        height = len(obs.semantic_grid)
        width = len(obs.semantic_grid[0])
        search_radius = 16

        objects_by_type: dict[int, list[tuple[int, int, int]]] = {}
        target_types = (
            CrafterObject.TREE,
            CrafterObject.WATER,
            CrafterObject.STONE,
            CrafterObject.COAL,
            CrafterObject.IRON,
            CrafterObject.DIAMOND,
            CrafterObject.CRAFTING_TABLE,
            CrafterObject.FURNACE,
            CrafterObject.COW,
            CrafterObject.ZOMBIE,
            CrafterObject.SKELETON,
        )

        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                x, y = px + dx, py + dy
                if 0 <= x < width and 0 <= y < height:
                    obj_id = obs.semantic_grid[y][x]
                    if obj_id in target_types:
                        dist = abs(px - x) + abs(py - y)
                        if obj_id not in objects_by_type:
                            objects_by_type[obj_id] = []
                        objects_by_type[obj_id].append((dist, x, y))

        for obj_id, locs in objects_by_type.items():
            locs.sort(key=lambda item: item[0])
            for dist, x, y in locs[:3]:
                node_id = f"ent_{CrafterObject(obj_id).name.lower()}_{x}_{y}"
                node = PhysicalEntityNode(
                    id=node_id,
                    entity_name=CrafterObject(obj_id).name.lower(),
                    entity_type="resource" if obj_id < 14 else "entity",
                    properties={
                        "obj_type": obj_id,
                        "x": x,
                        "y": y,
                        "distance": dist,
                        "coords": (x, y),
                        "passable": False,
                    },
                    entity_lifecycle=EntityLifecycle.TRACKED,
                )
                self.graph.add_node(node)

        return self.graph

    def find_nearest_object(
        self, obs: CrafterObservation, target_type: CrafterObject
    ) -> tuple[int, int] | None:
        """Find the coordinates of the nearest instance of target_type."""
        px, py = obs.player_pos
        height = len(obs.semantic_grid)
        width = len(obs.semantic_grid[0])
        best_pos = None
        best_dist = float("inf")

        radius = 24
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = px + dx, py + dy
                if 0 <= x < width and 0 <= y < height:
                    if obs.semantic_grid[y][x] == target_type:
                        dist = abs(px - x) + abs(py - y)
                        if dist < best_dist:
                            best_dist = dist
                            best_pos = (x, y)

        return best_pos

    def ingest_goal(self, goal: CrafterGoal | None, obs: CrafterObservation) -> GoalNode:
        """Translate CrafterGoal (or progressive tech-tree roadmap) into an active GoalNode."""
        target_conditions: list[str] = []

        # 1. Vital Survival Interrupts
        if obs.vitals.energy <= 2:
            target_conditions.append("vitals_safe(energy, 9)")
            ach = CrafterAchievement.WAKE_UP
        elif obs.vitals.drink <= 4:
            target_conditions.append("vitals_safe(drink, 5)")
            ach = CrafterAchievement.COLLECT_DRINK
        elif obs.vitals.food <= 4:
            target_conditions.append("vitals_safe(food, 5)")
            ach = CrafterAchievement.EAT_COW
        else:
            ach = goal.target_achievement if goal else None
            if ach is None:
                # Progressive roadmap
                achs = obs.achievements
                inv = obs.inventory
                if CrafterAchievement.COLLECT_WOOD not in achs or inv.wood < 2:
                    ach = CrafterAchievement.COLLECT_WOOD
                elif CrafterAchievement.PLACE_TABLE not in achs:
                    ach = CrafterAchievement.PLACE_TABLE
                elif CrafterAchievement.MAKE_WOOD_PICKAXE not in achs and inv.wood_pickaxe == 0:
                    ach = CrafterAchievement.MAKE_WOOD_PICKAXE
                elif CrafterAchievement.COLLECT_STONE not in achs or inv.stone < 1:
                    ach = CrafterAchievement.COLLECT_STONE
                elif CrafterAchievement.MAKE_STONE_PICKAXE not in achs and inv.stone_pickaxe == 0:
                    ach = CrafterAchievement.MAKE_STONE_PICKAXE
                elif CrafterAchievement.COLLECT_COAL not in achs or inv.coal < 1:
                    ach = CrafterAchievement.COLLECT_COAL
                elif CrafterAchievement.COLLECT_IRON not in achs or inv.iron < 1:
                    ach = CrafterAchievement.COLLECT_IRON
                elif CrafterAchievement.PLACE_FURNACE not in achs and inv.stone >= 4:
                    ach = CrafterAchievement.PLACE_FURNACE
                elif CrafterAchievement.MAKE_IRON_PICKAXE not in achs and inv.iron_pickaxe == 0:
                    ach = CrafterAchievement.MAKE_IRON_PICKAXE
                elif CrafterAchievement.COLLECT_DIAMOND not in achs:
                    ach = CrafterAchievement.COLLECT_DIAMOND
                else:
                    ach = CrafterAchievement.SURVIVE

            cond_map = {
                CrafterAchievement.COLLECT_WOOD: ["has(wood, 1)"],
                CrafterAchievement.PLACE_TABLE: ["has(table)"],
                CrafterAchievement.MAKE_WOOD_PICKAXE: ["has(wood_pickaxe)"],
                CrafterAchievement.COLLECT_STONE: ["has(stone, 1)"],
                CrafterAchievement.MAKE_STONE_PICKAXE: ["has(stone_pickaxe)"],
                CrafterAchievement.COLLECT_COAL: ["has(coal, 1)"],
                CrafterAchievement.COLLECT_IRON: ["has(iron, 1)"],
                CrafterAchievement.PLACE_FURNACE: ["has(furnace)"],
                CrafterAchievement.MAKE_IRON_PICKAXE: ["has(iron_pickaxe)"],
                CrafterAchievement.COLLECT_DIAMOND: ["has(diamond)"],
                CrafterAchievement.COLLECT_DRINK: ["vitals_safe(drink, 5)"],
                CrafterAchievement.EAT_COW: ["vitals_safe(food, 5)"],
                CrafterAchievement.SURVIVE: ["vitals_safe(health, 5)"],
            }
            target_conditions.extend(cond_map.get(ach, ["vitals_safe(health, 5)"]))

        goal_node = GoalNode(
            id="goal_active",
            properties={
                "target_conditions": target_conditions,
                "target_achievement": ach.value if ach else None,
            },
        )
        if self.graph.has_node("goal_active"):
            existing = self.graph.get_node("goal_active")
            if isinstance(existing, GoalNode):
                existing.properties.update(goal_node.properties)
        else:
            self.graph.add_node(goal_node)
        return goal_node
