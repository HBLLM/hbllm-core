"""
Crafter Action and Causal Planning Adapter.

Implements CrafterCausalPlanner which combines:
1. Urgent survival vital interrupts (energy, drink, food).
2. Recursive tech-tree causal recipe DAG (wood -> table -> pickaxe -> stone -> furnace -> iron -> diamond).
3. 2D grid BFS pathfinding and target interaction.
"""

from __future__ import annotations

import logging
from collections import deque

from .types import (
    CrafterAction,
    CrafterGoal,
    CrafterObject,
    CrafterObservation,
)

logger = logging.getLogger(__name__)


from hbllm.brain.reasoning.operators.base import ProblemType, ReasoningProblem
from hbllm.brain.reasoning.operators.registry import create_default_operator_registry
from hbllm.brain.reasoning.unified_runtime import UnifiedReasoningRuntime
from hbllm.hcir.graph import ActionNode, CognitiveGraph

from .perception import CrafterPerceptionAdapter
from .predicates import register_crafter_predicates


class CrafterActionAdapter:
    """
    Pure Device Driver for Crafter environment.

    Decoupled into:
    1. enumerate_affordances: Declares ActionNodes with causal preconditions and outcomes.
    2. execute_action: Low-level motor dispatch (grid BFS, facing, tool usage).
    3. plan_next_action: Routes causal planning directly through UnifiedReasoningRuntime.
    """

    def __init__(self) -> None:
        register_crafter_predicates()
        self.current_plan: list[CrafterAction] = []
        self.table_pos: tuple[int, int] | None = None
        self.furnace_pos: tuple[int, int] | None = None
        self.perception = CrafterPerceptionAdapter()
        self.runtime = UnifiedReasoningRuntime(create_default_operator_registry())

    def reset(self) -> None:
        """Reset internal plan and spatial landmarks."""
        self.current_plan.clear()
        self.table_pos = None
        self.furnace_pos = None
        self.perception = CrafterPerceptionAdapter()

    def enumerate_affordances(
        self, obs: CrafterObservation, graph: CognitiveGraph
    ) -> list[ActionNode]:
        """Declare candidate ActionNodes with explicit causal requirements and outputs."""
        affordances: list[ActionNode] = [
            # Survival & Defense
            ActionNode(
                id="act_defend",
                intent="defend",
                requirements=[],
                produces=["safe_from_monster"],
            ),
            ActionNode(
                id="act_sleep",
                intent="sleep",
                requirements=["no_mobs_adjacent"],
                produces=["vitals_safe(energy, 9)"],
            ),
            ActionNode(
                id="act_drink",
                intent="drink",
                requirements=["near(water)"],
                produces=["vitals_safe(drink, 5)"],
            ),
            ActionNode(
                id="act_eat",
                intent="eat",
                requirements=["near(cow)"],
                produces=["vitals_safe(food, 5)"],
            ),
            # Tech-Tree Crafting & Gathering
            ActionNode(
                id="act_collect_wood",
                intent="collect_wood",
                requirements=["near(tree)"],
                produces=["has(wood, 1)", "has(wood, 2)"],
            ),
            ActionNode(
                id="act_place_table",
                intent="place_table",
                requirements=["has(wood, 2)"],
                produces=["has(table)", "near(crafting_table)"],
            ),
            ActionNode(
                id="act_make_wood_pickaxe",
                intent="make_wood_pickaxe",
                requirements=["near(crafting_table)", "has(wood, 1)"],
                produces=["has(wood_pickaxe)"],
            ),
            ActionNode(
                id="act_collect_stone",
                intent="collect_stone",
                requirements=["has(wood_pickaxe)", "near(stone)"],
                produces=["has(stone, 1)", "has(stone, 4)"],
            ),
            ActionNode(
                id="act_make_stone_pickaxe",
                intent="make_stone_pickaxe",
                requirements=["near(crafting_table)", "has(wood, 1)", "has(stone, 1)"],
                produces=["has(stone_pickaxe)"],
            ),
            ActionNode(
                id="act_place_furnace",
                intent="place_furnace",
                requirements=["has(stone, 4)"],
                produces=["has(furnace)", "near(furnace)"],
            ),
            ActionNode(
                id="act_collect_coal",
                intent="collect_coal",
                requirements=["has(wood_pickaxe)", "near(coal)"],
                produces=["has(coal, 1)"],
            ),
            ActionNode(
                id="act_collect_iron",
                intent="collect_iron",
                requirements=["has(stone_pickaxe)", "near(iron)"],
                produces=["has(iron, 1)"],
            ),
            ActionNode(
                id="act_make_iron_pickaxe",
                intent="make_iron_pickaxe",
                requirements=[
                    "has(iron, 1)",
                    "has(coal, 1)",
                    "has(wood, 1)",
                    "near(crafting_table)",
                    "near(furnace)",
                ],
                produces=["has(iron_pickaxe)"],
            ),
            ActionNode(
                id="act_collect_diamond",
                intent="collect_diamond",
                requirements=["has(iron_pickaxe)", "near(diamond)"],
                produces=["has(diamond)"],
            ),
            # Spatial Approach Primitives
            ActionNode(
                id="act_approach_tree",
                intent="approach_tree",
                requirements=[],
                produces=["near(tree)"],
            ),
            ActionNode(
                id="act_approach_water",
                intent="approach_water",
                requirements=[],
                produces=["near(water)"],
            ),
            ActionNode(
                id="act_approach_cow",
                intent="approach_cow",
                requirements=[],
                produces=["near(cow)"],
            ),
            ActionNode(
                id="act_approach_table",
                intent="approach_crafting_table",
                requirements=[],
                produces=["near(crafting_table)"],
            ),
            ActionNode(
                id="act_approach_stone",
                intent="approach_stone",
                requirements=[],
                produces=["near(stone)"],
            ),
            ActionNode(
                id="act_approach_coal",
                intent="approach_coal",
                requirements=[],
                produces=["near(coal)"],
            ),
            ActionNode(
                id="act_approach_iron",
                intent="approach_iron",
                requirements=[],
                produces=["near(iron)"],
            ),
            ActionNode(
                id="act_approach_furnace",
                intent="approach_furnace",
                requirements=[],
                produces=["near(furnace)"],
            ),
            ActionNode(
                id="act_approach_diamond",
                intent="approach_diamond",
                requirements=[],
                produces=["near(diamond)"],
            ),
            # Default Exploration
            ActionNode(
                id="act_explore",
                intent="explore",
                requirements=[],
                produces=["explore_done"],
            ),
        ]

        for act in affordances:
            if graph.has_node(act.id):
                graph.remove_node(act.id)
            graph.add_node(act)

        return affordances

    def execute_action(self, intent: str, obs: CrafterObservation) -> CrafterAction:
        """Low-level actuator dispatch: translates declarative intent into discrete motor actions."""
        # 1. Self-defense mob engagement
        if intent == "defend":
            px, py = obs.player_pos
            height = len(obs.semantic_grid)
            width = len(obs.semantic_grid[0]) if height > 0 else 0
            if height > 0 and width > 0:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        if obs.semantic_grid[ny][nx] in (
                            CrafterObject.ZOMBIE,
                            CrafterObject.SKELETON,
                        ):
                            if obs.player_facing == (dx, dy):
                                return CrafterAction.DO
                            if (dx, dy) == (-1, 0):
                                return CrafterAction.MOVE_LEFT
                            if (dx, dy) == (1, 0):
                                return CrafterAction.MOVE_RIGHT
                            if (dx, dy) == (0, -1):
                                return CrafterAction.MOVE_UP
                            if (dx, dy) == (0, 1):
                                return CrafterAction.MOVE_DOWN
            return self._explore_passable(obs)

        # 2. Vitals & Sleep
        if intent == "sleep":
            return CrafterAction.SLEEP

        if intent in ("drink", "approach_water"):
            act = self._navigate_and_interact(obs, CrafterObject.WATER)
            return act or self._explore_passable(obs)

        if intent in ("eat", "approach_cow"):
            act = self._navigate_and_interact(obs, CrafterObject.COW)
            return act or self._explore_passable(obs)

        # 3. Wood & Table
        if intent in ("collect_wood", "approach_tree"):
            act = self._navigate_and_interact(obs, CrafterObject.TREE)
            return act or self._explore_passable(obs)

        if intent == "place_table":
            tx = obs.player_pos[0] + obs.player_facing[0]
            ty = obs.player_pos[1] + obs.player_facing[1]
            if 0 <= tx < len(obs.semantic_grid[0]) and 0 <= ty < len(obs.semantic_grid):
                if obs.semantic_grid[ty][tx] in (
                    CrafterObject.GRASS,
                    CrafterObject.PATH,
                    CrafterObject.SAND,
                ):
                    self.table_pos = (tx, ty)
                    return CrafterAction.PLACE_TABLE
            return CrafterAction.MOVE_LEFT

        if intent == "make_wood_pickaxe":
            if not self._is_near(obs, CrafterObject.CRAFTING_TABLE, radius=1):
                act = self._navigate_and_interact(obs, CrafterObject.CRAFTING_TABLE, face_only=True)
                if act:
                    return act
            return CrafterAction.MAKE_WOOD_PICKAXE

        # 4. Stone & Pickaxe
        if intent in ("collect_stone", "approach_stone"):
            act = self._navigate_and_interact(obs, CrafterObject.STONE)
            return act or self._explore_passable(obs)

        if intent == "make_stone_pickaxe":
            if not self._is_near(obs, CrafterObject.CRAFTING_TABLE, radius=1):
                act = self._navigate_and_interact(obs, CrafterObject.CRAFTING_TABLE, face_only=True)
                if act:
                    return act
            return CrafterAction.MAKE_STONE_PICKAXE

        # 5. Furnace & Metallurgy
        if intent == "place_furnace":
            if self.table_pos is not None and not self._is_near(
                obs, CrafterObject.CRAFTING_TABLE, radius=1
            ):
                act = self._navigate_and_interact(obs, CrafterObject.CRAFTING_TABLE, face_only=True)
                if act:
                    return act
            fx = obs.player_pos[0] + obs.player_facing[0]
            fy = obs.player_pos[1] + obs.player_facing[1]
            if 0 <= fx < len(obs.semantic_grid[0]) and 0 <= fy < len(obs.semantic_grid):
                if obs.semantic_grid[fy][fx] in (
                    CrafterObject.GRASS,
                    CrafterObject.PATH,
                    CrafterObject.SAND,
                ):
                    self.furnace_pos = (fx, fy)
                    return CrafterAction.PLACE_FURNACE
            return CrafterAction.MOVE_LEFT

        if intent in ("collect_coal", "approach_coal"):
            act = self._navigate_and_interact(obs, CrafterObject.COAL)
            return act or self._explore_passable(obs)

        if intent in ("collect_iron", "approach_iron"):
            act = self._navigate_and_interact(obs, CrafterObject.IRON)
            return act or self._explore_passable(obs)

        if intent == "make_iron_pickaxe":
            act = self._navigate_to_overlap(
                obs, CrafterObject.CRAFTING_TABLE, CrafterObject.FURNACE
            )
            if act:
                return act
            return CrafterAction.MAKE_IRON_PICKAXE

        if intent in ("collect_diamond", "approach_diamond"):
            act = self._navigate_and_interact(obs, CrafterObject.DIAMOND)
            return act or self._explore_passable(obs)

        if intent == "approach_crafting_table":
            act = self._navigate_and_interact(obs, CrafterObject.CRAFTING_TABLE, face_only=True)
            return act or self._explore_passable(obs)

        if intent == "approach_furnace":
            act = self._navigate_and_interact(obs, CrafterObject.FURNACE, face_only=True)
            return act or self._explore_passable(obs)

        return self._explore_passable(obs)

    def plan_next_action(
        self, obs: CrafterObservation, goal: CrafterGoal | None = None
    ) -> CrafterAction:
        """Select next action using UnifiedReasoningRuntime with EmbodiedCausalOperator."""
        # 1. Tactical Monster Combat Fast-Path
        px, py = obs.player_pos
        height = len(obs.semantic_grid)
        width = len(obs.semantic_grid[0]) if height > 0 else 0
        if height > 0 and width > 0:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = px + dx, py + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if obs.semantic_grid[ny][nx] in (CrafterObject.ZOMBIE, CrafterObject.SKELETON):
                        return self.execute_action("defend", obs)

        # 2. Perception Ingestion
        self.perception.ingest_observation(obs)

        # 3. Goal Translation
        goal_node = self.perception.ingest_goal(goal, obs)

        # 4. Affordance Declaration
        self.enumerate_affordances(obs, self.perception.graph)

        # 5. Cognitive Causal Planning via UnifiedReasoningRuntime
        problem = ReasoningProblem(
            problem_type=ProblemType.PLANNING,
            goal_node_ids=(goal_node.id,),
            description="Crafter tech-tree and survival goal resolution",
        )
        trace = self.runtime.reason(graph=self.perception.graph, problem=problem)

        # 6. Actuator Motor Dispatch
        if (
            trace
            and trace.final_result
            and trace.final_result.conclusions
            and "best_action" in trace.final_result.conclusions
        ):
            chosen_intent = trace.final_result.conclusions["best_action"]
            if chosen_intent and chosen_intent != "no_op":
                return self.execute_action(chosen_intent, obs)

        return self._explore_passable(obs)

    def _is_near(self, obs: CrafterObservation, obj_type: CrafterObject, radius: int = 2) -> bool:
        px, py = obs.player_pos
        if obj_type == CrafterObject.CRAFTING_TABLE and self.table_pos is not None:
            tx, ty = self.table_pos
            if abs(px - tx) <= radius and abs(py - ty) <= radius:
                return True
        if obj_type == CrafterObject.FURNACE and self.furnace_pos is not None:
            fx, fy = self.furnace_pos
            if abs(px - fx) <= radius and abs(py - fy) <= radius:
                return True
        height = len(obs.semantic_grid)
        width = len(obs.semantic_grid[0]) if height > 0 else 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = px + dx, py + dy
                if 0 <= x < width and 0 <= y < height:
                    if obs.semantic_grid[y][x] == obj_type:
                        return True
        return False

    def _navigate_and_interact(
        self,
        obs: CrafterObservation,
        target_type: CrafterObject,
        face_only: bool = False,
    ) -> CrafterAction | None:
        """Find nearest target, navigate adjacent via BFS, and interact."""
        px, py = obs.player_pos
        height = len(obs.semantic_grid)
        width = len(obs.semantic_grid[0])

        # 1. Find nearest instance of target_type
        target_pos = None
        best_dist = float("inf")
        for y in range(height):
            for x in range(width):
                if obs.semantic_grid[y][x] == target_type:
                    dist = abs(px - x) + abs(py - y)
                    if dist < best_dist:
                        best_dist = dist
                        target_pos = (x, y)

        if (
            target_pos is None
            and target_type == CrafterObject.CRAFTING_TABLE
            and self.table_pos is not None
        ):
            target_pos = self.table_pos

        if (
            target_pos is None
            and target_type == CrafterObject.FURNACE
            and self.furnace_pos is not None
        ):
            target_pos = self.furnace_pos

        if target_pos is None:
            return None

        tx, ty = target_pos
        # Check if already adjacent
        if abs(px - tx) + abs(py - ty) == 1:
            dx, dy = tx - px, ty - py
            if obs.player_facing == (dx, dy):
                return CrafterAction.NOOP if face_only else CrafterAction.DO
            # Turn to face target
            if dx == -1:
                return CrafterAction.MOVE_LEFT
            if dx == 1:
                return CrafterAction.MOVE_RIGHT
            if dy == -1:
                return CrafterAction.MOVE_UP
            if dy == 1:
                return CrafterAction.MOVE_DOWN

        # 2. BFS to adjacent passable tile
        return self._bfs_path_step(obs, target_pos)

    def _bfs_path_step(
        self, obs: CrafterObservation, target_pos: tuple[int, int]
    ) -> CrafterAction | None:
        """Compute one-step BFS movement toward an adjacent tile of target_pos."""
        start = obs.player_pos
        tx, ty = target_pos
        height = len(obs.semantic_grid)
        width = len(obs.semantic_grid[0])

        passable_ids = (
            CrafterObject.GRASS,
            CrafterObject.PATH,
            CrafterObject.SAND,
            CrafterObject.EMPTY,
        )

        queue = deque([(start[0], start[1], [])])
        visited = {start}

        while queue:
            cx, cy, path = queue.popleft()

            # Target reached if adjacent to target
            if abs(cx - tx) + abs(cy - ty) == 1:
                if path:
                    return path[0]
                return None

            if len(path) >= 60:  # Search depth cap
                continue

            for act, (dx, dy) in (
                (CrafterAction.MOVE_LEFT, (-1, 0)),
                (CrafterAction.MOVE_RIGHT, (1, 0)),
                (CrafterAction.MOVE_UP, (0, -1)),
                (CrafterAction.MOVE_DOWN, (0, 1)),
            ):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                    if obs.semantic_grid[ny][nx] in passable_ids:
                        visited.add((nx, ny))
                        queue.append((nx, ny, path + [act]))

        return None

    def _explore_passable(self, obs: CrafterObservation) -> CrafterAction:
        """Move in an available passable direction to discover new terrain."""
        px, py = obs.player_pos
        height = len(obs.semantic_grid)
        width = len(obs.semantic_grid[0]) if height > 0 else 0
        passable_ids = (
            CrafterObject.GRASS,
            CrafterObject.PATH,
            CrafterObject.SAND,
            CrafterObject.EMPTY,
        )
        fx, fy = obs.player_facing
        nx, ny = px + fx, py + fy
        if 0 <= nx < width and 0 <= ny < height and obs.semantic_grid[ny][nx] in passable_ids:
            if fx == -1:
                return CrafterAction.MOVE_LEFT
            if fx == 1:
                return CrafterAction.MOVE_RIGHT
            if fy == -1:
                return CrafterAction.MOVE_UP
            if fy == 1:
                return CrafterAction.MOVE_DOWN

        for act, (dx, dy) in (
            (CrafterAction.MOVE_UP, (0, -1)),
            (CrafterAction.MOVE_RIGHT, (1, 0)),
            (CrafterAction.MOVE_DOWN, (0, 1)),
            (CrafterAction.MOVE_LEFT, (-1, 0)),
        ):
            nx, ny = px + dx, py + dy
            if 0 <= nx < width and 0 <= ny < height and obs.semantic_grid[ny][nx] in passable_ids:
                return act

        return CrafterAction.DO

    def _navigate_to_overlap(
        self,
        obs: CrafterObservation,
        obj_a: CrafterObject,
        obj_b: CrafterObject,
    ) -> CrafterAction | None:
        """Find a passable tile within Chebyshev radius 1 of both objects and step toward it."""
        pos_a = self.table_pos if obj_a == CrafterObject.CRAFTING_TABLE else self.furnace_pos
        pos_b = self.furnace_pos if obj_b == CrafterObject.FURNACE else self.table_pos
        if pos_a is None or pos_b is None:
            return None

        px, py = obs.player_pos
        ax, ay = pos_a
        bx, by = pos_b
        height = len(obs.semantic_grid)
        width = len(obs.semantic_grid[0]) if height > 0 else 0

        passable_ids = (
            CrafterObject.GRASS,
            CrafterObject.PATH,
            CrafterObject.SAND,
            CrafterObject.EMPTY,
        )

        candidates = set()
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                cx, cy = ax + dx, ay + dy
                if 0 <= cx < width and 0 <= cy < height:
                    if abs(cx - bx) <= 1 and abs(cy - by) <= 1:
                        if obs.semantic_grid[cy][cx] in passable_ids:
                            candidates.add((cx, cy))

        if not candidates or (px, py) in candidates:
            return None

        queue = deque([(px, py, [])])
        visited = {(px, py)}
        while queue:
            cx, cy, path = queue.popleft()
            if (cx, cy) in candidates:
                if path:
                    return path[0]
                return None

            if len(path) >= 60:
                continue

            for act, (dx, dy) in (
                (CrafterAction.MOVE_LEFT, (-1, 0)),
                (CrafterAction.MOVE_RIGHT, (1, 0)),
                (CrafterAction.MOVE_UP, (0, -1)),
                (CrafterAction.MOVE_DOWN, (0, 1)),
            ):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                    if obs.semantic_grid[ny][nx] in passable_ids:
                        visited.add((nx, ny))
                        queue.append((nx, ny, path + [act]))
        return None
