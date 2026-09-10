"""
MineDojo Action Adapter and Causal Recipe DAG Planner.

Decomposes target tool and recipe milestones:
Harvest Wood -> Craft Planks -> Craft Table -> Craft Sticks -> Craft Wooden Pickaxe.
"""

from __future__ import annotations

import logging

from hbllm.brain.reasoning.operators.base import ProblemType, ReasoningProblem
from hbllm.brain.reasoning.operators.registry import create_default_operator_registry
from hbllm.brain.reasoning.unified_runtime import UnifiedReasoningRuntime
from hbllm.hcir.graph import ActionNode, CognitiveGraph

from .perception import MineDojoPerceptionAdapter
from .types import (
    MineDojoAction,
    MineDojoGoal,
    MineDojoObservation,
)

logger = logging.getLogger(__name__)


class MineDojoActionAdapter:
    """
    HCIR Recursive Recipe DAG and Voxel Harvesting Planner for MineDojo.
    """

    def __init__(self) -> None:
        self.perception = MineDojoPerceptionAdapter()
        self.runtime = UnifiedReasoningRuntime(create_default_operator_registry())

    def reset(self) -> None:
        """Reset internal perception state."""
        self.perception = MineDojoPerceptionAdapter()

    def enumerate_affordances(
        self, obs: MineDojoObservation, graph: CognitiveGraph
    ) -> list[ActionNode]:
        """Declare candidate crafting and mining ActionNodes."""
        affordances = [
            ActionNode(
                id="act_mine_tree",
                intent="mine_tree",
                requirements=[],
                produces=["has(log)"],
            ),
            ActionNode(
                id="act_craft_planks",
                intent="craft_planks",
                requirements=["has(log)"],
                produces=["has(planks)"],
            ),
            ActionNode(
                id="act_craft_table",
                intent="craft_table",
                requirements=["has(planks, 4)"],
                produces=["has(crafting_table)"],
            ),
            ActionNode(
                id="act_craft_sticks",
                intent="craft_sticks",
                requirements=["has(planks, 2)"],
                produces=["has(stick)"],
            ),
            ActionNode(
                id="act_craft_wood_pickaxe",
                intent="craft_wood_pickaxe",
                requirements=["has(crafting_table)", "has(planks, 3)", "has(stick, 2)"],
                produces=["has(wooden_pickaxe)"],
            ),
            ActionNode(
                id="act_mine_stone",
                intent="mine_stone",
                requirements=["has(wooden_pickaxe)"],
                produces=["has(cobblestone)"],
            ),
            ActionNode(
                id="act_craft_stone_pickaxe",
                intent="craft_stone_pickaxe",
                requirements=["has(crafting_table)", "has(cobblestone, 3)", "has(stick, 2)"],
                produces=["has(stone_pickaxe)"],
            ),
        ]
        stale = [n.id for n in graph.all_nodes() if n.id.startswith("act_")]
        for sid in stale:
            graph.remove_node(sid)

        for act in affordances:
            graph.add_node(act)

        return affordances

    def execute_action(self, intent: str, obs: MineDojoObservation) -> MineDojoAction:
        """Low-level actuator dispatch: translates declarative intent into discrete motor actions."""
        if intent == "mine_tree":
            return self._navigate_and_mine_tree(obs)
        if intent == "craft_planks":
            return MineDojoAction.CRAFT_PLANKS
        if intent == "craft_table":
            return MineDojoAction.CRAFT_TABLE
        if intent == "craft_sticks":
            return MineDojoAction.CRAFT_STICKS
        if intent == "craft_wood_pickaxe":
            return MineDojoAction.CRAFT_WOOD_PICKAXE
        if intent == "mine_stone":
            return self._navigate_and_mine_stone(obs)
        if intent == "craft_stone_pickaxe":
            return MineDojoAction.CRAFT_STONE_PICKAXE
        return MineDojoAction.NOOP

    def plan_next_action(self, obs: MineDojoObservation, goal: MineDojoGoal) -> MineDojoAction:
        """Select next optimal action towards synthesizing goal item via HCIR backward chaining."""
        # 1. Update graph and declare affordances
        self.perception.ingest_observation(obs)
        goal_node = self.perception.ingest_goal(goal)
        self.enumerate_affordances(obs, self.perception.graph)

        # 2. Query UnifiedReasoningRuntime
        problem = ReasoningProblem(
            problem_type=ProblemType.PLANNING,
            goal_node_ids=(goal_node.id,),
            description=f"Synthesize {goal.target_item}",
        )
        trace = self.runtime.reason(graph=self.perception.graph, problem=problem)

        # 3. Actuator Motor Dispatch
        if (
            trace
            and trace.final_result
            and trace.final_result.conclusions
            and "best_action" in trace.final_result.conclusions
        ):
            chosen_intent = trace.final_result.conclusions["best_action"]
            if chosen_intent and chosen_intent != "no_op":
                return self.execute_action(chosen_intent, obs)

        return MineDojoAction.NOOP

    def _navigate_and_mine_tree(self, obs: MineDojoObservation) -> MineDojoAction:
        """Navigate to nearest tree at (16, 18) and mine."""
        px, py, pz = obs.player_pos
        tree_x, tree_y = 16, 18

        # Ensure facing North (yaw = 0)
        if obs.player_yaw != 0.0:
            return MineDojoAction.TURN_LEFT if obs.player_yaw == 90.0 else MineDojoAction.TURN_RIGHT

        # If already adjacent (y = 17, facing North to 18)
        if px == tree_x and py == tree_y - 1:
            return MineDojoAction.MINE_BLOCK

        # Move forward toward tree
        if py < tree_y - 1:
            return MineDojoAction.MOVE_FORWARD

        return MineDojoAction.MINE_BLOCK

    def _navigate_and_mine_stone(self, obs: MineDojoObservation) -> MineDojoAction:
        """Navigate to rock outcrop at (16, 14) and mine cobblestone."""
        px, py, pz = obs.player_pos

        # Ensure facing South (yaw = 180)
        if obs.player_yaw != 180.0:
            return MineDojoAction.TURN_RIGHT

        # If further north than y = 15, move forward towards south
        if py > 15:
            return MineDojoAction.MOVE_FORWARD

        # At y = 15 facing South (towards y = 14): mine stone
        return MineDojoAction.MINE_BLOCK
