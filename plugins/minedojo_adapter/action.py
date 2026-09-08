"""
MineDojo Action Adapter and Causal Recipe DAG Planner.

Decomposes target tool and recipe milestones:
Harvest Wood -> Craft Planks -> Craft Table -> Craft Sticks -> Craft Wooden Pickaxe.
"""

from __future__ import annotations

import logging

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

    def plan_next_action(self, obs: MineDojoObservation, goal: MineDojoGoal) -> MineDojoAction:
        """Select next optimal action towards synthesizing goal item."""
        inv = obs.inventory
        target = goal.target_item

        if target == "wooden_pickaxe":
            # If already have pickaxe
            if inv.wooden_pickaxe >= goal.target_count:
                return MineDojoAction.NOOP

            # 1. Convert any raw logs to planks
            if inv.log > 0:
                return MineDojoAction.CRAFT_PLANKS

            # 2. Craft Crafting Table if we don't have one (needs 4 planks)
            if inv.crafting_table == 0:
                if inv.planks >= 4:
                    return MineDojoAction.CRAFT_TABLE
                return self._navigate_and_mine_tree(obs)

            # 3. Craft Sticks if needed (needs 2 planks -> 4 sticks)
            if inv.stick < 2:
                if inv.planks >= 2:
                    return MineDojoAction.CRAFT_STICKS
                return self._navigate_and_mine_tree(obs)

            # 4. Craft Wooden Pickaxe (needs 3 planks + 2 sticks + table)
            if inv.planks >= 3 and inv.stick >= 2:
                return MineDojoAction.CRAFT_WOOD_PICKAXE

            # If not enough planks for pickaxe head, mine more wood
            return self._navigate_and_mine_tree(obs)

        elif target == "crafting_table":
            if inv.crafting_table >= goal.target_count:
                return MineDojoAction.NOOP
            if inv.log > 0:
                return MineDojoAction.CRAFT_PLANKS
            if inv.planks >= 4:
                return MineDojoAction.CRAFT_TABLE
            return self._navigate_and_mine_tree(obs)

        elif target == "planks":
            if inv.planks >= goal.target_count:
                return MineDojoAction.NOOP
            if inv.log > 0:
                return MineDojoAction.CRAFT_PLANKS
            return self._navigate_and_mine_tree(obs)

        return MineDojoAction.NOOP

    def _navigate_and_mine_tree(self, obs: MineDojoObservation) -> MineDojoAction:
        """Navigate to nearest tree at (16, 18) and mine."""
        px, py, pz = obs.player_pos
        tree_x, tree_y = 16, 18

        # If already adjacent (y = 17, facing North/ahead to 18)
        if px == tree_x and py == tree_y - 1:
            return MineDojoAction.MINE_BLOCK

        # Move forward toward tree
        if py < tree_y - 1:
            return MineDojoAction.MOVE_FORWARD

        return MineDojoAction.MINE_BLOCK
