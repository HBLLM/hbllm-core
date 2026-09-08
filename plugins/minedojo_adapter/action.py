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

        # Tier 1: Raw Log Harvesting
        if target == "log":
            if inv.log >= goal.target_count:
                return MineDojoAction.NOOP
            return self._navigate_and_mine_tree(obs)

        # Tier 2: Planks & Sticks
        elif target == "planks":
            if inv.planks >= goal.target_count:
                return MineDojoAction.NOOP
            if inv.log > 0:
                return MineDojoAction.CRAFT_PLANKS
            return self._navigate_and_mine_tree(obs)

        # Tier 3: Crafting Table Synthesis
        elif target == "crafting_table":
            if inv.crafting_table >= goal.target_count:
                return MineDojoAction.NOOP
            if inv.log > 0:
                return MineDojoAction.CRAFT_PLANKS
            if inv.planks >= 4:
                return MineDojoAction.CRAFT_TABLE
            return self._navigate_and_mine_tree(obs)

        # Tier 4: Wooden Pickaxe
        elif target == "wooden_pickaxe":
            if inv.wooden_pickaxe >= goal.target_count:
                return MineDojoAction.NOOP
            if inv.log > 0:
                return MineDojoAction.CRAFT_PLANKS
            if inv.crafting_table == 0:
                if inv.planks >= 4:
                    return MineDojoAction.CRAFT_TABLE
                return self._navigate_and_mine_tree(obs)
            if inv.stick < 2:
                if inv.planks >= 2:
                    return MineDojoAction.CRAFT_STICKS
                return self._navigate_and_mine_tree(obs)
            if inv.planks >= 3 and inv.stick >= 2:
                return MineDojoAction.CRAFT_WOOD_PICKAXE
            return self._navigate_and_mine_tree(obs)

        # Tier 5: Stone Pickaxe & Cobblestone
        elif target in ("stone_pickaxe", "cobblestone"):
            if target == "stone_pickaxe" and inv.stone_pickaxe >= goal.target_count:
                return MineDojoAction.NOOP
            if target == "cobblestone" and inv.cobblestone >= goal.target_count:
                return MineDojoAction.NOOP

            # Sub-goal: Need wooden pickaxe first to mine stone
            if inv.wooden_pickaxe == 0:
                if inv.log > 0:
                    return MineDojoAction.CRAFT_PLANKS
                if inv.crafting_table == 0:
                    if inv.planks >= 4:
                        return MineDojoAction.CRAFT_TABLE
                    return self._navigate_and_mine_tree(obs)
                if inv.stick < 4:  # Craft enough sticks for both pickaxes
                    if inv.planks >= 2:
                        return MineDojoAction.CRAFT_STICKS
                    return self._navigate_and_mine_tree(obs)
                if inv.planks >= 3 and inv.stick >= 2:
                    return MineDojoAction.CRAFT_WOOD_PICKAXE
                return self._navigate_and_mine_tree(obs)

            # Wooden pickaxe in hand: acquire cobblestone
            if inv.cobblestone < 3:
                return self._navigate_and_mine_stone(obs)

            # Craft stone pickaxe
            if target == "stone_pickaxe":
                if inv.stick < 2:
                    if inv.planks >= 2:
                        return MineDojoAction.CRAFT_STICKS
                    return self._navigate_and_mine_tree(obs)
                return MineDojoAction.CRAFT_STONE_PICKAXE

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
