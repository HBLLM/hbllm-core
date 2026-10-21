"""
Overcooked-AI Native Kitchen Environment Wrapper.

Provides integration with authentic upstream overcooked_ai_py package for
multi-agent cooperative kitchen simulation, recipe progression, cooking timers,
counter exchange, and partner dynamics.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

if not hasattr(np, "Inf"):
    np.Inf = np.inf  # type: ignore
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_  # type: ignore

from .types import (
    AgentState,
    CulinaryItem,
    KitchenTile,
    OvercookedAction,
    OvercookedObservation,
    OvercookedTier,
    PotState,
    PotStatus,
)

logger = logging.getLogger(__name__)


class NativeOvercookedWrapper:
    """Wrapper around authentic upstream overcooked_ai_py package."""

    is_native: bool = True

    LAYOUT_MAP = {
        OvercookedTier.TIER_1_CRAMPED_ROOM_SOLO: "cramped_room",
        OvercookedTier.TIER_2_ASYMMETRIC_COORDINATION: "asymmetric_advantages",
        OvercookedTier.TIER_3_CORRIDOR_CONTENTION: "coordination_ring",
        OvercookedTier.TIER_4_DYNAMIC_PARTNER_ADAPTATION: "forced_coordination",
        OvercookedTier.TIER_5_MULTI_ORDER_SURGE: "counter_circuit_o_1order",
    }

    TILE_MAP = {
        " ": int(KitchenTile.FLOOR),
        "X": int(KitchenTile.COUNTER),
        "O": int(KitchenTile.ONION_DISPENSER),
        "D": int(KitchenTile.DISH_DISPENSER),
        "P": int(KitchenTile.POT),
        "S": int(KitchenTile.SERVING_STATION),
    }

    def __init__(
        self,
        tier: OvercookedTier | str = OvercookedTier.TIER_1_CRAMPED_ROOM_SOLO,
        seed: int = 42,
        max_steps: int = 150,
    ) -> None:
        try:
            from overcooked_ai_py.mdp.actions import Action, Direction
            from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
            from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
        except ImportError as err:
            raise ImportError(
                "overcooked-ai is required for NativeOvercookedWrapper. "
                "Install via 'pip install overcooked-ai'."
            ) from err

        self._action_mod = Action
        self._direction_mod = Direction
        self._overcooked_env_cls = OvercookedEnv
        self._gridworld_cls = OvercookedGridworld

        if isinstance(tier, int):
            tier_int_map = {
                1: OvercookedTier.TIER_1_CRAMPED_ROOM_SOLO,
                2: OvercookedTier.TIER_2_ASYMMETRIC_COORDINATION,
                3: OvercookedTier.TIER_3_CORRIDOR_CONTENTION,
                4: OvercookedTier.TIER_4_DYNAMIC_PARTNER_ADAPTATION,
                5: OvercookedTier.TIER_5_MULTI_ORDER_SURGE,
            }
            self.tier = tier_int_map.get(tier, OvercookedTier.TIER_1_CRAMPED_ROOM_SOLO)
        else:
            self.tier = OvercookedTier(tier)
        self.seed = seed
        self.max_steps = max_steps
        self.step_count = 0
        self.soups_delivered = 0
        self.target_soups = 1

        layout_name = self.LAYOUT_MAP.get(self.tier, "cramped_room")
        self.base_mdp = self._gridworld_cls.from_layout_name(layout_name)
        self.env = self._overcooked_env_cls.from_mdp(self.base_mdp, horizon=max_steps)

        # In overcooked-ai, terrain_mtx is [height][width] (row, col)
        self.height = self.base_mdp.height
        self.width = self.base_mdp.width
        self.grid = [
            [self.TILE_MAP.get(char, int(KitchenTile.FLOOR)) for char in row]
            for row in self.base_mdp.terrain_mtx
        ]

        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> OvercookedObservation:
        """Reset native OvercookedEnv state."""
        if seed is not None:
            self.seed = seed
        self.step_count = 0
        self.soups_delivered = 0
        self.env.reset()
        return self._build_obs(done=False, won=False)

    def _convert_action(self, action: OvercookedAction | int) -> Any:
        """Convert OvercookedAction enum to overcooked-ai primitive."""
        act_enum = OvercookedAction(action)
        if act_enum == OvercookedAction.UP:
            return self._direction_mod.NORTH
        if act_enum == OvercookedAction.DOWN:
            return self._direction_mod.SOUTH
        if act_enum == OvercookedAction.LEFT:
            return self._direction_mod.WEST
        if act_enum == OvercookedAction.RIGHT:
            return self._direction_mod.EAST
        if act_enum == OvercookedAction.INTERACT:
            return self._action_mod.INTERACT
        return self._action_mod.STAY

    def step(
        self,
        action: OvercookedAction | int,
        partner_action: OvercookedAction | int | None = None,
    ) -> tuple[OvercookedObservation, float, bool, dict[str, Any]]:
        """Step native Overcooked environment."""
        self.step_count += 1
        act0 = self._convert_action(action)
        act1 = (
            self._convert_action(partner_action)
            if partner_action is not None
            else self._action_mod.STAY
        )

        num_players = len(self.env.state.players)
        action_tuple = (act0, act1) if num_players > 1 else (act0,)

        next_state, reward, done, info = self.env.step(action_tuple)

        if reward > 0:
            # Overcooked delivers 20 reward per completed soup
            delivered = max(1, int(reward // 20))
            self.soups_delivered += delivered

        won = self.soups_delivered >= self.target_soups
        time_limit_done = self.step_count >= self.max_steps
        episode_done = done or won or time_limit_done

        obs = self._build_obs(done=episode_done, won=won, info=info)
        return obs, float(reward), episode_done, info or {}

    def _build_obs(
        self,
        done: bool = False,
        won: bool = False,
        info: dict[str, Any] | None = None,
    ) -> OvercookedObservation:
        """Project native overcooked_ai_py state into typed OvercookedObservation."""
        state = self.env.state

        def _map_held(player: Any) -> CulinaryItem:
            if not player.has_object():
                return CulinaryItem.NONE
            name = player.get_object().name
            if name == "onion":
                return CulinaryItem.ONION
            if name == "dish":
                return CulinaryItem.DISH
            if name == "soup":
                return CulinaryItem.SOUP
            return CulinaryItem.NONE

        p0 = state.players[0]
        # (x, y) col, row -> (r, c) row, col
        # (dx, dy) col_dir, row_dir -> (dr, dc) row_dir, col_dir
        p0_pos = (p0.position[1], p0.position[0])
        p0_orient = (p0.orientation[1], p0.orientation[0])
        agent = AgentState(
            agent_id=0,
            pos=p0_pos,
            orientation=p0_orient,
            held_item=_map_held(p0),
        )

        partner: AgentState | None = None
        if len(state.players) > 1:
            p1 = state.players[1]
            p1_pos = (p1.position[1], p1.position[0])
            p1_orient = (p1.orientation[1], p1.orientation[0])
            partner = AgentState(
                agent_id=1,
                pos=p1_pos,
                orientation=p1_orient,
                held_item=_map_held(p1),
            )

        # Pot states
        pots: list[PotState] = []
        pot_locs = set(self.base_mdp.get_pot_locations())
        for x, y in sorted(pot_locs):
            r, c = y, x
            if (x, y) in state.objects:
                obj = state.objects[(x, y)]
                ingredients = getattr(obj, "ingredients", [])
                num_onions = len(
                    [
                        ing
                        for ing in ingredients
                        if ing == "onion" or getattr(ing, "name", "") == "onion"
                    ]
                )
                if getattr(obj, "is_ready", False):
                    p_status = PotStatus.READY
                elif getattr(obj, "is_cooking", False):
                    p_status = PotStatus.COOKING
                elif num_onions > 0:
                    p_status = PotStatus.FILLING
                else:
                    p_status = PotStatus.EMPTY

                try:
                    cook_tick = getattr(obj, "_cooking_tick", getattr(obj, "cooking_tick", -1))
                except Exception:
                    cook_tick = -1
                try:
                    cook_time = getattr(obj, "_cook_time", getattr(obj, "cook_time", 20))
                except Exception:
                    cook_time = 20
                pots.append(
                    PotState(
                        pos=(r, c),
                        onions_in_pot=num_onions,
                        required_onions=3,
                        cooking_timer=max(0, cook_tick),
                        cooking_duration=cook_time,
                        status=p_status,
                    )
                )
            else:
                pots.append(PotState(pos=(r, c), status=PotStatus.EMPTY))

        # Counter items
        counter_items: dict[tuple[int, int], CulinaryItem] = {}
        for (ox, oy), obj in state.objects.items():
            if (ox, oy) not in pot_locs:
                name = getattr(obj, "name", "")
                item = CulinaryItem.NONE
                if name == "onion":
                    item = CulinaryItem.ONION
                elif name == "dish":
                    item = CulinaryItem.DISH
                elif name == "soup":
                    item = CulinaryItem.SOUP
                if item != CulinaryItem.NONE:
                    counter_items[(oy, ox)] = item

        return OvercookedObservation(
            grid=[row[:] for row in self.grid],
            agent=agent,
            partner=partner,
            pots=pots,
            counter_items=counter_items,
            soups_delivered=self.soups_delivered,
            step_count=self.step_count,
            max_steps=self.max_steps,
            done=done,
            won=won,
            info=info or {},
        )


def make_overcooked_env(
    tier: OvercookedTier | str = OvercookedTier.TIER_1_CRAMPED_ROOM_SOLO,
    seed: int = 42,
    max_steps: int = 150,
) -> NativeOvercookedWrapper:
    """Factory creating configured native Overcooked-AI environments."""
    try:
        wrapper = NativeOvercookedWrapper(tier=tier, seed=seed, max_steps=max_steps)
        logger.info("Successfully bound to native Overcooked-AI environment (tier=%s)", tier)
        return wrapper
    except Exception as err:
        raise RuntimeError(
            f"Native 'overcooked_ai_py' upstream package is required but failed: {err}"
        ) from err
