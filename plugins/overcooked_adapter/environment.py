"""
Overcooked-AI Standalone Kitchen Environment.

Provides a deterministic multi-agent cooperative kitchen simulator implementing
authentic recipe progression, cooking timers, counter exchange, and partner dynamics.
"""

from __future__ import annotations

import logging
import random
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

DIRECTION_DELTAS = {
    OvercookedAction.UP: (-1, 0),
    OvercookedAction.DOWN: (1, 0),
    OvercookedAction.LEFT: (0, -1),
    OvercookedAction.RIGHT: (0, 1),
}


class StandaloneOvercookedEnv:
    """Deterministic, high-fidelity cooperative kitchen engine."""

    def __init__(
        self,
        tier: OvercookedTier | str = OvercookedTier.TIER_1_CRAMPED_ROOM_SOLO,
        seed: int = 42,
        max_steps: int = 150,
    ) -> None:
        self.tier = OvercookedTier(tier)
        self.seed = seed
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.step_count = 0
        self.soups_delivered = 0
        self.target_soups = 1

        self.width = 5
        self.height = 5
        self.grid: list[list[int]] = []
        self.agent = AgentState(agent_id=0, pos=(2, 2))
        self.partner: AgentState | None = None
        self.pots: list[PotState] = []
        self.counter_items: dict[tuple[int, int], CulinaryItem] = {}

        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> OvercookedObservation:
        """Reset kitchen state according to tier layout."""
        if seed is not None:
            self.seed = seed
            self.rng = random.Random(seed)

        self.step_count = 0
        self.soups_delivered = 0
        self.counter_items.clear()
        self.pots.clear()

        self._build_tier_layout()
        return self._get_obs()

    def _build_tier_layout(self) -> None:
        """Construct canonical kitchen topology and agent placements."""
        if self.tier == OvercookedTier.TIER_1_CRAMPED_ROOM_SOLO:
            # 5x5 cramped kitchen: Solo agent
            self.target_soups = 1
            self.height, self.width = 5, 5
            self.grid = [[int(KitchenTile.COUNTER) for _ in range(5)] for _ in range(5)]
            for r in range(1, 4):
                for c in range(1, 4):
                    self.grid[r][c] = int(KitchenTile.FLOOR)

            self.grid[0][1] = int(KitchenTile.ONION_DISPENSER)
            self.grid[0][2] = int(KitchenTile.POT)
            self.grid[4][1] = int(KitchenTile.DISH_DISPENSER)
            self.grid[4][2] = int(KitchenTile.SERVING_STATION)

            self.pots = [PotState(pos=(0, 2), cooking_duration=3)]
            self.agent = AgentState(agent_id=0, pos=(2, 2), orientation=(-1, 0))
            self.partner = None

        elif self.tier == OvercookedTier.TIER_2_ASYMMETRIC_COORDINATION:
            # Shared counter divider: Agent on left room, Partner on right room
            self.target_soups = 1
            self.height, self.width = 5, 7
            self.grid = [[int(KitchenTile.COUNTER) for _ in range(7)] for _ in range(5)]
            # Left floor
            for r in range(1, 4):
                for c in range(1, 3):
                    self.grid[r][c] = int(KitchenTile.FLOOR)
            # Right floor
            for r in range(1, 4):
                for c in range(4, 6):
                    self.grid[r][c] = int(KitchenTile.FLOOR)
            # Column 3 is the shared counter divider
            self.grid[0][1] = int(KitchenTile.ONION_DISPENSER)
            self.grid[0][5] = int(KitchenTile.POT)
            self.grid[4][5] = int(KitchenTile.DISH_DISPENSER)
            self.grid[2][6] = int(KitchenTile.SERVING_STATION)

            self.pots = [PotState(pos=(0, 5), cooking_duration=3)]
            self.agent = AgentState(agent_id=0, pos=(2, 1), orientation=(0, 1))
            self.partner = AgentState(agent_id=1, pos=(2, 4), orientation=(0, -1))

        elif self.tier in (
            OvercookedTier.TIER_3_CORRIDOR_CONTENTION,
            OvercookedTier.TIER_4_DYNAMIC_PARTNER_ADAPTATION,
        ):
            # Shared kitchen with narrow corridor
            self.target_soups = 1
            self.height, self.width = 5, 7
            self.grid = [[int(KitchenTile.COUNTER) for _ in range(7)] for _ in range(5)]
            for r in range(1, 4):
                for c in range(1, 6):
                    self.grid[r][c] = int(KitchenTile.FLOOR)
            # Bottleneck pillar in center
            self.grid[2][3] = int(KitchenTile.COUNTER)

            self.grid[0][1] = int(KitchenTile.ONION_DISPENSER)
            self.grid[0][5] = int(KitchenTile.POT)
            self.grid[4][1] = int(KitchenTile.DISH_DISPENSER)
            self.grid[4][5] = int(KitchenTile.SERVING_STATION)

            self.pots = [PotState(pos=(0, 5), cooking_duration=3)]
            self.agent = AgentState(agent_id=0, pos=(1, 2), orientation=(0, 1))
            self.partner = AgentState(agent_id=1, pos=(3, 4), orientation=(0, -1))

        elif self.tier == OvercookedTier.TIER_5_MULTI_ORDER_SURGE:
            # 2 pots, surge of 2 soups
            self.target_soups = 2
            self.height, self.width = 5, 6
            self.grid = [[int(KitchenTile.COUNTER) for _ in range(6)] for _ in range(5)]
            for r in range(1, 4):
                for c in range(1, 5):
                    self.grid[r][c] = int(KitchenTile.FLOOR)

            self.grid[0][1] = int(KitchenTile.ONION_DISPENSER)
            self.grid[0][3] = int(KitchenTile.POT)
            self.grid[0][4] = int(KitchenTile.POT)
            self.grid[4][1] = int(KitchenTile.DISH_DISPENSER)
            self.grid[4][3] = int(KitchenTile.SERVING_STATION)

            self.pots = [
                PotState(pos=(0, 3), cooking_duration=3),
                PotState(pos=(0, 4), cooking_duration=3),
            ]
            self.agent = AgentState(agent_id=0, pos=(2, 2), orientation=(-1, 0))
            self.partner = None

    def step(
        self,
        action: OvercookedAction | int,
        partner_action: OvercookedAction | int | None = None,
    ) -> tuple[OvercookedObservation, float, bool, dict[str, Any]]:
        """Apply agent action, advance kitchen timers, and step partner agent."""
        self.step_count += 1
        act = OvercookedAction(action)

        # 1. Execute agent action
        self._apply_agent_action(self.agent, act)

        # 2. Step partner agent if present
        if self.partner is not None:
            if partner_action is not None:
                self._apply_agent_action(self.partner, OvercookedAction(partner_action))
            else:
                self._step_partner_agent()

        # 3. Advance cooking timers in all pots
        for pot in self.pots:
            if pot.status == PotStatus.COOKING:
                pot.cooking_timer -= 1
                if pot.cooking_timer <= 0:
                    pot.status = PotStatus.READY

        won = self.soups_delivered >= self.target_soups
        done = won or self.step_count >= self.max_steps
        reward = 20.0 if won else -0.01

        info = {
            "won": won,
            "soups_delivered": self.soups_delivered,
            "steps": self.step_count,
        }

        obs = self._get_obs(done=done, won=won, info=info)
        return obs, reward, done, info

    def _apply_agent_action(self, agent: AgentState, act: OvercookedAction) -> None:
        """Process primitive navigation or culinary interaction."""
        other_agent_pos = (
            self.partner.pos
            if agent == self.agent and self.partner
            else (self.agent.pos if agent != self.agent else None)
        )

        if act in DIRECTION_DELTAS:
            dr, dc = DIRECTION_DELTAS[act]
            agent.orientation = (dr, dc)
            nr, nc = agent.pos[0] + dr, agent.pos[1] + dc

            # Validate destination
            if 0 <= nr < self.height and 0 <= nc < self.width:
                if self.grid[nr][nc] == int(KitchenTile.FLOOR) and (nr, nc) != other_agent_pos:
                    agent.pos = (nr, nc)

        elif act == OvercookedAction.INTERACT:
            dr, dc = agent.orientation
            target_r, target_c = agent.pos[0] + dr, agent.pos[1] + dc

            if 0 <= target_r < self.height and 0 <= target_c < self.width:
                tile = self.grid[target_r][target_c]

                # Interact with Onion Dispenser
                if (
                    tile == int(KitchenTile.ONION_DISPENSER)
                    and agent.held_item == CulinaryItem.NONE
                ):
                    agent.held_item = CulinaryItem.ONION

                # Interact with Dish Dispenser
                elif (
                    tile == int(KitchenTile.DISH_DISPENSER) and agent.held_item == CulinaryItem.NONE
                ):
                    agent.held_item = CulinaryItem.DISH

                # Interact with Pot
                elif tile == int(KitchenTile.POT):
                    pot = next((p for p in self.pots if p.pos == (target_r, target_c)), None)
                    if pot:
                        if agent.held_item == CulinaryItem.ONION and pot.status in (
                            PotStatus.EMPTY,
                            PotStatus.FILLING,
                        ):
                            pot.onions_in_pot += 1
                            agent.held_item = CulinaryItem.NONE
                            if pot.onions_in_pot >= pot.required_onions:
                                pot.status = PotStatus.COOKING
                                pot.cooking_timer = pot.cooking_duration
                            else:
                                pot.status = PotStatus.FILLING
                        elif agent.held_item == CulinaryItem.DISH and pot.status == PotStatus.READY:
                            agent.held_item = CulinaryItem.SOUP
                            pot.status = PotStatus.EMPTY
                            pot.onions_in_pot = 0

                # Interact with Serving Station
                elif (
                    tile == int(KitchenTile.SERVING_STATION)
                    and agent.held_item == CulinaryItem.SOUP
                ):
                    agent.held_item = CulinaryItem.NONE
                    self.soups_delivered += 1

                # Interact with Counter
                elif tile == int(KitchenTile.COUNTER):
                    counter_pos = (target_r, target_c)
                    if counter_pos in self.counter_items and agent.held_item == CulinaryItem.NONE:
                        # Pick up from counter
                        agent.held_item = self.counter_items.pop(counter_pos)
                    elif (
                        counter_pos not in self.counter_items
                        and agent.held_item != CulinaryItem.NONE
                    ):
                        # Place onto counter
                        self.counter_items[counter_pos] = agent.held_item
                        agent.held_item = CulinaryItem.NONE

    def _step_partner_agent(self) -> None:
        """Execute autonomous partner behavior according to tier."""
        assert self.partner is not None
        if self.tier == OvercookedTier.TIER_2_ASYMMETRIC_COORDINATION:
            # Partner monitors counter (2, 3). If onion on counter, grabs it and puts in pot!
            # If pot is ready, grabs dish, scoops soup, and serves.
            pot = self.pots[0]
            shared_counter = (2, 3)

            if self.partner.held_item == CulinaryItem.SOUP:
                # Deliver soup to serving station at (2, 6)
                if self.partner.pos != (2, 5):
                    self._partner_move_towards((2, 5))
                else:
                    self.partner.orientation = (0, 1)  # Face serving station (2, 6)
                    self._apply_agent_action(self.partner, OvercookedAction.INTERACT)
            elif pot.status == PotStatus.READY:
                if self.partner.held_item == CulinaryItem.DISH:
                    # Scoop soup from pot (0, 5)
                    if self.partner.pos != (1, 5):
                        self._partner_move_towards((1, 5))
                    else:
                        self.partner.orientation = (-1, 0)
                        self._apply_agent_action(self.partner, OvercookedAction.INTERACT)
                else:
                    # Fetch dish from (4, 5)
                    if self.partner.pos != (3, 5):
                        self._partner_move_towards((3, 5))
                    else:
                        self.partner.orientation = (1, 0)
                        self._apply_agent_action(self.partner, OvercookedAction.INTERACT)
            elif (
                shared_counter in self.counter_items
                and self.counter_items[shared_counter] == CulinaryItem.ONION
                and self.partner.held_item == CulinaryItem.NONE
            ):
                # Pick up onion from shared counter
                if self.partner.pos != (2, 4):
                    self._partner_move_towards((2, 4))
                else:
                    self.partner.orientation = (0, -1)
                    self._apply_agent_action(self.partner, OvercookedAction.INTERACT)
            elif self.partner.held_item == CulinaryItem.ONION:
                # Put onion in pot (0, 5)
                if self.partner.pos != (1, 5):
                    self._partner_move_towards((1, 5))
                else:
                    self.partner.orientation = (-1, 0)
                    self._apply_agent_action(self.partner, OvercookedAction.INTERACT)
            else:
                # Idle wait
                pass

        elif self.tier == OvercookedTier.TIER_3_CORRIDOR_CONTENTION:
            # Partner walks back and forth, yielding if agent approaches
            if self.partner.pos[1] > 3:
                self.partner.orientation = (0, -1)
            else:
                self.partner.orientation = (0, 1)

        elif self.tier == OvercookedTier.TIER_4_DYNAMIC_PARTNER_ADAPTATION:
            # Partner takes occasional random step
            if self.rng.random() < 0.3:
                act = self.rng.choice(
                    [
                        OvercookedAction.UP,
                        OvercookedAction.DOWN,
                        OvercookedAction.LEFT,
                        OvercookedAction.RIGHT,
                        OvercookedAction.STAY,
                    ]
                )
                self._apply_agent_action(self.partner, act)

    def _partner_move_towards(self, target_pos: tuple[int, int]) -> None:
        """Greedy step towards target cell."""
        assert self.partner is not None
        pr, pc = self.partner.pos
        tr, tc = target_pos
        if pr < tr:
            self._apply_agent_action(self.partner, OvercookedAction.DOWN)
        elif pr > tr:
            self._apply_agent_action(self.partner, OvercookedAction.UP)
        elif pc < tc:
            self._apply_agent_action(self.partner, OvercookedAction.RIGHT)
        elif pc > tc:
            self._apply_agent_action(self.partner, OvercookedAction.LEFT)

    def _get_obs(
        self,
        done: bool = False,
        won: bool = False,
        info: dict[str, Any] | None = None,
    ) -> OvercookedObservation:
        """Construct full observation."""
        return OvercookedObservation(
            grid=[row[:] for row in self.grid],
            agent=AgentState(
                agent_id=self.agent.agent_id,
                pos=self.agent.pos,
                orientation=self.agent.orientation,
                held_item=self.agent.held_item,
            ),
            partner=AgentState(
                agent_id=self.partner.agent_id,
                pos=self.partner.pos,
                orientation=self.partner.orientation,
                held_item=self.partner.held_item,
            )
            if self.partner
            else None,
            pots=[
                PotState(
                    pos=p.pos,
                    onions_in_pot=p.onions_in_pot,
                    required_onions=p.required_onions,
                    cooking_timer=p.cooking_timer,
                    cooking_duration=p.cooking_duration,
                    status=p.status,
                )
                for p in self.pots
            ],
            counter_items=dict(self.counter_items),
            soups_delivered=self.soups_delivered,
            step_count=self.step_count,
            max_steps=self.max_steps,
            done=done,
            won=won,
            info=info or {},
        )


class NativeOvercookedWrapper:
    """Dual-mode wrapper wrapping authentic upstream overcooked_ai_py package."""

    is_native: bool = True

    LAYOUT_MAP = {
        OvercookedTier.TIER_1_CRAMPED_ROOM_SOLO: "cramped_room",
        OvercookedTier.TIER_2_ASYMMETRIC_COORDINATION: "asymmetric_advantages",
        OvercookedTier.TIER_3_CORRIDOR_CONTENTION: "coordination_ring",
        OvercookedTier.TIER_4_DYNAMIC_PARTNER_ADAPTATION: "forced_coordination",
        OvercookedTier.TIER_5_MULTI_ORDER_SURGE: "counter_circuit",
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
                "Install via 'pip install overcooked-ai' or use StandaloneOvercookedEnv."
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
    prefer_native: bool = False,
) -> StandaloneOvercookedEnv | NativeOvercookedWrapper:
    """Factory creating Overcooked environments with dual-mode native/standalone selection."""
    if prefer_native:
        try:
            return NativeOvercookedWrapper(tier=tier, seed=seed, max_steps=max_steps)
        except Exception as err:
            logger.warning(
                "Failed to initialize NativeOvercookedWrapper (%s), falling back to StandaloneOvercookedEnv",
                err,
            )
    return StandaloneOvercookedEnv(tier=tier, seed=seed, max_steps=max_steps)
