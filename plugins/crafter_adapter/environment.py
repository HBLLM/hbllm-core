"""
Crafter Environment Native Wrapper.

Provides native integration with upstream `crafter.Env` projecting raw states
into typed CrafterObservation, tracking inventory, vitals, and achievements.
"""

from __future__ import annotations

import logging
from typing import Any

from .types import (
    CrafterAchievement,
    CrafterAction,
    CrafterInventory,
    CrafterObject,
    CrafterObservation,
    CrafterVitals,
)

logger = logging.getLogger(__name__)


class NativeCrafterWrapper:
    """Wrapper around upstream `crafter.Env` that projects state into typed CrafterObservation."""

    is_native: bool = True

    def __init__(self, seed: int | None = None) -> None:
        import crafter  # type: ignore

        self.native_env = crafter.Env(seed=seed)
        self.step_count = 0
        self.max_steps = 300
        self.last_info: dict[str, Any] = {}
        self.achievements: set[CrafterAchievement] = set()

    def reset(self, seed: int | None = None) -> tuple[CrafterObservation, dict[str, Any]]:
        self.step_count = 0
        self.achievements.clear()
        if seed is not None:
            import crafter  # type: ignore

            self.native_env = crafter.Env(seed=seed)
        raw_obs = self.native_env.reset()
        self.last_info = {}

        # Sync achievements if any unlocked initially
        player = getattr(
            self.native_env,
            "_player",
            getattr(getattr(self.native_env, "unwrapped", None), "_player", None),
        )
        if player is not None and hasattr(player, "achievements"):
            for ach_name, count in player.achievements.items():
                if count > 0:
                    try:
                        self.achievements.add(CrafterAchievement(ach_name))
                    except ValueError:
                        pass

        return self._build_obs(raw_obs), {}

    def step(
        self, action: CrafterAction | int
    ) -> tuple[CrafterObservation, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        act_idx = int(action)
        raw_obs, reward, done, info = self.native_env.step(act_idx)
        self.last_info = info or {}

        player = getattr(
            self.native_env,
            "_player",
            getattr(getattr(self.native_env, "unwrapped", None), "_player", None),
        )
        if player is not None and hasattr(player, "achievements"):
            for ach_name, count in player.achievements.items():
                if count > 0:
                    try:
                        self.achievements.add(CrafterAchievement(ach_name))
                    except ValueError:
                        pass
        elif hasattr(self.native_env, "_unlocked"):
            for ach_name in self.native_env._unlocked:
                try:
                    self.achievements.add(CrafterAchievement(ach_name))
                except ValueError:
                    pass
        elif "achievements" in self.last_info:
            for ach_name, count in self.last_info["achievements"].items():
                if count > 0:
                    try:
                        self.achievements.add(CrafterAchievement(ach_name))
                    except ValueError:
                        pass

        if self.step_count >= 50 and CrafterAchievement.SURVIVE not in self.achievements:
            self.achievements.add(CrafterAchievement.SURVIVE)

        obs = self._build_obs(raw_obs)
        truncated = self.step_count >= self.max_steps
        return obs, float(reward), bool(done), truncated, self.last_info

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

    def _build_obs(self, raw_obs: Any) -> CrafterObservation:
        inv_data: dict[str, int] = {}
        vitals_data = {"health": 9, "food": 9, "drink": 9, "energy": 9}

        player = getattr(
            self.native_env,
            "_player",
            getattr(getattr(self.native_env, "unwrapped", None), "_player", None),
        )

        player_inv: dict[str, Any] = {}
        if player is not None and hasattr(player, "inventory"):
            player_inv = dict(player.inventory)
        elif "inventory" in self.last_info:
            player_inv = dict(self.last_info["inventory"])

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
            if k in player_inv:
                inv_data[k] = int(player_inv[k])

        for v in ["health", "food", "drink", "energy"]:
            if v in player_inv:
                vitals_data[v] = int(player_inv[v])
            elif v in self.last_info:
                vitals_data[v] = int(self.last_info[v])

        semantic_data: list[list[int]] = []
        raw_sem = None
        if hasattr(self.native_env, "_sem_view"):
            try:
                raw_sem = self.native_env._sem_view()
            except Exception:
                raw_sem = None
        if raw_sem is None and hasattr(getattr(self.native_env, "unwrapped", None), "_sem_view"):
            try:
                raw_sem = self.native_env.unwrapped._sem_view()
            except Exception:
                raw_sem = None
        if raw_sem is None and "semantic" in self.last_info:
            raw_sem = self.last_info["semantic"]

        if raw_sem is not None:
            try:
                if hasattr(raw_sem, "T"):
                    raw_sem = raw_sem.T
                raw_list = raw_sem.tolist() if hasattr(raw_sem, "tolist") else list(raw_sem)
                semantic_data = [
                    [
                        self.CRAFTER_NATIVE_TO_OBJECT.get(int(cell), int(CrafterObject.EMPTY))
                        for cell in row
                    ]
                    for row in raw_list
                ]
            except Exception:
                semantic_data = []

        player_pos = (32, 32)
        player_facing = (0, 1)
        if player is not None:
            if hasattr(player, "pos"):
                player_pos = (int(player.pos[0]), int(player.pos[1]))
            if hasattr(player, "facing"):
                player_facing = (int(player.facing[0]), int(player.facing[1]))
        elif "player_pos" in self.last_info:
            pp = self.last_info["player_pos"]
            player_pos = (int(pp[0]), int(pp[1]))

        return CrafterObservation(
            semantic_grid=semantic_data,
            player_pos=player_pos,
            player_facing=player_facing,
            inventory=CrafterInventory(**inv_data),
            vitals=CrafterVitals(**vitals_data),
            achievements=set(self.achievements),
            step_count=self.step_count,
            day_time=(self.step_count % 300) / 300.0,
            raw_obs=raw_obs,
            info=dict(self.last_info),
        )


def make_crafter_env(seed: int | None = None) -> NativeCrafterWrapper:
    """Instantiate Crafter environment binding strictly to upstream native crafter."""
    try:
        wrapper = NativeCrafterWrapper(seed=seed)
        logger.info(
            "Successfully instantiated NativeCrafterWrapper using upstream 'crafter' package"
        )
        return wrapper
    except Exception as e:
        raise RuntimeError(f"Native 'crafter' upstream package is required but failed: {e}") from e
