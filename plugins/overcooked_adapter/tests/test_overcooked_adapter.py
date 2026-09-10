"""
Unit and Integration Tests for Overcooked-AI Adapter Plugin.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_test_dir = Path(__file__).resolve().parent
_plugin_dir = _test_dir.parent
_plugins_root = _plugin_dir.parent
_core_root = _plugins_root.parent

for p in [str(_core_root), str(_plugins_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from overcooked_adapter import (
    PLUGIN_NAME,
    PLUGIN_VERSION,
    CulinaryItem,
    KitchenTile,
    NativeOvercookedWrapper,
    OvercookedAction,
    OvercookedPerceptionAdapter,
    OvercookedTier,
    PotStatus,
    make_overcooked_env,
    run_overcooked_benchmark,
    run_overcooked_tier,
)


def test_plugin_manifest() -> None:
    manifest_path = Path(__file__).resolve().parent.parent / "plugin.json"
    assert manifest_path.exists(), "plugin.json must exist"
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["name"] == PLUGIN_NAME
    assert data["version"] == PLUGIN_VERSION
    assert "overcooked_plan_recipe" in data["capabilities"]


def test_kitchen_mechanics_solo() -> None:
    env = make_overcooked_env(
        tier=OvercookedTier.TIER_1_CRAMPED_ROOM_SOLO, seed=42, prefer_native=False
    )
    obs = env.reset()
    assert obs.agent.held_item == CulinaryItem.NONE
    assert obs.pots[0].status == PotStatus.EMPTY

    # Move adjacent to Onion Dispenser (0, 1) -> agent starts at (2, 2)
    env.step(OvercookedAction.UP)  # pos (1, 2)
    env.step(OvercookedAction.LEFT)  # pos (1, 1)
    env.step(OvercookedAction.UP)  # faces (0, 1)
    obs, _r, done, info = env.step(OvercookedAction.INTERACT)
    assert obs.agent.held_item == CulinaryItem.ONION


def test_perception_appliance_detection() -> None:
    perception = OvercookedPerceptionAdapter()
    env = make_overcooked_env(tier=OvercookedTier.TIER_1_CRAMPED_ROOM_SOLO, seed=42)
    obs = env.reset()
    data = perception.process_observation(obs)
    assert len(data["onion_dispensers"]) >= 1
    assert len(data["dish_dispensers"]) >= 1
    assert len(data["serving_stations"]) >= 1
    assert len(data["filling_pots"]) == 1


def test_overcooked_all_tiers_completion() -> None:
    for tier in OvercookedTier:
        res = run_overcooked_tier(tier, episodes=2, seed=42)
        assert res["success_rate"] == 1.0, (
            f"Tier {tier.value} failed to achieve 100% success rate: {res}"
        )


def test_overcooked_benchmark_smoke() -> None:
    bench = run_overcooked_benchmark(episodes_per_tier=2, seed=42)
    assert bench["overall_success_rate"] == 1.0
    assert len(bench["tiers"]) == 5
    for tier_name, tier_res in bench["tiers"].items():
        assert tier_res["success_rate"] == 1.0


def test_native_overcooked_wrapper() -> None:
    try:
        import overcooked_ai_py  # noqa: F401
    except ImportError:
        return

    env = make_overcooked_env(
        tier=OvercookedTier.TIER_1_CRAMPED_ROOM_SOLO, seed=42, prefer_native=True
    )
    assert getattr(env, "is_native", False) is True
    assert isinstance(env, NativeOvercookedWrapper)
    obs = env.reset()
    assert obs.agent.held_item == CulinaryItem.NONE
    assert len(obs.pots) >= 1
    assert obs.grid[0][2] == int(KitchenTile.POT) or len(obs.pots) >= 1

    # Step in native environment
    next_obs, reward, done, info = env.step(OvercookedAction.UP)
    assert not done
    assert next_obs.step_count == 1
