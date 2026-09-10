"""
Unit and Integration Tests for Sokoban Adapter Plugin.
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

from sokoban_adapter import (
    PLUGIN_NAME,
    PLUGIN_VERSION,
    SokobanAction,
    SokobanDeadlockType,
    SokobanPerceptionAdapter,
    SokobanTier,
    make_sokoban_env,
    run_sokoban_benchmark,
    run_sokoban_tier,
)


def test_plugin_manifest() -> None:
    manifest_path = Path(__file__).resolve().parent.parent / "plugin.json"
    assert manifest_path.exists(), "plugin.json must exist"
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["name"] == PLUGIN_NAME
    assert data["version"] == PLUGIN_VERSION
    assert "sokoban_detect_deadlocks" in data["capabilities"]


def test_sokoban_environment_mechanics() -> None:
    env = make_sokoban_env(tier=SokobanTier.TIER_1_DIRECT_PUSH, seed=42, prefer_native=False)
    obs = env.reset()
    assert len(obs.boxes) == 1
    assert len(obs.targets) == 1
    assert obs.step_count == 0

    # Player at (2, 2), box at (2, 3), target at (2, 5). Push right moves box to (2, 4)
    obs, reward, done, info = env.step(SokobanAction.RIGHT)
    assert env.player_pos == (2, 3)
    assert (2, 4) in env.boxes
    assert not done

    # Push right again moves box to (2, 5) -> target reached!
    obs, reward, done, info = env.step(SokobanAction.RIGHT)
    assert env.player_pos == (2, 4)
    assert (2, 5) in env.boxes
    assert obs.won
    assert done


def test_corner_deadlock_detection() -> None:
    env = make_sokoban_env(
        tier=SokobanTier.TIER_3_CORNER_DEADLOCK_AVOIDANCE, seed=42, prefer_native=False
    )
    env.reset()
    # (1, 1) is a corner between top wall (0, 1) and left wall (1, 0)
    dl = env.check_deadlock((1, 1))
    assert dl == SokobanDeadlockType.CORNER


def test_perception_identifies_taboo_cells() -> None:
    perception = SokobanPerceptionAdapter()
    env = make_sokoban_env(
        tier=SokobanTier.TIER_3_CORNER_DEADLOCK_AVOIDANCE, seed=42, prefer_native=False
    )
    obs = env.reset()
    data = perception.process_observation(obs)
    assert "deadlock_taboo_cells" in data
    assert (1, 1) in data["deadlock_taboo_cells"]


def test_sokoban_all_tiers_completion() -> None:
    for tier in SokobanTier:
        res = run_sokoban_tier(tier, episodes=2, seed=10, prefer_native=False)
        assert res["success_rate"] == 1.0, (
            f"Tier {tier.value} failed to achieve 100% success rate: {res}"
        )
        assert res["deadlock_count"] == 0, f"Tier {tier.value} had unexpected deadlock: {res}"


def test_sokoban_full_benchmark_smoke() -> None:
    bench = run_sokoban_benchmark(episodes_per_tier=2, seed=42, prefer_native=False)
    assert bench["overall_success_rate"] == 1.0
    assert len(bench["tiers"]) == 5
    for tier_name, tier_res in bench["tiers"].items():
        assert tier_res["success_rate"] == 1.0


def test_native_sokoban_wrapper() -> None:
    env = make_sokoban_env(
        tier=SokobanTier.TIER_1_DIRECT_PUSH,
        seed=42,
        prefer_native=True,
        require_native=True,
    )
    assert getattr(env, "is_native", False) is True
    obs = env.reset()
    assert obs is not None
    obs, reward, done, info = env.step(SokobanAction.RIGHT)
    assert obs.step_count == 1
