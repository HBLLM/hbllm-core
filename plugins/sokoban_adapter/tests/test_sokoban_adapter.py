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
    NativeSokobanWrapper,
    SokobanAction,
    SokobanPerceptionAdapter,
    SokobanTier,
    make_sokoban_env,
    run_sokoban_tier,
)

from hbllm.hcir.world.predictors.physics import PhysicsPredictor


def test_plugin_manifest() -> None:
    manifest_path = Path(__file__).resolve().parent.parent / "plugin.json"
    assert manifest_path.exists(), "plugin.json must exist"
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["name"] == PLUGIN_NAME
    assert data["version"] == PLUGIN_VERSION
    assert "sokoban_detect_deadlocks" in data["capabilities"]
    assert data.get("supports_native_execution") is True
    assert "supports_standalone_fallback" not in data


def test_sokoban_environment_mechanics() -> None:
    env = make_sokoban_env(tier=SokobanTier.TIER_1_DIRECT_PUSH, seed=42)
    obs = env.reset()
    assert len(obs.boxes) >= 1
    assert len(obs.targets) >= 1
    assert obs.step_count == 0
    assert env.player_pos == obs.player_pos
    assert env.boxes == obs.boxes
    assert env.targets == obs.targets

    obs, reward, done, info = env.step(SokobanAction.RIGHT)
    assert obs.step_count == 1
    assert not done or obs.won


def test_corner_deadlock_detection() -> None:
    walls = {(0, 1), (1, 0)}
    targets = {(2, 2)}
    assert PhysicsPredictor.is_corner_deadlock((1, 1), walls, targets, (8, 8)) is True
    assert PhysicsPredictor.is_corner_deadlock((1, 1), walls, {(1, 1)}, (8, 8)) is False


def test_perception_identifies_taboo_cells() -> None:
    perception = SokobanPerceptionAdapter()
    env = make_sokoban_env(tier=SokobanTier.TIER_1_DIRECT_PUSH, seed=42)
    obs = env.reset()
    data = perception.process_observation(obs)
    assert "deadlock_taboo_cells" in data


def test_sokoban_tier_completion() -> None:
    res = run_sokoban_tier(SokobanTier.TIER_1_DIRECT_PUSH, episodes=1, seed=10)
    assert res["success_rate"] == 1.0, f"Tier 1 failed: {res}"
    assert res["deadlock_count"] == 0


def test_sokoban_full_benchmark_smoke() -> None:
    res = run_sokoban_tier(SokobanTier.TIER_2_OBSTACLE_NAVIGATION, episodes=1, seed=12)
    assert res["success_rate"] == 1.0, f"Tier 2 failed: {res}"


def test_native_sokoban_wrapper() -> None:
    env = make_sokoban_env(
        tier=SokobanTier.TIER_1_DIRECT_PUSH,
        seed=42,
    )
    assert isinstance(env, NativeSokobanWrapper)
    assert getattr(env, "is_native", False) is True
    obs = env.reset()
    assert obs is not None
    obs, reward, done, info = env.step(SokobanAction.RIGHT)
    assert obs.step_count == 1
