"""
Tests for NetHack Adapter Plugin.
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

from nethack_adapter import (
    PLUGIN_NAME,
    PLUGIN_VERSION,
    NativeNetHackWrapper,
    NetHackAction,
    NetHackPerceptionAdapter,
    PureHCIRNetHackAgent,
    make_nethack_env,
)


def test_plugin_manifest() -> None:
    manifest_path = Path(__file__).resolve().parent.parent / "plugin.json"
    assert manifest_path.exists(), "plugin.json must exist"
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["name"] == PLUGIN_NAME
    assert data["version"] == PLUGIN_VERSION
    assert "nethack_explore_dungeon" in data["capabilities"]
    assert data.get("supports_native_execution") is True
    assert "supports_standalone_fallback" not in data


def test_nethack_environment_lifecycle() -> None:
    env = make_nethack_env(seed=42, tier=1)
    assert isinstance(env, NativeNetHackWrapper)
    obs, info = env.reset(seed=42)

    assert obs.player_pos is not None
    assert obs.stats.hp > 0
    assert len(obs.glyphs) == 21
    assert len(obs.glyphs[0]) == 79
    assert len(obs.chars) == 21

    # Execute a movement step
    next_obs, reward, term, trunc, info = env.step(NetHackAction.EAST)
    assert next_obs.step_count == 1


def test_nethack_perception_adapter() -> None:
    env = make_nethack_env(seed=101, tier=1)
    obs, _ = env.reset(seed=101)

    perception = NetHackPerceptionAdapter()
    graph = perception.ingest_observation(obs)

    agent_node = graph.get_node("agent")
    assert agent_node is not None
    assert agent_node.properties["hp"] == obs.stats.hp


def test_nethack_stairs_descent_native() -> None:
    env = make_nethack_env(seed=42, tier=1)
    obs, _ = env.reset(seed=42)
    agent = PureHCIRNetHackAgent()

    max_steps = 50
    success = False
    for _ in range(max_steps):
        act = agent.select_action(obs)
        obs, r, term, trunc, info = env.step(act)
        if term and r > 0:
            success = True
            break

    assert success, f"Failed to reach stairs in {obs.step_count} steps"


def test_native_nethack_wrapper() -> None:
    env = make_nethack_env(seed=42, tier=1)
    assert env.is_native is True
    assert isinstance(env, NativeNetHackWrapper)
    obs, info = env.reset(seed=42)
    assert obs is not None
    assert obs.step_count == 0
    assert obs.stats.hp > 0
