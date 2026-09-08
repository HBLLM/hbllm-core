"""
Tests for NetHack Adapter Plugin.
"""

from __future__ import annotations

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
    NetHackAction,
    NetHackPerceptionAdapter,
    PureHCIRNetHackAgent,
    make_nethack_env,
    run_nethack_benchmark,
)


def test_nethack_environment_lifecycle() -> None:
    env = make_nethack_env(seed=42)
    obs, info = env.reset(seed=42)

    assert obs.player_pos is not None
    assert obs.stats.hp == 15
    assert obs.stats.dungeon_level == 1
    assert len(obs.glyphs) == 21
    assert len(obs.glyphs[0]) == 79
    assert len(obs.chars) == 21

    # Execute a movement step
    next_obs, reward, term, trunc, info = env.step(NetHackAction.EAST)
    assert not term
    assert next_obs.step_count == 1


def test_nethack_perception_adapter() -> None:
    env = make_nethack_env(seed=101)
    obs, _ = env.reset(seed=101)

    perception = NetHackPerceptionAdapter()
    graph = perception.ingest_observation(obs)

    agent_node = graph.get_node("agent")
    assert agent_node is not None
    assert agent_node.properties["hp"] == 15
    assert agent_node.properties["dungeon_level"] == 1


def test_nethack_door_opening_and_stairs_descent() -> None:
    env = make_nethack_env(seed=77)
    obs, _ = env.reset(seed=77)
    agent = PureHCIRNetHackAgent()

    # Step agent until stairs are reached and descended
    max_steps = 100
    success = False
    for _ in range(max_steps):
        act = agent.select_action(obs)
        obs, r, term, trunc, info = env.step(act)
        if term and obs.stats.dungeon_level >= 2:
            success = True
            break

    assert success, (
        f"Failed to descend stairs in {obs.step_count} steps (lvl: {obs.stats.dungeon_level})"
    )


def test_nethack_benchmark_smoke() -> None:
    data = run_nethack_benchmark("pure-hcir", episodes=3, base_seed=500)
    assert data["episodes"] == 3
    assert data["cohort"] == "pure-hcir"
    assert "ci_95" in data
    assert len(data["results"]) == 3
