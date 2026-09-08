"""
Tests for MineDojo Adapter Plugin.
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

from minedojo_adapter import (
    MineDojoAction,
    MineDojoGoal,
    MineDojoPerceptionAdapter,
    PureHCIRMineDojoAgent,
    make_minedojo_env,
    run_minedojo_benchmark,
)


def test_minedojo_environment_lifecycle() -> None:
    env = make_minedojo_env(seed=42)
    obs, info = env.reset(seed=42)

    assert obs.player_pos == (16, 16, 6)
    assert len(obs.voxels) > 0
    assert obs.inventory.log == 0

    # Execute a movement step
    next_obs, reward, term, trunc, info = env.step(MineDojoAction.MOVE_FORWARD)
    assert not term
    assert next_obs.step_count == 1


def test_minedojo_perception_adapter() -> None:
    env = make_minedojo_env(seed=101)
    obs, _ = env.reset(seed=101)

    perception = MineDojoPerceptionAdapter()
    graph = perception.ingest_observation(obs)

    agent_node = graph.get_node("agent")
    assert agent_node is not None
    assert agent_node.properties["x"] == 16


def test_minedojo_causal_recipe_dag() -> None:
    env = make_minedojo_env(seed=77)
    obs, info = env.reset(seed=77)
    goal: MineDojoGoal = info["goal"]
    agent = PureHCIRMineDojoAgent()

    # Step agent until wooden pickaxe is crafted
    max_steps = 30
    success = False
    for _ in range(max_steps):
        act = agent.select_action(obs, goal)
        obs, r, term, trunc, info = env.step(act)
        if term and obs.inventory.wooden_pickaxe > 0:
            success = True
            break

    assert success, (
        f"Failed to synthesize wooden pickaxe in {obs.step_count} steps (inv: {obs.inventory.to_dict()})"
    )


def test_minedojo_benchmark_smoke() -> None:
    data = run_minedojo_benchmark("pure-hcir", episodes=3, base_seed=500)
    assert data["episodes"] == 3
    assert data["cohort"] == "pure-hcir"
    assert "ci_95" in data
    assert len(data["results"]) == 3
