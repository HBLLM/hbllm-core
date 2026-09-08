"""
Tests for Crafter Adapter Plugin.
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

from crafter_adapter import (
    CrafterAchievement,
    CrafterAction,
    CrafterActionAdapter,
    CrafterObject,
    CrafterPerceptionAdapter,
    PureHCIRCrafterAgent,
    make_crafter_env,
    run_crafter_benchmark,
)


def test_crafter_environment_lifecycle() -> None:
    env = make_crafter_env(seed=42)
    obs, info = env.reset(seed=42)

    assert obs.player_pos is not None
    assert obs.vitals.health == 9
    assert obs.vitals.food == 9
    assert obs.vitals.drink == 9
    assert obs.vitals.energy == 9
    assert len(obs.semantic_grid) == 64
    assert len(obs.semantic_grid[0]) == 64

    # Perform a movement step
    next_obs, reward, term, trunc, info = env.step(CrafterAction.MOVE_RIGHT)
    assert not term
    assert next_obs.step_count == 1


def test_crafter_perception_adapter() -> None:
    env = make_crafter_env(seed=101)
    obs, _ = env.reset(seed=101)

    perception = CrafterPerceptionAdapter(width=64, height=64)
    graph = perception.ingest_observation(obs)

    agent_node = graph.get_node("agent")
    assert agent_node is not None
    assert agent_node.properties["health"] == 9
    assert "inventory" in agent_node.properties

    # Check nearest object finder
    tree_pos = perception.find_nearest_object(obs, CrafterObject.TREE)
    assert tree_pos is not None
    tx, ty = tree_pos
    assert obs.semantic_grid[ty][tx] == CrafterObject.TREE


def test_crafter_vital_interrupts() -> None:
    env = make_crafter_env(seed=77)
    obs, _ = env.reset(seed=77)

    action_adapter = CrafterActionAdapter()

    # When energy is depleted, action must be SLEEP
    obs.vitals.energy = 1
    act = action_adapter.plan_next_action(obs)
    assert act == CrafterAction.SLEEP


def test_crafter_wood_collection() -> None:
    env = make_crafter_env(seed=12)
    obs, _ = env.reset(seed=12)
    agent = PureHCIRCrafterAgent()

    # Step agent until wood is collected
    max_steps = 100
    for _ in range(max_steps):
        act = agent.select_action(obs)
        obs, r, term, trunc, info = env.step(act)
        if CrafterAchievement.COLLECT_WOOD in obs.achievements:
            break

    assert CrafterAchievement.COLLECT_WOOD in obs.achievements
    assert obs.inventory.wood >= 1


def test_crafter_benchmark_smoke() -> None:
    data = run_crafter_benchmark("pure-hcir", episodes=3, base_seed=500)
    assert data["episodes"] == 3
    assert data["cohort"] == "pure-hcir"
    assert "ci_95" in data
    assert len(data["results"]) == 3
