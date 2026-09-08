"""
Tests for Safety-Gymnasium Adapter Plugin.
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

from safety_gym_adapter import (
    PureHCIRSafetyAgent,
    SafetyGymAction,
    SafetyGymPerceptionAdapter,
    make_safety_gym_env,
    run_safety_gym_benchmark,
)


def test_safety_gym_environment_lifecycle() -> None:
    env = make_safety_gym_env(seed=42)
    obs, info = env.reset(seed=42)

    assert obs.agent_pos == (-2.0, -2.0)
    assert obs.goal_pos == (2.0, 2.0)
    assert len(obs.hazards) > 0
    assert len(obs.gremlins) > 0
    assert len(obs.lidar_distances) == 16
    assert obs.cumulative_cost == 0.0

    # Execute a step
    next_obs, reward, term, trunc, info = env.step(SafetyGymAction.FORWARD)
    assert not term
    assert next_obs.step_count == 1


def test_safety_gym_perception_adapter() -> None:
    env = make_safety_gym_env(seed=101)
    obs, _ = env.reset(seed=101)

    perception = SafetyGymPerceptionAdapter()
    graph = perception.ingest_observation(obs)

    agent_node = graph.get_node("agent")
    assert agent_node is not None
    assert agent_node.properties["current_cost"] == 0.0

    goal_node = graph.get_node("goal")
    assert goal_node is not None
    assert goal_node.properties["x"] == 2.0


def test_safety_gym_constrained_navigation() -> None:
    env = make_safety_gym_env(seed=77)
    obs, _ = env.reset(seed=77)
    agent = PureHCIRSafetyAgent()

    # Step agent toward goal
    max_steps = 100
    goal_reached = False
    for _ in range(max_steps):
        act = agent.select_action(obs)
        obs, r, term, trunc, info = env.step(act)
        if term:
            goal_reached = True
            break

    assert goal_reached, f"Failed to reach goal in {obs.step_count} steps"
    assert obs.cumulative_cost == 0.0, f"Incurred safety violation cost: {obs.cumulative_cost}"


def test_safety_gym_benchmark_smoke() -> None:
    data = run_safety_gym_benchmark("pure-hcir", episodes=3, base_seed=500)
    assert data["episodes"] == 3
    assert data["cohort"] == "pure-hcir"
    assert "ci_zero_95" in data
    assert len(data["results"]) == 3
