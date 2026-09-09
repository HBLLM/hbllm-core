"""
Tests for AI2-THOR Adapter Plugin.
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

from ai2thor_adapter import (
    AI2ThorActionType,
    AI2ThorGoal,
    AI2ThorPerceptionAdapter,
    PureHCIRAI2ThorAgent,
    make_ai2thor_env,
    run_ai2thor_benchmark,
)


def test_ai2thor_environment_lifecycle() -> None:
    env = make_ai2thor_env(seed=42)
    obs, info = env.reset(seed=42)

    assert obs.agent_pose.position.x == 0.0
    assert len(obs.objects) > 0
    assert "Mug_1" in [o.objectId for o in obs.objects]
    assert "Microwave_1" in [o.objectId for o in obs.objects]

    # Execute a movement step
    next_obs, reward, term, trunc, info = env.step(AI2ThorActionType.MOVE_AHEAD)
    assert not term
    assert next_obs.step_count == 1


def test_ai2thor_perception_adapter() -> None:
    env = make_ai2thor_env(seed=101)
    obs, _ = env.reset(seed=101)

    perception = AI2ThorPerceptionAdapter()
    graph = perception.ingest_observation(obs)

    agent_node = graph.get_node("agent")
    assert agent_node is not None
    assert agent_node.properties["rotation"] == 0.0

    mug_node = graph.get_node("Mug_1")
    assert mug_node is not None
    assert mug_node.properties["is_pickupable"] is True


def test_ai2thor_3d_pick_and_place() -> None:
    env = make_ai2thor_env(seed=77)
    obs, info = env.reset(seed=77)
    goal: AI2ThorGoal = info["goal"]
    agent = PureHCIRAI2ThorAgent()

    # Step agent until task is completed
    max_steps = 30
    success = False
    for _ in range(max_steps):
        act = agent.select_action(obs, goal)
        obs, r, term, trunc, info = env.step(act)
        if term and r > 0.0:
            success = True
            break

    assert success, f"Failed to complete AI2-THOR pick and place in {obs.step_count} steps"


def test_ai2thor_benchmark_smoke() -> None:
    data = run_ai2thor_benchmark("pure-hcir", episodes=4, base_seed=500)
    assert data["episodes"] >= 4
    assert data["cohort"] == "pure-hcir"
    assert "ci_95" in data
    assert len(data["results"]) >= 4


def test_native_ai2thor_wrapper_fallback_and_channels() -> None:
    env = make_ai2thor_env(seed=42, tier=1, prefer_native=True)
    assert env is not None
    obs, info = env.reset(seed=42)
    assert obs is not None
    assert obs.step_count == 0
    # Verify observation carries typed 3D scene-graph metadata
    assert len(obs.objects) > 0
    assert hasattr(obs.objects[0], "objectId")
    assert hasattr(obs.objects[0], "position")
