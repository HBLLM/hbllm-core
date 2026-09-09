"""
Tests for ALFWorld Adapter Plugin.
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

from alfworld_adapter import (
    ALFWorldGoal,
    ALFWorldPerceptionAdapter,
    ALFWorldTaskType,
    PureHCIRALFWorldAgent,
    make_alfworld_env,
    run_alfworld_benchmark,
)


def test_alfworld_environment_lifecycle() -> None:
    env = make_alfworld_env(task_type=ALFWorldTaskType.PICK_AND_PLACE, seed=42)
    obs, info = env.reset(seed=42)

    assert obs.text_obs is not None
    assert obs.current_location is not None
    assert len(obs.admissible_commands) > 0
    assert "look" in obs.admissible_commands

    # Step to a receptacle
    next_obs, reward, term, trunc, info = env.step("go to countertop 1")
    assert not term
    assert next_obs.current_location == "countertop 1"


def test_alfworld_mission_parser() -> None:
    perception = ALFWorldPerceptionAdapter()

    goal1 = perception.parse_instruction("clean soapbar with sinkbasin 1 and put on countertop 1")
    assert goal1.task_type == ALFWorldTaskType.CLEAN_AND_PLACE
    assert goal1.target_object_type == "soapbar"
    assert goal1.apparatus_receptacle_type == "sinkbasin 1"
    assert goal1.target_receptacle_type == "countertop 1"

    goal2 = perception.parse_instruction("heat apple with microwave 1 and put in desk 1")
    assert goal2.task_type == ALFWorldTaskType.HEAT_AND_PLACE
    assert goal2.target_object_type == "apple"
    assert goal2.apparatus_receptacle_type == "microwave 1"

    goal3 = perception.parse_instruction("put a sponge on countertop 1")
    assert goal3.task_type == ALFWorldTaskType.PICK_AND_PLACE
    assert goal3.target_object_type == "sponge"


def test_alfworld_perception_adapter() -> None:
    env = make_alfworld_env(task_type=ALFWorldTaskType.PICK_AND_PLACE, seed=10)
    obs, _ = env.reset(seed=10)

    perception = ALFWorldPerceptionAdapter()
    graph = perception.ingest_observation(obs)

    agent_node = graph.get_node("agent")
    assert agent_node is not None
    assert "current_location" in agent_node.properties


def test_alfworld_pure_hcir_pick_and_place() -> None:
    env = make_alfworld_env(task_type=ALFWorldTaskType.PICK_AND_PLACE, seed=99)
    obs, info = env.reset(seed=99)
    goal: ALFWorldGoal = info["goal"]

    agent = PureHCIRALFWorldAgent()

    # Step agent until success
    max_steps = 25
    success = False
    for _ in range(max_steps):
        act = agent.select_action(obs, goal)
        obs, r, term, trunc, info = env.step(act)
        if term and r > 0.0:
            success = True
            break

    assert success, f"Pure HCIR failed to complete pick_and_place in {obs.step_count} steps"


def test_alfworld_benchmark_smoke() -> None:
    data = run_alfworld_benchmark("pure-hcir", episodes=6, base_seed=100)
    assert data["episodes"] >= 6
    assert data["cohort"] == "pure-hcir"
    assert "ci_95" in data
    assert len(data["results"]) >= 6


def test_native_alfworld_wrapper_fallback() -> None:
    env = make_alfworld_env(task_type=ALFWorldTaskType.PICK_AND_PLACE, seed=42, prefer_native=True)
    assert env is not None
    obs, info = env.reset(seed=42)
    assert obs is not None
    assert obs.step_count == 0
