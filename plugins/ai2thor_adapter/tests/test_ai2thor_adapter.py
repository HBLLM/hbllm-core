"""Tests for AI2-THOR Adapter Plugin, dual-mode wrapper, and benchmark utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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
from ai2thor_adapter.action import AI2ThorActionAdapter
from ai2thor_adapter.benchmark import resolve_cohort, wilson_score_interval
from ai2thor_adapter.types import (
    AI2ThorAgentPose,
    AI2ThorObjectMetadata,
    AI2ThorObservation,
    AI2ThorVector3,
)


def test_cohort_resolution_valid() -> None:
    assert resolve_cohort("pure-hcir") == "pure-hcir"
    assert resolve_cohort("pure_hcir") == "pure-hcir"
    assert resolve_cohort("HBLLM-Core") == "pure-hcir"
    assert resolve_cohort("llm-only") == "llm-only"
    assert resolve_cohort("baseline") == "llm-only"


def test_cohort_resolution_invalid_fails_loud() -> None:
    with pytest.raises(ValueError, match="Unrecognized cohort 'unknown-cohort'"):
        resolve_cohort("unknown-cohort")


def test_wilson_score_interval() -> None:
    low, high = wilson_score_interval(10, 10)
    assert low > 0.6
    assert high == 1.0

    low, high = wilson_score_interval(0, 10)
    assert low == 0.0
    assert high < 0.4

    low, high = wilson_score_interval(0, 0)
    assert (low, high) == (0.0, 0.0)


def test_ai2thor_environment_lifecycle() -> None:
    env = make_ai2thor_env(seed=42)
    try:
        obs, info = env.reset(seed=42)

        assert obs.agent_pose.position.x == 0.0
        assert len(obs.objects) > 0
        assert "Mug_1" in [o.objectId for o in obs.objects]
        assert "Microwave_1" in [o.objectId for o in obs.objects]

        # Execute a movement step
        next_obs, reward, term, trunc, info = env.step(AI2ThorActionType.MOVE_AHEAD)
        assert not term
        assert next_obs.step_count == 1
    finally:
        env.close()


def test_standalone_ai2thor_env_tiers() -> None:
    for tier in [1, 2, 3, 4]:
        env = make_ai2thor_env(tier=tier, prefer_native=False)
        try:
            obs, info = env.reset(seed=100)
            assert isinstance(obs, AI2ThorObservation)
            assert len(obs.objects) > 0
            assert info["goal"] is not None

            # Test action step
            obs, r, term, trunc, info = env.step(AI2ThorActionType.MOVE_AHEAD)
            assert isinstance(obs, AI2ThorObservation)
            assert obs.step_count == 1
        finally:
            env.close()


def test_ai2thor_perception_adapter() -> None:
    env = make_ai2thor_env(seed=101)
    try:
        obs, _ = env.reset(seed=101)

        perception = AI2ThorPerceptionAdapter()
        graph = perception.ingest_observation(obs)

        agent_node = graph.get_node("agent")
        assert agent_node is not None
        assert agent_node.properties["rotation"] == 0.0

        mug_node = graph.get_node("Mug_1")
        assert mug_node is not None
        assert mug_node.properties["is_pickupable"] is True
    finally:
        env.close()


def test_ai2thor_action_adapter_planning() -> None:
    adapter = AI2ThorActionAdapter()
    obs = AI2ThorObservation(
        agent_pose=AI2ThorAgentPose(
            position=AI2ThorVector3(0.0, 0.9, 0.0),
            rotation=0.0,
            horizon=0.0,
        ),
        objects=[
            AI2ThorObjectMetadata(
                objectId="Apple_1",
                objectType="Apple",
                position=AI2ThorVector3(0.0, 0.9, 3.0),
                distance=3.0,
                isInteractable=True,
                isPickupable=True,
            )
        ],
        held_object_id=None,
        last_action_success=True,
        last_action_error="",
        step_count=0,
    )
    goal = AI2ThorGoal(
        target_object_id="Apple_1",
        target_receptacle_id=None,
        raw_instruction="Pick up Apple_1",
    )

    action = adapter.plan_next_action(obs, goal)
    assert action == AI2ThorActionType.MOVE_AHEAD


def test_ai2thor_3d_pick_and_place() -> None:
    env = make_ai2thor_env(seed=77)
    try:
        obs, info = env.reset(seed=77)
        goal: AI2ThorGoal = info["goal"]
        agent = PureHCIRAI2ThorAgent()

        max_steps = 30
        success = False
        for _ in range(max_steps):
            act = agent.select_action(obs, goal)
            obs, r, term, trunc, info = env.step(act)
            if term and r > 0.0:
                success = True
                break

        assert success, f"Failed to complete AI2-THOR pick and place in {obs.step_count} steps"
    finally:
        env.close()


def test_ai2thor_benchmark_smoke() -> None:
    data = run_ai2thor_benchmark("pure-hcir", episodes=4, base_seed=500, prefer_native=False)
    assert data["episodes"] >= 4
    assert data["cohort"] == "pure-hcir"
    assert "ci_95" in data
    assert len(data["results"]) >= 4


def test_native_ai2thor_wrapper_lifecycle() -> None:
    env = make_ai2thor_env(seed=42, tier=1, prefer_native=True, require_native=True)
    try:
        assert getattr(env, "is_native", False) is True
        obs, info = env.reset(seed=42)
        assert obs is not None
        assert obs.step_count == 0
        # Verify observation carries typed 3D scene-graph metadata
        assert len(obs.objects) > 0
        assert hasattr(obs.objects[0], "objectId")
        assert hasattr(obs.objects[0], "position")
    finally:
        env.close()
