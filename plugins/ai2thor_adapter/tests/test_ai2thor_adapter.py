"""Tests for AI2-THOR Adapter Plugin, native wrapper, and benchmark utilities."""

from __future__ import annotations

import json
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
    PLUGIN_NAME,
    PLUGIN_VERSION,
    AI2ThorActionType,
    AI2ThorGoal,
    AI2ThorPerceptionAdapter,
    NativeAI2ThorWrapper,
    make_ai2thor_env,
)
from ai2thor_adapter.action import AI2ThorActionAdapter
from ai2thor_adapter.benchmark import resolve_cohort, wilson_score_interval
from ai2thor_adapter.types import (
    AI2ThorAgentPose,
    AI2ThorObjectMetadata,
    AI2ThorObservation,
    AI2ThorVector3,
)


def test_plugin_manifest() -> None:
    manifest_path = Path(__file__).resolve().parent.parent / "plugin.json"
    assert manifest_path.exists(), "plugin.json must exist"
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["name"] == PLUGIN_NAME
    assert data["version"] == PLUGIN_VERSION
    assert "ai2thor_observe" in data["capabilities"]
    assert data.get("supports_native_execution") is True
    assert "supports_standalone_fallback" not in data


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
    env = make_ai2thor_env(seed=42, tier=1)
    try:
        assert isinstance(env, NativeAI2ThorWrapper)
        obs, info = env.reset(seed=42)

        assert obs.agent_pose is not None
        assert len(obs.objects) > 0
        assert any("Apple" in o.objectType for o in obs.objects)

        # Execute a movement step
        next_obs, reward, term, trunc, info = env.step(AI2ThorActionType.MOVE_AHEAD)
        assert next_obs.step_count == 1
    finally:
        env.close()


def test_ai2thor_perception_adapter() -> None:
    env = make_ai2thor_env(seed=101, tier=1)
    try:
        obs, _ = env.reset(seed=101)

        perception = AI2ThorPerceptionAdapter()
        graph = perception.ingest_observation(obs)

        agent_node = graph.get_node("agent")
        assert agent_node is not None
        assert "rotation" in agent_node.properties
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


def test_native_ai2thor_wrapper_lifecycle() -> None:
    env = make_ai2thor_env(seed=42, tier=1)
    try:
        assert getattr(env, "is_native", False) is True
        assert isinstance(env, NativeAI2ThorWrapper)
        obs, info = env.reset(seed=42)
        assert obs is not None
        assert obs.step_count == 0
        assert len(obs.objects) > 0
        assert hasattr(obs.objects[0], "objectId")
        assert hasattr(obs.objects[0], "position")
    finally:
        env.close()
