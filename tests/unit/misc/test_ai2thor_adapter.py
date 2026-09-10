"""Unit tests for AI2-THOR embodied adapter, dual-mode wrapper, and benchmark utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_plugins_dir = Path(__file__).resolve().parents[3] / "plugins"
if str(_plugins_dir) not in sys.path:
    sys.path.insert(0, str(_plugins_dir))

from ai2thor_adapter.action import AI2ThorActionAdapter
from ai2thor_adapter.benchmark import (
    resolve_cohort,
    wilson_score_interval,
)
from ai2thor_adapter.environment import (
    make_ai2thor_env,
)
from ai2thor_adapter.types import (
    AI2ThorActionType,
    AI2ThorAgentPose,
    AI2ThorGoal,
    AI2ThorObjectMetadata,
    AI2ThorObservation,
    AI2ThorVector3,
)


def test_cohort_resolution_valid():
    assert resolve_cohort("pure-hcir") == "pure-hcir"
    assert resolve_cohort("pure_hcir") == "pure-hcir"
    assert resolve_cohort("HBLLM-Core") == "pure-hcir"
    assert resolve_cohort("llm-only") == "llm-only"
    assert resolve_cohort("baseline") == "llm-only"


def test_cohort_resolution_invalid_fails_loud():
    with pytest.raises(ValueError, match="Unrecognized cohort 'unknown-cohort'"):
        resolve_cohort("unknown-cohort")


def test_wilson_score_interval():
    low, high = wilson_score_interval(10, 10)
    assert low > 0.6
    assert high == 1.0

    low, high = wilson_score_interval(0, 10)
    assert low == 0.0
    assert high < 0.4

    low, high = wilson_score_interval(0, 0)
    assert (low, high) == (0.0, 0.0)


def test_standalone_ai2thor_env_tiers():
    for tier in [1, 2, 3, 4]:
        env = make_ai2thor_env(tier=tier, prefer_native=False)
        obs, info = env.reset(seed=100)
        assert isinstance(obs, AI2ThorObservation)
        assert len(obs.objects) > 0
        assert info["goal"] is not None

        # Test action step
        obs, r, term, trunc, info = env.step(AI2ThorActionType.MOVE_AHEAD)
        assert isinstance(obs, AI2ThorObservation)
        assert obs.step_count == 1
        env.close()


def test_ai2thor_action_adapter_planning():
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


def test_ai2thor_require_native_when_available():
    env = make_ai2thor_env(tier=1, prefer_native=True, require_native=True)
    assert getattr(env, "is_native", False) is True
    obs, info = env.reset(seed=123)
    assert obs.step_count == 0
    assert len(obs.objects) > 0
    env.close()
