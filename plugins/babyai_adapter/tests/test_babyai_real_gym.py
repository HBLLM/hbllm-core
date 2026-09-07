"""
Integration tests against the official upstream Farama Gymnasium & MiniGrid packages.

Verifies that the HBLLM perception and action adapters execute directly against
the real published 'BabyAI-GoToObj-v0' and 'BabyAI-PickupObj-v0' benchmark environments.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add core/ and core/plugins/ to sys.path
_test_dir = Path(__file__).resolve().parent
_plugin_dir = _test_dir.parent
_plugins_root = _plugin_dir.parent
_core_root = _plugins_root.parent

for p in [str(_core_root), str(_plugins_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import gymnasium  # noqa: F401
    import minigrid  # noqa: F401

    HAS_REAL_MINIGRID = True
except ImportError:
    HAS_REAL_MINIGRID = False

pytestmark = pytest.mark.skipif(
    not HAS_REAL_MINIGRID,
    reason="Requires upstream 'minigrid' and 'gymnasium' packages installed.",
)

from babyai_adapter import (
    BabyAIActionAdapter,
    BabyAIMissionParser,
    BabyAIPerceptionAdapter,
    MiniGridAction,
    make_gym_babyai_level,
)


def test_real_farama_babyai_goto_obj() -> None:
    """Test HCIR adapter on official Farama BabyAI-GoToObj-v0."""
    env = make_gym_babyai_level("BabyAI-GoToObj-v0")
    obs, info = env.reset(seed=123)

    adapter = BabyAIPerceptionAdapter()
    planner = BabyAIActionAdapter(room_size=8)
    parser = BabyAIMissionParser()

    mission = obs["mission"]
    goal = parser.parse(mission)

    # Initial 360 scan: agent takes in all angles to construct full allocentric room map
    for _ in range(4):
        adapter.ingest_observation(obs, known_agent_pos=env.unwrapped.agent_pos)
        obs, r, term, trunc, info = env.step(int(MiniGridAction.LEFT))

    adapter.ingest_observation(obs, known_agent_pos=env.unwrapped.agent_pos)

    # Resolve target from CognitiveGraph
    target_node = planner.find_target_entity(adapter.graph, goal)
    assert target_node is not None, f"Could not find target for goal {goal} in mapped room"

    # Plan trajectory
    trajectory = planner.plan_trajectory(adapter.graph, goal)
    assert len(trajectory) > 0

    success = False
    for act in trajectory:
        obs, r, term, trunc, info = env.step(int(act))
        adapter.ingest_observation(obs, known_agent_pos=env.unwrapped.agent_pos)
        if term and r > 0.0:
            success = True
            break

    env.close()
    assert success, "Failed to solve real BabyAI-GoToObj-v0 episode"


def test_real_farama_babyai_pickup_obj() -> None:
    """Test HCIR adapter on official Farama BabyAI-PickupDist-v0 (single-room pickup with distractors)."""
    env = make_gym_babyai_level("BabyAI-PickupDist-v0")
    obs, info = env.reset(seed=42)

    adapter = BabyAIPerceptionAdapter()
    planner = BabyAIActionAdapter(room_size=7)
    parser = BabyAIMissionParser()

    mission = obs["mission"]
    goal = parser.parse(mission)

    # Initial 360 scan
    for _ in range(4):
        adapter.ingest_observation(obs, known_agent_pos=env.unwrapped.agent_pos)
        obs, r, term, trunc, info = env.step(int(MiniGridAction.LEFT))

    adapter.ingest_observation(obs, known_agent_pos=env.unwrapped.agent_pos)

    target_node = planner.find_target_entity(adapter.graph, goal)
    assert target_node is not None, f"Could not find target for goal {goal} in mapped room"

    trajectory = planner.plan_trajectory(adapter.graph, goal)
    assert len(trajectory) > 0

    success = False
    for act in trajectory:
        obs, r, term, trunc, info = env.step(int(act))
        adapter.ingest_observation(obs, known_agent_pos=env.unwrapped.agent_pos)
        if term and r > 0.0:
            success = True
            break

    env.close()
    assert success, "Failed to solve real BabyAI-PickupDist-v0 episode"


def test_real_farama_babyai_open_red_door() -> None:
    """Test HCIR adapter on official Farama BabyAI-OpenRedDoor-v0 (9x5 partition room with red door)."""
    env = make_gym_babyai_level("BabyAI-OpenRedDoor-v0")
    obs, info = env.reset(seed=42)

    adapter = BabyAIPerceptionAdapter()
    planner = BabyAIActionAdapter(room_size=(9, 5))
    parser = BabyAIMissionParser()

    mission = obs["mission"]
    goal = parser.parse(mission)

    # Initial 360 scan
    for _ in range(4):
        adapter.ingest_observation(obs, known_agent_pos=env.unwrapped.agent_pos)
        obs, r, term, trunc, info = env.step(int(MiniGridAction.LEFT))

    adapter.ingest_observation(obs, known_agent_pos=env.unwrapped.agent_pos)

    target_node = planner.find_target_entity(adapter.graph, goal)
    assert target_node is not None, f"Could not find target for goal {goal} in mapped room"

    trajectory = planner.plan_trajectory(adapter.graph, goal)
    assert len(trajectory) > 0
    assert MiniGridAction.TOGGLE in trajectory

    success = False
    for act in trajectory:
        obs, r, term, trunc, info = env.step(int(act))
        adapter.ingest_observation(obs, known_agent_pos=env.unwrapped.agent_pos)
        if term and r > 0.0:
            success = True
            break

    env.close()
    assert success, "Failed to solve real BabyAI-OpenRedDoor-v0 episode"


def test_real_farama_babyai_unlock_local() -> None:
    """Test HCIR adapter on official Farama BabyAI-UnlockLocal-v0 (key-door prerequisite)."""
    env = make_gym_babyai_level("BabyAI-UnlockLocal-v0")
    obs, info = env.reset(seed=10)

    adapter = BabyAIPerceptionAdapter()
    planner = BabyAIActionAdapter(room_size=22)
    parser = BabyAIMissionParser()

    mission = obs["mission"]
    goal = parser.parse(mission)

    # Initial 360 scan
    for _ in range(4):
        adapter.ingest_observation(obs, known_agent_pos=env.unwrapped.agent_pos)
        obs, r, term, trunc, info = env.step(int(MiniGridAction.LEFT))

    adapter.ingest_observation(obs, known_agent_pos=env.unwrapped.agent_pos)

    target_node = planner.find_target_entity(adapter.graph, goal)
    assert target_node is not None, f"Could not find target door for goal {goal} in mapped room"

    trajectory = planner.plan_trajectory(adapter.graph, goal)
    assert len(trajectory) > 0
    assert MiniGridAction.PICKUP in trajectory, "Trajectory must include picking up the key"
    assert MiniGridAction.TOGGLE in trajectory, "Trajectory must include toggling the door"

    success = False
    for act in trajectory:
        obs, r, term, trunc, info = env.step(int(act))
        adapter.ingest_observation(obs, known_agent_pos=env.unwrapped.agent_pos)
        if term and r > 0.0:
            success = True
            break

    env.close()
    assert success, "Failed to solve real BabyAI-UnlockLocal-v0 episode"
