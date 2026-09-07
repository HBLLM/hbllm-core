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


def test_real_farama_babyai_unlock_pickup() -> None:
    """Test HCIR adapter on official Farama BabyAI-UnlockPickup-v0 (closed-loop prerequisite chaining)."""
    env = make_gym_babyai_level("BabyAI-UnlockPickup-v0")
    obs, info = env.reset(seed=5)

    adapter = BabyAIPerceptionAdapter()
    planner = BabyAIActionAdapter(room_size=16)
    parser = BabyAIMissionParser()

    goal = parser.parse(obs["mission"])

    success = False
    for _ in range(72):
        adapter.ingest_observation(obs, known_agent_pos=env.unwrapped.agent_pos)
        act = planner.plan_next_action(adapter.graph, goal)
        obs, r, term, trunc, info = env.step(int(act))
        if term and r > 0.0:
            success = True
            break

    env.close()
    assert success, (
        f"Failed to solve real BabyAI-UnlockPickup-v0 seed 5: mission='{goal.raw_instruction}'"
    )


def test_real_farama_babyai_key_corridor() -> None:
    """Test HCIR adapter on official Farama BabyAI-KeyCorridorS3R1-v0 (multi-room search and chaining)."""
    env = make_gym_babyai_level("BabyAI-KeyCorridorS3R1-v0")
    obs, info = env.reset(seed=42)

    adapter = BabyAIPerceptionAdapter()
    planner = BabyAIActionAdapter(room_size=10)
    parser = BabyAIMissionParser()

    goal = parser.parse(obs["mission"])

    success = False
    for _ in range(env.unwrapped.max_steps):
        adapter.ingest_observation(obs, known_agent_pos=env.unwrapped.agent_pos)
        act = planner.plan_next_action(adapter.graph, goal)
        obs, r, term, trunc, info = env.step(int(act))
        if term and r > 0.0:
            success = True
            break

    env.close()
    assert success, (
        f"Failed to solve real BabyAI-KeyCorridorS3R1-v0: mission='{goal.raw_instruction}'"
    )


def test_real_farama_babyai_put_next_local() -> None:
    """Test HCIR adapter on official Farama BabyAI-PutNextLocal-v0 (Tier 4 Relational PutNext)."""
    env = make_gym_babyai_level("BabyAI-PutNextLocal-v0")
    obs, info = env.reset(seed=42)

    adapter = BabyAIPerceptionAdapter()
    planner = BabyAIActionAdapter(room_size=8)
    parser = BabyAIMissionParser()

    goal = parser.parse(obs["mission"])

    for _ in range(4):
        adapter.ingest_observation(
            obs,
            known_agent_pos=env.unwrapped.agent_pos,
            known_carrying=env.unwrapped.carrying,
        )
        obs, r, term, trunc, info = env.step(int(MiniGridAction.LEFT))

    adapter.ingest_observation(
        obs,
        known_agent_pos=env.unwrapped.agent_pos,
        known_carrying=env.unwrapped.carrying,
    )

    success = False
    for _ in range(25):
        adapter.ingest_observation(
            obs,
            known_agent_pos=env.unwrapped.agent_pos,
            known_carrying=env.unwrapped.carrying,
        )
        act = planner.plan_next_action(adapter.graph, goal)
        obs, r, term, trunc, info = env.step(int(act))
        if term and r > 0.0:
            success = True
            break

    env.close()
    assert success, f"Failed to solve real BabyAI-PutNextLocal-v0: mission='{goal.raw_instruction}'"


def test_real_farama_babyai_blocked_unlock_pickup() -> None:
    """Test HCIR adapter on official Farama BabyAI-BlockedUnlockPickup-v0 (Tier 5 Causal Unblocking)."""
    env = make_gym_babyai_level("BabyAI-BlockedUnlockPickup-v0")
    obs, info = env.reset(seed=42)

    adapter = BabyAIPerceptionAdapter()
    planner = BabyAIActionAdapter(room_size=16)
    parser = BabyAIMissionParser()

    goal = parser.parse(obs["mission"])

    for _ in range(4):
        adapter.ingest_observation(
            obs,
            known_agent_pos=env.unwrapped.agent_pos,
            known_carrying=env.unwrapped.carrying,
        )
        obs, r, term, trunc, info = env.step(int(MiniGridAction.LEFT))

    adapter.ingest_observation(
        obs,
        known_agent_pos=env.unwrapped.agent_pos,
        known_carrying=env.unwrapped.carrying,
    )

    success = False
    for _ in range(40):
        adapter.ingest_observation(
            obs,
            known_agent_pos=env.unwrapped.agent_pos,
            known_carrying=env.unwrapped.carrying,
        )
        act = planner.plan_next_action(adapter.graph, goal)
        obs, r, term, trunc, info = env.step(int(act))
        if term and r > 0.0:
            success = True
            break

    env.close()
    assert success, (
        f"Failed to solve real BabyAI-BlockedUnlockPickup-v0: mission='{goal.raw_instruction}'"
    )


def test_real_farama_babyai_goto_seq() -> None:
    """Test HCIR adapter on official Farama BabyAI-GoToSeqS5R2-v0 (Tier 6 Sequential Instructions)."""
    env = make_gym_babyai_level("BabyAI-GoToSeqS5R2-v0")
    obs, info = env.reset(seed=42)

    adapter = BabyAIPerceptionAdapter()
    planner = BabyAIActionAdapter(room_size=10)
    parser = BabyAIMissionParser()

    goal = parser.parse(obs["mission"])

    for _ in range(4):
        adapter.ingest_observation(
            obs,
            known_agent_pos=env.unwrapped.agent_pos,
            known_carrying=env.unwrapped.carrying,
        )
        obs, r, term, trunc, info = env.step(int(MiniGridAction.LEFT))

    adapter.ingest_observation(
        obs,
        known_agent_pos=env.unwrapped.agent_pos,
        known_carrying=env.unwrapped.carrying,
    )

    success = False
    for _ in range(30):
        adapter.ingest_observation(
            obs,
            known_agent_pos=env.unwrapped.agent_pos,
            known_carrying=env.unwrapped.carrying,
        )
        act = planner.plan_next_action(adapter.graph, goal)
        obs, r, term, trunc, info = env.step(int(act))
        if term and r > 0.0:
            success = True
            break

    env.close()
    assert success, f"Failed to solve real BabyAI-GoToSeqS5R2-v0: mission='{goal.raw_instruction}'"


def test_real_farama_babyai_synth_loc() -> None:
    """Test HCIR adapter on official Farama BabyAI-SynthLoc-v0 (Tier 7 Synthesis)."""
    env = make_gym_babyai_level("BabyAI-SynthLoc-v0")
    obs, info = env.reset(seed=42)

    adapter = BabyAIPerceptionAdapter()
    planner = BabyAIActionAdapter(room_size=10)
    parser = BabyAIMissionParser()

    goal = parser.parse(obs["mission"])

    for _ in range(4):
        adapter.ingest_observation(
            obs,
            known_agent_pos=env.unwrapped.agent_pos,
            known_carrying=env.unwrapped.carrying,
        )
        obs, r, term, trunc, info = env.step(int(MiniGridAction.LEFT))

    adapter.ingest_observation(
        obs,
        known_agent_pos=env.unwrapped.agent_pos,
        known_carrying=env.unwrapped.carrying,
    )

    success = False
    for _ in range(30):
        adapter.ingest_observation(
            obs,
            known_agent_pos=env.unwrapped.agent_pos,
            known_carrying=env.unwrapped.carrying,
        )
        act = planner.plan_next_action(adapter.graph, goal)
        obs, r, term, trunc, info = env.step(int(act))
        if term and r > 0.0:
            success = True
            break

    env.close()
    assert success, f"Failed to solve real BabyAI-SynthLoc-v0: mission='{goal.raw_instruction}'"
