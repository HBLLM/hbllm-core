"""
Unit and adversarial tests for BabyAI Tier 3: Keys, Locked Doors, and Causal Prerequisite Subgoaling.

Verifies:
1. Multilingual parsing of unlock instructions (English, Sinhala, Tamil).
2. Hermetic physical unlock mechanics (matching key required, wrong key fails, empty hand fails).
3. Adversarial key selection with distractors (wrong-color keys, same-color wrong-type objects).
4. Full causal subgoaling execution (picking up key prerequisite, then unlocking door).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add core/ and core/plugins/ to sys.path
_test_dir = Path(__file__).resolve().parent
_plugin_dir = _test_dir.parent
_plugins_root = _plugin_dir.parent
_core_root = _plugins_root.parent

for p in [str(_core_root), str(_plugins_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from babyai_adapter import (
    BabyAIActionAdapter,
    BabyAIMissionParser,
    BabyAIPerceptionAdapter,
    MiniGridAction,
    MiniGridColor,
    MiniGridObjectType,
    MiniGridState,
    create_unlock_door_level,
)
from babyai_adapter.environment import GridCell


def test_unlock_mission_parsing_multilingual() -> None:
    """Verify that unlock instructions in English, Sinhala, and Tamil are parsed into typed goals."""
    parser = BabyAIMissionParser()

    # English
    g_en = parser.parse("unlock the red door")
    assert g_en.action == "open"
    assert g_en.target_type == "door"
    assert g_en.target_color == "red"

    g_en_any = parser.parse("unlock the door")
    assert g_en_any.action == "open"
    assert g_en_any.target_type == "door"
    assert g_en_any.target_color is None

    # Sinhala
    g_si = parser.parse("රතු දොර අගුළු අරින්න")
    assert g_si.action == "open"
    assert g_si.target_type == "door"
    assert g_si.target_color == "red"
    assert g_si.language == "si"

    # Tamil
    g_ta = parser.parse("சிவப்பு கதவை பூட்டு திறக்கவும்")
    assert g_ta.action == "open"
    assert g_ta.target_type == "door"
    assert g_ta.target_color == "red"
    assert g_ta.language == "ta"


def test_hermetic_unlock_mechanics() -> None:
    """Verify hermetic physical mechanics: empty hand fails, wrong key fails, matching key unlocks."""
    # Case A: Empty hand toggle fails
    env = create_unlock_door_level(
        mission="open the red door",
        door_color="red",
        door_pos=(4, 2),
        key_pos=(2, 2),
        agent_pos=(3, 2),
        agent_dir=0,  # facing East (towards door at 4, 2)
    )
    obs, r, term, trunc, info = env.step(int(MiniGridAction.TOGGLE))
    assert env.grid[4][2].state == MiniGridState.LOCKED
    assert not term
    assert r == 0.0

    # Case B: Carrying wrong color key (blue key) toggle fails
    env.carrying = GridCell(
        object_type=MiniGridObjectType.KEY,
        color=MiniGridColor.BLUE,
        state=MiniGridState.OPEN,
    )
    obs, r, term, trunc, info = env.step(int(MiniGridAction.TOGGLE))
    assert env.grid[4][2].state == MiniGridState.LOCKED
    assert not term
    assert r == 0.0

    # Case C: Carrying matching key (red key) toggle succeeds!
    env.carrying = GridCell(
        object_type=MiniGridObjectType.KEY,
        color=MiniGridColor.RED,
        state=MiniGridState.OPEN,
    )
    obs, r, term, trunc, info = env.step(int(MiniGridAction.TOGGLE))
    assert env.grid[4][2].state == MiniGridState.OPEN
    assert term
    assert r > 0.0


def test_adversarial_key_selection_with_distractor_keys() -> None:
    """Test adversarial setup: target is purple locked door.

    Environment contains:
    - purple key (matching)
    - yellow key (wrong color key distractor)
    - purple ball (same color wrong object distractor)
    - grey box (distractor)
    Agent must selectively pick up the purple key and unlock the purple door.
    """
    env = create_unlock_door_level(
        mission="open the purple door",
        door_color="purple",
        door_pos=(4, 2),
        key_pos=(2, 3),  # purple key
        distractors=[
            ("key", "yellow", (1, 3)),  # yellow key (distractor)
            ("ball", "purple", (3, 1)),  # purple ball (distractor)
            ("box", "grey", (1, 2)),  # grey box (distractor)
        ],
        agent_pos=(1, 1),
        agent_dir=0,
    )

    adapter = BabyAIPerceptionAdapter()
    planner = BabyAIActionAdapter(room_size=(9, 5))
    parser = BabyAIMissionParser()

    obs = env.gen_obs()
    # Map initial view
    for _ in range(4):
        adapter.ingest_observation(obs, known_agent_pos=env.agent_pos)
        obs, r, term, trunc, info = env.step(int(MiniGridAction.LEFT))
    adapter.ingest_observation(obs, known_agent_pos=env.agent_pos)

    goal = parser.parse(env.mission)
    trajectory = planner.plan_trajectory(adapter.graph, goal)

    assert MiniGridAction.PICKUP in trajectory, "Trajectory must include picking up the key"
    assert MiniGridAction.TOGGLE in trajectory, "Trajectory must include toggling the door"

    success = False
    for act in trajectory:
        obs, r, term, trunc, info = env.step(int(act))
        adapter.ingest_observation(obs, known_agent_pos=env.agent_pos)
        if term and r > 0.0:
            success = True
            break

    assert success, "Agent failed to solve adversarial unlock level"
    assert env.grid[4][2].state == MiniGridState.OPEN, "Purple door must be open"
    # Yellow key should still be on the ground at (1, 3)
    assert env.grid[1][3].object_type == MiniGridObjectType.KEY
    assert env.grid[1][3].color == MiniGridColor.YELLOW


def test_closed_loop_unlock_subgoaling() -> None:
    """Verify closed-loop step-by-step re-planning on an unlock task."""
    env = create_unlock_door_level(
        mission="unlock the red door",
        door_color="red",
        door_pos=(4, 2),
        key_pos=(2, 2),
        agent_pos=(1, 1),
        agent_dir=0,
    )

    adapter = BabyAIPerceptionAdapter()
    planner = BabyAIActionAdapter(room_size=(9, 5))
    parser = BabyAIMissionParser()

    goal = parser.parse(env.mission)

    # Initial scan
    obs = env.gen_obs()
    for _ in range(4):
        adapter.ingest_observation(obs, known_agent_pos=env.agent_pos)
        obs, r, term, trunc, info = env.step(int(MiniGridAction.LEFT))

    success = False
    for step_num in range(40):
        adapter.ingest_observation(obs, known_agent_pos=env.agent_pos)
        act = planner.plan_next_action(adapter.graph, goal)
        obs, r, term, trunc, info = env.step(int(act))
        if term and r > 0.0:
            success = True
            break

    assert success, "Closed-loop subgoaling failed to unlock red door within step budget"
    assert env.grid[4][2].state == MiniGridState.OPEN
