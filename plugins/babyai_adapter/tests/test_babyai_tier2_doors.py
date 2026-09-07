"""
Unit and Adversarial Verification Suite for BabyAI Tier 2: Doors and Multi-Room Navigation.

Tests:
1. Multilingual parsing of door-related instructions (English, Sinhala, Tamil).
2. Dynamic physical state transitions of doors (CLOSED -> OPEN via TOGGLE).
3. Adversarial door selection with confounding distractors (color vs. shape matches).
4. Multi-room traversal through closed partition doors.
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
    BabyAIGoal,
    BabyAIMissionParser,
    BabyAIPerceptionAdapter,
    MiniGridAction,
    MiniGridDirection,
    MiniGridState,
    create_two_room_door_level,
)


def test_door_mission_parsing() -> None:
    parser = BabyAIMissionParser()

    # English
    g1 = parser.parse("open the red door")
    assert g1.action == "open"
    assert g1.target_type == "door"
    assert g1.target_color == "red"

    g2 = parser.parse("open a green door")
    assert g2.action == "open"
    assert g2.target_type == "door"
    assert g2.target_color == "green"

    g3 = parser.parse("go to the door")
    assert g3.action == "go_to"
    assert g3.target_type == "door"
    assert g3.target_color is None

    # Sinhala
    g_si = parser.parse("රතු දොර අරින්න")
    assert g_si.action == "open"
    assert g_si.target_type == "door"
    assert g_si.target_color == "red"
    assert g_si.language == "si"

    # Tamil
    g_ta = parser.parse("சிவப்பு கதவை திறக்கவும்")
    assert g_ta.action == "open"
    assert g_ta.target_type == "door"
    assert g_ta.target_color == "red"
    assert g_ta.language == "ta"


def test_hermetic_door_open_mechanics() -> None:
    """Verify physical state change from CLOSED to OPEN using TOGGLE action."""
    env = create_two_room_door_level(
        mission="open the red door",
        door_color="red",
        door_pos=(4, 2),
        door_state=MiniGridState.CLOSED,
        room_width=9,
        room_height=5,
        agent_pos=(2, 2),
        agent_dir=int(MiniGridDirection.EAST),
    )

    adapter = BabyAIPerceptionAdapter()
    planner = BabyAIActionAdapter(room_size=(9, 5))
    parser = BabyAIMissionParser()

    obs = env.gen_obs()
    goal = parser.parse(env.mission)

    # Initial scan to observe door
    for _ in range(4):
        adapter.ingest_observation(obs, known_agent_pos=env.agent_pos)
        obs, r, term, trunc, info = env.step(int(MiniGridAction.LEFT))

    adapter.ingest_observation(obs, known_agent_pos=env.agent_pos)

    # Locate door in CognitiveGraph
    door_node = planner.find_target_entity(adapter.graph, goal)
    assert door_node is not None
    assert door_node.entity_type == "door"
    assert door_node.properties["color"] == "red"
    assert door_node.properties["state"] == "closed"
    assert door_node.properties["passable"] is False

    # Plan trajectory to open door
    trajectory = planner.plan_trajectory(adapter.graph, goal)
    assert len(trajectory) > 0
    assert MiniGridAction.TOGGLE in trajectory

    # Execute trajectory
    success = False
    for act in trajectory:
        obs, r, term, trunc, info = env.step(act)
        adapter.ingest_observation(obs, known_agent_pos=info["agent_pos"])
        if term and info["goal_achieved"]:
            success = True
            break

    assert success, "Agent failed to open the red door"
    # Verify environment cell state updated to OPEN
    assert env.grid[4][2].state == MiniGridState.OPEN
    # Verify CognitiveGraph node updated to OPEN and passable
    updated_door = adapter.graph.get_node(door_node.id)
    assert updated_door.properties["state"] == "open"
    assert updated_door.properties["passable"] is True


def test_adversarial_door_selection_with_confounders() -> None:
    """Adversarial door selection: 'open the purple door' amidst confounders.

    Confounders:
    - Red door at (4, 1): wrong color, same type
    - Purple box at (2, 3): same color, wrong type
    - Yellow key at (1, 1): unrelated
    """
    mission = "open the purple door"
    env = create_two_room_door_level(
        mission=mission,
        door_color="purple",
        door_pos=(4, 2),
        door_state=MiniGridState.CLOSED,
        room_width=9,
        room_height=5,
        agent_pos=(2, 2),
        agent_dir=int(MiniGridDirection.EAST),
        distractors=[
            ("door", "red", (4, 1)),
            ("box", "purple", (2, 3)),
            ("key", "yellow", (1, 1)),
        ],
    )

    adapter = BabyAIPerceptionAdapter()
    planner = BabyAIActionAdapter(room_size=(9, 5))
    parser = BabyAIMissionParser()

    obs = env.gen_obs()
    goal = parser.parse(mission)

    for _ in range(4):
        adapter.ingest_observation(obs, known_agent_pos=env.agent_pos)
        obs, r, term, trunc, info = env.step(int(MiniGridAction.LEFT))

    adapter.ingest_observation(obs, known_agent_pos=env.agent_pos)

    # Candidate resolution
    target_node = planner.find_target_entity(adapter.graph, goal)
    assert target_node is not None
    assert target_node.entity_type == "door"
    assert target_node.properties["color"] == "purple"
    assert target_node.properties["coords"] == (4, 2)

    # Strictly reject distractors
    assert target_node.properties["coords"] != (4, 1), "Incorrectly targeted red door distractor!"
    assert target_node.properties["coords"] != (2, 3), "Incorrectly targeted purple box distractor!"

    trajectory = planner.plan_trajectory(adapter.graph, goal)
    success = False
    for act in trajectory:
        obs, r, term, trunc, info = env.step(act)
        adapter.ingest_observation(obs, known_agent_pos=info["agent_pos"])
        if term and info["goal_achieved"]:
            success = True
            break

    assert success, "Failed to navigate to and open the purple door"
    assert env.grid[4][2].state == MiniGridState.OPEN
    # Red door remains closed
    assert env.grid[4][1].state == MiniGridState.CLOSED


def test_multi_room_traversal_through_closed_door() -> None:
    """Multi-room test: navigate to a green box in Room 2 by traversing through a closed red door."""
    mission = "go to the green box"
    env = create_two_room_door_level(
        mission=mission,
        door_color="red",
        door_pos=(4, 2),
        door_state=MiniGridState.CLOSED,
        room_width=9,
        room_height=5,
        agent_pos=(1, 2),
        agent_dir=int(MiniGridDirection.EAST),
        target_in_room2=("box", "green", (7, 2)),
        distractors=[("ball", "blue", (1, 1))],
    )

    adapter = BabyAIPerceptionAdapter()
    planner = BabyAIActionAdapter(room_size=(9, 5))
    parser = BabyAIMissionParser()

    obs = env.gen_obs()
    goal = parser.parse(mission)

    # Agent in Room 1 scans room
    for _ in range(4):
        adapter.ingest_observation(obs, known_agent_pos=env.agent_pos)
        obs, r, term, trunc, info = env.step(int(MiniGridAction.LEFT))

    adapter.ingest_observation(obs, known_agent_pos=env.agent_pos)

    # Step-by-step agent loop:
    # 1. Target (green box in Room 2) is not yet in sight.
    # 2. Agent detects closed door at (4, 2), paths to it, opens it, and enters Room 2.
    # 3. Upon entering Room 2, target is spotted and agent navigates to it.
    success = False
    max_agent_steps = 30

    for step_i in range(max_agent_steps):
        target = planner.find_target_entity(adapter.graph, goal)
        if target:
            # Target is spotted: plan directly to green box
            act = planner.plan_next_action(adapter.graph, goal)
        else:
            # Target not yet spotted: plan through closed door to explore
            closed_door = planner.find_closed_door(adapter.graph)
            if closed_door:
                door_goal = BabyAIGoal(
                    action="open",
                    target_type="door",
                    target_color=closed_door.properties.get("color"),
                )
                act = planner.plan_next_action(adapter.graph, door_goal)
            else:
                act = MiniGridAction.LEFT

        obs, r, term, trunc, info = env.step(int(act))
        adapter.ingest_observation(obs, known_agent_pos=info["agent_pos"])

        if term and info["goal_achieved"]:
            success = True
            break

    assert success, "Agent failed to traverse through closed door and reach green box in Room 2"
    # Agent must be in Room 2 (x > 4)
    assert env.agent_pos[0] >= 5, f"Agent ended in room 1 at {env.agent_pos}"
