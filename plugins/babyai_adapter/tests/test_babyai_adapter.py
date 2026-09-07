"""
Unit tests for the BabyAI / MiniGrid Adapter Plugin.

Validates:
- Enums, mappings, and types
- Multilingual mission parsing (English, Sinhala, Tamil)
- Environment simulation (collision, line-of-sight, actions)
- Perception adapter graph translation
- Allocentric object persistence across field-of-view occlusion
- Action adapter path planning and task completion
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
    BabyAIEnvironment,
    BabyAIGoal,
    BabyAIMissionParser,
    BabyAIPerceptionAdapter,
    MiniGridAction,
    MiniGridColor,
    MiniGridDirection,
    MiniGridObjectType,
    create_babyai_level,
)

from hbllm.hcir.graph import EntityLifecycle, PhysicalEntityNode


def test_minigrid_enums_and_types() -> None:
    assert int(MiniGridObjectType.WALL) == 2
    assert int(MiniGridObjectType.BALL) == 6
    assert int(MiniGridColor.RED) == 0
    assert int(MiniGridColor.PURPLE) == 3
    assert int(MiniGridAction.FORWARD) == 2
    assert int(MiniGridDirection.EAST) == 0

    goal = BabyAIGoal(action="go_to", target_type="ball", target_color="red")
    assert goal.matches_attributes("ball", "red")
    assert not goal.matches_attributes("box", "red")
    assert not goal.matches_attributes("ball", "blue")


def test_mission_parser_english() -> None:
    parser = BabyAIMissionParser()

    g1 = parser.parse("go to the red ball")
    assert g1.action == "go_to"
    assert g1.target_type == "ball"
    assert g1.target_color == "red"

    g2 = parser.parse("pick up a yellow key")
    assert g2.action == "pickup"
    assert g2.target_type == "key"
    assert g2.target_color == "yellow"

    g3 = parser.parse("go to green box")
    assert g3.action == "go_to"
    assert g3.target_type == "box"
    assert g3.target_color == "green"


def test_mission_parser_multilingual() -> None:
    parser = BabyAIMissionParser()

    # Sinhala
    g_si = parser.parse("රතු බෝලය වෙත යන්න")
    assert g_si.action == "go_to"
    assert g_si.target_type == "ball"
    assert g_si.target_color == "red"
    assert g_si.language == "si"

    g_si_pick = parser.parse("කොළ පෙට්ටිය ගන්න")
    assert g_si_pick.action == "pickup"
    assert g_si_pick.target_type == "box"
    assert g_si_pick.target_color == "green"

    # Tamil
    g_ta = parser.parse("சிவப்பு பந்துக்கு செல்லவும்")
    assert g_ta.action == "go_to"
    assert g_ta.target_type == "ball"
    assert g_ta.target_color == "red"
    assert g_ta.language == "ta"

    g_ta_pick = parser.parse("பச்சை பெட்டியை எடுக்கவும்")
    assert g_ta_pick.action == "pickup"
    assert g_ta_pick.target_type == "box"
    assert g_ta_pick.target_color == "green"


def test_environment_simulation() -> None:
    env = BabyAIEnvironment(room_size=8, mission="go to the red ball")
    env.agent_pos = (1, 1)
    env.agent_dir = int(MiniGridDirection.NORTH)  # Facing wall at (1, 0)

    # Moving into wall should fail / keep agent at (1, 1)
    obs, reward, term, trunc, info = env.step(MiniGridAction.FORWARD)
    assert env.agent_pos == (1, 1)

    # Turn right (now facing East towards (2, 1))
    env.step(MiniGridAction.RIGHT)
    assert env.agent_dir == int(MiniGridDirection.EAST)

    # Move forward into empty tile (2, 1)
    env.step(MiniGridAction.FORWARD)
    assert env.agent_pos == (2, 1)


def test_perception_adapter_graph_generation() -> None:
    env = create_babyai_level(
        mission="go to the red ball",
        target=("ball", "red", (3, 1)),
        distractors=[("box", "blue", (1, 3))],
        agent_pos=(1, 1),
        agent_dir=int(MiniGridDirection.EAST),
    )
    obs = env.gen_obs()

    adapter = BabyAIPerceptionAdapter()
    graph = adapter.ingest_observation(obs)

    # Verify agent node
    assert graph.has_node("agent_primary")
    agent_node = graph.get_node("agent_primary")
    assert isinstance(agent_node, PhysicalEntityNode)
    assert agent_node.properties["coords"] == (1, 1)

    # Verify red ball entity node in graph
    entities = adapter.get_known_entities()
    ball_nodes = [e for e in entities if e.entity_type == "ball"]
    assert len(ball_nodes) == 1
    ball = ball_nodes[0]
    assert ball.properties["color"] == "red"
    assert ball.properties["coords"] == (3, 1)


def test_object_persistence_across_camera_turn() -> None:
    """Critical test: turning away from an object must retain it in CognitiveGraph."""
    env = create_babyai_level(
        mission="go to the red ball",
        target=("ball", "red", (3, 1)),
        distractors=[],
        agent_pos=(1, 1),
        agent_dir=int(MiniGridDirection.EAST),  # Facing East, looking directly at (3, 1)
    )
    adapter = BabyAIPerceptionAdapter()

    # Step 1: Observe ball in front
    obs1 = env.gen_obs()
    graph = adapter.ingest_observation(obs1)
    ball_id = [e.id for e in adapter.get_known_entities() if e.entity_type == "ball"][0]
    assert graph.has_node(ball_id)

    # Step 2: Turn 180 degrees to West (dir=2). The ball at (3, 1) is now behind agent
    env.step(MiniGridAction.LEFT)
    env.step(MiniGridAction.LEFT)
    assert env.agent_dir == int(MiniGridDirection.WEST)

    obs2 = env.gen_obs()
    # The agent is at (1, 1) facing West. Front is (0, 1). Behind is (2, 1), (3, 1).
    graph = adapter.ingest_observation(obs2)

    # The CognitiveGraph MUST still retain the red ball node with occluded lifecycle!
    assert graph.has_node(ball_id)
    persisted_node = graph.get_node(ball_id)
    assert isinstance(persisted_node, PhysicalEntityNode)
    assert persisted_node.entity_lifecycle == EntityLifecycle.OCCLUDED
    assert persisted_node.properties["color"] == "red"
    assert persisted_node.properties["coords"] == (3, 1)


def test_action_adapter_navigation() -> None:
    env = create_babyai_level(
        mission="go to the red ball",
        target=("ball", "red", (4, 1)),
        distractors=[("box", "blue", (2, 2))],
        agent_pos=(1, 1),
        agent_dir=int(MiniGridDirection.EAST),
    )
    adapter = BabyAIPerceptionAdapter()
    planner = BabyAIActionAdapter(room_size=8)
    parser = BabyAIMissionParser()

    # Ingest initial state
    obs = env.gen_obs()
    graph = adapter.ingest_observation(obs)
    goal = parser.parse(env.mission)

    trajectory = planner.plan_trajectory(graph, goal)
    assert len(trajectory) > 0

    # Execute planned trajectory in environment
    success = False
    for act in trajectory:
        obs, reward, term, trunc, info = env.step(act)
        if term and info["goal_achieved"]:
            success = True
            break

    assert success
