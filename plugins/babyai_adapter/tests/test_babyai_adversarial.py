"""
Adversarial Compositional Generalization Verification Suite for BabyAI Plugin.

Tests that the HCIR perception and action stack solves held-out (unseen)
attribute bindings (color x object x location combinations never seen in single episodes)
in the presence of confounding distractors that share single attributes (color-only or shape-only matches).
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
    MiniGridColor,
    MiniGridDirection,
    MiniGridObjectType,
    create_babyai_level,
)


def test_adversarial_held_out_goto_with_confounding_distractors() -> None:
    """Adversarial Test 1: GoTo held-out (purple, ball).

    Environment contains:
    - Target: purple ball at (5, 3)  [HELD-OUT COMBINATION]
    - Distractor 1: purple box at (3, 2)  [Color matches, shape wrong]
    - Distractor 2: red ball at (2, 4)    [Shape matches, color wrong]
    - Distractor 3: yellow key at (1, 5)  [Completely unrelated]
    """
    mission = "go to the purple ball"
    env = create_babyai_level(
        mission=mission,
        target=("ball", "purple", (5, 3)),
        distractors=[
            ("box", "purple", (3, 2)),
            ("ball", "red", (2, 4)),
            ("key", "yellow", (1, 5)),
        ],
        agent_pos=(1, 1),
        agent_dir=int(MiniGridDirection.EAST),
        room_size=8,
    )

    adapter = BabyAIPerceptionAdapter()
    planner = BabyAIActionAdapter(room_size=8)
    parser = BabyAIMissionParser()

    # Ingest perception
    obs = env.gen_obs()
    graph = adapter.ingest_observation(obs)

    # Parse mission
    goal = parser.parse(mission)
    assert goal.target_type == "ball"
    assert goal.target_color == "purple"

    # Verify candidate resolution against CognitiveGraph
    target_node = planner.find_target_entity(graph, goal)
    assert target_node is not None, "Failed to locate target entity in CognitiveGraph"
    assert target_node.entity_type == "ball"
    assert target_node.properties["color"] == "purple"
    assert target_node.properties["coords"] == (5, 3)

    # Verify that single-attribute distractors are strictly NOT matched as target
    assert target_node.properties["coords"] != (3, 2), "Incorrectly matched purple box distractor!"
    assert target_node.properties["coords"] != (2, 4), "Incorrectly matched red ball distractor!"

    # Plan trajectory
    trajectory = planner.plan_trajectory(graph, goal)
    assert len(trajectory) > 0

    # Step through trajectory and execute
    success = False
    for step_num, act in enumerate(trajectory):
        obs, reward, term, trunc, info = env.step(act)
        # Update perception dynamically
        adapter.ingest_observation(obs, known_agent_pos=info["agent_pos"])
        if term and info["goal_achieved"]:
            success = True
            break

    assert success, "Agent failed to reach and face the held-out purple ball"
    # Verify agent final position is adjacent to target (5, 3)
    fx, fy = env.get_front_pos()
    assert (fx, fy) == (5, 3), f"Agent ended facing {(fx, fy)} instead of target (5, 3)"


def test_adversarial_held_out_pickup_with_confounding_distractors() -> None:
    """Adversarial Test 2: Pickup held-out (grey, key).

    Environment contains:
    - Target: grey key at (4, 4)         [HELD-OUT COMBINATION]
    - Distractor 1: grey ball at (2, 3)  [Color matches, shape wrong]
    - Distractor 2: green key at (3, 2)  [Shape matches, color wrong]
    """
    mission = "pick up the grey key"
    env = create_babyai_level(
        mission=mission,
        target=("key", "grey", (4, 4)),
        distractors=[
            ("ball", "grey", (2, 3)),
            ("key", "green", (3, 2)),
        ],
        agent_pos=(1, 1),
        agent_dir=int(MiniGridDirection.EAST),
        room_size=8,
    )

    adapter = BabyAIPerceptionAdapter()
    planner = BabyAIActionAdapter(room_size=8)
    parser = BabyAIMissionParser()

    obs = env.gen_obs()
    graph = adapter.ingest_observation(obs)
    goal = parser.parse(mission)

    target_node = planner.find_target_entity(graph, goal)
    assert target_node is not None
    assert target_node.entity_type == "key"
    assert target_node.properties["color"] == "grey"
    assert target_node.properties["coords"] == (4, 4)

    trajectory = planner.plan_trajectory(graph, goal)
    assert len(trajectory) > 0

    success = False
    for act in trajectory:
        obs, reward, term, trunc, info = env.step(act)
        adapter.ingest_observation(obs, known_agent_pos=info["agent_pos"])
        if term and info["goal_achieved"]:
            success = True
            break

    assert success, "Agent failed to pick up the held-out grey key"
    assert env.carrying is not None
    assert int(env.carrying.object_type) == int(MiniGridObjectType.KEY)
    assert int(env.carrying.color) == int(MiniGridColor.GREY)


def test_adversarial_multilingual_held_out_combination() -> None:
    """Adversarial Test 3: Multilingual instruction with held-out attribute binding.

    Sinhala: "දම් බෝලය වෙත යන්න" -> Go to the purple ball
    Target: purple ball at (5, 3)
    Distractors: purple box at (3, 2), red ball at (2, 4)
    """
    sinhala_mission = "දම් බෝලය වෙත යන්න"
    env = create_babyai_level(
        mission="go to the purple ball",  # Internal env goal check uses canonical English
        target=("ball", "purple", (5, 3)),
        distractors=[
            ("box", "purple", (3, 2)),
            ("ball", "red", (2, 4)),
        ],
        agent_pos=(1, 1),
        agent_dir=int(MiniGridDirection.EAST),
        room_size=8,
    )

    adapter = BabyAIPerceptionAdapter()
    planner = BabyAIActionAdapter(room_size=8)
    parser = BabyAIMissionParser()

    obs = env.gen_obs()
    graph = adapter.ingest_observation(obs)

    # Parse Sinhala mission
    goal = parser.parse(sinhala_mission)
    assert goal.language == "si"
    assert goal.target_type == "ball"
    assert goal.target_color == "purple"

    target_node = planner.find_target_entity(graph, goal)
    assert target_node is not None
    assert target_node.properties["coords"] == (5, 3)

    trajectory = planner.plan_trajectory(graph, goal)
    success = False
    for act in trajectory:
        obs, reward, term, trunc, info = env.step(act)
        adapter.ingest_observation(obs, known_agent_pos=info["agent_pos"])
        if term and info["goal_achieved"]:
            success = True
            break

    assert success
