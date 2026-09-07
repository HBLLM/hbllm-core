"""
Unit and adversarial tests for BabyAI Tier 4: Relational Spatial Goal Planning (PutNext).

Verifies:
1. Multilingual mission parsing for PutNext (English, Sinhala, Tamil).
2. Hermetic relational placement execution (pickup -> transport -> drop adjacent).
3. Graph topological relation registration (HCIREdgeType.NEAR).
4. Adversarial distractor discrimination with swapped attribute confounders.
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
    create_put_next_level,
)

from hbllm.hcir.graph import HCIREdgeType


def test_putnext_mission_parsing_multilingual() -> None:
    """Verify that PutNext instructions in English, Sinhala, and Tamil parse accurately."""
    parser = BabyAIMissionParser()

    # English
    g_en = parser.parse("put the yellow key next to the yellow box")
    assert g_en.action == "put_next"
    assert g_en.target_type == "key"
    assert g_en.target_color == "yellow"
    assert g_en.fixed_type == "box"
    assert g_en.fixed_color == "yellow"

    # Sinhala
    g_si = parser.parse("කහ පෙට්ටිය ළඟින් කහ යතුර තියන්න")
    assert g_si.action == "put_next"
    assert g_si.target_type == "key"
    assert g_si.target_color == "yellow"
    assert g_si.fixed_type == "box"
    assert g_si.fixed_color == "yellow"
    assert g_si.language == "si"

    # Tamil
    g_ta = parser.parse("மஞ்சள் பெட்டியின் அருகில் மஞ்சள் சாவியை வைக்கவும்")
    assert g_ta.action == "put_next"
    assert g_ta.target_type == "key"
    assert g_ta.target_color == "yellow"
    assert g_ta.fixed_type == "box"
    assert g_ta.fixed_color == "yellow"
    assert g_ta.language == "ta"


def test_hermetic_putnext_execution() -> None:
    """Verify hermetic execution: pick up object, navigate to landmark, drop next to it."""
    env = create_put_next_level(
        mission="put the yellow key next to the yellow box",
        move_obj=("key", "yellow", (2, 2)),
        fixed_obj=("box", "yellow", (5, 4)),
        agent_pos=(1, 1),
        agent_dir=0,
        room_size=8,
    )

    adapter = BabyAIPerceptionAdapter()
    planner = BabyAIActionAdapter(room_size=8)
    parser = BabyAIMissionParser()

    # Map initial view
    obs = env.gen_obs()
    for _ in range(4):
        adapter.ingest_observation(obs, known_agent_pos=env.agent_pos)
        obs, r, term, trunc, info = env.step(int(MiniGridAction.LEFT))
    adapter.ingest_observation(obs, known_agent_pos=env.agent_pos)

    goal = parser.parse(env.mission)
    trajectory = planner.plan_trajectory(adapter.graph, goal)

    assert MiniGridAction.PICKUP in trajectory
    assert MiniGridAction.DROP in trajectory

    success = False
    for act in trajectory:
        obs, r, term, trunc, info = env.step(int(act))
        adapter.ingest_observation(obs, known_agent_pos=env.agent_pos)
        if term and r > 0.0:
            success = True
            break

    assert success, "Failed to execute hermetic PutNext task"

    # Verify that CognitiveGraph registers a NEAR edge between key and box
    near_edges = [e for e in adapter.graph.all_edges() if e.edge_type == HCIREdgeType.NEAR]
    key_box_near = any(
        ("key" in str(e.sources[0]) and "box" in str(e.targets[0]))
        or ("box" in str(e.sources[0]) and "key" in str(e.targets[0]))
        for e in near_edges
    )
    assert key_box_near, "CognitiveGraph must register a NEAR edge between key and box"


def test_adversarial_putnext_with_confounders() -> None:
    """Verify that PutNext accurately binds attributes under confounding distractors."""
    env = create_put_next_level(
        mission="put the purple ball next to the green box",
        move_obj=("ball", "purple", (2, 2)),
        fixed_obj=("box", "green", (5, 4)),
        distractors=[
            ("box", "purple", (3, 3)),  # purple box (same color as target, same type as landmark!)
            ("ball", "green", (1, 4)),  # green ball (same color as landmark, same type as target!)
            ("key", "yellow", (5, 2)),  # yellow key (unrelated distractor)
        ],
        agent_pos=(1, 1),
        agent_dir=0,
        room_size=8,
    )

    adapter = BabyAIPerceptionAdapter()
    planner = BabyAIActionAdapter(room_size=8)
    parser = BabyAIMissionParser()

    obs = env.gen_obs()
    for _ in range(4):
        adapter.ingest_observation(obs, known_agent_pos=env.agent_pos)
        obs, r, term, trunc, info = env.step(int(MiniGridAction.LEFT))
    adapter.ingest_observation(obs, known_agent_pos=env.agent_pos)

    goal = parser.parse(env.mission)
    trajectory = planner.plan_trajectory(adapter.graph, goal)

    success = False
    for act in trajectory:
        obs, r, term, trunc, info = env.step(int(act))
        adapter.ingest_observation(obs, known_agent_pos=env.agent_pos)
        if term and r > 0.0:
            success = True
            break

    assert success, "Agent failed to solve adversarial PutNext level"
