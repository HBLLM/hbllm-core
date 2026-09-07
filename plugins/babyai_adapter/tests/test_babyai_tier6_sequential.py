"""
Unit and adversarial tests for BabyAI Tier 6: Sequential & Compound Instruction Execution (Seq).

Verifies:
1. Multilingual parsing of sequential instructions ('then', 'and', 'after you' inversion).
2. Multi-phase execution: intermediate goal verification and sequential advancement.
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
    create_sequential_level,
)


def test_sequential_mission_parsing_multilingual() -> None:
    """Verify that compound sequential instructions parse and order subgoals accurately."""
    parser = BabyAIMissionParser()

    # English forward sequence ("then")
    g_then = parser.parse("open the red door, then pick up the green box")
    assert g_then.is_compound()
    assert len(g_then.subgoals) == 2
    assert g_then.subgoals[0].action == "open"
    assert g_then.subgoals[0].target_color == "red"
    assert g_then.subgoals[1].action == "pickup"
    assert g_then.subgoals[1].target_color == "green"

    # English inverted sequence ("after you")
    g_after = parser.parse("pick up the green box after you open the red door")
    assert g_after.is_compound()
    assert len(g_after.subgoals) == 2
    # Inverted: opening the door must be subgoals[0]!
    assert g_after.subgoals[0].action == "open"
    assert g_after.subgoals[0].target_color == "red"
    assert g_after.subgoals[1].action == "pickup"
    assert g_after.subgoals[1].target_color == "green"

    # Sinhala sequence
    g_si = parser.parse("රතු දොර අරින්න, පසුව කොළ පෙට්ටිය ගන්න")
    assert g_si.is_compound()
    assert len(g_si.subgoals) == 2
    assert g_si.subgoals[0].action == "open"
    assert g_si.subgoals[1].action == "pickup"

    # Tamil sequence
    g_ta = parser.parse("சிவப்பு கதவை திறக்கவும், பிறகு பச்சை பெட்டியை எடுக்கவும்")
    assert g_ta.is_compound()
    assert len(g_ta.subgoals) == 2
    assert g_ta.subgoals[0].action == "open"
    assert g_ta.subgoals[1].action == "pickup"


def test_hermetic_sequential_execution() -> None:
    """Verify end-to-end execution of a 2-phase sequential instruction."""
    env = create_sequential_level(
        mission="open the red door, then pick up the green box",
        door_pos=(4, 2),
        target_pos=(7, 2),
        agent_pos=(1, 1),
        room_width=9,
        room_height=5,
    )

    adapter = BabyAIPerceptionAdapter()
    planner = BabyAIActionAdapter(room_size=(9, 5))
    parser = BabyAIMissionParser()

    obs = env.gen_obs()
    for _ in range(4):
        adapter.ingest_observation(obs, known_agent_pos=env.agent_pos)
        obs, r, term, trunc, info = env.step(int(MiniGridAction.LEFT))
    adapter.ingest_observation(obs, known_agent_pos=env.agent_pos)

    goal = parser.parse(env.mission)

    success = False
    for step_num in range(50):
        adapter.ingest_observation(obs, known_agent_pos=env.agent_pos)
        act = planner.plan_next_action(adapter.graph, goal)
        obs, r, term, trunc, info = env.step(int(act))
        if term and r > 0.0:
            success = True
            break

    assert success, "Agent failed to complete multi-phase sequential mission"
