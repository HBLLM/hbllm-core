"""Unit tests for Stage D5: Tool Use & Compositional Causal Chains Engine."""

from __future__ import annotations

from plugins.developmental_adapter.blank_brain import create_blank_brain_substrate
from plugins.developmental_adapter.environment import BabyWorldEnvironment
from plugins.developmental_adapter.perception import DevelopmentalPerceptionAdapter
from plugins.developmental_adapter.tool_learning import ToolLearningEngine


def test_tool_use_direct_manipulation_fails():
    """Verify that distant target cannot be reached without an intermediate tool."""
    env = BabyWorldEnvironment(seed=42)
    env.reset("tool_use_world")

    substrate = create_blank_brain_substrate()
    perception = DevelopmentalPerceptionAdapter()
    engine = ToolLearningEngine(substrate, perception, env)

    res = engine.attempt_direct_manipulation("obj_distant_reward")
    assert res["direct_grasp_success"] is False
    assert res["direct_pull_success"] is False
    assert res["out_of_reach"] is True


def test_tool_discovery_and_compositional_chain_execution():
    """Verify evaluation of candidate tools and synthesis of 3-step action chain."""
    env = BabyWorldEnvironment(seed=42)
    env.reset("tool_use_world")

    substrate = create_blank_brain_substrate()
    perception = DevelopmentalPerceptionAdapter()
    engine = ToolLearningEngine(substrate, perception, env)

    candidate_tools = [
        "obj_heavy_boulder",  # Ungraspable distractor
        "obj_short_twig",  # Ineffective tool (too short)
        "obj_stick_tool",  # Functional tool
    ]

    res = engine.discover_and_execute_tool_chain(
        target_id="obj_distant_reward",
        candidate_tool_ids=candidate_tools,
    )

    assert res["success"] is True
    assert res["effective_tool_id"] == "obj_stick_tool"
    assert res["target_secured"] is True
    assert res["steps_executed"] == 3

    # Check that substrate causal rules was updated
    assert len(substrate.causal_rules) >= 1
    rule = substrate.causal_rules[0]
    assert "PULL" in rule["action_chain"][1]
    assert rule["effective_tool"] == "obj_stick_tool"


def test_novel_tool_transfer():
    """Verify that acquired tool schema transfers to novel tool geometries and distances."""
    env = BabyWorldEnvironment(seed=42)
    env.reset("tool_use_world")

    substrate = create_blank_brain_substrate()
    perception = DevelopmentalPerceptionAdapter()
    engine = ToolLearningEngine(substrate, perception, env)

    engine.discover_and_execute_tool_chain(
        target_id="obj_distant_reward",
        candidate_tool_ids=["obj_stick_tool"],
    )

    held_out_scenarios = [
        # Novel long rake: length 1.2, target distance 2.5 -> reach 1.5 + 1.2 = 2.7 >= 2.5 -> Success
        {
            "id": "long_rake_test",
            "tool_length": 1.2,
            "target_distance": 2.5,
            "tool_mass": 1.2,
            "actual_success": True,
        },
        # Novel short fork: length 0.3, target distance 2.5 -> reach 1.5 + 0.3 = 1.8 < 2.5 -> Fail
        {
            "id": "short_fork_test",
            "tool_length": 0.3,
            "target_distance": 2.5,
            "tool_mass": 0.5,
            "actual_success": False,
        },
        # Heavy steel bar: length 2.0, target distance 2.0, but mass 30.0 > 10.0 -> Ungraspable -> Fail
        {
            "id": "heavy_bar_test",
            "tool_length": 2.0,
            "target_distance": 2.0,
            "tool_mass": 30.0,
            "actual_success": False,
        },
    ]

    acc, records = engine.evaluate_novel_tool_transfer(held_out_scenarios)
    assert acc == 1.0
    assert len(records) == 3
    assert all(r["match"] for r in records)
