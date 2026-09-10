"""Tests for Compositional Language Understanding Engine (Stage D11)."""

from plugins.developmental_adapter.blank_brain import create_blank_brain_substrate
from plugins.developmental_adapter.compositional_language import CompositionalLanguageEngine
from plugins.developmental_adapter.environment import BabyWorldEnvironment
from plugins.developmental_adapter.goal_planning import GoalDirectedPlanningEngine
from plugins.developmental_adapter.language_grounding import LanguageGroundingEngine
from plugins.developmental_adapter.types import (
    BabyActionType,
    BabyObjectType,
    BabyRelationType,
)


def test_compositional_sentence_to_goal_and_execution():
    """Verify novel compound sentence parsing into structured goal and zero-shot execution."""
    substrate = create_blank_brain_substrate()
    env = BabyWorldEnvironment(scenario="containment_world")
    grounding = LanguageGroundingEngine(substrate, env)
    planner = GoalDirectedPlanningEngine(substrate, env)

    # Train vocabulary
    grounding.observe_paired_demonstration("ball", {"entity_type": BabyObjectType.BALL})
    grounding.observe_paired_demonstration("box", {"entity_type": BabyObjectType.BOX})
    grounding.observe_paired_demonstration("inside", {"relation": BabyRelationType.INSIDE})
    grounding.observe_paired_demonstration("put", {"action": BabyActionType.PLACE})

    engine = CompositionalLanguageEngine(substrate, env, grounding, planner)

    # Novel sentence never seen during lexical learning
    instruction = "put ball inside box"
    goal = engine.parse_instruction_to_goal(instruction)

    assert goal is not None
    assert goal.predicate == "INSIDE"

    # Execute instruction zero-shot
    result = engine.execute_instruction(instruction)
    assert result.success is True
