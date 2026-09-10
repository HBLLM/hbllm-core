"""Tests for Metacognitive Calibration Engine (Stage D13)."""

from plugins.developmental_adapter.blank_brain import create_blank_brain_substrate
from plugins.developmental_adapter.environment import BabyWorldEnvironment
from plugins.developmental_adapter.metacognition import MetacognitiveEngine
from plugins.developmental_adapter.types import PredicateGoal


def test_metacognitive_confidence_assessment_and_calibration():
    """Verify confidence prediction, strategic abstention, and Brier score."""
    substrate = create_blank_brain_substrate()
    env = BabyWorldEnvironment(scenario="tool_use_world")
    engine = MetacognitiveEngine(substrate, env, abstain_threshold=0.6)

    # Goal that agent is equipped for
    easy_goal = PredicateGoal(predicate="REACHABLE", subject_id="obj_stick_tool")
    conf_easy = engine.assess_confidence(easy_goal)
    assert conf_easy >= 0.8
    assert engine.should_abstain_or_explore(easy_goal) is False

    # Goal for non-existent or impossible target
    hard_goal = PredicateGoal(predicate="REACHABLE", subject_id="ghost_entity")
    conf_hard = engine.assess_confidence(hard_goal)
    assert conf_hard < 0.6
    assert engine.should_abstain_or_explore(hard_goal) is True

    # Record outcomes
    engine.record_outcome(easy_goal, conf_easy, actual_success=True)
    engine.record_outcome(hard_goal, conf_hard, actual_success=False)

    report = engine.compute_calibration_report()
    assert report.brier_score < 0.15
    assert report.abstention_accuracy == 1.0
