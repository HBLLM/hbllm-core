"""Advanced tests for Compositional Language and Metacognitive Active Inquiries (Stages D11-D13)."""

from plugins.developmental_adapter.blank_brain import create_blank_brain_substrate
from plugins.developmental_adapter.compositional_language import CompositionalLanguageEngine
from plugins.developmental_adapter.environment import BabyWorldEnvironment
from plugins.developmental_adapter.goal_planning import GoalDirectedPlanningEngine
from plugins.developmental_adapter.language_grounding import LanguageGroundingEngine
from plugins.developmental_adapter.metacognition import MetacognitiveEngine
from plugins.developmental_adapter.types import (
    BabyActionType,
    BabyObjectType,
    BabyRelationType,
    PredicateGoal,
    Vector2D,
)


def test_multi_clause_compound_instruction_parsing_and_execution() -> None:
    """Verify parsing and sequential zero-shot execution of multi-clause compound instructions."""
    substrate = create_blank_brain_substrate()
    env = BabyWorldEnvironment(scenario="containment_world")
    grounding = LanguageGroundingEngine(substrate, env)
    planner = GoalDirectedPlanningEngine(substrate, env)

    # Train lexicon
    grounding.observe_paired_demonstration("ball", {"entity_type": BabyObjectType.BALL})
    grounding.observe_paired_demonstration("box", {"entity_type": BabyObjectType.BOX})
    grounding.observe_paired_demonstration("inside", {"relation": BabyRelationType.INSIDE})
    grounding.observe_paired_demonstration("put", {"action": BabyActionType.PLACE})
    grounding.observe_paired_demonstration("pick", {"action": BabyActionType.GRASP})

    engine = CompositionalLanguageEngine(substrate, env, grounding, planner)

    # Test splitting clauses
    instruction = "pick ball and then put ball inside box"
    clauses = engine.split_compound_clauses(instruction)
    assert len(clauses) == 2
    assert "pick ball" in clauses[0]
    assert "put ball inside box" in clauses[1]

    # Test parsing into ordered sequence of goals
    goals = engine.parse_compound_instruction(instruction)
    assert len(goals) == 2
    assert goals[0].predicate == "REACHABLE"
    assert goals[1].predicate == "INSIDE"

    # Execute compound instruction sequentially
    exec_res = engine.execute_compound_instruction(instruction)
    assert len(exec_res) == 2
    assert all(r.success for r in exec_res)

    # Test single entry-point execution with aggregated result
    single_res = engine.execute_instruction("pick ball and put ball inside box")
    assert single_res.success is True
    assert len(single_res.steps) > 0


def test_metacognitive_active_clarification_queries() -> None:
    """Verify that MetacognitiveEngine generates actionable clarification queries on uncertainty."""
    substrate = create_blank_brain_substrate()
    env = BabyWorldEnvironment(scenario="tool_world")
    metacog = MetacognitiveEngine(substrate, env, abstain_threshold=0.65)

    # 1. Distant target out of reach without viable tool
    target_id = "obj_red_ball"
    distant_goal = PredicateGoal(predicate="REACHABLE", subject_id=target_id)
    # Move target far away
    env.objects[target_id].position = Vector2D(5.0, 5.0)
    # Remove any tools
    for obj in list(env.objects.values()):
        if obj.is_tool:
            obj.is_tool = False
            obj.tool_length = 0.0

    conf = metacog.assess_confidence(distant_goal)
    assert conf < 0.65
    assert metacog.should_abstain_or_explore(distant_goal) is True

    query = metacog.generate_clarification_query(distant_goal)
    assert "tool" in query.lower() or "reach" in query.lower()

    # 2. Unknown object referent
    unknown_goal = PredicateGoal(predicate="REACHABLE", subject_id="nonexistent_object")
    query_unknown = metacog.generate_clarification_query(unknown_goal)
    assert "Which object" in query_unknown

    # 3. Compound confidence evaluation
    goals = [
        PredicateGoal(predicate="REACHABLE", subject_id="target_sphere"),
        PredicateGoal(predicate="STATE", subject_id="target_sphere"),
    ]
    joint_conf = metacog.assess_compound_confidence(goals)
    assert 0.0 < joint_conf <= 1.0


def test_metacognitive_calibration_reporting() -> None:
    """Verify Brier score and Expected Calibration Error computation."""
    substrate = create_blank_brain_substrate()
    env = BabyWorldEnvironment()
    metacog = MetacognitiveEngine(substrate, env)

    # Simulate well-calibrated sequence of outcomes
    goal_easy = PredicateGoal(predicate="REACHABLE", subject_id="obj_1")
    metacog.record_outcome(goal_easy, confidence=0.9, actual_success=True)

    goal_hard = PredicateGoal(predicate="INSIDE", subject_id="obj_2")
    metacog.record_outcome(goal_hard, confidence=0.2, actual_success=False)

    report = metacog.compute_calibration_report()
    assert report.brier_score < 0.1
    assert report.expected_calibration_error < 0.2
    assert report.abstention_accuracy == 1.0
