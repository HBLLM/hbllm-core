"""Tests for Goal-Directed Behavior & Multi-Step Compositional Planning (Stages D6 & D7)."""

from plugins.developmental_adapter.blank_brain import create_blank_brain_substrate
from plugins.developmental_adapter.environment import BabyWorldEnvironment
from plugins.developmental_adapter.goal_planning import GoalDirectedPlanningEngine
from plugins.developmental_adapter.types import (
    PredicateGoal,
    Vector2D,
)


def test_goal_synthesis_and_execution_inside_container():
    """Verify that agent synthesizes a multi-step plan to put an object inside a container."""
    substrate = create_blank_brain_substrate()
    env = BabyWorldEnvironment(scenario="containment_world")
    planner = GoalDirectedPlanningEngine(substrate, env)

    goal = PredicateGoal(
        predicate="INSIDE",
        subject_id="obj_outside_ball",
        target_id="obj_container_box",
    )

    steps = planner.synthesize_plan(goal)
    assert len(steps) >= 3

    # Execute plan
    result = planner.execute_with_replanning(goal)
    assert result.success is True
    assert planner._is_goal_satisfied(goal) is True


def test_goal_replanning_on_perturbation():
    """Verify dynamic replanning when target object is perturbed during execution."""
    substrate = create_blank_brain_substrate()
    env = BabyWorldEnvironment(scenario="containment_world")
    planner = GoalDirectedPlanningEngine(substrate, env)

    goal = PredicateGoal(
        predicate="INSIDE",
        subject_id="obj_outside_ball",
        target_id="obj_container_box",
    )

    # Perturb initial distance to force replanning
    toy = env.objects["obj_outside_ball"]
    toy.position = Vector2D(0.8, 0.4)

    result = planner.execute_with_replanning(goal, max_replans=3)
    assert result.success is True
    assert planner._is_goal_satisfied(goal) is True


def test_goal_reach_with_tool_synthesis():
    """Verify tool retrieval sub-plan is composed when target is out of reach."""
    substrate = create_blank_brain_substrate()
    env = BabyWorldEnvironment(scenario="tool_use_world")
    planner = GoalDirectedPlanningEngine(substrate, env)

    goal = PredicateGoal(
        predicate="REACHABLE",
        subject_id="obj_distant_reward",
    )

    result = planner.execute_with_replanning(goal)
    assert result.success is True
    apple = env.objects["obj_distant_reward"]
    dist = env.agent_hand_position.distance_to(apple.position)
    assert dist <= env.REACH_DISTANCE
