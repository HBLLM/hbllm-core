"""Goal-Directed Behavior & Multi-Step Compositional Planning (Stages D6 & D7).

Implements backward causal chaining from target desire states to subgoals,
action sequencing, and dynamic replanning on environmental perturbation.
"""

from __future__ import annotations

import logging
from typing import Any

from .blank_brain import BlankBrainSubstrate
from .environment import BabyWorldEnvironment
from .types import (
    BabyActionType,
    BeliefTransitionEvent,
    BeliefTransitionType,
    PlanExecutionResult,
    PlanStep,
    PredicateGoal,
    Vector2D,
)

logger = logging.getLogger(__name__)


class GoalDirectedPlanningEngine:
    """Synthesizes and executes multi-step causal plans to achieve predicate desires."""

    def __init__(self, substrate: BlankBrainSubstrate, env: BabyWorldEnvironment) -> None:
        self.substrate = substrate
        self.env = env
        self.execution_history: list[PlanExecutionResult] = []

    def synthesize_plan(self, goal: PredicateGoal) -> list[PlanStep]:
        """Synthesize a causal action chain via backward goal decomposition."""
        steps: list[PlanStep] = []
        target_obj = self.env.objects.get(goal.subject_id)
        if not target_obj:
            logger.warning(f"Subject {goal.subject_id} not found in environment.")
            return steps

        # Case 1: Desired State change (e.g. door open via button)
        if goal.predicate == "STATE":
            # Search causal rules for action causing door to open
            # Typically: PUSH button causes door to open
            button_id = None
            for rule in self.substrate.causal_rules:
                if (
                    rule.get("consequence") == "OPEN"
                    or "open" in str(rule.get("consequence")).lower()
                ):
                    button_id = rule.get("subject_id")
            if not button_id:
                # Find any button in the environment
                for oid, obj in self.env.objects.items():
                    if obj.object_type.value == "button":
                        button_id = oid
                        break
            if button_id:
                b_obj = self.env.objects[button_id]
                steps.append(
                    PlanStep(
                        action=BabyActionType.MOVE,
                        target_id=button_id,
                        parameter=b_obj.position,
                        expected_outcome="HAND_AT_BUTTON",
                    )
                )
                steps.append(
                    PlanStep(
                        action=BabyActionType.PUSH,
                        target_id=button_id,
                        expected_outcome=f"{goal.subject_id}_STATE_{goal.target_value}",
                    )
                )
            return steps

        # Case 2: REACHABLE predicate
        dist_to_hand = self.env.agent_hand_position.distance_to(target_obj.position)
        if goal.predicate == "REACHABLE":
            if dist_to_hand <= self.env.REACH_DISTANCE:
                return []  # Already reachable

            # Need tool retrieval chain
            tool_id = self._find_best_tool_for(target_obj)
            if tool_id:
                tool_obj = self.env.objects[tool_id]
                steps.append(
                    PlanStep(
                        action=BabyActionType.MOVE,
                        target_id=tool_id,
                        parameter=tool_obj.position,
                        expected_outcome="HAND_AT_TOOL",
                    )
                )
                steps.append(
                    PlanStep(
                        action=BabyActionType.GRASP,
                        target_id=tool_id,
                        expected_outcome="TOOL_IN_HAND",
                    )
                )
                steps.append(
                    PlanStep(
                        action=BabyActionType.EXTEND,
                        target_id=goal.subject_id,
                        tool_id=tool_id,
                        expected_outcome="TOOL_TOUCHING_TARGET",
                    )
                )
                steps.append(
                    PlanStep(
                        action=BabyActionType.PULL,
                        target_id=goal.subject_id,
                        expected_outcome="TARGET_IN_REACH",
                    )
                )
            return steps

        # Case 3: INSIDE predicate (e.g. INSIDE(ball, box))
        if goal.predicate == "INSIDE" and goal.target_id:
            container_obj = self.env.objects.get(goal.target_id)
            if not container_obj:
                return steps

            # Check if container is closed and needs opening
            if container_obj.is_open is False:
                steps.append(
                    PlanStep(
                        action=BabyActionType.MOVE,
                        target_id=goal.target_id,
                        parameter=container_obj.position,
                        expected_outcome="HAND_AT_CONTAINER",
                    )
                )
                steps.append(
                    PlanStep(
                        action=BabyActionType.OPEN,
                        target_id=goal.target_id,
                        expected_outcome="CONTAINER_OPENED",
                    )
                )

            # Check if target is directly reachable or requires tool
            if dist_to_hand > self.env.REACH_DISTANCE:
                # Add tool reach sub-plan
                tool_id = self._find_best_tool_for(target_obj)
                if tool_id:
                    tool_obj = self.env.objects[tool_id]
                    steps.append(
                        PlanStep(
                            action=BabyActionType.MOVE,
                            target_id=tool_id,
                            parameter=tool_obj.position,
                            expected_outcome="HAND_AT_TOOL",
                        )
                    )
                    steps.append(
                        PlanStep(
                            action=BabyActionType.GRASP,
                            target_id=tool_id,
                            expected_outcome="TOOL_IN_HAND",
                        )
                    )
                    steps.append(
                        PlanStep(
                            action=BabyActionType.EXTEND,
                            target_id=goal.subject_id,
                            tool_id=tool_id,
                            expected_outcome="TOOL_TOUCHING_TARGET",
                        )
                    )
                    steps.append(
                        PlanStep(
                            action=BabyActionType.PULL,
                            target_id=goal.subject_id,
                            expected_outcome="TARGET_IN_REACH",
                        )
                    )
                    # Release tool before grasping object
                    steps.append(
                        PlanStep(
                            action=BabyActionType.RELEASE,
                            target_id=tool_id,
                            expected_outcome="TOOL_RELEASED",
                        )
                    )

            # Move to object, grasp it, move to container, place inside
            steps.append(
                PlanStep(
                    action=BabyActionType.MOVE,
                    target_id=goal.subject_id,
                    parameter=target_obj.position,
                    expected_outcome="HAND_AT_OBJECT",
                )
            )
            steps.append(
                PlanStep(
                    action=BabyActionType.GRASP,
                    target_id=goal.subject_id,
                    expected_outcome="OBJECT_IN_HAND",
                )
            )
            steps.append(
                PlanStep(
                    action=BabyActionType.MOVE,
                    target_id=goal.target_id,
                    parameter=container_obj.position,
                    expected_outcome="HAND_AT_CONTAINER",
                )
            )
            steps.append(
                PlanStep(
                    action=BabyActionType.PLACE,
                    target_id=goal.subject_id,
                    parameter=goal.target_id,
                    expected_outcome=f"INSIDE({goal.subject_id}, {goal.target_id})",
                )
            )
            return steps

        # Case 4: ON predicate (e.g. ON(block_a, table_b))
        if goal.predicate == "ON" and goal.target_id:
            support_obj = self.env.objects.get(goal.target_id)
            if support_obj:
                steps.append(
                    PlanStep(
                        action=BabyActionType.MOVE,
                        target_id=goal.subject_id,
                        parameter=target_obj.position,
                        expected_outcome="HAND_AT_OBJECT",
                    )
                )
                steps.append(
                    PlanStep(
                        action=BabyActionType.GRASP,
                        target_id=goal.subject_id,
                        expected_outcome="OBJECT_IN_HAND",
                    )
                )
                steps.append(
                    PlanStep(
                        action=BabyActionType.MOVE,
                        target_id=goal.target_id,
                        parameter=Vector2D(support_obj.position.x, support_obj.position.y + 0.3),
                        expected_outcome="HAND_ABOVE_SUPPORT",
                    )
                )
                steps.append(
                    PlanStep(
                        action=BabyActionType.PLACE,
                        target_id=goal.subject_id,
                        parameter=goal.target_id,
                        expected_outcome=f"ON({goal.subject_id}, {goal.target_id})",
                    )
                )
            return steps

        return steps

    def execute_with_replanning(
        self, goal: PredicateGoal, max_replans: int = 3
    ) -> PlanExecutionResult:
        """Execute plan step-by-step, monitoring environment and replanning if perturbed."""
        steps = self.synthesize_plan(goal)
        replan_count = 0
        wasted_actions = 0
        executed: list[str] = []

        step_idx = 0
        max_total_steps = 25

        while step_idx < len(steps) and len(executed) < max_total_steps:
            current_step = steps[step_idx]

            # Check if goal already satisfied early
            if self._is_goal_satisfied(goal):
                break

            # Execute step in environment
            executed.append(f"{current_step.action.value}({current_step.target_id})")
            self.env.step(
                action=current_step.action,
                target_id=current_step.target_id,
                parameter=current_step.parameter,
            )

            # Verify step effect or detect environmental perturbation
            if not self._verify_step_postcondition(current_step):
                # Step failed or world perturbed: trigger replan
                wasted_actions += 1
                if replan_count < max_replans:
                    replan_count += 1
                    logger.info(
                        f"Step {current_step.action.value} failed. Replanning attempt {replan_count}..."
                    )
                    steps = self.synthesize_plan(goal)
                    step_idx = 0
                    continue
                else:
                    break

            step_idx += 1

        is_success = self._is_goal_satisfied(goal)

        result = PlanExecutionResult(
            goal=goal,
            steps=steps,
            success=is_success,
            replan_count=replan_count,
            wasted_actions=wasted_actions,
            executed_actions=executed,
        )
        self.execution_history.append(result)

        # Log event to substrate
        if hasattr(self.substrate, "profile") and hasattr(
            self.substrate.profile, "belief_transitions"
        ):
            self.substrate.profile.belief_transitions.append(
                BeliefTransitionEvent(
                    event_type=(
                        BeliefTransitionType.PLAN_REPLANNED
                        if replan_count > 0
                        else BeliefTransitionType.PLAN_EXECUTED
                    ),
                    step_index=len(executed),
                    variable=goal.predicate,
                    condition=f"{goal.subject_id} -> {goal.target_id}",
                    posterior_confidence=1.0 if is_success else 0.0,
                    evidence={"success": is_success, "replans": replan_count},
                )
            )

        return result

    def _find_best_tool_for(self, target_obj: Any) -> str | None:
        """Identify viable tool based on reach extension and graspability."""
        dist = self.env.agent_hand_position.distance_to(target_obj.position)
        for oid, obj in self.env.objects.items():
            if obj.mass <= 2.5 and obj.tool_length > 0:
                if self.env.REACH_DISTANCE + obj.tool_length >= dist:
                    return oid
        # Fallback to any tool
        for oid, obj in self.env.objects.items():
            if obj.is_tool:
                return oid
        return None

    def _verify_step_postcondition(self, step: PlanStep) -> bool:
        """Check if executed step reached intended state."""
        if step.action == BabyActionType.GRASP:
            return self.env.agent_held_object_id == step.target_id
        if step.action == BabyActionType.RELEASE:
            return self.env.agent_held_object_id is None
        if step.action == BabyActionType.OPEN:
            obj = self.env.objects.get(step.target_id or "")
            return obj is not None and obj.is_open is True
        return True

    def _is_goal_satisfied(self, goal: PredicateGoal) -> bool:
        """Check if goal predicate holds in current environment state."""
        subj = self.env.objects.get(goal.subject_id)
        if not subj:
            return False

        if goal.predicate == "INSIDE" and goal.target_id:
            container = self.env.objects.get(goal.target_id)
            if not container:
                return False
            # Check if coordinates inside container bounds or contained_in pointer set
            is_inside = (
                subj.contained_in == goal.target_id
                or goal.subject_id in container.contained_object_ids
                or (
                    abs(subj.position.x - container.position.x) <= container.size.x / 2.0
                    and abs(subj.position.y - container.position.y) <= container.size.y / 2.0
                )
            )
            return is_inside

        if goal.predicate == "REACHABLE":
            dist = self.env.agent_hand_position.distance_to(subj.position)
            return dist <= self.env.REACH_DISTANCE

        if goal.predicate == "ON" and goal.target_id:
            support = self.env.objects.get(goal.target_id)
            if not support:
                return False
            return (
                abs(subj.position.x - support.position.x) <= support.size.x / 2.0
                and abs(subj.position.y - (support.position.y + support.size.y / 2.0)) < 0.2
            )

        if goal.predicate == "STATE":
            return subj.is_open == goal.target_value

        return False
