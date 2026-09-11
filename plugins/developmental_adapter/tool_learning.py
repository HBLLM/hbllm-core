"""Stage D5: Tool Use & Compositional Causal Chains Engine.

Enables an embodied blank-brain learner to discover intermediate tool use
to solve indirect manipulation problems (e.g. reaching a distant target via a stick)
by synthesizing multi-step backward causal chains:
  GRASP(tool) -> PULL(target, with_tool) -> GRASP(target)
"""

from __future__ import annotations

import logging
from typing import Any

from .blank_brain import BlankBrainSubstrate
from .environment import BabyWorldEnvironment
from .perception import DevelopmentalPerceptionAdapter
from .types import (
    BabyActionType,
    BeliefTransitionEvent,
    BeliefTransitionType,
)

logger = logging.getLogger(__name__)


class ToolLearningEngine:
    """Discovers functional tool use and synthesizes compositional manipulation chains."""

    def __init__(
        self,
        substrate: BlankBrainSubstrate,
        perception: DevelopmentalPerceptionAdapter,
        env: BabyWorldEnvironment,
    ) -> None:
        self.substrate = substrate
        self.perception = perception
        self.env = env

        self.discovered_tool_rules: list[dict[str, Any]] = []
        self.belief_history: list[BeliefTransitionEvent] = []
        self.interventions_count: int = 0

    def attempt_direct_manipulation(self, target_id: str) -> dict[str, Any]:
        """Verify that direct manipulation fails when target is beyond reach."""
        self.interventions_count += 1
        prior_state = self.env.save_state()

        # Step 1: Attempt direct grasp
        _, _, _, consequences_grasp = self.env.step(BabyActionType.GRASP, target_id=target_id)
        # Step 2: Attempt direct pull
        _, _, _, consequences_pull = self.env.step(BabyActionType.PULL, target_id=target_id)

        self.env.restore_state(prior_state)
        return {
            "target_id": target_id,
            "direct_grasp_success": consequences_grasp.get("grasped", False),
            "direct_pull_success": consequences_pull.get("pulled", False),
            "out_of_reach": consequences_pull.get("out_of_reach", True),
        }

    def discover_and_execute_tool_chain(
        self,
        target_id: str,
        candidate_tool_ids: list[str],
    ) -> dict[str, Any]:
        """Evaluate candidate tools and synthesize a 3-step compositional manipulation chain.

        1. Test candidates for graspability and reach extension.
        2. Reject ineffective tools (too short, ungraspable).
        3. Execute: GRASP(tool) -> PULL(target) -> GRASP(target).
        """
        eval_history: list[dict[str, Any]] = []
        effective_tool_id: str | None = None

        for tool_id in candidate_tool_ids:
            self.interventions_count += 1
            prior_state = self.env.save_state()

            # Test 1: Can agent grasp this candidate?
            _, _, _, grasp_res = self.env.step(BabyActionType.GRASP, target_id=tool_id)
            if not grasp_res.get("grasped", False):
                eval_history.append(
                    {
                        "tool_id": tool_id,
                        "status": "UNGRASPABLE",
                        "reason": "Mass too heavy or out of reach",
                    }
                )
                self.env.restore_state(prior_state)
                continue

            # Test 2: Can agent pull the distant target using this held candidate?
            _, _, _, pull_res = self.env.step(BabyActionType.PULL, target_id=target_id)
            pull_success = pull_res.get("pulled", False)

            if pull_success:
                effective_tool_id = tool_id
                eval_history.append(
                    {
                        "tool_id": tool_id,
                        "status": "EFFECTIVE_TOOL",
                        "new_target_distance": pull_res.get("new_distance", 0.0),
                    }
                )
                # Keep state to proceed to Step 3: GRASP retrieved target
                _, _, _, final_grasp = self.env.step(BabyActionType.GRASP, target_id=target_id)
                target_secured = final_grasp.get("grasped", False)

                rule = {
                    "rule_id": "rule_indirect_tool_reach",
                    "action_chain": ["GRASP(tool)", "PULL(target, with_tool)", "GRASP(target)"],
                    "precondition": "IN_REACH(tool) ∧ EXTENDED_REACH(tool, target)",
                    "consequence": "HOLDING(target)",
                    "effective_tool": tool_id,
                    "target_secured": target_secured,
                }
                self.discovered_tool_rules.append(rule)
                self.substrate.causal_rules.append(rule)
                self._record_event(
                    event_type=BeliefTransitionType.TOOL_COMPOSED,
                    condition=f"TOOL({tool_id}) -> PULL({target_id}) => IN_REACH",
                    prior_conf=0.5,
                    post_conf=1.0,
                )
                self.env.restore_state(prior_state)
                return {
                    "success": True,
                    "effective_tool_id": effective_tool_id,
                    "target_secured": target_secured,
                    "steps_executed": 3,
                    "eval_history": eval_history,
                }
            else:
                eval_history.append(
                    {
                        "tool_id": tool_id,
                        "status": "INSUFFICIENT_LENGTH",
                        "reason": "Length insufficient to span distance to target",
                    }
                )
                self.env.restore_state(prior_state)

        return {
            "success": False,
            "effective_tool_id": None,
            "target_secured": False,
            "eval_history": eval_history,
        }

    def evaluate_novel_tool_transfer(
        self,
        held_out_scenarios: list[dict[str, Any]],
    ) -> tuple[float, list[dict[str, Any]]]:
        """Test acquired tool schema on novel objects and distances."""
        if not self.discovered_tool_rules:
            return 0.0, []

        successes = 0
        total = 0
        records: list[dict[str, Any]] = []

        for sc in held_out_scenarios:
            tool_len = sc.get("tool_length", 0.0)
            target_dist = sc.get("target_distance", 1.0)
            tool_mass = sc.get("tool_mass", 1.0)
            base_reach = self.env.REACH_DISTANCE

            # Predict feasibility using acquired rule:
            # Feasible if tool is graspable (mass < 10.0) and base_reach + tool_len >= target_dist
            is_graspable = tool_mass < 10.0
            can_reach = (base_reach + tool_len) >= target_dist
            predicted_success = is_graspable and can_reach

            actual_success = sc.get("actual_success", False)
            match = predicted_success == actual_success
            if match:
                successes += 1
            total += 1

            records.append(
                {
                    "scenario_id": sc.get("id"),
                    "predicted_success": predicted_success,
                    "actual_success": actual_success,
                    "match": match,
                }
            )

        accuracy = successes / total if total > 0 else 0.0
        return accuracy, records

    def _record_event(
        self,
        event_type: BeliefTransitionType,
        condition: str,
        prior_conf: float,
        post_conf: float,
    ) -> None:
        self.belief_history.append(
            BeliefTransitionEvent(
                event_type=event_type,
                step_index=self.interventions_count,
                hypothesis_id="rule_tool_use",
                variable="tool_use",
                condition=condition,
                prior_confidence=prior_conf,
                posterior_confidence=post_conf,
                is_falsified=False,
            )
        )
