"""Planning Capabilities — HCIR §10.

Registers planning.generate_candidates, planning.validate, and planning.expand
capability backends for CounterfactualPlanner and HierarchicalGoalDecomposer.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from hbllm.hcir.graph import ActionNode, GoalNode
from hbllm.hcir.kernel.capability_resolver import ICapabilityExecutor
from hbllm.hcir.kernel.cognitive_abi import CapabilityCall, CapabilityManifest, CapabilityResult

if TYPE_CHECKING:
    from hbllm.hcir.counterfactual_planner import CounterfactualPlanner
    from hbllm.hcir.kernel.services import KernelServices
    from hbllm.hcir.subgoal_decomposer import HierarchicalGoalDecomposer
    from hbllm.hcir.workspace import HCIRWorkspaceState

logger = logging.getLogger(__name__)

PLANNING_MANIFESTS = [
    CapabilityManifest(name="planning.generate_candidates", owner="HCIR.Planner"),
    CapabilityManifest(name="planning.validate", owner="HCIR.Planner"),
    CapabilityManifest(name="planning.expand", owner="HCIR.Planner"),
]


def execute_planning_capability(
    call: CapabilityCall,
    planner: CounterfactualPlanner | None = None,
    decomposer: HierarchicalGoalDecomposer | None = None,
    workspace: HCIRWorkspaceState | None = None,
) -> CapabilityResult:
    """Execute a planning capability call backed by CounterfactualPlanner or HierarchicalGoalDecomposer."""
    cap_name = call.capability_name
    params = getattr(call, "arguments", None) or getattr(call, "parameters", None) or {}
    logger.debug("Executing planning capability: %s with params %s", cap_name, params)

    if cap_name == "planning.generate_candidates":
        candidate_data = params.get("candidate_actions")
        candidates: list[ActionNode] = []

        if isinstance(candidate_data, list) and candidate_data:
            for item in candidate_data:
                if isinstance(item, ActionNode):
                    candidates.append(item)
                elif isinstance(item, dict):
                    candidates.append(
                        ActionNode(
                            id=item.get("id") or f"act_{uuid.uuid4().hex[:6]}",
                            action_type=item.get("action_type") or item.get("intent") or "NAVIGATE",
                            intent=item.get("intent") or item.get("action_type") or "NAVIGATE",
                            parameters=item.get("parameters") or {},
                        )
                    )
                elif isinstance(item, str):
                    candidates.append(
                        ActionNode(
                            id=f"act_{uuid.uuid4().hex[:6]}",
                            action_type=item,
                            intent=item,
                        )
                    )
        else:
            # Default general action candidates
            default_intents = ["NAVIGATE", "INTERACT", "OBSERVE", "BRANCH"]
            for intent in default_intents:
                candidates.append(
                    ActionNode(
                        id=f"act_{intent.lower()}_{uuid.uuid4().hex[:4]}",
                        action_type=intent,
                        intent=intent,
                    )
                )

        selected_plan = None
        goal_param = params.get("goal")
        goal_node = None
        if isinstance(goal_param, GoalNode):
            goal_node = goal_param
        elif isinstance(goal_param, dict):
            goal_node = GoalNode(
                id=goal_param.get("id") or "goal_eval",
                description=goal_param.get("description") or "Active Goal",
                properties=goal_param.get("properties") or {},
            )

        if planner is not None and goal_node is not None and candidates:
            try:
                res = planner.evaluate_and_select_sync(goal_node, candidates)
                selected_plan = res.action.id
            except Exception as e:
                logger.debug("Counterfactual evaluation skipped in candidate generation: %s", e)
                selected_plan = candidates[0].id if candidates else None

        output = {
            "capability": cap_name,
            "candidates_generated": len(candidates),
            "candidates": [c.model_dump() for c in candidates],
            "candidate_ids": [c.id for c in candidates],
            "selected_plan": selected_plan or (candidates[0].id if candidates else "none"),
        }
        return CapabilityResult(capability_name=cap_name, status="SUCCESS", output=output)

    elif cap_name == "planning.validate":
        action_id = params.get("action_id") or params.get("candidate_id") or "action_default"
        action_intent = params.get("intent") or params.get("action_type") or ""
        preconditions = params.get("preconditions") or {}

        # Validate that required keys or conditions are present
        is_valid = True
        reasons: list[str] = []

        if "energy_required" in preconditions:
            energy_avail = params.get("energy_available", 100)
            if energy_avail < preconditions["energy_required"]:
                is_valid = False
                reasons.append("Insufficient energy for action execution")

        if "collision" in params and params["collision"]:
            is_valid = False
            reasons.append("Trajectory intersects with impassable barrier")

        output = {
            "capability": cap_name,
            "action_id": action_id,
            "intent": action_intent,
            "valid": is_valid,
            "reasons": reasons,
        }
        return CapabilityResult(
            capability_name=cap_name,
            status="SUCCESS" if is_valid else "VALIDATION_FAILED",
            output=output,
        )

    elif cap_name == "planning.expand":
        primary_goal_data = params.get("primary_goal") or params.get("goal")
        avatar_pos = params.get("avatar_pos") or (0, 0)
        barrier_cells = set(tuple(p) for p in params.get("barrier_cells", []))
        grid_shape = tuple(params.get("grid_shape", (10, 10)))
        candidate_subgoals = params.get("candidate_subgoals")

        primary_goal = None
        if isinstance(primary_goal_data, GoalNode):
            primary_goal = primary_goal_data
        elif isinstance(primary_goal_data, dict):
            primary_goal = GoalNode(
                id=primary_goal_data.get("id") or "goal_root",
                description=primary_goal_data.get("description") or "Primary Goal",
                properties=primary_goal_data.get("properties") or {},
            )
        else:
            primary_goal = GoalNode(id="goal_root", description="Default Primary Goal")

        if decomposer is not None and workspace is not None:
            active_leaf = decomposer.decompose_goal(
                workspace=workspace,
                primary_goal=primary_goal,
                avatar_pos=avatar_pos,
                barrier_cells=barrier_cells,
                grid_shape=grid_shape,
                candidate_subgoals=candidate_subgoals,
            )
            output = {
                "capability": cap_name,
                "primary_goal_id": primary_goal.id,
                "active_subgoal_id": active_leaf.id,
                "active_subgoal_description": active_leaf.description,
                "is_primary": active_leaf.id == primary_goal.id,
            }
        else:
            output = {
                "capability": cap_name,
                "primary_goal_id": primary_goal.id,
                "active_subgoal_id": primary_goal.id,
                "active_subgoal_description": primary_goal.description,
                "is_primary": True,
            }

        return CapabilityResult(capability_name=cap_name, status="SUCCESS", output=output)

    else:
        return CapabilityResult(
            capability_name=cap_name,
            status="UNKNOWN_CAPABILITY",
            output={"error": f"Planning capability '{cap_name}' is not recognized"},
        )


class PlanningCapabilityExecutor(ICapabilityExecutor):
    """Capability executor for planning.* capabilities registered with CapabilityResolver."""

    def __init__(
        self,
        workspace: HCIRWorkspaceState | None = None,
        services: KernelServices | None = None,
        planner: CounterfactualPlanner | None = None,
        decomposer: HierarchicalGoalDecomposer | None = None,
        default_capability: str = "planning.generate_candidates",
    ) -> None:
        self._workspace = workspace or (services.workspace if services else None)
        self._services = services
        self._planner = planner
        self._decomposer = decomposer
        self._default_capability = default_capability

    @property
    def is_available(self) -> bool:
        return True

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute planning capability via CapabilityResolver."""
        cap_name = (
            params.get("capability") or params.get("capability_name") or self._default_capability
        )
        call = CapabilityCall(
            capability_name=cap_name,
            arguments=params.get("params") or params,
        )
        res = execute_planning_capability(
            call=call,
            planner=self._planner,
            decomposer=self._decomposer,
            workspace=self._workspace,
        )
        return {
            "status": res.status,
            **res.output,
        }
