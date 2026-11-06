"""Unit tests for multi-modal action effector planning in HCIR."""

import pytest

from hbllm.hcir.counterfactual_planner import CounterfactualPlanner
from hbllm.hcir.graph import ActionModality, ActionNode, GoalNode
from hbllm.hcir.kernel.capability_resolver import CapabilityResolver
from hbllm.hcir.kernel.scheduler import KernelInstructionScheduler
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.workspace import HCIRWorkspaceState


@pytest.mark.asyncio
async def test_action_modality_enum_and_filtering() -> None:
    """Verify that ActionModality cleanly categorizes effector types."""
    assert ActionModality.LOCOMOTION == "locomotion"
    assert ActionModality.MANIPULATION == "manipulation"
    assert ActionModality.TARGETING == "targeting"
    assert ActionModality.COGNITIVE == "cognitive"

    loco_act = ActionNode(
        id="act_move_up",
        intent="move_north",
        modality=ActionModality.LOCOMOTION,
    )
    manip_act = ActionNode(
        id="act_grab",
        intent="interact_grab",
        modality=ActionModality.MANIPULATION,
    )
    target_act = ActionNode(
        id="act_point",
        intent="point_click",
        modality=ActionModality.TARGETING,
    )

    assert loco_act.modality == ActionModality.LOCOMOTION
    assert manip_act.modality == ActionModality.MANIPULATION
    assert target_act.modality == ActionModality.TARGETING


@pytest.mark.asyncio
async def test_multimodal_planner_affordance_gating() -> None:
    """Verify that evaluate_multimodal_plan filters manipulation actions when affordance is absent."""
    workspace = HCIRWorkspaceState()
    services = KernelServices(
        workspace=workspace,
        transaction_manager=TransactionManager(workspace),
        capability_resolver=CapabilityResolver(),
        scheduler=KernelInstructionScheduler(),
    )
    planner = CounterfactualPlanner(workspace, services)

    goal = GoalNode(id="goal_solve", description="Deliver item to target")

    loco_act = ActionNode(
        id="act_move",
        intent="move_east",
        modality=ActionModality.LOCOMOTION,
        properties={"delta_r": 0, "delta_c": 1},
    )
    manip_act = ActionNode(
        id="act_grab",
        intent="grab_item",
        modality=ActionModality.MANIPULATION,
    )

    # 1. When CAN_MANIPULATE is not active, manipulation action should not be selected
    result_without_manip = await planner.evaluate_multimodal_plan(
        goal=goal,
        candidate_actions=[loco_act, manip_act],
        active_affordances=[],
    )
    assert result_without_manip.action.id == "act_move"

    # 2. When CAN_MANIPULATE is active, manipulation action is valid and selected if affordance aligns
    manip_act_with_bonus = ActionNode(
        id="act_grab_adjacent",
        intent="grab_item",
        modality=ActionModality.MANIPULATION,
        properties={"can_interact": True},
    )
    result_with_manip = await planner.evaluate_multimodal_plan(
        goal=goal,
        candidate_actions=[loco_act, manip_act_with_bonus],
        active_affordances=["CAN_MANIPULATE"],
    )
    assert result_with_manip.action.id == "act_grab_adjacent"
