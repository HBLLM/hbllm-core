"""Unit tests for Multi-Step Lookahead Beam Search in CounterfactualPlanner."""

from __future__ import annotations

import pytest

from hbllm.hcir.counterfactual_planner import CounterfactualPlanner
from hbllm.hcir.graph import ActionNode, GoalNode, PhysicalEntityNode
from hbllm.hcir.kernel.capability_resolver import CapabilityResolver
from hbllm.hcir.kernel.scheduler import KernelInstructionScheduler
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.workspace import HCIRWorkspaceState


@pytest.fixture
def hcir_setup() -> tuple[HCIRWorkspaceState, KernelServices]:
    """Create a workspace and kernel services for testing."""
    ws = HCIRWorkspaceState()
    services = KernelServices(
        workspace=ws,
        transaction_manager=TransactionManager(ws),
        capability_resolver=CapabilityResolver(),
        scheduler=KernelInstructionScheduler(),
    )
    return ws, services


@pytest.mark.asyncio
async def test_multi_step_counterfactual_planner_accumulates_discounted_utility(
    hcir_setup: tuple[HCIRWorkspaceState, KernelServices],
) -> None:
    """Verify that horizon=2 evaluates higher cumulative utility than horizon=1 on a clear path."""
    ws, services = hcir_setup

    ws.upsert_node(
        PhysicalEntityNode(
            id="avatar",
            entity_name="avatar",
            properties={"position": [4, 4], "is_avatar": True, "movable": True, "passable": False},
        )
    )
    ws.upsert_node(
        PhysicalEntityNode(
            id="goal",
            entity_name="goal",
            properties={"position": [1, 4], "is_goal": True, "movable": False, "passable": True},
        )
    )

    planner = CounterfactualPlanner(ws, services)
    goal = GoalNode(id="g_nav", description="Navigate towards goal")

    candidates = [
        ActionNode(
            id="act_up",
            intent="MOVE_UP",
            properties={"delta_r": -1, "delta_c": 0, "action_id": 1},
        ),
        ActionNode(
            id="act_down",
            intent="MOVE_DOWN",
            properties={"delta_r": 1, "delta_c": 0, "action_id": 2},
        ),
    ]

    # Run single-step horizon=1
    best_h1 = await planner.evaluate_and_select(goal, candidates, horizon=1)
    assert best_h1.action.id == "act_up"
    u1 = best_h1.utility_score

    # Run multi-step horizon=2
    best_h2 = await planner.evaluate_and_select(goal, candidates, horizon=2)
    assert best_h2.action.id == "act_up"
    u2 = best_h2.utility_score

    # Multi-step should accumulate positive discounted utility from forward rollout
    assert u2 > u1


@pytest.mark.asyncio
async def test_multi_step_counterfactual_planner_horizon_3(
    hcir_setup: tuple[HCIRWorkspaceState, KernelServices],
) -> None:
    """Verify that horizon=3 runs multi-step mental simulation cleanly."""
    ws, services = hcir_setup

    ws.upsert_node(
        PhysicalEntityNode(
            id="avatar",
            entity_name="avatar",
            properties={"position": [5, 5], "is_avatar": True, "movable": True, "passable": False},
        )
    )
    ws.upsert_node(
        PhysicalEntityNode(
            id="goal",
            entity_name="goal",
            properties={"position": [2, 5], "is_goal": True, "movable": False, "passable": True},
        )
    )

    planner = CounterfactualPlanner(ws, services)
    goal = GoalNode(id="g_nav", description="Navigate towards goal")

    candidates = [
        ActionNode(
            id="act_up",
            intent="MOVE_UP",
            properties={"delta_r": -1, "delta_c": 0, "action_id": 1},
        ),
        ActionNode(
            id="act_left",
            intent="MOVE_LEFT",
            properties={"delta_r": 0, "delta_c": -1, "action_id": 3},
        ),
    ]

    best_h3 = await planner.evaluate_and_select(goal, candidates, horizon=3)
    assert best_h3.action.id == "act_up"
    assert best_h3.utility_score > 1.0
