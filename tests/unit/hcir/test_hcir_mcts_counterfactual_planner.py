"""Unit tests for Hierarchical Monte Carlo Tree Search (MCTS) in CounterfactualPlanner."""

from __future__ import annotations

import pytest

from hbllm.hcir.counterfactual_planner import (
    CandidatePlanResult,
    CounterfactualPlanner,
    MCTSConfig,
    MCTSNode,
)
from hbllm.hcir.graph import ActionModality, ActionNode, GoalNode, PhysicalEntityNode
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


def test_mcts_node_uct_score() -> None:
    root = MCTSNode(node_id="root", state_branch="main", visit_count=10)
    unvisited = MCTSNode(node_id="c1", state_branch="b1", parent=root, visit_count=0)
    visited_low_val = MCTSNode(
        node_id="c2",
        state_branch="b2",
        parent=root,
        visit_count=5,
        total_value=2.0,
    )
    visited_high_val = MCTSNode(
        node_id="c3",
        state_branch="b3",
        parent=root,
        visit_count=5,
        total_value=8.0,
    )

    # Unvisited node must have infinite UCT score
    assert unvisited.uct_score(parent_visits=10, exploration_constant=1.414) == float("inf")

    # High value node has higher UCT score than low value with equal visits
    score_low = visited_low_val.uct_score(parent_visits=10, exploration_constant=1.414)
    score_high = visited_high_val.uct_score(parent_visits=10, exploration_constant=1.414)
    assert score_high > score_low


@pytest.mark.asyncio
async def test_mcts_counterfactual_planner_selects_goal_directed_action(
    hcir_setup: tuple[HCIRWorkspaceState, KernelServices],
) -> None:
    ws, services = hcir_setup

    # Avatar at (4, 4), Goal at (1, 4) -> UP reduces distance
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
        ActionNode(
            id="act_left",
            intent="MOVE_LEFT",
            properties={"delta_r": 0, "delta_c": -1, "action_id": 3},
        ),
        ActionNode(
            id="act_right",
            intent="MOVE_RIGHT",
            properties={"delta_r": 0, "delta_c": 1, "action_id": 4},
        ),
    ]

    result = await planner.mcts_evaluate_and_select(
        goal=goal,
        candidate_actions=candidates,
        config=MCTSConfig(max_simulations=12, max_depth=3),
    )

    assert isinstance(result, CandidatePlanResult)
    assert result.candidate_id == "act_up"
    assert result.utility_score > 0.5


@pytest.mark.asyncio
async def test_mcts_counterfactual_planner_prunes_deadlock(
    hcir_setup: tuple[HCIRWorkspaceState, KernelServices],
) -> None:
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
            id="box",
            entity_name="box",
            properties={"position": [4, 5], "movable": True, "passable": False},
        )
    )
    ws.upsert_node(
        PhysicalEntityNode(
            id="wall_corner",
            entity_name="wall",
            properties={"position": [4, 6], "movable": False, "passable": False},
        )
    )

    planner = CounterfactualPlanner(ws, services)
    goal = GoalNode(id="g_escape", description="Avoid deadlocks")

    candidates = [
        ActionNode(
            id="act_push_corner",
            intent="PUSH_INTO_CORNER",
            properties={
                "predicted_state": {"spatial_outcome": {"deadlock": True, "progress": -1.0}}
            },
        ),
        ActionNode(
            id="act_step_open",
            intent="MOVE_OPEN_PATH",
            properties={
                "predicted_state": {"spatial_outcome": {"deadlock": False, "progress": 1.0}}
            },
        ),
    ]

    result = await planner.mcts_evaluate_and_select(
        goal=goal,
        candidate_actions=candidates,
        config=MCTSConfig(max_simulations=8, max_depth=2),
    )

    assert result.candidate_id == "act_step_open"
    assert result.utility_score > 0.5


@pytest.mark.asyncio
async def test_evaluate_and_select_dual_mode_with_mcts(
    hcir_setup: tuple[HCIRWorkspaceState, KernelServices],
) -> None:
    ws, services = hcir_setup

    ws.upsert_node(
        PhysicalEntityNode(
            id="avatar",
            entity_name="avatar",
            properties={"position": [2, 2], "is_avatar": True, "movable": True, "passable": False},
        )
    )

    planner = CounterfactualPlanner(ws, services)
    goal = GoalNode(id="g_test", description="Test dual mode")

    candidates = [
        ActionNode(
            id="act_1",
            intent="ACTION_ONE",
            properties={"can_interact": True},
            modality=ActionModality.MANIPULATION,
        ),
        ActionNode(
            id="act_2",
            intent="ACTION_TWO",
            properties={"can_interact": False},
            modality=ActionModality.COGNITIVE,
        ),
    ]

    # Legacy mode (use_mcts=False)
    res_legacy = await planner.evaluate_and_select(
        goal=goal,
        candidate_actions=candidates,
        horizon=1,
        use_mcts=False,
    )
    assert res_legacy.candidate_id == "act_1"

    # MCTS mode (use_mcts=True)
    res_mcts = await planner.evaluate_and_select(
        goal=goal,
        candidate_actions=candidates,
        horizon=2,
        use_mcts=True,
    )
    assert res_mcts.candidate_id == "act_1"
    assert res_mcts.utility_score > 0.0
