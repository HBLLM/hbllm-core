"""Unit tests for CounterfactualPlanner with spatial kinematics and deadlock pruning."""

from __future__ import annotations

import pytest

from hbllm.hcir.counterfactual_planner import CounterfactualPlanner
from hbllm.hcir.graph import ActionNode, GoalNode, PhysicalEntityNode
from hbllm.hcir.kernel.capability_resolver import CapabilityResolver
from hbllm.hcir.kernel.scheduler import KernelInstructionScheduler
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.workspace import HCIRWorkspaceState


@pytest.mark.asyncio
async def test_spatial_counterfactual_planner_selects_goal_directed_action() -> None:
    """Verify CounterfactualPlanner evaluates candidate actions and chooses the action moving to goal."""
    ws = HCIRWorkspaceState()
    services = KernelServices(
        workspace=ws,
        transaction_manager=TransactionManager(ws),
        capability_resolver=CapabilityResolver(),
        scheduler=KernelInstructionScheduler(),
    )

    # Populate workspace with avatar at (4, 4) and goal at (2, 4) [ABOVE]
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
            properties={"position": [2, 4], "is_goal": True, "movable": False, "passable": True},
        )
    )

    planner = CounterfactualPlanner(ws, services)
    goal = GoalNode(id="g_nav", description="Reach goal location")

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

    best = await planner.evaluate_and_select(goal, candidates)

    assert best is not None
    assert best.action.id == "act_up"
    assert best.utility_score > 0.85


@pytest.mark.asyncio
async def test_spatial_counterfactual_planner_prunes_deadlock() -> None:
    """Verify CounterfactualPlanner drops candidate branch that causes a corner deadlock."""
    ws = HCIRWorkspaceState()
    services = KernelServices(
        workspace=ws,
        transaction_manager=TransactionManager(ws),
        capability_resolver=CapabilityResolver(),
        scheduler=KernelInstructionScheduler(),
    )

    # Setup room with corner at (1, 1). Avatar is at (3, 1). Box is at (2, 1).
    # Moving UP pushes box to (1, 1) [corner deadlock between wall at row 0 and wall at col 0]
    # Moving RIGHT leaves box alone and explores safe area.
    ws.upsert_node(
        PhysicalEntityNode(
            id="avatar",
            entity_name="avatar",
            properties={"position": [3, 1], "is_avatar": True, "movable": True, "passable": False},
        )
    )
    ws.upsert_node(
        PhysicalEntityNode(
            id="box_1",
            entity_name="box",
            properties={
                "position": [2, 1],
                "movable": True,
                "passable": False,
                "affordances": ["PUSHABLE"],
            },
        )
    )
    ws.upsert_node(
        PhysicalEntityNode(
            id="goal",
            entity_name="goal",
            properties={"position": [7, 7], "is_goal": True, "movable": False, "passable": True},
        )
    )

    # Set barrier cells in workspace digital twin via world kernel or properties
    barrier_cells = [[0, c] for c in range(8)] + [[r, 0] for r in range(8)]

    planner = CounterfactualPlanner(ws, services)
    goal = GoalNode(id="g_sokoban", description="Solve puzzle without corner deadlock")

    candidates = [
        # This push causes deadlock:
        ActionNode(
            id="act_push_up",
            intent="MOVE_UP",
            properties={"delta_r": -1, "delta_c": 0, "barrier_cells": barrier_cells},
        ),
        # This move is safe:
        ActionNode(
            id="act_move_right",
            intent="MOVE_RIGHT",
            properties={"delta_r": 0, "delta_c": 1, "barrier_cells": barrier_cells},
        ),
    ]

    best = await planner.evaluate_and_select(goal, candidates)

    assert best is not None
    # act_push_up should have been penalized heavily due to corner deadlock
    assert best.action.id == "act_move_right"


def test_predict_outcome_click_destruction() -> None:
    """Verify CounterfactualPlanner.predict_outcome clears clicked obstacles on Action 6."""
    import numpy as np

    grid = np.zeros((5, 5), dtype=int)
    grid[2, 2] = 4  # Obstacle

    out = CounterfactualPlanner.predict_outcome(
        grid=grid,
        action_id=6,
        action_data={"x": 2, "y": 2},
        destructible_colors={4},
    )

    assert out[2, 2] == 0  # Cleared to background


def test_predict_outcome_push_dynamics() -> None:
    """Verify CounterfactualPlanner.predict_outcome displaces movable objects when pushed."""
    from types import SimpleNamespace

    import numpy as np

    grid = np.zeros((5, 5), dtype=int)
    grid[2, 1] = 1  # Avatar
    grid[2, 2] = 2  # Movable box
    # (2, 3) is 0 (empty)

    action_model = SimpleNamespace(delta=(0, 1), confidence=0.9)

    out = CounterfactualPlanner.predict_outcome(
        grid=grid,
        action_id=4,  # Move right
        action_model=action_model,
        avatar_color=1,
        movable_colors={2},
    )

    assert out[2, 1] == 0  # Old avatar pos empty
    assert out[2, 2] == 1  # Avatar moved here
    assert out[2, 3] == 2  # Box pushed here


def test_predict_outcome_cellular_gravity() -> None:
    """Verify CounterfactualPlanner.predict_outcome settles unsupported movable pixels downward."""
    from types import SimpleNamespace

    import numpy as np

    grid = np.zeros((5, 5), dtype=int)
    grid[1, 2] = 3  # Falling sand/block
    # Rows 2, 3, 4 are 0 (empty)

    action_model = SimpleNamespace(delta=(0, 0), confidence=0.8)

    out = CounterfactualPlanner.predict_outcome(
        grid=grid,
        action_id=0,
        action_model=action_model,
        avatar_color=1,
        movable_colors={3},
        gravity=(1, 0),
    )

    assert out[1, 2] == 0  # Fell from row 1
    assert out[4, 2] == 3  # Settled at floor (row 4)
