"""Unit tests for compound precondition subgoal decomposition in HCIR."""

from hbllm.hcir.graph import GoalNode
from hbllm.hcir.subgoal_decomposer import HierarchicalGoalDecomposer
from hbllm.hcir.workspace import HCIRWorkspaceState


def test_compound_precondition_decomposition() -> None:
    """Verify that HierarchicalGoalDecomposer resolves conjunctions of preconditions."""
    workspace = HCIRWorkspaceState()
    decomposer = HierarchicalGoalDecomposer()

    primary_goal = GoalNode(
        id="compound_gate",
        description="Unlock multi-attribute exit gate",
        priority=1.0,
        properties={"target_position": (9, 9)},
    )

    avatar_pos = (0, 0)
    current_state = {"rotation": 0, "color": 1, "shape": 2}
    required_preconditions = {"rotation": 90, "color": 4, "shape": 2}

    available_modifiers = [
        {"id": "rot_90", "property": "rotation", "value": 90, "position": (2, 2)},
        {"id": "col_4", "property": "color", "value": 4, "position": (5, 5)},
        {"id": "shape_3", "property": "shape", "value": 3, "position": (7, 7)},
    ]

    barrier_cells: set[tuple[int, int]] = set()
    grid_shape = (10, 10)

    # Step 1: Decompose compound preconditions
    active_subgoal = decomposer.decompose_compound_preconditions(
        workspace=workspace,
        primary_goal=primary_goal,
        avatar_pos=avatar_pos,
        current_state=current_state,
        required_preconditions=required_preconditions,
        available_modifiers=available_modifiers,
        barrier_cells=barrier_cells,
        grid_shape=grid_shape,
    )

    assert active_subgoal.id != primary_goal.id
    # Immediate subgoal should be rotation (closest modifier at (2, 2))
    assert "rotation" in active_subgoal.id
    assert active_subgoal.properties.get("target_position") == (2, 2)

    # Step 2: Simulate resolving rotation
    decomposer.resolve_subgoal(workspace, active_subgoal.id)
    current_state["rotation"] = 90

    # Step 3: Next decomposition should target color
    next_subgoal = decomposer.decompose_compound_preconditions(
        workspace=workspace,
        primary_goal=primary_goal,
        avatar_pos=(2, 2),
        current_state=current_state,
        required_preconditions=required_preconditions,
        available_modifiers=available_modifiers,
        barrier_cells=barrier_cells,
        grid_shape=grid_shape,
    )

    assert next_subgoal.id != primary_goal.id
    assert "color" in next_subgoal.id
    assert next_subgoal.properties.get("target_position") == (5, 5)

    # Step 4: Resolve color
    decomposer.resolve_subgoal(workspace, next_subgoal.id)
    current_state["color"] = 4

    # Step 5: All preconditions met -> returns primary_goal
    final_goal = decomposer.decompose_compound_preconditions(
        workspace=workspace,
        primary_goal=primary_goal,
        avatar_pos=(5, 5),
        current_state=current_state,
        required_preconditions=required_preconditions,
        available_modifiers=available_modifiers,
        barrier_cells=barrier_cells,
        grid_shape=grid_shape,
    )

    assert final_goal.id == primary_goal.id
