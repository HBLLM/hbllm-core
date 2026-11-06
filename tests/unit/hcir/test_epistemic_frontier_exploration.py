"""Unit tests for EpistemicFrontierDetector and epistemic exploration in HCIR."""

import numpy as np

from hbllm.hcir.graph import GoalNode, HCIREdgeType
from hbllm.hcir.subgoal_decomposer import EpistemicFrontierDetector, HierarchicalGoalDecomposer
from hbllm.hcir.workspace import HCIRWorkspaceState


def test_epistemic_frontier_detection() -> None:
    """Verify that EpistemicFrontierDetector correctly extracts and ranks boundary cells."""
    grid_shape = (10, 10)
    unobserved_mask = np.zeros(grid_shape, dtype=bool)
    unobserved_mask[:, 5:] = True  # columns 5..9 are unobserved

    avatar_pos = (5, 2)
    barrier_cells = {(3, 4), (4, 4)}

    frontiers = EpistemicFrontierDetector.detect_frontiers(
        avatar_pos=avatar_pos,
        unobserved_mask=unobserved_mask,
        barrier_cells=barrier_cells,
        grid_shape=grid_shape,
        step_size=1,
    )

    assert len(frontiers) > 0
    for (r, c), score in frontiers:
        assert c == 4
        assert (r, c) not in barrier_cells
        assert score > 0.0

    top_pos, _top_score = frontiers[0]
    assert abs(top_pos[0] - avatar_pos[0]) <= 2


def test_decompose_goal_with_epistemic_frontier() -> None:
    """Verify that HierarchicalGoalDecomposer generates an epistemic exploration GoalNode when path is occluded."""
    workspace = HCIRWorkspaceState()
    decomposer = HierarchicalGoalDecomposer()

    primary_goal = GoalNode(
        id="primary_exit",
        description="Escape maze",
        priority=1.0,
        properties={"target_position": (9, 9)},
    )

    avatar_pos = (1, 1)
    grid_shape = (10, 10)

    unobserved_mask = np.zeros(grid_shape, dtype=bool)
    unobserved_mask[4:, :] = True  # rows 4..9 unobserved

    barrier_cells: set[tuple[int, int]] = set()

    active_goal = decomposer.decompose_goal(
        workspace=workspace,
        primary_goal=primary_goal,
        avatar_pos=avatar_pos,
        barrier_cells=barrier_cells,
        grid_shape=grid_shape,
        candidate_subgoals=[],
        unobserved_mask=unobserved_mask,
    )

    assert active_goal.id != primary_goal.id
    assert active_goal.properties.get("is_epistemic") is True
    assert "epistemic_frontier" in active_goal.id
    t_pos = active_goal.properties.get("target_position")
    assert t_pos is not None
    assert t_pos[0] == 3

    edges = workspace.graph.edges_from(primary_goal.id)
    dep_edges = [e for e in edges if e.edge_type == HCIREdgeType.DEPENDS_ON]
    assert len(dep_edges) > 0
    assert active_goal.id in dep_edges[0].targets
