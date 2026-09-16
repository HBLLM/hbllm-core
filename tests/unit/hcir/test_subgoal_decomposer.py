"""Unit tests for HierarchicalGoalDecomposer in core HCIR."""

from __future__ import annotations

from hbllm.hcir.graph import GoalNode, HCIREdgeType
from hbllm.hcir.subgoal_decomposer import HierarchicalGoalDecomposer
from hbllm.hcir.workspace import HCIRWorkspaceState


def test_decomposer_unobstructed_returns_primary_goal() -> None:
    """Verify decomposer returns primary goal when unobstructed."""
    ws = HCIRWorkspaceState()
    decomposer = HierarchicalGoalDecomposer()

    primary_goal = GoalNode(
        id="g_exit",
        description="Reach the exit zone",
        properties={"target_position": (1, 4)},
    )
    ws.upsert_node(primary_goal)

    active_goal = decomposer.decompose_goal(
        workspace=ws,
        primary_goal=primary_goal,
        avatar_pos=(4, 4),
        barrier_cells=set(),
        grid_shape=(8, 8),
        candidate_subgoals=[{"id": "switch_1", "position": (4, 2)}],
    )

    assert active_goal.id == "g_exit"


def test_decomposer_obstructed_synthesizes_reachable_subgoal() -> None:
    """Verify decomposer generates a prerequisite subgoal when primary goal is blocked."""
    ws = HCIRWorkspaceState()
    decomposer = HierarchicalGoalDecomposer()

    primary_goal = GoalNode(
        id="g_exit",
        description="Reach the exit zone",
        properties={"target_position": (1, 4)},
    )
    ws.upsert_node(primary_goal)

    # Wall completely blocking access to row 1
    barrier_cells = {(2, c) for c in range(8)}

    active_goal = decomposer.decompose_goal(
        workspace=ws,
        primary_goal=primary_goal,
        avatar_pos=(4, 4),
        barrier_cells=barrier_cells,
        grid_shape=(8, 8),
        candidate_subgoals=[
            {"id": "switch_yellow", "position": (4, 2), "description": "Step on yellow switch"}
        ],
    )

    assert active_goal.id != "g_exit"
    assert active_goal.id.startswith("subgoal_")
    assert active_goal.properties["target_position"] == (4, 2)

    # Verify DEPENDS_ON edge in workspace graph
    edges = ws.graph.edges_from(primary_goal.id)
    depends_edges = [e for e in edges if e.edge_type == HCIREdgeType.DEPENDS_ON]
    assert len(depends_edges) == 1
    assert active_goal.id in depends_edges[0].targets


def test_decomposer_resolving_subgoal_advances_to_primary_goal() -> None:
    """Verify that resolving the prerequisite allows progression to the primary goal."""
    ws = HCIRWorkspaceState()
    decomposer = HierarchicalGoalDecomposer()

    primary_goal = GoalNode(
        id="g_exit",
        description="Reach the exit zone",
        properties={"target_position": (1, 4)},
    )
    ws.upsert_node(primary_goal)

    # 1. Obstructed -> returns subgoal
    barrier_cells = {(2, c) for c in range(8)}
    subgoal = decomposer.decompose_goal(
        workspace=ws,
        primary_goal=primary_goal,
        avatar_pos=(4, 4),
        barrier_cells=barrier_cells,
        grid_shape=(8, 8),
        candidate_subgoals=[{"id": "key_blue", "position": (4, 2)}],
    )
    assert subgoal.id.startswith("subgoal_")

    # 2. Resolve the subgoal and open the door (clear barrier)
    decomposer.resolve_subgoal(ws, subgoal.id)
    cleared_barriers = set()  # Door opens!

    # 3. Next decomposition step returns the primary goal!
    next_goal = decomposer.decompose_goal(
        workspace=ws,
        primary_goal=primary_goal,
        avatar_pos=(4, 2),
        barrier_cells=cleared_barriers,
        grid_shape=(8, 8),
    )
    assert next_goal.id == "g_exit"
