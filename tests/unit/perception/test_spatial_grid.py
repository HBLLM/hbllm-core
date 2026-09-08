"""
Unit tests for EpistemicSpatialGrid and CausalObstacleResolver.
"""

from __future__ import annotations

from hbllm.actions.causal_planner import (
    CausalObstacleResolver,
    TopologicalFrontierNavigator,
)
from hbllm.hcir.graph import CognitiveGraph, PhysicalEntityNode
from hbllm.perception.spatial_grid import EpistemicSpatialGrid


def test_epistemic_spatial_grid_visibility_and_bounds() -> None:
    """Verify visibility tracking and dynamic bounding box computation."""
    graph = CognitiveGraph()
    grid = EpistemicSpatialGrid(graph=graph, default_bounds=10)

    grid.mark_visited((1, 1))
    grid.mark_seen([(1, 1), (1, 2), (2, 1), (2, 2)])

    assert (1, 1) in grid.visited_cells
    assert (2, 2) in grid.seen_cells
    assert (3, 3) not in grid.seen_cells

    # Add entity beyond default bounds
    wall_node = PhysicalEntityNode(
        id="wall_15_15",
        entity_name="wall",
        entity_type="wall",
        properties={"coords": (15, 15), "passable": False},
    )
    graph.add_node(wall_node)

    max_x, max_y = grid.get_bounds()
    assert max_x >= 17
    assert max_y >= 17


def test_epistemic_spatial_grid_partition_and_frontiers() -> None:
    """Verify cell partitioning into walls, doors, and frontiers."""
    graph = CognitiveGraph()
    grid = EpistemicSpatialGrid(graph=graph, default_bounds=(8, 8))

    # Add agent
    agent_node = PhysicalEntityNode(
        id="agent_primary",
        entity_name="agent",
        entity_type="agent",
        properties={"coords": (1, 1)},
    )
    graph.add_node(agent_node)

    # Add open door at (3, 1) and closed door at (1, 3)
    open_door = PhysicalEntityNode(
        id="door_open",
        entity_name="door",
        entity_type="door",
        properties={"coords": (3, 1), "state": "open", "is_door": True, "passable": True},
    )
    closed_door = PhysicalEntityNode(
        id="door_closed",
        entity_name="door",
        entity_type="door",
        properties={"coords": (1, 3), "state": "closed", "is_door": True, "passable": False},
    )
    graph.add_node(open_door)
    graph.add_node(closed_door)

    partition = grid.partition_cells()
    assert (3, 1) in partition.open_doors
    assert (1, 3) in partition.closed_doors

    # Mark cells around agent as seen
    grid.mark_visited((1, 1))
    grid.mark_seen([(1, 1), (2, 1), (3, 1)])
    grid.sync_to_agent_node()

    # Frontier should discover through the open door (3, 1) to unvisited cell (4, 1)
    frontier = grid.find_frontiers(agent_pos=(1, 1))
    assert frontier is not None
    target_cell, portal_node = frontier
    assert target_cell == (4, 1)
    assert portal_node is not None
    assert portal_node.id == "door_open"


def test_causal_obstacle_resolver() -> None:
    """Verify obstacle detection and safe drop cell candidate calculation."""
    graph = CognitiveGraph()
    resolver = CausalObstacleResolver()

    # Target is at (4, 2). Blocked by a box at (3, 2).
    target_door = PhysicalEntityNode(
        id="target_door",
        entity_name="door",
        entity_type="door",
        properties={"coords": (4, 2), "state": "closed", "is_door": True, "passable": False},
    )
    blocking_box = PhysicalEntityNode(
        id="blocking_box",
        entity_name="box",
        entity_type="box",
        properties={"coords": (3, 2), "pickupable": True, "passable": False},
    )
    graph.add_node(target_door)
    graph.add_node(blocking_box)

    # Agent at (1, 2) facing East (dir=0)
    blocker = resolver.find_blocking_obstacle(
        graph=graph,
        agent_pos=(1, 2),
        agent_dir=0,
        target_pos=(4, 2),
    )
    assert blocker is not None
    assert blocker.id == "blocking_box"

    # Verify safe drop candidate excludes the doorway (4, 2) and blocker pos (3, 2)
    candidates = resolver.find_safe_drop_candidates(
        graph=graph,
        curr_pos=(2, 2),
        critical_positions={(4, 2), (3, 2)},
    )
    assert (4, 2) not in candidates
    assert (3, 2) not in candidates
    assert len(candidates) > 0


def test_topological_frontier_navigator_hysteresis() -> None:
    """Verify frontier commitment and hysteresis invalidation."""
    navigator = TopologicalFrontierNavigator()
    assert navigator.active_frontier is None

    # Commit to (4, 1)
    navigator.commit((4, 1))
    assert navigator.is_valid(agent_pos=(1, 1), visited_cells={(1, 1)}) is True

    # Reaching or visiting target invalidates commitment
    assert navigator.is_valid(agent_pos=(4, 1), visited_cells={(1, 1), (4, 1)}) is False
    assert navigator.active_frontier is None
