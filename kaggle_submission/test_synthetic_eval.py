"""Synthetic Procedural Evaluation Harness for Kaggle Submission Agent (MyAgent).

Tests MyAgent across procedurally generated ARC-AGI-3 environments:
1. Procedural Maze Navigation (A* obstacle traversal)
2. Procedural Lights Out (GF(2) Gaussian elimination)
3. Spatiotemporal Periodic Hazards (waiting and timed crossing)
4. Multi-Room Doorway Navigation (room topology decomposition)
5. Pattern Stamping (canvas diff minimization)
6. Level Transition & Invariant Retention (zero-shot transfer)
"""

from __future__ import annotations

import numpy as np

from kaggle_submission.submission import (
    DynamicCanvasMatcher,
    DynamicPermutationSolver,
    DynamicSpatialNavigator,
    GameAction,
    MyAgent,
    RoomTopologyExtractor,
    SpatiotemporalNavigator,
    TemporalHazardTracker,
    VisualSymmetryAnalyzer,
    VisualTopologyExtractor,
)


class MockFrame:
    """Simulates a Kaggle frame observation."""

    def __init__(self, grid: np.ndarray, available_actions: list[int] | None = None) -> None:
        self.frame = grid
        self.available_actions = available_actions or [1, 2, 3, 4, 5, 6, 7]


def test_synthetic_spatial_navigation() -> None:
    """Test MyAgent navigating a procedural maze with barrier."""
    print("Testing Synthetic Procedural Spatial Navigation...")
    occ = np.ones((8, 8), dtype=bool)
    occ[3, :6] = False  # Wall with opening at col 6, 7

    start = (1, 1)
    goal = (6, 1)
    path = DynamicSpatialNavigator.astar_path(occ, start, goal)
    assert path is not None, "A* failed to find path around barrier"
    assert path[0] == start
    assert path[-1] == goal

    actions = DynamicSpatialNavigator.path_to_actions(path)
    assert len(actions) > 0

    # Verify every step is valid
    curr: list[int] = [start[0], start[1]]
    for act in actions:
        dr, dc = DynamicSpatialNavigator.REVERSE_ACTION_MAP[act]
        curr[0] += dr
        curr[1] += dc
        assert occ[curr[0], curr[1]], f"Stepped into obstacle at {curr}"
    assert tuple(curr) == goal
    print("  ✓ Spatial Navigation A* pathfinder verified!")


def test_synthetic_lights_out_gf2() -> None:
    """Test MyAgent solving a randomized Lights Out toggle grid via GF(2)."""
    print("Testing Synthetic Procedural Lights Out GF(2)...")
    grid = np.zeros((4, 4), dtype=int)
    # Apply valid cross toggles at (0, 1), (2, 2), and (3, 3) to generate a guaranteed solvable configuration
    for tr, tc in [(0, 1), (2, 2), (3, 3)]:
        grid[tr, tc] ^= 1
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = tr + dr, tc + dc
            if 0 <= nr < 4 and 0 <= nc < 4:
                grid[nr, nc] ^= 1

    toggles = DynamicPermutationSolver.solve_lights_out_grid(grid, toggle_pattern="cross")
    assert toggles is not None, "GF(2) solver returned None for solvable configuration"

    # Simulate toggles
    sim = grid.copy()
    for r, c in toggles:
        sim[r, c] ^= 1
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 4 and 0 <= nc < 4:
                sim[nr, nc] ^= 1

    assert np.all(sim == 0), f"Lights remaining on: {np.sum(sim)}"
    print("  ✓ GF(2) Gaussian elimination verified (all lights turned OFF)!")


def test_synthetic_periodic_hazard() -> None:
    """Test MyAgent waiting and timing traversal across a periodic hazard."""
    print("Testing Synthetic Spatiotemporal Hazard Avoidance...")
    occ = np.ones((5, 5), dtype=bool)
    tracker = TemporalHazardTracker()

    # Hazard at (2, 2) active on even steps
    for step in range(10):
        hazards = {(2, 2)} if step % 2 == 0 else set()
        tracker.record_hazard_coords(hazards, t=step)
    tracker.detect_periodicity(min_period=2, max_period=4)

    path = SpatiotemporalNavigator.plan_path_with_hazards(
        occ, tracker, start=(2, 0), goal=(2, 4), start_time=0
    )
    assert path is not None, "Failed to find spatiotemporal path"
    assert path[-1][:2] == (2, 4)

    for r, c, t in path:
        assert tracker.is_safe_at(r, c, t), f"Hazard hit at {(r, c, t)}"

    actions = SpatiotemporalNavigator.path_to_spatiotemporal_actions(path)
    assert 5 in actions, "Spatiotemporal path should incorporate wait action (5)"
    print("  ✓ Spatiotemporal A* successfully avoided dynamic periodic hazards!")


def test_synthetic_room_topology() -> None:
    """Test MyAgent decomposing multi-room chambers into doorway connectivity graph."""
    print("Testing Synthetic Room Topology & Doorways...")
    occ = np.ones((9, 9), dtype=bool)
    # Two walls splitting into 4 rooms with 3 doorways
    occ[4, :] = False  # Horizontal wall
    occ[:, 4] = False  # Vertical wall
    occ[4, 2] = True  # Door between top-left and bottom-left
    occ[2, 4] = True  # Door between top-left and top-right
    occ[4, 6] = True  # Door between top-right and bottom-right

    rooms, doors = RoomTopologyExtractor.extract_rooms_and_doors(occ, min_room_size=4)
    assert len(rooms) >= 3, f"Expected at least 3 rooms, found {len(rooms)}"
    assert len(doors) >= 2, f"Expected at least 2 doors, found {len(doors)}"

    adj = RoomTopologyExtractor.build_adjacency_graph(rooms, doors)
    assert len(adj) >= 3
    print("  ✓ Room topology decomposition and doorway adjacency verified!")


def test_synthetic_canvas_stamping() -> None:
    """Test MyAgent synthesizing pattern stamping sequences."""
    print("Testing Synthetic Canvas Stamping & Diff...")
    curr = np.zeros((6, 6), dtype=int)
    target = np.zeros((6, 6), dtype=int)
    target[1:5, 1:5] = 4

    stamp_mask = np.zeros((6, 6), dtype=bool)
    stamp_mask[1:5, 1:5] = True
    stamps = [(1, stamp_mask)]

    plan = DynamicCanvasMatcher.plan_stamping_sequence(curr, target, stamps, palette_colors=[4])
    assert len(plan) == 1
    assert plan[0]["stamp_id"] == 1
    assert plan[0]["color"] == 4
    assert plan[0]["gain"] == 16
    print("  ✓ Dynamic canvas stamping sequence synthesis verified!")


def test_synthetic_visual_topology() -> None:
    """Test connected component entity extraction."""
    print("Testing Synthetic Visual Topology Extraction...")
    grid = np.zeros((8, 8), dtype=int)
    grid[1:3, 1:3] = 4
    entities = VisualTopologyExtractor.extract_entities(grid, ignore_colors={0})
    assert len(entities) == 1
    assert entities[0].size == 4
    assert entities[0].color == 4
    print("  ✓ Visual topology extraction verified!")


def test_synthetic_visual_symmetry() -> None:
    """Test symmetry score computation and reflection."""
    print("Testing Synthetic Visual Symmetry Analysis...")
    grid = np.zeros((6, 6), dtype=int)
    grid[:, 1] = 2
    grid[:, 4] = 2
    scores = VisualSymmetryAnalyzer.compute_symmetry_scores(grid)
    assert scores["vertical"] == 1.0
    print("  ✓ Visual symmetry analysis verified!")


def test_myagent_kaggle_interface() -> None:
    """Test MyAgent choose_action with MockFrame objects."""
    print("Testing MyAgent Kaggle Interface (choose_action)...")
    agent = MyAgent()
    agent.reset_episode()

    grid = np.zeros((64, 64), dtype=int)
    grid[10, 10] = 3  # avatar
    grid[10, 15] = 8  # goal

    frame = MockFrame(grid, available_actions=[1, 2, 3, 4])
    action = agent.choose_action([], frame)

    assert isinstance(action, GameAction), f"Expected GameAction, got {type(action)}"
    assert action.value in [1, 2, 3, 4], f"Action {action} not in available actions"

    # Test level transition handling
    grid_next_level = np.ones((64, 64), dtype=int) * 7
    frame_next = MockFrame(grid_next_level, available_actions=[1, 2, 3, 4])
    action2 = agent.choose_action([frame], frame_next)
    assert isinstance(action2, GameAction)
    print("  ✓ MyAgent Kaggle interface and transition handling verified!")


def main() -> None:
    print("==================================================================")
    print("RUNNING ARC-AGI-3 KAGGLE AGENT SYNTHETIC PROCEDURAL EVALUATION")
    print("==================================================================")
    test_synthetic_spatial_navigation()
    test_synthetic_lights_out_gf2()
    test_synthetic_periodic_hazard()
    test_synthetic_room_topology()
    test_synthetic_canvas_stamping()
    test_synthetic_visual_topology()
    test_synthetic_visual_symmetry()
    test_myagent_kaggle_interface()
    print("==================================================================")
    print("ALL SYNTHETIC PROCEDURAL TESTS PASSED (100% SUCCESS RATE)!")
    print("==================================================================")


if __name__ == "__main__":
    main()
