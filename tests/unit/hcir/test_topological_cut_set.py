"""Unit tests for TopologicalCutSetAnalyzer."""

from __future__ import annotations

from hbllm.hcir.topological_cut_set import TopologicalCutSetAnalyzer


def test_unpartitioned_path() -> None:
    """Verify that an open grid is detected as not partitioned."""
    grid_shape = (10, 10)
    barriers: set[tuple[int, int]] = set()
    start = (1, 1)
    goal = (8, 8)

    res = TopologicalCutSetAnalyzer.analyze_cut_set(
        start=start, goal=goal, barrier_cells=barriers, grid_shape=grid_shape
    )
    assert not res.is_partitioned
    assert goal in res.start_component
    assert res.best_gate_cell is None


def test_partitioned_maze_detects_cut_set_and_gate() -> None:
    """Verify that a wall separating left and right halves is identified as cut-set."""
    grid_shape = (10, 10)
    # Vertical wall down column 5 from row 0 to 9
    barriers: set[tuple[int, int]] = {(r, 5) for r in range(10)}
    start = (2, 2)
    goal = (2, 8)

    res = TopologicalCutSetAnalyzer.analyze_cut_set(
        start=start, goal=goal, barrier_cells=barriers, grid_shape=grid_shape
    )
    assert res.is_partitioned
    assert goal not in res.start_component
    assert len(res.barrier_cut_set) > 0
    # The best gate cell should be at column 5, closest to row 2
    assert res.best_gate_cell is not None
    assert res.best_gate_cell[1] == 5
    assert res.best_gate_cell[0] == 2  # Closest to (2, 8)
    assert res.approach_cell == (2, 4)  # Accessible cell adjacent to (2, 5)
