"""Unit Test Suite for Milestone 3 Phase 2: Topological Graph & Geodesic Maze Induction."""

import ast
from pathlib import Path

import numpy as np

from experiments.benchmarks.task_generators.topological_maze_generator import (
    TopologicalMazeTaskGenerator,
)
from hbllm.hcir.world.representation_expansion import RepresentationExpansionEngine
from hbllm.hcir.world.topological_graph_induction import (
    GeodesicPathOperator,
    GridTopologyGraph,
    InductiveTopologicalGraphSynthesizer,
)


def test_generator_ast_independence() -> None:
    """Verify that TopologicalMazeTaskGenerator has 0 imports from hbllm/hcir solver internals."""
    gen_file = (
        Path(__file__).resolve().parents[3]
        / "experiments"
        / "benchmarks"
        / "task_generators"
        / "topological_maze_generator.py"
    )
    assert gen_file.exists(), f"File {gen_file} not found"

    tree = ast.parse(gen_file.read_text(encoding="utf-8"))
    forbidden_prefixes = ("hbllm", "core.hbllm")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prefix in forbidden_prefixes:
                    assert not alias.name.startswith(prefix), (
                        f"Forbidden import '{alias.name}' found in task generator"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for prefix in forbidden_prefixes:
                    assert not node.module.startswith(prefix), (
                        f"Forbidden import from '{node.module}' found in task generator"
                    )


def test_grid_topology_graph_shortest_path() -> None:
    """Verify BFS shortest path computation and unreachable detection."""
    # 5x5 grid with wall barrier
    grid = np.zeros((5, 5), dtype=int)
    grid[1, 1:4] = 1  # Horizontal wall
    grid[0, 2] = 2  # Start
    grid[2, 2] = 3  # Goal behind wall

    path = GridTopologyGraph.shortest_path(
        grid=grid,
        start_coord=(0, 2),
        goal_coord=(2, 2),
        traversable_colors={0},
        connectivity="4_connected",
    )
    assert path is not None
    assert path[0] == (0, 2)
    assert path[-1] == (2, 2)
    # Must route around wall: length should be at least 5 steps
    assert len(path) >= 5

    # Completely sealed goal
    grid[2, 1] = 1
    grid[2, 3] = 1
    grid[3, 2] = 1
    unreachable = GridTopologyGraph.shortest_path(
        grid=grid,
        start_coord=(0, 2),
        goal_coord=(2, 2),
        traversable_colors={0},
        connectivity="4_connected",
    )
    assert unreachable is None


def test_ambiguity_detection_multiple_shortest_paths() -> None:
    """Verify detection of ambiguous multiple shortest paths."""
    # 5x5 symmetric grid with center obstacle
    grid = np.zeros((5, 5), dtype=int)
    grid[2, 2] = 1  # Center obstacle
    grid[2, 0] = 2  # Start
    grid[2, 4] = 3  # Goal

    op = GeodesicPathOperator(
        name="test_geodesic_op",
        start_color=2,
        goal_color=3,
        path_color=4,
        traversable_colors=(0,),
        obstacle_colors=(1,),
    )

    # There are two symmetric shortest paths around the center obstacle (top or bottom)
    assert op.has_multiple_shortest_paths(grid) is True


def test_inductive_synthesizer_distinct_and_same_landmarks() -> None:
    """Verify induction for distinct landmarks and same-color landmarks."""
    # Case A: Distinct landmarks
    x0 = np.zeros((6, 6), dtype=int)
    x0[1, 1] = 2  # Start
    x0[4, 4] = 3  # Goal
    x0[2, 1:4] = 1  # Wall
    y0 = np.copy(x0)
    # Shortest path routes through (1, 4), (2, 4), (3, 4)
    y0[1, 2] = 5
    y0[1, 3] = 5
    y0[1, 4] = 5
    y0[2, 4] = 5
    y0[3, 4] = 5

    x1 = np.zeros((6, 6), dtype=int)
    x1[0, 0] = 2
    x1[0, 3] = 3
    y1 = np.copy(x1)
    y1[0, 1] = 5
    y1[0, 2] = 5

    op_distinct = InductiveTopologicalGraphSynthesizer.synthesize_from_residuals(
        [(x0, y0), (x1, y1)]
    )
    assert op_distinct is not None
    assert op_distinct.start_color == 2
    assert op_distinct.goal_color == 3
    assert op_distinct.path_color == 5

    # Case B: Same-color landmarks
    xs0 = np.zeros((5, 5), dtype=int)
    xs0[0, 0] = 7
    xs0[0, 4] = 7
    ys0 = np.copy(xs0)
    ys0[0, 1:4] = 4

    xs1 = np.zeros((5, 5), dtype=int)
    xs1[2, 1] = 7
    xs1[2, 3] = 7
    ys1 = np.copy(xs1)
    ys1[2, 2] = 4

    op_same = InductiveTopologicalGraphSynthesizer.synthesize_from_residuals(
        [(xs0, ys0), (xs1, ys1)]
    )
    assert op_same is not None
    assert op_same.start_color == 7
    assert op_same.goal_color == 7
    assert op_same.path_color == 4


def test_end_to_end_maze_benchmark_suite_generation() -> None:
    """Verify suite generation and end-to-end synthesis across families."""
    suite = TopologicalMazeTaskGenerator.generate_benchmark_suite(seed=123)
    assert len(suite) == 20

    # Test one instance of each family
    for fam in TopologicalMazeTaskGenerator.FAMILIES:
        task = next(t for t in suite if t.family == fam)
        trace = RepresentationExpansionEngine.synthesize_and_predict(
            list(task.train_pairs), task.test_input, task_id=task.task_id, family=task.family
        )
        if fam == "ambiguous_multi_path_abstention":
            assert trace.phase2_epistemic_uncertainty == 1.0
        else:
            assert trace.phase2_solved is True
            assert trace.phase2_predicted_test is not None
            assert np.array_equal(trace.phase2_predicted_test, task.test_output)
