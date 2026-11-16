"""Unit Test Suite for Milestone 3 Phase 2: Topological Graph & Geodesic Maze Induction."""

import ast
from pathlib import Path

import numpy as np

from hbllm.hcir.world.topological_graph_induction import GridTopologyGraph


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
