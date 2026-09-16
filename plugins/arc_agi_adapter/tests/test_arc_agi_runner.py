"""Unit tests for ARC-AGI Benchmark Runner & Relational Transformation Engine."""

from __future__ import annotations

import pytest

from plugins.arc_agi_adapter.arc_agi_runner import (
    ARCBenchmarkReport,
    ARCGrid,
    ARCRelationalSolver,
    GridTopologyExtractor,
)


def test_arc_grid_primitives() -> None:
    """Verify ARCGrid indexing, cloning, cropping, and accuracy scoring."""
    raw = [
        [0, 1, 0],
        [1, 2, 1],
        [0, 1, 0],
    ]
    grid = ARCGrid.from_list(raw)
    assert grid.height == 3
    assert grid.width == 3
    assert grid.get(1, 1) == 2
    assert grid.get(10, 10, default=-1) == -1

    clone = grid.clone()
    assert clone == grid
    clone.set(1, 1, 9)
    assert clone != grid
    assert clone.get(1, 1) == 9

    cropped = grid.crop(1, 1, 2, 2)
    assert cropped.height == 2
    assert cropped.width == 2
    assert cropped.cells == [[2, 1], [1, 0]]

    # Accuracy and Brier scores
    assert grid.pixel_accuracy(grid) == 1.0
    assert grid.brier_score(grid, confidence=1.0) == 0.0
    assert grid.pixel_accuracy(clone) == pytest.approx(8 / 9)


def test_grid_topology_extraction_and_cognitive_graph() -> None:
    """Verify connected components, closed boundary frame detection, and CognitiveGraph mapping."""
    # 5x5 grid with a closed frame (color 1) enclosing a 1x1 void at (2, 2)
    raw = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    grid = ARCGrid.from_list(raw)
    objs = GridTopologyExtractor.extract_objects(grid)
    assert len(objs) == 1
    frame_obj = objs[0]
    assert frame_obj.color == 1
    assert frame_obj.area == 8
    assert frame_obj.is_frame is True
    assert (2, 2) in frame_obj.enclosed_coords

    # Lift to HCIR CognitiveGraph
    graph = GridTopologyExtractor.to_cognitive_graph(grid)
    assert len(graph._nodes) == 1
    node = list(graph._nodes.values())[0]
    assert node.properties["is_frame"] is True
    assert node.properties["enclosed_count"] == 1


def test_arc_boundary_containment_fill_task() -> None:
    """Verify ARC-AGI relational solver on Boundary Containment Fill (Stage D2)."""
    # Demonstration pair 1: hollow frame gets filled with yellow (4)
    train_in_1 = [
        [0, 0, 0, 0, 0],
        [0, 2, 2, 2, 0],
        [0, 2, 0, 2, 0],
        [0, 2, 2, 2, 0],
        [0, 0, 0, 0, 0],
    ]
    train_out_1 = [
        [0, 0, 0, 0, 0],
        [0, 2, 2, 2, 0],
        [0, 2, 4, 2, 0],
        [0, 2, 2, 2, 0],
        [0, 0, 0, 0, 0],
    ]

    # Test pair: hollow frame of different dimensions/location
    test_in = [
        [0, 0, 0, 0, 0, 0],
        [0, 2, 2, 2, 2, 0],
        [0, 2, 0, 0, 2, 0],
        [0, 2, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    test_out = [
        [0, 0, 0, 0, 0, 0],
        [0, 2, 2, 2, 2, 0],
        [0, 2, 4, 4, 2, 0],
        [0, 2, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    task = {
        "train": [{"input": train_in_1, "output": train_out_1}],
        "test": [{"input": test_in, "output": test_out}],
    }

    solver = ARCRelationalSolver()
    result = solver.solve_task(task, task_id="arc_containment_fill")

    assert result.exact_match is True
    assert result.pixel_accuracy == 1.0
    assert result.schema_name == "boundary_containment_fill"
    assert result.brier_score < 0.05


def test_arc_tool_reach_ray_extension_task() -> None:
    """Verify ARC-AGI relational solver on Tool Reach & Ray Extension (Stage D4)."""
    # Center tool anchor (color 3) shoots green beams to borders
    train_in = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    train_out = [
        [0, 0, 3, 0, 0],
        [0, 0, 3, 0, 0],
        [3, 3, 3, 3, 3],
        [0, 0, 3, 0, 0],
        [0, 0, 3, 0, 0],
    ]

    # Test pair: anchor at (1, 1) with obstacle at (1, 3)
    test_in = [
        [0, 0, 0, 0, 0],
        [0, 3, 0, 5, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    # Beam extends up to row 0, down to row 4, left to col 0, and right until stopping at obstacle 5!
    test_out = [
        [0, 3, 0, 0, 0],
        [3, 3, 3, 5, 0],
        [0, 3, 0, 0, 0],
        [0, 3, 0, 0, 0],
        [0, 3, 0, 0, 0],
    ]

    task = {
        "train": [{"input": train_in, "output": train_out}],
        "test": [{"input": test_in, "output": test_out}],
    }

    solver = ARCRelationalSolver()
    result = solver.solve_task(task, task_id="arc_tool_ray")

    assert result.exact_match is True
    assert result.pixel_accuracy == 1.0
    assert result.schema_name == "tool_reach_ray_extension"


def test_arc_obstacle_clearance_displacement_task() -> None:
    """Verify ARC-AGI relational solver on Obstacle Clearance Displacement (Stage D1/D2)."""
    # Dot (color 8) translates downwards until resting against floor/obstacle
    train_in = [
        [0, 8, 0],
        [0, 0, 0],
        [0, 0, 0],
        [5, 5, 5],
    ]
    train_out = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 8, 0],
        [5, 5, 5],
    ]

    test_in = [
        [8, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [5, 5, 5],
    ]
    test_out = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [8, 0, 0],
        [5, 5, 5],
    ]

    task = {
        "train": [{"input": train_in, "output": train_out}],
        "test": [{"input": test_in, "output": test_out}],
    }

    solver = ARCRelationalSolver()
    result = solver.solve_task(task, task_id="arc_obstacle_displacement")

    assert result.exact_match is True
    assert result.pixel_accuracy == 1.0
    assert result.schema_name == "obstacle_clearance_displacement"


def test_arc_attribute_mapping_and_benchmark_report() -> None:
    """Verify 1-to-1 attribute mapping and benchmark report generation."""
    task_mapping = {
        "train": [
            {
                "input": [[1, 2], [2, 1]],
                "output": [[6, 7], [7, 6]],
            }
        ],
        "test": [
            {
                "input": [[2, 2], [1, 1]],
                "output": [[7, 7], [6, 6]],
            }
        ],
    }

    solver = ARCRelationalSolver()
    tasks = {"task_color_map": task_mapping}
    report = solver.run_benchmark(tasks)

    assert isinstance(report, ARCBenchmarkReport)
    assert report.total_tasks == 1
    assert report.tasks_solved == 1
    assert report.exact_accuracy == 1.0
    md = report.format_markdown()
    assert "ARC-AGI Cognitive Relational Benchmark Report" in md
    assert "attribute_color_mapping" in md
