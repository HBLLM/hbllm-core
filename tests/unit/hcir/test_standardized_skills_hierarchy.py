"""Unit tests verifying the HCIR Standardized Skills Hierarchy and Subskills Building Blocks.

Validates that:
1. Every domain skill adheres to the BaseHierarchicalSkill protocol
2. Every skill exposes semantic_intent and canonical skill_name
3. Common subskills (DiscreteVectorTranslator, LatticeQuantizer, Raycaster2D,
   PerceptualClusterDetector, RemoteActuator, GF2LinearSolver, TemporalSyncPlanner)
   operate as composable, hierarchical building blocks.
"""

from __future__ import annotations

import inspect

import numpy as np

import hbllm.hcir.skills as skills_module
from hbllm.hcir.skills.base import BaseHierarchicalSkill
from hbllm.hcir.skills.common_subskills import (
    BlockPushPlanner,
    DiscreteVectorTranslator,
    GF2LinearSolver,
    LatticeQuantizer,
    PerceptualClusterDetector,
    Raycaster2D,
    RemoteActuator,
    TemporalSyncPlanner,
)
from hbllm.hcir.spatial_planner import SpatialActionIntent


def test_all_skills_inherit_and_conform_to_protocol():
    """Verify that all concrete skills in hbllm.hcir.skills subclass BaseHierarchicalSkill and implement protocol."""
    skill_classes = []
    for name, obj in inspect.getmembers(skills_module):
        if (
            inspect.isclass(obj)
            and issubclass(obj, BaseHierarchicalSkill)
            and obj is not BaseHierarchicalSkill
        ):
            skill_classes.append((name, obj))

    assert len(skill_classes) >= 20, f"Expected at least 20 skills, found {len(skill_classes)}"

    dummy_grid = np.zeros((64, 64), dtype=int)
    dummy_actions = [1, 2, 3, 4, 5, 6, 7]

    for name, cls in skill_classes:
        instance = cls()
        # Verify required attributes
        assert hasattr(instance, "skill_name") and isinstance(instance.skill_name, str), (
            f"{name} must define a string skill_name"
        )
        assert hasattr(instance, "semantic_intent") and isinstance(
            instance.semantic_intent, SpatialActionIntent
        ), f"{name} must define a valid SpatialActionIntent"

        # Verify can_handle callable signature
        assert callable(getattr(instance, "can_handle", None)), f"{name} must implement can_handle"
        can_res = instance.can_handle(dummy_grid, dummy_actions)
        assert isinstance(can_res, (bool, np.bool_)), f"{name}.can_handle must return a boolean"

        # Verify plan callable signature
        assert callable(getattr(instance, "plan", None)), f"{name} must implement plan"


def test_discrete_vector_translator_building_block():
    """Verify lowest-level movement translation building block."""
    # Cardinal decomposition: move (dx=-2, dy=3)
    # y increases downward: dy=3 is DOWN (2), dx=-2 is LEFT (3)
    actions = DiscreteVectorTranslator.delta_to_actions(dx=-2, dy=3, horizontal_first=False)
    assert actions == [(2, None), (2, None), (2, None), (3, None), (3, None)]

    # Points to actions
    pt_actions = DiscreteVectorTranslator.points_to_actions((10, 10), (10, 12))
    assert pt_actions == [(2, None), (2, None)]


def test_lattice_quantizer_building_block():
    """Verify coordinate quantization downsampling building block."""
    grid = np.zeros((12, 12), dtype=int)
    grid[0:6, 0:6] = 3
    grid[6:12, 6:12] = 5

    # Block majority 6x6
    macro = LatticeQuantizer.block_majority(grid, block_size=6)
    assert macro.shape == (2, 2)
    assert macro[0, 0] == 3
    assert macro[1, 1] == 5


def test_perceptual_cluster_detector():
    """Verify connected component visual cluster extraction."""
    grid = np.zeros((10, 10), dtype=int)
    grid[2:5, 3:7] = 4  # Cluster of color 4

    clusters = PerceptualClusterDetector.find_color_clusters(grid, color=4)
    assert len(clusters) == 1
    assert clusters[0].pixel_count == 12
    assert clusters[0].color == 4


def test_remote_actuator_building_block():
    """Verify click and macro action formulation."""
    click_act = RemoteActuator.click(x=32, y=48)
    assert click_act == (6, {"x": 32, "y": 48})


def test_temporal_sync_planner():
    """Verify periodic wait and sync actions."""
    waits = TemporalSyncPlanner.wait(3, noop_action=5)
    assert len(waits) == 3
    assert waits == [(5, None), (5, None), (5, None)]

    padded = TemporalSyncPlanner.pad_to_length([(1, None)], target_length=4)
    assert len(padded) == 4
    assert padded == [(1, None), (5, None), (5, None), (5, None)]


def test_raycaster_and_push_planner():
    """Verify 2D raycasting and push stance calculation."""
    grid = np.zeros((10, 10), dtype=int)
    grid[0, 5] = 8  # Obstacle at x=5, y=0

    hit_pos, hit_color, path = Raycaster2D.trace_ray(
        origin=(5, 5), direction=Raycaster2D.UP, grid=grid, obstacle_colors={8}
    )
    assert hit_pos == (5, 0)
    assert hit_color == 8

    stance = BlockPushPlanner.get_push_stance(block_pos=(4, 4), push_dir=(1, 0))
    assert stance == (3, 4)


def test_gf2_linear_solver():
    """Verify GF(2) linear solver building block."""
    # System:
    # x0 + x1 = 1
    # x1 = 1
    # Solution: x1 = 1, x0 = 0
    A = np.array([[1, 1], [0, 1]], dtype=int)
    b = np.array([1, 1], dtype=int)
    x = GF2LinearSolver.solve(A, b)
    assert x is not None
    assert list(x) == [0, 1]
