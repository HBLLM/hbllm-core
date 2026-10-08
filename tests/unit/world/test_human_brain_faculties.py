"""Unit tests for Advanced Human Brain Faculties:
1. Orbitofrontal Cortex (OFC) Counterfactual Deadlock & Irreversibility Detector
2. dlPFC/IPS Visuospatial Working Memory & Feature-Location Binding
3. IPL Extended Body Schema & Tool Affordance Incorporation
4. LOC Ventral Stream Geometric Symmetry Discrepancy Completion
5. Premotor Spatiotemporal Collision Cones & Trajectory Extrapolation
"""

from __future__ import annotations

import numpy as np

from hbllm.hcir.world.counterfactual_simulation import CounterfactualDeadlockDetector
from hbllm.hcir.world.extended_body_schema import ExtendedBodySchema
from hbllm.hcir.world.kinetic_stream import KineticEntity
from hbllm.hcir.world.spatiotemporal_collision import SpatiotemporalCollisionCones
from hbllm.hcir.world.visual_symmetry import VisualSymmetryAnalyzer
from hbllm.hcir.world.visuospatial_working_memory import VisuospatialWorkingMemory


class TestCounterfactualDeadlockDetector:
    """Test OFC counterfactual deadlock simulation and irreversible state pruning."""

    def test_corner_deadlock_detected(self) -> None:
        goals = {(5, 5)}
        static_barriers = {(0, 1), (1, 0)}
        grid_shape = (10, 10)
        eval_res = CounterfactualDeadlockDetector.evaluate_deadlock(
            (0, 0), goals, static_barriers, {(0, 0)}, grid_shape
        )
        assert eval_res.is_deadlock is True
        assert eval_res.deadlock_type == "corner"

    def test_goal_reversal_prevents_deadlock(self) -> None:
        goals = {(0, 0)}
        static_barriers = {(0, 1), (1, 0)}
        grid_shape = (10, 10)
        eval_res = CounterfactualDeadlockDetector.evaluate_deadlock(
            (0, 0), goals, static_barriers, {(0, 0)}, grid_shape
        )
        assert eval_res.is_deadlock is False

    def test_2x2_square_deadlock_detected(self) -> None:
        goals = {(9, 9)}
        static_barriers = {(2, 2), (2, 3)}
        all_blocks = {(3, 2), (3, 3)}
        grid_shape = (10, 10)
        eval_res = CounterfactualDeadlockDetector.evaluate_deadlock(
            (3, 2), goals, static_barriers, all_blocks, grid_shape
        )
        assert eval_res.is_deadlock is True
        assert eval_res.deadlock_type == "square_2x2"

    def test_line_freeze_deadlock_detected(self) -> None:
        goals = {(9, 9)}
        static_barriers = {(1, 2), (1, 3)}
        all_blocks = {(2, 2), (2, 3)}
        grid_shape = (10, 10)
        eval_res = CounterfactualDeadlockDetector.evaluate_deadlock(
            (2, 2), goals, static_barriers, all_blocks, grid_shape
        )
        assert eval_res.is_deadlock is True
        assert (
            CounterfactualDeadlockDetector.is_line_freeze_deadlock(
                (2, 2), goals, static_barriers, all_blocks, grid_shape
            )
            is True
        )

    def test_interior_wall_deadlock_detected(self) -> None:
        goals = {(9, 9)}
        static_barriers = {
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (3, 1),
            (3, 6),
        }
        grid_shape = (10, 10)
        eval_res = CounterfactualDeadlockDetector.evaluate_deadlock(
            (3, 3), goals, static_barriers, {(3, 3)}, grid_shape
        )
        assert eval_res.is_deadlock is True
        assert eval_res.deadlock_type == "interior_wall"


class TestVisuospatialWorkingMemory:
    """Test dlPFC visuospatial feature-location binding and pair matching."""

    def test_feature_location_binding_and_pair_matching(self) -> None:
        wm = VisuospatialWorkingMemory(capacity=16)

        wm.record_probe((2, 3), feature_id=7, step=1)
        assert wm.last_probed_coord == (2, 3)
        assert wm.last_probed_feature == 7

        wm.record_probe((8, 4), feature_id=3, step=2)
        assert wm.find_matching_pair((8, 4), 3) is None

        wm.record_probe((5, 6), feature_id=7, step=3)
        pair = wm.find_matching_pair((5, 6), 7)
        assert pair == (2, 3)

    def test_diff_stencil_recording(self) -> None:
        wm = VisuospatialWorkingMemory()
        center = (5, 5)
        changed = {(5, 5), (4, 5), (6, 5), (5, 4), (5, 6)}
        wm.record_diff_stencil(center, changed)
        assert len(wm.discovered_stencils) == 1
        assert (0, 0) in wm.discovered_stencils[0]
        assert (-1, 0) in wm.discovered_stencils[0]
        assert (1, 0) in wm.discovered_stencils[0]

    def test_systematic_raster_scan(self) -> None:
        wm = VisuospatialWorkingMemory()
        wm.record_probe((1, 1), feature_id=2, step=1)
        candidates = [(3, 2), (1, 1), (1, 5), (2, 0)]
        next_cand = wm.get_systematic_unprobed_candidate(candidates)
        assert next_cand == (1, 5)


class TestExtendedBodySchema:
    """Test IPL extended body schema and tool-gated barrier affordances."""

    def test_tool_incorporation_and_resonance(self) -> None:
        schema = ExtendedBodySchema()
        assert schema.is_holding(3) is False

        schema.acquire_tool(feature_id=3, step=10, role="key")
        assert schema.is_holding(3) is True

        assert schema.is_barrier_permeable(8) is False

        schema.register_resonance(tool_feature=3, barrier_feature=8)
        assert schema.is_barrier_permeable(8) is True
        assert 8 in schema.get_permeable_barrier_features()

        assert schema.expend_tool(3) is True
        assert schema.is_holding(3) is False
        assert schema.is_barrier_permeable(8) is False


class TestVisualSymmetryAnalyzer:
    """Test LOC geometric symmetry discrepancy completion."""

    def test_extract_discrepancy_targets(self) -> None:
        # Create a 6x6 grid with dominant vertical symmetry: 4 matching pairs, 1 missing at (2, 4)
        grid = np.zeros((6, 6), dtype=int)
        grid[0, 1] = 4
        grid[0, 4] = 4
        grid[1, 1] = 4
        grid[1, 4] = 4
        grid[3, 1] = 4
        grid[3, 4] = 4
        grid[4, 1] = 4
        grid[4, 4] = 4
        grid[2, 1] = 4  # Left side has color 4, right side at (2, 4) is 0
        discrepancies = VisualSymmetryAnalyzer.extract_discrepancy_targets(
            grid, background_color=0, threshold=0.55
        )
        assert len(discrepancies) > 0
        coords = [(d[0], d[1]) for d in discrepancies]
        assert (2, 4) in coords
        target_feat = [d[2] for d in discrepancies if (d[0], d[1]) == (2, 4)][0]
        assert target_feat == 4


class TestSpatiotemporalCollisionCones:
    """Test Premotor collision cone projection with static boundary rebound."""

    def test_trajectory_extrapolation_and_hazard_check(self) -> None:
        cones = SpatiotemporalCollisionCones(default_horizon=6)
        entity = KineticEntity(
            centroid=(2.0, 2.0),
            velocity=(0.0, 1.0),
            cells=[(2, 2)],
            bounding_box=(2, 2, 2, 2),
            area=1,
            features={5},
        )
        static_barriers = {(2, 4)}
        grid_shape = (10, 10)

        cones.update_trajectories([entity], static_barriers, grid_shape, horizon=5)

        assert cones.is_collision_hazard(2, 3, time_step=1) is True
        assert cones.is_collision_hazard(2, 2, time_step=2) is True
        assert cones.is_collision_hazard(5, 5, time_step=1) is False
