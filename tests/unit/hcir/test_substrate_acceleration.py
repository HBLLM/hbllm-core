"""Decisional Hot-Path Substrate Acceleration and Latency Budget Verification.

Validates that native Rust acceleration kernels provide 100% mathematical parity
with pure Python references, satisfy the sub-10ms decisional hot-path latency budget,
and gracefully degrade to pure Python execution when native binaries are unavailable.
"""

from __future__ import annotations

import unittest.mock as mock

from hbllm.benchmarks.substrate_profiler import (
    benchmark_astar_geodesic,
    benchmark_collision_cone_sweep,
    benchmark_connected_components,
    benchmark_graph_snapshotting,
    run_substrate_profile,
)
from hbllm.hcir.world.predictors.physics import PhysicsPredictor
from hbllm.hcir.world.spatiotemporal_collision import (
    ProjectedTrajectory,
    SpatiotemporalCollisionCones,
)

try:
    import hbllm_hcir_graph as _rust_graph
except ImportError:
    _rust_graph = None


class TestSubstrateAccelerationAndHotPaths:
    """Substrate acceleration, latency envelope, and fallback parity battery."""

    def test_native_graph_chunked_snapshot_and_hash_parity(self) -> None:
        """Verify native graph chunked snapshotting and canonical BLAKE3 hashing."""
        result = benchmark_graph_snapshotting(node_count=100, iterations=10)
        assert result.parity_verified is True
        assert result.rust_latency_ms <= result.python_latency_ms

    def test_topological_connected_components_parity(self) -> None:
        """Verify topological connected components between pure Python and native Rust."""
        result = benchmark_connected_components(height=24, width=24, iterations=10)
        assert result.parity_verified is True
        assert result.speedup >= 1.0

    def test_fast_collision_cone_sweep_parity(self) -> None:
        """Verify space-time collision cone intercepts between Python loop and native Rust."""
        result = benchmark_collision_cone_sweep(iterations=20)
        assert result.parity_verified is True
        assert result.details.get("intercept_step") is not None

    def test_fast_geodesic_astar_parity(self) -> None:
        """Verify A* geodesic obstacle-clearing path between Python reference and native Rust."""
        result = benchmark_astar_geodesic(height=20, width=20, iterations=10)
        assert result.parity_verified is True
        assert result.details.get("path_length", 0) > 0

    def test_decisional_hot_path_latency_budget(self) -> None:
        """Verify that combined decisional hot paths operate within the sub-10ms budget."""
        report = run_substrate_profile()
        assert report.budget_met is True
        assert report.total_native_p95_ms <= 10.0
        for r in report.results:
            assert r.parity_verified is True

    def test_physics_predictor_native_integration_and_fallback(self) -> None:
        """Verify PhysicsPredictor geodesic computation uses native path and falls back seamlessly."""
        start = (1, 1)
        goal = (10, 10)
        grid_shape = (15, 15)
        barriers = {(5, 5), (5, 6), (5, 7)}

        # Direct execution (will use native if compiled)
        path = PhysicsPredictor.compute_geodesic_path(
            start=start,
            goal=goal,
            barrier_cells=barriers,
            grid_shape=grid_shape,
            step_size=1,
        )
        assert path[0] == start
        assert path[-1] == goal
        for p in path:
            assert p not in barriers

        # Fallback simulation with native disabled
        with mock.patch("hbllm.hcir.world.predictors.physics._rust_graph", None):
            fallback_path = PhysicsPredictor.compute_geodesic_path(
                start=start,
                goal=goal,
                barrier_cells=barriers,
                grid_shape=grid_shape,
                step_size=1,
            )
            assert fallback_path[0] == start
            assert fallback_path[-1] == goal
            assert len(fallback_path) == len(path)

    def test_spatiotemporal_collision_native_integration_and_fallback(self) -> None:
        """Verify SpatiotemporalCollisionCones intercept prediction uses native sweep and falls back."""
        cones = SpatiotemporalCollisionCones()
        traj = ProjectedTrajectory(
            entity_id="hazard_alpha",
            feature_id=3,
            initial_pos=(0, 5),
            velocity=(1, 0),
            positions=[(t, 5) for t in range(12)],
        )
        cones.projected_trajectories.append(traj)
        cones.step_occupied_cells = {t: {(t, 5)} for t in range(12)}

        # Direct execution
        step = cones.predict_intercept_step(target_pos=(6, 5), max_horizon=10, clearance=0)
        assert step == 6

        # Fallback simulation with native disabled
        with mock.patch("hbllm.hcir.world.spatiotemporal_collision._rust_graph", None):
            fallback_step = cones.predict_intercept_step(
                target_pos=(6, 5), max_horizon=10, clearance=0
            )
            assert fallback_step == 6
