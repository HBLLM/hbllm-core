from __future__ import annotations

"""Grid Spatial Benchmark — ARC-AGI-style grid puzzles.

Tests entity discovery, transformation tracking, and concept formation
on 2D grid observations.
"""

import time

import numpy as np

from benchmarks.sge_integration_benchmark import BenchmarkMetrics, _make_engine


class GridSpatialScenario:
    """10-step grid observation scenario with entity tracking and concept learning."""

    def run(self) -> BenchmarkMetrics:
        from hbllm.hcir.world.autonomous_epistemic_engine import EmbodiedObservation

        metrics = BenchmarkMetrics(scenario_name="grid_spatial")
        engine, bridge = _make_engine()
        t0 = time.perf_counter()

        try:
            # Phase 1: Introduce entities over 5 steps
            for step in range(5):
                grid = np.zeros((10, 10), dtype=int)
                grid[2, 3 + step % 3] = 4  # Moving blue entity
                grid[7, 7] = 6  # Stationary green entity
                entities = [
                    {"position": (2.0, float(3 + step % 3)), "color": 4, "role": "agent"},
                    {"position": (7.0, 7.0), "color": 6, "role": "goal"},
                ]
                obs = EmbodiedObservation(
                    lattice=grid,
                    available_actions=[1, 2, 3, 4],
                    entities=entities,
                )
                engine.step(obs)
                metrics.steps += 1

            # Phase 2: Track entity transformations for 5 more steps
            for step in range(5, 10):
                grid = np.zeros((10, 10), dtype=int)
                grid[3, 5] = 4  # Agent moved to new position
                grid[7, 7] = 6  # Goal unchanged
                grid[5, 5] = 9  # New unfamiliar entity appears
                entities = [
                    {"position": (3.0, 5.0), "color": 4, "role": "agent"},
                    {"position": (7.0, 7.0), "color": 6, "role": "goal"},
                    {"position": (5.0, 5.0), "color": 9, "role": "obstacle"},
                ]
                obs = EmbodiedObservation(
                    lattice=grid,
                    available_actions=[1, 2, 3, 4],
                    entities=entities,
                )
                engine.step(obs)
                metrics.steps += 1

            # Collect metrics
            active = bridge.tracker.get_active_candidates()
            metrics.entities_tracked = len(active)
            metrics.concepts_formed = len(bridge.concept_store.get_all_concepts())
            metrics.features_extracted = sum(1 for _ in bridge.state.last_feature_vectors)

            # Validate minimum quality
            if metrics.entities_tracked < 2:
                metrics.errors.append(
                    f"Expected >=2 tracked entities, got {metrics.entities_tracked}"
                )

        except Exception as exc:
            metrics.errors.append(f"Scenario failed: {exc}")

        elapsed = (time.perf_counter() - t0) * 1000
        metrics.total_time_ms = elapsed
        metrics.mean_step_time_ms = elapsed / max(1, metrics.steps)
        return metrics
