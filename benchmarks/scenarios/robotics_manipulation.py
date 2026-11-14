from __future__ import annotations

"""Robotics Manipulation Benchmark — pick-place with unfamiliar objects.

Tests concept formation, causal learning, and action validation
for robotics-style observations.
"""

import time

import numpy as np

from benchmarks.sge_integration_benchmark import BenchmarkMetrics, _make_engine


class RoboticsManipulationScenario:
    """Simulated pick-place: discover object → attempt grasp → verify outcome."""

    def run(self) -> BenchmarkMetrics:
        from hbllm.hcir.world.autonomous_epistemic_engine import EmbodiedObservation

        metrics = BenchmarkMetrics(scenario_name="robotics_manipulation")
        engine, bridge = _make_engine()
        t0 = time.perf_counter()

        try:
            # Phase 1: Observe workspace with objects
            for step in range(3):
                grid = np.zeros((8, 8), dtype=int)
                grid[1, 2] = 3  # Object to grasp
                grid[6, 6] = 5  # Placement target
                obs = EmbodiedObservation(
                    lattice=grid,
                    available_actions=[1, 2, 3, 4],
                    entities=[
                        {"position": (1.0, 2.0), "color": 3, "role": "graspable"},
                        {"position": (6.0, 6.0), "color": 5, "role": "target_zone"},
                    ],
                )
                engine.step(obs)
                metrics.steps += 1

            # Phase 2: Attempt action through gateway
            candidates = bridge.tracker.get_active_candidates()
            if candidates:
                target_ref = candidates[0].candidate_ref
                world_st = {
                    "tracked_entities": {
                        target_ref.id: {"confidence": 0.9, "position": (1.0, 2.0)},
                    },
                    "gripper_empty": True,
                }
                act, rej = bridge.process_motor(
                    ctx=None,  # type: ignore[arg-type]
                    proposed_action="GRASP",
                    action_data={
                        "target_id": target_ref.id,
                        "preconditions": ["gripper_empty == True"],
                        "speed": 1.0,
                    },
                    world_state=world_st,
                )
                if act is not None:
                    metrics.actions_authorised += 1
                if rej is not None:
                    metrics.actions_rejected += 1

            # Phase 3: Observe post-action state
            for step in range(3):
                grid = np.zeros((8, 8), dtype=int)
                grid[6, 6] = 5  # Object moved to target
                obs = EmbodiedObservation(
                    lattice=grid,
                    available_actions=[1, 2, 3, 4],
                    entities=[
                        {"position": (6.0, 6.0), "color": 5, "role": "target_zone"},
                    ],
                )
                engine.step(obs)
                metrics.steps += 1

            metrics.entities_tracked = len(bridge.tracker.get_active_candidates())
            metrics.concepts_formed = len(bridge.concept_store.get_all_concepts())
            metrics.outcomes_verified = bridge.state.verified_outcomes_count

        except Exception as exc:
            metrics.errors.append(f"Scenario failed: {exc}")

        elapsed = (time.perf_counter() - t0) * 1000
        metrics.total_time_ms = elapsed
        metrics.mean_step_time_ms = elapsed / max(1, metrics.steps)
        return metrics
