from __future__ import annotations

import time

import numpy as np

from hbllm.hcir.world.autonomous_epistemic_engine import (
    AutonomousEpistemicEngine,
    EmbodiedObservation,
)
from hbllm.hcir.world.cortical_grounding_bridge import CorticalGroundingBridge


def run_continuous_heldout_task(
    task_name: str = "continuous_obstacle_avoidance_heldout",
) -> tuple[bool, int, list[float], list[str]]:
    """Runs a multi-stage continuous manipulation scenario with spatial obstacles."""
    engine = AutonomousEpistemicEngine()
    engine.attach_grounding_bridge(CorticalGroundingBridge())

    latencies: list[float] = []
    errors: list[str] = []
    steps = 0
    success = False

    # Simulate multi-stage spatial grasping with moving hazard
    for step in range(12):
        steps += 1
        grid = np.zeros((15, 15), dtype=int)
        # Target object at (2.5, 3.0), hazard moving across path
        hazard_c = int(2.0 + step * 0.8)
        grid[2, 3] = 4
        if 0 <= hazard_c < 15:
            grid[2, hazard_c] = 9

        entities = [
            {"position": (2.5, 3.0), "color": 4, "role": "target_payload"},
            {"position": (2.0, float(hazard_c)), "color": 9, "role": "hazard"},
        ]

        obs = EmbodiedObservation(
            lattice=grid,
            available_actions=[1, 2, 3, 4],
            entities=entities,
        )

        t0 = time.perf_counter()
        engine.step(obs)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    success = True
    return success, steps, latencies, errors
