from __future__ import annotations

import time

import numpy as np

from hbllm.hcir.world.autonomous_epistemic_engine import (
    AutonomousEpistemicEngine,
    EmbodiedObservation,
)
from hbllm.hcir.world.cortical_grounding_bridge import CorticalGroundingBridge


def run_grid_dev_task(task_name: str = "grid_dev_nav") -> tuple[bool, int, list[float], list[str]]:
    """Runs a development grid navigation and interaction task."""
    engine = AutonomousEpistemicEngine()
    engine.attach_grounding_bridge(CorticalGroundingBridge())

    latencies: list[float] = []
    errors: list[str] = []
    steps = 0
    success = False

    # A simple 10x10 environment: avatar at (1, 1), target at (5, 5)
    avatar_r, avatar_c = 1, 1
    target_r, target_c = 5, 5

    for step in range(20):
        steps = step + 1
        grid = np.zeros((10, 10), dtype=int)
        grid[avatar_r, avatar_c] = 1  # Avatar
        grid[target_r, target_c] = 3  # Goal

        entities = [
            {"position": (float(avatar_r), float(avatar_c)), "color": 1, "role": "avatar"},
            {"position": (float(target_r), float(target_c)), "color": 3, "role": "goal"},
        ]

        obs = EmbodiedObservation(
            lattice=grid,
            available_actions=[1, 2, 3, 4],  # 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT
            entities=entities,
        )

        t0 = time.perf_counter()
        _ = engine.step(obs)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        # Move avatar toward target (simulating environment response)
        if avatar_r < target_r:
            avatar_r += 1
        elif avatar_c < target_c:
            avatar_c += 1

        if avatar_r == target_r and avatar_c == target_c:
            success = True
            break

    return success, steps, latencies, errors
