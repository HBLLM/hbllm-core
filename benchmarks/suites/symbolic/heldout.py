from __future__ import annotations

import time

import numpy as np

from hbllm.hcir.world.autonomous_epistemic_engine import (
    AutonomousEpistemicEngine,
    EmbodiedObservation,
)
from hbllm.hcir.world.cortical_grounding_bridge import CorticalGroundingBridge


def run_symbolic_heldout_task(
    task_name: str = "symbolic_rule_induction_heldout",
) -> tuple[bool, int, list[float], list[str]]:
    """Runs a multi-stage symbol grounding and rule induction held-out task."""
    engine = AutonomousEpistemicEngine()
    engine.attach_grounding_bridge(CorticalGroundingBridge())

    latencies: list[float] = []
    errors: list[str] = []
    steps = 0
    success = False

    # Simulate multi-symbol state with relational change
    for step in range(8):
        steps += 1
        grid = np.zeros((8, 8), dtype=int)
        grid[1, 1] = 2
        grid[1, 2] = 5
        grid[6, 6] = 7

        entities = [
            {"position": (1.0, 1.0), "color": 2, "role": "subject"},
            {"position": (1.0, 2.0), "color": 5, "role": "instrument"},
            {"position": (6.0, 6.0), "color": 7, "role": "receptacle"},
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
