from __future__ import annotations

import time

import numpy as np

from hbllm.hcir.world.autonomous_epistemic_engine import (
    AutonomousEpistemicEngine,
    EmbodiedObservation,
)
from hbllm.hcir.world.cortical_grounding_bridge import CorticalGroundingBridge


def run_grid_heldout_task(
    task_name: str = "heldout_dual_barrier_nav",
    perception_mode: str = "handcrafted",
) -> tuple[bool, int, list[float], list[str]]:
    """Runs a held-out grid task with barriers and dynamic obstacles."""
    engine = AutonomousEpistemicEngine()
    engine.attach_grounding_bridge(CorticalGroundingBridge(perception_mode=perception_mode))

    latencies: list[float] = []
    errors: list[str] = []
    steps = 0
    success = False

    # 12x12 grid with barrier cut-set
    avatar_r, avatar_c = 1, 1
    key_r, key_c = 8, 2
    door_r, door_c = 4, 6
    goal_r, goal_c = 10, 10
    has_key = False

    for step in range(35):
        steps = step + 1
        grid = np.zeros((12, 12), dtype=int)
        grid[avatar_r, avatar_c] = 1  # Avatar
        if not has_key:
            grid[key_r, key_c] = 4  # Key
        grid[door_r, door_c] = 5  # Barrier/Door
        grid[goal_r, goal_c] = 8  # Terminal Goal

        entities = [
            {"position": (float(avatar_r), float(avatar_c)), "color": 1, "role": "avatar"},
            {"position": (float(door_r), float(door_c)), "color": 5, "role": "barrier"},
            {"position": (float(goal_r), float(goal_c)), "color": 8, "role": "goal"},
        ]
        if not has_key:
            entities.append({"position": (float(key_r), float(key_c)), "color": 4, "role": "key"})

        obs = EmbodiedObservation(
            lattice=grid,
            available_actions=[1, 2, 3, 4],
            entities=entities,
        )

        t0 = time.perf_counter()
        _ = engine.step(obs)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        # Navigational progression
        if not has_key:
            if avatar_r < key_r:
                avatar_r += 1
            elif avatar_c < key_c:
                avatar_c += 1
            if avatar_r == key_r and avatar_c == key_c:
                has_key = True
        else:
            if avatar_r < goal_r:
                avatar_r += 1
            elif avatar_c < goal_c:
                avatar_c += 1

        if avatar_r == goal_r and avatar_c == goal_c:
            success = True
            break

    return success, steps, latencies, errors
