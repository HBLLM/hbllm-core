from __future__ import annotations

import time

from benchmarks.scenarios.robotics_manipulation import RoboticsManipulationScenario


def run_continuous_dev_task(
    task_name: str = "robotics_pick_place_dev",
) -> tuple[bool, int, list[float], list[str]]:
    """Runs the continuous robotics manipulation scenario (dev split)."""
    scenario = RoboticsManipulationScenario()
    t0 = time.perf_counter()
    metrics = scenario.run()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    step_latencies = [elapsed_ms / max(1, metrics.steps)] * metrics.steps
    return metrics.passed, metrics.steps, step_latencies, metrics.errors
