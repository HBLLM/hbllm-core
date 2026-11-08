from __future__ import annotations

import time

from benchmarks.scenarios.language_grounding import LanguageGroundingScenario


def run_symbolic_dev_task(
    task_name: str = "language_grounding_dev",
) -> tuple[bool, int, list[float], list[str]]:
    """Runs the language grounding scenario (dev split)."""
    scenario = LanguageGroundingScenario()
    t0 = time.perf_counter()
    metrics = scenario.run()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    step_latencies = [elapsed_ms / max(1, metrics.steps)] * metrics.steps
    return metrics.passed, metrics.steps, step_latencies, metrics.errors
