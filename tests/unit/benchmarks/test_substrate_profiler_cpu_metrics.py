"""Tests verifying CPU + wall latency reporting, git SHA stamping, and JSON serialization (H-4)."""

from __future__ import annotations

from hbllm.benchmarks.substrate_profiler import (
    benchmark_e2e_cognitive_step_32x32,
    run_substrate_profile,
    serialize_profile_report,
)


def test_profiler_reports_cpu_wall_metrics_and_git_sha() -> None:
    substrate = run_substrate_profile()
    e2e = benchmark_e2e_cognitive_step_32x32(steps=8, warmup_steps=2)

    assert e2e.steps == 8
    assert e2e.e2e_p50_ms >= 0.0
    assert e2e.e2e_p95_ms >= e2e.e2e_p50_ms
    assert e2e.e2e_cpu_p50_ms >= 0.0
    assert e2e.e2e_cpu_p95_ms >= e2e.e2e_cpu_p50_ms
    assert e2e.e2e_cpu_max_ms >= e2e.e2e_cpu_p95_ms
    assert isinstance(e2e.git_sha, str) and len(e2e.git_sha) > 0

    payload = serialize_profile_report(substrate, e2e)
    assert payload["git_sha"] == e2e.git_sha
    assert payload["e2e_32x32"]["e2e_cpu_p95_ms"] == e2e.e2e_cpu_p95_ms
    assert "results" in payload["substrate"]
