"""Tests for the benchmark runner."""

import pytest
import asyncio
import json

from hbllm.benchmarks.runner import (
    BenchmarkResult, BenchmarkReport,
    LatencyBenchmark, MemoryBenchmark,
    SpecializationBenchmark, MultiTenantBenchmark,
    SUITES, run_suite,
)


# ── Result / Report Types ───────────────────────────────────────────────────

def test_benchmark_result_to_dict():
    r = BenchmarkResult(name="Test", metric="latency", value=1.234, unit="ms")
    d = r.to_dict()
    assert d["name"] == "Test"
    assert d["value"] == 1.234
    assert d["unit"] == "ms"


def test_benchmark_report():
    report = BenchmarkReport(suite="test")
    report.add(BenchmarkResult(name="A", metric="m", value=1.0, unit="ms"))
    report.add(BenchmarkResult(name="B", metric="m", value=2.0, unit="ms"))
    d = report.to_dict()
    assert d["suite"] == "test"
    assert len(d["results"]) == 2


def test_report_save_and_load(tmp_path):
    report = BenchmarkReport(suite="test_save")
    report.add(BenchmarkResult(name="Speed", metric="s", value=42.0, unit="ops/s"))
    path = tmp_path / "report.json"
    report.save(path)
    loaded = json.loads(path.read_text())
    assert loaded["suite"] == "test_save"
    assert len(loaded["results"]) == 1


# ── Latency Benchmark ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_latency_benchmark():
    bench = LatencyBenchmark()
    report = await bench.run()
    assert report.suite == "latency"
    assert len(report.results) >= 3
    # Bus latency should be sub-millisecond
    p50 = next(r for r in report.results if r.metric == "latency_p50")
    assert p50.value < 10  # less than 10ms


# ── Memory Benchmark ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_memory_benchmark():
    bench = MemoryBenchmark()
    report = await bench.run()
    assert report.suite == "memory"
    # Zoning should be much smaller than monolithic
    zoning = next(r for r in report.results if r.metric == "zoning_total")
    mono = next(r for r in report.results if r.metric == "monolithic_total")
    assert zoning.value < mono.value


# ── Specialization Benchmark ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_specialization_benchmark():
    bench = SpecializationBenchmark()
    report = await bench.run()
    assert report.suite == "specialization"
    accuracy = next(r for r in report.results if r.metric == "routing_accuracy")
    assert accuracy.value >= 50  # at least 50% with keyword heuristics


# ── Multi-Tenant Benchmark ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_multi_tenant_benchmark():
    bench = MultiTenantBenchmark()
    report = await bench.run()
    assert report.suite == "multi_tenant"
    isolation = next(r for r in report.results if r.metric == "isolation")
    assert isolation.value == 1  # perfect isolation


# ── Suite Registry ───────────────────────────────────────────────────────────

def test_suites_registered():
    assert "latency" in SUITES
    assert "memory" in SUITES
    assert "specialization" in SUITES
    assert "multi_tenant" in SUITES


@pytest.mark.asyncio
async def test_run_suite():
    report = await run_suite("memory")
    assert report.suite == "memory"


@pytest.mark.asyncio
async def test_run_invalid_suite():
    with pytest.raises(ValueError):
        await run_suite("nonexistent")
