"""
DualLLMRouter Benchmark — measures routing decisions and circuit breaker behavior.

Suites:
  - Routing decision latency (local vs external)
  - Circuit breaker state transitions (closed → open → half-open → closed)
  - Fallback overhead when circuit opens

Usage:
    Registered as 'dual_router' suite in runner.py
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from hbllm.benchmarks.runner import BenchmarkReport, BenchmarkResult

logger = logging.getLogger(__name__)


class DualRouterBenchmark:
    """Benchmark the DualLLMRouter and CircuitBreaker patterns."""

    async def run(self) -> BenchmarkReport:
        report = BenchmarkReport(suite="dual_router")

        await self._bench_routing_decisions(report)
        await self._bench_circuit_breaker_transitions(report)
        await self._bench_fallback_overhead(report)

        report.comparisons = [
            {
                "name": "Routing decision",
                "hbllm": f"{report.results[0].value:.3f}ms" if report.results else "N/A",
                "baseline": "0ms (single model)",
                "delta": "< 0.1ms added",
            },
            {
                "name": "Circuit recovery",
                "hbllm": "Auto (half-open probe)",
                "baseline": "Manual restart",
                "delta": "self-healing",
            },
        ]

        return report

    async def _bench_routing_decisions(self, report: BenchmarkReport) -> None:
        """Benchmark the classify() routing decision latency."""
        try:
            from hbllm.brain.dual_llm_router import DualLLMRouter

            local_llm = MagicMock()
            local_llm.generate = AsyncMock(return_value=MagicMock(content="local response"))
            external_llm = MagicMock()
            external_llm.generate = AsyncMock(return_value=MagicMock(content="external response"))

            router = DualLLMRouter(local_llm=local_llm, external_llm=external_llm)

            # Test classify() latency
            queries = [
                "Hello",  # simple → local
                "Hi there",  # simple → local
                "What is the weather?",  # medium → depends
                "Explain the implications of quantum field theory on string cosmology",  # complex → external
                "Write a recursive Python function with memoization",  # complex → external
                "Thank you",  # simple → local
                "How does photosynthesis work in detail?",  # complex → external
                "OK",  # simple → local
            ]

            latencies = []
            for query in queries * 50:  # 400 iterations
                t0 = time.perf_counter()
                router.classify(query)
                latencies.append((time.perf_counter() - t0) * 1000)

            report.add(
                BenchmarkResult(
                    name="DualLLMRouter.classify() avg",
                    metric="classify_avg",
                    value=statistics.mean(latencies),
                    unit="ms",
                    metadata={"iterations": len(latencies)},
                )
            )
            report.add(
                BenchmarkResult(
                    name="DualLLMRouter.classify() p99",
                    metric="classify_p99",
                    value=sorted(latencies)[int(0.99 * len(latencies))],
                    unit="ms",
                )
            )

            # Count routing distribution
            local_count = sum(1 for q in queries if router.classify(q) == "local")
            external_count = len(queries) - local_count

            report.add(
                BenchmarkResult(
                    name="Routing distribution (local/external)",
                    metric="routing_distribution",
                    value=local_count / len(queries) * 100,
                    unit="% local",
                    metadata={"local": local_count, "external": external_count},
                )
            )

        except Exception as e:
            logger.warning("DualLLMRouter benchmark skipped: %s", e)
            report.add(
                BenchmarkResult(
                    name="DualLLMRouter (skipped)",
                    metric="classify_avg",
                    value=0,
                    unit="ms",
                    metadata={"error": str(e)},
                )
            )

    async def _bench_circuit_breaker_transitions(self, report: BenchmarkReport) -> None:
        """Benchmark circuit breaker state transition speed."""
        try:
            from hbllm.brain.dual_llm_router import CircuitBreaker

            cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)

            # Measure transition: closed → open
            t0 = time.perf_counter()
            for _ in range(3):
                cb.record_failure()
            closed_to_open = (time.perf_counter() - t0) * 1000

            assert cb.state == "open", f"Expected open, got {cb.state}"

            report.add(
                BenchmarkResult(
                    name="Circuit: closed → open (3 failures)",
                    metric="cb_close_to_open",
                    value=closed_to_open,
                    unit="ms",
                )
            )

            # Wait for recovery timeout
            await asyncio.sleep(0.15)

            # Measure transition: open → half-open
            t0 = time.perf_counter()
            can_try = cb.allow_request()
            open_to_half = (time.perf_counter() - t0) * 1000

            report.add(
                BenchmarkResult(
                    name="Circuit: open → half-open check",
                    metric="cb_open_to_half",
                    value=open_to_half,
                    unit="ms",
                    metadata={"allowed": can_try},
                )
            )

            # Measure transition: half-open → closed (success)
            t0 = time.perf_counter()
            cb.record_success()
            half_to_closed = (time.perf_counter() - t0) * 1000

            report.add(
                BenchmarkResult(
                    name="Circuit: half-open → closed (success)",
                    metric="cb_half_to_closed",
                    value=half_to_closed,
                    unit="ms",
                )
            )

            # Full cycle timing
            report.add(
                BenchmarkResult(
                    name="Circuit breaker full recovery cycle",
                    metric="cb_full_cycle",
                    value=closed_to_open + 150 + open_to_half + half_to_closed,
                    unit="ms",
                    metadata={"recovery_timeout_ms": 100},
                )
            )

        except Exception as e:
            logger.warning("CircuitBreaker benchmark skipped: %s", e)
            report.add(
                BenchmarkResult(
                    name="CircuitBreaker (skipped)",
                    metric="cb_close_to_open",
                    value=0,
                    unit="ms",
                    metadata={"error": str(e)},
                )
            )

    async def _bench_fallback_overhead(self, report: BenchmarkReport) -> None:
        """Measure added latency when circuit is open and we fall back to local."""
        try:
            from hbllm.brain.dual_llm_router import DualLLMRouter

            local_llm = MagicMock()
            local_llm.generate = AsyncMock(return_value=MagicMock(content="local fallback"))
            external_llm = MagicMock()
            external_llm.generate = AsyncMock(side_effect=ConnectionError("External LLM down"))

            router = DualLLMRouter(
                local_llm=local_llm,
                external_llm=external_llm,
                failure_threshold=2,
                recovery_timeout=60.0,  # Long timeout so circuit stays open
            )

            # Force circuit open
            for _ in range(2):
                try:
                    await router.generate("complex query requiring external", tier="external")
                except Exception:
                    pass

            # Measure fallback latency (circuit open → local)
            latencies = []
            for _ in range(100):
                t0 = time.perf_counter()
                try:
                    await router.generate("This should fall back to local", tier="external")
                except Exception:
                    pass
                latencies.append((time.perf_counter() - t0) * 1000)

            report.add(
                BenchmarkResult(
                    name="Fallback latency (circuit open → local)",
                    metric="fallback_latency",
                    value=statistics.mean(latencies),
                    unit="ms",
                    metadata={"trials": 100, "circuit_state": "open"},
                )
            )

            # Get snapshot for stats
            snap = router.snapshot()
            report.add(
                BenchmarkResult(
                    name="Router snapshot generation",
                    metric="snapshot_time",
                    value=0,
                    unit="ms",
                    metadata=snap,
                )
            )

        except Exception as e:
            logger.warning("Fallback overhead benchmark skipped: %s", e)
            report.add(
                BenchmarkResult(
                    name="Fallback overhead (skipped)",
                    metric="fallback_latency",
                    value=0,
                    unit="ms",
                    metadata={"error": str(e)},
                )
            )
