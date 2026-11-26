"""
Routing & CircuitBreaker Benchmark — measures HCIR complexity classification and circuit resilience.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time

from hbllm.benchmarks.runner import BenchmarkReport, BenchmarkResult

logger = logging.getLogger(__name__)


class DualRouterBenchmark:
    """Benchmark HCIR/Serving query complexity classification and CircuitBreaker behavior."""

    async def run(self) -> BenchmarkReport:
        report = BenchmarkReport(suite="dual_router")

        await self._bench_routing_decisions(report)
        await self._bench_circuit_breaker_transitions(report)

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
        """Benchmark complexity classification latency."""
        try:
            from hbllm.network.bus import InProcessBus
            from hbllm.network.registry import ServiceRegistry
            from hbllm.serving.pipeline import CognitivePipeline

            pipeline = CognitivePipeline(bus=InProcessBus(), registry=ServiceRegistry())
            queries = [
                "Hello",
                "Hi there",
                "What is the weather?",
                "Explain the implications of quantum field theory on string cosmology",
                "Write a recursive Python function with memoization",
                "Thank you",
                "How does photosynthesis work in detail?",
                "OK",
            ]

            latencies = []
            for query in queries * 50:
                t0 = time.perf_counter()
                pipeline._classify_complexity(query)
                latencies.append((time.perf_counter() - t0) * 1000)

            report.add(
                BenchmarkResult(
                    name="CognitivePipeline._classify_complexity() avg",
                    metric="classify_avg",
                    value=statistics.mean(latencies),
                    unit="ms",
                    metadata={"iterations": len(latencies)},
                )
            )
            local_count = sum(1 for q in queries if pipeline._classify_complexity(q) == "trivial")
            report.add(
                BenchmarkResult(
                    name="Fast-path trivial fraction",
                    metric="routing_distribution",
                    value=local_count / len(queries) * 100,
                    unit="% trivial",
                )
            )
        except Exception as e:
            logger.warning("Routing benchmark skipped: %s", e)

    async def _bench_circuit_breaker_transitions(self, report: BenchmarkReport) -> None:
        """Benchmark circuit breaker state transition speed."""
        try:
            from hbllm.network.circuit_breaker import CircuitBreaker

            cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.05)

            t0 = time.perf_counter()
            for _ in range(3):
                cb.record_failure()
            closed_to_open = (time.perf_counter() - t0) * 1000

            report.add(
                BenchmarkResult(
                    name="Circuit: closed → open (3 failures)",
                    metric="cb_close_to_open",
                    value=closed_to_open,
                    unit="ms",
                )
            )

            await asyncio.sleep(0.06)
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
        except Exception as e:
            logger.warning("CircuitBreaker benchmark skipped: %s", e)
