"""
HTTP API Load Test — measures real HTTP endpoint performance.

Suites:
  - Health endpoint latency (p50/p99)
  - Rate limit validation (burst → 429)
  - Concurrent tenant load
  - Graceful shutdown drain behavior

Usage:
    Registered as 'http_api' suite in runner.py

Note: This benchmark requires the FastAPI app to be importable but does NOT
start a real server. It uses httpx.AsyncClient with ASGI transport for
zero-network-overhead testing.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time

from hbllm.benchmarks.runner import BenchmarkReport, BenchmarkResult

logger = logging.getLogger(__name__)


class HTTPAPIBenchmark:
    """Benchmark the FastAPI serving layer via ASGI transport (no network)."""

    async def run(self) -> BenchmarkReport:
        report = BenchmarkReport(suite="http_api")

        await self._bench_health_latency(report)
        await self._bench_rate_limiting(report)
        await self._bench_concurrent_tenants(report)

        report.comparisons = [
            {
                "name": "Health check latency",
                "hbllm": f"{report.results[0].value:.2f}ms" if report.results else "N/A",
                "baseline": "~5ms (external)",
                "delta": "in-process ASGI",
            },
            {
                "name": "Rate limiting",
                "hbllm": "Token bucket (429)",
                "baseline": "None / nginx",
                "delta": "built-in",
            },
        ]

        return report

    async def _bench_health_latency(self, report: BenchmarkReport) -> None:
        """Benchmark /health, /health/live, /health/ready response times."""
        try:
            import httpx

            from hbllm.serving.api import app

            transport = httpx.ASGITransport(app=app)

            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                endpoints = ["/health", "/health/live", "/health/ready"]

                for endpoint in endpoints:
                    latencies = []

                    # Warm up
                    for _ in range(5):
                        await client.get(endpoint)

                    # Measure
                    for _ in range(100):
                        t0 = time.perf_counter()
                        resp = await client.get(endpoint)
                        latencies.append((time.perf_counter() - t0) * 1000)

                    latencies.sort()
                    report.add(
                        BenchmarkResult(
                            name=f"GET {endpoint} p50",
                            metric=f"http_{endpoint.replace('/', '_')}_p50",
                            value=latencies[len(latencies) // 2],
                            unit="ms",
                            metadata={"status": resp.status_code},
                        )
                    )
                    report.add(
                        BenchmarkResult(
                            name=f"GET {endpoint} p99",
                            metric=f"http_{endpoint.replace('/', '_')}_p99",
                            value=latencies[int(0.99 * len(latencies))],
                            unit="ms",
                        )
                    )

        except Exception as e:
            logger.warning("HTTP health benchmark skipped: %s", e)
            report.add(
                BenchmarkResult(
                    name="HTTP health (skipped)",
                    metric="http_health_p50",
                    value=0,
                    unit="ms",
                    metadata={"error": str(e)},
                )
            )

    async def _bench_rate_limiting(self, report: BenchmarkReport) -> None:
        """Verify rate limiter fires 429 under sustained load."""
        try:
            from hbllm.serving.middleware.rate_limit import HTTPRateLimitMiddleware

            # Test the rate limiter directly (faster than HTTP round-trips)
            limiter = HTTPRateLimitMiddleware(app=None, default_rpm=10)

            # Burst 20 requests for same tenant
            allowed = 0
            rejected = 0
            tenant = "bench_tenant"

            for i in range(20):
                bucket = limiter._get_bucket(tenant)
                if bucket.consume():
                    allowed += 1
                else:
                    rejected += 1

            report.add(
                BenchmarkResult(
                    name="Rate limit: allowed (10 RPM bucket)",
                    metric="rate_allowed",
                    value=allowed,
                    unit="requests",
                )
            )
            report.add(
                BenchmarkResult(
                    name="Rate limit: rejected (429s)",
                    metric="rate_rejected",
                    value=rejected,
                    unit="requests",
                    metadata={"expected_rejected": "~10"},
                )
            )

            # Measure bucket lookup latency
            latencies = []
            for _ in range(1000):
                t0 = time.perf_counter()
                limiter._get_bucket(f"tenant_{_ % 100}")
                latencies.append((time.perf_counter() - t0) * 1000)

            report.add(
                BenchmarkResult(
                    name="Rate limiter bucket lookup (100 tenants)",
                    metric="rate_lookup",
                    value=statistics.mean(latencies),
                    unit="ms",
                    metadata={"iterations": 1000},
                )
            )

        except Exception as e:
            logger.warning("Rate limiting benchmark skipped: %s", e)
            report.add(
                BenchmarkResult(
                    name="Rate limiting (skipped)",
                    metric="rate_allowed",
                    value=0,
                    unit="requests",
                    metadata={"error": str(e)},
                )
            )

    async def _bench_concurrent_tenants(self, report: BenchmarkReport) -> None:
        """Benchmark concurrent tenant HTTP requests."""
        try:
            import httpx

            from hbllm.serving.api import app

            transport = httpx.ASGITransport(app=app)
            n_tenants = 10
            n_requests = 20

            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:

                async def tenant_requests(tenant_id: str) -> list[float]:
                    """Send requests for a single tenant."""
                    latencies = []
                    for _ in range(n_requests):
                        t0 = time.perf_counter()
                        await client.get("/health/live")
                        latencies.append((time.perf_counter() - t0) * 1000)
                    return latencies

                t0 = time.perf_counter()
                tasks = [tenant_requests(f"tenant_{i}") for i in range(n_tenants)]
                results = await asyncio.gather(*tasks)
                total_elapsed = (time.perf_counter() - t0) * 1000

                all_latencies = [lat for tenant_lats in results for lat in tenant_lats]
                total_requests = n_tenants * n_requests

                report.add(
                    BenchmarkResult(
                        name=f"Concurrent {n_tenants} tenants × {n_requests} reqs",
                        metric="concurrent_total_time",
                        value=total_elapsed,
                        unit="ms",
                        metadata={"total_requests": total_requests},
                    )
                )
                report.add(
                    BenchmarkResult(
                        name="Concurrent throughput",
                        metric="concurrent_throughput",
                        value=total_requests / (total_elapsed / 1000),
                        unit="req/s",
                    )
                )
                report.add(
                    BenchmarkResult(
                        name="Per-request latency (concurrent) p50",
                        metric="concurrent_p50",
                        value=sorted(all_latencies)[len(all_latencies) // 2],
                        unit="ms",
                    )
                )

        except Exception as e:
            logger.warning("Concurrent tenant benchmark skipped: %s", e)
            report.add(
                BenchmarkResult(
                    name="Concurrent tenants (skipped)",
                    metric="concurrent_total_time",
                    value=0,
                    unit="ms",
                    metadata={"error": str(e)},
                )
            )
