"""
Prometheus Metrics Collector for HBLLM.

Provides counters, histograms, and gauges for monitoring the cognitive
pipeline in production. Falls back to in-memory counters when the
prometheus_client library is not installed.

Usage:
    metrics = MetricsCollector.get_instance()
    metrics.record_request("router.query", tenant_id="t1")

    with metrics.measure_latency("workspace"):
        await do_work()

    # Expose at /metrics for Prometheus scraping
    app.mount("/metrics", metrics.asgi_app())
"""

from __future__ import annotations

import contextlib
import logging
import time
import threading
from collections import defaultdict
from typing import Any, Generator

logger = logging.getLogger(__name__)

# Try to import prometheus_client, fall back to in-memory
try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Info,
        generate_latest,
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        REGISTRY,
    )
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


class MetricsCollector:
    """
    Enterprise metrics collection for HBLLM.

    Uses Prometheus client when available, falls back to in-memory
    counters for environments without the dependency.

    Thread-safe singleton.
    """

    _instance: MetricsCollector | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        if HAS_PROMETHEUS:
            self._init_prometheus()
        else:
            self._init_inmemory()

    @classmethod
    def get_instance(cls) -> MetricsCollector:
        """Get or create the singleton metrics collector."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    # ─── Prometheus Backend ───────────────────────────────────────────────

    def _init_prometheus(self) -> None:
        self._backend = "prometheus"

        self._requests_total = Counter(
            "hbllm_requests_total",
            "Total requests processed",
            ["topic", "tenant_id", "status"],
        )
        self._messages_total = Counter(
            "hbllm_messages_total",
            "Total messages published on the bus",
            ["topic", "message_type"],
        )
        self._errors_total = Counter(
            "hbllm_errors_total",
            "Total errors by node",
            ["node_id", "error_type"],
        )
        self._request_duration = Histogram(
            "hbllm_request_duration_seconds",
            "Request processing duration",
            ["stage"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
        )
        self._node_latency = Histogram(
            "hbllm_node_latency_seconds",
            "Per-node processing latency",
            ["node_id"],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
        )
        self._active_nodes = Gauge(
            "hbllm_active_nodes",
            "Number of active nodes",
        )
        self._healthy_nodes = Gauge(
            "hbllm_healthy_nodes",
            "Number of healthy nodes",
        )
        self._active_requests = Gauge(
            "hbllm_active_requests",
            "Currently in-flight requests",
        )
        self._info = Info(
            "hbllm_build",
            "Build information",
        )
        self._info.info({"version": "1.0.0", "bus": "unknown"})

        logger.info("MetricsCollector initialized with Prometheus backend")

    # ─── In-Memory Backend ────────────────────────────────────────────────

    def _init_inmemory(self) -> None:
        self._backend = "inmemory"
        self._mem_counters: dict[str, float] = defaultdict(float)
        self._mem_histograms: dict[str, list[float]] = defaultdict(list)
        self._mem_gauges: dict[str, float] = defaultdict(float)
        logger.info("MetricsCollector initialized with in-memory backend (install prometheus_client for full metrics)")

    # ─── Public API ───────────────────────────────────────────────────────

    @property
    def backend(self) -> str:
        return self._backend

    def record_request(
        self,
        topic: str,
        tenant_id: str = "default",
        status: str = "success",
    ) -> None:
        """Record a request (counter increment)."""
        if HAS_PROMETHEUS:
            self._requests_total.labels(topic=topic, tenant_id=tenant_id, status=status).inc()
        else:
            self._mem_counters[f"requests:{topic}:{status}"] += 1

    def record_message(self, topic: str, message_type: str = "event") -> None:
        """Record a bus message (counter increment)."""
        if HAS_PROMETHEUS:
            self._messages_total.labels(topic=topic, message_type=message_type).inc()
        else:
            self._mem_counters[f"messages:{topic}:{message_type}"] += 1

    def record_error(self, node_id: str, error_type: str = "exception") -> None:
        """Record an error (counter increment)."""
        if HAS_PROMETHEUS:
            self._errors_total.labels(node_id=node_id, error_type=error_type).inc()
        else:
            self._mem_counters[f"errors:{node_id}:{error_type}"] += 1

    def observe_duration(self, stage: str, duration_seconds: float) -> None:
        """Record a request duration (histogram observation)."""
        if HAS_PROMETHEUS:
            self._request_duration.labels(stage=stage).observe(duration_seconds)
        else:
            self._mem_histograms[f"duration:{stage}"].append(duration_seconds)

    def observe_node_latency(self, node_id: str, latency_seconds: float) -> None:
        """Record per-node latency (histogram observation)."""
        if HAS_PROMETHEUS:
            self._node_latency.labels(node_id=node_id).observe(latency_seconds)
        else:
            self._mem_histograms[f"latency:{node_id}"].append(latency_seconds)

    def set_active_nodes(self, count: int) -> None:
        """Set the active node count (gauge)."""
        if HAS_PROMETHEUS:
            self._active_nodes.set(count)
        else:
            self._mem_gauges["active_nodes"] = count

    def set_healthy_nodes(self, count: int) -> None:
        """Set the healthy node count (gauge)."""
        if HAS_PROMETHEUS:
            self._healthy_nodes.set(count)
        else:
            self._mem_gauges["healthy_nodes"] = count

    def inc_active_requests(self) -> None:
        """Increment active requests gauge."""
        if HAS_PROMETHEUS:
            self._active_requests.inc()
        else:
            self._mem_gauges["active_requests"] += 1

    def dec_active_requests(self) -> None:
        """Decrement active requests gauge."""
        if HAS_PROMETHEUS:
            self._active_requests.dec()
        else:
            self._mem_gauges["active_requests"] = max(0, self._mem_gauges["active_requests"] - 1)

    @contextlib.contextmanager
    def measure_latency(self, stage: str) -> Generator[None, None, None]:
        """Context manager to measure and record stage latency."""
        start = time.monotonic()
        try:
            yield
        finally:
            duration = time.monotonic() - start
            self.observe_duration(stage, duration)

    def get_metrics_text(self) -> str:
        """Generate Prometheus-format metrics text."""
        if HAS_PROMETHEUS:
            return generate_latest(REGISTRY).decode()
        else:
            lines = []
            for key, val in sorted(self._mem_counters.items()):
                lines.append(f"# COUNTER {key} {val}")
            for key, vals in sorted(self._mem_histograms.items()):
                if vals:
                    avg = sum(vals) / len(vals)
                    lines.append(f"# HISTOGRAM {key} count={len(vals)} avg={avg:.4f}")
            for key, val in sorted(self._mem_gauges.items()):
                lines.append(f"# GAUGE {key} {val}")
            return "\n".join(lines)

    def snapshot(self) -> dict[str, Any]:
        """Get a JSON-friendly snapshot of all metrics."""
        if not HAS_PROMETHEUS:
            return {
                "backend": "inmemory",
                "counters": dict(self._mem_counters),
                "gauges": dict(self._mem_gauges),
                "histograms": {
                    k: {"count": len(v), "avg": sum(v) / len(v) if v else 0}
                    for k, v in self._mem_histograms.items()
                },
            }
        return {"backend": "prometheus", "note": "Use /metrics endpoint for full data"}
