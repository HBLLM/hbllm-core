"""
Prometheus Metrics Middleware — exports /metrics/prometheus endpoint.

Provides request-level instrumentation compatible with Prometheus/Grafana
and Datadog. All metrics are computed from in-memory counters (zero external deps).

Exported metrics:
  - hbllm_http_requests_total{method, path, status}
  - hbllm_http_request_duration_seconds{method, path}
  - hbllm_http_requests_in_flight
  - hbllm_llm_calls_total{tier}  (from DualLLMRouter stats if available)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """In-memory Prometheus-style metrics collector."""

    def __init__(self) -> None:
        # Counter: {(method, path, status): count}
        self.request_counts: dict[tuple[str, str, int], int] = defaultdict(int)
        # Histogram buckets for request duration
        self._duration_buckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        # {(method, path): [bucket_counts..., +Inf_count, sum, count]}
        self._durations: dict[tuple[str, str], list[float]] = defaultdict(
            lambda: [0.0] * (len(self._duration_buckets) + 3)  # +Inf, sum, count
        )
        self.in_flight = 0

    def record_request(self, method: str, path: str, status: int, duration: float) -> None:
        """Record a completed request."""
        normalized = self._normalize_path(path)
        self.request_counts[(method, normalized, status)] += 1

        # Duration histogram
        entry = self._durations[(method, normalized)]
        for i, bucket in enumerate(self._duration_buckets):
            if duration <= bucket:
                entry[i] += 1
        entry[-3] += 1  # +Inf (always incremented)
        entry[-2] += duration  # sum
        entry[-1] += 1  # count

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize path to prevent cardinality explosion from IDs."""
        parts = path.strip("/").split("/")
        normalized = []
        for part in parts:
            # Replace UUIDs and numeric IDs
            if len(part) > 20 or part.isdigit():
                normalized.append(":id")
            else:
                normalized.append(part)
        return "/" + "/".join(normalized) if normalized else "/"

    def render(self, extra_lines: list[str] | None = None) -> str:
        """Render metrics in Prometheus text exposition format."""
        lines: list[str] = []

        # Request counter
        lines.append("# HELP hbllm_http_requests_total Total HTTP requests.")
        lines.append("# TYPE hbllm_http_requests_total counter")
        for (method, path, status), count in sorted(self.request_counts.items()):
            lines.append(
                f'hbllm_http_requests_total{{method="{method}",path="{path}",status="{status}"}} {count}'
            )

        # Duration histogram
        lines.append("# HELP hbllm_http_request_duration_seconds Request duration in seconds.")
        lines.append("# TYPE hbllm_http_request_duration_seconds histogram")
        for (method, path), entry in sorted(self._durations.items()):
            cum = 0.0
            for i, bucket in enumerate(self._duration_buckets):
                cum += entry[i]
                lines.append(
                    f'hbllm_http_request_duration_seconds_bucket{{method="{method}",path="{path}",le="{bucket}"}} {int(cum)}'
                )
            cum += entry[-3]
            lines.append(
                f'hbllm_http_request_duration_seconds_bucket{{method="{method}",path="{path}",le="+Inf"}} {int(cum)}'
            )
            lines.append(
                f'hbllm_http_request_duration_seconds_sum{{method="{method}",path="{path}"}} {entry[-2]:.6f}'
            )
            lines.append(
                f'hbllm_http_request_duration_seconds_count{{method="{method}",path="{path}"}} {int(entry[-1])}'
            )

        # In-flight gauge
        lines.append("# HELP hbllm_http_requests_in_flight Current in-flight requests.")
        lines.append("# TYPE hbllm_http_requests_in_flight gauge")
        lines.append(f"hbllm_http_requests_in_flight {self.in_flight}")

        # Extra metrics from brain (e.g., DualLLMRouter stats)
        if extra_lines:
            lines.extend(extra_lines)

        lines.append("")
        return "\n".join(lines)


# Global metrics instance
_metrics = PrometheusMetrics()


def get_prometheus_metrics() -> PrometheusMetrics:
    """Get the global Prometheus metrics collector."""
    return _metrics


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware that records request-level metrics for Prometheus export."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Skip the metrics endpoint itself
        if request.url.path == "/metrics/prometheus":
            return await call_next(request)

        _metrics.in_flight += 1
        start = time.monotonic()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            status_code = 500
            raise
        finally:
            _metrics.in_flight -= 1
            duration = time.monotonic() - start
            _metrics.record_request(request.method, request.url.path, status_code, duration)


def prometheus_endpoint(brain_getter: Any = None) -> PlainTextResponse:
    """FastAPI endpoint handler for /metrics/prometheus."""
    extra: list[str] = []

    # Include DualLLMRouter stats if available
    if brain_getter is not None:
        try:
            brain = brain_getter()
            if brain is not None:
                dual_router = getattr(brain, "dual_router", None)
                if dual_router is not None:
                    snap = dual_router.snapshot()
                    stats = snap.get("stats", {})
                    extra.append("# HELP hbllm_llm_calls_total LLM calls by tier.")
                    extra.append("# TYPE hbllm_llm_calls_total counter")
                    extra.append(
                        f'hbllm_llm_calls_total{{tier="local"}} {stats.get("local_calls", 0)}'
                    )
                    extra.append(
                        f'hbllm_llm_calls_total{{tier="external"}} {stats.get("external_calls", 0)}'
                    )
                    extra.append(
                        f'hbllm_llm_calls_total{{tier="fallback"}} {stats.get("fallbacks", 0)}'
                    )

                    cb = snap.get("circuit_breaker", {})
                    extra.append("# HELP hbllm_circuit_breaker_state External LLM circuit state.")
                    extra.append("# TYPE hbllm_circuit_breaker_state gauge")
                    state_val = {"closed": 0, "open": 1, "half_open": 2, "partial_open": 3}.get(
                        cb.get("state", "closed"), 0
                    )
                    extra.append(f'hbllm_circuit_breaker_state{{node="external_llm"}} {state_val}')
        except Exception:
            logger.debug("Error collecting brain metrics for Prometheus", exc_info=True)

    return PlainTextResponse(
        content=_metrics.render(extra_lines=extra if extra else None),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
