"""
OpenTelemetry Tracing for the HBLLM MessageBus.

Provides automatic span creation for all message publish, subscribe dispatch,
and request/response flows. Traces propagate correlation IDs as span links
so the full cognitive pipeline is visible in Jaeger / Zipkin.

Usage:
    Set OTEL_ENABLED=1 to activate tracing (disabled by default).
    Configure the exporter via standard OTEL env vars:
        OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
        OTEL_SERVICE_NAME=hbllm
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)

# ── Feature flag ──────────────────────────────────────────────────────────────

_OTEL_ENABLED = os.environ.get("OTEL_ENABLED", "0") == "1"
_tracer = None
_meter = None

def _init_otel():
    """Lazily initialize OpenTelemetry SDK."""
    global _tracer, _meter
    if _tracer is not None:
        return
    
    try:
        from opentelemetry import trace, metrics
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.resources import Resource
        
        resource = Resource.create({
            "service.name": os.environ.get("OTEL_SERVICE_NAME", "hbllm"),
            "service.version": "1.0.0",
        })
        
        # Trace provider
        tracer_provider = TracerProvider(resource=resource)
        
        # Try OTLP exporter, fall back to console
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter()
        except ImportError:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            exporter = ConsoleSpanExporter()
        
        tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(tracer_provider)
        _tracer = trace.get_tracer("hbllm.network")
        
        # Metrics provider
        meter_provider = MeterProvider(resource=resource)
        metrics.set_meter_provider(meter_provider)
        _meter = metrics.get_meter("hbllm.network")
        
        logger.info("OpenTelemetry tracing initialized (exporter: %s)", exporter.__class__.__name__)
        
    except ImportError:
        logger.warning("OpenTelemetry SDK not installed. Tracing disabled.")
        _OTEL_ENABLED_RUNTIME = False


def get_tracer():
    """Get the HBLLM tracer (may be None if OTEL is disabled)."""
    if _OTEL_ENABLED:
        _init_otel()
    return _tracer


def get_meter():
    """Get the HBLLM meter (may be None if OTEL is disabled)."""
    if _OTEL_ENABLED:
        _init_otel()
    return _meter


# ── Lightweight Metrics (always active, zero-dep) ────────────────────────────

class BusMetrics:
    """
    Lightweight, always-on metrics for the MessageBus.
    
    These work without OpenTelemetry and provide real-time counters/histograms
    that the /health and /metrics endpoints can expose.
    """
    
    def __init__(self):
        self.messages_published: int = 0
        self.messages_delivered: int = 0
        self.messages_dropped: int = 0
        self.handler_errors: int = 0
        self.active_subscriptions: int = 0
        self._latency_samples: list[float] = []
        self._max_samples = 1000  # Rolling window
    
    def record_publish(self, topic: str) -> None:
        self.messages_published += 1
    
    def record_delivery(self, topic: str, latency_ms: float) -> None:
        self.messages_delivered += 1
        self._latency_samples.append(latency_ms)
        if len(self._latency_samples) > self._max_samples:
            self._latency_samples = self._latency_samples[-self._max_samples:]
    
    def record_drop(self, topic: str) -> None:
        self.messages_dropped += 1
    
    def record_error(self, topic: str) -> None:
        self.handler_errors += 1
    
    def record_subscribe(self) -> None:
        self.active_subscriptions += 1
    
    def record_unsubscribe(self) -> None:
        self.active_subscriptions = max(0, self.active_subscriptions - 1)
    
    @property
    def avg_latency_ms(self) -> float:
        if not self._latency_samples:
            return 0.0
        return sum(self._latency_samples) / len(self._latency_samples)
    
    @property
    def p99_latency_ms(self) -> float:
        if not self._latency_samples:
            return 0.0
        sorted_samples = sorted(self._latency_samples)
        idx = int(len(sorted_samples) * 0.99)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]
    
    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of all metrics."""
        return {
            "messages_published": self.messages_published,
            "messages_delivered": self.messages_delivered,
            "messages_dropped": self.messages_dropped,
            "handler_errors": self.handler_errors,
            "active_subscriptions": self.active_subscriptions,
            "avg_latency_ms": round(self.avg_latency_ms, 3),
            "p99_latency_ms": round(self.p99_latency_ms, 3),
        }


@contextmanager
def trace_span(name: str, attributes: dict[str, str] | None = None) -> Generator[Any, None, None]:
    """
    Context manager that creates an OTEL span if tracing is enabled,
    otherwise acts as a no-op.
    """
    tracer = get_tracer()
    if tracer:
        with tracer.start_as_current_span(name, attributes=attributes or {}) as span:
            yield span
    else:
        yield None
