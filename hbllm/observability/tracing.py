"""
OpenTelemetry Tracing Integration for HBLLM Core.

Provides distributed tracing with graceful degradation: if the
``opentelemetry-api`` package is not installed, all functions
become silent no-ops. This allows the core package to run without
the ``[observability]`` extra while still having tracing wired in.

Configuration via environment variables:
  - ``OTEL_ENABLED``: Set to ``1`` to activate tracing (default: ``0``)
  - ``OTEL_SERVICE_NAME``: Service name for spans (default: ``hbllm``)
  - ``OTEL_EXPORTER_OTLP_ENDPOINT``: OTLP gRPC endpoint (default: ``http://localhost:4317``)

Usage::

    from hbllm.observability.tracing import init_tracing, get_tracer, trace_span

    # Initialize once at boot
    if init_tracing():
        print("Tracing active")

    # Create spans in your code
    tracer = get_tracer("hbllm.brain")
    with trace_span("process_query", {"tenant_id": "acme"}):
        ...
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

# ── Sentinel for OTel availability ───────────────────────────────────────────

_OTEL_AVAILABLE = False
_tracer_provider: Any = None
_initialized = False


def _check_otel() -> bool:
    """Check if OpenTelemetry packages are installed."""
    try:
        import opentelemetry.api  # noqa: F401
        import opentelemetry.sdk  # noqa: F401

        return True
    except ImportError:
        return False


# ── No-Op Tracer (used when OTel is not available) ───────────────────────────


class _NoOpSpan:
    """Minimal span interface that does nothing."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: Any, description: str | None = None) -> None:
        pass

    def record_exception(self, exception: BaseException) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpTracer:
    """Tracer that produces no-op spans."""

    def start_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()

    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()


_noop_tracer = _NoOpTracer()


# ── Public API ───────────────────────────────────────────────────────────────


def init_tracing(
    service_name: str | None = None,
    endpoint: str | None = None,
) -> bool:
    """Initialize OpenTelemetry tracing.

    Reads configuration from environment variables if not explicitly provided.
    Returns True if tracing was successfully initialized, False if OTel
    packages are not installed or tracing is disabled.

    Args:
        service_name: Service name for spans. Defaults to ``OTEL_SERVICE_NAME`` or ``hbllm``.
        endpoint: OTLP gRPC exporter endpoint. Defaults to ``OTEL_EXPORTER_OTLP_ENDPOINT``.

    Returns:
        True if tracing is active, False otherwise.
    """
    global _OTEL_AVAILABLE, _tracer_provider, _initialized

    if _initialized:
        return _OTEL_AVAILABLE

    # Check if tracing is enabled
    enabled = os.environ.get("OTEL_ENABLED", "0").strip()
    if enabled not in ("1", "true", "yes"):
        logger.debug("[Tracing] OTEL_ENABLED is not set — tracing disabled")
        _initialized = True
        return False

    # Check if OTel packages are available
    if not _check_otel():
        logger.info(
            "[Tracing] OpenTelemetry packages not installed. "
            "Install with: pip install hbllm[observability]"
        )
        _initialized = True
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        svc_name = service_name or os.environ.get("OTEL_SERVICE_NAME", "hbllm")
        otlp_endpoint = endpoint or os.environ.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
        )

        resource = Resource.create({"service.name": svc_name})
        provider = TracerProvider(resource=resource)

        exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)
        _tracer_provider = provider
        _OTEL_AVAILABLE = True
        _initialized = True

        logger.info(
            "[Tracing] OpenTelemetry initialized (service=%s, endpoint=%s)",
            svc_name,
            otlp_endpoint,
        )
        return True

    except Exception as e:
        logger.warning("[Tracing] Failed to initialize OpenTelemetry: %s", e)
        _initialized = True
        return False


def get_tracer(name: str = "hbllm") -> Any:
    """Get a tracer instance.

    Returns a real OpenTelemetry tracer if tracing is initialized,
    otherwise returns a no-op tracer.

    Args:
        name: Tracer name, typically the module path (e.g. ``hbllm.brain``).

    Returns:
        A tracer instance (real or no-op).
    """
    if _OTEL_AVAILABLE:
        from opentelemetry import trace

        return trace.get_tracer(name)
    return _noop_tracer


@contextmanager
def trace_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    tracer_name: str = "hbllm",
) -> Generator[Any, None, None]:
    """Context manager that creates a tracing span.

    If tracing is not initialized, this is a silent no-op.

    Args:
        name: Span name (e.g. ``process_query``, ``brain.reason``).
        attributes: Optional span attributes.
        tracer_name: Tracer name for this span.

    Yields:
        The span object (real or no-op).
    """
    if not _OTEL_AVAILABLE:
        yield _NoOpSpan()
        return

    from opentelemetry import trace

    tracer = trace.get_tracer(tracer_name)
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        yield span


def shutdown_tracing() -> None:
    """Flush and shut down the tracer provider.

    Should be called during graceful shutdown to ensure all
    pending spans are exported.
    """
    global _tracer_provider, _initialized, _OTEL_AVAILABLE

    if _tracer_provider is not None:
        try:
            _tracer_provider.shutdown()
            logger.info("[Tracing] Tracer provider shut down")
        except Exception as e:
            logger.debug("[Tracing] Error during shutdown: %s", e)
        _tracer_provider = None

    _initialized = False
    _OTEL_AVAILABLE = False
