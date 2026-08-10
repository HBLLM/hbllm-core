"""
HBLLM Observability — Distributed tracing and telemetry.

Provides OpenTelemetry integration with graceful degradation:
if the ``opentelemetry-api`` package is not installed, all
functions become silent no-ops.

Usage::

    from hbllm.observability import init_tracing, get_tracer

    # Initialize once at boot (no-op if OTel not installed)
    init_tracing()

    # Get a tracer for your module
    tracer = get_tracer("hbllm.brain")
"""

from hbllm.observability.tracing import get_tracer, init_tracing, trace_span

__all__ = ["init_tracing", "get_tracer", "trace_span"]
