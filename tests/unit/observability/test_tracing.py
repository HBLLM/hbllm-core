"""Unit tests for the HBLLM observability module.

Tests graceful degradation (no OTel installed) and core functionality.
"""

from __future__ import annotations

from unittest.mock import patch

# ── Graceful Degradation Tests ───────────────────────────────────────────────


class TestTracingGracefulDegradation:
    """Verify tracing works as silent no-ops when OTel is not installed."""

    def setup_method(self) -> None:
        """Reset tracing state between tests."""
        import hbllm.observability.tracing as mod

        mod._initialized = False
        mod._OTEL_AVAILABLE = False
        mod._tracer_provider = None

    def test_init_tracing_returns_false_when_disabled(self) -> None:
        """init_tracing() returns False when OTEL_ENABLED is not set."""
        from hbllm.observability.tracing import init_tracing

        with patch.dict("os.environ", {"OTEL_ENABLED": "0"}, clear=False):
            result = init_tracing()
        assert result is False

    def test_init_tracing_returns_false_when_otel_missing(self) -> None:
        """init_tracing() returns False when OTel packages are not installed."""
        from hbllm.observability.tracing import init_tracing

        with (
            patch.dict("os.environ", {"OTEL_ENABLED": "1"}, clear=False),
            patch("hbllm.observability.tracing._check_otel", return_value=False),
        ):
            result = init_tracing()
        assert result is False

    def test_get_tracer_returns_noop_when_disabled(self) -> None:
        """get_tracer() returns a no-op tracer when tracing is not initialized."""
        from hbllm.observability.tracing import _NoOpTracer, get_tracer

        tracer = get_tracer("test")
        assert isinstance(tracer, _NoOpTracer)

    def test_trace_span_is_noop_when_disabled(self) -> None:
        """trace_span() context manager works without error when tracing is disabled."""
        from hbllm.observability.tracing import _NoOpSpan, trace_span

        with trace_span("test_operation", {"key": "value"}) as span:
            assert isinstance(span, _NoOpSpan)
            # Should not raise
            span.set_attribute("test", "value")
            span.record_exception(RuntimeError("test"))

    def test_shutdown_tracing_is_safe_when_not_initialized(self) -> None:
        """shutdown_tracing() doesn't raise when tracing was never initialized."""
        from hbllm.observability.tracing import shutdown_tracing

        # Should not raise
        shutdown_tracing()


# ── No-Op Span Tests ─────────────────────────────────────────────────────────


class TestNoOpSpan:
    """Verify the no-op span implements the required interface."""

    def test_noop_span_context_manager(self) -> None:
        from hbllm.observability.tracing import _NoOpSpan

        span = _NoOpSpan()
        with span as s:
            assert s is span

    def test_noop_span_set_attribute(self) -> None:
        from hbllm.observability.tracing import _NoOpSpan

        span = _NoOpSpan()
        span.set_attribute("key", "value")  # Should not raise

    def test_noop_span_record_exception(self) -> None:
        from hbllm.observability.tracing import _NoOpSpan

        span = _NoOpSpan()
        span.record_exception(ValueError("test"))  # Should not raise

    def test_noop_span_set_status(self) -> None:
        from hbllm.observability.tracing import _NoOpSpan

        span = _NoOpSpan()
        span.set_status("OK")  # Should not raise

    def test_noop_span_end(self) -> None:
        from hbllm.observability.tracing import _NoOpSpan

        span = _NoOpSpan()
        span.end()  # Should not raise


# ── No-Op Tracer Tests ───────────────────────────────────────────────────────


class TestNoOpTracer:
    """Verify the no-op tracer produces no-op spans."""

    def test_start_span_returns_noop(self) -> None:
        from hbllm.observability.tracing import _NoOpSpan, _NoOpTracer

        tracer = _NoOpTracer()
        span = tracer.start_span("test")
        assert isinstance(span, _NoOpSpan)

    def test_start_as_current_span_returns_noop(self) -> None:
        from hbllm.observability.tracing import _NoOpSpan, _NoOpTracer

        tracer = _NoOpTracer()
        span = tracer.start_as_current_span("test")
        assert isinstance(span, _NoOpSpan)


# ── Middleware Tests ─────────────────────────────────────────────────────────


class TestTracingMiddleware:
    """Verify middleware passes through when tracing is disabled."""

    def test_middleware_passthrough_when_disabled(self) -> None:
        """Middleware should not interfere with requests when tracing is off."""
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route
        from starlette.testclient import TestClient

        from hbllm.observability.middleware import TracingMiddleware

        async def homepage(request):
            return JSONResponse({"status": "ok"})

        app = Starlette(routes=[Route("/test", homepage)])
        app.add_middleware(TracingMiddleware)

        client = TestClient(app)
        response = client.get("/test")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_middleware_skips_health_endpoints(self) -> None:
        """Middleware should skip health check paths."""
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route
        from starlette.testclient import TestClient

        from hbllm.observability.middleware import TracingMiddleware

        async def health(request):
            return JSONResponse({"status": "healthy"})

        app = Starlette(routes=[Route("/health", health)])
        app.add_middleware(TracingMiddleware)

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200


# ── Integration Sanity ───────────────────────────────────────────────────────


class TestObservabilityPackage:
    """Verify the package exports are correct."""

    def test_package_imports(self) -> None:
        from hbllm.observability import get_tracer, init_tracing, trace_span

        assert callable(init_tracing)
        assert callable(get_tracer)
        assert callable(trace_span)

    def test_init_tracing_idempotent(self) -> None:
        """Calling init_tracing() multiple times should be safe."""
        import hbllm.observability.tracing as mod

        mod._initialized = False
        mod._OTEL_AVAILABLE = False

        from hbllm.observability.tracing import init_tracing

        with patch.dict("os.environ", {"OTEL_ENABLED": "0"}, clear=False):
            r1 = init_tracing()
            r2 = init_tracing()
        assert r1 is False
        assert r2 is False
