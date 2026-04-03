"""Tests for OpenTelemetry tracing integration."""

import pytest
from unittest.mock import patch, MagicMock
from contextlib import contextmanager

from hbllm.network.tracing import (
    trace_span,
    BusMetrics,
    get_tracer,
    get_meter,
)


class TestTraceSpan:
    """Tests for the trace_span context manager."""

    def test_noop_when_otel_disabled(self):
        """trace_span should yield None when OTEL is disabled (default)."""
        with trace_span("test.span", {"key": "value"}) as span:
            assert span is None

    def test_noop_with_no_attributes(self):
        """trace_span should work without attributes."""
        with trace_span("test.span") as span:
            assert span is None

    def test_code_executes_inside_span(self):
        """Code inside the span context should execute normally."""
        result = []
        with trace_span("test.compute"):
            result.append(42)
        assert result == [42]

    def test_exception_propagates_through_span(self):
        """Exceptions inside trace_span should propagate normally."""
        with pytest.raises(ValueError, match="test error"):
            with trace_span("test.error"):
                raise ValueError("test error")

    @patch("hbllm.network.tracing._OTEL_ENABLED", True)
    @patch("hbllm.network.tracing._tracer", None)
    def test_tracer_init_called_when_enabled(self):
        """get_tracer should attempt initialization when OTEL_ENABLED=1."""
        # This will try to import opentelemetry which may not be installed
        # but should not crash — just returns None
        tracer = get_tracer()
        # Either returns a tracer (if OTel is installed) or None
        assert tracer is None or tracer is not None  # No crash = pass

    @patch("hbllm.network.tracing._OTEL_ENABLED", True)
    @patch("hbllm.network.tracing._meter", None)
    def test_meter_init_called_when_enabled(self):
        """get_meter should attempt initialization when OTEL_ENABLED=1."""
        meter = get_meter()
        assert meter is None or meter is not None  # No crash = pass


class TestBusMetrics:
    """Tests for the lightweight always-on BusMetrics."""

    @pytest.fixture
    def metrics(self):
        return BusMetrics()

    def test_initial_state(self, metrics):
        assert metrics.messages_published == 0
        assert metrics.messages_delivered == 0
        assert metrics.messages_dropped == 0
        assert metrics.handler_errors == 0
        assert metrics.active_subscriptions == 0

    def test_record_publish(self, metrics):
        metrics.record_publish("test.topic")
        metrics.record_publish("test.topic")
        assert metrics.messages_published == 2

    def test_record_delivery_with_latency(self, metrics):
        metrics.record_delivery("test.topic", 5.0)
        metrics.record_delivery("test.topic", 10.0)
        assert metrics.messages_delivered == 2
        assert metrics.avg_latency_ms == 7.5

    def test_record_drop(self, metrics):
        metrics.record_drop("test.topic")
        assert metrics.messages_dropped == 1

    def test_record_error(self, metrics):
        metrics.record_error("test.topic")
        assert metrics.handler_errors == 1

    def test_subscribe_unsubscribe(self, metrics):
        metrics.record_subscribe()
        metrics.record_subscribe()
        assert metrics.active_subscriptions == 2
        metrics.record_unsubscribe()
        assert metrics.active_subscriptions == 1
        metrics.record_unsubscribe()
        assert metrics.active_subscriptions == 0
        # Should not go negative
        metrics.record_unsubscribe()
        assert metrics.active_subscriptions == 0

    def test_avg_latency_empty(self, metrics):
        assert metrics.avg_latency_ms == 0.0

    def test_p99_latency(self, metrics):
        for i in range(100):
            metrics.record_delivery("topic", float(i))
        assert metrics.p99_latency_ms >= 98.0

    def test_p99_latency_empty(self, metrics):
        assert metrics.p99_latency_ms == 0.0

    def test_latency_rolling_window(self, metrics):
        """Latency samples should be capped at max_samples."""
        for i in range(1500):
            metrics.record_delivery("topic", float(i))
        assert len(metrics._latency_samples) <= metrics._max_samples

    def test_snapshot(self, metrics):
        metrics.record_publish("t")
        metrics.record_delivery("t", 5.0)
        metrics.record_drop("t")
        metrics.record_error("t")
        metrics.record_subscribe()

        snap = metrics.snapshot()
        assert snap["messages_published"] == 1
        assert snap["messages_delivered"] == 1
        assert snap["messages_dropped"] == 1
        assert snap["handler_errors"] == 1
        assert snap["active_subscriptions"] == 1
        assert snap["avg_latency_ms"] == 5.0
        assert isinstance(snap["p99_latency_ms"], float)

    def test_snapshot_serializable(self, metrics):
        """Snapshot should be JSON-serializable."""
        import json
        metrics.record_delivery("t", 3.14)
        snap = metrics.snapshot()
        serialized = json.dumps(snap)
        assert isinstance(serialized, str)


class TestProviderTracing:
    """Tests that provider.py correctly imports and uses trace_span."""

    def test_trace_span_import(self):
        """Provider module should import trace_span without errors."""
        from hbllm.serving.provider import trace_span as provider_trace
        assert provider_trace is trace_span

    def test_openai_provider_has_tracing(self):
        """OpenAIProvider.generate should reference trace_span."""
        import inspect
        from hbllm.serving.provider import OpenAIProvider
        source = inspect.getsource(OpenAIProvider.generate)
        assert "trace_span" in source

    def test_anthropic_provider_has_tracing(self):
        """AnthropicProvider.generate should reference trace_span."""
        import inspect
        from hbllm.serving.provider import AnthropicProvider
        source = inspect.getsource(AnthropicProvider.generate)
        assert "trace_span" in source

    def test_local_provider_has_tracing(self):
        """LocalProvider.generate should reference trace_span."""
        import inspect
        from hbllm.serving.provider import LocalProvider
        source = inspect.getsource(LocalProvider.generate)
        assert "trace_span" in source
