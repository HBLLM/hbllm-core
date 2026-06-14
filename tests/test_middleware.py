"""
Unit tests for the serving middleware stack:
  - HTTPRateLimitMiddleware (rate_limit.py)
  - PrometheusMiddleware (prometheus.py)
  - APIVersionMiddleware (api_version.py)

These test the middleware in isolation without starting the full Brain.
"""

from __future__ import annotations

import time
import unittest


# ── Rate Limiting Middleware ─────────────────────────────────────────────────


class TestTenantBucket(unittest.TestCase):
    """Test the _TenantBucket rate limiting primitive."""

    def test_initial_state(self):
        from hbllm.serving.middleware.rate_limit import _TenantBucket

        bucket = _TenantBucket(rpm=60, burst=60)
        assert bucket.rpm == 60

    def test_try_consume_succeeds_within_limit(self):
        from hbllm.serving.middleware.rate_limit import _TenantBucket

        bucket = _TenantBucket(rpm=5, burst=5)
        results = [bucket.try_consume() for _ in range(5)]
        assert all(results), "All 5 requests within limit should succeed"

    def test_try_consume_fails_over_limit(self):
        from hbllm.serving.middleware.rate_limit import _TenantBucket

        bucket = _TenantBucket(rpm=3, burst=3)
        for _ in range(3):
            bucket.try_consume()
        assert not bucket.try_consume(), "4th request over limit should fail"

    def test_tokens_refill_over_time(self):
        from hbllm.serving.middleware.rate_limit import _TenantBucket

        bucket = _TenantBucket(rpm=60, burst=60)
        # Drain all tokens
        for _ in range(60):
            bucket.try_consume()
        assert not bucket.try_consume(), "Should be empty"

        # Simulate time passing (push _last_refill back)
        bucket.last_refill = time.monotonic() - 2.0
        assert bucket.try_consume(), "Should have refilled after 2s"

    def test_retry_after_is_positive_when_empty(self):
        from hbllm.serving.middleware.rate_limit import _TenantBucket

        bucket = _TenantBucket(rpm=1, burst=1)
        bucket.try_consume()  # Use the only token
        bucket.try_consume()  # Over limit
        retry = bucket.retry_after
        assert retry > 0, f"retry_after should be positive, got {retry}"
        assert retry <= 60, f"retry_after should be <= 60s, got {retry}"


class TestHTTPRateLimitMiddleware(unittest.TestCase):
    """Test the middleware's tenant extraction and bucket management."""

    def test_get_bucket_returns_consistent_bucket(self):
        from hbllm.serving.middleware.rate_limit import HTTPRateLimitMiddleware

        mw = HTTPRateLimitMiddleware(app=None, default_rpm=10)
        b1 = mw._get_bucket("tenant_a")
        b2 = mw._get_bucket("tenant_a")
        assert b1 is b2, "Same tenant should return same bucket"

    def test_different_tenants_get_different_buckets(self):
        from hbllm.serving.middleware.rate_limit import HTTPRateLimitMiddleware

        mw = HTTPRateLimitMiddleware(app=None, default_rpm=10)
        b1 = mw._get_bucket("tenant_a")
        b2 = mw._get_bucket("tenant_b")
        assert b1 is not b2, "Different tenants should get different buckets"

    def test_burst_generates_rejections(self):
        from hbllm.serving.middleware.rate_limit import _TenantBucket

        # Use a bucket directly with matching rpm and burst
        bucket = _TenantBucket(rpm=5, burst=5)
        accepted = sum(1 for _ in range(10) if bucket.try_consume())
        rejected = 10 - accepted
        assert accepted <= 5, f"Should accept at most 5, got {accepted}"
        assert rejected >= 5, f"Should reject at least 5, got {rejected}"


# ── Prometheus Middleware ────────────────────────────────────────────────────


class TestPrometheusMetrics(unittest.TestCase):
    """Test Prometheus metrics collection."""

    def test_metrics_initializes(self):
        from hbllm.serving.middleware.prometheus import PrometheusMetrics

        metrics = PrometheusMetrics()
        assert hasattr(metrics, "request_counts")
        assert hasattr(metrics, "_durations")
        assert hasattr(metrics, "in_flight")

    def test_record_request_works(self):
        from hbllm.serving.middleware.prometheus import PrometheusMetrics

        metrics = PrometheusMetrics()
        # record_request(method, path, status, duration)
        metrics.record_request("GET", "/health", 200, 0.01)
        assert len(metrics.request_counts) > 0

    def test_render_produces_text(self):
        from hbllm.serving.middleware.prometheus import PrometheusMetrics

        metrics = PrometheusMetrics()
        metrics.record_request("GET", "/health", 200, 0.01)
        metrics.record_request("POST", "/v1/chat", 200, 0.5)
        text = metrics.render()
        assert "hbllm_http_requests_total" in text
        assert "hbllm_http_request_duration_seconds" in text

    def test_render_empty_metrics(self):
        from hbllm.serving.middleware.prometheus import PrometheusMetrics

        metrics = PrometheusMetrics()
        text = metrics.render()
        assert isinstance(text, str)

    def test_path_normalization(self):
        from hbllm.serving.middleware.prometheus import PrometheusMetrics

        # The _normalize_path static method should handle paths
        normalized = PrometheusMetrics._normalize_path("/v1/chat/completions")
        assert isinstance(normalized, str)
        assert len(normalized) > 0


# ── API Version Middleware ───────────────────────────────────────────────────


class TestAPIVersionConstants(unittest.TestCase):
    """Test API version module constants."""

    def test_supported_versions_is_not_empty(self):
        from hbllm.serving.middleware.api_version import SUPPORTED_VERSIONS

        assert len(SUPPORTED_VERSIONS) > 0

    def test_current_version_is_in_supported(self):
        from hbllm.serving.middleware.api_version import (
            CURRENT_VERSION,
            SUPPORTED_VERSIONS,
        )

        assert CURRENT_VERSION in SUPPORTED_VERSIONS

    def test_v1_is_supported(self):
        from hbllm.serving.middleware.api_version import SUPPORTED_VERSIONS

        assert "v1" in SUPPORTED_VERSIONS

    def test_unsupported_version_not_in_set(self):
        from hbllm.serving.middleware.api_version import SUPPORTED_VERSIONS

        assert "v99" not in SUPPORTED_VERSIONS
        assert "invalid" not in SUPPORTED_VERSIONS

    def test_current_version_format(self):
        from hbllm.serving.middleware.api_version import CURRENT_VERSION

        assert CURRENT_VERSION.startswith("v"), f"Should start with 'v', got {CURRENT_VERSION}"
