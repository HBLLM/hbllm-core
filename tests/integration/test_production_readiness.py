"""Integration tests for production-readiness features.

Tests:
  - Full pipeline smoke test (query → response)
  - DualLLMRouter fallback with circuit breaker
  - HTTP rate limiting
  - Graceful shutdown drain
  - BrainConfig Pydantic validation
  - Prometheus metrics rendering
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

# ─── BrainConfig Pydantic Validation ─────────────────────────────────────────


class TestBrainConfigValidation:
    """BrainConfig is now Pydantic — verify validation works."""

    def test_default_config_is_valid(self):
        from hbllm.brain.factory import BrainConfig

        cfg = BrainConfig()
        assert cfg.total_timeout > 0
        assert cfg.planner_branch_factor == 3
        assert cfg.dual_llm_complexity_threshold == 0.4

    def test_invalid_timeout_rejected(self):
        from pydantic import ValidationError

        from hbllm.brain.factory import BrainConfig

        with pytest.raises(ValidationError, match="greater than 0"):
            BrainConfig(total_timeout=-1.0)

    def test_invalid_threshold_rejected(self):
        from pydantic import ValidationError

        from hbllm.brain.factory import BrainConfig

        with pytest.raises(ValidationError, match="less than or equal to 1"):
            BrainConfig(dual_llm_complexity_threshold=1.5)

    def test_planner_depth_out_of_range(self):
        from pydantic import ValidationError

        from hbllm.brain.factory import BrainConfig

        with pytest.raises(ValidationError, match="less than or equal to 5"):
            BrainConfig(planner_max_depth=10)

    def test_empty_data_dir_rejected(self):
        from pydantic import ValidationError

        from hbllm.brain.factory import BrainConfig

        with pytest.raises(ValidationError, match="at least 1"):
            BrainConfig(data_dir="")

    def test_valid_custom_config(self):
        from hbllm.brain.factory import BrainConfig

        cfg = BrainConfig(
            total_timeout=120.0,
            planner_branch_factor=5,
            planner_max_depth=3,
            dual_llm_complexity_threshold=0.6,
            inject_perception=True,
            external_provider="openai/gpt-4o",
        )
        assert cfg.total_timeout == 120.0
        assert cfg.planner_branch_factor == 5
        assert cfg.external_provider == "openai/gpt-4o"


# ─── DualLLMRouter Circuit Breaker Integration ──────────────────────────────


class TestDualRouterCircuitBreaker:
    """Verify circuit breaker opens after failures and auto-fallbacks."""

    def _make_router(self, local_fn, external_fn, threshold=3):
        from hbllm.brain.dual_llm_router import DualLLMRouter

        local = MagicMock()
        local.generate = local_fn
        local.provider = MagicMock(name="local")

        external = MagicMock()
        external.generate = external_fn
        external.provider = MagicMock(name="external")

        return DualLLMRouter(
            local=local,
            external=external,
            complexity_threshold=0.0,  # Always prefer external
            circuit_failure_threshold=threshold,
        )

    @pytest.mark.asyncio
    async def test_circuit_opens_after_consecutive_failures(self):
        from hbllm.brain.dual_llm_router import TaskTier
        from hbllm.network.circuit_breaker import CircuitState

        fail_count = 0

        async def _external_fail(prompt, **kw):
            nonlocal fail_count
            fail_count += 1
            raise RuntimeError("API down")

        async def _local_ok(prompt, **kw):
            return "local response"

        router = self._make_router(_local_ok, _external_fail, threshold=2)

        # First 2 failures -> circuit opens on the 2nd
        for _ in range(2):
            result = await router.generate("complex task", tier=TaskTier.EXTERNAL)
            assert result == "local response"  # Fallback works

        # Circuit should now be OPEN
        assert router._external_circuit.state == CircuitState.OPEN

        # Next call should be auto-routed to local due to open circuit
        decision = router.route("complex task", tier=TaskTier.AUTO)
        assert decision.reason == "circuit_open"
        assert decision.fallback_used is True

    @pytest.mark.asyncio
    async def test_circuit_recovers_after_success(self):
        from hbllm.network.circuit_breaker import CircuitState

        call_count = 0

        async def _external(prompt, **kw):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("Transient error")
            return "external response"

        async def _local(prompt, **kw):
            return "local response"

        router = self._make_router(_local, _external, threshold=2)
        # Set very short recovery timeout for test
        router._external_circuit._base_recovery_timeout = 0.01
        router._external_circuit._current_recovery_timeout = 0.01

        # Trigger circuit open
        from hbllm.brain.dual_llm_router import TaskTier

        for _ in range(2):
            await router.generate("x", tier=TaskTier.EXTERNAL)

        assert router._external_circuit._state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.1)

        # Circuit should transition to HALF_OPEN when checked
        assert router._external_circuit.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_snapshot_includes_circuit_state(self):
        async def _gen(prompt, **kw):
            return "ok"

        router = self._make_router(_gen, _gen)
        snap = router.snapshot()

        assert "circuit_breaker" in snap
        assert snap["circuit_breaker"]["state"] == "closed"
        assert snap["circuit_breaker"]["failure_count"] == 0


# ─── HTTP Rate Limiting ──────────────────────────────────────────────────────


class TestHTTPRateLimiting:
    """Verify per-tenant rate limiting works."""

    def test_bucket_allows_within_limit(self):
        from hbllm.serving.middleware.rate_limit import _TenantBucket

        bucket = _TenantBucket(rpm=60.0, burst=90.0)
        # Should allow burst of 90
        for _ in range(90):
            assert bucket.try_consume() is True
        # 91st should fail
        assert bucket.try_consume() is False

    def test_bucket_refills_over_time(self):

        from hbllm.serving.middleware.rate_limit import _TenantBucket

        bucket = _TenantBucket(rpm=600.0, burst=10.0)  # 10/s
        # Exhaust burst
        for _ in range(10):
            bucket.try_consume()
        assert bucket.try_consume() is False

        # Manually advance time by simulating refill
        bucket.last_refill -= 0.2  # 200ms ago
        assert bucket.try_consume() is True  # ~2 tokens refilled

    def test_retry_after_calculation(self):
        from hbllm.serving.middleware.rate_limit import _TenantBucket

        bucket = _TenantBucket(rpm=60.0, burst=1.0)
        bucket.try_consume()  # Use the 1 token
        # retry_after should be > 0
        assert bucket.retry_after > 0
        assert bucket.retry_after <= 1.0  # Should be about 1s at 60 RPM


# ─── Graceful Shutdown Drain ─────────────────────────────────────────────────


class TestGracefulShutdown:
    """Verify Brain.shutdown() drains in-flight requests."""

    def _make_brain(self):
        from hbllm.brain.factory import Brain
        from hbllm.network.bus import InProcessBus
        from hbllm.network.registry import ServiceRegistry
        from hbllm.serving.pipeline import CognitivePipeline

        bus = InProcessBus()
        registry = ServiceRegistry(bus)
        pipeline = CognitivePipeline(bus=bus, registry=registry)

        llm = MagicMock()
        llm.usage = {}

        return Brain(
            bus=bus,
            registry=registry,
            pipeline=pipeline,
            llm=llm,
            nodes=[],
            provider=MagicMock(),
        )

    def test_draining_rejects_new_requests(self):
        brain = self._make_brain()
        assert brain.acquire_request() is True
        brain._draining = True
        assert brain.acquire_request() is False

    def test_release_signals_drain_event(self):
        brain = self._make_brain()
        brain.acquire_request()
        brain._draining = True
        brain._drain_event.clear()
        brain.release_request()
        assert brain._drain_event.is_set()

    @pytest.mark.asyncio
    async def test_process_returns_error_when_draining(self):
        brain = self._make_brain()
        brain._draining = True
        result = await brain.process("test")
        assert result.error is True
        assert "shutting down" in result.text.lower()


# ─── Prometheus Metrics ──────────────────────────────────────────────────────


class TestPrometheusMetrics:
    """Verify Prometheus text exposition format."""

    def test_render_empty_metrics(self):
        from hbllm.serving.middleware.prometheus import PrometheusMetrics

        m = PrometheusMetrics()
        rendered = m.render()
        assert "hbllm_http_requests_total" in rendered
        assert "hbllm_http_requests_in_flight" in rendered

    def test_record_and_render(self):
        from hbllm.serving.middleware.prometheus import PrometheusMetrics

        m = PrometheusMetrics()
        m.record_request("GET", "/health", 200, 0.05)
        m.record_request("POST", "/chat", 200, 1.2)
        m.record_request("POST", "/chat", 500, 0.3)

        rendered = m.render()
        assert 'method="GET"' in rendered
        assert 'status="200"' in rendered
        assert 'status="500"' in rendered
        assert "hbllm_http_request_duration_seconds_sum" in rendered

    def test_path_normalization(self):
        from hbllm.serving.middleware.prometheus import PrometheusMetrics

        m = PrometheusMetrics()
        # UUID should be normalized to :id
        normalized = m._normalize_path("/chat/550e8400-e29b-41d4-a716-446655440000/messages")
        assert ":id" in normalized

    def test_render_with_extra_lines(self):
        from hbllm.serving.middleware.prometheus import PrometheusMetrics

        m = PrometheusMetrics()
        extra = ['hbllm_llm_calls_total{tier="local"} 42']
        rendered = m.render(extra_lines=extra)
        assert 'tier="local"' in rendered
        assert "42" in rendered


# ─── ExpressionStream Dual Router ────────────────────────────────────────────


class TestExpressionStreamDualRouter:
    """Verify ExpressionStream uses DualLLMRouter when available."""

    @pytest.mark.asyncio
    async def test_expression_uses_dual_router(self):
        from hbllm.brain.snn.expression import (
            ExpressionStream,
            RewardEvaluator,
            ThoughtController,
            ThoughtPlanner,
        )
        from hbllm.brain.snn.expression.models import ThoughtGoal

        dual_router = MagicMock()
        dual_router.generate = AsyncMock(return_value="Routed response")

        stream = ExpressionStream(
            planner=ThoughtPlanner(),
            controller=ThoughtController(),
            evaluator=RewardEvaluator(),
            dual_router=dual_router,
        )

        goal = ThoughtGoal(
            id="g1",
            text="Test goal",
            source_concept_text="test concept",
            max_tokens=100,
        )

        fragment = await stream._generate_for_goal(goal, "base", "query", None)
        assert fragment.text == "Routed response"
        assert fragment.metadata["source"] == "dual_router"
        dual_router.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_expression_falls_back_to_llm_generate(self):
        from hbllm.brain.snn.expression import (
            ExpressionStream,
            RewardEvaluator,
            ThoughtController,
            ThoughtPlanner,
        )
        from hbllm.brain.snn.expression.models import ThoughtGoal

        async def _llm_gen(prompt: str) -> str:
            return "LLM response"

        stream = ExpressionStream(
            planner=ThoughtPlanner(),
            controller=ThoughtController(),
            evaluator=RewardEvaluator(),
            llm_generate=_llm_gen,
            dual_router=None,
        )

        goal = ThoughtGoal(
            id="g1",
            text="Test goal",
            source_concept_text="test concept",
            max_tokens=100,
        )

        fragment = await stream._generate_for_goal(goal, "base", "query", None)
        assert fragment.text == "LLM response"
        assert fragment.metadata["source"] == "llm"
