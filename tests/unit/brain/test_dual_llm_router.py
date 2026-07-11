"""Tests for DualLLMRouter — dual model routing with auto-complexity detection."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from hbllm.brain.control.dual_llm_router import (
    DualLLMRouter,
    DualLLMStats,
    TaskTier,
    estimate_complexity,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_provider_llm(name: str = "test") -> MagicMock:
    """Create a mock ProviderLLM."""
    mock = MagicMock()
    mock.provider = MagicMock()
    mock.provider.name = name
    mock.generate = AsyncMock(return_value=f"response from {name}")
    mock.generate_json = AsyncMock(return_value={"answer": name})
    mock.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "call_count": 1}

    async def _stream(*args, **kwargs):
        for token in ["hello", " ", "world"]:
            yield token

    mock.generate_stream = _stream
    return mock


def _make_state_machine(allow_heavy: bool = True) -> MagicMock:
    """Create a mock CognitiveStateMachine."""
    sm = MagicMock()
    profile = MagicMock()
    profile.allow_heavy_llm = allow_heavy
    sm.current_profile = profile
    return sm


# ── Complexity Estimation ────────────────────────────────────────────────────


class TestComplexityEstimation:
    """Tests for the prompt complexity heuristic."""

    def test_simple_greeting(self):
        score = estimate_complexity("Hello, how are you?")
        assert score < 0.3

    def test_complex_reasoning(self):
        score = estimate_complexity(
            "Analyze the trade-off between microservices and monolithic architecture. "
            "Compare their deployment complexity, then evaluate the debugging story. "
            "Finally, design a migration strategy."
        )
        assert score >= 0.2  # Multiple complex keywords detected

    def test_code_generation(self):
        score = estimate_complexity("Write a function to sort a list using quicksort")
        assert score >= 0.1  # Code pattern detected

    def test_empty_prompt(self):
        score = estimate_complexity("")
        assert score == 0.0

    def test_long_simple_prompt(self):
        """Long prompts get a length bonus but simple words keep score low."""
        score = estimate_complexity("hello " * 60)
        # Length adds score but simple keywords suppress
        assert 0.0 <= score <= 1.0

    def test_multi_step_prompt(self):
        score = estimate_complexity(
            "First do this, then do that, after which step 3 is to finalize"
        )
        # Should detect multi-step indicators
        assert score >= 0.05

    def test_score_bounds(self):
        """Score should always be between 0.0 and 1.0."""
        for prompt in ["", "hi", "x" * 1000, "analyze debug implement optimize prove reason"]:
            score = estimate_complexity(prompt)
            assert 0.0 <= score <= 1.0


# ── Routing Decisions ────────────────────────────────────────────────────────


class TestRoutingDecisions:
    """Tests for explicit and auto routing."""

    def test_explicit_local(self):
        local = _make_provider_llm("local")
        external = _make_provider_llm("external")
        router = DualLLMRouter(local=local, external=external)

        decision = router.route("any prompt", tier=TaskTier.LOCAL)
        assert decision.tier == TaskTier.LOCAL
        assert decision.reason == "explicit_local"

    def test_explicit_external(self):
        local = _make_provider_llm("local")
        external = _make_provider_llm("external")
        router = DualLLMRouter(local=local, external=external)

        decision = router.route("any prompt", tier=TaskTier.EXTERNAL)
        assert decision.tier == TaskTier.EXTERNAL
        assert decision.reason == "explicit_external"

    def test_explicit_external_no_external_available(self):
        local = _make_provider_llm("local")
        router = DualLLMRouter(local=local, external=None)

        decision = router.route("any prompt", tier=TaskTier.EXTERNAL)
        assert decision.tier == TaskTier.LOCAL  # Fallback
        assert decision.fallback_used is True

    def test_auto_simple_goes_local(self):
        local = _make_provider_llm("local")
        external = _make_provider_llm("external")
        router = DualLLMRouter(local=local, external=external)

        decision = router.route("Hello there", tier=TaskTier.AUTO)
        assert decision.tier == TaskTier.LOCAL

    def test_auto_complex_goes_external(self):
        local = _make_provider_llm("local")
        external = _make_provider_llm("external")
        router = DualLLMRouter(local=local, external=external, complexity_threshold=0.2)

        decision = router.route(
            "Analyze and compare the trade-off of this design then implement the solution",
            tier=TaskTier.AUTO,
        )
        assert decision.tier == TaskTier.EXTERNAL
        assert decision.complexity_score >= 0.2

    def test_auto_no_external(self):
        local = _make_provider_llm("local")
        router = DualLLMRouter(local=local, external=None)

        decision = router.route(
            "Analyze and compare complex architecture",
            tier=TaskTier.AUTO,
        )
        assert decision.tier == TaskTier.LOCAL
        assert decision.reason == "no_external_configured"

    def test_state_machine_forbids_heavy(self):
        local = _make_provider_llm("local")
        external = _make_provider_llm("external")
        sm = _make_state_machine(allow_heavy=False)
        router = DualLLMRouter(local=local, external=external, state_machine=sm)

        decision = router.route(
            "Analyze complex trade-off and design architecture",
            tier=TaskTier.AUTO,
        )
        assert decision.tier == TaskTier.LOCAL
        assert "state_forbids" in decision.reason

    def test_state_machine_allows_heavy(self):
        local = _make_provider_llm("local")
        external = _make_provider_llm("external")
        sm = _make_state_machine(allow_heavy=True)
        router = DualLLMRouter(
            local=local, external=external, state_machine=sm, complexity_threshold=0.2
        )

        decision = router.route(
            "Analyze and debug the implementation trade-off",
            tier=TaskTier.AUTO,
        )
        assert decision.tier == TaskTier.EXTERNAL
        assert decision.state_allows_heavy is True


# ── Generation ───────────────────────────────────────────────────────────────


class TestGeneration:
    """Tests for actual generation routing."""

    @pytest.mark.asyncio
    async def test_generate_local(self):
        local = _make_provider_llm("local")
        external = _make_provider_llm("external")
        router = DualLLMRouter(local=local, external=external)

        result = await router.generate("hi", tier=TaskTier.LOCAL)
        assert result == "response from local"
        local.generate.assert_awaited_once()
        external.generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_generate_external(self):
        local = _make_provider_llm("local")
        external = _make_provider_llm("external")
        router = DualLLMRouter(local=local, external=external)

        result = await router.generate("complex task", tier=TaskTier.EXTERNAL)
        assert result == "response from external"
        external.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_external_fallback(self):
        """If external fails, should fallback to local."""
        local = _make_provider_llm("local")
        external = _make_provider_llm("external")
        external.generate = AsyncMock(side_effect=ConnectionError("API down"))
        router = DualLLMRouter(local=local, external=external)

        result = await router.generate("complex task", tier=TaskTier.EXTERNAL)
        assert result == "response from local"
        assert router.stats.fallbacks == 1

    @pytest.mark.asyncio
    async def test_generate_json_routes(self):
        local = _make_provider_llm("local")
        external = _make_provider_llm("external")
        router = DualLLMRouter(local=local, external=external)

        result = await router.generate_json("get json", tier=TaskTier.LOCAL)
        assert result == {"answer": "local"}

    @pytest.mark.asyncio
    async def test_generate_stream_routes(self):
        local = _make_provider_llm("local")
        router = DualLLMRouter(local=local, external=None)

        tokens = []
        async for token in router.generate_stream("hello", tier=TaskTier.LOCAL):
            tokens.append(token)
        assert tokens == ["hello", " ", "world"]


# ── Stats ────────────────────────────────────────────────────────────────────


class TestStats:
    """Tests for usage statistics tracking."""

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        local = _make_provider_llm("local")
        external = _make_provider_llm("external")
        router = DualLLMRouter(local=local, external=external)

        await router.generate("hi", tier=TaskTier.LOCAL)
        await router.generate("hi", tier=TaskTier.LOCAL)
        await router.generate("complex", tier=TaskTier.EXTERNAL)

        assert router.stats.local_calls == 2
        assert router.stats.external_calls == 1
        assert router.stats.total_calls == 3
        assert 0.5 < router.stats.local_ratio < 0.8

    @pytest.mark.asyncio
    async def test_stats_auto_tracking(self):
        local = _make_provider_llm("local")
        external = _make_provider_llm("external")
        router = DualLLMRouter(local=local, external=external)

        # Simple prompt → auto local
        await router.generate("hello", tier=TaskTier.AUTO)
        assert router.stats.auto_routed_local >= 1

    def test_stats_snapshot(self):
        stats = DualLLMStats(
            local_calls=10,
            external_calls=5,
            fallbacks=1,
        )
        d = stats.to_dict()
        assert d["local_calls"] == 10
        assert d["external_calls"] == 5
        assert d["fallbacks"] == 1
        assert d["local_ratio"] == round(10 / 15, 3)

    def test_usage_combined(self):
        local = _make_provider_llm("local")
        external = _make_provider_llm("external")
        router = DualLLMRouter(local=local, external=external)

        usage = router.usage
        assert usage["total_tokens"] == 30  # 15 + 15

    def test_snapshot(self):
        local = _make_provider_llm("local")
        router = DualLLMRouter(local=local, external=None)

        snap = router.snapshot()
        assert snap["external_provider"] == "none"
        assert snap["has_state_machine"] is False


# ── TaskTier Enum ────────────────────────────────────────────────────────────


class TestTaskTier:
    def test_values(self):
        assert TaskTier.LOCAL == "local"
        assert TaskTier.EXTERNAL == "external"
        assert TaskTier.AUTO == "auto"

    def test_from_string(self):
        assert TaskTier("local") == TaskTier.LOCAL
        assert TaskTier("auto") == TaskTier.AUTO
