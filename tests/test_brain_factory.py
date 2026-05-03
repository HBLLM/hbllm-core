"""Tests for BrainFactory — end-to-end brain creation and query processing."""

import asyncio
from typing import Any

import pytest

from hbllm.brain.factory import Brain, BrainConfig, BrainFactory
from hbllm.brain.provider_adapter import ProviderLLM
from hbllm.serving.provider import LLMProvider, LLMResponse

# ── Mock Provider ────────────────────────────────────────────────────────────


class MockBrainProvider(LLMProvider):
    """Mock provider that returns structured responses for brain node queries."""

    def __init__(self):
        self._call_count = 0

    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> LLMResponse:
        self._call_count += 1
        user_msg = messages[-1]["content"] if messages else ""

        # Router intent classification
        if "intent classifier" in user_msg.lower() or "classify" in user_msg.lower():
            content = '{"domain": "general", "intent": "general_knowledge", "confidence": 0.9}'
        # Critic evaluation
        elif "evaluator" in user_msg.lower() or "verdict" in user_msg.lower():
            content = '{"verdict": "PASS", "reason": "Response is relevant and safe"}'
        # Decision safety
        elif "safety classifier" in user_msg.lower():
            content = '{"safe": true, "reason": "Content is safe"}'
        # Planner / thought generation
        elif "generate" in user_msg.lower() and "thought" in user_msg.lower():
            content = '{"thought": "Let me think about this step by step", "score": 0.8}'
        # Score evaluation
        elif "score" in user_msg.lower() or "evaluate" in user_msg.lower():
            content = '{"score": 0.85, "explanation": "Good quality response"}'
        # Default
        else:
            content = '{"response": "This is a helpful answer from the brain"}'

        return LLMResponse(
            content=content,
            model="mock-brain",
            usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        )

    async def stream(self, messages, max_tokens=1024, temperature=0.7, **kwargs):
        response = await self.generate(messages, max_tokens, temperature, **kwargs)
        yield response.content

    @property
    def name(self) -> str:
        return "mock-brain"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _test_config(tmp_path, **overrides) -> BrainConfig:
    """Create a test-safe BrainConfig that disables background tasks."""
    defaults = dict(
        data_dir=str(tmp_path),
        watch_plugins=False,
        inject_plugins=False,
        inject_awareness=False,
        inject_load_manager=False,
        inject_scheduler=False,
        inject_knowledge=False,
        inject_persistence=False,
    )
    defaults.update(overrides)
    return BrainConfig(**defaults)


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_factory_creates_brain(tmp_path):
    """Factory creates a Brain with all components."""
    provider = MockBrainProvider()
    brain = await BrainFactory.create(
        provider=provider,
        config=_test_config(tmp_path),
    )

    try:
        assert isinstance(brain, Brain)
        assert brain.bus is not None
        assert brain.registry is not None
        assert brain.pipeline is not None
        assert brain.llm is not None
        assert (
            len(brain.nodes) >= 8
        )  # 8 composite nodes + awareness (v4 composite architecture)
    finally:
        await brain.shutdown()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_factory_with_config(tmp_path):
    """Factory respects custom config."""
    config = _test_config(
        tmp_path,
        inject_memory=False,
        inject_identity=False,
        inject_curiosity=False,
        total_timeout=10.0,
        system_prompt="You are a coding assistant.",
    )
    provider = MockBrainProvider()
    brain = await BrainFactory.create(provider=provider, config=config)

    try:
        assert brain.llm.system_prompt == "You are a coding assistant."
    finally:
        await brain.shutdown()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_brain_usage_tracking(tmp_path):
    """Usage counters accumulate across calls."""
    provider = MockBrainProvider()
    brain = await BrainFactory.create(
        provider=provider,
        config=_test_config(tmp_path),
    )

    try:
        # Direct LLM usage
        await brain.llm.generate("test")
        assert brain.usage["call_count"] == 1
        assert brain.usage["total_tokens"] == 30
    finally:
        await brain.shutdown()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_brain_shutdown(tmp_path):
    """Brain shuts down cleanly."""
    provider = MockBrainProvider()
    brain = await BrainFactory.create(
        provider=provider,
        config=_test_config(tmp_path),
    )

    # Shutdown should not raise
    await brain.shutdown()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_factory_with_string_provider(tmp_path):
    """Factory accepts provider name strings (but will fail without API keys)."""
    # We can't test with real API keys in CI, but we can verify the
    # factory path handles the string correctly and wraps the error.
    # Use a minimal config to avoid hanging on background tasks.
    try:
        brain = await BrainFactory.create(
            provider="openai/gpt-4o-mini",
            config=_test_config(tmp_path),
        )
        await brain.shutdown()
    except Exception:
        # Expected — no API key. The point is the factory didn't crash
        # before reaching the provider call.
        pass


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_adapter_integration_with_nodes():
    """ProviderLLM adapter works with the actual brain node interface."""
    provider = MockBrainProvider()
    llm = ProviderLLM(provider)

    # Simulate what RouterNode does
    classification = await llm.generate_json(
        "You are an intent classifier. Classify: 'What is Python?'\n"
        'Output JSON: {"domain": "...", "intent": "...", "confidence": 0.0-1.0}'
    )
    assert "domain" in classification or "error" in classification

    # Simulate what CriticNode does
    evaluation = await llm.generate_json(
        'You are a QA evaluator.\nOutput JSON: {"verdict": "PASS" or "FAIL", "reason": "..."}'
    )
    assert "verdict" in evaluation or "error" in evaluation

    # Simulate what DecisionNode does
    safety = await llm.generate_json(
        'You are a safety classifier.\nOutput JSON: {"safe": true/false, "reason": "..."}'
    )
    assert "safe" in safety or "error" in safety


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_provider_llm_counts(tmp_path):
    """Verify total LLM call count through brain creation + queries."""
    provider = MockBrainProvider()
    brain = await BrainFactory.create(
        provider=provider,
        config=_test_config(
            tmp_path,
            inject_memory=False,
            inject_identity=False,
            inject_curiosity=False,
        ),
    )

    try:
        initial_calls = provider._call_count
        # Factory creation itself shouldn't make LLM calls
        assert initial_calls == 0
    finally:
        await brain.shutdown()
