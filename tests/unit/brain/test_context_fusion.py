"""Unit tests for ContextFusionEngine — token-budgeted context assembly."""

import pytest

from hbllm.brain.reasoning.context_fusion import (
    ContextFusionEngine,
    ContextSlice,
    FusedContext,
)


class TestContextSlice:
    def test_defaults(self):
        s = ContextSlice(source="memory", content="some text")
        assert s.source == "memory"
        assert s.priority == 0.5
        assert s.token_estimate > 0
        assert s.timestamp > 0

    def test_token_estimate(self):
        s = ContextSlice(source="test", content="a" * 400)
        assert s.token_estimate == 100  # 400 / 4

    def test_explicit_token_estimate(self):
        s = ContextSlice(source="test", content="short", token_estimate=50)
        assert s.token_estimate == 50


class TestFusedContext:
    def test_empty(self):
        ctx = FusedContext(sections=[], total_tokens=0, budget_used_pct=0, assembly_time_ms=1.0)
        assert ctx.to_system_prompt() == ""
        d = ctx.to_dict()
        assert d["total_tokens"] == 0

    def test_with_sections(self):
        sections = [
            ContextSlice(source="episodic_memory", content="Recent events here"),
            ContextSlice(source="world_state", content="Temperature: 22C"),
        ]
        ctx = FusedContext(
            sections=sections,
            total_tokens=20,
            budget_used_pct=50.0,
            assembly_time_ms=5.0,
        )
        prompt = ctx.to_system_prompt()
        assert "Recent Context" in prompt  # section header
        assert "Current Environment" in prompt  # section header
        assert "Recent events here" in prompt
        assert "22C" in prompt

    def test_to_dict(self):
        sections = [
            ContextSlice(source="memory", content="data", priority=0.9),
        ]
        ctx = FusedContext(
            sections=sections,
            total_tokens=10,
            budget_used_pct=25.0,
            assembly_time_ms=2.5,
        )
        d = ctx.to_dict()
        assert d["total_tokens"] == 10
        assert d["budget_used_pct"] == 25.0
        assert len(d["sections"]) == 1
        assert d["sections"][0]["source"] == "memory"


class TestContextFusionEngine:
    def test_init(self):
        engine = ContextFusionEngine(token_budget=2000)
        assert engine.token_budget == 2000

    def test_register_source(self):
        engine = ContextFusionEngine()

        async def dummy(q, t, b):
            return "context"

        engine.register_source("test_source", dummy, priority=0.8)
        assert "test_source" in engine._sources

    @pytest.mark.asyncio
    async def test_fuse_empty(self):
        engine = ContextFusionEngine()
        result = await engine.fuse(query="test")
        assert result.total_tokens == 0
        assert len(result.sections) == 0

    @pytest.mark.asyncio
    async def test_fuse_with_provider(self):
        engine = ContextFusionEngine(token_budget=1000)

        async def provider(query, tenant_id, budget):
            return f"Context for: {query}"

        engine.register_source("test", provider, priority=0.8)
        result = await engine.fuse(query="hello")

        assert len(result.sections) == 1
        assert "Context for: hello" in result.sections[0].content

    @pytest.mark.asyncio
    async def test_fuse_priority_ordering(self):
        engine = ContextFusionEngine(token_budget=1000)

        async def high_priority(q, t, b):
            return "high priority content"

        async def low_priority(q, t, b):
            return "low priority content"

        engine.register_source("high", high_priority, priority=0.9)
        engine.register_source("low", low_priority, priority=0.1)

        result = await engine.fuse(query="test")
        assert len(result.sections) == 2
        # High priority should come first
        assert result.sections[0].source == "high"

    @pytest.mark.asyncio
    async def test_fuse_budget_limit(self):
        engine = ContextFusionEngine(token_budget=10)  # Very tight budget

        async def big_content(q, t, b):
            return "x" * 200  # 50 tokens, exceeds budget

        engine.register_source("big", big_content, priority=0.8)
        result = await engine.fuse(query="test")

        # Should truncate to fit budget
        assert result.total_tokens <= 10 + 1  # Allow small rounding

    @pytest.mark.asyncio
    async def test_fuse_extra_context(self):
        engine = ContextFusionEngine()
        result = await engine.fuse(
            query="test",
            extra_context={"calendar": "Meeting at 3pm"},
        )
        assert len(result.sections) == 1
        assert result.sections[0].source == "calendar"

    @pytest.mark.asyncio
    async def test_fuse_failing_provider(self):
        engine = ContextFusionEngine()

        async def failing_provider(q, t, b):
            raise RuntimeError("Provider failed")

        engine.register_source("broken", failing_provider)
        result = await engine.fuse(query="test")

        # Should not crash, just skip the failed source
        assert result.total_tokens == 0

    @pytest.mark.asyncio
    async def test_fuse_concurrent_sources(self):
        engine = ContextFusionEngine()

        async def source_a(q, t, b):
            return "Data A"

        async def source_b(q, t, b):
            return "Data B"

        async def source_c(q, t, b):
            return "Data C"

        engine.register_source("a", source_a, priority=0.8)
        engine.register_source("b", source_b, priority=0.6)
        engine.register_source("c", source_c, priority=0.4)

        result = await engine.fuse(query="test")
        assert len(result.sections) == 3
        assert result.assembly_time_ms > 0


class TestWorldStateProvider:
    def test_factory_creates_callable(self):
        provider = ContextFusionEngine.world_state_provider(None)
        assert callable(provider)

    @pytest.mark.asyncio
    async def test_empty_world_state(self):
        class MockWS:
            _graph = {}

        provider = ContextFusionEngine.world_state_provider(MockWS())
        result = await provider("query", "tenant", 100)
        assert result == ""


class TestEmotionProvider:
    @pytest.mark.asyncio
    async def test_no_engine(self):
        provider = ContextFusionEngine.emotion_provider(None)
        result = await provider("query", "tenant", 100)
        assert result == ""

    @pytest.mark.asyncio
    async def test_with_state(self):
        class MockEmotionEngine:
            def get_state(self, tenant_id):
                return {"dominant_emotion": "joy", "valence": 0.8, "arousal": 0.5}

        provider = ContextFusionEngine.emotion_provider(MockEmotionEngine())
        result = await provider("query", "tenant", 100)
        assert "joy" in result
