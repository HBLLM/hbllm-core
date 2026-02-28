"""Tests for CognitivePipeline and ContextWindowManager."""

import asyncio
import pytest

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.network.registry import ServiceRegistry
from hbllm.serving.pipeline import CognitivePipeline, PipelineConfig, PipelineResult
from hbllm.brain.context_window import (
    ContextWindowManager,
    ContextBlock,
    ContextResult,
    estimate_tokens,
)


# ─── ContextWindowManager Tests ──────────────────────────────────────────────

class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_short(self):
        assert estimate_tokens("hi") >= 1

    def test_longer(self):
        # 100 chars ≈ 25 tokens
        text = "a" * 100
        tokens = estimate_tokens(text)
        assert 20 <= tokens <= 30


class TestContextBlock:
    def test_auto_estimate(self):
        block = ContextBlock(content="Hello world", priority=1)
        assert block.token_estimate > 0

    def test_explicit_estimate(self):
        block = ContextBlock(content="Hello", priority=1, token_estimate=50)
        assert block.token_estimate == 50


class TestContextWindowManager:
    def test_add_and_build(self):
        cwm = ContextWindowManager(max_tokens=1000)
        cwm.add("system", "You are helpful.", priority=1)
        cwm.add("user", "What is Python?", priority=2)
        result = cwm.build()
        assert "helpful" in result.text
        assert "Python" in result.text
        assert result.used_tokens > 0

    def test_priority_ordering(self):
        cwm = ContextWindowManager(max_tokens=100, reserve_for_output=10)
        cwm.add("low", "x" * 200, priority=5)  # ~50 tokens, won't all fit
        cwm.add("high", "IMPORTANT", priority=1)  # ~2 tokens, fits first
        result = cwm.build()
        assert "IMPORTANT" in result.text

    def test_truncation(self):
        cwm = ContextWindowManager(max_tokens=50, reserve_for_output=10)
        # Budget = 40 tokens = 160 chars
        cwm.add("big", "a" * 500, priority=1)  # 125 tokens, must truncate
        result = cwm.build()
        assert result.is_truncated
        assert len(result.blocks_truncated) > 0

    def test_empty_content_skipped(self):
        cwm = ContextWindowManager(max_tokens=500)
        cwm.add("empty", "", priority=1)
        cwm.add("spaces", "   ", priority=1)
        cwm.add("real", "real content", priority=1)
        result = cwm.build()
        assert result.blocks_included == 1

    def test_add_messages(self):
        cwm = ContextWindowManager(max_tokens=500)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]
        cwm.add_messages(messages)
        result = cwm.build()
        assert "Hello" in result.text
        assert "Hi there" in result.text
        assert result.blocks_included == 3

    def test_clear(self):
        cwm = ContextWindowManager(max_tokens=100)
        cwm.add("test", "content", priority=1)
        cwm.clear()
        result = cwm.build()
        assert result.blocks_included == 0

    def test_stats(self):
        cwm = ContextWindowManager(max_tokens=1000, reserve_for_output=200)
        cwm.add("a", "hello world", priority=1)
        stats = cwm.stats()
        assert stats["blocks"] == 1
        assert stats["max_tokens"] == 1000
        assert stats["available_tokens"] == 800
        assert stats["would_fit"] is True

    def test_summary(self):
        cwm = ContextWindowManager(max_tokens=100)
        cwm.add("test", "content", priority=1)
        result = cwm.build()
        summary = result.summary()
        assert "tokens" in summary
        assert "%" in summary


# ─── PipelineResult Tests ────────────────────────────────────────────────────

class TestPipelineResult:
    def test_to_dict(self):
        r = PipelineResult(
            text="Hello",
            correlation_id="abc",
            latency_ms=42.0,
            stages_completed=["pre_process", "route"],
        )
        d = r.to_dict()
        assert d["text"] == "Hello"
        assert d["correlation_id"] == "abc"
        assert d["latency_ms"] == 42.0
        assert d["error"] is False

    def test_error_result(self):
        r = PipelineResult(text="Error", correlation_id="x", error=True)
        assert r.error is True


# ─── CognitivePipeline Tests ─────────────────────────────────────────────────

class TestCognitivePipeline:
    @pytest.fixture
    async def pipeline(self):
        bus = InProcessBus()
        await bus.start()
        registry = ServiceRegistry()
        await registry.start()

        config = PipelineConfig(
            total_timeout=5.0,
            inject_memory=False,
            inject_identity=False,
            inject_curiosity=False,
        )
        pipe = CognitivePipeline(bus=bus, registry=registry, config=config)
        await pipe.start()
        yield pipe, bus
        await pipe.stop()
        await bus.stop()

    async def test_start_stop(self, pipeline):
        pipe, _ = pipeline
        assert pipe._subscription is not None

    async def test_timeout_returns_error(self, pipeline):
        pipe, _ = pipeline
        # No nodes registered, so the query will time out
        result = await pipe.process("test query", tenant_id="t1")
        assert result.error is True
        assert "pre_process" in result.stages_completed
        assert result.latency_ms > 0

    async def test_health(self, pipeline):
        pipe, _ = pipeline
        h = await pipe.health()
        assert h["status"] == "healthy"
        assert h["bus_type"] == "InProcessBus"

    async def test_decision_output_resolves(self, pipeline):
        """Test that a decision.output message resolves a pending pipeline."""
        pipe, bus = pipeline

        # Simulate: router receives and immediately emits decision.output
        async def mock_router(msg: Message) -> Message | None:
            # Simulate the full pipeline completing by publishing decision output
            decision_msg = Message(
                type=MessageType.RESPONSE,
                source_node_id="decision_01",
                topic="decision.output",
                tenant_id=msg.tenant_id,
                correlation_id=msg.id,
                payload={"text": "I know the answer!", "confidence": 0.9},
            )
            await bus.publish("decision.output", decision_msg)
            return None

        await bus.subscribe("router.query", mock_router)

        result = await pipe.process("What is 2+2?", tenant_id="t1")
        assert not result.error
        assert result.text == "I know the answer!"
        assert result.confidence == 0.9
        assert "route" in result.stages_completed

    async def test_pre_process_graceful_timeout(self):
        """Pre-processing gracefully handles missing nodes."""
        bus = InProcessBus()
        await bus.start()
        config = PipelineConfig(
            inject_memory=True,
            inject_identity=True,
            inject_curiosity=True,
            total_timeout=3.0,
        )
        pipe = CognitivePipeline(bus=bus, config=config)
        await pipe.start()

        # Pre-process should not crash even with no handlers
        context = await pipe._pre_process("test", "t1", "s1", "corr1")
        assert context["memory"] == []
        assert context["identity"] == {}
        assert context["curiosity_goals"] == []

        await pipe.stop()
        await bus.stop()
