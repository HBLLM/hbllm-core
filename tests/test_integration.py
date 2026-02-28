"""
Integration Tests — full pipeline with multiple nodes wired together.

Tests the actual message flow: Router → Workspace → Domain → Critic → Decision
with mock LLM to verify the cognitive pipeline works end-to-end.
"""

import asyncio
import pytest

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.network.registry import ServiceRegistry
from hbllm.serving.pipeline import CognitivePipeline, PipelineConfig, PipelineResult


class TestEndToEndPipeline:
    """Tests full pipeline flow with simulated node handlers."""

    @pytest.fixture
    async def full_pipeline(self):
        """Set up bus, pipeline, and simulated nodes."""
        bus = InProcessBus()
        await bus.start()
        registry = ServiceRegistry()
        await registry.start()

        config = PipelineConfig(
            total_timeout=10.0,
            inject_memory=False,
            inject_identity=False,
            inject_curiosity=False,
        )
        pipe = CognitivePipeline(bus=bus, registry=registry, config=config)
        await pipe.start()

        # ── Simulated Router ──
        async def mock_router(msg: Message) -> Message | None:
            """Classify and forward to workspace."""
            text = msg.payload.get("text", "")
            domain = "math" if any(w in text.lower() for w in ["calculate", "sum", "+", "math"]) else "general"

            workspace_msg = Message(
                type=MessageType.EVENT,
                source_node_id="router_sim",
                topic="workspace.update",
                tenant_id=msg.tenant_id,
                session_id=msg.session_id,
                correlation_id=msg.id,
                payload={
                    "text": text,
                    "intent": "math_reasoning" if domain == "math" else "general_knowledge",
                    "domain_hint": domain,
                },
            )
            await bus.publish("workspace.update", workspace_msg)
            return None

        # ── Simulated Workspace ──
        async def mock_workspace(msg: Message) -> Message | None:
            """Aggregate and forward to decision."""
            decision_msg = Message(
                type=MessageType.RESPONSE,
                source_node_id="workspace_sim",
                topic="decision.output",
                tenant_id=msg.tenant_id,
                session_id=msg.session_id,
                correlation_id=msg.correlation_id,
                payload={
                    "text": f"Processed: {msg.payload.get('text', '')}",
                    "confidence": 0.85,
                    "source_node": "workspace_sim",
                    "domain": msg.payload.get("domain_hint", "general"),
                },
            )
            await bus.publish("decision.output", decision_msg)
            return None

        await bus.subscribe("router.query", mock_router)
        await bus.subscribe("workspace.update", mock_workspace)

        yield pipe, bus

        await pipe.stop()
        await bus.stop()

    async def test_full_flow(self, full_pipeline):
        """Test a query flows through router → workspace → decision."""
        pipe, _ = full_pipeline
        result = await pipe.process("What is Python?", tenant_id="t1")

        assert not result.error
        assert "Processed" in result.text
        assert result.confidence == 0.85
        assert result.latency_ms > 0
        assert "route" in result.stages_completed

    async def test_math_routing(self, full_pipeline):
        """Test domain-aware routing for math queries."""
        pipe, _ = full_pipeline
        result = await pipe.process("Calculate the sum of 2 + 3", tenant_id="t1")

        assert not result.error
        assert "Processed" in result.text
        assert result.metadata.get("domain") == "math"

    async def test_multi_tenant_isolation(self, full_pipeline):
        """Test that concurrent queries from different tenants don't interfere."""
        pipe, _ = full_pipeline

        results = await asyncio.gather(
            pipe.process("Hello from tenant A", tenant_id="tenant_a"),
            pipe.process("Hello from tenant B", tenant_id="tenant_b"),
        )

        assert len(results) == 2
        assert results[0].tenant_id == "tenant_a"
        assert results[1].tenant_id == "tenant_b"
        assert not results[0].error
        assert not results[1].error
        # Each got their own response
        assert "tenant A" in results[0].text or "Processed" in results[0].text
        assert "tenant B" in results[1].text or "Processed" in results[1].text

    async def test_concurrent_queries(self, full_pipeline):
        """Test 5 concurrent queries all resolve correctly."""
        pipe, _ = full_pipeline

        queries = [f"Query number {i}" for i in range(5)]
        results = await asyncio.gather(
            *(pipe.process(q, tenant_id="t1") for q in queries)
        )

        assert len(results) == 5
        assert all(not r.error for r in results)
        assert all(r.latency_ms > 0 for r in results)


class TestPipelineWithContextInjection:
    """Test pre-processing context injection."""

    @pytest.fixture
    async def pipeline_with_context(self):
        """Pipeline with context injection enabled and mock handlers."""
        bus = InProcessBus()
        await bus.start()

        config = PipelineConfig(
            total_timeout=10.0,
            inject_memory=True,
            inject_identity=True,
            inject_curiosity=True,
        )
        pipe = CognitivePipeline(bus=bus, config=config)
        await pipe.start()

        # Mock memory search
        async def mock_memory(msg: Message) -> Message | None:
            return Message(
                type=MessageType.RESPONSE,
                source_node_id="memory_sim",
                topic="memory.search",
                correlation_id=msg.id,
                tenant_id=msg.tenant_id,
                payload={"results": [{"text": "User likes cats", "score": 0.9}]},
            )

        # Mock identity query
        async def mock_identity(msg: Message) -> Message | None:
            return Message(
                type=MessageType.RESPONSE,
                source_node_id="identity_sim",
                topic="identity.query",
                correlation_id=msg.id,
                tenant_id=msg.tenant_id,
                payload={"persona_name": "Assistant", "system_prompt": "Be helpful"},
            )

        # Mock router that captures context
        captured = {}

        async def mock_router(msg: Message) -> Message | None:
            captured["context"] = msg.payload.get("context", {})
            # Emit decision to complete pipeline
            decision = Message(
                type=MessageType.RESPONSE,
                source_node_id="decision_sim",
                topic="decision.output",
                tenant_id=msg.tenant_id,
                correlation_id=msg.id,
                payload={"text": "Response with context", "confidence": 0.9},
            )
            await bus.publish("decision.output", decision)
            return None

        await bus.subscribe("memory.search", mock_memory)
        await bus.subscribe("identity.query", mock_identity)
        await bus.subscribe("router.query", mock_router)

        yield pipe, bus, captured

        await pipe.stop()
        await bus.stop()

    async def test_context_injected(self, pipeline_with_context):
        """Test that memory and identity are injected into the query."""
        pipe, _, captured = pipeline_with_context
        result = await pipe.process("Tell me about cats", tenant_id="t1")

        assert not result.error
        context = captured.get("context", {})
        assert len(context.get("memory", [])) > 0
        assert context["memory"][0]["text"] == "User likes cats"
        assert context["identity"]["persona_name"] == "Assistant"


class TestPipelineFallback:
    """Test graceful degradation when nodes are unavailable."""

    async def test_no_nodes_returns_timeout(self):
        """Pipeline returns error when no nodes respond."""
        bus = InProcessBus()
        await bus.start()
        config = PipelineConfig(total_timeout=2.0, inject_memory=False, inject_identity=False, inject_curiosity=False)
        pipe = CognitivePipeline(bus=bus, config=config)
        await pipe.start()

        result = await pipe.process("test")
        assert result.error is True
        assert result.latency_ms > 0

        await pipe.stop()
        await bus.stop()

    async def test_partial_context_failure(self):
        """Pipeline works even when some context sources fail."""
        bus = InProcessBus()
        await bus.start()
        config = PipelineConfig(
            total_timeout=5.0,
            inject_memory=True,  # No handler → will timeout gracefully
            inject_identity=False,
            inject_curiosity=False,
        )
        pipe = CognitivePipeline(bus=bus, config=config)
        await pipe.start()

        # Mock router that completes immediately
        async def quick_router(msg: Message) -> Message | None:
            decision = Message(
                type=MessageType.RESPONSE,
                source_node_id="d",
                topic="decision.output",
                tenant_id=msg.tenant_id,
                correlation_id=msg.id,
                payload={"text": "Quick answer", "confidence": 0.5},
            )
            await bus.publish("decision.output", decision)
            return None

        await bus.subscribe("router.query", quick_router)

        result = await pipe.process("test")
        assert not result.error
        assert result.text == "Quick answer"

        await pipe.stop()
        await bus.stop()
