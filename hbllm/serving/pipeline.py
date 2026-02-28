"""
Cognitive Pipeline — the single entry point for all HBLLM inference.

Orchestrates the full flow:
  Router → Workspace → Domain Experts → Critic → Decision → Response

Used by FastAPI, MCP server, and CLI. Injects memory context, identity
persona, and curiosity goals before routing. Timeout/fallback at each stage.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from hbllm.network.bus import MessageBus, InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.network.registry import ServiceRegistry

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a cognitive pipeline execution."""
    text: str
    correlation_id: str
    source_node: str = "decision"
    confidence: float = 0.0
    tenant_id: str = "default"
    session_id: str = "default"
    latency_ms: float = 0.0
    stages_completed: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "correlation_id": self.correlation_id,
            "source_node": self.source_node,
            "confidence": self.confidence,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "latency_ms": self.latency_ms,
            "stages_completed": self.stages_completed,
            "error": self.error,
        }


@dataclass
class PipelineConfig:
    """Configuration for the cognitive pipeline."""
    router_timeout: float = 15.0
    workspace_timeout: float = 30.0
    decision_timeout: float = 30.0
    total_timeout: float = 60.0
    max_context_tokens: int = 2048
    inject_memory: bool = True
    inject_identity: bool = True
    inject_curiosity: bool = True


class CognitivePipeline:
    """
    Orchestrates the full HBLLM cognitive inference flow.

    This is the single, canonical entry point for all inference—used by
    the FastAPI server, MCP server, CLI, and integration tests.

    Flow:
    1. Pre-processing: inject memory context, identity, curiosity goals
    2. Routing: send to RouterNode for intent classification
    3. Workspace: blackboard aggregates thoughts from domain experts
    4. Critic: evaluates and scores proposals
    5. Decision: selects best approach and generates final response
    6. Post-processing: store in memory, update value signals

    Each stage has independent timeouts and graceful fallbacks.
    """

    def __init__(
        self,
        bus: MessageBus,
        registry: ServiceRegistry | None = None,
        config: PipelineConfig | None = None,
    ):
        self.bus = bus
        self.registry = registry
        self.config = config or PipelineConfig()
        self._response_futures: dict[str, asyncio.Future[Message]] = {}
        self._subscription = None

    async def start(self) -> None:
        """Subscribe to decision output to capture final responses."""
        self._subscription = await self.bus.subscribe(
            "decision.output", self._handle_decision_output
        )
        logger.info("CognitivePipeline started")

    async def stop(self) -> None:
        """Clean up subscriptions and pending futures."""
        if self._subscription:
            await self.bus.unsubscribe(self._subscription)

        for future in self._response_futures.values():
            if not future.done():
                future.cancel()
        self._response_futures.clear()
        logger.info("CognitivePipeline stopped")

    async def process(
        self,
        text: str,
        tenant_id: str = "default",
        session_id: str = "default",
        model_size: str = "125M",
    ) -> PipelineResult:
        """
        Process a user query through the full cognitive pipeline.

        Args:
            text: The user's query text
            tenant_id: Tenant for multi-tenant isolation
            session_id: Session for conversation continuity
            model_size: Model size hint

        Returns:
            PipelineResult with the final response text, latency, and metadata.
        """
        start_time = time.monotonic()
        correlation_id = str(uuid.uuid4())
        stages: list[str] = []

        try:
            # ── Stage 1: Pre-processing (memory + identity injection) ──
            context = await self._pre_process(
                text, tenant_id, session_id, correlation_id
            )
            stages.append("pre_process")

            # ── Stage 2: Route through cognitive pipeline ──
            response = await self._route_and_wait(
                text=text,
                context=context,
                tenant_id=tenant_id,
                session_id=session_id,
                correlation_id=correlation_id,
                model_size=model_size,
            )
            stages.append("route")
            stages.append("workspace")
            stages.append("decision")

            # ── Stage 3: Post-processing (memory storage) ──
            await self._post_process(
                text, response, tenant_id, session_id, correlation_id
            )
            stages.append("post_process")

            latency_ms = (time.monotonic() - start_time) * 1000

            return PipelineResult(
                text=response.get("text", response.get("response", "")),
                correlation_id=correlation_id,
                source_node=response.get("source_node", "decision"),
                confidence=float(response.get("confidence", 0.0)),
                tenant_id=tenant_id,
                session_id=session_id,
                latency_ms=latency_ms,
                stages_completed=stages,
                metadata=response,
            )

        except asyncio.TimeoutError:
            latency_ms = (time.monotonic() - start_time) * 1000
            logger.warning(
                "Pipeline timed out after %.0fms for '%s...'",
                latency_ms, text[:30],
            )
            return PipelineResult(
                text="I'm taking longer than expected to think about this. Please try again.",
                correlation_id=correlation_id,
                tenant_id=tenant_id,
                session_id=session_id,
                latency_ms=latency_ms,
                stages_completed=stages,
                error=True,
            )
        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            logger.exception("Pipeline error: %s", e)
            return PipelineResult(
                text=f"An error occurred while processing your request: {e}",
                correlation_id=correlation_id,
                tenant_id=tenant_id,
                session_id=session_id,
                latency_ms=latency_ms,
                stages_completed=stages,
                error=True,
            )

    async def _pre_process(
        self,
        text: str,
        tenant_id: str,
        session_id: str,
        correlation_id: str,
    ) -> dict[str, Any]:
        """
        Gather context from memory, identity, and curiosity before routing.
        Each sub-query has its own timeout and fails gracefully.
        """
        context: dict[str, Any] = {}

        # Memory retrieval
        if self.config.inject_memory:
            try:
                mem_msg = Message(
                    type=MessageType.QUERY,
                    source_node_id="pipeline",
                    topic="memory.search",
                    tenant_id=tenant_id,
                    session_id=session_id,
                    payload={"query": text, "limit": 5},
                )
                mem_resp = await asyncio.wait_for(
                    self.bus.request("memory.search", mem_msg, timeout=5.0),
                    timeout=5.0,
                )
                context["memory"] = mem_resp.payload.get("results", [])
            except (TimeoutError, asyncio.TimeoutError):
                logger.debug("Memory retrieval timed out, continuing without")
                context["memory"] = []
            except Exception:
                context["memory"] = []

        # Identity retrieval
        if self.config.inject_identity:
            try:
                id_msg = Message(
                    type=MessageType.QUERY,
                    source_node_id="pipeline",
                    topic="identity.query",
                    tenant_id=tenant_id,
                    payload={},
                )
                id_resp = await asyncio.wait_for(
                    self.bus.request("identity.query", id_msg, timeout=3.0),
                    timeout=3.0,
                )
                context["identity"] = id_resp.payload
            except (TimeoutError, asyncio.TimeoutError):
                context["identity"] = {}
            except Exception:
                context["identity"] = {}

        # Curiosity goals
        if self.config.inject_curiosity:
            try:
                cur_msg = Message(
                    type=MessageType.QUERY,
                    source_node_id="pipeline",
                    topic="curiosity.goals",
                    tenant_id=tenant_id,
                    payload={},
                )
                cur_resp = await asyncio.wait_for(
                    self.bus.request("curiosity.goals", cur_msg, timeout=3.0),
                    timeout=3.0,
                )
                context["curiosity_goals"] = cur_resp.payload.get("goals", [])
            except (TimeoutError, asyncio.TimeoutError):
                context["curiosity_goals"] = []
            except Exception:
                context["curiosity_goals"] = []

        return context

    async def _route_and_wait(
        self,
        text: str,
        context: dict[str, Any],
        tenant_id: str,
        session_id: str,
        correlation_id: str,
        model_size: str,
    ) -> dict[str, Any]:
        """
        Send query to the router and wait for the decision output.
        """
        # Create a future for the final decision output
        future: asyncio.Future[Message] = asyncio.get_event_loop().create_future()
        self._response_futures[correlation_id] = future

        # Build the router message with enriched context
        query_msg = Message(
            id=correlation_id,
            type=MessageType.QUERY,
            source_node_id="pipeline",
            topic="router.query",
            tenant_id=tenant_id,
            session_id=session_id,
            payload={
                "text": text,
                "model_size": model_size,
                "context": context,
            },
        )

        # Publish to router
        await self.bus.publish("router.query", query_msg)

        # Wait for decision output with total timeout
        try:
            response = await asyncio.wait_for(
                future, timeout=self.config.total_timeout
            )
            return response.payload
        finally:
            self._response_futures.pop(correlation_id, None)

    async def _handle_decision_output(self, message: Message) -> None:
        """Capture decision output and resolve the corresponding future."""
        corr_id = message.correlation_id
        if corr_id and corr_id in self._response_futures:
            future = self._response_futures.get(corr_id)
            if future and not future.done():
                future.set_result(message)

    async def _post_process(
        self,
        query: str,
        response: dict[str, Any],
        tenant_id: str,
        session_id: str,
        correlation_id: str,
    ) -> None:
        """Store the interaction in memory (fire-and-forget)."""
        try:
            store_msg = Message(
                type=MessageType.EVENT,
                source_node_id="pipeline",
                topic="memory.store",
                tenant_id=tenant_id,
                session_id=session_id,
                payload={
                    "query": query,
                    "response": response.get("text", ""),
                    "correlation_id": correlation_id,
                },
            )
            await self.bus.publish("memory.store", store_msg)
        except Exception:
            logger.debug("Post-process memory store failed, non-critical")

    async def health(self) -> dict[str, Any]:
        """Pipeline health check."""
        node_count = 0
        if self.registry:
            all_nodes = await self.registry.discover(healthy_only=False)
            node_count = len(all_nodes)

        return {
            "status": "healthy",
            "bus_type": type(self.bus).__name__,
            "nodes": node_count,
            "config": {
                "total_timeout": self.config.total_timeout,
                "max_context_tokens": self.config.max_context_tokens,
            },
        }

    # ─── Multi-Modal Support ──────────────────────────────────────────────

    async def process_multimodal(
        self,
        text: str = "",
        images: list[bytes] | None = None,
        audio: bytes | None = None,
        tenant_id: str = "default",
        session_id: str = "default",
        model_size: str = "125M",
    ) -> PipelineResult:
        """
        Process a multi-modal query through the cognitive pipeline.

        Images are captioned via VisionNode, audio is transcribed via
        AudioInputNode. Results are injected as context before routing.

        Args:
            text: Optional text query
            images: Optional list of image bytes
            audio: Optional audio bytes
            tenant_id: Tenant for isolation
            session_id: Session continuity
            model_size: Model size hint

        Returns:
            PipelineResult with unified response.
        """
        context_parts: list[str] = []

        # Process images → captions
        if images:
            for i, img_bytes in enumerate(images):
                try:
                    caption_msg = Message(
                        type=MessageType.QUERY,
                        source_node_id="pipeline",
                        topic="vision.caption",
                        tenant_id=tenant_id,
                        payload={"image_data": img_bytes.hex(), "index": i},
                    )
                    resp = await asyncio.wait_for(
                        self.bus.request("vision.caption", caption_msg, timeout=10.0),
                        timeout=10.0,
                    )
                    caption = resp.payload.get("caption", "")
                    if caption:
                        context_parts.append(f"[Image {i+1}]: {caption}")
                except (TimeoutError, asyncio.TimeoutError):
                    context_parts.append(f"[Image {i+1}]: (could not process)")
                except Exception:
                    context_parts.append(f"[Image {i+1}]: (processing error)")

        # Process audio → transcript
        if audio:
            try:
                audio_msg = Message(
                    type=MessageType.QUERY,
                    source_node_id="pipeline",
                    topic="audio.transcribe",
                    tenant_id=tenant_id,
                    payload={"audio_data": audio.hex()},
                )
                resp = await asyncio.wait_for(
                    self.bus.request("audio.transcribe", audio_msg, timeout=15.0),
                    timeout=15.0,
                )
                transcript = resp.payload.get("transcript", "")
                if transcript:
                    context_parts.append(f"[Audio transcript]: {transcript}")
            except (TimeoutError, asyncio.TimeoutError):
                context_parts.append("[Audio]: (could not transcribe)")
            except Exception:
                context_parts.append("[Audio]: (transcription error)")

        # Combine text + multi-modal context
        combined = text
        if context_parts:
            modal_context = "\n".join(context_parts)
            combined = f"{modal_context}\n\n{text}" if text else modal_context

        return await self.process(
            text=combined,
            tenant_id=tenant_id,
            session_id=session_id,
            model_size=model_size,
        )

