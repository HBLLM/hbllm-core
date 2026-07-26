"""
Generation Node — translation boundary between cognitive OS and Execution OS.

This is the ONLY place where cognitive outputs (RoutingResult) are translated
into execution requests (ExecutionRequest). The cognitive layer never sees
execution details (providers, models, modifiers). The execution layer never
sees cognitive metadata (domain, style, persona).

Data flow:
    RouterNode → RoutingResult (cognitive)
        ↓
    GenerationNode (translation boundary)
        ↓
    ExecutionRequest → ExecutionOrchestrator → ExecutionBus → Runtime → Result

The GenerationNode translates:
    - domain → constraints (e.g. "medical" → high accuracy, JSON format)
    - complexity → max_tokens
    - language → payload (system prompt with language instructions)
    - style → prompt modifier hint (NOT cognitive metadata in ExecutionRequest)
    - audience → constraints (safety level, formality)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.network.messages import Message, MessageType, QueryPayload
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


@dataclass
class RoutingResult:
    """
    Pure cognitive output from the RouterNode.

    Contains ONLY cognitive assessments — no execution knowledge.
    The GenerationNode translates this into execution constraints.
    """

    domain: str = "general"
    domain_weights: dict[str, float] = field(default_factory=dict)
    language: str = "en"
    complexity: str = "medium"  # "low", "medium", "high"
    audience: str = "general"  # "general", "technical", "academic"
    safety_level: str = "standard"  # "standard", "strict", "minimal"
    intent: str = "general_knowledge"
    confidence: float = 0.8
    requires_planning: bool = False
    requires_tools: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Complexity → Execution Constraints ────────────────────────────────────────

_COMPLEXITY_TOKENS: dict[str, int] = {
    "low": 256,
    "medium": 1024,
    "high": 4096,
}

_COMPLEXITY_TEMP: dict[str, float] = {
    "low": 0.7,
    "medium": 0.7,
    "high": 0.5,
}


class GenerationNode(Node):
    """
    Translation boundary: cognitive → execution.

    Receives cognitive outputs (RoutingResult, QueryPayload) from the
    RouterNode and translates them into ExecutionRequests that are sent
    to the Execution Orchestrator.

    This node knows both vocabularies but keeps them strictly separated:
    - Reads cognitive: RoutingResult, domain, style, language, complexity
    - Writes execution: ExecutionRequest, constraints, payload (no cognitive metadata)

    The GenerationNode does NOT:
    - Know about LoRA adapters
    - Know about providers
    - Know about model architectures
    - Make execution decisions

    Those are the Orchestrator's responsibilities.
    """

    def __init__(
        self,
        node_id: str = "generation_node",
        orchestrator: Any = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.DOMAIN_MODULE,
            capabilities=["generation", "text"],
        )
        self._orchestrator = orchestrator
        self._requests_handled = 0
        self._total_latency_ms = 0.0

    async def on_start(self) -> None:
        """Subscribe to generation requests from the cognitive layer."""
        logger.info("Starting GenerationNode '%s'", self.node_id)
        await self.bus.subscribe("module.evaluate", self.handle_message)

    async def on_stop(self) -> None:
        logger.info(
            "Stopping GenerationNode '%s' (requests=%d, avg_latency=%.1fms)",
            self.node_id,
            self._requests_handled,
            self._total_latency_ms / max(self._requests_handled, 1),
        )

    async def handle_message(self, message: Message) -> Message | None:
        """
        Translate cognitive outputs into execution requests.

        This is the ONLY translation boundary. Cognitive metadata
        is consumed here and translated into execution constraints
        and payload — it never appears in the ExecutionRequest.
        """
        if message.topic != "module.evaluate":
            return None

        try:
            payload = QueryPayload(**message.payload)
        except (TypeError, ValueError, KeyError) as e:
            return message.create_error(f"Invalid QueryPayload: {e}")

        # Extract cognitive outputs
        payload_metadata = getattr(payload, "metadata", {}) or {}
        prompt = payload_metadata.get("augmented_prompt") or getattr(payload, "text", "")
        domain_hint = getattr(payload, "domain_hint", "general")

        # Build RoutingResult from cognitive layer outputs
        routing_result = self._build_routing_result(domain_hint, payload_metadata)

        # Translate to execution request and execute
        start_time = time.monotonic()
        try:
            response_text = await self._execute(prompt, routing_result, message)
        except Exception as e:
            logger.error("GenerationNode execution failed: %s", e)
            return message.create_error(f"Generation failed: {e}")

        latency_ms = (time.monotonic() - start_time) * 1000
        self._requests_handled += 1
        self._total_latency_ms += latency_ms

        # Publish result as workspace thought
        thought_msg = Message(
            type=MessageType.EVENT,
            source_node_id=self.node_id,
            tenant_id=message.tenant_id,
            session_id=message.session_id,
            topic="workspace.thought",
            payload={
                "type": f"intuition_{routing_result.domain}",
                "confidence": routing_result.confidence,
                "content": response_text,
                "metrics": {
                    "latency_ms": int(latency_ms),
                    "domain": routing_result.domain,
                },
            },
            correlation_id=message.correlation_id,
        )
        await self.bus.publish("workspace.thought", thought_msg)
        return None

    async def _execute(
        self,
        prompt: str,
        routing_result: RoutingResult,
        message: Message,
    ) -> str:
        """
        Translate cognitive outputs into ExecutionRequest and dispatch.

        This is where the vocabulary translation happens:
            complexity → max_tokens
            language → system prompt instructions
            domain → constraints
            audience → safety constraints
        """
        if self._orchestrator is None:
            raise RuntimeError("GenerationNode has no orchestrator configured")

        from hbllm.execution.payload import ExecutionPayload
        from hbllm.execution.plan import ExecutionConstraints, ExecutionRequest, TaskType

        # ── Translate cognitive → execution ──────────────────────────────

        # 1. Complexity → token limit
        max_tokens = _COMPLEXITY_TOKENS.get(routing_result.complexity, 1024)

        # 2. Build constraints (from cognitive assessment, not execution knowledge)
        constraints = ExecutionConstraints(
            max_tokens=max_tokens,
            require_streaming=False,
        )

        # 3. Build payload (cognitive content rendered into messages)
        system_parts: list[str] = []
        if routing_result.language != "en":
            system_parts.append(f"Respond in {routing_result.language}.")

        system_prompt = " ".join(system_parts) if system_parts else None
        execution_payload = ExecutionPayload.from_prompt(prompt, system=system_prompt)

        # 4. Build the request — zero cognitive metadata
        request = ExecutionRequest(
            task_type=TaskType.TEXT_GENERATION,
            payload=execution_payload,
            constraints=constraints,
            tenant_id=message.tenant_id,
            session_id=message.session_id,
        )

        # 5. Dispatch to Execution OS
        result = await self._orchestrator.execute(request)
        return result.content

    def _build_routing_result(
        self,
        domain_hint: str | dict[str, float],
        metadata: dict[str, Any],
    ) -> RoutingResult:
        """Build a RoutingResult from cognitive layer outputs."""
        if isinstance(domain_hint, dict):
            # Weighted routing — pick the top domain
            domain = max(domain_hint, key=domain_hint.get) if domain_hint else "general"
            return RoutingResult(
                domain=domain,
                domain_weights=domain_hint,
                language=metadata.get("language", "en"),
                complexity=metadata.get("complexity", "medium"),
                audience=metadata.get("audience", "general"),
                intent=metadata.get("intent", "general_knowledge"),
                confidence=metadata.get("confidence", 0.8),
            )
        return RoutingResult(
            domain=domain_hint if isinstance(domain_hint, str) else "general",
            language=metadata.get("language", "en"),
            complexity=metadata.get("complexity", "medium"),
            audience=metadata.get("audience", "general"),
            intent=metadata.get("intent", "general_knowledge"),
            confidence=metadata.get("confidence", 0.8),
        )
