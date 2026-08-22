"""LLM Cognition Adapter — concrete CognitionProvider wrapping LLMProvider instances.

Translates structured CognitionRequest objects into model invocations (Local, OpenAI,
Anthropic, Ollama, Groq) and packages outputs into ThoughtResult objects.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from hbllm.runtime.providers.capability import ProviderCapability
from hbllm.runtime.providers.cognition import CognitionProvider, CognitionRequest, ThoughtResult
from hbllm.serving.provider import LLMProvider, get_provider

logger = logging.getLogger(__name__)


class LLMCognitionAdapter:
    """Concrete CognitionProvider adapting LLMProvider instances.

    Conforms to ``CognitionProvider``.
    Translates structured ``CognitionRequest`` (intent, state summary, evidence refs,
    constraints, goals) into provider calls and captures reasoning traces.

    Usage::

        adapter = LLMCognitionAdapter(provider_name="openai", model="gpt-4o-mini")
        await adapter.initialize()
        result = await adapter.reason(cognition_request)
    """

    def __init__(
        self,
        provider_id: str = "llm_cognition",
        provider_name: str = "local",
        model: str | None = None,
        underlying_provider: LLMProvider | None = None,
        latency_profile: str = "medium",
        quality_profile: str = "high",
        requires_network: bool = False,
    ) -> None:
        self._provider_id = provider_id
        self._provider_name = provider_name
        self._model = model
        self._underlying_provider = underlying_provider
        self._latency_profile = latency_profile
        self._quality_profile = quality_profile
        self._requires_network = requires_network

    @property
    def capability(self) -> ProviderCapability:
        """Declarative capability manifest for LLM reasoning."""
        return ProviderCapability(
            provider_id=self._provider_id,
            provider_type="cognition",
            capabilities=["text_reasoning", "planning", "intent_resolution", "evidence_synthesis"],
            modalities=["text"],
            latency_profile=self._latency_profile,
            quality_profile=self._quality_profile,
            requires_network=self._requires_network,
            max_input_tokens=8192,
        )

    async def initialize(self) -> None:
        """Initialize underlying LLM provider."""
        if self._underlying_provider is None:
            try:
                self._underlying_provider = get_provider(self._provider_name, model=self._model)
                logger.info(
                    "Initialized LLMCognitionAdapter with %s provider (model=%s)",
                    self._provider_name,
                    self._model,
                )
            except Exception as e:
                logger.warning(
                    "Could not initialize provider '%s' directly: %s. Provider calls will handle fallback.",
                    self._provider_name,
                    e,
                )

    async def shutdown(self) -> None:
        """Release LLM provider resources."""
        self._underlying_provider = None
        logger.info("Shutdown LLMCognitionAdapter (%s)", self._provider_id)

    async def reason(self, request: CognitionRequest) -> ThoughtResult:
        """Perform reasoning over a structured CognitionRequest.

        Args:
            request: Structured cognition request with intent, state, constraints, and goals.

        Returns:
            ThoughtResult with conclusion, confidence, and reasoning trace.
        """
        start_time = time.time()
        trace: list[str] = [
            f"Cognition intent: {request.intent}",
            f"Evidence references considered: {len(request.evidence_refs)}",
            f"Active goals: {len(request.goals)}",
            f"Active constraints: {len(request.constraints)}",
        ]

        # Construct prompt representation from structured state
        prompt = self._format_request_prompt(request)

        # Execute generation if underlying provider is available
        conclusion = ""
        confidence = 0.85
        tokens_used = 0

        if self._underlying_provider is not None:
            try:
                response = await self._underlying_provider.generate(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are the cognitive core of HBLLM. Reason over the grounded "
                                "state and provided evidence to achieve the active goal while "
                                "strictly respecting all constraints."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=min(request.reasoning_budget_tokens, 2048),
                )
                conclusion = response.content
                tokens_used = response.usage.get("total_tokens", 0) if response.usage else 0
                trace.append(f"Generated reasoning via {response.model}")
            except Exception as e:
                logger.warning("LLM reasoning error: %s; falling back to synthesis", e)
                conclusion = f"Synthesized analysis for intent '{request.intent}' over evidence {request.evidence_refs}."
                confidence = 0.6
                trace.append(f"Fallback synthesis due to provider error: {e}")
        else:
            conclusion = f"Synthesized analysis for intent '{request.intent}' over evidence {request.evidence_refs}."
            confidence = 0.7
            trace.append("Synthesized reasoning output (provider offline)")

        duration_ms = (time.time() - start_time) * 1000.0

        return ThoughtResult(
            conclusion=conclusion,
            confidence=confidence,
            reasoning_trace=trace,
            evidence_produced=[],
            tokens_used=tokens_used,
            latency_ms=duration_ms,
            provider_id=self._provider_id,
        )

    def _format_request_prompt(self, request: CognitionRequest) -> str:
        """Format structured request into model prompt text."""
        parts: list[str] = [f"INTENT: {request.intent}"]

        if request.cognitive_state_summary:
            parts.append("STATE SUMMARY:")
            for k, v in request.cognitive_state_summary.items():
                parts.append(f"  - {k}: {v}")

        if request.goals:
            parts.append("GOALS:")
            for g in request.goals:
                parts.append(f"  - {g}")

        if request.constraints:
            parts.append("CONSTRAINTS:")
            for c in request.constraints:
                parts.append(f"  - {c}")

        if request.evidence_refs:
            parts.append(f"EVIDENCE NODES: {', '.join(request.evidence_refs)}")

        return "\n".join(parts)
