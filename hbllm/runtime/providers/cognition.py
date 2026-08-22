"""Cognition Provider Protocol — HBLLM Cognitive Runtime.

Cognition providers reason about evidence and state.  The interface
is NOT prompt-based — it receives structured ``CognitionRequest``
objects so that:

- A local LLM can translate the request into a prompt.
- A symbolic reasoner can process it without an LLM.
- A remote model can receive a transformed representation.

This keeps HBLLM cognition independent of LLM prompting.

Architecture::

    CognitionRequest
        ├── local 3B LLM
        ├── local VLM
        ├── symbolic reasoner
        ├── planning engine
        └── cloud LLM

The provider should not determine the cognitive interface.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from hbllm.runtime.providers.capability import ProviderCapability

# ═══════════════════════════════════════════════════════════════════════════
# Request & Response Types
# ═══════════════════════════════════════════════════════════════════════════


class CognitionRequest(BaseModel):
    """Structured cognition request — NOT a prompt string.

    Encapsulates everything a cognition provider needs to reason,
    independent of the underlying implementation (LLM, VLM,
    symbolic, mathematical).

    Attributes:
        intent: What cognitive operation is being requested
            (``"evaluate_scene"``, ``"plan_action"``, ``"classify"``,
            ``"explain_evidence"``).
        cognitive_state_summary: Relevant HCIR state projection
            (key facts, active beliefs, current context).
        evidence_refs: Evidence node IDs to reason about.
        constraints: Active constraints or rules that must be respected.
        goals: Active goal descriptions (from HCIR GoalNodes).
        latency_budget_ms: Maximum acceptable latency.
        reasoning_budget_tokens: Maximum reasoning tokens to spend.
        output_schema: Optional expected output structure for
            structured generation.
    """

    intent: str = ""
    cognitive_state_summary: dict = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    latency_budget_ms: int = 3000
    reasoning_budget_tokens: int = 2048
    output_schema: dict | None = None


class ThoughtResult(BaseModel):
    """Result from any cognition provider.

    Captures the conclusion, confidence, reasoning trace, and
    resource usage — regardless of whether the provider was a
    local LLM, VLM, symbolic reasoner, or cloud API.

    Attributes:
        conclusion: The primary output/answer.
        confidence: Self-assessed confidence ``[0.0, 1.0]``.
        reasoning_trace: Step-by-step reasoning chain (if available).
        evidence_produced: IDs of new evidence nodes produced.
        tokens_used: Total tokens consumed (0 for non-LLM providers).
        latency_ms: Wall-clock latency in milliseconds.
        provider_id: Which provider produced this result.
    """

    conclusion: str = ""
    confidence: float = 0.5
    reasoning_trace: list[str] = Field(default_factory=list)
    evidence_produced: list[str] = Field(default_factory=list)
    tokens_used: int = 0
    latency_ms: float = 0.0
    provider_id: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Protocol
# ═══════════════════════════════════════════════════════════════════════════


@runtime_checkable
class CognitionProvider(Protocol):
    """Reasoning provider — local LLM, VLM, symbolic reasoner, or cloud API.

    Receives structured ``CognitionRequest`` objects rather than
    raw prompts, keeping HBLLM cognition independent of LLM
    prompting conventions.

    A local LLM implementation would translate the request into
    a prompt.  A symbolic reasoner could process it directly.
    A remote model could receive a transformed representation.
    """

    @property
    def capability(self) -> ProviderCapability:
        """Declarative capability manifest for this provider."""
        ...

    async def reason(self, request: CognitionRequest) -> ThoughtResult:
        """Perform a reasoning operation.

        Args:
            request: Structured cognition request with intent,
                evidence refs, constraints, goals, and budgets.

        Returns:
            ThoughtResult with conclusion, confidence, and trace.
        """
        ...

    async def initialize(self) -> None:
        """Initialize provider resources (load model, etc.)."""
        ...

    async def shutdown(self) -> None:
        """Release provider resources."""
        ...
