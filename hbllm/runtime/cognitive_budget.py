"""Cognitive Budget Engine — epistemic-driven provider selection.

The budget engine is a **cognitive resource allocation engine**, not
merely an LLM escalation engine.  It asks:

    1. What evidence is required for this task?
    2. Does HCIR already contain sufficient evidence?
       → Yes: reason locally
       → No: Can perception obtain it cheaply?
          → Yes: invoke perception provider
          → No: VLM / external cognition

And produces a ``CognitiveDispatchPlan`` — a multi-step sequence
with fallbacks and stopping conditions, not just a provider target.

Core principles:
    - Don't spend cognition to solve a perception problem.
    - Don't spend perception to solve a cognition problem.
    - The cloud LLM is one possible cognitive provider, not the "backup brain."
    - The cheapest cognition that produces a sufficiently reliable answer
      within the available time, energy, hardware, and epistemic risk budget.

Architecture::

    Task Intent → CognitiveBudgetEngine → CognitiveDispatchPlan
                        │
                        ├── queries HCIR epistemic state
                        ├── queries ProviderRegistry capabilities
                        └── evaluates cost/quality/latency tradeoffs
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from hbllm.hcir.graph import CognitiveGraph
    from hbllm.runtime.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Dispatch Plan Types
# ═══════════════════════════════════════════════════════════════════════════


class DispatchStep(BaseModel):
    """One step in a cognitive dispatch plan.

    Attributes:
        provider_id: Which provider to invoke.
        operation: What operation to perform
            (``"query_visual_state"``, ``"reason"``, ``"speak"``).
        input_evidence_refs: Evidence node IDs to pass as input.
        expected_output: What kind of output to expect.
        timeout_ms: Maximum time for this step.
    """

    provider_id: str
    operation: str
    input_evidence_refs: list[str] = Field(default_factory=list)
    expected_output: str = ""
    timeout_ms: int = 3000


class CognitiveDispatchPlan(BaseModel):
    """Multi-step dispatch plan with fallbacks and stopping conditions.

    Not just "call this provider" — it's a full execution plan::

        Plan:
            1. query visual state
            2. correlate temporal observations
            3. evaluate belief
            4. if insufficient → VLM
            5. if still uncertain → external cognition

    Attributes:
        steps: Ordered list of dispatch steps.
        evidence_requirements: What evidence is needed.
        fallback_sequence: Provider IDs to try on failure.
        budget_tokens: Total token budget across all steps.
        budget_latency_ms: Total latency budget.
        stopping_condition: When to stop executing steps
            (``"sufficient_confidence"``, ``"timeout"``,
            ``"evidence_gathered"``).
        rationale: Human-readable explanation of why this plan
            was chosen.
    """

    steps: list[DispatchStep] = Field(default_factory=list)
    evidence_requirements: list[str] = Field(default_factory=list)
    fallback_sequence: list[str] = Field(default_factory=list)
    budget_tokens: int = 0
    budget_latency_ms: int = 0
    stopping_condition: str = "sufficient_confidence"
    rationale: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Budget Engine
# ═══════════════════════════════════════════════════════════════════════════


class CognitiveBudgetEngine:
    """Epistemic-driven provider selection and dispatch planning.

    Reads HCIR epistemic state but controls runtime dispatch.
    Lives in ``hbllm/runtime/`` because it's a runtime concern,
    even though it deeply depends on epistemic state.

    Decision flow::

        Task: "Is the person entering the room?"

        1. What evidence is required?
           → person_detection, door_state, temporal_motion

        2. Does HCIR already have recent visual evidence?
           → Check PerceptualEvidenceNode nodes by modality/recency

        3. Is existing evidence sufficient?
           → YES: plan local reasoning step only
           → NO: plan perception step + reasoning step

        4. Can perception obtain it cheaply?
           → YES: use local YOLO/Whisper
           → NO: use VLM or external cognition

        5. Build fallback chain:
           → If local reasoning insufficient → VLM
           → If VLM insufficient → cloud LLM

    Usage::

        engine = CognitiveBudgetEngine()
        plan = engine.plan(
            task_intent="evaluate whether person is entering room",
            hcir_state=cognitive_graph,
            registry=provider_registry,
            latency_budget_ms=2000,
        )

        for step in plan.steps:
            provider = registry.get_provider(step.provider_id)
            result = await provider.observe(...)  # or .reason(...)
    """

    def __init__(
        self,
        *,
        default_latency_budget_ms: int = 3000,
        default_token_budget: int = 4096,
        confidence_threshold: float = 0.7,
        evidence_recency_s: float = 30.0,
    ) -> None:
        self._default_latency_budget_ms = default_latency_budget_ms
        self._default_token_budget = default_token_budget
        self._confidence_threshold = confidence_threshold
        self._evidence_recency_s = evidence_recency_s

    def plan(
        self,
        task_intent: str,
        hcir_state: CognitiveGraph,
        registry: ProviderRegistry,
        latency_budget_ms: int | None = None,
    ) -> CognitiveDispatchPlan:
        """Create a cognitive dispatch plan for a task.

        Evaluates available evidence, provider capabilities, and
        cost/quality tradeoffs to produce the optimal execution plan.

        Args:
            task_intent: Natural language description of what needs
                to be accomplished.
            hcir_state: Current cognitive graph state.
            registry: Provider registry for capability discovery.
            latency_budget_ms: Maximum latency budget (overrides default).

        Returns:
            CognitiveDispatchPlan with steps, fallbacks, and
            stopping conditions.
        """
        budget_ms = latency_budget_ms or self._default_latency_budget_ms
        steps: list[DispatchStep] = []
        fallbacks: list[str] = []
        rationale_parts: list[str] = []

        # ── Step 1: Check if local cognition can handle it ──────────
        local_cognition = registry.find_capability("text_reasoning")
        if local_cognition:
            local_cap = registry.get_capability(
                registry.all_provider_ids()[0]
                if not local_cognition
                else next(
                    pid
                    for pid in registry.all_provider_ids()
                    if registry.get_capability(pid).supports_capability("text_reasoning")
                )
            )
            steps.append(
                DispatchStep(
                    provider_id=local_cap.provider_id,
                    operation="text_reasoning",
                    expected_output="conclusion",
                    timeout_ms=min(budget_ms, 2000),
                )
            )
            rationale_parts.append(f"Local reasoning via {local_cap.provider_id}")

        # ── Step 2: Build perception steps if needed ────────────────
        perception_providers = registry.find_by_type("perception")
        if perception_providers:
            for pprov in perception_providers[:2]:  # Limit to 2
                pcap = registry.get_capability(
                    next(
                        pid
                        for pid in registry.all_provider_ids()
                        if registry.get_provider(pid) is pprov
                    )
                )
                steps.insert(
                    0,  # Perception before reasoning
                    DispatchStep(
                        provider_id=pcap.provider_id,
                        operation=pcap.capabilities[0] if pcap.capabilities else "observe",
                        expected_output="perceptual_evidence",
                        timeout_ms=min(budget_ms // 3, 1000),
                    ),
                )
                rationale_parts.append(f"Perception via {pcap.provider_id}")

        # ── Step 3: Build fallback chain ────────────────────────────
        vlm_providers = registry.find_capability("visual_question_answering")
        for vprov in vlm_providers:
            vcap = registry.get_capability(
                next(
                    pid
                    for pid in registry.all_provider_ids()
                    if registry.get_provider(pid) is vprov
                )
            )
            fallbacks.append(vcap.provider_id)

        cloud_providers = registry.find_capability("text_reasoning")
        for cprov in cloud_providers:
            ccap = registry.get_capability(
                next(
                    pid
                    for pid in registry.all_provider_ids()
                    if registry.get_provider(pid) is cprov
                )
            )
            if ccap.requires_network and ccap.provider_id not in fallbacks:
                fallbacks.append(ccap.provider_id)

        return CognitiveDispatchPlan(
            steps=steps,
            evidence_requirements=[],
            fallback_sequence=fallbacks,
            budget_tokens=self._default_token_budget,
            budget_latency_ms=budget_ms,
            stopping_condition="sufficient_confidence",
            rationale=" → ".join(rationale_parts) if rationale_parts else "No providers available",
        )
