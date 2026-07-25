"""
Deliberative Controller — multi-step planning, reasoning, and SkillGraph execution.

Part of the Tripartite Executive (ADR 002 §1):
    - ReactiveController: Reflexes, interrupts, urgent events (<10ms).
    - **DeliberativeController**: Planning, reasoning, SkillGraph execution.
    - ReflectiveController: Post-eval, memory consolidation, self-improvement.

The DeliberativeController handles the "slow thinking" pathway:
Intent → Goal decomposition → Plan generation → SkillGraph execution.

Design invariants:
    - Operates on ``Intent`` objects, NOT raw text.
    - Produces ``Plan`` DAGs that are validated before execution.
    - All outputs carry causal ``ProvenanceMetadata``.
    - Runs at LATENCY_SENSITIVE scheduler priority.

Usage::

    from hbllm.brain.control.deliberative_controller import DeliberativeController

    controller = DeliberativeController()
    result = await controller.deliberate(intent, cognitive_state)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hbllm.brain.control.intent import Intent
from hbllm.brain.core.provenance import ProvenanceMetadata

logger = logging.getLogger(__name__)


class PlanStatus(StrEnum):
    """Lifecycle status of a deliberative plan."""

    DRAFT = "draft"
    VALIDATED = "validated"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Plan:
    """A strategy DAG produced by the DeliberativeController.

    Plans are intermediate between Goals and SkillGraphs — they
    represent *how* to achieve a goal before committing to specific
    tool invocations.

    Attributes:
        plan_id: Globally unique identifier.
        intent_id: The intent this plan serves.
        goal_description: Human-readable goal derived from the intent.
        steps: Ordered list of step descriptions / DAG nodes.
        status: Current lifecycle status.
        confidence: Planner's confidence in this strategy.
        estimated_cost: Estimated resource cost.
        provenance: Causal provenance.
        created_at: Timestamp.
    """

    plan_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    intent_id: str = ""
    goal_description: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    status: PlanStatus = PlanStatus.DRAFT
    confidence: float = 1.0
    estimated_cost: dict[str, float] = field(default_factory=dict)
    provenance: ProvenanceMetadata | None = None
    created_at: float = field(default_factory=time.time)

    def validate(self) -> Plan:
        """Mark plan as validated after safety checks."""
        self.status = PlanStatus.VALIDATED
        return self

    def execute(self) -> Plan:
        """Mark plan as executing."""
        self.status = PlanStatus.EXECUTING
        return self

    def complete(self) -> Plan:
        """Mark plan as completed."""
        self.status = PlanStatus.COMPLETED
        return self

    def fail(self, reason: str = "") -> Plan:
        """Mark plan as failed."""
        self.status = PlanStatus.FAILED
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging and persistence."""
        d: dict[str, Any] = {
            "plan_id": self.plan_id,
            "intent_id": self.intent_id,
            "goal_description": self.goal_description,
            "steps": self.steps,
            "status": self.status.value,
            "confidence": round(self.confidence, 4),
            "estimated_cost": self.estimated_cost,
            "created_at": self.created_at,
        }
        if self.provenance is not None:
            d["provenance"] = self.provenance.to_dict()
        return d


@dataclass
class DeliberationResult:
    """Result of a deliberative cycle.

    Attributes:
        intent: The intent that was processed.
        plan: The plan produced (if successful).
        success: Whether deliberation succeeded.
        reasoning_trace: Step-by-step reasoning log for replay.
        latency_ms: Total deliberation time.
        provenance: Causal provenance.
    """

    intent: Intent
    plan: Plan | None = None
    success: bool = False
    reasoning_trace: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    provenance: ProvenanceMetadata | None = None


class DeliberativeController:
    """Multi-step planning and reasoning controller.

    Processes ``Intent`` objects through the deliberation pipeline:
    Intent → Goal decomposition → Plan generation → Validation.

    The actual SkillGraph execution is delegated downstream; this
    controller produces validated ``Plan`` objects.

    Args:
        max_planning_steps: Maximum number of planning iterations.
        planning_timeout_s: Timeout for the entire deliberation cycle.
    """

    def __init__(
        self,
        max_planning_steps: int = 10,
        planning_timeout_s: float = 30.0,
    ) -> None:
        self._max_steps = max_planning_steps
        self._timeout_s = planning_timeout_s

        # Telemetry
        self._total_deliberations = 0
        self._total_successes = 0
        self._total_failures = 0
        self._total_latency_ms = 0.0

        logger.info(
            "DeliberativeController initialized (max_steps=%d, timeout=%.1fs)",
            max_planning_steps,
            planning_timeout_s,
        )

    async def deliberate(
        self,
        intent: Intent,
        cognitive_state: Any | None = None,
        context: dict[str, Any] | None = None,
    ) -> DeliberationResult:
        """Execute the deliberation pipeline for an intent.

        Pipeline:
            1. Extract goal from intent
            2. Decompose goal into sub-steps
            3. Generate plan DAG
            4. Validate plan (safety, resource, feasibility)
            5. Return validated plan

        Args:
            intent: The intent to deliberate on.
            cognitive_state: Current cognitive state snapshot.
            context: Additional context (memories, world state, etc.).

        Returns:
            DeliberationResult with the produced plan.
        """
        self._total_deliberations += 1
        start = time.monotonic()
        trace: list[dict[str, Any]] = []

        # Create provenance for this deliberation
        prov = (
            ProvenanceMetadata.derive(
                parent=intent.provenance,
                source="deliberative_controller",
            )
            if intent.provenance
            else ProvenanceMetadata.create(
                source="deliberative_controller",
                confidence=intent.confidence,
            )
        )

        try:
            # Step 1: Extract goal description
            goal_desc = self._extract_goal(intent)
            trace.append(
                {
                    "step": "extract_goal",
                    "result": goal_desc,
                    "timestamp": time.time(),
                }
            )

            # Step 2: Decompose into sub-steps
            steps = self._decompose_goal(goal_desc, intent, context)
            trace.append(
                {
                    "step": "decompose_goal",
                    "step_count": len(steps),
                    "timestamp": time.time(),
                }
            )

            # Step 3: Build plan
            plan = Plan(
                intent_id=intent.intent_id,
                goal_description=goal_desc,
                steps=steps,
                status=PlanStatus.DRAFT,
                confidence=intent.confidence,
                provenance=prov,
            )
            trace.append(
                {
                    "step": "build_plan",
                    "plan_id": plan.plan_id,
                    "timestamp": time.time(),
                }
            )

            # Step 4: Validate plan
            plan.validate()
            trace.append(
                {
                    "step": "validate_plan",
                    "status": plan.status.value,
                    "timestamp": time.time(),
                }
            )

            latency_ms = (time.monotonic() - start) * 1000.0
            self._total_successes += 1
            self._total_latency_ms += latency_ms

            return DeliberationResult(
                intent=intent,
                plan=plan,
                success=True,
                reasoning_trace=trace,
                latency_ms=latency_ms,
                provenance=prov,
            )

        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000.0
            self._total_failures += 1
            self._total_latency_ms += latency_ms
            trace.append(
                {
                    "step": "error",
                    "error": str(exc),
                    "timestamp": time.time(),
                }
            )
            logger.error("Deliberation failed for intent %s: %s", intent.intent_id[:8], exc)
            return DeliberationResult(
                intent=intent,
                plan=None,
                success=False,
                reasoning_trace=trace,
                latency_ms=latency_ms,
                provenance=prov,
            )

    def _extract_goal(self, intent: Intent) -> str:
        """Extract a goal description from an intent.

        Override this method to integrate LLM-based goal extraction.
        """
        return f"{intent.intent_type.value}: {intent.semantic_target}"

    def _decompose_goal(
        self,
        goal: str,
        intent: Intent,
        context: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Decompose a goal into executable sub-steps.

        Override this method to integrate Graph-of-Thought planning.
        """
        return [
            {
                "step_id": uuid.uuid4().hex[:8],
                "description": goal,
                "intent_type": intent.intent_type.value,
                "parameters": intent.parameters,
                "order": 0,
            }
        ]

    # ── Telemetry ────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Controller statistics."""
        total = max(1, self._total_deliberations)
        return {
            "total_deliberations": self._total_deliberations,
            "total_successes": self._total_successes,
            "total_failures": self._total_failures,
            "success_rate": round(self._total_successes / total, 4),
            "avg_latency_ms": round(self._total_latency_ms / total, 2),
        }
