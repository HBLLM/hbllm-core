"""
Reflective Controller — post-execution evaluation and memory consolidation.

Part of the Tripartite Executive (ADR 002 §1):
    - ReactiveController: Reflexes, interrupts, urgent events (<10ms).
    - DeliberativeController: Planning, reasoning, SkillGraph execution.
    - **ReflectiveController**: Post-eval, memory consolidation, self-improvement.

The ReflectiveController runs asynchronously after task completion to:
    1. Evaluate outcome quality via reward scoring.
    2. Emit ``ReflectionEvent`` payloads to the Memory Service.
    3. Trigger skill effectiveness scoring.
    4. Coordinate sleep cycle compaction requests.

Design invariants (ADR 002):
    - Does NOT own memory consolidation logic — emits events to MemoryService.
    - Runs at BACKGROUND scheduler priority.
    - All outputs carry causal ``ProvenanceMetadata``.

Usage::

    from hbllm.brain.control.reflective_controller import ReflectiveController

    controller = ReflectiveController()
    reflection = await controller.reflect(deliberation_result)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hbllm.brain.core.provenance import ProvenanceMetadata

logger = logging.getLogger(__name__)


class ReflectionType(StrEnum):
    """Categories of reflection events emitted to the Memory Service."""

    OUTCOME_EVALUATION = "outcome_evaluation"
    SKILL_SCORING = "skill_scoring"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    STRATEGY_UPDATE = "strategy_update"
    COMPACTION_REQUEST = "compaction_request"


@dataclass
class ReflectionEvent:
    """An event emitted by the ReflectiveController to the Memory Service.

    The Memory Service is responsible for acting on these events
    (consolidation, pruning, skill updates, etc.).

    Attributes:
        reflection_id: Globally unique identifier.
        reflection_type: Category of the reflection.
        source_task_id: Task/deliberation ID being reflected upon.
        outcome_score: Quality score of the outcome [0.0, 1.0].
        insights: Structured insights extracted from the execution.
        recommendations: Suggested improvements or learning signals.
        provenance: Causal provenance metadata.
        created_at: Timestamp.
    """

    reflection_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    reflection_type: ReflectionType = ReflectionType.OUTCOME_EVALUATION
    source_task_id: str = ""
    outcome_score: float = 0.5
    insights: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    provenance: ProvenanceMetadata | None = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging and bus emission."""
        d: dict[str, Any] = {
            "reflection_id": self.reflection_id,
            "reflection_type": self.reflection_type.value,
            "source_task_id": self.source_task_id,
            "outcome_score": round(self.outcome_score, 4),
            "insights": self.insights,
            "recommendations": self.recommendations,
            "created_at": self.created_at,
        }
        if self.provenance is not None:
            d["provenance"] = self.provenance.to_dict()
        return d


@dataclass
class ReflectionResult:
    """Result of a reflection cycle.

    Attributes:
        events: List of ``ReflectionEvent`` objects emitted.
        total_score: Aggregated quality score across all evaluations.
        latency_ms: Reflection processing time.
        provenance: Causal provenance.
    """

    events: list[ReflectionEvent] = field(default_factory=list)
    total_score: float = 0.0
    latency_ms: float = 0.0
    provenance: ProvenanceMetadata | None = None


class ReflectiveController:
    """Asynchronous post-execution evaluator and memory consolidation trigger.

    After a task completes, the ReflectiveController:
        1. Scores the outcome quality.
        2. Extracts execution insights (latency, errors, tool usage).
        3. Emits ``ReflectionEvent`` objects to the Memory Service.
        4. Optionally requests sleep cycle compaction.

    The controller does NOT own consolidation logic — it only emits
    structured events that the Memory Service processes independently.

    Args:
        score_threshold: Minimum outcome score to emit SKILL_SCORING events.
        compaction_threshold: Number of reflections before requesting compaction.
    """

    def __init__(
        self,
        score_threshold: float = 0.3,
        compaction_threshold: int = 50,
    ) -> None:
        self._score_threshold = score_threshold
        self._compaction_threshold = compaction_threshold

        # Counters
        self._total_reflections = 0
        self._pending_compaction_count = 0
        self._total_events_emitted = 0

        logger.info(
            "ReflectiveController initialized (score_thresh=%.2f, compaction_thresh=%d)",
            score_threshold,
            compaction_threshold,
        )

    async def reflect(
        self,
        task_id: str,
        outcome: dict[str, Any],
        reasoning_trace: list[dict[str, Any]] | None = None,
        parent_provenance: ProvenanceMetadata | None = None,
    ) -> ReflectionResult:
        """Execute a reflection cycle for a completed task.

        Args:
            task_id: The identifier of the completed task/deliberation.
            outcome: Execution outcome data (success, errors, tool results).
            reasoning_trace: Optional reasoning trace from the deliberation.
            parent_provenance: Provenance of the parent deliberation.

        Returns:
            ReflectionResult with emitted events and aggregate score.
        """
        self._total_reflections += 1
        self._pending_compaction_count += 1
        start = time.monotonic()

        prov = (
            ProvenanceMetadata.derive(
                parent=parent_provenance,
                source="reflective_controller",
            )
            if parent_provenance
            else ProvenanceMetadata.create(
                source="reflective_controller",
            )
        )

        events: list[ReflectionEvent] = []

        # 1. Outcome evaluation
        outcome_score = self._evaluate_outcome(outcome)
        events.append(
            ReflectionEvent(
                reflection_type=ReflectionType.OUTCOME_EVALUATION,
                source_task_id=task_id,
                outcome_score=outcome_score,
                insights={
                    "success": outcome.get("success", False),
                    "error_count": len(outcome.get("errors", [])),
                    "tool_count": len(outcome.get("tools_used", [])),
                },
                provenance=prov,
            )
        )

        # 2. Skill scoring (only if outcome is above threshold)
        if outcome_score >= self._score_threshold:
            events.append(
                ReflectionEvent(
                    reflection_type=ReflectionType.SKILL_SCORING,
                    source_task_id=task_id,
                    outcome_score=outcome_score,
                    insights={
                        "skills_used": outcome.get("skills_used", []),
                        "improvement_delta": outcome_score - 0.5,
                    },
                    provenance=ProvenanceMetadata.derive(
                        parent=prov, source="reflective.skill_scorer"
                    ),
                )
            )

        # 3. Memory consolidation request
        if reasoning_trace:
            events.append(
                ReflectionEvent(
                    reflection_type=ReflectionType.MEMORY_CONSOLIDATION,
                    source_task_id=task_id,
                    outcome_score=outcome_score,
                    insights={
                        "trace_length": len(reasoning_trace),
                        "key_decisions": [
                            step.get("step", "")
                            for step in reasoning_trace
                            if step.get("step") not in ("error",)
                        ],
                    },
                    recommendations=self._extract_recommendations(outcome, outcome_score),
                    provenance=ProvenanceMetadata.derive(
                        parent=prov, source="reflective.consolidator"
                    ),
                )
            )

        # 4. Compaction request (periodic)
        if self._pending_compaction_count >= self._compaction_threshold:
            events.append(
                ReflectionEvent(
                    reflection_type=ReflectionType.COMPACTION_REQUEST,
                    source_task_id=task_id,
                    outcome_score=0.0,
                    insights={"pending_count": self._pending_compaction_count},
                    provenance=ProvenanceMetadata.derive(
                        parent=prov, source="reflective.compaction"
                    ),
                )
            )
            self._pending_compaction_count = 0

        self._total_events_emitted += len(events)
        latency_ms = (time.monotonic() - start) * 1000.0

        return ReflectionResult(
            events=events,
            total_score=outcome_score,
            latency_ms=latency_ms,
            provenance=prov,
        )

    def _evaluate_outcome(self, outcome: dict[str, Any]) -> float:
        """Score the quality of a task outcome.

        Override this method to integrate the TrainedPRM SNN or
        other reward models.
        """
        if outcome.get("success", False):
            base = 0.7
        else:
            base = 0.2

        # Penalize errors
        error_count = len(outcome.get("errors", []))
        penalty = min(0.3, error_count * 0.1)

        return max(0.0, min(1.0, base - penalty))

    def _extract_recommendations(
        self,
        outcome: dict[str, Any],
        score: float,
    ) -> list[str]:
        """Extract improvement recommendations from an outcome."""
        recs: list[str] = []
        if score < 0.5:
            recs.append("Consider alternative planning strategies for similar intents.")
        if outcome.get("errors"):
            recs.append(f"Address {len(outcome['errors'])} error(s) in tool execution.")
        if score > 0.8:
            recs.append("Reinforce successful skill patterns in procedural memory.")
        return recs

    # ── Telemetry ────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Controller statistics."""
        return {
            "total_reflections": self._total_reflections,
            "total_events_emitted": self._total_events_emitted,
            "pending_compaction_count": self._pending_compaction_count,
        }
