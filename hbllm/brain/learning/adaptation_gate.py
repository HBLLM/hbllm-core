"""Adaptation Gate — the sole authority over model mutation for A14.

**Critical architectural invariant:**

    Neither ErrorClassifier, LearningSignalRouter, nor CognitiveAdaptationLoop
    can directly modify models. Only AdaptationGate can authorize model changes.

Internally structured as:

    ErrorEvidenceAccumulator → AdaptationGate

The accumulator maintains deterministic evidence state.
The gate makes deterministic decisions from that state.
This separation makes testing dramatically easier.

Gate decisions::

    DEFER    — insufficient evidence, accumulate more
    REJECT   — noise, no adaptation needed
    ADAPT    — sufficient model error evidence, authorize adaptation
    EXPLORE  — novelty detected, authorize exploration
    WORLD_UPDATE — environment changed, route to A13

Anti-oscillation: If a model has been adapted N times in the last T
seconds, the gate tightens (higher evidence threshold).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum

from hbllm.brain.learning.error_classifier import ErrorClassification

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Gate Decision
# ═══════════════════════════════════════════════════════════════════════════


class GateDecision(StrEnum):
    """Possible decisions from the AdaptationGate."""

    DEFER = "defer"  # Insufficient evidence — accumulate more
    REJECT = "reject"  # Noise — no adaptation needed
    ADAPT = "adapt"  # Sufficient model error — authorize adaptation
    EXPLORE = "explore"  # Novelty — authorize exploration
    WORLD_UPDATE = "world_update"  # Environment changed — route to A13


@dataclass(frozen=True)
class GateVerdict:
    """The gate's decision with supporting evidence."""

    decision: GateDecision
    confidence: float = 0.0  # How confident the gate is in this decision
    evidence_count: int = 0  # How many errors contributed to this verdict
    dominant_classification: str = ""  # The dominant error class
    reasoning: str = ""  # Human-readable explanation


# ═══════════════════════════════════════════════════════════════════════════
# Error Evidence Accumulator — separated from gate decision
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ErrorEvidence:
    """Accumulated evidence for a specific prediction domain/model."""

    model_id: str
    domain: str = ""
    occurrences: int = 0
    total_magnitude: float = 0.0
    weighted_magnitude: float = 0.0  # Recency-weighted
    avg_model_error_prob: float = 0.0  # Average P(model_error) across errors
    avg_env_change_prob: float = 0.0
    avg_noise_prob: float = 0.0
    avg_novelty_prob: float = 0.0
    max_model_error_prob: float = 0.0
    temporal_density: float = 0.0  # Errors per second (recent window)
    first_error_time: float = 0.0
    last_error_time: float = 0.0
    recent_adaptation_count: int = 0  # Adaptations in the last window
    error_ids: list[str] = field(default_factory=list)
    classifications: list[ErrorClassification] = field(default_factory=list)


class ErrorEvidenceAccumulator:
    """Maintains deterministic evidence state per model/domain.

    Separated from AdaptationGate so that:
    - accumulator → deterministic evidence state
    - gate → deterministic decision from that state

    This makes testing dramatically easier.
    """

    def __init__(
        self,
        recency_window_s: float = 300.0,  # 5 minute window for temporal density
    ) -> None:
        self._evidence: dict[str, ErrorEvidence] = {}  # model_id → evidence
        self._recency_window = recency_window_s
        self._adaptation_timestamps: dict[str, list[float]] = {}  # model_id → timestamps

    def accumulate(
        self,
        model_id: str,
        error_id: str,
        classification: ErrorClassification,
        error_magnitude: float = 0.0,
        domain: str = "",
        timestamp: float | None = None,
    ) -> ErrorEvidence:
        """Accumulate a classified error into the evidence for a model.

        Returns the updated evidence state (deterministic).
        """
        now = timestamp if timestamp is not None else time.time()

        if model_id not in self._evidence:
            self._evidence[model_id] = ErrorEvidence(
                model_id=model_id,
                domain=domain,
                first_error_time=now,
            )

        ev = self._evidence[model_id]
        ev.occurrences += 1
        ev.total_magnitude += error_magnitude
        ev.last_error_time = now
        ev.error_ids.append(error_id)
        ev.classifications.append(classification)

        # Update running averages
        n = ev.occurrences
        ev.avg_model_error_prob = (
            (ev.avg_model_error_prob * (n - 1) + classification.model_error) / n
        )
        ev.avg_env_change_prob = (
            (ev.avg_env_change_prob * (n - 1) + classification.environment_change) / n
        )
        ev.avg_noise_prob = (
            (ev.avg_noise_prob * (n - 1) + classification.noise) / n
        )
        ev.avg_novelty_prob = (
            (ev.avg_novelty_prob * (n - 1) + classification.novelty) / n
        )
        ev.max_model_error_prob = max(ev.max_model_error_prob, classification.model_error)

        # Recency-weighted magnitude (exponential decay)
        decay = 0.9 ** max(0, n - 1)
        ev.weighted_magnitude = ev.weighted_magnitude * 0.9 + error_magnitude * decay

        # Temporal density
        time_span = now - ev.first_error_time
        if time_span > 0:
            ev.temporal_density = ev.occurrences / time_span

        # Count recent adaptations
        adapt_times = self._adaptation_timestamps.get(model_id, [])
        ev.recent_adaptation_count = sum(
            1 for t in adapt_times if (now - t) < self._recency_window
        )

        return ev

    def record_adaptation(self, model_id: str, timestamp: float | None = None) -> None:
        """Record that an adaptation was performed for a model."""
        now = timestamp if timestamp is not None else time.time()
        if model_id not in self._adaptation_timestamps:
            self._adaptation_timestamps[model_id] = []
        self._adaptation_timestamps[model_id].append(now)

    def get_evidence(self, model_id: str) -> ErrorEvidence | None:
        """Get current evidence state for a model."""
        return self._evidence.get(model_id)

    def clear_evidence(self, model_id: str) -> None:
        """Clear accumulated evidence after an adaptation decision."""
        if model_id in self._evidence:
            del self._evidence[model_id]

    @property
    def tracked_models(self) -> set[str]:
        return set(self._evidence.keys())


# ═══════════════════════════════════════════════════════════════════════════
# Adaptation Gate — the sole authority over model mutation
# ═══════════════════════════════════════════════════════════════════════════


class AdaptationGate:
    """The sole authority that can authorize model mutation.

    Takes deterministic evidence from ErrorEvidenceAccumulator and
    produces a GateVerdict.

    **Anti-oscillation:** If a model has been adapted recently,
    the evidence threshold increases, preventing rapid oscillation.

    Usage::

        gate = AdaptationGate(accumulator=accumulator)

        # After accumulating errors...
        verdict = gate.evaluate(model_id)

        if verdict.decision == GateDecision.ADAPT:
            # Only now can adaptation proceed
            engine.adapt(...)
            accumulator.record_adaptation(model_id)
            accumulator.clear_evidence(model_id)
    """

    def __init__(
        self,
        accumulator: ErrorEvidenceAccumulator,
        # Thresholds
        min_evidence_count: int = 3,  # Minimum errors before adapting
        model_error_threshold: float = 0.6,  # P(model_error) threshold
        noise_threshold: float = 0.5,  # P(noise) threshold for rejection
        novelty_threshold: float = 0.4,  # P(novelty) threshold for exploration
        env_change_threshold: float = 0.5,  # P(env_change) threshold
        # Anti-oscillation
        max_recent_adaptations: int = 3,  # Max adaptations in window
        oscillation_penalty: float = 0.15,  # Extra threshold per recent adaptation
    ) -> None:
        self._accumulator = accumulator
        self._min_evidence = min_evidence_count
        self._model_error_threshold = model_error_threshold
        self._noise_threshold = noise_threshold
        self._novelty_threshold = novelty_threshold
        self._env_change_threshold = env_change_threshold
        self._max_recent_adaptations = max_recent_adaptations
        self._oscillation_penalty = oscillation_penalty

    def evaluate(self, model_id: str) -> GateVerdict:
        """Evaluate whether a model should be adapted.

        The gate makes a deterministic decision from accumulated evidence.

        Args:
            model_id: The model to evaluate.

        Returns:
            GateVerdict with the decision and supporting evidence.
        """
        evidence = self._accumulator.get_evidence(model_id)

        if evidence is None or evidence.occurrences == 0:
            return GateVerdict(
                decision=GateDecision.DEFER,
                reasoning="No error evidence accumulated.",
            )

        # Anti-oscillation: tighten threshold if recently adapted
        threshold_boost = (
            evidence.recent_adaptation_count * self._oscillation_penalty
        )
        effective_model_threshold = min(
            self._model_error_threshold + threshold_boost, 0.95,
        )

        # Decision 1: REJECT if noise dominates
        if evidence.avg_noise_prob >= self._noise_threshold:
            return GateVerdict(
                decision=GateDecision.REJECT,
                confidence=evidence.avg_noise_prob,
                evidence_count=evidence.occurrences,
                dominant_classification="noise",
                reasoning=(
                    f"Noise probability {evidence.avg_noise_prob:.2f} "
                    f">= threshold {self._noise_threshold:.2f}"
                ),
            )

        # Decision 2: WORLD_UPDATE if environment change dominates
        if evidence.avg_env_change_prob >= self._env_change_threshold:
            return GateVerdict(
                decision=GateDecision.WORLD_UPDATE,
                confidence=evidence.avg_env_change_prob,
                evidence_count=evidence.occurrences,
                dominant_classification="environment_change",
                reasoning=(
                    f"Environment change probability {evidence.avg_env_change_prob:.2f} "
                    f">= threshold {self._env_change_threshold:.2f}"
                ),
            )

        # Decision 3: EXPLORE if novelty is dominant
        if evidence.avg_novelty_prob >= self._novelty_threshold:
            return GateVerdict(
                decision=GateDecision.EXPLORE,
                confidence=evidence.avg_novelty_prob,
                evidence_count=evidence.occurrences,
                dominant_classification="novelty",
                reasoning=(
                    f"Novelty probability {evidence.avg_novelty_prob:.2f} "
                    f">= threshold {self._novelty_threshold:.2f}"
                ),
            )

        # Decision 4: DEFER if insufficient evidence count
        if evidence.occurrences < self._min_evidence:
            return GateVerdict(
                decision=GateDecision.DEFER,
                confidence=evidence.avg_model_error_prob,
                evidence_count=evidence.occurrences,
                dominant_classification="model_error",
                reasoning=(
                    f"Only {evidence.occurrences} errors accumulated, "
                    f"need {self._min_evidence}"
                ),
            )

        # Decision 5: DEFER if too many recent adaptations (anti-oscillation)
        if evidence.recent_adaptation_count >= self._max_recent_adaptations:
            return GateVerdict(
                decision=GateDecision.DEFER,
                confidence=evidence.avg_model_error_prob,
                evidence_count=evidence.occurrences,
                dominant_classification="model_error",
                reasoning=(
                    f"Anti-oscillation: {evidence.recent_adaptation_count} "
                    f"recent adaptations >= max {self._max_recent_adaptations}"
                ),
            )

        # Decision 6: ADAPT if model error is dominant with sufficient evidence
        if evidence.avg_model_error_prob >= effective_model_threshold:
            return GateVerdict(
                decision=GateDecision.ADAPT,
                confidence=evidence.avg_model_error_prob,
                evidence_count=evidence.occurrences,
                dominant_classification="model_error",
                reasoning=(
                    f"Model error probability {evidence.avg_model_error_prob:.2f} "
                    f">= effective threshold {effective_model_threshold:.2f} "
                    f"with {evidence.occurrences} errors"
                ),
            )

        # Default: DEFER — not enough signal to act
        return GateVerdict(
            decision=GateDecision.DEFER,
            confidence=evidence.avg_model_error_prob,
            evidence_count=evidence.occurrences,
            dominant_classification="model_error",
            reasoning=(
                f"Model error probability {evidence.avg_model_error_prob:.2f} "
                f"< effective threshold {effective_model_threshold:.2f}"
            ),
        )
