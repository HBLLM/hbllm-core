"""Contextual Competence Profile, Provenance Evidence, and Uncertainty Models for A21.

Enforces strict separation between historical task competence, calibration,
epistemic/aleatoric/structural uncertainty, and metacognitive confidence.
Tracks provenance-bearing evidence for all self-model performance updates.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class UncertaintyType(str, Enum):
    """The nature of internal uncertainty."""

    EPISTEMIC = "epistemic"  # Lack of observation data / sparse sample count
    ALEATORIC = "aleatoric"  # Inherent physical environmental stochasticity
    STRUCTURAL_MODEL = "structural"  # Inadequate/missing model representation or wrong schema


class EpistemicMaturity(str, Enum):
    """Maturity stage of the self-model's experience in a domain."""

    NOVICE = "novice"  # Few or no historical samples (< 3)
    CALIBRATING = "calibrating"  # Moderate sample size (3 - 10)
    MATURE = "mature"  # Robust sample history (> 10)


@dataclass
class SelfModelEvidence:
    """Provenance-bearing empirical evidence record for self-model updates."""

    evidence_id: str = field(default_factory=lambda: f"sme_{uuid.uuid4().hex[:8]}")
    domain: str = ""
    context_props: dict[str, Any] = field(default_factory=dict)
    attempt_id: str = ""
    predicted_confidence: float = 0.50
    actual_outcome: bool = True
    prediction_error: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class UncertaintyBreakdown:
    """Decomposed multi-signal uncertainty vector."""

    epistemic: float = 0.0  # 0.0 (well sampled) to 1.0 (zero data)
    aleatoric: float = 0.0  # 0.0 (deterministic) to 1.0 (pure coin toss)
    structural_model: float = 0.0  # 0.0 (model accurate) to 1.0 (repeated model failure)

    @property
    def total_uncertainty(self) -> float:
        """Composite bounded uncertainty in [0.0, 1.0]."""
        raw = max(self.epistemic, self.aleatoric, self.structural_model)
        return min(1.0, max(0.0, raw))


@dataclass
class CompetenceProfile:
    """Domain competence profile grounded strictly in historical empirical evidence."""

    domain: str
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    brier_score_sum: float = 0.0
    squared_errors: list[float] = field(default_factory=list)
    evidences: list[SelfModelEvidence] = field(default_factory=list)

    # Contextual boundaries
    known_competent_conditions: list[dict[str, Any]] = field(default_factory=list)
    uncalibrated_conditions: list[dict[str, Any]] = field(default_factory=list)
    structural_failure_count: int = 0

    @property
    def competence(self) -> float:
        """Historical empirical task success rate."""
        if self.attempts == 0:
            return 0.50
        return round(self.successes / float(self.attempts), 4)

    @property
    def brier_score(self) -> float:
        """Mean squared error between predicted confidence and binary outcome: (f - o)^2."""
        if not self.squared_errors:
            return 0.25  # Prior uncalibrated error
        return round(sum(self.squared_errors) / len(self.squared_errors), 4)

    @property
    def epistemic_maturity(self) -> EpistemicMaturity:
        if self.attempts < 3:
            return EpistemicMaturity.NOVICE
        elif self.attempts <= 10:
            return EpistemicMaturity.CALIBRATING
        return EpistemicMaturity.MATURE

    def record_attempt(
        self,
        predicted_confidence: float,
        actual_success: bool,
        context_props: dict[str, Any] | None = None,
        is_structural_mismatch: bool = False,
    ) -> SelfModelEvidence:
        """Record an empirical attempt with provenance-bearing evidence."""
        context = dict(context_props or {})
        outcome_val = 1.0 if actual_success else 0.0
        error = abs(predicted_confidence - outcome_val)
        sq_err = error**2

        self.attempts += 1
        if actual_success:
            self.successes += 1
            if context and context not in self.known_competent_conditions:
                self.known_competent_conditions.append(context)
        else:
            self.failures += 1
            if is_structural_mismatch:
                self.structural_failure_count += 1
            if context and context not in self.uncalibrated_conditions:
                self.uncalibrated_conditions.append(context)

        self.squared_errors.append(sq_err)

        evidence = SelfModelEvidence(
            domain=self.domain,
            context_props=context,
            attempt_id=f"att_{self.attempts}",
            predicted_confidence=predicted_confidence,
            actual_outcome=actual_success,
            prediction_error=error,
        )
        self.evidences.append(evidence)
        return evidence

    def evaluate_context_uncertainty(self, context_props: dict[str, Any]) -> UncertaintyBreakdown:
        """Compute decomposed uncertainty vector given specific context features."""
        # 1. Epistemic uncertainty: high when sample count is sparse
        if self.attempts == 0:
            epistemic = 0.90
        elif self.attempts < 5:
            epistemic = round(max(0.20, 1.0 - (self.attempts * 0.18)), 4)
        else:
            epistemic = round(max(0.05, 0.30 - (self.attempts * 0.02)), 4)

        # Contextual check: novel context props elevate epistemic uncertainty
        if context_props:
            is_known = any(
                all(k in c and c[k] == v for k, v in context_props.items())
                for c in self.known_competent_conditions
            )
            if not is_known and self.attempts > 0:
                epistemic = min(1.0, epistemic + 0.35)

        # 2. Aleatoric uncertainty: empirical variance around base rate
        p = self.competence
        aleatoric = round(4.0 * p * (1.0 - p) * 0.25, 4)  # Peaks at 0.25 when p = 0.50

        # 3. Structural model uncertainty: elevated by repeated unexpected failures
        structural = round(min(1.0, self.structural_failure_count * 0.40), 4)

        return UncertaintyBreakdown(
            epistemic=epistemic,
            aleatoric=aleatoric,
            structural_model=structural,
        )

    def compute_metacognitive_confidence(
        self, context_props: dict[str, Any], prior_confidence: float = 0.70
    ) -> float:
        """Derive context-specific metacognitive confidence from competence, calibration, maturity, and uncertainty."""
        uncertainty = self.evaluate_context_uncertainty(context_props)
        comp = self.competence
        calib_penalty = min(0.30, self.brier_score * 0.5)

        # Base confidence blends empirical competence and prior
        if self.attempts == 0:
            base = 0.40  # Honest ungrounded baseline
        else:
            weight_emp = min(0.85, 0.3 + (self.attempts * 0.1))
            base = (weight_emp * comp) + ((1.0 - weight_emp) * prior_confidence)

        # Demote by total uncertainty and calibration penalty
        final_conf = base * (1.0 - (uncertainty.total_uncertainty * 0.60)) - calib_penalty
        return round(max(0.05, min(0.99, final_conf)), 4)
