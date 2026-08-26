"""Concept Consolidator — validates concept hypotheses for A15.

**Core invariant:** Similarity proposes a concept; prediction justifies it.

Formation criteria (ALL required):
1. Minimum exemplars
2. Temporal stability
3. Non-redundancy
4. Behavioral coherence
5. **Predictive utility** (decisive) — measured against a counterfactual baseline

Predictive utility is measured as::

    utility_delta = concept_prediction_accuracy - baseline_prediction_accuracy

A concept is only admitted if ``utility_delta > minimum_gain`` with enough samples.

Also provides:
- Merge: two concepts with >80% overlapping exemplars
- Split signal: heterogeneous prediction profiles within a concept
- ConceptDegradationSignal for low-confidence concepts (not deletion)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

from hbllm.brain.concepts.concept_hypothesis_generator import ConceptHypothesis

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Consolidation Types
# ═══════════════════════════════════════════════════════════════════════════


class ConsolidationDecision(StrEnum):
    """Result of consolidating a concept hypothesis."""

    ACCEPT = "accept"  # Predictive utility confirmed
    REJECT = "reject"  # Insufficient utility or coherence
    DEFER = "defer"  # Not enough evidence yet
    MERGE = "merge"  # Subsumes or overlaps existing concept


@dataclass(frozen=True)
class ConsolidationResult:
    """Outcome of consolidating a concept hypothesis."""

    decision: ConsolidationDecision
    hypothesis_id: str = ""
    utility_delta: float = 0.0  # concept_accuracy - baseline_accuracy
    concept_accuracy: float = 0.0
    baseline_accuracy: float = 0.0
    sample_count: int = 0
    reasoning: str = ""
    merge_with_concept_id: str = ""  # If MERGE, which existing concept


@dataclass
class ConceptRefinementSignal:
    """Signal that a concept is internally heterogeneous.

    Emitted when members show divergent prediction profiles.
    v1 emits the signal; later versions can auto-split.
    """

    concept_id: str = ""
    subgroup_a: list[str] = field(default_factory=list)  # Entity IDs
    subgroup_b: list[str] = field(default_factory=list)
    accuracy_a: float = 0.0
    accuracy_b: float = 0.0
    divergence: float = 0.0  # How different the subgroups are


@dataclass
class ConceptDegradationSignal:
    """Signal that a concept has degraded in predictive utility.

    Low confidence → refinement/retirement signal, NOT immediate deletion.
    Consolidation decides whether to retain, refine, merge, split, or retire.
    """

    concept_id: str = ""
    current_confidence: float = 0.0
    prediction_accuracy: float = 0.0
    recent_failure_rate: float = 0.0
    recommendation: str = ""  # "refine", "merge", "retire"


# ═══════════════════════════════════════════════════════════════════════════
# Predictive Utility Evaluator
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PredictiveUtilityTest:
    """Container for a predictive utility comparison.

    Baseline: predict each entity independently.
    Concept: predict entity using shared concept model.
    """

    # Per-entity outcomes: entity_id → list of (predicted_correct: bool)
    individual_outcomes: dict[str, list[bool]] = field(default_factory=dict)
    concept_outcomes: list[bool] = field(default_factory=list)

    @property
    def baseline_accuracy(self) -> float:
        """Average accuracy when predicting individually."""
        all_outcomes: list[bool] = []
        for outcomes in self.individual_outcomes.values():
            all_outcomes.extend(outcomes)
        return sum(all_outcomes) / len(all_outcomes) if all_outcomes else 0.0

    @property
    def concept_accuracy(self) -> float:
        """Accuracy when predicting using shared concept model."""
        return (
            sum(self.concept_outcomes) / len(self.concept_outcomes)
            if self.concept_outcomes
            else 0.0
        )

    @property
    def utility_delta(self) -> float:
        """Concept accuracy minus baseline accuracy."""
        return self.concept_accuracy - self.baseline_accuracy

    @property
    def sample_count(self) -> int:
        return len(self.concept_outcomes)


# ═══════════════════════════════════════════════════════════════════════════
# Concept Consolidator
# ═══════════════════════════════════════════════════════════════════════════


class ConceptConsolidator:
    """Validates concept hypotheses using predictive utility.

    **Similarity proposes a concept; prediction justifies a concept.**

    The consolidator measures predictive utility against an explicit
    counterfactual baseline:

        utility_delta = concept_prediction_accuracy - baseline_accuracy

    A concept is only admitted if utility_delta > minimum_gain.

    Also provides:
    - ``merge(a, b)`` — merge overlapping concepts
    - ``detect_heterogeneity(concept)`` — split signal for divergent members
    - ``detect_degradation(concept)`` — degradation signal for low confidence

    Usage::

        consolidator = ConceptConsolidator()

        # Test a hypothesis
        test = PredictiveUtilityTest(
            individual_outcomes={"e1": [True, False, True], ...},
            concept_outcomes=[True, True, True, ...],
        )
        result = consolidator.consolidate(hypothesis, test)

        # Check existing concepts for heterogeneity
        signal = consolidator.detect_heterogeneity(
            concept_id="c_001",
            member_outcomes={"e1": [T, T, F], "e2": [F, F, T]},
        )
    """

    def __init__(
        self,
        min_exemplars: int = 2,
        min_utility_gain: float = 0.05,  # 5% improvement required
        min_coherence: float = 0.4,
        min_sample_count: int = 3,
        merge_overlap_threshold: float = 0.8,
        heterogeneity_threshold: float = 0.2,
        degradation_confidence: float = 0.3,
    ) -> None:
        self._min_exemplars = min_exemplars
        self._min_utility_gain = min_utility_gain
        self._min_coherence = min_coherence
        self._min_sample_count = min_sample_count
        self._merge_overlap = merge_overlap_threshold
        self._heterogeneity_threshold = heterogeneity_threshold
        self._degradation_confidence = degradation_confidence

    # ── Primary Consolidation ─────────────────────────────────────────

    def consolidate(
        self,
        hypothesis: ConceptHypothesis,
        utility_test: PredictiveUtilityTest,
        existing_concept_members: dict[str, set[str]] | None = None,
    ) -> ConsolidationResult:
        """Consolidate a concept hypothesis against predictive utility.

        Args:
            hypothesis: The candidate concept.
            utility_test: Predictive utility comparison results.
            existing_concept_members: concept_id → set of member entity IDs.

        Returns:
            ConsolidationResult with decision and evidence.
        """
        # Check 1: Minimum exemplars
        if len(hypothesis.member_ids) < self._min_exemplars:
            return ConsolidationResult(
                decision=ConsolidationDecision.REJECT,
                hypothesis_id=hypothesis.hypothesis_id,
                reasoning=(
                    f"Insufficient exemplars: {len(hypothesis.member_ids)} < {self._min_exemplars}"
                ),
            )

        # Check 2: Behavioral coherence
        if hypothesis.overall_coherence < self._min_coherence:
            return ConsolidationResult(
                decision=ConsolidationDecision.REJECT,
                hypothesis_id=hypothesis.hypothesis_id,
                reasoning=(
                    f"Insufficient coherence: {hypothesis.overall_coherence:.2f} "
                    f"< {self._min_coherence}"
                ),
            )

        # Check 3: Non-redundancy — check for overlap with existing concepts
        if existing_concept_members:
            members = set(hypothesis.member_ids)
            for concept_id, existing_members in existing_concept_members.items():
                overlap = len(members & existing_members) / max(len(members), 1)
                if overlap >= self._merge_overlap:
                    return ConsolidationResult(
                        decision=ConsolidationDecision.MERGE,
                        hypothesis_id=hypothesis.hypothesis_id,
                        reasoning=(f"Overlaps {overlap:.0%} with existing concept {concept_id}"),
                        merge_with_concept_id=concept_id,
                    )

        # Check 4: Sufficient sample count for utility test
        if utility_test.sample_count < self._min_sample_count:
            return ConsolidationResult(
                decision=ConsolidationDecision.DEFER,
                hypothesis_id=hypothesis.hypothesis_id,
                reasoning=(
                    f"Insufficient evaluation samples: {utility_test.sample_count} "
                    f"< {self._min_sample_count}"
                ),
            )

        # Check 5 (DECISIVE): Predictive utility
        delta = utility_test.utility_delta

        if delta < self._min_utility_gain:
            return ConsolidationResult(
                decision=ConsolidationDecision.REJECT,
                hypothesis_id=hypothesis.hypothesis_id,
                utility_delta=delta,
                concept_accuracy=utility_test.concept_accuracy,
                baseline_accuracy=utility_test.baseline_accuracy,
                sample_count=utility_test.sample_count,
                reasoning=(
                    f"Predictive utility too low: Δ={delta:.3f} "
                    f"< minimum gain {self._min_utility_gain}"
                ),
            )

        return ConsolidationResult(
            decision=ConsolidationDecision.ACCEPT,
            hypothesis_id=hypothesis.hypothesis_id,
            utility_delta=delta,
            concept_accuracy=utility_test.concept_accuracy,
            baseline_accuracy=utility_test.baseline_accuracy,
            sample_count=utility_test.sample_count,
            reasoning=(
                f"Predictive utility confirmed: Δ={delta:.3f} "
                f"(concept={utility_test.concept_accuracy:.2f} "
                f"vs baseline={utility_test.baseline_accuracy:.2f})"
            ),
        )

    # ── Heterogeneity Detection (split signal) ────────────────────────

    def detect_heterogeneity(
        self,
        concept_id: str,
        member_outcomes: dict[str, list[bool]],
    ) -> ConceptRefinementSignal | None:
        """Detect whether a concept's members show divergent prediction profiles.

        If members split into subgroups with significantly different
        prediction accuracy, emit a ConceptRefinementSignal.

        Args:
            concept_id: The concept to check.
            member_outcomes: entity_id → list of prediction outcomes.

        Returns:
            ConceptRefinementSignal if heterogeneity detected, else None.
        """
        if len(member_outcomes) < 2:
            return None

        # Compute per-entity accuracy
        accuracies: dict[str, float] = {}
        for entity_id, outcomes in member_outcomes.items():
            if outcomes:
                accuracies[entity_id] = sum(outcomes) / len(outcomes)
            else:
                accuracies[entity_id] = 0.5

        # Sort by accuracy and try median split
        sorted_members = sorted(accuracies.items(), key=lambda x: x[1])
        mid = len(sorted_members) // 2
        group_a = [m[0] for m in sorted_members[:mid]]
        group_b = [m[0] for m in sorted_members[mid:]]

        acc_a = sum(accuracies[m] for m in group_a) / max(len(group_a), 1)
        acc_b = sum(accuracies[m] for m in group_b) / max(len(group_b), 1)

        divergence = abs(acc_a - acc_b)

        if divergence >= self._heterogeneity_threshold:
            return ConceptRefinementSignal(
                concept_id=concept_id,
                subgroup_a=group_a,
                subgroup_b=group_b,
                accuracy_a=acc_a,
                accuracy_b=acc_b,
                divergence=divergence,
            )

        return None

    # ── Degradation Detection ─────────────────────────────────────────

    def detect_degradation(
        self,
        concept_id: str,
        confidence: float,
        prediction_accuracy: float,
        recent_failure_rate: float = 0.0,
    ) -> ConceptDegradationSignal | None:
        """Detect whether a concept has degraded.

        Low confidence → refinement/retirement signal, NOT deletion.

        Returns:
            ConceptDegradationSignal if degraded, else None.
        """
        if confidence >= self._degradation_confidence:
            return None

        # Determine recommendation
        if recent_failure_rate > 0.7:
            recommendation = "retire"
        elif recent_failure_rate > 0.4:
            recommendation = "refine"
        else:
            recommendation = "merge"

        return ConceptDegradationSignal(
            concept_id=concept_id,
            current_confidence=confidence,
            prediction_accuracy=prediction_accuracy,
            recent_failure_rate=recent_failure_rate,
            recommendation=recommendation,
        )
