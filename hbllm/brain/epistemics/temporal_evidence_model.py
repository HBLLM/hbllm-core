"""Temporal Evidence Model — multidimensional novelty, identity, and dependence correction.

This module answers four independent questions about incoming evidence:

1. **Identity**: "Have I already incorporated this exact evidence for this proposition?"
2. **Novelty**: "How much new information does this evidence contribute?"
   - Temporal novelty: n_t = 1 − 2^(−Δt / T½)
   - Semantic novelty: 1 − Jaccard(current_tags, previous_tags)
   - State-change novelty: 1.0 if transition detected, 0.0 if persistent
3. **Temporal Pattern**: Is this PERSISTENT, TRANSITION, TRANSIENT, or PERIODIC?
4. **Dependence**: Correlated observations from the same sensor pipeline
   are not independent evidence. The composite novelty score is the
   dependence-correction exponent applied to the Likelihood Ratio:
   LR_effective = LR^novelty

Epistemic Pipeline Invariant::

    Perception produces evidence.
    Temporal modeling determines evidence novelty and dependence.
    Likelihood evaluation determines proposition discrimination.
    Belief management performs state transitions.
    HCIR owns the resulting state and immutable history.
    Replay reconstructs the same state from the same causal inputs.

Replay Contract:
    This model is deterministic and effectively stateless with respect
    to authoritative cognition. All temporal calculations during replay
    are derived from ``event history + EpistemicRuntimeConfig``, never
    from mutable caches. ``last_incorporated_at`` on EvidenceNode is a
    convenience cache, not authoritative replay state.

Architecture::

    EvidenceNode
         │
         ▼
    TemporalEvidenceModel
         │
         ├── identity check (idempotent)
         ├── temporal novelty
         ├── semantic novelty (Jaccard on tags/labels)
         ├── state-change novelty (label transition detection)
         ├── composite novelty (via NoveltyPolicy)
         └── temporal pattern classification
         │
         ▼
    NoveltyAssessment
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from hbllm.hcir.graph import BeliefNode, CognitiveGraph, EvidenceNode, PerceptualEvidenceNode
from hbllm.hcir.types import EvidenceTemporalPattern, NoveltyPolicy

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data Types
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class NoveltyAssessment:
    """Result of multidimensional novelty computation for a piece of evidence.

    Used by EpistemicLikelihoodEvaluator to compute the dependence-corrected
    effective Likelihood Ratio: LR_effective = LR^composite_novelty.
    """

    temporal_novelty: float = 1.0
    semantic_novelty: float = 1.0
    state_change_novelty: float = 0.0
    composite_novelty: float = 1.0
    temporal_pattern: EvidenceTemporalPattern = EvidenceTemporalPattern.UNKNOWN
    already_incorporated: bool = False
    temporal_delta_seconds: float = 0.0
    semantic_delta: float = 0.0


@dataclass
class StateChangeAssessment:
    """Result of state-change detection between consecutive evidence."""

    is_transition: bool = False
    previous_state: str = ""
    current_state: str = ""
    change_magnitude: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# State Change Detection Protocol
# ═══════════════════════════════════════════════════════════════════════════


@runtime_checkable
class StateChangeDetector(Protocol):
    """Protocol for detecting state transitions between evidence observations."""

    def detect(
        self,
        previous: EvidenceNode | None,
        current: EvidenceNode,
    ) -> StateChangeAssessment:
        """Detect whether a state transition occurred between observations."""
        ...

    def get_evidence_label(self, evidence: EvidenceNode) -> str:
        """Extract primary label/state description from evidence."""
        ...


class LabelStateChangeDetector:
    """Detects state transitions via direct label/tag comparison.

    Initial implementation: deterministic, cheap, inspectable, replayable.
    Future implementations (ProviderTransitionDetector, SNNTransitionDetector)
    can be swapped via the StateChangeDetector protocol.
    """

    def detect(
        self,
        previous: EvidenceNode | None,
        current: EvidenceNode,
    ) -> StateChangeAssessment:
        """Detect state change by comparing evidence labels and candidate tags."""
        curr_state = self.get_evidence_label(current).lower().strip()
        if previous is None:
            return StateChangeAssessment(
                is_transition=False,
                current_state=curr_state,
                change_magnitude=0.0,
            )

        prev_tags = self._extract_tags(previous)
        curr_tags = self._extract_tags(current)

        prev_state = self.get_evidence_label(previous).lower().strip()

        # Check for label change
        label_changed = prev_state != curr_state

        # Jaccard distance on tags
        if prev_tags or curr_tags:
            intersection = prev_tags & curr_tags
            union = prev_tags | curr_tags
            jaccard_sim = len(intersection) / max(len(union), 1)
            change_magnitude = 1.0 - jaccard_sim
        else:
            change_magnitude = 1.0 if label_changed else 0.0

        is_transition = label_changed or change_magnitude > 0.3

        return StateChangeAssessment(
            is_transition=is_transition,
            previous_state=prev_state,
            current_state=curr_state,
            change_magnitude=change_magnitude,
        )

    @classmethod
    def get_evidence_label(cls, evidence: EvidenceNode | PerceptualEvidenceNode | Any) -> str:
        """Extract a primary label/state description from an EvidenceNode or PerceptualEvidenceNode."""
        if hasattr(evidence, "proposition") and evidence.proposition is not None:
            return f"{evidence.proposition.subject} {evidence.proposition.predicate} {evidence.proposition.object_value}"
        if getattr(evidence, "candidates", None):
            top_cand = evidence.candidates[0]
            if isinstance(top_cand, dict) and "label" in top_cand and top_cand["label"]:
                return str(top_cand["label"])
        if getattr(evidence, "tags", None):
            return " ".join(evidence.tags)
        if getattr(evidence, "methodology", None):
            return evidence.methodology
        if getattr(evidence, "claim_id", None):
            return evidence.claim_id
        return getattr(evidence, "id", "")

    @classmethod
    def _extract_tags(cls, evidence: EvidenceNode | PerceptualEvidenceNode | Any) -> set[str]:
        """Extract normalized tags from evidence candidates, tags, and label."""
        tags: set[str] = set(getattr(evidence, "tags", []))
        label = cls.get_evidence_label(evidence).lower().strip()
        if label:
            tags.update(w for w in re.findall(r"\w+", label) if len(w) > 2)

        for cand in getattr(evidence, "candidates", []):
            cand_label = str(cand.get("label", "")).lower()
            if cand_label:
                tags.update(w for w in re.findall(r"\w+", cand_label) if len(w) > 2)

        return tags


# ═══════════════════════════════════════════════════════════════════════════
# Temporal Evidence Model
# ═══════════════════════════════════════════════════════════════════════════


class TemporalEvidenceModel:
    """Multidimensional novelty assessment and dependence correction.

    Determines how much new epistemic information an incoming piece of
    evidence contributes relative to previously incorporated evidence
    for a given belief/proposition.

    Internal state is reconstructible from event history + configuration
    (no mutable caches as source of truth). The sliding window of recently
    incorporated evidence per belief is deterministically derivable from
    the HCIR graph.
    """

    def __init__(
        self,
        graph: CognitiveGraph,
        policy: NoveltyPolicy | None = None,
        state_change_detector: StateChangeDetector | None = None,
    ) -> None:
        self._graph = graph
        self._policy = policy or NoveltyPolicy()
        self._state_detector = state_change_detector or LabelStateChangeDetector()

    @property
    def policy(self) -> NoveltyPolicy:
        """Current novelty policy."""
        return self._policy

    def assess(
        self,
        evidence: EvidenceNode,
        belief: BeliefNode,
    ) -> NoveltyAssessment:
        """Compute full multidimensional novelty assessment.

        Args:
            evidence: Incoming evidence node to assess.
            belief: Target belief/proposition that this evidence might revise.

        Returns:
            NoveltyAssessment with temporal, semantic, state-change, and
            composite novelty scores plus temporal pattern classification.
        """
        # 1. Identity check (idempotency)
        if self.check_identity(evidence, belief):
            return NoveltyAssessment(
                temporal_novelty=0.0,
                semantic_novelty=0.0,
                state_change_novelty=0.0,
                composite_novelty=0.0,
                already_incorporated=True,
                temporal_pattern=EvidenceTemporalPattern.UNKNOWN,
            )

        # 2. Gather recent incorporated evidence for this belief
        recent = self._get_recent_evidence_for_belief(belief.id)

        # 3. Compute temporal novelty
        delta_t = self._compute_temporal_delta(evidence, recent)
        temporal_novelty = self._policy.compute_temporal_novelty(delta_t)

        # 4. Compute semantic novelty (Jaccard distance)
        semantic_delta, semantic_novelty = self._compute_semantic_novelty(evidence, recent)

        # 5. Detect state change
        most_recent = recent[0] if recent else None
        state_assessment = self._state_detector.detect(most_recent, evidence)
        state_change_novelty = (
            state_assessment.change_magnitude if state_assessment.is_transition else 0.0
        )

        # 6. Classify temporal pattern
        pattern = self.classify_pattern(evidence, recent)

        # 7. Compute composite novelty via policy
        composite = self._compute_composite(
            temporal_novelty,
            semantic_novelty,
            state_change_novelty,
        )

        assessment = NoveltyAssessment(
            temporal_novelty=temporal_novelty,
            semantic_novelty=semantic_novelty,
            state_change_novelty=state_change_novelty,
            composite_novelty=composite,
            temporal_pattern=pattern,
            already_incorporated=False,
            temporal_delta_seconds=delta_t,
            semantic_delta=semantic_delta,
        )

        logger.debug(
            "Novelty assessment for evidence=%s, belief=%s: "
            "temporal=%.3f, semantic=%.3f, state_change=%.3f, composite=%.3f, pattern=%s",
            evidence.id,
            belief.id,
            temporal_novelty,
            semantic_novelty,
            state_change_novelty,
            composite,
            pattern,
        )

        return assessment

    def check_identity(self, evidence: EvidenceNode, belief: BeliefNode) -> bool:
        """Check if this evidence has already been incorporated for this belief.

        Idempotency key: (evidence_id, belief_id).
        Same evidence may legitimately affect multiple independent propositions.
        """
        return belief.id in evidence.incorporated_transitions

    def classify_pattern(
        self,
        evidence: EvidenceNode,
        recent_history: list[EvidenceNode],
    ) -> EvidenceTemporalPattern:
        """Classify the temporal pattern of evidence relative to recent history.

        - PERSISTENT: Same label/state repeated consistently (>3 consecutive similar).
        - TRANSITION: Significant state change from previous observation.
        - TRANSIENT: Single instantaneous event (knock, flash).
        - PERIODIC: Recurring pattern with detectable periodicity.
        - UNKNOWN: Insufficient history to classify.
        """
        if len(recent_history) < 2:
            return EvidenceTemporalPattern.UNKNOWN

        # Check for transition (most recent differs)
        most_recent = recent_history[0]
        state_change = self._state_detector.detect(most_recent, evidence)
        if state_change.is_transition:
            return EvidenceTemporalPattern.TRANSITION

        # Check for persistent (>= 3 consecutive similar labels)
        current_label_norm = self._state_detector.get_evidence_label(evidence).lower().strip()
        similar_count = 0
        for prev in recent_history:
            if self._state_detector.get_evidence_label(prev).lower().strip() == current_label_norm:
                similar_count += 1
            else:
                break

        if similar_count >= 3:
            return EvidenceTemporalPattern.PERSISTENT

        # Check for transient characteristics
        duration = getattr(evidence, "duration", 0.0)
        if isinstance(duration, (int, float)) and duration > 0.0 and duration < 0.5:
            return EvidenceTemporalPattern.TRANSIENT

        # Check for periodicity (simplified: look for alternating labels)
        if len(recent_history) >= 4:
            labels = [self._state_detector.get_evidence_label(evidence).lower().strip()] + [
                self._state_detector.get_evidence_label(e).lower().strip()
                for e in recent_history[:5]
            ]
            unique = set(labels)
            if len(unique) <= 2 and len(labels) >= 4:
                # Simple heuristic: alternating between ≤2 labels
                return EvidenceTemporalPattern.PERIODIC

        return EvidenceTemporalPattern.UNKNOWN

    # ───────────────────────────────────────────────────────────────────
    # Internal Methods
    # ───────────────────────────────────────────────────────────────────

    def _get_recent_evidence_for_belief(
        self,
        belief_id: str,
        max_count: int = 10,
    ) -> list[EvidenceNode | PerceptualEvidenceNode | Any]:
        """Get recently incorporated evidence for a specific belief.

        Reconstructed deterministically from the HCIR graph by scanning
        EvidenceNodes whose incorporated_transitions contain the belief_id.
        """
        results: list[Any] = []
        for node in self._graph.all_nodes():
            if not isinstance(node, (EvidenceNode, PerceptualEvidenceNode)):
                continue
            if belief_id in getattr(node, "incorporated_transitions", {}):
                results.append(node)

        # Sort by last_incorporated_at descending (most recent first)
        results.sort(key=lambda e: getattr(e, "last_incorporated_at", 0.0), reverse=True)
        return results[:max_count]

    def _compute_temporal_delta(
        self,
        evidence: EvidenceNode | PerceptualEvidenceNode | Any,
        recent: list[Any],
    ) -> float:
        """Compute time delta from the most recently incorporated evidence."""
        if not recent:
            return float("inf")  # First evidence → full novelty

        most_recent_time = getattr(recent[0], "last_incorporated_at", 0.0)
        if most_recent_time <= 0.0:
            return float("inf")

        current_time = getattr(getattr(evidence, "provenance", None), "timestamp", 0.0)
        if current_time <= 0.0:
            import time

            current_time = time.time()

        return max(0.0, current_time - most_recent_time)

    def _compute_semantic_novelty(
        self,
        evidence: EvidenceNode,
        recent: list[EvidenceNode],
    ) -> tuple[float, float]:
        """Compute semantic novelty using Jaccard distance on tags.

        Returns:
            (jaccard_distance, semantic_novelty_score)
        """
        if not recent:
            return (1.0, 1.0)  # First evidence → fully novel

        current_tags = LabelStateChangeDetector._extract_tags(evidence)
        if not current_tags:
            return (0.5, 0.5)  # Unknown → moderate novelty

        # Compare against the most recent incorporated evidence
        prev_tags = LabelStateChangeDetector._extract_tags(recent[0])
        if not prev_tags:
            return (0.5, 0.5)

        intersection = current_tags & prev_tags
        union = current_tags | prev_tags
        jaccard_similarity = len(intersection) / max(len(union), 1)
        jaccard_distance = 1.0 - jaccard_similarity

        return (jaccard_distance, jaccard_distance)

    def _compute_composite(
        self,
        temporal: float,
        semantic: float,
        state_change: float,
    ) -> float:
        """Compute composite novelty via the configured NoveltyPolicy.

        Default policy: state transitions override temporal × semantic.
        """
        if self._policy.state_change_override and state_change > 0.5:
            return min(1.0, state_change * self._policy.state_change_weight)

        weighted = temporal * self._policy.temporal_weight + semantic * self._policy.semantic_weight
        total_weight = self._policy.temporal_weight + self._policy.semantic_weight
        if total_weight > 0:
            weighted /= total_weight

        return min(1.0, max(0.0, weighted))


__all__ = [
    "LabelStateChangeDetector",
    "NoveltyAssessment",
    "StateChangeAssessment",
    "StateChangeDetector",
    "TemporalEvidenceModel",
]
