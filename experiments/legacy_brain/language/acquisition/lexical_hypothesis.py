"""Epistemic Lexical Hypothesis Data Structures for A17 Grounded Language Learning.

Defines the epistemic candidate types, evidence source categories,
competing hypothesis sets, and provenance-backed LexicalSense representations.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LexicalTargetType(str, Enum):
    """Semantic target categories for lexical hypotheses."""

    CONCEPT = "concept"  # Category / Noun prototype (e.g. cylinder, cup)
    INDIVIDUAL = "individual"  # Specific physical entity instance (e.g. instance_42)
    PROPERTY = "property"  # Attribute / Adjective (e.g. color=red, size=large)
    RELATION = "relation"  # Spatial / Prepositional link (e.g. ON, INSIDE, UNDER)
    ACTION = "action"  # Agentive state transition (e.g. PUSH, LIFT, DROP)
    EVENT = "event"  # Non-agentive state transition (e.g. FALL, COLLISION, RAIN)


class LexicalCandidateStatus(str, Enum):
    """Lifecycle status of a lexical candidate hypothesis."""

    UNKNOWN = "unknown"
    HYPOTHESIS = "hypothesis"
    TENTATIVE = "tentative"
    GROUNDED = "grounded"
    CONTRADICTED = "contradicted"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"


class EvidenceSourceType(str, Enum):
    """Epistemic source types with calibrated epistemic weights."""

    OSTENSIVE_POSITIVE = "ostensive_positive"  # Weight: 1.0 (Teacher: "This is a cup")
    OSTENSIVE_NEGATIVE = "ostensive_negative"  # Weight: 1.2 (Teacher: "No, that is not an apple")
    CROSS_SITUATIONAL = "cross_situational"  # Weight: 0.4 (Incidental multi-scene co-occurrence)
    CONTRASTIVE = "contrastive"  # Weight: 0.8 (Explicit contrast cup != bowl)
    PREDICTIVE = "predictive"  # Weight: 0.9 (Forward prediction verified)
    ACTION_TRANSITION = "action_transition"  # Weight: 0.85 (Temporal delta s_t0 -> s_t1)
    SPATIAL_RELATIONAL = "spatial_relational"  # Weight: 0.75 (Topological graph alignment)
    CONTEXTUAL = "contextual"  # Weight: 0.3 (Discourse context)


# Source weight mapping for deterministic scoring
SOURCE_WEIGHTS: dict[EvidenceSourceType, float] = {
    EvidenceSourceType.OSTENSIVE_POSITIVE: 1.0,
    EvidenceSourceType.OSTENSIVE_NEGATIVE: 1.2,
    EvidenceSourceType.CROSS_SITUATIONAL: 0.4,
    EvidenceSourceType.CONTRASTIVE: 0.8,
    EvidenceSourceType.PREDICTIVE: 0.9,
    EvidenceSourceType.ACTION_TRANSITION: 0.85,
    EvidenceSourceType.SPATIAL_RELATIONAL: 0.75,
    EvidenceSourceType.CONTEXTUAL: 0.3,
}


@dataclass(frozen=True)
class LexicalEvidence:
    """An observation event providing evidence for/against lexical hypotheses."""

    id: str = field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:8]}")
    source_type: EvidenceSourceType = EvidenceSourceType.CROSS_SITUATIONAL
    token: str = ""
    language: str = "en"
    target_type: LexicalTargetType = LexicalTargetType.CONCEPT
    target_value: Any = None
    is_positive: bool = True
    context_entities: list[str] = field(default_factory=list)
    state_delta: dict[str, Any] = field(default_factory=dict)
    speaker: str = "teacher"
    timestamp: float = 0.0

    @property
    def epistemic_weight(self) -> float:
        return SOURCE_WEIGHTS.get(self.source_type, 0.4)


TARGET_TYPE_PRIORITY: dict[LexicalTargetType, int] = {
    LexicalTargetType.CONCEPT: 1,
    LexicalTargetType.PROPERTY: 2,
    LexicalTargetType.ACTION: 3,
    LexicalTargetType.RELATION: 4,
    LexicalTargetType.EVENT: 5,
    LexicalTargetType.INDIVIDUAL: 6,
}


@dataclass
class LexicalCandidate:
    """A single hypothesis within a competing hypothesis set."""

    target_type: LexicalTargetType
    target_id: str  # Concept ID, Entity ID, Property name, Action name, etc.
    target_value: Any = None  # Specific prototype, property value, or state transition spec
    support_weight: float = 0.0
    contradiction_weight: float = 0.0
    predictive_score: float = 0.5  # A15 predictive utility baseline
    evidence_ids: list[str] = field(default_factory=list)
    contradiction_ids: list[str] = field(default_factory=list)
    first_observed: float = 0.0
    last_updated: float = 0.0
    status: LexicalCandidateStatus = LexicalCandidateStatus.HYPOTHESIS

    @property
    def raw_score(self) -> float:
        """Unclamped score for precise ranking and margin calculation."""
        pred_delta = (self.predictive_score - 0.5) * 0.5
        return 0.1 + self.support_weight - (1.5 * self.contradiction_weight) + pred_delta

    @property
    def total_score(self) -> float:
        """Deterministic bounded composite score in [0.0, 1.0]."""
        return max(0.0, min(1.0, self.raw_score))

    @property
    def confidence(self) -> float:
        """Quantitative confidence metric bounded in [0.0, 1.0]."""
        total_ev = len(self.evidence_ids) + len(self.contradiction_ids)
        if total_ev == 0:
            return 0.0
        ratio = len(self.evidence_ids) / total_ev
        return max(0.0, min(1.0, ratio * min(1.0, self.support_weight / 1.2)))


@dataclass
class LexicalHypothesisSet:
    """A set of competing lexical candidates for a single token in a language."""

    token: str
    language: str
    candidates: list[LexicalCandidate] = field(default_factory=list)

    def get_candidate(
        self, target_type: LexicalTargetType, target_id: str
    ) -> LexicalCandidate | None:
        for c in self.candidates:
            if c.target_type == target_type and c.target_id == target_id:
                return c
        return None

    def add_or_get_candidate(
        self,
        target_type: LexicalTargetType,
        target_id: str,
        target_value: Any = None,
        timestamp: float = 0.0,
    ) -> LexicalCandidate:
        existing = self.get_candidate(target_type, target_id)
        if existing is not None:
            return existing
        cand = LexicalCandidate(
            target_type=target_type,
            target_id=target_id,
            target_value=target_value,
            first_observed=timestamp,
            last_updated=timestamp,
        )
        self.candidates.append(cand)
        return cand

    def ranked_candidates(self) -> list[LexicalCandidate]:
        """Deterministic ranking with strict tie-breaking order:

        1. Raw composite score (desc)
        2. Predictive score (desc)
        3. Support weight (desc)
        4. Target type priority (asc: CONCEPT > PROPERTY > ACTION > RELATION > EVENT > INDIVIDUAL)
        5. Contradiction count (asc)
        6. Earliest observed timestamp (asc)
        7. Canonical target ID (asc)
        """
        return sorted(
            self.candidates,
            key=lambda c: (
                -c.raw_score,
                -c.predictive_score,
                -c.support_weight,
                TARGET_TYPE_PRIORITY.get(c.target_type, 10),
                len(c.contradiction_ids),
                c.first_observed,
                c.target_id,
            ),
        )

    @property
    def winner(self) -> LexicalCandidate | None:
        ranked = self.ranked_candidates()
        return ranked[0] if ranked else None

    @property
    def runner_up(self) -> LexicalCandidate | None:
        ranked = self.ranked_candidates()
        return ranked[1] if len(ranked) > 1 else None

    @property
    def margin_of_victory(self) -> float:
        w = self.winner
        r = self.runner_up
        if not w:
            return 0.0
        if not r:
            return max(0.0, w.raw_score)
        return max(0.0, w.raw_score - r.raw_score)

    @property
    def is_ambiguous(self) -> bool:
        """True if top candidates are closely contested."""
        if not self.winner or not self.runner_up:
            return False
        return self.margin_of_victory < 0.20 and self.winner.total_score > 0.3


@dataclass
class LexicalSense:
    """A committed, authoritative grounded lexical entry in HCIR with full provenance."""

    id: str = field(default_factory=lambda: f"sense_{uuid.uuid4().hex[:8]}")
    token: str = ""
    language: str = "en"
    target_type: LexicalTargetType = LexicalTargetType.CONCEPT
    target_id: str = ""
    target_value: Any = None
    supporting_evidence_ids: list[str] = field(default_factory=list)
    contradicting_evidence_ids: list[str] = field(default_factory=list)
    contrast_ids: list[str] = field(default_factory=list)
    predictive_score: float = 0.5
    comprehension_confidence: float = 0.0
    generation_confidence: float = 0.0
    status: LexicalCandidateStatus = LexicalCandidateStatus.TENTATIVE
    first_observed_event: float = 0.0
    last_updated_event: float = 0.0
