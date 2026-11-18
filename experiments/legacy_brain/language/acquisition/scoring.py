"""Deterministic Bounded Scoring and Epistemic Status Resolution for A17.

Implements the explicit scoring function and deterministic tie-breaking logic
for evaluating competing lexical hypotheses.
"""

from __future__ import annotations

from hbllm.brain.language.acquisition.lexical_hypothesis import (
    LexicalCandidate,
    LexicalCandidateStatus,
    LexicalEvidence,
)


def apply_evidence_to_candidate(
    candidate: LexicalCandidate,
    evidence: LexicalEvidence,
) -> None:
    """Update candidate weights deterministically upon observing LexicalEvidence."""
    weight = evidence.epistemic_weight
    if evidence.is_positive:
        candidate.support_weight += weight
        if evidence.id not in candidate.evidence_ids:
            candidate.evidence_ids.append(evidence.id)
    else:
        candidate.contradiction_weight += weight
        if evidence.id not in candidate.contradiction_ids:
            candidate.contradiction_ids.append(evidence.id)

    candidate.last_updated = evidence.timestamp
    update_candidate_status(candidate)


def update_candidate_status(candidate: LexicalCandidate) -> None:
    """Derive epistemic candidate status from evidence weights and confidence."""
    # If contradictions decisively overwhelm support
    if round(candidate.contradiction_weight, 4) >= round(candidate.support_weight + 0.79, 4):
        candidate.status = LexicalCandidateStatus.CONTRADICTED
        return

    # If contradiction count is high and support is zero/negligible
    if len(candidate.contradiction_ids) >= 2 and candidate.support_weight < 0.3:
        candidate.status = LexicalCandidateStatus.REJECTED
        return

    # Fast map / initial hypothesis: 1 weak observation
    if len(candidate.evidence_ids) == 1 and candidate.support_weight < 0.6:
        candidate.status = LexicalCandidateStatus.HYPOTHESIS
        return

    # Tentative status: moderate support or moderate confidence
    if candidate.support_weight >= 0.6 and candidate.confidence < 0.70:
        candidate.status = LexicalCandidateStatus.TENTATIVE
        return

    # Grounded status: strong support and high confidence
    if candidate.support_weight >= 1.2 and candidate.confidence >= 0.70:
        candidate.status = LexicalCandidateStatus.GROUNDED
        return

    candidate.status = LexicalCandidateStatus.TENTATIVE
