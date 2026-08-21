"""Epistemic Likelihood Evaluator — evaluates proposition-specific evidence likelihoods.

Separates general evidence reliability (evaluated by PerceptualEvidenceEvaluator)
from the semantic/statistical likelihood of evidence given a specific hypothesis H
versus its negation ¬H:

- P(E | H)    = Probability of observing evidence E if hypothesis H is true
- P(E | ¬H)   = Probability of observing evidence E if hypothesis H is false
- LR          = P(E | H) / P(E | ¬H)  (Likelihood Ratio)

Architecture::

    EvidenceNode + EvidenceAssessment + BeliefNode
                        │
                        ▼
    EpistemicLikelihoodEvaluator.evaluate_likelihood(belief, evidence, assessment)
                        │
                        ▼
    PropositionLikelihood (P(E|H), P(E|¬H), LR, status)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from hbllm.hcir.graph import BeliefNode, CognitiveGraph, EvidenceNode, HCIREdgeType
from hbllm.hcir.types import EvidenceAssessment, PropositionLikelihood

logger = logging.getLogger(__name__)


class EpistemicLikelihoodEvaluator:
    """Evaluates proposition-specific likelihoods for a candidate belief and evidence."""

    def __init__(
        self,
        graph: CognitiveGraph | None = None,
        llm: Any | None = None,
    ) -> None:
        self._graph = graph
        self._llm = llm

    def evaluate_likelihood(
        self,
        belief: BeliefNode,
        evidence: EvidenceNode,
        assessment: EvidenceAssessment,
        direction: str = "auto",  # "auto" | "supporting" | "contradicting"
    ) -> PropositionLikelihood:
        """Compute proposition-specific P(E | H), P(E | ¬H), and Likelihood Ratio (LR).

        Args:
            belief: The candidate BeliefNode (hypothesis H).
            evidence: The sensory EvidenceNode E.
            assessment: General evidence quality assessment.
            direction: Explicit direction or auto-inferred from semantic alignment.

        Returns:
            PropositionLikelihood containing P(E|H), P(E|¬H), LR, and categorization.
        """
        reliability = float(assessment.reliability)

        # 1. Determine direction (supporting vs contradicting)
        resolved_direction = direction
        if resolved_direction == "auto":
            resolved_direction = self._infer_direction(belief.claim, evidence)

        # 2. Compute P(E | H) and P(E | ¬H) based on reliability and direction
        # High reliability evidence strongly discriminates between H and ¬H.
        if resolved_direction == "supporting":
            # If H is true, probability of seeing supporting evidence is high
            p_e_given_h = 0.5 + 0.45 * reliability
            # If H is false, probability of seeing supporting evidence is low (false positive rate)
            p_e_given_not_h = max(0.05, 0.5 - 0.45 * reliability)
        elif resolved_direction == "contradicting":
            # If H is true, probability of seeing contradicting evidence is low
            p_e_given_h = max(0.05, 0.5 - 0.45 * reliability)
            # If H is false, probability of seeing contradicting evidence is high
            p_e_given_not_h = 0.5 + 0.45 * reliability
        else:
            # Neutral / irrelevant
            p_e_given_h = 0.5
            p_e_given_not_h = 0.5

        # 3. Calculate Likelihood Ratio (LR)
        lr = p_e_given_h / max(1e-6, p_e_given_not_h)

        # 4. Categorize evidence incorporation decision
        # - insufficient: reliability too low or evidence too weak
        # - redundant: LR near 1.0 (no informational change)
        # - contradictory: strongly disconfirms a high-confidence belief
        # - informative: significant shift warranting Bayesian transition
        if reliability < 0.45:
            status = "insufficient"
        elif 0.95 <= lr <= 1.05:
            status = "redundant"
        elif lr < 0.35 and belief.uncertainty.confidence > 0.65:
            status = "contradictory"
        else:
            status = "informative"

        prop_likelihood = PropositionLikelihood(
            belief_id=belief.id,
            evidence_id=evidence.id,
            p_e_given_h=float(p_e_given_h),
            p_e_given_not_h=float(p_e_given_not_h),
            likelihood_ratio=float(lr),
            status=status,
        )

        logger.debug(
            "Proposition likelihood for belief=%s, evidence=%s: P(E|H)=%.2f, P(E|¬H)=%.2f, LR=%.2f, status=%s",
            belief.id,
            evidence.id,
            p_e_given_h,
            p_e_given_not_h,
            lr,
            status,
        )

        return prop_likelihood

    def _infer_direction(self, claim: str, evidence: EvidenceNode) -> str:
        """Infer whether evidence supports or contradicts the belief claim."""
        # 1. Check if an explicit edge exists in the HCIR graph
        if self._graph is not None:
            for edge in self._graph.edges_from(evidence.id):
                if edge.edge_type == HCIREdgeType.WEAKENS and any(t == claim or self._graph.get_node(t) is not None for t in edge.targets):
                    return "contradicting"
                elif edge.edge_type == HCIREdgeType.STRENGTHENS and any(t == claim or self._graph.get_node(t) is not None for t in edge.targets):
                    return "supporting"

        claim_norm = claim.lower()

        # 2. Check for semantic absence vs presence opposition
        absence_terms = {"empty", "quiet", "nobody", "vacant", "silent", "dark", "no person", "clear", "unoccupied"}
        presence_terms = {
            "person", "human", "speech", "talking", "voice", "applause", "sound",
            "crowd", "noise", "active", "asr", "speaker", "conversation",
            "movement", "occupant", "occupied", "utterance",
        }

        has_absence_claim = any(term in claim_norm for term in absence_terms)
        has_presence_claim = any(term in claim_norm for term in presence_terms)

        evidence_text = (
            " ".join([str(cand.get("label", "")) for cand in evidence.candidates]).lower()
            + " " + evidence.methodology.lower()
            + " " + evidence.modality.lower()
        )

        evidence_has_presence = any(term in evidence_text for term in presence_terms) or evidence.modality in ("audio", "visual")
        evidence_has_absence = any(term in evidence_text for term in absence_terms)

        if has_absence_claim and evidence_has_presence:
            return "contradicting"
        if has_presence_claim and evidence_has_absence:
            return "contradicting"
        if has_absence_claim and evidence_has_absence:
            return "supporting"
        if has_presence_claim and evidence_has_presence:
            return "supporting"

        # Check candidate labels in evidence for direct word matches
        for cand in evidence.candidates:
            label = str(cand.get("label", "")).lower()
            if not label:
                continue

            words = [w for w in re.findall(r"\w+", label) if len(w) > 3]
            for w in words:
                if w in claim_norm:
                    return "supporting"

        # Check method / description
        method = evidence.methodology.lower()
        if any(neg in method for neg in ["negation", "contradicts", "conflict", "false"]):
            return "contradicting"

        return "supporting"
