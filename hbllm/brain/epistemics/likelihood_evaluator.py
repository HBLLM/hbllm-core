"""Epistemic Likelihood Evaluator — evaluates proposition-specific evidence likelihoods.

Separates general evidence reliability (evaluated by PerceptualEvidenceEvaluator)
from the semantic/statistical likelihood of evidence given a specific hypothesis H
versus its negation ¬H:

- P(E | H)    = Probability of observing evidence E if hypothesis H is true
- P(E | ¬H)   = Probability of observing evidence E if hypothesis H is false
- LR          = P(E | H) / P(E | ¬H)  (Likelihood Ratio)
- LR_eff      = LR^novelty  (dependence-corrected effective LR)

Architecture::

    EvidenceNode + EvidenceAssessment + BeliefNode
                        │
                        ▼
    EpistemicLikelihoodEvaluator.evaluate_likelihood(belief, evidence, assessment)
                        │
                        ├── raw LR from proposition semantics
                        ├── novelty from TemporalEvidenceModel (if available)
                        ├── effective LR = raw_LR^novelty
                        └── information_gain = |log2(LR_eff)| × novelty
                        │
                        ▼
    PropositionLikelihood (P(E|H), P(E|¬H), raw_LR, effective_LR, status)
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from hbllm.hcir.graph import BeliefNode, CognitiveGraph, HCIREdgeType
from hbllm.hcir.types import EvidenceAssessment, PropositionLikelihood

logger = logging.getLogger(__name__)


class EpistemicLikelihoodEvaluator:
    """Evaluates proposition-specific likelihoods for a candidate belief and evidence.

    When a TemporalEvidenceModel is provided, applies dependence correction:
    LR_effective = LR^novelty, where novelty ∈ [0.0, 1.0] corrects for
    correlated observations from the same sensor pipeline.
    """

    def __init__(
        self,
        graph: CognitiveGraph | None = None,
        llm: Any | None = None,
        temporal_model: Any | None = None,
    ) -> None:
        self._graph = graph
        self._llm = llm
        self._temporal_model = temporal_model

    def evaluate_likelihood(
        self,
        belief: BeliefNode,
        evidence: Any,
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
            PropositionLikelihood containing P(E|H), P(E|¬H), raw LR,
            effective LR (after dependence correction), and categorization.
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

        # 3. Calculate raw Likelihood Ratio (LR)
        raw_lr = p_e_given_h / max(1e-6, p_e_given_not_h)

        # 4. Apply dependence correction via TemporalEvidenceModel
        novelty_discount = 1.0
        if self._temporal_model is not None:
            novelty_assessment = self._temporal_model.assess(evidence, belief)
            novelty_discount = novelty_assessment.composite_novelty

            # If already incorporated for this proposition, force redundant
            if novelty_assessment.already_incorporated:
                return PropositionLikelihood(
                    belief_id=belief.id,
                    evidence_id=evidence.id,
                    p_e_given_h=float(p_e_given_h),
                    p_e_given_not_h=float(p_e_given_not_h),
                    likelihood_ratio=float(raw_lr),
                    raw_likelihood_ratio=float(raw_lr),
                    effective_likelihood_ratio=1.0,
                    novelty_discount=0.0,
                    status="redundant",
                )

        # 5. Compute effective LR: LR_effective = LR^novelty
        if novelty_discount <= 0.0 or raw_lr <= 0.0:
            effective_lr = 1.0
        elif novelty_discount >= 1.0:
            effective_lr = raw_lr
        else:
            effective_lr = math.pow(raw_lr, novelty_discount)

        # 6. Calculate information gain: |log2(LR_eff)| × novelty
        if effective_lr > 0.0 and effective_lr != 1.0:
            information_gain = abs(math.log2(effective_lr)) * novelty_discount
        else:
            information_gain = 0.0

        # 7. Determine novelty threshold for redundancy
        novelty_threshold = 0.05
        if self._temporal_model is not None:
            novelty_threshold = self._temporal_model.policy.novelty_threshold

        # 8. Categorize evidence incorporation decision
        if reliability < 0.45:
            status = "insufficient"
        elif novelty_discount < novelty_threshold:
            status = "redundant"
        elif 0.95 <= effective_lr <= 1.05:
            status = "redundant"
        elif effective_lr < 0.35 and belief.uncertainty.confidence > 0.65:
            status = "contradictory"
        else:
            status = "informative"

        prop_likelihood = PropositionLikelihood(
            belief_id=belief.id,
            evidence_id=evidence.id,
            p_e_given_h=float(p_e_given_h),
            p_e_given_not_h=float(p_e_given_not_h),
            likelihood_ratio=float(raw_lr),
            raw_likelihood_ratio=float(raw_lr),
            effective_likelihood_ratio=float(effective_lr),
            novelty_discount=float(novelty_discount),
            status=status,
        )

        logger.debug(
            "Proposition likelihood for belief=%s, evidence=%s: "
            "P(E|H)=%.2f, P(E|¬H)=%.2f, raw_LR=%.2f, eff_LR=%.2f, "
            "novelty=%.3f, IG=%.3f, status=%s",
            belief.id,
            evidence.id,
            p_e_given_h,
            p_e_given_not_h,
            raw_lr,
            effective_lr,
            novelty_discount,
            information_gain,
            status,
        )

        return prop_likelihood

    def _infer_direction(self, claim: str, evidence: Any) -> str:
        """Infer whether evidence supports or contradicts the belief claim."""
        # 1. Check if an explicit edge exists in the HCIR graph
        if self._graph is not None:
            for edge in self._graph.edges_from(evidence.id):
                if edge.edge_type == HCIREdgeType.WEAKENS and any(
                    t == claim or self._graph.get_node(t) is not None for t in edge.targets
                ):
                    return "contradicting"
                elif edge.edge_type == HCIREdgeType.STRENGTHENS and any(
                    t == claim or self._graph.get_node(t) is not None for t in edge.targets
                ):
                    return "supporting"

        claim_norm = claim.lower()

        # 2. Check for semantic absence vs presence opposition
        absence_terms = {
            "empty",
            "quiet",
            "nobody",
            "vacant",
            "silent",
            "dark",
            "no person",
            "clear",
            "unoccupied",
        }
        presence_terms = {
            "person",
            "human",
            "speech",
            "talking",
            "voice",
            "applause",
            "sound",
            "crowd",
            "noise",
            "active",
            "asr",
            "speaker",
            "conversation",
            "movement",
            "occupant",
            "occupied",
            "utterance",
        }

        has_absence_claim = any(term in claim_norm for term in absence_terms)
        has_presence_claim = any(term in claim_norm for term in presence_terms)

        prop_text = ""
        if hasattr(evidence, "proposition") and evidence.proposition is not None:
            prop_text = f"{evidence.proposition.subject} {evidence.proposition.predicate} {evidence.proposition.object_value}"

        method_text = getattr(evidence, "methodology", "") or ""
        modality_text = getattr(evidence, "modality", "") or ""

        evidence_text = (
            " ".join(
                [str(cand.get("label", "")) for cand in getattr(evidence, "candidates", [])]
            ).lower()
            + " "
            + prop_text.lower()
            + " "
            + method_text.lower()
            + " "
            + modality_text.lower()
        )

        evidence_has_presence = any(
            term in evidence_text for term in presence_terms
        ) or modality_text in ("audio", "visual")
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
        for cand in getattr(evidence, "candidates", []):
            label = str(cand.get("label", "")).lower()
            if not label:
                continue

            words = [w for w in re.findall(r"\w+", label) if len(w) > 3]
            for w in words:
                if w in claim_norm:
                    return "supporting"

        # Check method / description
        method = method_text.lower()
        if any(neg in method for neg in ["negation", "contradicts", "conflict", "false"]):
            return "contradicting"

        return "supporting"
