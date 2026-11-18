from __future__ import annotations

import time
import uuid

from hbllm.hcir.grounding.contracts.beliefs import BeliefState
from hbllm.hcir.grounding.contracts.claims import GroundingClaim
from hbllm.hcir.grounding.contracts.decisions import GroundingDecision
from hbllm.hcir.grounding.contracts.enums import DecisionOutcome
from hbllm.hcir.grounding.contracts.evidence import EvidenceRecord
from hbllm.hcir.grounding.epistemics.calibration import EvidenceCalibrator
from hbllm.hcir.grounding.epistemics.fusion import DependencyGroupEvidenceFusion, FusionResult
from hbllm.hcir.grounding.epistemics.revision import (
    BeliefRevisionEngine,
    ContradictionRecord,
)


def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class EpistemicEvaluator:
    """Evaluates GroundingClaims against evidence and world state to emit GroundingDecisions.

    Coordinates calibration, dependency-group Bayesian fusion, contradiction checks,
    and belief revision to determine whether a claim should be accepted, rejected, deferred,
    or flagged for active information gathering.
    """

    def __init__(
        self,
        calibrator: EvidenceCalibrator | None = None,
        fusion_engine: DependencyGroupEvidenceFusion | None = None,
        revision_engine: BeliefRevisionEngine | None = None,
        acceptance_threshold: float = 0.85,
        rejection_threshold: float = 0.20,
    ) -> None:
        self._calibrator = calibrator or EvidenceCalibrator()
        self._fusion = fusion_engine or DependencyGroupEvidenceFusion()
        self._revision = revision_engine or BeliefRevisionEngine()
        self._acceptance_threshold = acceptance_threshold
        self._rejection_threshold = rejection_threshold

    def evaluate_claim(
        self,
        claim: GroundingClaim,
        evidence_records: list[EvidenceRecord],
        active_beliefs: list[BeliefState] | None = None,
        prior_probability: float = 0.5,
    ) -> tuple[GroundingDecision, BeliefState | None]:
        """Evaluates a GroundingClaim proposition and determines its epistemic disposition."""
        beliefs = active_beliefs or []

        # 1. Calibrate raw evidence records
        calibrated_records = [self._calibrator.calibrate_record(r) for r in evidence_records]

        # 2. Dependency-group Bayesian fusion
        fusion: FusionResult = self._fusion.fuse(
            calibrated_records,
            prior_probability=prior_probability,
        )

        # 3. Contradiction detection
        contradictions: list[ContradictionRecord] = self._revision.detect_contradictions(
            claim,
            active_beliefs=beliefs,
        )

        severe_contradictions = [c for c in contradictions if c.severity >= 0.8]
        contradiction_ids = tuple(c.contradiction_id for c in contradictions)
        evidence_ids = tuple(r.evidence_id for r in calibrated_records)

        # 4. Epistemic Decision Disposition
        outcome: DecisionOutcome
        rationale: str
        tx_proposal_id: str | None = None
        resulting_belief: BeliefState | None = None

        if severe_contradictions:
            outcome = DecisionOutcome.REQUEST_EVIDENCE
            rationale = (
                f"Severe contradiction detected ({len(severe_contradictions)} conflict records); "
                "requesting sensory saccade or active verification"
            )
        elif fusion.posterior_probability >= self._acceptance_threshold:
            outcome = DecisionOutcome.ACCEPT
            tx_proposal_id = _gen_id("txprop")
            rationale = (
                f"Accepted with high posterior {fusion.posterior_probability:.3f} >= {self._acceptance_threshold} "
                f"across {fusion.independent_group_count} independent evidence group(s)"
            )
            resulting_belief = BeliefState(
                belief_id=_gen_id("belief"),
                subject_ref=claim.subject_ref,
                predicate_ref=claim.predicate_ref,
                object_ref=claim.object_ref,
                polarity=claim.polarity,
                posterior_probability=fusion.posterior_probability,
                evidence_strength=fusion.effective_evidence_strength,
                supporting_evidence_ids=evidence_ids,
                provenance_claims=(claim.claim_id,),
            )
        elif fusion.posterior_probability <= self._rejection_threshold:
            outcome = DecisionOutcome.REJECT
            rationale = f"Rejected with low posterior {fusion.posterior_probability:.3f} <= {self._rejection_threshold}"
        elif fusion.effective_evidence_strength < 0.3:
            outcome = DecisionOutcome.REQUEST_EVIDENCE
            rationale = (
                f"Uncertain claim with low evidence SNR ({fusion.effective_evidence_strength:.2f} < 0.30); "
                "requesting exploratory observation"
            )
        else:
            outcome = DecisionOutcome.DEFER
            rationale = (
                f"Deferred with intermediate posterior {fusion.posterior_probability:.3f}; "
                "held in working memory awaiting further corroboration"
            )

        decision = GroundingDecision(
            claim_id=claim.claim_id,
            outcome=outcome,
            posterior_belief=fusion.posterior_probability,
            calibrated_evidence_strength=fusion.effective_evidence_strength,
            contradictions=contradiction_ids,
            supporting_evidence_ids=evidence_ids,
            transaction_proposal_id=tx_proposal_id,
            decision_rationale=rationale,
            timestamp=time.time(),
        )

        return decision, resulting_belief
