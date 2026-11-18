from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hbllm.hcir.grounding.contracts.beliefs import BeliefState
from hbllm.hcir.grounding.contracts.claims import GroundingClaim
from hbllm.hcir.grounding.contracts.enums import Polarity
from hbllm.hcir.grounding.contracts.references import LiteralValueRef
from hbllm.hcir.grounding.epistemics.fusion import FusionResult


class ContradictionType(str, Enum):
    """Classification of an epistemic contradiction."""

    POLARITY_CONFLICT = "polarity_conflict"
    PROPERTY_EXCLUSIVITY = "property_exclusivity"
    SPATIAL_IMPOSSIBILITY = "spatial_impossibility"
    STATE_EXCLUSIVITY = "state_exclusivity"


class RevisionActionType(str, Enum):
    """The structured belief revision action taken."""

    CONFIRMED = "confirmed"
    WITHDRAWN = "withdrawn"
    REPLACED = "replaced"
    SUSPENDED = "suspended"
    PRESERVED = "preserved"


@dataclass(frozen=True)
class ContradictionRecord:
    """Audit record of a detected epistemic contradiction."""

    contradiction_id: str
    contradiction_type: ContradictionType
    conflicting_claim_ids: tuple[str, ...]
    conflicting_belief_ids: tuple[str, ...]
    severity: float
    rationale: str
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class RevisionResult:
    """The outcome of an explicit belief revision operation."""

    action_type: RevisionActionType
    target_belief_id: str
    updated_belief: BeliefState | None
    active_contradictions: tuple[ContradictionRecord, ...] = field(default_factory=tuple)
    rationale: str = ""
    timestamp: float = field(default_factory=time.time)


def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class BeliefRevisionEngine:
    """Structured belief revision engine for HCIR.

    Avoids naive in-place state overwrites. Evaluates spatial, temporal, and
    logical contradictions, and executes principled belief confirmation,
    replacement, withdrawal, or suspension with full historical preservation.
    """

    def __init__(
        self,
        confirmation_threshold: float = 0.85,
        withdrawal_threshold: float = 0.20,
        max_physical_velocity_mps: float = 50.0,
    ) -> None:
        self._confirmation_threshold = confirmation_threshold
        self._withdrawal_threshold = withdrawal_threshold
        self._max_physical_velocity_mps = max_physical_velocity_mps
        self._contradictions: list[ContradictionRecord] = []

    def detect_contradictions(
        self,
        claim: GroundingClaim,
        active_beliefs: list[BeliefState],
    ) -> list[ContradictionRecord]:
        """Classifies contradictions between a proposed claim and active beliefs."""
        detected: list[ContradictionRecord] = []

        for b in active_beliefs:
            # 1. Polarity Conflict: same subject, same predicate, same object, opposite polarity
            if (
                b.subject_ref.id == claim.subject_ref.id
                and b.predicate_ref == claim.predicate_ref
                and self._objects_match(b.object_ref, claim.object_ref)
                and b.polarity != claim.polarity
            ):
                rec = ContradictionRecord(
                    contradiction_id=_gen_id("contra_pol"),
                    contradiction_type=ContradictionType.POLARITY_CONFLICT,
                    conflicting_claim_ids=(claim.claim_id,),
                    conflicting_belief_ids=(b.belief_id,),
                    severity=0.9,
                    rationale=(
                        f"Polarity conflict on {claim.predicate_ref}: "
                        f"existing={b.polarity.value}, new={claim.polarity.value}"
                    ),
                )
                detected.append(rec)

            # 2. Mutually exclusive functional properties (e.g. single-valued attributes like COLOR)
            if (
                b.subject_ref.id == claim.subject_ref.id
                and b.predicate_ref == claim.predicate_ref
                and claim.predicate_ref in ("COLOR", "CATEGORY", "TYPE", "SHAPE")
                and not self._objects_match(b.object_ref, claim.object_ref)
                and b.polarity == Polarity.AFFIRMATIVE
                and claim.polarity == Polarity.AFFIRMATIVE
            ):
                rec = ContradictionRecord(
                    contradiction_id=_gen_id("contra_prop"),
                    contradiction_type=ContradictionType.PROPERTY_EXCLUSIVITY,
                    conflicting_claim_ids=(claim.claim_id,),
                    conflicting_belief_ids=(b.belief_id,),
                    severity=0.85,
                    rationale=(
                        f"Mutually exclusive property {claim.predicate_ref}: "
                        f"existing={b.object_ref}, new={claim.object_ref}"
                    ),
                )
                detected.append(rec)

            # 3. Spatial Impossibility (same entity observed at distance > v_max * dt)
            if (
                b.subject_ref.id == claim.subject_ref.id
                and b.predicate_ref in ("LOCATED_AT", "POSITION")
                and claim.predicate_ref in ("LOCATED_AT", "POSITION")
            ):
                pos1 = b.temporal_validity.get("coords")
                pos2 = claim.spatial_scope.get("coords")
                t1 = b.last_updated
                t2 = claim.created_at
                dt = abs(t2 - t1)
                if pos1 and pos2 and isinstance(pos1, tuple) and isinstance(pos2, tuple) and dt > 0:
                    dist = math.dist(pos1, pos2)
                    velocity = dist / dt
                    if velocity > self._max_physical_velocity_mps:
                        rec = ContradictionRecord(
                            contradiction_id=_gen_id("contra_spatial"),
                            contradiction_type=ContradictionType.SPATIAL_IMPOSSIBILITY,
                            conflicting_claim_ids=(claim.claim_id,),
                            conflicting_belief_ids=(b.belief_id,),
                            severity=0.95,
                            rationale=(
                                f"Spatial impossibility for {b.subject_ref.id}: "
                                f"distance={dist:.1f}m in {dt:.2f}s implies {velocity:.1f} m/s > limit"
                            ),
                        )
                        detected.append(rec)

        self._contradictions.extend(detected)
        return detected

    def revise_belief(
        self,
        current_belief: BeliefState,
        fusion_result: FusionResult,
        contradictions: list[ContradictionRecord],
        new_evidence_ids: tuple[str, ...] = (),
        claim_ref: str = "",
    ) -> RevisionResult:
        """Executes principled belief revision based on evidence fusion and contradictions."""
        posterior = fusion_result.posterior_probability
        evidence_strength = fusion_result.effective_evidence_strength

        # Check for unresolvable severe contradictions
        severe_contradictions = [c for c in contradictions if c.severity >= 0.8]
        if severe_contradictions:
            # If evidence strength of new observation is approximately equal to current belief, suspend
            if abs(evidence_strength - current_belief.evidence_strength) < 0.2:
                # Suspend belief pending further disambiguation
                suspended_belief = BeliefState(
                    belief_id=current_belief.belief_id,
                    subject_ref=current_belief.subject_ref,
                    predicate_ref=current_belief.predicate_ref,
                    object_ref=current_belief.object_ref,
                    polarity=current_belief.polarity,
                    posterior_probability=0.5,  # Max uncertainty
                    evidence_strength=min(current_belief.evidence_strength, evidence_strength),
                    supporting_evidence_ids=current_belief.supporting_evidence_ids,
                    contradicting_evidence_ids=tuple(
                        set(current_belief.contradicting_evidence_ids) | set(new_evidence_ids)
                    ),
                    revision_count=current_belief.revision_count + 1,
                    last_updated=time.time(),
                )
                return RevisionResult(
                    action_type=RevisionActionType.SUSPENDED,
                    target_belief_id=current_belief.belief_id,
                    updated_belief=suspended_belief,
                    active_contradictions=tuple(severe_contradictions),
                    rationale="Belief suspended due to unresolved high-severity contradiction",
                )

        # Check for Withdrawal
        if posterior <= self._withdrawal_threshold:
            return RevisionResult(
                action_type=RevisionActionType.WITHDRAWN,
                target_belief_id=current_belief.belief_id,
                updated_belief=None,
                rationale=f"Belief withdrawn: posterior {posterior:.2f} <= {self._withdrawal_threshold}",
            )

        # Check for Confirmation
        if posterior >= self._confirmation_threshold:
            confirmed_belief = BeliefState(
                belief_id=current_belief.belief_id,
                subject_ref=current_belief.subject_ref,
                predicate_ref=current_belief.predicate_ref,
                object_ref=current_belief.object_ref,
                polarity=current_belief.polarity,
                posterior_probability=posterior,
                evidence_strength=evidence_strength,
                supporting_evidence_ids=tuple(
                    set(current_belief.supporting_evidence_ids) | set(new_evidence_ids)
                ),
                contradicting_evidence_ids=current_belief.contradicting_evidence_ids,
                revision_count=current_belief.revision_count + 1,
                last_updated=time.time(),
                provenance_claims=(
                    tuple(set(current_belief.provenance_claims) | {claim_ref})
                    if claim_ref
                    else current_belief.provenance_claims
                ),
            )
            return RevisionResult(
                action_type=RevisionActionType.CONFIRMED,
                target_belief_id=current_belief.belief_id,
                updated_belief=confirmed_belief,
                rationale=f"Belief confirmed with posterior {posterior:.2f}",
            )

        # General replacement / revision
        replaced_belief = BeliefState(
            belief_id=current_belief.belief_id,
            subject_ref=current_belief.subject_ref,
            predicate_ref=current_belief.predicate_ref,
            object_ref=current_belief.object_ref,
            polarity=current_belief.polarity,
            posterior_probability=posterior,
            evidence_strength=evidence_strength,
            supporting_evidence_ids=tuple(
                set(current_belief.supporting_evidence_ids) | set(new_evidence_ids)
            ),
            contradicting_evidence_ids=current_belief.contradicting_evidence_ids,
            revision_count=current_belief.revision_count + 1,
            last_updated=time.time(),
        )
        return RevisionResult(
            action_type=RevisionActionType.REPLACED,
            target_belief_id=current_belief.belief_id,
            updated_belief=replaced_belief,
            rationale=f"Belief updated with revised posterior {posterior:.2f}",
        )

    @staticmethod
    def _objects_match(o1: Any, o2: Any) -> bool:
        if o1 is None and o2 is None:
            return True
        if o1 is None or o2 is None:
            return False
        if isinstance(o1, LiteralValueRef) and isinstance(o2, LiteralValueRef):
            return str(o1.value).lower() == str(o2.value).lower()
        if hasattr(o1, "id") and hasattr(o2, "id"):
            return o1.id == o2.id
        return o1 == o2
