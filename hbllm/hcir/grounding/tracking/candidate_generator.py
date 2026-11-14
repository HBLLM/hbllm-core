from __future__ import annotations

import time
import uuid

from hbllm.hcir.grounding.contracts.claims import GroundingClaim
from hbllm.hcir.grounding.contracts.enums import ClaimOrigin, Polarity
from hbllm.hcir.grounding.contracts.evidence import EvidenceRecord
from hbllm.hcir.grounding.contracts.references import (
    EntityCandidateRef,
    GroundedConceptRef,
    GroundedRef,
    LiteralValueRef,
    UnresolvedRef,
)


def _gen_claim_id() -> str:
    return f"claim_{uuid.uuid4().hex[:10]}"


class CandidateGenerator:
    """Generates uncommitted GroundingClaim proposals from sensory EvidenceRecords.

    Translates perceptual detections and tracked entity states into candidate propositions
    for Epistemic evaluation, without directly mutating the authoritative world state.
    """

    def __init__(self, generator_name: str = "default_candidate_generator") -> None:
        self._generator_name = generator_name

    def generate_property_claim(
        self,
        subject_ref: GroundedRef,
        property_name: str,
        value: float | int | str | bool,
        evidence_records: list[EvidenceRecord],
        origin: ClaimOrigin = ClaimOrigin.DIRECT_SENSORY,
        polarity: Polarity = Polarity.AFFIRMATIVE,
        temporal_scope: dict[str, float] | None = None,
        spatial_scope: dict[str, float] | None = None,
    ) -> GroundingClaim:
        """Proposes a property claim (e.g. COLOR(ecand_1, RED) or HEIGHT(ecand_1, 15.0))."""
        if not evidence_records:
            raise ValueError("CandidateGenerator requires at least one EvidenceRecord")

        ev_ids = tuple(ev.evidence_id for ev in evidence_records)
        literal_ref = LiteralValueRef(value=value)

        return GroundingClaim(
            claim_id=_gen_claim_id(),
            origin=origin,
            subject_ref=subject_ref,
            predicate_ref=property_name.upper(),
            object_ref=literal_ref,
            polarity=polarity,
            evidence_ids=ev_ids,
            temporal_scope=temporal_scope or {"created": time.time()},
            spatial_scope=spatial_scope or {},
            generated_by=self._generator_name,
        )

    def generate_relation_claim(
        self,
        subject_ref: GroundedRef,
        relation_name: str,
        object_ref: GroundedRef | UnresolvedRef,
        evidence_records: list[EvidenceRecord],
        origin: ClaimOrigin = ClaimOrigin.DIRECT_SENSORY,
        polarity: Polarity = Polarity.AFFIRMATIVE,
        temporal_scope: dict[str, float] | None = None,
        spatial_scope: dict[str, float] | None = None,
    ) -> GroundingClaim:
        """Proposes a dyadic relational claim (e.g. LOCATED_NEAR(ecand_1, e_sofa) or ON(ecand_1, ecand_2))."""
        if not evidence_records:
            raise ValueError("CandidateGenerator requires at least one EvidenceRecord")

        ev_ids = tuple(ev.evidence_id for ev in evidence_records)

        return GroundingClaim(
            claim_id=_gen_claim_id(),
            origin=origin,
            subject_ref=subject_ref,
            predicate_ref=relation_name.upper(),
            object_ref=object_ref,
            polarity=polarity,
            evidence_ids=ev_ids,
            temporal_scope=temporal_scope or {"created": time.time()},
            spatial_scope=spatial_scope or {},
            generated_by=self._generator_name,
        )

    def generate_concept_membership_claim(
        self,
        candidate_ref: EntityCandidateRef,
        concept_ref: GroundedConceptRef,
        evidence_records: list[EvidenceRecord],
        origin: ClaimOrigin = ClaimOrigin.DIRECT_SENSORY,
        polarity: Polarity = Polarity.AFFIRMATIVE,
    ) -> GroundingClaim:
        """Proposes that a candidate entity is an instance of a GroundedConcept (e.g. INSTANCE_OF(ecand_1, C-00387))."""
        if not evidence_records:
            raise ValueError("CandidateGenerator requires at least one EvidenceRecord")

        ev_ids = tuple(ev.evidence_id for ev in evidence_records)

        return GroundingClaim(
            claim_id=_gen_claim_id(),
            origin=origin,
            subject_ref=candidate_ref,
            predicate_ref="INSTANCE_OF",
            object_ref=concept_ref,
            polarity=polarity,
            evidence_ids=ev_ids,
            generated_by=self._generator_name,
        )
