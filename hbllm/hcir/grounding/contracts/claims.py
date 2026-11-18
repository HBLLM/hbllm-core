from __future__ import annotations

import time
from dataclasses import dataclass, field

from hbllm.hcir.grounding.contracts.enums import ClaimOrigin, Polarity
from hbllm.hcir.grounding.contracts.references import (
    GroundedRef,
    LatentRelationRef,
    LiteralValueRef,
    PredicateRef,
    UnresolvedRef,
)


@dataclass(frozen=True)
class GroundingClaim:
    """An uncommitted semantic proposition generated from observations.

    References multiple evidence IDs rather than collapsing them into a single score.
    Uses typed reference objects for subject, predicate, and object, with first-class support
    for UnresolvedRef.
    """

    claim_id: str
    origin: ClaimOrigin
    subject_ref: GroundedRef
    predicate_ref: str | PredicateRef | LatentRelationRef
    object_ref: GroundedRef | LiteralValueRef | UnresolvedRef | None
    polarity: Polarity
    evidence_ids: tuple[str, ...]
    temporal_scope: dict[str, float] = field(default_factory=dict)
    spatial_scope: dict[str, float] = field(default_factory=dict)
    generated_by: str = ""
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.claim_id:
            raise ValueError("claim_id must be non-empty")
        if not isinstance(self.subject_ref, GroundedRef):
            raise TypeError(
                f"subject_ref must be an instance of GroundedRef, got {type(self.subject_ref)}"
            )
        if not self.predicate_ref:
            raise ValueError("predicate_ref must be non-empty")
        if self.object_ref is not None and not isinstance(
            self.object_ref, (GroundedRef, LiteralValueRef, UnresolvedRef)
        ):
            raise TypeError(
                f"object_ref must be GroundedRef, LiteralValueRef, UnresolvedRef, or None; got {type(self.object_ref)}"
            )
        if not isinstance(self.polarity, Polarity):
            raise TypeError(f"polarity must be an instance of Polarity, got {type(self.polarity)}")
        if not isinstance(self.evidence_ids, tuple):
            object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
