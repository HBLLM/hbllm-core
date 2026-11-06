from __future__ import annotations

import time
from dataclasses import dataclass, field

from hbllm.hcir.grounding.contracts.enums import Polarity
from hbllm.hcir.grounding.contracts.references import (
    GroundedRef,
    LiteralValueRef,
    UnresolvedRef,
)


@dataclass(frozen=True)
class BeliefState:
    """The system's current epistemic position regarding a proposition.

    Maintains explicit calibrated posterior probability, supporting and
    contradicting evidence tracking, revision history, and temporal validity.
    """

    belief_id: str
    subject_ref: GroundedRef
    predicate_ref: str
    object_ref: GroundedRef | LiteralValueRef | UnresolvedRef | None
    polarity: Polarity
    posterior_probability: float
    evidence_strength: float
    supporting_evidence_ids: tuple[str, ...]
    contradicting_evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    revision_count: int = 0
    last_updated: float = field(default_factory=time.time)
    temporal_validity: dict[str, float] = field(default_factory=dict)
    provenance_claims: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.belief_id:
            raise ValueError("belief_id must be non-empty")
        if not isinstance(self.subject_ref, GroundedRef):
            raise TypeError(f"subject_ref must be GroundedRef, got {type(self.subject_ref)}")
        if not self.predicate_ref:
            raise ValueError("predicate_ref must be non-empty")
        if not (0.0 <= self.posterior_probability <= 1.0):
            raise ValueError(
                f"posterior_probability must be in [0.0, 1.0], got {self.posterior_probability}"
            )
        if not (0.0 <= self.evidence_strength <= 1.0):
            raise ValueError(
                f"evidence_strength must be in [0.0, 1.0], got {self.evidence_strength}"
            )
        if not isinstance(self.supporting_evidence_ids, tuple):
            object.__setattr__(self, "supporting_evidence_ids", tuple(self.supporting_evidence_ids))
        if not isinstance(self.contradicting_evidence_ids, tuple):
            object.__setattr__(
                self, "contradicting_evidence_ids", tuple(self.contradicting_evidence_ids)
            )
        if not isinstance(self.provenance_claims, tuple):
            object.__setattr__(self, "provenance_claims", tuple(self.provenance_claims))
