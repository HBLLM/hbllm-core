from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from hbllm.hcir.grounding.contracts.references import (
    EntityCandidateRef,
    GroundedEntityRef,
    GroundedRef,
    TrackletRef,
)


class IdentityOperationType(str, Enum):
    """The type of reversible entity identity operation."""

    ASSOCIATION = "association"
    MERGE = "merge"
    SPLIT = "split"
    REIDENTIFICATION = "reidentification"
    ALIAS = "alias"


@dataclass(frozen=True)
class IdentityOperation:
    """Base dataclass for reversible entity identity operations."""

    operation_id: str
    confidence: float
    rationale: str
    op_type: IdentityOperationType = IdentityOperationType.ASSOCIATION
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    timestamp: float = field(default_factory=time.time)
    reversible: bool = True

    def __post_init__(self) -> None:
        if not self.operation_id:
            raise ValueError("operation_id must be non-empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if not isinstance(self.evidence_ids, tuple):
            object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))


@dataclass(frozen=True)
class EntityAssociationOp(IdentityOperation):
    """Binds a sensor-local TrackletRef to a persistent EntityCandidateRef."""

    tracklet_ref: TrackletRef = field(default_factory=lambda: TrackletRef(id="trk_dummy"))
    candidate_ref: EntityCandidateRef = field(
        default_factory=lambda: EntityCandidateRef(id="ecand_dummy")
    )
    op_type: IdentityOperationType = IdentityOperationType.ASSOCIATION

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.tracklet_ref, TrackletRef):
            raise TypeError(f"tracklet_ref must be TrackletRef, got {type(self.tracklet_ref)}")
        if not isinstance(self.candidate_ref, EntityCandidateRef):
            raise TypeError(
                f"candidate_ref must be EntityCandidateRef, got {type(self.candidate_ref)}"
            )


@dataclass(frozen=True)
class EntityMergeOp(IdentityOperation):
    """Proposes merging two or more candidate entities into one unified candidate."""

    source_candidates: tuple[EntityCandidateRef, ...] = field(default_factory=tuple)
    merged_candidate: EntityCandidateRef = field(
        default_factory=lambda: EntityCandidateRef(id="ecand_dummy")
    )
    prior_tracklet_map: dict[str, tuple[str, ...]] = field(default_factory=dict)
    op_type: IdentityOperationType = IdentityOperationType.MERGE

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.source_candidates, tuple):
            object.__setattr__(self, "source_candidates", tuple(self.source_candidates))
        if len(self.source_candidates) < 2:
            raise ValueError("EntityMergeOp requires at least 2 source candidates")
        for sc in self.source_candidates:
            if not isinstance(sc, EntityCandidateRef):
                raise TypeError(f"source candidate must be EntityCandidateRef, got {type(sc)}")
        if not isinstance(self.merged_candidate, EntityCandidateRef):
            raise TypeError(
                f"merged_candidate must be EntityCandidateRef, got {type(self.merged_candidate)}"
            )


@dataclass(frozen=True)
class EntitySplitOp(IdentityOperation):
    """Reverses an erroneous merge or splits a heterogeneous candidate entity."""

    split_candidate: EntityCandidateRef = field(
        default_factory=lambda: EntityCandidateRef(id="ecand_dummy")
    )
    resulting_candidates: tuple[EntityCandidateRef, ...] = field(default_factory=tuple)
    merge_operation_ref: str | None = None
    tracklet_partition: dict[str, str] = field(default_factory=dict)
    op_type: IdentityOperationType = IdentityOperationType.SPLIT

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.split_candidate, EntityCandidateRef):
            raise TypeError(
                f"split_candidate must be EntityCandidateRef, got {type(self.split_candidate)}"
            )
        if not isinstance(self.resulting_candidates, tuple):
            object.__setattr__(self, "resulting_candidates", tuple(self.resulting_candidates))
        if len(self.resulting_candidates) < 2:
            raise ValueError("EntitySplitOp requires at least 2 resulting candidates")
        for rc in self.resulting_candidates:
            if not isinstance(rc, EntityCandidateRef):
                raise TypeError(f"resulting candidate must be EntityCandidateRef, got {type(rc)}")


@dataclass(frozen=True)
class EntityReidentificationOp(IdentityOperation):
    """Associates an EntityCandidateRef with an authoritative GroundedEntityRef."""

    candidate_ref: EntityCandidateRef = field(
        default_factory=lambda: EntityCandidateRef(id="ecand_dummy")
    )
    grounded_ref: GroundedEntityRef = field(default_factory=lambda: GroundedEntityRef(id="e_dummy"))
    similarity_metrics: dict[str, float] = field(default_factory=dict)
    op_type: IdentityOperationType = IdentityOperationType.REIDENTIFICATION

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.candidate_ref, EntityCandidateRef):
            raise TypeError(
                f"candidate_ref must be EntityCandidateRef, got {type(self.candidate_ref)}"
            )
        if not isinstance(self.grounded_ref, GroundedEntityRef):
            raise TypeError(
                f"grounded_ref must be GroundedEntityRef, got {type(self.grounded_ref)}"
            )


@dataclass(frozen=True)
class EntityAliasOp(IdentityOperation):
    """Binds an alias label or contextual name to a candidate or grounded entity."""

    target_ref: GroundedRef = field(default_factory=lambda: GroundedRef(id="dummy"))
    alias_label: str = ""
    alias_context: str = ""
    op_type: IdentityOperationType = IdentityOperationType.ALIAS

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.target_ref, GroundedRef):
            raise TypeError(f"target_ref must be GroundedRef, got {type(self.target_ref)}")
        if not self.alias_label:
            raise ValueError("alias_label must be non-empty")
