from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GroundedRef:
    """Base immutable identifier for grounded entities, concepts, and tracklets."""

    id: str

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValueError(f"GroundedRef id must be a non-empty string, got {self.id!r}")


@dataclass(frozen=True)
class TrackletRef(GroundedRef):
    """Temporary spatio-temporal observation trajectory (single sensor channel)."""

    channel_id: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.id.startswith("trk_"):
            raise ValueError(f"TrackletRef id must start with 'trk_', got {self.id!r}")


@dataclass(frozen=True)
class EntityCandidateRef(GroundedRef):
    """Provisional persistent entity hypothesis bridging occlusions or modalities."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.id.startswith("ecand_"):
            raise ValueError(f"EntityCandidateRef id must start with 'ecand_', got {self.id!r}")


@dataclass(frozen=True)
class GroundedEntityRef(GroundedRef):
    """Authoritative physical or symbolic entity committed to Canonical CognitiveGraph."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.id.startswith("e_"):
            raise ValueError(f"GroundedEntityRef id must start with 'e_', got {self.id!r}")


@dataclass(frozen=True)
class GroundedConceptRef(GroundedRef):
    """Grounded structural/predictive concept identifier (e.g. 'C-00387')."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.id.startswith("C-") and not self.id.startswith("concept_"):
            raise ValueError(
                f"GroundedConceptRef id must start with 'C-' or 'concept_', got {self.id!r}"
            )


@dataclass(frozen=True)
class PredicateRef:
    """Canonical or predefined relational predicate reference (e.g. 'NEAR', 'CONTAINS')."""

    name: str

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError(f"PredicateRef name must be a non-empty string, got {self.name!r}")


@dataclass(frozen=True)
class LatentRelationRef(GroundedRef):
    """Dynamically discovered relational invariant (e.g. 'rel_sync_1', 'rel_dynamic_3')."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.id.startswith("rel_"):
            raise ValueError(f"LatentRelationRef id must start with 'rel_', got {self.id!r}")


@dataclass(frozen=True)
class LiteralValueRef:
    """Typed primitive literal value (scalar, string, color code, or measurement)."""

    value: float | int | str | bool
    unit: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.value, (float, int, str, bool)):
            raise TypeError(
                f"LiteralValueRef value must be primitive (float, int, str, bool), got {type(self.value)}"
            )


@dataclass(frozen=True)
class UnresolvedRef:
    """Explicit placeholder for unresolved references awaiting epistemic binding.

    An unresolved reference is a legitimate, first-class semantic state (e.g.
    'the thing left beside the sofa'), not an error condition.
    """

    placeholder_id: str
    constraints: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.placeholder_id:
            raise ValueError("UnresolvedRef placeholder_id cannot be empty")
