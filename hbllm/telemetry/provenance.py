"""
Universal Causal & Epistemic Provenance.

Every state object, memory triple, perception snapshot, and cognitive event
carries standardized causal provenance metadata enabling:

    - Belief revision: stale or unverified facts are deprioritized.
    - Conflict resolution: overlapping sources are ranked by confidence.
    - Deterministic replay: parent/correlation chains reconstruct causality.
    - Distributed sync: event IDs and vector clocks prevent duplication.

Usage::

    from hbllm.telemetry.provenance import ProvenanceMetadata, VerificationState

    prov = ProvenanceMetadata.create(
        source="perception.audio_in",
        confidence=0.92,
        expiry_seconds=300.0,
    )
    child = ProvenanceMetadata.derive(parent=prov, source="hcir.snn.comprehension")
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class VerificationState(StrEnum):
    """Epistemic status of a provenance-tagged object."""

    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    HYPOTHETICAL = "hypothetical"
    STALE = "stale"


@dataclass(frozen=True)
class ProvenanceMetadata:
    """Immutable causal and epistemic provenance wrapper."""

    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_event_id: str | None = None
    correlation_id: str = ""
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    expiry: float | None = None
    verification_state: VerificationState = VerificationState.UNVERIFIED

    @classmethod
    def create(
        cls,
        source: str,
        confidence: float = 1.0,
        expiry_seconds: float | None = None,
        correlation_id: str = "",
        verification_state: VerificationState = VerificationState.UNVERIFIED,
    ) -> ProvenanceMetadata:
        """Create a root provenance (no causal parent)."""
        now = time.time()
        return cls(
            event_id=uuid.uuid4().hex,
            parent_event_id=None,
            correlation_id=correlation_id or uuid.uuid4().hex[:12],
            source=source,
            timestamp=now,
            confidence=max(0.0, min(1.0, confidence)),
            expiry=now + expiry_seconds if expiry_seconds is not None else None,
            verification_state=verification_state,
        )

    @classmethod
    def derive(
        cls,
        parent: ProvenanceMetadata,
        source: str,
        confidence: float | None = None,
        expiry_seconds: float | None = None,
        verification_state: VerificationState | None = None,
    ) -> ProvenanceMetadata:
        """Derive child provenance from a causal parent."""
        now = time.time()
        conf = confidence if confidence is not None else parent.confidence
        exp_s = expiry_seconds
        if exp_s is not None:
            exp = now + exp_s
        elif parent.expiry is not None:
            exp = parent.expiry
        else:
            exp = None
        return cls(
            event_id=uuid.uuid4().hex,
            parent_event_id=parent.event_id,
            correlation_id=parent.correlation_id,
            source=source,
            timestamp=now,
            confidence=max(0.0, min(1.0, conf)),
            expiry=exp,
            verification_state=(
                verification_state if verification_state is not None else parent.verification_state
            ),
        )

    @property
    def is_expired(self) -> bool:
        """Check whether this data has passed its TTL."""
        if self.expiry is None:
            return False
        return time.time() > self.expiry

    @property
    def is_trustworthy(self) -> bool:
        """Quick check: verified, not expired, confidence >= 0.5."""
        return (
            self.verification_state == VerificationState.VERIFIED
            and not self.is_expired
            and self.confidence >= 0.5
        )

    @property
    def effective_verification(self) -> VerificationState:
        """Return the effective verification state, accounting for expiry."""
        if self.is_expired:
            return VerificationState.STALE
        return self.verification_state

    def with_verification(self, state: VerificationState) -> ProvenanceMetadata:
        """Return a copy with updated verification state."""
        return ProvenanceMetadata(
            event_id=self.event_id,
            parent_event_id=self.parent_event_id,
            correlation_id=self.correlation_id,
            source=self.source,
            timestamp=self.timestamp,
            confidence=self.confidence,
            expiry=self.expiry,
            verification_state=state,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging, persistence, and replay."""
        return {
            "event_id": self.event_id,
            "parent_event_id": self.parent_event_id,
            "correlation_id": self.correlation_id,
            "source": self.source,
            "timestamp": self.timestamp,
            "confidence": round(self.confidence, 4),
            "expiry": self.expiry,
            "verification_state": self.verification_state.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProvenanceMetadata:
        """Deserialize from a plain dict."""
        return cls(
            event_id=data.get("event_id", uuid.uuid4().hex),
            parent_event_id=data.get("parent_event_id"),
            correlation_id=data.get("correlation_id", ""),
            source=data.get("source", ""),
            timestamp=data.get("timestamp", time.time()),
            confidence=data.get("confidence", 1.0),
            expiry=data.get("expiry"),
            verification_state=VerificationState(data.get("verification_state", "unverified")),
        )

    def __repr__(self) -> str:
        return (
            f"Provenance(id={self.event_id[:8]}…, "
            f"src={self.source!r}, "
            f"conf={self.confidence:.2f}, "
            f"state={self.verification_state.value})"
        )
