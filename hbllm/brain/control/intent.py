"""
Lightweight Language-Independent Intent — ADR 002 §3.

Intent is the first-class abstraction that normalizes *what* the user
requests, independent of language, phrasing, or prompt structure.

Execution hierarchy (ADR 002)::

    Conversation → Intent → Goal → Plan → SkillGraph

Design invariants:
    - Intent represents normalized semantics, NOT raw text.
    - Raw text is stored only inside ``ProvenanceMetadata`` as source trace.
    - Intent survives paraphrasing: two differently worded requests
      producing the same semantic action share the same ``IntentType``.
    - Intent is deliberately lightweight — it captures *what*, not *how*.

Usage::

    from hbllm.brain.control.intent import Intent, IntentType

    intent = Intent.create(
        intent_type=IntentType.QUERY,
        semantic_target="weather forecast",
        parameters={"location": "Colombo", "time_range": "today"},
        source_text="What's the weather like in Colombo?",
        source_node="perception.audio_in",
    )
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hbllm.brain.core.provenance import ProvenanceMetadata


class IntentType(StrEnum):
    """Normalized intent categories.

    These represent *what* kind of action the user wants, independent
    of the specific domain or phrasing.
    """

    QUERY = "query"  # Requesting information
    COMMAND = "command"  # Requesting an action/tool execution
    CONVERSATION = "conversation"  # Social/conversational interaction
    CLARIFICATION = "clarification"  # Asking to clarify or elaborate
    CORRECTION = "correction"  # Correcting a previous statement
    CONFIRMATION = "confirmation"  # Confirming a proposed action
    REJECTION = "rejection"  # Rejecting a proposed action
    CONTINUATION = "continuation"  # Continuing a prior thread
    META = "meta"  # Controlling the system itself


class IntentStatus(StrEnum):
    """Lifecycle status of an intent."""

    PENDING = "pending"  # Extracted, not yet routed to a goal
    ACTIVE = "active"  # Mapped to a goal and being processed
    COMPLETED = "completed"  # Successfully resolved
    FAILED = "failed"  # Could not be resolved
    CANCELLED = "cancelled"  # Cancelled by user or system


@dataclass(frozen=True)
class Intent:
    """Normalized semantic intent — language-independent.

    Captures *what* is being requested without prescribing *how* it
    will be achieved.  Raw user text is NOT stored here — only in
    the ``provenance`` field as source trace.

    Attributes:
        intent_id: Globally unique immutable UUID4.
        intent_type: Normalized category of the request.
        semantic_target: The core semantic object of the intent
            (e.g., ``"weather forecast"``, ``"send email"``,
            ``"explain recursion"``).
        parameters: Structured key-value parameters extracted from
            the input (e.g., ``{"location": "Colombo"}``).
        confidence: Extraction confidence [0.0, 1.0].
        status: Current lifecycle status.
        provenance: Causal provenance metadata (carries source text
            in its ``source`` field).
        created_at: Timestamp when the intent was created.
    """

    intent_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    intent_type: IntentType = IntentType.QUERY
    semantic_target: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    status: IntentStatus = IntentStatus.PENDING
    provenance: ProvenanceMetadata | None = None
    created_at: float = field(default_factory=time.time)

    # ── Factory methods ──────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        intent_type: IntentType,
        semantic_target: str,
        parameters: dict[str, Any] | None = None,
        confidence: float = 1.0,
        source_text: str = "",
        source_node: str = "intent_extractor",
        correlation_id: str = "",
    ) -> Intent:
        """Create an intent with automatic provenance generation.

        The ``source_text`` (raw user input) is stored only inside
        the provenance metadata, NOT in the intent body.

        Args:
            intent_type: Normalized intent category.
            semantic_target: Core semantic object of the request.
            parameters: Extracted structured parameters.
            confidence: Extraction confidence [0.0, 1.0].
            source_text: Original raw text (stored as provenance).
            source_node: Subsystem that extracted this intent.
            correlation_id: Session or conversation trace ID.
        """
        prov = ProvenanceMetadata.create(
            source=source_node,
            confidence=confidence,
            correlation_id=correlation_id,
        )
        return cls(
            intent_id=uuid.uuid4().hex,
            intent_type=intent_type,
            semantic_target=semantic_target,
            parameters=parameters or {},
            confidence=max(0.0, min(1.0, confidence)),
            status=IntentStatus.PENDING,
            provenance=prov,
        )

    # ── Lifecycle transitions ────────────────────────────────────────

    def activate(self) -> Intent:
        """Mark intent as actively being processed."""
        return Intent(
            intent_id=self.intent_id,
            intent_type=self.intent_type,
            semantic_target=self.semantic_target,
            parameters=self.parameters,
            confidence=self.confidence,
            status=IntentStatus.ACTIVE,
            provenance=self.provenance,
            created_at=self.created_at,
        )

    def complete(self) -> Intent:
        """Mark intent as successfully resolved."""
        return Intent(
            intent_id=self.intent_id,
            intent_type=self.intent_type,
            semantic_target=self.semantic_target,
            parameters=self.parameters,
            confidence=self.confidence,
            status=IntentStatus.COMPLETED,
            provenance=self.provenance,
            created_at=self.created_at,
        )

    def fail(self) -> Intent:
        """Mark intent as failed."""
        return Intent(
            intent_id=self.intent_id,
            intent_type=self.intent_type,
            semantic_target=self.semantic_target,
            parameters=self.parameters,
            confidence=self.confidence,
            status=IntentStatus.FAILED,
            provenance=self.provenance,
            created_at=self.created_at,
        )

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging and persistence."""
        d: dict[str, Any] = {
            "intent_id": self.intent_id,
            "intent_type": self.intent_type.value,
            "semantic_target": self.semantic_target,
            "parameters": self.parameters,
            "confidence": round(self.confidence, 4),
            "status": self.status.value,
            "created_at": self.created_at,
        }
        if self.provenance is not None:
            d["provenance"] = self.provenance.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Intent:
        """Deserialize from a plain dict."""
        prov_data = data.get("provenance")
        prov = ProvenanceMetadata.from_dict(prov_data) if prov_data else None
        return cls(
            intent_id=data.get("intent_id", uuid.uuid4().hex),
            intent_type=IntentType(data.get("intent_type", "query")),
            semantic_target=data.get("semantic_target", ""),
            parameters=data.get("parameters", {}),
            confidence=data.get("confidence", 1.0),
            status=IntentStatus(data.get("status", "pending")),
            provenance=prov,
            created_at=data.get("created_at", time.time()),
        )

    def __repr__(self) -> str:
        return (
            f"Intent(type={self.intent_type.value!r}, "
            f"target={self.semantic_target!r}, "
            f"status={self.status.value}, "
            f"conf={self.confidence:.2f})"
        )
