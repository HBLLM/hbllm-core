"""
HCIR Identity — stable, structured object identity for distributed operation.

Every HCIR object (node, edge, event, transaction) gets a structured
identity that encodes its scope hierarchy:

    tenant → device → namespace → object_type → uuid → version

This replaces bare UUID strings with identifiers that carry provenance
and ownership information, enabling:

    - Deterministic merge conflict resolution across devices
    - Natural tenant isolation at the ID level
    - Causal ordering within a device's event stream
    - Human-readable debugging
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# ═══════════════════════════════════════════════════════════════════════════
# Object Namespace
# ═══════════════════════════════════════════════════════════════════════════


class HCIRNamespace(StrEnum):
    """Logical namespaces for HCIR objects."""

    KNOWLEDGE = "knowledge"
    EXECUTION = "execution"
    MEMORY = "memory"
    SIMULATION = "simulation"
    SYSTEM = "system"


# ═══════════════════════════════════════════════════════════════════════════
# Structured Object ID
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class HCIRObjectID:
    """Structured identity for all HCIR objects.

    Encodes the full scope hierarchy in a compact, hashable format.
    Compatible with the existing ``HCIRNode.id`` string field via
    ``to_string()`` / ``from_string()`` conversion.

    Format::

        {tenant_id}:{device_id}:{namespace}:{object_type}:{uuid}:{version}

    Example::

        acme:laptop_01:knowledge:belief:83fa1e2b:3
    """

    tenant_id: str = "default"
    device_id: str = "local"
    namespace: HCIRNamespace = HCIRNamespace.KNOWLEDGE
    object_type: str = "node"  # node, edge, event, transaction
    uuid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    version: int = 1

    def to_string(self) -> str:
        """Serialize to a compact string representation."""
        return (
            f"{self.tenant_id}:{self.device_id}:"
            f"{self.namespace}:{self.object_type}:"
            f"{self.uuid}:{self.version}"
        )

    @classmethod
    def from_string(cls, s: str) -> HCIRObjectID:
        """Parse a structured ID from its string representation."""
        parts = s.split(":")
        if len(parts) != 6:
            # Fallback: treat as legacy bare UUID
            return cls(uuid=s, object_type="legacy")
        return cls(
            tenant_id=parts[0],
            device_id=parts[1],
            namespace=HCIRNamespace(parts[2])
            if parts[2] in HCIRNamespace.__members__.values()
            else HCIRNamespace.KNOWLEDGE,
            object_type=parts[3],
            uuid=parts[4],
            version=int(parts[5]),
        )

    def next_version(self) -> HCIRObjectID:
        """Return a new ID with incremented version."""
        return HCIRObjectID(
            tenant_id=self.tenant_id,
            device_id=self.device_id,
            namespace=self.namespace,
            object_type=self.object_type,
            uuid=self.uuid,
            version=self.version + 1,
        )

    def with_namespace(self, namespace: HCIRNamespace) -> HCIRObjectID:
        """Return a copy with a different namespace."""
        return HCIRObjectID(
            tenant_id=self.tenant_id,
            device_id=self.device_id,
            namespace=namespace,
            object_type=self.object_type,
            uuid=self.uuid,
            version=self.version,
        )

    @property
    def is_legacy(self) -> bool:
        return self.object_type == "legacy"

    def __str__(self) -> str:
        return self.to_string()


# ═══════════════════════════════════════════════════════════════════════════
# ID Factory
# ═══════════════════════════════════════════════════════════════════════════


class IDFactory:
    """Factory for generating scoped HCIR object IDs.

    Bound to a specific tenant and device context.

    Usage::

        factory = IDFactory(tenant_id="acme", device_id="laptop_01")
        node_id = factory.node_id(namespace=HCIRNamespace.KNOWLEDGE, object_type="belief")
        event_id = factory.event_id()
    """

    def __init__(
        self,
        tenant_id: str = "default",
        device_id: str = "local",
    ) -> None:
        self._tenant_id = tenant_id
        self._device_id = device_id

    def node_id(
        self,
        namespace: HCIRNamespace = HCIRNamespace.KNOWLEDGE,
        object_type: str = "node",
    ) -> HCIRObjectID:
        return HCIRObjectID(
            tenant_id=self._tenant_id,
            device_id=self._device_id,
            namespace=namespace,
            object_type=object_type,
        )

    def edge_id(self) -> HCIRObjectID:
        return HCIRObjectID(
            tenant_id=self._tenant_id,
            device_id=self._device_id,
            namespace=HCIRNamespace.KNOWLEDGE,
            object_type="edge",
        )

    def event_id(self) -> HCIRObjectID:
        return HCIRObjectID(
            tenant_id=self._tenant_id,
            device_id=self._device_id,
            namespace=HCIRNamespace.SYSTEM,
            object_type="event",
        )

    def transaction_id(self) -> HCIRObjectID:
        return HCIRObjectID(
            tenant_id=self._tenant_id,
            device_id=self._device_id,
            namespace=HCIRNamespace.EXECUTION,
            object_type="transaction",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Causal Event — explainability layer
# ═══════════════════════════════════════════════════════════════════════════


class CauseRelation(StrEnum):
    """How one event relates causally to another."""

    TRIGGERED_BY = "triggered_by"
    DERIVED_FROM = "derived_from"
    RESPONSE_TO = "response_to"
    ROLLBACK_OF = "rollback_of"
    FORK_OF = "fork_of"
    MERGE_OF = "merge_of"


@dataclass(frozen=True)
class CausalLink:
    """A directed causal relationship between events."""

    source_event_id: str
    target_event_id: str
    relation: CauseRelation
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalEvent:
    """An HCIR event enriched with causal provenance.

    Extends the base ``GraphEvent`` model with:
        - Structured event ID
        - Parent event references (DAG of causality)
        - Actor identity
        - Transaction binding
        - Content hash for tamper detection

    This enables the explainability query:
        "Why did belief B change?"
    → Transaction T932 → PlannerNode → Observation O12 → SensorEvent E44
    """

    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    parent_event_ids: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    actor: str = ""  # Node ID, "kernel", or "external"
    transaction_id: str = ""
    event_type: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def compute_hash(self) -> str:
        """Compute a deterministic content hash for tamper detection."""
        import json

        payload = json.dumps(
            {
                "event_id": self.event_id,
                "parent_event_ids": sorted(self.parent_event_ids),
                "actor": self.actor,
                "transaction_id": self.transaction_id,
                "event_type": self.event_type,
                "data": self.data,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def seal(self) -> CausalEvent:
        """Seal the event with its content hash (immutable after this)."""
        self.content_hash = self.compute_hash()
        return self


@dataclass
class CausalGraph:
    """A directed acyclic graph of causal events.

    Enables tracing "why did X happen?" by walking the causal chain.

    Usage::

        cg = CausalGraph()
        e1 = CausalEvent(event_id="e1", event_type="observation")
        e2 = CausalEvent(event_id="e2", parent_event_ids=["e1"], event_type="belief_update")
        cg.add_event(e1)
        cg.add_event(e2)
        chain = cg.trace_causes("e2")  # Returns [e1]
    """

    def __init__(self) -> None:
        self._events: dict[str, CausalEvent] = {}
        self._children: dict[str, list[str]] = {}  # parent → children

    def add_event(self, event: CausalEvent) -> None:
        """Add an event to the causal graph."""
        self._events[event.event_id] = event
        for parent_id in event.parent_event_ids:
            self._children.setdefault(parent_id, []).append(event.event_id)

    def get_event(self, event_id: str) -> CausalEvent | None:
        return self._events.get(event_id)

    def trace_causes(self, event_id: str, max_depth: int = 50) -> list[CausalEvent]:
        """Walk up the causal chain to find root causes.

        Returns events in causal order (root cause first).
        """
        visited: set[str] = set()
        chain: list[CausalEvent] = []

        def _walk(eid: str, depth: int) -> None:
            if depth > max_depth or eid in visited:
                return
            visited.add(eid)
            event = self._events.get(eid)
            if event is None:
                return
            for parent_id in event.parent_event_ids:
                _walk(parent_id, depth + 1)
            chain.append(event)

        _walk(event_id, 0)
        # Remove the query event itself
        return [e for e in chain if e.event_id != event_id]

    def trace_effects(self, event_id: str, max_depth: int = 50) -> list[CausalEvent]:
        """Walk down the causal graph to find all effects."""
        visited: set[str] = set()
        effects: list[CausalEvent] = []

        def _walk(eid: str, depth: int) -> None:
            if depth > max_depth or eid in visited:
                return
            visited.add(eid)
            for child_id in self._children.get(eid, []):
                child = self._events.get(child_id)
                if child:
                    effects.append(child)
                    _walk(child_id, depth + 1)

        _walk(event_id, 0)
        return effects

    @property
    def event_count(self) -> int:
        return len(self._events)
