"""
HBLLM Graph Types — shared primitives for knowledge and memory graphs.

These are general-purpose graph concepts used across the memory and
knowledge subsystems. They live here (not in memory/) because they
are graph-structural primitives, not memory-specific concepts.

Both ``hbllm.memory.knowledge_graph`` and ``hbllm.knowledge.extractor``
import from this module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Entity:
    """A node in a knowledge or memory graph.

    Attributes:
        id: Unique, stable identifier (e.g. ``class::path/file.py#ClassName``).
        label: Human-readable display name.
        entity_type: Category (``concept``, ``class``, ``module``, ``folder``, etc.).
        attributes: Extensible key-value metadata.
        created_at: Epoch timestamp of creation.
        confidence: Knowledge confidence — decays over time, reinforced by evidence.
        evidence_count: Number of evidence events supporting this entity.
        verified: Whether this entity has been explicitly verified.
        last_reinforced: Epoch timestamp of last reinforcement.
    """

    id: str
    label: str
    entity_type: str = "concept"
    attributes: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    # Knowledge confidence — decays over time, reinforced by evidence
    confidence: float = 1.0
    evidence_count: int = 1
    verified: bool = False
    last_reinforced: float = field(default_factory=time.time)

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class Relation:
    """A directed edge in a knowledge or memory graph.

    Attributes:
        source_id: ID of the source entity.
        target_id: ID of the target entity.
        relation_type: Edge label (``contains``, ``is_a``, ``uses``, etc.).
        weight: Edge strength [0.0, ∞).
        metadata: Extensible key-value metadata.
        created_at: Epoch timestamp of creation.
        valid_from: Optional temporal validity start.
        valid_until: Optional temporal validity end.
    """

    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    valid_from: float | None = None
    valid_until: float | None = None

    @property
    def key(self) -> str:
        return f"{self.source_id}--{self.relation_type}-->{self.target_id}"
