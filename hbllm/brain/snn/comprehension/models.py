"""
Comprehension pipeline data models.

Defines the core data structures used throughout the cognitive
comprehension stream — from individual concept units to the
aggregated understanding state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActivatedMemory:
    """A lightweight memory reference retrieved during comprehension.

    Decouples comprehension from the full MemoryNode interface —
    only the fields needed for downstream reasoning are kept.
    """

    id: str
    content: str
    score: float = 0.0


@dataclass
class ComprehensionUnit:
    """A single perceived concept from the input.

    Produced when the clause neuron fires (concept boundary).
    Contains the text span, its embedding, activated memories,
    domain classification, salience score, and which specialized
    SNN channels contributed metadata.
    """

    text: str
    embedding: Any  # numpy array from ONNX encoder
    activated_memories: list[ActivatedMemory] = field(default_factory=list)
    domain_activation: dict[str, float] = field(default_factory=dict)
    salience: float = 1.0
    channel_metadata: dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class UnderstandingState:
    """Aggregated comprehension result from the full input.

    Combines all individual concept units into a single structured
    representation that the RouterNode can use for refined routing.
    """

    concepts: list[ComprehensionUnit] = field(default_factory=list)
    domain_activations: dict[str, float] = field(default_factory=dict)
    all_memories: list[ActivatedMemory] = field(default_factory=list)
    salience_map: list[float] = field(default_factory=list)
