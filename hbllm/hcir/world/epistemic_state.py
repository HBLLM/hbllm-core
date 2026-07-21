"""
Epistemic State — Belief Metadata, Evidence Tracking, & Certainty Classification.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum


class EvidenceSource(str, Enum):
    """Origin of cognitive evidence."""

    SENSOR = "sensor"
    MODEL = "model"
    HUMAN = "human"
    MEMORY = "memory"
    SIMULATION = "simulation"


class CertaintyLevel(str, Enum):
    """Certainty level of a cognitive belief."""

    FACT = "fact"
    OBSERVATION = "observation"
    INFERENCE = "inference"
    HYPOTHESIS = "hypothesis"


@dataclass
class EpistemicState:
    """Metadata container describing the epistemic backing of a belief."""

    belief_id: str
    confidence: float = 0.9
    evidence_sources: Sequence[EvidenceSource] = field(
        default_factory=lambda: (EvidenceSource.SENSOR,)
    )
    certainty: CertaintyLevel = CertaintyLevel.OBSERVATION
    last_verified: float = field(default_factory=time.time)

    def is_empirically_grounded(self) -> bool:
        """Return True if evidence includes direct empirical sensor observation."""
        return EvidenceSource.SENSOR in self.evidence_sources
