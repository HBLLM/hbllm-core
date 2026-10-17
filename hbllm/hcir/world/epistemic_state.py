"""Epistemic State — Belief Metadata, Evidence Tracking, & Certainty Classification.

Metadata container describing the epistemic backing of a belief, lifted to
declarative CausalPredicates and atomic TransactionOperations.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate


class EvidenceSource(StrEnum):
    """Origin of cognitive evidence."""

    SENSOR = "sensor"
    MODEL = "model"
    HUMAN = "human"
    MEMORY = "memory"
    SIMULATION = "simulation"


class CertaintyLevel(StrEnum):
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
    spatial_observability_mask: Any = None
    information_gain: float = 0.0

    def is_empirically_grounded(self) -> bool:
        """Return True if evidence includes direct empirical sensor observation."""
        return EvidenceSource.SENSOR in self.evidence_sources

    def to_causal_predicate(self) -> CausalPredicate:
        """Export epistemic backing as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"epistemic:{self.belief_id}",
            operator="==",
            value=self.certainty.value,
            relation=str(self.confidence),
            target_entity=self.belief_id,
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction updating belief epistemic backing."""
        return [
            TransactionOperation(
                op=TransactionOp.MODIFY_NODE,
                node_id=self.belief_id,
                changes={
                    "confidence": self.confidence,
                    "certainty": self.certainty.value,
                    "evidence_sources": [str(s) for s in self.evidence_sources],
                    "last_verified": self.last_verified,
                    "information_gain": self.information_gain,
                },
            )
        ]
