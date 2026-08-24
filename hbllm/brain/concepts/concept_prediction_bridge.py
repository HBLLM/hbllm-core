"""Concept Prediction Bridge — makes concepts generate A12-compatible predictions.

**Critical invariant:** Concepts NEVER bypass A14.

The bridge is an **adapter** into A12, not another prediction engine.
Concepts provide constraints/priors/abstractions to the existing
prediction substrate.

Architecture::

    GroundedConcept
          ↓
    ConceptPredictionSpec
          ↓
    A12 PredictionOperator
          ↓
    PredictionNode
          ↓
    A13 observes reality
          ↓
    A14 measures prediction error
          ↓
    concept reinforced OR refined

This preserves::

    A12 = prediction machinery
    A13 = world state
    A14 = adaptation
    A15 = abstraction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Concept Prediction Spec
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ConceptPredictionSpec:
    """A prediction specification derived from a concept.

    Provides constraints/priors for A12 prediction operators.
    Does NOT contain prediction logic — it's an adapter.
    """

    concept_id: str
    concept_name: str
    target_entity_id: str  # Entity being predicted about
    predicted_behavior: str  # e.g., "supports_objects"
    predicted_value: Any = None  # Expected value
    confidence: float = 0.5  # From concept confidence
    domain: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Concept Prediction Bridge
# ═══════════════════════════════════════════════════════════════════════════


class ConceptPredictionBridge:
    """Adapter between GroundedConcepts and A12 prediction substrate.

    Generates ConceptPredictionSpecs from concepts, which A12
    prediction operators can consume.

    The bridge itself contains NO prediction intelligence.

    Usage::

        bridge = ConceptPredictionBridge(registry)

        # Generate predictions for all concepts
        specs = bridge.generate_predictions(entity_ids)

        # Record outcome (feeds through A14)
        bridge.record_outcome(concept_id, entity_id, correct=True)
    """

    def __init__(self, registry: Any) -> None:
        self._registry = registry

    def generate_predictions(
        self,
        entity_ids: list[str] | None = None,
    ) -> list[ConceptPredictionSpec]:
        """Generate prediction specs from concepts.

        For each entity that is INSTANCE_OF a concept, generate
        a prediction based on the concept's behavioral regularities.

        Args:
            entity_ids: Optional filter. If None, generates for all concept members.

        Returns:
            List of ConceptPredictionSpecs for A12.
        """
        specs: list[ConceptPredictionSpec] = []

        for concept in self._registry.all_concepts():
            members = self._registry.concept_members(concept.id)

            target_members = (
                [m for m in members if m in entity_ids]
                if entity_ids
                else members
            )

            for member_id in target_members:
                for behavior in concept.behavioral_regularities:
                    specs.append(ConceptPredictionSpec(
                        concept_id=concept.id,
                        concept_name=concept.concept_name,
                        target_entity_id=member_id,
                        predicted_behavior=behavior,
                        confidence=concept.confidence,
                        domain=concept.domain,
                    ))

        return specs

    def record_outcome(
        self,
        concept_id: str,
        correct: bool,
    ) -> None:
        """Record a prediction outcome, feeding through to A14.

        Updates concept confidence via the registry.
        """
        self._registry.record_prediction(concept_id, correct)

    def concept_predictions_for_entity(
        self,
        entity_id: str,
    ) -> list[ConceptPredictionSpec]:
        """Get all concept-based predictions for a specific entity."""
        concepts = self._registry.entity_concepts(entity_id)
        specs: list[ConceptPredictionSpec] = []

        for concept_id in concepts:
            concept = self._registry.get_concept(concept_id)
            if concept is None:
                continue

            for behavior in concept.behavioral_regularities:
                specs.append(ConceptPredictionSpec(
                    concept_id=concept.id,
                    concept_name=concept.concept_name,
                    target_entity_id=entity_id,
                    predicted_behavior=behavior,
                    confidence=concept.confidence,
                    domain=concept.domain,
                ))

        return specs
