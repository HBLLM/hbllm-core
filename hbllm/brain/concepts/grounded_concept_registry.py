"""Grounded Concept Registry — HCIR-native concept management for A15.

Each concept is a GroundedConceptNode in the graph with:
- EXEMPLAR_OF edges during hypothesis (candidate phase)
- INSTANCE_OF edges after predictive utility validation

Concept confidence is derived from recent predictive performance —
NOT time decay or instance count.

Low-confidence concepts emit ConceptDegradationSignal
rather than being immediately deleted.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.hcir.graph import (
    CognitiveGraph,
    GroundedConceptNode,
    HCIREdge,
    HCIREdgeType,
    HCIRNodeType,
)

logger = logging.getLogger(__name__)


class GroundedConceptRegistry:
    """HCIR-native registry of grounded concepts.

    Usage::

        registry = GroundedConceptRegistry(graph)
        concept_id = registry.register(
            concept_name="C-00417",
            feature_prototype={...},
            member_ids=["e1", "e2", "e3"],
            ...
        )

        registry.update_confidence(concept_id, prediction_success=True)
    """

    def __init__(self, graph: CognitiveGraph) -> None:
        self._graph = graph
        self._concept_counter = 0
        # Prediction tracking: concept_id → list of (correct: bool)
        self._prediction_history: dict[str, list[bool]] = {}

    def register(
        self,
        feature_prototype: dict[str, Any],
        member_ids: list[str],
        behavioral_regularities: list[str] | None = None,
        causal_rule_ids: list[str] | None = None,
        domain: str = "",
        formation_source: str = "",
        formation_score: dict[str, float] | None = None,
        utility_delta: float = 0.0,
        concept_name: str = "",
    ) -> str:
        """Register a validated grounded concept in HCIR.

        Creates INSTANCE_OF edges (validated membership, not EXEMPLAR_OF).

        Returns:
            The ID of the created GroundedConceptNode.
        """
        self._concept_counter += 1
        name = concept_name or f"C-{self._concept_counter:05d}"

        concept = GroundedConceptNode(
            concept_name=name,
            feature_prototype=feature_prototype,
            behavioral_regularities=behavioral_regularities or [],
            causal_rule_ids=causal_rule_ids or [],
            prediction_accuracy=0.5,
            confidence=0.5 + min(utility_delta, 0.3),  # Initial from utility
            formation_score=formation_score or {},
            exemplar_count=len(member_ids),
            domain=domain,
            formation_source=formation_source,
            utility_delta=utility_delta,
            tags=["grounded_concept", domain, formation_source],
        )
        self._graph.add_node(concept)

        # Create INSTANCE_OF edges (validated — passed predictive utility test)
        for member_id in member_ids:
            if self._graph.get_node(member_id) is not None:
                self._graph.add_edge(HCIREdge(
                    edge_type=HCIREdgeType.INSTANCE_OF,
                    sources=[member_id],
                    targets=[concept.id],
                ))

        self._prediction_history[concept.id] = []

        logger.debug(
            "GroundedConceptRegistry: registered concept %s (%s, %d members, Δ=%.3f)",
            concept.id, name, len(member_ids), utility_delta,
        )

        return concept.id

    def get_concept(self, concept_id: str) -> GroundedConceptNode | None:
        """Retrieve a concept by ID."""
        node = self._graph.get_node(concept_id)
        if isinstance(node, GroundedConceptNode):
            return node
        return None

    def record_prediction(
        self,
        concept_id: str,
        correct: bool,
    ) -> None:
        """Record a prediction outcome for confidence tracking.

        Confidence is derived from prediction performance:
            confidence = f(recent_success_rate, behavioral_coherence)
        """
        if concept_id not in self._prediction_history:
            self._prediction_history[concept_id] = []
        self._prediction_history[concept_id].append(correct)

        concept = self.get_concept(concept_id)
        if concept is None:
            return

        concept.prediction_count += 1

        # Update prediction accuracy (exponential moving average)
        alpha = 0.15
        outcome = 1.0 if correct else 0.0
        concept.prediction_accuracy = (
            concept.prediction_accuracy * (1 - alpha) + outcome * alpha
        )

        # Update confidence from prediction performance
        recent = self._prediction_history[concept_id][-20:]
        if len(recent) >= 3:
            success_rate = sum(recent) / len(recent)
            concept.confidence = success_rate

        self._graph.upsert_node(concept)

    def concept_members(self, concept_id: str) -> list[str]:
        """Get all entity IDs that are INSTANCE_OF this concept."""
        members: list[str] = []
        for edge in self._graph.edges_to(concept_id):
            if edge.edge_type == HCIREdgeType.INSTANCE_OF:
                members.extend(edge.sources)
        return members

    def entity_concepts(self, entity_id: str) -> list[str]:
        """Get all concept IDs that this entity is INSTANCE_OF."""
        concepts: list[str] = []
        for edge in self._graph.edges_from(entity_id):
            if edge.edge_type == HCIREdgeType.INSTANCE_OF:
                concepts.extend(edge.targets)
        return concepts

    def all_concepts(self) -> list[GroundedConceptNode]:
        """Return all grounded concepts."""
        return [
            node for node in self._graph.nodes_by_type(HCIRNodeType.GROUNDED_CONCEPT)
            if isinstance(node, GroundedConceptNode)
        ]

    @property
    def total_concepts(self) -> int:
        return len(list(self._graph.nodes_by_type(HCIRNodeType.GROUNDED_CONCEPT)))
