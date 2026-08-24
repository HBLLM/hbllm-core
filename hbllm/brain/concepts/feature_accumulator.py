"""Feature Accumulator — multi-dimensional entity feature extraction for A15.

Builds rich feature vectors from A13 world state + A14 epistemic state.

Five feature dimensions::

    EntityFeatureVector
    │
    ├── appearance   — observable properties from PhysicalEntityNode
    ├── behavior     — event patterns from EventChronicle
    ├── relational   — spatial/identity relations
    ├── temporal     — persistence, transitions, lifecycle
    └── epistemic    — prediction accuracy, error profile, rules (from A14)

The **epistemic features** are critical: two objects might look different
but behave identically and belong to the same functional concept.

Derived epistemic features carry provenance — they are traceable to
the predictions and adaptations that produced them, preventing A15
from treating an epistemic estimate as an unquestionable property.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Feature Vector
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AppearanceFeatures:
    """Observable properties from PhysicalEntityNode."""

    entity_type: str = ""
    properties: dict[str, Any] = field(default_factory=dict)  # color, shape, size, etc.


@dataclass
class BehaviorFeatures:
    """Event patterns and interaction dynamics."""

    stationary_rate: float = 0.0  # Fraction of time stationary
    motion_patterns: list[str] = field(default_factory=list)
    interaction_frequency: float = 0.0
    event_type_distribution: dict[str, float] = field(default_factory=dict)


@dataclass
class RelationalFeatures:
    """Spatial and identity relations."""

    spatial_relations: dict[str, int] = field(default_factory=dict)  # relation_type → count
    containment_role: str = ""  # "container", "contained", "none"
    support_role: str = ""  # "supporter", "supported", "none"
    co_occurrence_partners: list[str] = field(default_factory=list)  # entity types seen with


@dataclass
class TemporalFeatures:
    """Persistence and lifecycle dynamics."""

    persistence_duration: float = 0.0  # Seconds observed
    state_transition_count: int = 0
    lifecycle_stage: str = ""  # "new", "stable", "disappearing"


@dataclass
class EpistemicFeatures:
    """Prediction and learning features from A14.

    These carry implicit provenance — each value is traceable to
    the prediction history that produced it.
    """

    prediction_accuracy: float = 0.5
    error_profile: dict[str, float] = field(default_factory=dict)  # error_type → rate
    model_confidence: float = 0.5
    associated_rule_count: int = 0
    associated_rule_ids: list[str] = field(default_factory=list)
    adaptation_count: int = 0  # How many times models adapted for this entity


@dataclass
class EntityFeatureVector:
    """Multi-dimensional feature vector for a single entity.

    Combines raw observable features with derived epistemic features.
    Used by ConceptHypothesisGenerator for concept candidate discovery.
    """

    entity_id: str = ""
    entity_type: str = ""
    appearance: AppearanceFeatures = field(default_factory=AppearanceFeatures)
    behavior: BehaviorFeatures = field(default_factory=BehaviorFeatures)
    relational: RelationalFeatures = field(default_factory=RelationalFeatures)
    temporal: TemporalFeatures = field(default_factory=TemporalFeatures)
    epistemic: EpistemicFeatures = field(default_factory=EpistemicFeatures)


# ═══════════════════════════════════════════════════════════════════════════
# Feature Accumulator
# ═══════════════════════════════════════════════════════════════════════════


class FeatureAccumulator:
    """Builds multi-dimensional feature vectors from A13/A14 state.

    Queries the HCIR graph to extract features across all five
    dimensions for each entity.

    Usage::

        accumulator = FeatureAccumulator(graph)
        features = accumulator.accumulate_all()
        # → dict[entity_id, EntityFeatureVector]
    """

    def __init__(self, graph: Any) -> None:
        self._graph = graph

    def accumulate_all(self) -> dict[str, EntityFeatureVector]:
        """Extract feature vectors for all PhysicalEntityNodes."""
        from hbllm.hcir.graph import HCIRNodeType, PhysicalEntityNode

        features: dict[str, EntityFeatureVector] = {}

        for node in self._graph.nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY):
            if isinstance(node, PhysicalEntityNode):
                fv = self.accumulate_entity(node)
                features[node.id] = fv

        return features

    def accumulate_entity(self, entity: Any) -> EntityFeatureVector:
        """Extract a feature vector for a single entity."""
        fv = EntityFeatureVector(
            entity_id=entity.id,
            entity_type=getattr(entity, "entity_type", ""),
        )

        # Appearance
        fv.appearance = self._extract_appearance(entity)

        # Behavior
        fv.behavior = self._extract_behavior(entity)

        # Relational
        fv.relational = self._extract_relational(entity)

        # Temporal
        fv.temporal = self._extract_temporal(entity)

        # Epistemic
        fv.epistemic = self._extract_epistemic(entity)

        return fv

    # ── Dimension Extractors ──────────────────────────────────────────

    def _extract_appearance(self, entity: Any) -> AppearanceFeatures:
        """Extract observable properties."""
        props = {}
        if hasattr(entity, "observed_properties"):
            props = dict(entity.observed_properties)
        elif hasattr(entity, "properties"):
            props = dict(entity.properties)

        return AppearanceFeatures(
            entity_type=getattr(entity, "entity_type", ""),
            properties=props,
        )

    def _extract_behavior(self, entity: Any) -> BehaviorFeatures:
        """Extract behavioral features from event history."""
        from hbllm.hcir.graph import HCIRNodeType

        events: list[Any] = []
        for edge in self._graph.edges_from(entity.id):
            for target_id in edge.targets:
                target = self._graph.get_node(target_id)
                if target and target.node_type == HCIRNodeType.EVENT:
                    events.append(target)

        event_dist: dict[str, float] = {}
        for event in events:
            event_type = getattr(event, "event_type", "unknown")
            event_dist[event_type] = event_dist.get(event_type, 0) + 1

        # Normalize
        total = sum(event_dist.values()) or 1
        event_dist = {k: v / total for k, v in event_dist.items()}

        return BehaviorFeatures(
            stationary_rate=event_dist.get("stationary", 0.0),
            interaction_frequency=len(events),
            event_type_distribution=event_dist,
        )

    def _extract_relational(self, entity: Any) -> RelationalFeatures:
        """Extract spatial/relational features."""
        from hbllm.hcir.graph import HCIREdgeType

        spatial: dict[str, int] = {}
        spatial_types = {
            HCIREdgeType.LOCATED_IN, HCIREdgeType.ABOVE, HCIREdgeType.BELOW,
            HCIREdgeType.NEAR, HCIREdgeType.TOUCHING,
        }

        for edge in self._graph.edges_from(entity.id):
            if edge.edge_type in spatial_types:
                rel = str(edge.edge_type)
                spatial[rel] = spatial.get(rel, 0) + 1

        containment = ""
        support = ""
        # Check if entity contains or is contained
        for edge in self._graph.edges_from(entity.id):
            if edge.edge_type == HCIREdgeType.LOCATED_IN:
                containment = "contained"
        for edge in self._graph.edges_to(entity.id):
            if edge.edge_type == HCIREdgeType.LOCATED_IN:
                containment = "container"

        return RelationalFeatures(
            spatial_relations=spatial,
            containment_role=containment,
            support_role=support,
        )

    def _extract_temporal(self, entity: Any) -> TemporalFeatures:
        """Extract temporal/lifecycle features."""
        persistence = getattr(entity, "persistence_duration", 0.0)
        transitions = getattr(entity, "state_transition_count", 0)
        lifecycle = getattr(entity, "lifecycle_stage", "stable")

        return TemporalFeatures(
            persistence_duration=persistence,
            state_transition_count=transitions,
            lifecycle_stage=lifecycle,
        )

    def _extract_epistemic(self, entity: Any) -> EpistemicFeatures:
        """Extract A14 epistemic features with provenance."""
        from hbllm.hcir.graph import HCIREdgeType, HCIRNodeType

        prediction_accuracy = 0.5
        rule_ids: list[str] = []
        adaptation_count = 0

        # Find associated LearnedRules via APPLIES_TO edges
        for edge in self._graph.edges_to(entity.id):
            if edge.edge_type == HCIREdgeType.APPLIES_TO:
                for src_id in edge.sources:
                    src = self._graph.get_node(src_id)
                    if src and src.node_type == HCIRNodeType.LEARNED_RULE:
                        rule_ids.append(src_id)

        # Find associated PredictiveModel accuracy
        for edge in self._graph.edges_from(entity.id):
            for target_id in edge.targets:
                target = self._graph.get_node(target_id)
                if target and target.node_type == HCIRNodeType.PREDICTIVE_MODEL:
                    prediction_accuracy = getattr(target, "accuracy", 0.5)
                    adaptation_count = getattr(target, "adaptation_count", 0)

        return EpistemicFeatures(
            prediction_accuracy=prediction_accuracy,
            associated_rule_count=len(rule_ids),
            associated_rule_ids=rule_ids,
            adaptation_count=adaptation_count,
        )

    # ── Feature Distance ──────────────────────────────────────────────

    @staticmethod
    def feature_distance(a: EntityFeatureVector, b: EntityFeatureVector) -> dict[str, float]:
        """Compute per-dimension distance between two feature vectors.

        Returns a dict of dimension → distance (0 = identical, 1 = maximally different).
        """
        distances: dict[str, float] = {}

        # Appearance: property overlap
        a_props = set(a.appearance.properties.keys())
        b_props = set(b.appearance.properties.keys())
        if a_props or b_props:
            overlap = len(a_props & b_props) / max(len(a_props | b_props), 1)
            # Value similarity for shared keys
            shared = a_props & b_props
            if shared:
                matching = sum(
                    1 for k in shared
                    if a.appearance.properties.get(k) == b.appearance.properties.get(k)
                )
                val_sim = matching / len(shared)
                distances["appearance"] = 1.0 - (overlap * 0.5 + val_sim * 0.5)
            else:
                distances["appearance"] = 1.0 - overlap
        else:
            distances["appearance"] = 0.5  # Unknown

        # Behavior: event distribution similarity (cosine-like)
        a_events = a.behavior.event_type_distribution
        b_events = b.behavior.event_type_distribution
        all_events = set(a_events.keys()) | set(b_events.keys())
        if all_events:
            dot = sum(a_events.get(e, 0) * b_events.get(e, 0) for e in all_events)
            norm_a = math.sqrt(sum(v ** 2 for v in a_events.values())) or 1
            norm_b = math.sqrt(sum(v ** 2 for v in b_events.values())) or 1
            cosine = dot / (norm_a * norm_b)
            distances["behavior"] = 1.0 - max(0, cosine)
        else:
            distances["behavior"] = 0.5

        # Relational: spatial relation overlap
        a_rels = set(a.relational.spatial_relations.keys())
        b_rels = set(b.relational.spatial_relations.keys())
        if a_rels or b_rels:
            distances["relational"] = 1.0 - len(a_rels & b_rels) / max(len(a_rels | b_rels), 1)
        else:
            distances["relational"] = 0.5

        # Temporal: persistence similarity
        max_p = max(a.temporal.persistence_duration, b.temporal.persistence_duration, 1)
        distances["temporal"] = abs(
            a.temporal.persistence_duration - b.temporal.persistence_duration
        ) / max_p

        # Epistemic: prediction accuracy similarity
        distances["epistemic"] = abs(
            a.epistemic.prediction_accuracy - b.epistemic.prediction_accuracy
        )

        return distances
