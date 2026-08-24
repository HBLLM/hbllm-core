"""Object Permanence — multi-dimensional persistence predictions for A13.

When an entity becomes occluded (no longer observed), the system should
not simply forget it.  Object permanence generates *multiple independent
predictions* about the hidden entity's continued state.

Prediction dimensions::

    Object permanence
        ├── existence persistence    → "E17 still exists"
        ├── location persistence     → "E17 remains inside R3"
        ├── property persistence     → "E17 remains approximately round"
        └── relation persistence     → "E17 remains near E22"

Each prediction has its own confidence decay curve.  A later observation
can invalidate ONE prediction without invalidating the entity itself::

    Observation: E17 is now outside R3
        → PredictionError: location persistence violated
        → But existence/property predictions remain valid

**HCIR invariant:** ObjectPermanence produces PredictionNode and
PredictionErrorNode entries but NEVER mutates world state directly.
It generates cognitive objects for HCIR; the reconciler and reasoning
runtime decide what to do with them.

Architecture::

    E17 becomes OCCLUDED
        ↓
    ObjectPermanence
        ↓
    PredictionNode("E17 still exists", confidence=0.95)
    PredictionNode("E17 remains in R3", confidence=0.80)
    PredictionNode("E17 is still round", confidence=0.92)
        ↓
    HCIR (canonical state)
        ↓
    A12 reasoning (AbductionOperator, etc.)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hbllm.hcir.graph import (
    CognitiveGraph,
    HCIREdge,
    HCIREdgeType,
    PhysicalEntityNode,
    PredictionErrorNode,
    PredictionNode,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Persistence Dimension
# ═══════════════════════════════════════════════════════════════════════════


class PersistenceDimension(StrEnum):
    """The dimension of permanence being predicted."""

    EXISTENCE = "existence"  # Entity still exists
    LOCATION = "location"  # Entity remains at last-known location
    PROPERTY = "property"  # Entity's properties remain stable
    RELATION = "relation"  # Entity's relations remain stable


# ═══════════════════════════════════════════════════════════════════════════
# Decay Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DecayConfig:
    """Confidence decay parameters for a persistence dimension.

    Confidence decays exponentially: c(t) = c₀ · exp(-λ · Δt)

    Different dimensions decay at different rates:
    - Location decays fastest (things move)
    - Relations decay somewhat fast (things separate)
    - Properties decay slowly (things don't change color easily)
    - Existence decays slowest (things don't vanish)
    """

    lambda_rate: float  # Decay rate (higher = faster decay)
    min_confidence: float = 0.01  # Floor confidence


DEFAULT_DECAY: dict[PersistenceDimension, DecayConfig] = {
    PersistenceDimension.EXISTENCE: DecayConfig(lambda_rate=0.001),  # Very slow
    PersistenceDimension.LOCATION: DecayConfig(lambda_rate=0.01),  # Fast
    PersistenceDimension.PROPERTY: DecayConfig(lambda_rate=0.003),  # Slow
    PersistenceDimension.RELATION: DecayConfig(lambda_rate=0.007),  # Moderate
}


# ═══════════════════════════════════════════════════════════════════════════
# Permanence Prediction — descriptor for a single persistence prediction
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PermanencePrediction:
    """A prediction about a hidden entity's continued state."""

    entity_id: str
    dimension: PersistenceDimension
    claim: str  # Human-readable claim
    initial_confidence: float = 0.95
    occlusion_time: float = 0.0  # When the entity was last observed
    predicted_state: dict[str, Any] = field(default_factory=dict)
    prediction_node_id: str | None = None  # ID of the HCIR PredictionNode


# ═══════════════════════════════════════════════════════════════════════════
# Object Permanence Engine
# ═══════════════════════════════════════════════════════════════════════════


class ObjectPermanence:
    """Multi-dimensional persistence prediction engine for A13.

    Generates predictions about occluded entities across multiple
    dimensions (existence, location, property, relation), each with
    independent confidence decay.

    **Responsibility:** "What should continue to exist while unobserved?"

    **Does NOT:**
    - Mutate entity state directly
    - Own world state
    - Decide truth

    **Does:**
    - Generate PredictionNode entries for HCIR
    - Compute confidence decay over time
    - Detect prediction errors when re-observation contradicts predictions
    - Generate PredictionErrorNode entries

    Usage::

        permanence = ObjectPermanence(graph)

        # When entity becomes occluded
        predictions = permanence.generate_predictions(entity_id, t_occlusion)

        # Query current confidence
        conf = permanence.current_confidence(entity_id, PersistenceDimension.LOCATION)

        # When entity is re-observed, check for errors
        errors = permanence.check_against_observation(entity_id, observed_props)
    """

    def __init__(
        self,
        graph: CognitiveGraph,
        decay_config: dict[PersistenceDimension, DecayConfig] | None = None,
    ) -> None:
        self._graph = graph
        self._decay = decay_config or dict(DEFAULT_DECAY)
        # Tracking: entity_id → list of active permanence predictions
        self._active_predictions: dict[str, list[PermanencePrediction]] = {}

    # ── Prediction Generation ─────────────────────────────────────────

    def generate_predictions(
        self,
        entity_id: str,
        occlusion_time: float | None = None,
        spatial_context: dict[str, Any] | None = None,
        relation_context: list[tuple[str, str]] | None = None,
    ) -> list[PermanencePrediction]:
        """Generate multi-dimensional persistence predictions for an entity.

        Called when an entity transitions to OCCLUDED.  Creates a set of
        PredictionNode entries in HCIR, one per persistence dimension.

        Args:
            entity_id: The occluded entity.
            occlusion_time: When the entity was last observed.
            spatial_context: Last-known spatial information
                (e.g., {"container": "box", "region": "room"}).
            relation_context: Last-known relations as (relation_type, other_entity_id) pairs.

        Returns:
            List of generated PermanencePrediction descriptors.
        """
        t_occ = occlusion_time if occlusion_time is not None else time.time()
        entity = self._graph.get_node(entity_id)

        if entity is None or not isinstance(entity, PhysicalEntityNode):
            logger.warning("ObjectPermanence: entity %s not found", entity_id)
            return []

        predictions: list[PermanencePrediction] = []

        # 1. Existence prediction
        existence_pred = self._make_prediction(
            entity_id=entity_id,
            dimension=PersistenceDimension.EXISTENCE,
            claim=f"{entity.entity_name} still exists",
            initial_confidence=0.98,
            occlusion_time=t_occ,
            predicted_state={"exists": True},
        )
        predictions.append(existence_pred)

        # 2. Location prediction
        loc_state: dict[str, Any] = {}
        if spatial_context:
            loc_state = dict(spatial_context)
            location_pred = self._make_prediction(
                entity_id=entity_id,
                dimension=PersistenceDimension.LOCATION,
                claim=f"{entity.entity_name} remains at last-known location",
                initial_confidence=0.85,
                occlusion_time=t_occ,
                predicted_state=loc_state,
            )
            predictions.append(location_pred)

        # 3. Property predictions
        if entity.properties:
            prop_pred = self._make_prediction(
                entity_id=entity_id,
                dimension=PersistenceDimension.PROPERTY,
                claim=f"{entity.entity_name} properties remain stable",
                initial_confidence=0.92,
                occlusion_time=t_occ,
                predicted_state=dict(entity.properties),
            )
            predictions.append(prop_pred)

        # 4. Relation predictions
        if relation_context:
            rel_state = {
                "relations": [
                    {"type": rtype, "entity": eid}
                    for rtype, eid in relation_context
                ]
            }
            rel_pred = self._make_prediction(
                entity_id=entity_id,
                dimension=PersistenceDimension.RELATION,
                claim=f"{entity.entity_name} relations remain stable",
                initial_confidence=0.80,
                occlusion_time=t_occ,
                predicted_state=rel_state,
            )
            predictions.append(rel_pred)

        # Store active predictions
        self._active_predictions[entity_id] = predictions

        logger.debug(
            "ObjectPermanence: generated %d predictions for entity %s",
            len(predictions),
            entity_id,
        )

        return predictions

    # ── Confidence Computation ────────────────────────────────────────

    def current_confidence(
        self,
        entity_id: str,
        dimension: PersistenceDimension,
        current_time: float | None = None,
    ) -> float:
        """Compute the current confidence for a persistence prediction.

        Uses exponential decay: c(t) = c₀ · exp(-λ · Δt)

        Args:
            entity_id: The occluded entity.
            dimension: Which persistence dimension to query.
            current_time: Current time (defaults to now).

        Returns:
            Current confidence value (0.0 to 1.0).
        """
        now = current_time if current_time is not None else time.time()
        predictions = self._active_predictions.get(entity_id, [])

        for pred in predictions:
            if pred.dimension == dimension:
                decay_cfg = self._decay.get(dimension, DecayConfig(lambda_rate=0.005))
                dt = now - pred.occlusion_time
                confidence = pred.initial_confidence * math.exp(-decay_cfg.lambda_rate * dt)
                return max(confidence, decay_cfg.min_confidence)

        return 0.0

    def all_confidences(
        self,
        entity_id: str,
        current_time: float | None = None,
    ) -> dict[PersistenceDimension, float]:
        """Compute current confidence for all active prediction dimensions."""
        now = current_time if current_time is not None else time.time()
        result: dict[PersistenceDimension, float] = {}

        for dim in PersistenceDimension:
            conf = self.current_confidence(entity_id, dim, now)
            if conf > 0.0:
                result[dim] = conf

        return result

    # ── Observation Checking ──────────────────────────────────────────

    def check_against_observation(
        self,
        entity_id: str,
        observed_properties: dict[str, Any] | None = None,
        observed_location: dict[str, Any] | None = None,
        observed_relations: list[tuple[str, str]] | None = None,
        observation_time: float | None = None,
    ) -> list[PredictionErrorNode]:
        """Check active predictions against a new observation.

        For each prediction that is contradicted by the observation,
        generates a PredictionErrorNode in HCIR.

        A prediction violation in one dimension does NOT invalidate
        predictions in other dimensions.

        Args:
            entity_id: The entity being re-observed.
            observed_properties: Newly observed properties.
            observed_location: Newly observed location.
            observed_relations: Newly observed relations.
            observation_time: When the observation was made.

        Returns:
            List of PredictionErrorNode entries created in HCIR.
        """
        now = observation_time if observation_time is not None else time.time()
        predictions = self._active_predictions.get(entity_id, [])
        errors: list[PredictionErrorNode] = []

        for pred in predictions:
            error_node = self._check_single_prediction(
                pred, observed_properties, observed_location,
                observed_relations, now,
            )
            if error_node is not None:
                errors.append(error_node)

        # Clear predictions for this entity (they've been tested)
        if entity_id in self._active_predictions:
            del self._active_predictions[entity_id]

        return errors

    # ── Queries ───────────────────────────────────────────────────────

    def active_predictions_for(
        self,
        entity_id: str,
    ) -> list[PermanencePrediction]:
        """Return active predictions for an entity."""
        return list(self._active_predictions.get(entity_id, []))

    @property
    def tracked_entities(self) -> set[str]:
        """Entities with active permanence predictions."""
        return set(self._active_predictions.keys())

    # ── Internals ─────────────────────────────────────────────────────

    def _make_prediction(
        self,
        entity_id: str,
        dimension: PersistenceDimension,
        claim: str,
        initial_confidence: float,
        occlusion_time: float,
        predicted_state: dict[str, Any],
    ) -> PermanencePrediction:
        """Create a PredictionNode in HCIR and return a PermanencePrediction."""
        node = PredictionNode(
            claim=claim,
            predicted_outcome=f"permanence_{dimension}",
            tags=["object_permanence", str(dimension), entity_id],
        )
        self._graph.add_node(node)

        # Link prediction to entity
        edge = HCIREdge(
            edge_type=HCIREdgeType.PREDICTS,
            sources=[node.id],
            targets=[entity_id],
            properties={
                "dimension": str(dimension),
                "initial_confidence": initial_confidence,
            },
        )
        self._graph.add_edge(edge)

        return PermanencePrediction(
            entity_id=entity_id,
            dimension=dimension,
            claim=claim,
            initial_confidence=initial_confidence,
            occlusion_time=occlusion_time,
            predicted_state=predicted_state,
            prediction_node_id=node.id,
        )

    def _check_single_prediction(
        self,
        pred: PermanencePrediction,
        observed_properties: dict[str, Any] | None,
        observed_location: dict[str, Any] | None,
        observed_relations: list[tuple[str, str]] | None,
        observation_time: float,
    ) -> PredictionErrorNode | None:
        """Check a single prediction against observation data."""
        violated = False
        predicted_val: Any = None
        observed_val: Any = None

        if pred.dimension == PersistenceDimension.EXISTENCE:
            # Existence is confirmed by re-observation — no error
            return None

        elif pred.dimension == PersistenceDimension.LOCATION:
            if observed_location is not None and pred.predicted_state:
                # Compare predicted location with observed location
                predicted_val = pred.predicted_state
                observed_val = observed_location
                # Check if any key differs
                for key in pred.predicted_state:
                    if key in observed_location:
                        if pred.predicted_state[key] != observed_location[key]:
                            violated = True
                            break

        elif pred.dimension == PersistenceDimension.PROPERTY:
            if observed_properties is not None and pred.predicted_state:
                predicted_val = pred.predicted_state
                observed_val = observed_properties
                for key in pred.predicted_state:
                    if key in observed_properties:
                        if pred.predicted_state[key] != observed_properties[key]:
                            violated = True
                            break

        elif pred.dimension == PersistenceDimension.RELATION:
            if observed_relations is not None and pred.predicted_state:
                predicted_rels = pred.predicted_state.get("relations", [])
                predicted_val = predicted_rels
                observed_val = observed_relations
                # Check if predicted relations still hold
                observed_set = {(r, e) for r, e in observed_relations}
                for pred_rel in predicted_rels:
                    key = (pred_rel.get("type", ""), pred_rel.get("entity", ""))
                    if key not in observed_set:
                        violated = True
                        break

        if not violated:
            return None

        # Create PredictionErrorNode
        delta = self.current_confidence(
            pred.entity_id, pred.dimension, observation_time
        )

        error_node = PredictionErrorNode(
            prediction_id=pred.prediction_node_id or "",
            predicted_value=predicted_val,
            observed_value=observed_val,
            delta=delta,
            error_magnitude=delta,
            suspected_cause=f"{pred.dimension}_changed",
            tags=["object_permanence", str(pred.dimension), pred.entity_id],
        )
        self._graph.add_node(error_node)

        # Link error to prediction
        if pred.prediction_node_id:
            edge = HCIREdge(
                edge_type=HCIREdgeType.CONTRADICTS,
                sources=[error_node.id],
                targets=[pred.prediction_node_id],
            )
            self._graph.add_edge(edge)

        logger.debug(
            "ObjectPermanence: prediction error for entity %s dimension %s",
            pred.entity_id,
            pred.dimension,
        )

        return error_node
