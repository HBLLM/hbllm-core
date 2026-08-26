"""World State Reconciler — perception ↔ world model boundary for A13.

The reconciliation boundary between new observations and the existing
world model.  When new observations arrive, the reconciler compares
them against the current world belief and produces structured deltas.

**Critical invariant:** The reconciler NEVER decides truth.  It produces
structured evidence for HCIR/epistemics to evaluate.

Architecture::

    current world belief
            VS
    new observations
            ↓
    ┌───────────────────┐
    │ state transition   │
    │ prediction confirm │
    │ prediction error   │
    │ identity candidate │
    │ relation update    │
    │ contradiction      │
    └───────────────────┘

Data flow::

    PERCEPTION
        ↓
    OBSERVATION
        ↓
    WORLD RECONCILER
        ↓
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
    EntityRegistry SpatialOntology Chronicle
        ↓              ↓              ↓
    Object Permanence
        ↓
    HCIR STATE
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hbllm.brain.world.entity_registry import EntityRegistry, IdentityCandidate
from hbllm.brain.world.event_chronicle import EventChronicle
from hbllm.brain.world.object_permanence import ObjectPermanence
from hbllm.brain.world.spatial_ontology import SpatialOntology
from hbllm.hcir.graph import (
    CognitiveGraph,
    EntityLifecycle,
    HCIRNodeType,
    ObservationNode,
    PhysicalEntityNode,
    PredictionErrorNode,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Delta Types — structured reconciliation output
# ═══════════════════════════════════════════════════════════════════════════


class DeltaType(StrEnum):
    """Type of reconciliation delta."""

    STATE_TRANSITION = "state_transition"  # Entity property changed
    PREDICTION_CONFIRMATION = "prediction_confirmation"  # Observation matches prediction
    PREDICTION_ERROR = "prediction_error"  # Observation contradicts prediction
    IDENTITY_CANDIDATE = "identity_candidate"  # Observation may match known entity
    RELATION_UPDATE = "relation_update"  # Spatial/temporal relation changed
    CONTRADICTION = "contradiction"  # Observation contradicts current world belief
    NEW_ENTITY = "new_entity"  # Observation represents unknown entity


@dataclass
class ReconciliationDelta:
    """A structured delta produced by the reconciler.

    Each delta describes a specific difference between the observed
    state and the believed state.  The reconciler does NOT resolve
    these — it produces them as evidence for HCIR/epistemics.
    """

    delta_type: DeltaType
    observation_id: str
    entity_id: str | None = None  # None if identity not yet resolved
    confidence: float = 1.0
    evidence: dict[str, Any] = field(default_factory=dict)
    details: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Reconciliation Result
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ReconciliationResult:
    """The complete result of reconciling an observation against the world model."""

    observation_id: str
    deltas: list[ReconciliationDelta] = field(default_factory=list)
    matched_entity_id: str | None = None  # Entity the observation was matched to
    identity_candidates: list[IdentityCandidate] = field(default_factory=list)
    prediction_errors: list[PredictionErrorNode] = field(default_factory=list)
    created_entity_id: str | None = None  # If a new entity was created
    timestamp: float = field(default_factory=time.time)


# ═══════════════════════════════════════════════════════════════════════════
# World State Reconciler
# ═══════════════════════════════════════════════════════════════════════════


class WorldStateReconciler:
    """The reconciliation boundary between perception and world model.

    Compares new observations against the current world belief and
    produces structured deltas.  Never resolves truth — only produces
    evidence.

    **Responsibility:** "What changed relative to what we believed?"

    Connects the A13 subsystems:
    - EntityRegistry: identity resolution
    - SpatialOntology: spatial relation updates
    - EventChronicle: state transition recording
    - ObjectPermanence: prediction checking

    Usage::

        reconciler = WorldStateReconciler(
            graph=graph,
            entity_registry=registry,
            spatial_ontology=ontology,
            event_chronicle=chronicle,
            object_permanence=permanence,
        )

        # Reconcile a new observation
        result = reconciler.reconcile(observation_node)
    """

    def __init__(
        self,
        graph: CognitiveGraph,
        entity_registry: EntityRegistry,
        spatial_ontology: SpatialOntology,
        event_chronicle: EventChronicle,
        object_permanence: ObjectPermanence,
    ) -> None:
        self._graph = graph
        self._registry = entity_registry
        self._spatial = spatial_ontology
        self._chronicle = event_chronicle
        self._permanence = object_permanence

    # ── Main Reconciliation ───────────────────────────────────────────

    def reconcile(
        self,
        observation: ObservationNode,
        entity_hint: str | None = None,
        timestamp: float | None = None,
    ) -> ReconciliationResult:
        """Reconcile a new observation against the current world model.

        This is the main entry point.  The reconciliation process:

        1. Try to match the observation to a known entity
        2. If matched: compare observed state with believed state
        3. If occluded entity: check permanence predictions
        4. If no match: create identity candidates or new entity

        Args:
            observation: The new ObservationNode from perception.
            entity_hint: Optional hint for which entity this observation
                belongs to (e.g., from a tracker).
            timestamp: Optional reconciliation timestamp.

        Returns:
            A ReconciliationResult with all produced deltas.
        """
        now = timestamp if timestamp is not None else time.time()
        result = ReconciliationResult(
            observation_id=observation.id,
            timestamp=now,
        )

        # Step 1: Attempt identity resolution
        matched_entity = self._resolve_identity(observation, entity_hint, result)

        if matched_entity is not None:
            result.matched_entity_id = matched_entity.id

            # Step 2: Compare observed state with believed state
            self._compare_state(observation, matched_entity, result, now)

            # Step 3: Check permanence predictions if entity was occluded
            if matched_entity.entity_lifecycle == EntityLifecycle.OCCLUDED:
                self._check_permanence(observation, matched_entity, result, now)

            # Step 4: Update entity tracking
            self._registry.track_observation(
                entity_id=matched_entity.id,
                observation_id=observation.id,
                updated_properties=observation.payload,
                timestamp=now,
            )

        else:
            # Step 5: No match — create new entity or identity candidates
            self._handle_unmatched(observation, result, now)

        logger.debug(
            "Reconciler: observation %s produced %d deltas (entity: %s)",
            observation.id,
            len(result.deltas),
            result.matched_entity_id or result.created_entity_id or "none",
        )

        return result

    # ── Identity Resolution ───────────────────────────────────────────

    def _resolve_identity(
        self,
        observation: ObservationNode,
        entity_hint: str | None,
        result: ReconciliationResult,
    ) -> PhysicalEntityNode | None:
        """Try to match an observation to a known entity.

        Resolution priority:
        1. Direct entity hint (from tracker or tag)
        2. Observation payload matching against tracked entities
        """
        # Priority 1: Direct hint
        if entity_hint is not None:
            entity = self._registry.get_entity(entity_hint)
            if entity is not None:
                return entity

        # Priority 2: Payload-based matching
        obs_type = observation.payload.get("entity_type", "")
        obs_name = observation.payload.get("entity_name", "")

        if not obs_type and not obs_name:
            return None

        # Search tracked entities for matches
        candidates: list[tuple[PhysicalEntityNode, float]] = []

        for node in self._graph.nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY):
            if not isinstance(node, PhysicalEntityNode):
                continue

            score = self._compute_match_score(observation, node)
            if score > 0.5:
                candidates.append((node, score))

        if not candidates:
            return None

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_entity, best_score = candidates[0]

        # High confidence match → direct identity
        if best_score >= 0.8:
            return best_entity

        # Medium confidence → identity candidate (hypothesis)
        for entity, score in candidates:
            candidate = self._registry.propose_reidentification(
                observation_id=observation.id,
                candidate_entity_id=entity.id,
                similarity_score=score,
                evidence={"payload_match": True, "score": score},
            )
            result.identity_candidates.append(candidate)

            result.deltas.append(
                ReconciliationDelta(
                    delta_type=DeltaType.IDENTITY_CANDIDATE,
                    observation_id=observation.id,
                    entity_id=entity.id,
                    confidence=score,
                    evidence={"similarity_score": score},
                    details=f"Potential match with {entity.entity_name} (score={score:.2f})",
                )
            )

        # If best candidate is strong enough, auto-confirm
        if best_score >= 0.7:
            self._registry.confirm_reidentification(
                observation_id=observation.id,
                entity_id=best_entity.id,
            )
            return best_entity

        return None

    def _compute_match_score(
        self,
        observation: ObservationNode,
        entity: PhysicalEntityNode,
    ) -> float:
        """Compute similarity between an observation and a known entity."""
        score = 0.0
        total_factors = 0

        obs_payload = observation.payload

        # Name match
        obs_name = obs_payload.get("entity_name", "")
        if obs_name and entity.entity_name:
            total_factors += 1
            if obs_name == entity.entity_name:
                score += 1.0

        # Type match
        obs_type = obs_payload.get("entity_type", "")
        if obs_type and entity.entity_type:
            total_factors += 1
            if obs_type == entity.entity_type:
                score += 1.0

        # Property overlap
        obs_props = {
            k: v for k, v in obs_payload.items() if k not in ("entity_name", "entity_type")
        }
        if obs_props and entity.properties:
            total_factors += 1
            matching = sum(
                1
                for k, v in obs_props.items()
                if k in entity.properties and entity.properties[k] == v
            )
            total_prop_keys = max(len(obs_props), len(entity.properties))
            if total_prop_keys > 0:
                score += matching / total_prop_keys

        if total_factors == 0:
            return 0.0

        return score / total_factors

    # ── State Comparison ──────────────────────────────────────────────

    def _compare_state(
        self,
        observation: ObservationNode,
        entity: PhysicalEntityNode,
        result: ReconciliationResult,
        now: float,
    ) -> None:
        """Compare observed state against believed state."""
        obs_props = observation.payload
        entity_props = entity.properties

        changed_keys: list[str] = []
        confirmed_keys: list[str] = []

        for key, obs_val in obs_props.items():
            if key in ("entity_name", "entity_type"):
                continue

            if key in entity_props:
                if entity_props[key] != obs_val:
                    changed_keys.append(key)
                else:
                    confirmed_keys.append(key)

        # Produce state transition deltas for changed properties
        if changed_keys:
            result.deltas.append(
                ReconciliationDelta(
                    delta_type=DeltaType.STATE_TRANSITION,
                    observation_id=observation.id,
                    entity_id=entity.id,
                    confidence=0.9,
                    evidence={
                        "changed_keys": changed_keys,
                        "old_values": {k: entity_props.get(k) for k in changed_keys},
                        "new_values": {k: obs_props[k] for k in changed_keys},
                    },
                    details=f"Properties changed: {', '.join(changed_keys)}",
                )
            )

        # Produce confirmation deltas
        if confirmed_keys:
            result.deltas.append(
                ReconciliationDelta(
                    delta_type=DeltaType.PREDICTION_CONFIRMATION,
                    observation_id=observation.id,
                    entity_id=entity.id,
                    confidence=1.0,
                    evidence={"confirmed_keys": confirmed_keys},
                    details=f"Properties confirmed: {', '.join(confirmed_keys)}",
                )
            )

        # Check for contradictions (entity claims one thing, observation claims opposite)
        for key in changed_keys:
            old_val = entity_props.get(key)
            new_val = obs_props[key]

            # Strong contradiction: boolean flip or categorical change
            if isinstance(old_val, bool) and isinstance(new_val, bool) and old_val != new_val:
                result.deltas.append(
                    ReconciliationDelta(
                        delta_type=DeltaType.CONTRADICTION,
                        observation_id=observation.id,
                        entity_id=entity.id,
                        confidence=0.9,
                        evidence={
                            "key": key,
                            "believed": old_val,
                            "observed": new_val,
                        },
                        details=f"Contradiction: {key} believed={old_val}, observed={new_val}",
                    )
                )

    # ── Permanence Checking ───────────────────────────────────────────

    def _check_permanence(
        self,
        observation: ObservationNode,
        entity: PhysicalEntityNode,
        result: ReconciliationResult,
        now: float,
    ) -> None:
        """Check permanence predictions against a re-observation."""
        obs_props = {
            k: v for k, v in observation.payload.items() if k not in ("entity_name", "entity_type")
        }

        obs_location = observation.payload.get("location")
        if isinstance(obs_location, dict):
            location_data = obs_location
        else:
            location_data = None

        errors = self._permanence.check_against_observation(
            entity_id=entity.id,
            observed_properties=obs_props,
            observed_location=location_data,
            observation_time=now,
        )

        result.prediction_errors.extend(errors)

        for error in errors:
            result.deltas.append(
                ReconciliationDelta(
                    delta_type=DeltaType.PREDICTION_ERROR,
                    observation_id=observation.id,
                    entity_id=entity.id,
                    confidence=0.9,
                    evidence={
                        "predicted": error.predicted_value,
                        "observed": error.observed_value,
                        "error_magnitude": error.error_magnitude,
                    },
                    details=f"Permanence prediction violated: {error.suspected_cause}",
                )
            )

    # ── Unmatched Observation Handling ─────────────────────────────────

    def _handle_unmatched(
        self,
        observation: ObservationNode,
        result: ReconciliationResult,
        now: float,
    ) -> None:
        """Handle an observation that couldn't be matched to any entity."""
        obs_name = observation.payload.get("entity_name", "unknown")
        obs_type = observation.payload.get("entity_type", "unknown")

        # Create a new entity
        props = {
            k: v for k, v in observation.payload.items() if k not in ("entity_name", "entity_type")
        }

        entity_id = self._registry.discover(
            entity_name=obs_name,
            entity_type=obs_type,
            observation_id=observation.id,
            properties=props,
            timestamp=now,
        )

        result.created_entity_id = entity_id
        result.deltas.append(
            ReconciliationDelta(
                delta_type=DeltaType.NEW_ENTITY,
                observation_id=observation.id,
                entity_id=entity_id,
                confidence=1.0,
                evidence={
                    "entity_name": obs_name,
                    "entity_type": obs_type,
                    "properties": props,
                },
                details=f"New entity discovered: {obs_name} ({obs_type})",
            )
        )
