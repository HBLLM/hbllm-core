"""Entity Registry — persistent entity identity management for A13 world model.

The central registry tracks every entity the system has perceived.  Each
entity is an *identity hypothesis* — the system's current best guess about
a persistent thing in the external world.

Core architectural distinction::

    EntityIdentity
        └── PhysicalEntityNode
                ├── ObservationNodes (linked via IDENTIFIES edge)
                ├── StateVersions (timestamped property snapshots)
                ├── SpatialRelations
                └── LifecycleEvents (via EventChronicle)

    "I saw something"           → ObservationNode
            ↓
    "That corresponds to E17"   → IDENTIFIES edge
            ↓
    "E17 currently exists"      → PhysicalEntityNode (with believed state)

Three observations of the same person produce three ObservationNode
entries, all linked to ONE PhysicalEntityNode via IDENTIFIES edges.

**HCIR invariant:** EntityRegistry writes PhysicalEntityNode, HCIREdge,
and ObservationNode entries into the CognitiveGraph.  Epistemic beliefs
(existence, identity, location) are NOT stored on the entity — they are
projections of epistemic state.

**Re-identification:** The registry does NOT eagerly merge entities.
It creates ``POTENTIAL_SAME_AS`` edges as identity hypotheses.  Epistemic
evaluation determines when identity becomes sufficiently believed,
at which point the edge is upgraded to ``IDENTIFIES``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.brain.world.event_chronicle import (
    ChronicleEvent,
    EventChronicle,
    WorldEventKind,
)
from hbllm.hcir.graph import (
    CognitiveGraph,
    EntityLifecycle,
    HCIREdge,
    HCIREdgeType,
    HCIRNodeType,
    PhysicalEntityNode,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# State Version — timestamped property snapshot
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class StateVersion:
    """A timestamped snapshot of an entity's properties.

    Enables temporal queries: "What was entity E17's state at time T?"
    """

    timestamp: float
    properties: dict[str, Any]
    observation_id: str | None = None  # Which observation triggered this version
    event_id: str | None = None  # Chronicle event recording this change


# ═══════════════════════════════════════════════════════════════════════════
# Identity Candidate — re-identification hypothesis
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class IdentityCandidate:
    """A candidate identity link between an observation and an entity.

    Represents the hypothesis: "this observation might correspond to
    this known entity."  The registry creates POTENTIAL_SAME_AS edges
    for these candidates until epistemic evaluation confirms them.
    """

    observation_id: str
    entity_id: str
    similarity_score: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)
    edge_id: str | None = None  # ID of the POTENTIAL_SAME_AS edge


# ═══════════════════════════════════════════════════════════════════════════
# Entity Registry
# ═══════════════════════════════════════════════════════════════════════════


class EntityRegistry:
    """Persistent entity identity manager for the A13 world model.

    Manages the lifecycle of persistent entity hypotheses in HCIR.
    The registry answers: "Which persistent entity does this observation
    belong to?"

    **Key design decisions:**

    1. Entity identity ≠ observation identity.  Multiple observations
       link to one entity via IDENTIFIES edges.

    2. No eager merge during re-identification.  Candidate identity
       links use POTENTIAL_SAME_AS edges until confirmed.

    3. RE_IDENTIFIED is an event, not a state.  When an entity is
       re-identified, it transitions OCCLUDED → TRACKED and a
       re-identification EventNode is recorded.

    4. No belief fields on entities.  Epistemic state owns beliefs.

    Usage::

        registry = EntityRegistry(graph, chronicle)

        # Discover a new entity from an observation
        entity_id = registry.discover("ball", "toy", observation_id, props)

        # Track subsequent observations
        registry.track_observation(entity_id, new_observation_id)

        # Entity becomes occluded
        registry.occlude(entity_id)

        # Later re-identification
        registry.propose_reidentification(new_obs_id, entity_id, score=0.85)
        registry.confirm_reidentification(new_obs_id, entity_id)
    """

    def __init__(
        self,
        graph: CognitiveGraph,
        chronicle: EventChronicle,
    ) -> None:
        self._graph = graph
        self._chronicle = chronicle
        # In-memory index: entity_id → list of StateVersion
        self._state_history: dict[str, list[StateVersion]] = {}

    # ── Discovery ─────────────────────────────────────────────────────

    def discover(
        self,
        entity_name: str,
        entity_type: str,
        observation_id: str | None = None,
        properties: dict[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> str:
        """Discover a new persistent entity.

        Creates a PhysicalEntityNode in DISCOVERED state, optionally
        linked to the triggering observation via an IDENTIFIES edge.

        Args:
            entity_name: Human-readable name for the entity.
            entity_type: Type classification (e.g., "toy", "person", "furniture").
            observation_id: Optional ID of the observation that discovered this entity.
            properties: Optional initial properties.
            timestamp: Optional discovery timestamp (defaults to now).

        Returns:
            The ID of the created PhysicalEntityNode.
        """
        now = timestamp if timestamp is not None else time.time()
        props = properties or {}

        # Create PhysicalEntityNode
        entity = PhysicalEntityNode(
            entity_name=entity_name,
            entity_type=entity_type,
            properties=props,
            entity_lifecycle=EntityLifecycle.DISCOVERED,
            first_observed_at=now,
            last_observed_at=now,
        )
        self._graph.add_node(entity)

        # Link to triggering observation
        if observation_id is not None:
            obs_node = self._graph.get_node(observation_id)
            if obs_node is not None:
                edge = HCIREdge(
                    edge_type=HCIREdgeType.IDENTIFIES,
                    sources=[observation_id],
                    targets=[entity.id],
                    properties={"association_type": "discovery"},
                )
                self._graph.add_edge(edge)

        # Record initial state version
        version = StateVersion(
            timestamp=now,
            properties=dict(props),
            observation_id=observation_id,
        )
        self._state_history[entity.id] = [version]

        # Record discovery event in chronicle
        event_id = self._chronicle.record(
            ChronicleEvent(
                event_kind=WorldEventKind.ENTITY_DISCOVERED,
                subject_entity_id=entity.id,
                timestamp=now,
                state_after=props,
                metadata={
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "observation_id": observation_id or "",
                },
            )
        )

        # Update state version with event ID
        self._state_history[entity.id][-1] = StateVersion(
            timestamp=now,
            properties=dict(props),
            observation_id=observation_id,
            event_id=event_id,
        )

        logger.debug(
            "EntityRegistry: discovered entity %s (%s/%s)",
            entity.id,
            entity_name,
            entity_type,
        )

        return entity.id

    # ── Lifecycle Transitions ─────────────────────────────────────────

    def track(self, entity_id: str, timestamp: float | None = None) -> None:
        """Transition an entity to TRACKED state.

        Called after discovery is confirmed or after re-identification.
        """
        now = timestamp if timestamp is not None else time.time()
        entity = self._get_entity(entity_id)
        if entity is None:
            return

        old_lifecycle = entity.entity_lifecycle
        entity.entity_lifecycle = EntityLifecycle.TRACKED
        entity.last_observed_at = now
        self._graph.upsert_node(entity)

        self._chronicle.record(
            ChronicleEvent(
                event_kind=WorldEventKind.ENTITY_TRACKED,
                subject_entity_id=entity_id,
                timestamp=now,
                state_before={"lifecycle": old_lifecycle},
                state_after={"lifecycle": EntityLifecycle.TRACKED},
            )
        )

    def occlude(self, entity_id: str, timestamp: float | None = None) -> None:
        """Transition an entity to OCCLUDED state.

        Called when the entity is no longer observable (e.g., sensor
        lost contact, object moved out of view).
        """
        now = timestamp if timestamp is not None else time.time()
        entity = self._get_entity(entity_id)
        if entity is None:
            return

        old_lifecycle = entity.entity_lifecycle
        entity.entity_lifecycle = EntityLifecycle.OCCLUDED
        self._graph.upsert_node(entity)

        self._chronicle.record(
            ChronicleEvent(
                event_kind=WorldEventKind.ENTITY_OCCLUDED,
                subject_entity_id=entity_id,
                timestamp=now,
                state_before={"lifecycle": old_lifecycle},
                state_after={"lifecycle": EntityLifecycle.OCCLUDED},
            )
        )

    def forget(self, entity_id: str, timestamp: float | None = None) -> None:
        """Transition an entity to FORGOTTEN state.

        Called when the system determines the entity is no longer relevant
        or has been occluded beyond the confidence threshold.
        """
        now = timestamp if timestamp is not None else time.time()
        entity = self._get_entity(entity_id)
        if entity is None:
            return

        old_lifecycle = entity.entity_lifecycle
        entity.entity_lifecycle = EntityLifecycle.FORGOTTEN
        self._graph.upsert_node(entity)

        self._chronicle.record(
            ChronicleEvent(
                event_kind=WorldEventKind.ENTITY_FORGOTTEN,
                subject_entity_id=entity_id,
                timestamp=now,
                state_before={"lifecycle": old_lifecycle},
                state_after={"lifecycle": EntityLifecycle.FORGOTTEN},
            )
        )

    # ── Observation Tracking ──────────────────────────────────────────

    def track_observation(
        self,
        entity_id: str,
        observation_id: str,
        updated_properties: dict[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Link a new observation to an existing entity.

        Creates an IDENTIFIES edge from the observation to the entity
        and optionally updates the entity's properties, recording a
        new state version.
        """
        now = timestamp if timestamp is not None else time.time()
        entity = self._get_entity(entity_id)
        if entity is None:
            return

        # Create IDENTIFIES edge
        edge = HCIREdge(
            edge_type=HCIREdgeType.IDENTIFIES,
            sources=[observation_id],
            targets=[entity_id],
            properties={"association_type": "tracking"},
        )
        self._graph.add_edge(edge)

        # Update entity temporal tracking
        entity.last_observed_at = now

        # If entity was occluded, transition back to tracked
        if entity.entity_lifecycle == EntityLifecycle.OCCLUDED:
            entity.entity_lifecycle = EntityLifecycle.TRACKED
        elif entity.entity_lifecycle == EntityLifecycle.DISCOVERED:
            entity.entity_lifecycle = EntityLifecycle.TRACKED

        # Update properties if provided
        if updated_properties:
            old_props = dict(entity.properties)
            entity.properties.update(updated_properties)

            # Record property change event
            self._chronicle.record(
                ChronicleEvent(
                    event_kind=WorldEventKind.PROPERTY_CHANGED,
                    subject_entity_id=entity_id,
                    timestamp=now,
                    state_before=old_props,
                    state_after=dict(entity.properties),
                )
            )

            # Record state version
            version = StateVersion(
                timestamp=now,
                properties=dict(entity.properties),
                observation_id=observation_id,
            )
            if entity_id not in self._state_history:
                self._state_history[entity_id] = []
            self._state_history[entity_id].append(version)

        self._graph.upsert_node(entity)

    # ── Re-identification ─────────────────────────────────────────────

    def propose_reidentification(
        self,
        observation_id: str,
        candidate_entity_id: str,
        similarity_score: float = 0.0,
        evidence: dict[str, Any] | None = None,
    ) -> IdentityCandidate:
        """Propose a candidate identity link (hypothesis).

        Creates a POTENTIAL_SAME_AS edge — NOT a confirmed identity.
        Epistemic evaluation must confirm before the observation is
        linked as IDENTIFIES.

        Args:
            observation_id: The new observation.
            candidate_entity_id: The entity it might correspond to.
            similarity_score: How similar the observation is to the entity.
            evidence: Supporting evidence for the identity hypothesis.

        Returns:
            An IdentityCandidate descriptor.
        """
        ev = evidence or {}

        edge = HCIREdge(
            edge_type=HCIREdgeType.POTENTIAL_SAME_AS,
            sources=[observation_id],
            targets=[candidate_entity_id],
            properties={
                "similarity_score": similarity_score,
                "evidence": ev,
            },
        )
        self._graph.add_edge(edge)

        return IdentityCandidate(
            observation_id=observation_id,
            entity_id=candidate_entity_id,
            similarity_score=similarity_score,
            evidence=ev,
            edge_id=edge.id,
        )

    def confirm_reidentification(
        self,
        observation_id: str,
        entity_id: str,
        timestamp: float | None = None,
    ) -> None:
        """Confirm a candidate identity — upgrade to IDENTIFIES edge.

        Removes the POTENTIAL_SAME_AS edge and creates an IDENTIFIES edge.
        If the entity was OCCLUDED, transitions it back to TRACKED and
        records a re-identification event in the chronicle.
        """
        now = timestamp if timestamp is not None else time.time()

        # Remove POTENTIAL_SAME_AS edges between this observation and entity
        edges_to_remove: list[str] = []
        for edge in self._graph.edges_from(observation_id):
            if (
                edge.edge_type == HCIREdgeType.POTENTIAL_SAME_AS
                and entity_id in edge.targets
            ):
                edges_to_remove.append(edge.id)

        for eid in edges_to_remove:
            self._graph.remove_edge(eid)

        # Create confirmed IDENTIFIES edge
        edge = HCIREdge(
            edge_type=HCIREdgeType.IDENTIFIES,
            sources=[observation_id],
            targets=[entity_id],
            properties={"association_type": "re_identification"},
        )
        self._graph.add_edge(edge)

        # Transition entity lifecycle if needed
        entity = self._get_entity(entity_id)
        if entity is not None:
            if entity.entity_lifecycle == EntityLifecycle.OCCLUDED:
                entity.entity_lifecycle = EntityLifecycle.TRACKED
                entity.last_observed_at = now
                self._graph.upsert_node(entity)

            # Record re-identification as a transition event (not a state)
            self._chronicle.record(
                ChronicleEvent(
                    event_kind=WorldEventKind.ENTITY_RE_IDENTIFIED,
                    subject_entity_id=entity_id,
                    timestamp=now,
                    metadata={
                        "observation_id": observation_id,
                        "association_type": "re_identification",
                    },
                )
            )

    # ── State History Queries ─────────────────────────────────────────

    def state_at(
        self,
        entity_id: str,
        timestamp: float,
    ) -> dict[str, Any] | None:
        """Query an entity's properties at a specific point in time.

        Returns the most recent state version at or before the given
        timestamp, or None if the entity has no recorded history.
        """
        versions = self._state_history.get(entity_id, [])
        if not versions:
            return None

        # Find the latest version at or before the timestamp
        result: StateVersion | None = None
        for v in versions:
            if v.timestamp <= timestamp:
                result = v
            else:
                break

        return dict(result.properties) if result is not None else None

    def state_history(self, entity_id: str) -> list[StateVersion]:
        """Return the full state history for an entity."""
        return list(self._state_history.get(entity_id, []))

    # ── Queries ───────────────────────────────────────────────────────

    def get_entity(self, entity_id: str) -> PhysicalEntityNode | None:
        """Retrieve a PhysicalEntityNode by ID."""
        return self._get_entity(entity_id)

    def observations_for_entity(self, entity_id: str) -> list[str]:
        """Return all observation IDs linked to an entity via IDENTIFIES edges."""
        obs_ids: list[str] = []
        for edge in self._graph.edges_to(entity_id):
            if edge.edge_type == HCIREdgeType.IDENTIFIES:
                obs_ids.extend(edge.sources)
        return obs_ids

    def entities_by_lifecycle(
        self,
        lifecycle: EntityLifecycle,
    ) -> list[PhysicalEntityNode]:
        """Return all entities in a given lifecycle state."""
        results: list[PhysicalEntityNode] = []
        for node in self._graph.nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY):
            if isinstance(node, PhysicalEntityNode):
                if node.entity_lifecycle == lifecycle:
                    results.append(node)
        return results

    def entities_by_type(self, entity_type: str) -> list[PhysicalEntityNode]:
        """Return all entities of a given type."""
        results: list[PhysicalEntityNode] = []
        for node in self._graph.nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY):
            if isinstance(node, PhysicalEntityNode):
                if node.entity_type == entity_type:
                    results.append(node)
        return results

    @property
    def total_entities(self) -> int:
        """Total number of registered entities."""
        return len(list(self._graph.nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY)))

    # ── Internals ─────────────────────────────────────────────────────

    def _get_entity(self, entity_id: str) -> PhysicalEntityNode | None:
        """Retrieve and validate a PhysicalEntityNode."""
        node = self._graph.get_node(entity_id)
        if node is None or not isinstance(node, PhysicalEntityNode):
            logger.warning("EntityRegistry: entity %s not found", entity_id)
            return None
        return node
