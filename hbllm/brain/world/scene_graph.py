"""Scene Graph — unified read-only projection of A13 world state.

The SceneGraph is a materialized view of world state, NEVER the owner
of world state.  It provides a coherent projection of the entity registry,
spatial ontology, event chronicle, and object permanence into a single
queryable structure.

**Critical invariant:**

    HCIR = source of truth

    EntityRegistry
    SpatialOntology
    EventChronicle
    ObjectPermanence
            ↓
        SceneGraph (read-only projection)
            ↓
       FrozenGraphView
            ↓
          A12

SceneGraph NEVER writes independently into HCIR.

**Responsibility:** "What does the world currently look like as a
queryable whole?"
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hbllm.brain.world.entity_registry import EntityRegistry
from hbllm.brain.world.event_chronicle import EventChronicle
from hbllm.brain.world.object_permanence import ObjectPermanence, PersistenceDimension
from hbllm.brain.world.spatial_ontology import SpatialOntology, SpatialRelation
from hbllm.hcir.graph import (
    CognitiveGraph,
    EntityLifecycle,
    EventNode,
    HCIRNodeType,
    PhysicalEntityNode,
)

if TYPE_CHECKING:
    from hbllm.brain.reasoning.operators.base import FrozenGraphView

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Scene Entity — projected entity with all context
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SceneEntity:
    """A projected entity with all available world-model context.

    This is a read-only view combining entity state, spatial relations,
    recent events, and permanence predictions into a single structure.
    """

    entity_id: str
    entity_name: str
    entity_type: str
    lifecycle: EntityLifecycle
    properties: dict[str, Any] = field(default_factory=dict)
    last_observed_at: float | None = None
    first_observed_at: float | None = None
    observation_count: int = 0

    # Spatial context
    spatial_relations: list[SpatialRelation] = field(default_factory=list)
    containers: list[str] = field(default_factory=list)  # LOCATED_IN targets
    nearby_entities: list[str] = field(default_factory=list)  # NEAR targets

    # Temporal context
    recent_events: list[EventNode] = field(default_factory=list)

    # Permanence predictions (only for occluded entities)
    permanence_confidences: dict[str, float] = field(default_factory=dict)

    # Self/world boundary
    is_self: bool = False  # True if this entity represents the agent


# ═══════════════════════════════════════════════════════════════════════════
# Scene Snapshot — point-in-time world projection
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SceneSnapshot:
    """A point-in-time projection of the entire world state.

    Contains all tracked entities with their full context, organized
    by lifecycle state.
    """

    timestamp: float = field(default_factory=time.time)
    entities: list[SceneEntity] = field(default_factory=list)

    # Convenience accessors
    @property
    def tracked_entities(self) -> list[SceneEntity]:
        """Entities currently being tracked (actively observed)."""
        return [e for e in self.entities if e.lifecycle == EntityLifecycle.TRACKED]

    @property
    def occluded_entities(self) -> list[SceneEntity]:
        """Entities that are occluded (hidden but believed to exist)."""
        return [e for e in self.entities if e.lifecycle == EntityLifecycle.OCCLUDED]

    @property
    def discovered_entities(self) -> list[SceneEntity]:
        """Newly discovered entities not yet fully tracked."""
        return [e for e in self.entities if e.lifecycle == EntityLifecycle.DISCOVERED]

    @property
    def entity_count(self) -> int:
        """Total number of entities in the scene."""
        return len(self.entities)

    def entity_by_id(self, entity_id: str) -> SceneEntity | None:
        """Look up a specific entity by ID."""
        for e in self.entities:
            if e.entity_id == entity_id:
                return e
        return None

    def entities_in_region(self, region_id: str) -> list[SceneEntity]:
        """Return all entities located in a specific region."""
        return [e for e in self.entities if region_id in e.containers]

    def entities_by_type(self, entity_type: str) -> list[SceneEntity]:
        """Return all entities of a given type."""
        return [e for e in self.entities if e.entity_type == entity_type]


# ═══════════════════════════════════════════════════════════════════════════
# Scene Graph
# ═══════════════════════════════════════════════════════════════════════════


class SceneGraph:
    """Unified read-only projection of the A13 world model.

    Combines information from all A13 subsystems into a coherent,
    queryable scene representation.  This is a projection — it reads
    from HCIR and never writes back.

    **Responsibility:** "What does the world currently look like?"

    Usage::

        scene_graph = SceneGraph(
            graph=graph,
            entity_registry=registry,
            spatial_ontology=ontology,
            event_chronicle=chronicle,
            object_permanence=permanence,
        )

        # Get a full scene snapshot
        snapshot = scene_graph.snapshot()

        # Query specific aspects
        entity = scene_graph.entity(entity_id)
        entities = scene_graph.entities_in_region(region_id)

        # Get a FrozenGraphView scoped to world-model nodes
        view = scene_graph.as_frozen_view()
    """

    def __init__(
        self,
        graph: CognitiveGraph,
        entity_registry: EntityRegistry,
        spatial_ontology: SpatialOntology,
        event_chronicle: EventChronicle,
        object_permanence: ObjectPermanence,
        self_entity_id: str | None = None,
    ) -> None:
        self._graph = graph
        self._registry = entity_registry
        self._spatial = spatial_ontology
        self._chronicle = event_chronicle
        self._permanence = object_permanence
        self._self_entity_id = self_entity_id

    # ── Scene Snapshot ────────────────────────────────────────────────

    def snapshot(
        self,
        include_forgotten: bool = False,
        recent_events_limit: int = 5,
        current_time: float | None = None,
    ) -> SceneSnapshot:
        """Build a point-in-time scene snapshot from current HCIR state.

        Collects all entities with their spatial relations, recent events,
        and permanence predictions into a unified SceneSnapshot.

        Args:
            include_forgotten: Whether to include FORGOTTEN entities.
            recent_events_limit: Max recent events per entity.
            current_time: Override for current time (for testing).

        Returns:
            A SceneSnapshot with all entity context.
        """
        now = current_time if current_time is not None else time.time()
        entities: list[SceneEntity] = []

        for node in self._graph.nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY):
            if not isinstance(node, PhysicalEntityNode):
                continue

            # Skip forgotten entities unless requested
            if (
                not include_forgotten
                and node.entity_lifecycle == EntityLifecycle.FORGOTTEN
            ):
                continue

            scene_entity = self._project_entity(
                node, recent_events_limit, now,
            )
            entities.append(scene_entity)

        return SceneSnapshot(timestamp=now, entities=entities)

    # ── Single Entity Query ───────────────────────────────────────────

    def entity(
        self,
        entity_id: str,
        recent_events_limit: int = 10,
        current_time: float | None = None,
    ) -> SceneEntity | None:
        """Get a fully projected SceneEntity for a specific entity ID."""
        now = current_time if current_time is not None else time.time()
        node = self._registry.get_entity(entity_id)
        if node is None:
            return None

        return self._project_entity(node, recent_events_limit, now)

    # ── Spatial Queries ───────────────────────────────────────────────

    def entities_in_region(
        self,
        region_id: str,
        current_time: float | None = None,
    ) -> list[SceneEntity]:
        """Return all entities located in a specific region."""
        now = current_time if current_time is not None else time.time()
        content_ids = self._spatial.contents_of(region_id)

        entities: list[SceneEntity] = []
        for eid in content_ids:
            entity = self.entity(eid, current_time=now)
            if entity is not None:
                entities.append(entity)
        return entities

    def entities_near(
        self,
        entity_id: str,
        current_time: float | None = None,
    ) -> list[SceneEntity]:
        """Return all entities near a specific entity."""
        now = current_time if current_time is not None else time.time()
        near_ids = self._spatial.entities_near(entity_id)

        entities: list[SceneEntity] = []
        for eid in near_ids:
            entity = self.entity(eid, current_time=now)
            if entity is not None:
                entities.append(entity)
        return entities

    # ── Temporal Queries ──────────────────────────────────────────────

    def recent_changes(
        self,
        seconds: float = 5.0,
        current_time: float | None = None,
    ) -> list[EventNode]:
        """Return events that occurred in the last N seconds."""
        now = current_time if current_time is not None else time.time()
        since = now - seconds
        return self._chronicle.all_events(since=since, until=now)

    # ── FrozenGraphView Bridge ────────────────────────────────────────

    def as_frozen_view(self) -> FrozenGraphView:
        """Create a FrozenGraphView scoped to world-model nodes.

        Returns a frozen view containing only PhysicalEntityNode,
        EventNode, PredictionNode, PredictionErrorNode, and
        ObservationNode entries plus their connecting edges.

        This provides A12 operators with a targeted reasoning context.
        """
        from hbllm.brain.reasoning.operators.base import FrozenGraphView

        # Collect world-model node IDs
        world_node_types = {
            HCIRNodeType.PHYSICAL_ENTITY,
            HCIRNodeType.EVENT,
            HCIRNodeType.OBSERVATION,
            HCIRNodeType.PREDICTION,
            HCIRNodeType.PREDICTION_ERROR,
            HCIRNodeType.ENVIRONMENT_STATE,
            HCIRNodeType.WORLD_VARIABLE,
        }

        world_node_ids: set[str] = set()
        for node_type in world_node_types:
            for node in self._graph.nodes_by_type(node_type):
                world_node_ids.add(node.id)

        return FrozenGraphView.from_graph(
            self._graph,
            node_ids=world_node_ids,
        )

    # ── Self/World Boundary ───────────────────────────────────────────

    @property
    def self_entity_id(self) -> str | None:
        """The entity ID representing the agent itself, if set."""
        return self._self_entity_id

    @self_entity_id.setter
    def self_entity_id(self, entity_id: str | None) -> None:
        self._self_entity_id = entity_id

    # ── Summary ───────────────────────────────────────────────────────

    def summary(self, current_time: float | None = None) -> dict[str, Any]:
        """Generate a concise summary of the current scene."""
        snap = self.snapshot(current_time=current_time)
        return {
            "timestamp": snap.timestamp,
            "total_entities": snap.entity_count,
            "tracked": len(snap.tracked_entities),
            "occluded": len(snap.occluded_entities),
            "discovered": len(snap.discovered_entities),
            "total_chronicle_events": self._chronicle.total_events,
        }

    # ── Internals ─────────────────────────────────────────────────────

    def _project_entity(
        self,
        node: PhysicalEntityNode,
        recent_events_limit: int,
        current_time: float,
    ) -> SceneEntity:
        """Project a PhysicalEntityNode into a full SceneEntity."""
        # Spatial relations
        relations = self._spatial.relations_of(node.id)
        containers = self._spatial.containers_of(node.id)
        nearby = self._spatial.entities_near(node.id)

        # Recent events
        recent_events = self._chronicle.events_for_entity(node.id)
        if len(recent_events) > recent_events_limit:
            recent_events = recent_events[-recent_events_limit:]

        # Observation count
        obs_ids = self._registry.observations_for_entity(node.id)

        # Permanence predictions (for occluded entities)
        perm_confs: dict[str, float] = {}
        if node.entity_lifecycle == EntityLifecycle.OCCLUDED:
            for dim in PersistenceDimension:
                conf = self._permanence.current_confidence(
                    node.id, dim, current_time,
                )
                if conf > 0.0:
                    perm_confs[str(dim)] = round(conf, 4)

        return SceneEntity(
            entity_id=node.id,
            entity_name=node.entity_name,
            entity_type=node.entity_type,
            lifecycle=node.entity_lifecycle,
            properties=dict(node.properties),
            last_observed_at=node.last_observed_at,
            first_observed_at=node.first_observed_at,
            observation_count=len(obs_ids),
            spatial_relations=relations,
            containers=containers,
            nearby_entities=nearby,
            recent_events=recent_events,
            permanence_confidences=perm_confs,
            is_self=(node.id == self._self_entity_id),
        )
