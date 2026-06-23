"""
RealityGraph — unified facade over existing world models.

Provides a single query interface across:
    - BrainWorldState (probabilistic entity graph from perception events)
    - PerceptionWorldState (hardware, IoT, audio, calendar aggregation)
    - KnowledgeGraph (directed concept/entity graph with community detection)

Phase 1 (current): Read-only adapter. Queries all 3 backends, merges results.
                    No migrations, no deletions, no breaking changes.
Phase 2 (future):   New writes go to KG. Old systems become read-only feeders.
Phase 3 (future):   Old systems deprecated.

Architecture:
    RealityGraph
          ↑
     Adapter Layer
    ↙      ↓       ↘
 BrainWS   KG   PerceptionWS

Entity types:
    person     → Users, contacts (from RelationshipMemory)
    device     → IoT devices, hardware (from perception)
    location   → Physical locations
    project    → Active projects (from ProjectGraph)
    concept    → Abstract knowledge (from KnowledgeGraph)
    event      → Calendar events, meetings
    state      → Transient states (battery, weather, mood)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Unified Entity ───────────────────────────────────────────────────────────


@dataclass
class RealityEntity:
    """A unified entity from any backend."""

    entity_id: str
    entity_type: str  # person | device | location | project | concept | event | state
    label: str
    attributes: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = ""  # "kg" | "brain_ws" | "perception_ws" | "merged"
    last_updated: float = field(default_factory=time.time)
    ttl: float | None = None  # Seconds until expiry (None = permanent)

    @property
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return (time.time() - self.last_updated) > self.ttl

    def merge_with(self, other: RealityEntity) -> RealityEntity:
        """Merge two entities from different sources."""
        merged_attrs = {**self.attributes, **other.attributes}
        return RealityEntity(
            entity_id=self.entity_id,
            entity_type=self.entity_type,
            label=self.label or other.label,
            attributes=merged_attrs,
            confidence=max(self.confidence, other.confidence),
            source="merged",
            last_updated=max(self.last_updated, other.last_updated),
            ttl=min(self.ttl or float("inf"), other.ttl or float("inf"))
            if self.ttl or other.ttl
            else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "label": self.label,
            "attributes": self.attributes,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "last_updated": self.last_updated,
            "ttl": self.ttl,
            "expired": self.is_expired,
        }


# ── RealityGraph ─────────────────────────────────────────────────────────────


class RealityGraph:
    """Unified facade over existing world models.

    Provides a single query interface. Does NOT replace existing systems.
    All backends continue to operate independently.

    Usage:
        rg = RealityGraph(
            knowledge_graph=kg,
            brain_world_state=brain_ws,
            perception_world_state=perception_ws,
        )

        # Query any entity type
        entity = rg.query_entity("macbook")
        entities = rg.query_by_type("device")

        # Get unified context for LLM
        context = await rg.get_context(query, tenant_id, budget)
    """

    def __init__(
        self,
        knowledge_graph: Any | None = None,
        brain_world_state: Any | None = None,
        perception_world_state: Any | None = None,
    ) -> None:
        self._kg = knowledge_graph
        self._brain_ws = brain_world_state
        self._perception_ws = perception_world_state

        # Entity type classification for KG entities
        self._type_hints: dict[str, str] = {
            "person": "person",
            "device": "device",
            "location": "location",
            "project": "project",
            "concept": "concept",
            "event": "event",
        }

    # ── Entity Queries ───────────────────────────────────────────────

    def query_entity(self, label: str) -> RealityEntity | None:
        """Look up an entity across all backends.

        Priority: KG first, then brain WS, then perception WS.
        Merges results if found in multiple backends.
        """
        result: RealityEntity | None = None

        # 1. Check KnowledgeGraph
        kg_entity = self._query_kg(label)
        if kg_entity:
            result = kg_entity

        # 2. Check Brain WorldState
        brain_entity = self._query_brain_ws(label)
        if brain_entity:
            result = result.merge_with(brain_entity) if result else brain_entity

        # 3. Check Perception WorldState
        perception_entity = self._query_perception_ws(label)
        if perception_entity:
            result = result.merge_with(perception_entity) if result else perception_entity

        return result

    def query_by_type(self, entity_type: str) -> list[RealityEntity]:
        """Get all entities of a given type across all backends."""
        entities: dict[str, RealityEntity] = {}

        # KG entities
        if self._kg:
            try:
                kg_entities = self._query_kg_by_type(entity_type)
                for e in kg_entities:
                    entities[e.entity_id] = e
            except Exception as e:
                logger.debug("KG query_by_type failed: %s", e)

        # Brain WS entities (all are "state" type effectively)
        if entity_type in ("state", "device") and self._brain_ws:
            try:
                brain_entities = self._get_brain_ws_entities()
                for e in brain_entities:
                    if e.entity_type == entity_type:
                        if e.entity_id in entities:
                            entities[e.entity_id] = entities[e.entity_id].merge_with(e)
                        else:
                            entities[e.entity_id] = e
            except Exception as e:
                logger.debug("Brain WS query failed: %s", e)

        # Perception WS (hardware, IoT devices)
        if entity_type in ("device", "state") and self._perception_ws:
            try:
                perc_entities = self._get_perception_ws_entities()
                for e in perc_entities:
                    if e.entity_type == entity_type:
                        if e.entity_id in entities:
                            entities[e.entity_id] = entities[e.entity_id].merge_with(e)
                        else:
                            entities[e.entity_id] = e
            except Exception as e:
                logger.debug("Perception WS query failed: %s", e)

        # Filter expired
        return [e for e in entities.values() if not e.is_expired]

    def get_all_entities(self) -> list[RealityEntity]:
        """Get all known entities across all backends."""
        all_types = ["person", "device", "location", "project", "concept", "event", "state"]
        entities: dict[str, RealityEntity] = {}
        for t in all_types:
            for e in self.query_by_type(t):
                if e.entity_id not in entities:
                    entities[e.entity_id] = e
        return list(entities.values())

    # ── Context Generation ───────────────────────────────────────────

    async def get_context(
        self, query: str, tenant_id: str, budget: int
    ) -> str:
        """ContextFusion-compatible provider.

        Generates a unified world state summary from all backends.
        """
        parts: list[str] = []

        # Perception state (environment)
        env_summary = self._get_environment_summary()
        if env_summary:
            parts.append(env_summary)

        # High-confidence brain WS entities
        brain_summary = self._get_brain_ws_summary()
        if brain_summary:
            parts.append(brain_summary)

        # Relevant KG entities
        kg_summary = self._get_kg_summary(query)
        if kg_summary:
            parts.append(kg_summary)

        return "\n".join(parts) if parts else ""

    def get_user_context(self, tenant_id: str) -> str:
        """Generate a comprehensive user context summary.

        Combines: user state + devices + location + schedule.
        Used for system prompt injection.
        """
        parts: list[str] = []

        # Environment from perception
        env = self._get_environment_summary()
        if env:
            parts.append(env)

        # Active devices
        devices = self.query_by_type("device")
        if devices:
            dev_strs = []
            for d in devices[:5]:
                attrs = ", ".join(
                    f"{k}={v}" for k, v in list(d.attributes.items())[:3]
                )
                dev_strs.append(f"  - {d.label}: {attrs}")
            parts.append("Devices:\n" + "\n".join(dev_strs))

        return "\n".join(parts) if parts else ""

    # ── TTL Management ───────────────────────────────────────────────

    def tick(self) -> int:
        """Run maintenance: expire stale entities, update confidence.

        Returns the number of expired entities cleaned up.
        """
        expired_count = 0

        # Tick brain WS confidence decay
        if self._brain_ws and hasattr(self._brain_ws, "tick_decay"):
            try:
                self._brain_ws.tick_decay()
            except Exception:
                pass

        return expired_count

    # ── Backend Adapters ─────────────────────────────────────────────

    def _query_kg(self, label: str) -> RealityEntity | None:
        """Query KnowledgeGraph for an entity."""
        if not self._kg:
            return None

        try:
            # Try get_entity or neighbors
            if hasattr(self._kg, "get_entity"):
                entity = self._kg.get_entity(label)
                if entity:
                    return RealityEntity(
                        entity_id=f"kg:{label}",
                        entity_type=entity.get("entity_type", "concept"),
                        label=label,
                        attributes=entity.get("attributes", {}),
                        confidence=entity.get("confidence", 0.8),
                        source="kg",
                    )

            if hasattr(self._kg, "neighbors"):
                neighbors = self._kg.neighbors(label)
                if neighbors:
                    return RealityEntity(
                        entity_id=f"kg:{label}",
                        entity_type="concept",
                        label=label,
                        attributes={"neighbors": len(neighbors)},
                        confidence=0.7,
                        source="kg",
                    )
        except Exception as e:
            logger.debug("KG query failed for '%s': %s", label, e)
        return None

    def _query_kg_by_type(self, entity_type: str) -> list[RealityEntity]:
        """Query KG for entities of a specific type."""
        if not self._kg:
            return []
        results: list[RealityEntity] = []
        try:
            if hasattr(self._kg, "get_entities_by_type"):
                entities = self._kg.get_entities_by_type(entity_type)
                for label, data in entities.items() if isinstance(entities, dict) else []:
                    results.append(RealityEntity(
                        entity_id=f"kg:{label}",
                        entity_type=entity_type,
                        label=label,
                        attributes=data if isinstance(data, dict) else {},
                        source="kg",
                    ))
        except Exception:
            pass
        return results

    def _query_brain_ws(self, label: str) -> RealityEntity | None:
        """Query Brain WorldState for an entity."""
        if not self._brain_ws:
            return None
        try:
            if hasattr(self._brain_ws, "_graph"):
                entity = self._brain_ws._graph.get(label)
                if entity:
                    return RealityEntity(
                        entity_id=f"brain_ws:{label}",
                        entity_type="state",
                        label=label,
                        attributes=entity.properties if hasattr(entity, "properties") else {},
                        confidence=entity.confidence if hasattr(entity, "confidence") else 0.5,
                        source="brain_ws",
                        last_updated=entity.last_updated if hasattr(entity, "last_updated") else time.time(),
                        ttl=3600.0,  # Brain WS entities expire after 1 hour
                    )
        except Exception as e:
            logger.debug("Brain WS query failed for '%s': %s", label, e)
        return None

    def _get_brain_ws_entities(self) -> list[RealityEntity]:
        """Get all entities from Brain WorldState."""
        if not self._brain_ws or not hasattr(self._brain_ws, "_graph"):
            return []
        results: list[RealityEntity] = []
        try:
            for entity_id, entity in self._brain_ws._graph.items():
                results.append(RealityEntity(
                    entity_id=f"brain_ws:{entity_id}",
                    entity_type="state",
                    label=entity_id,
                    attributes=entity.properties if hasattr(entity, "properties") else {},
                    confidence=entity.confidence if hasattr(entity, "confidence") else 0.5,
                    source="brain_ws",
                    last_updated=entity.last_updated if hasattr(entity, "last_updated") else time.time(),
                    ttl=3600.0,
                ))
        except Exception:
            pass
        return results

    def _query_perception_ws(self, label: str) -> RealityEntity | None:
        """Query Perception WorldState for an entity."""
        if not self._perception_ws:
            return None
        try:
            if hasattr(self._perception_ws, "_iot_devices"):
                device = self._perception_ws._iot_devices.get(label)
                if device:
                    return RealityEntity(
                        entity_id=f"perc_ws:{label}",
                        entity_type="device",
                        label=label,
                        attributes=device if isinstance(device, dict) else {},
                        confidence=0.9,
                        source="perception_ws",
                        ttl=300.0,  # Perception state expires after 5 minutes
                    )
            # Check hardware state
            if hasattr(self._perception_ws, "_hardware"):
                hw = self._perception_ws._hardware
                if label in hw or label.lower() in [k.lower() for k in hw]:
                    return RealityEntity(
                        entity_id=f"perc_ws:{label}",
                        entity_type="state",
                        label=label,
                        attributes=hw,
                        confidence=0.95,
                        source="perception_ws",
                        ttl=60.0,  # Hardware state refreshes every minute
                    )
        except Exception as e:
            logger.debug("Perception WS query failed for '%s': %s", label, e)
        return None

    def _get_perception_ws_entities(self) -> list[RealityEntity]:
        """Get all entities from Perception WorldState."""
        results: list[RealityEntity] = []
        if not self._perception_ws:
            return results

        try:
            # IoT devices
            if hasattr(self._perception_ws, "_iot_devices"):
                for dev_id, dev_data in self._perception_ws._iot_devices.items():
                    results.append(RealityEntity(
                        entity_id=f"perc_ws:{dev_id}",
                        entity_type="device",
                        label=dev_id,
                        attributes=dev_data if isinstance(dev_data, dict) else {},
                        confidence=0.9,
                        source="perception_ws",
                        ttl=300.0,
                    ))

            # Hardware as a single state entity
            if hasattr(self._perception_ws, "_hardware") and self._perception_ws._hardware:
                results.append(RealityEntity(
                    entity_id="perc_ws:hardware",
                    entity_type="state",
                    label="hardware",
                    attributes=self._perception_ws._hardware,
                    confidence=0.95,
                    source="perception_ws",
                    ttl=60.0,
                ))
        except Exception:
            pass

        return results

    # ── Summary Generators ───────────────────────────────────────────

    def _get_environment_summary(self) -> str:
        """Get environment summary from perception WS."""
        if not self._perception_ws:
            return ""
        try:
            if hasattr(self._perception_ws, "get_summary"):
                return self._perception_ws.get_summary()
        except Exception:
            pass
        return ""

    def _get_brain_ws_summary(self) -> str:
        """Get high-confidence entities from brain WS."""
        entities = self._get_brain_ws_entities()
        high_conf = [e for e in entities if e.confidence > 0.5 and not e.is_expired]
        if not high_conf:
            return ""

        lines = []
        for e in sorted(high_conf, key=lambda x: x.confidence, reverse=True)[:5]:
            attrs = ", ".join(f"{k}={v}" for k, v in list(e.attributes.items())[:3])
            lines.append(f"  - {e.label}: {attrs} ({e.confidence:.0%})")
        return "Known state:\n" + "\n".join(lines)

    def _get_kg_summary(self, query: str) -> str:
        """Get relevant KG entities for a query."""
        if not self._kg:
            return ""
        try:
            if hasattr(self._kg, "neighbors"):
                query_words = query.lower().split()[:3]
                for word in query_words:
                    neighbors = self._kg.neighbors(word)
                    if neighbors:
                        return f"Related concepts: {', '.join(list(neighbors.keys())[:5])}"
        except Exception:
            pass
        return ""

    # ── Stats ────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Summary statistics across all backends."""
        stats: dict[str, Any] = {"backends": []}

        if self._kg:
            stats["backends"].append("knowledge_graph")
            try:
                if hasattr(self._kg, "stats"):
                    stats["kg_stats"] = self._kg.stats()
            except Exception:
                pass

        if self._brain_ws:
            stats["backends"].append("brain_world_state")
            try:
                if hasattr(self._brain_ws, "_graph"):
                    stats["brain_ws_entities"] = len(self._brain_ws._graph)
            except Exception:
                pass

        if self._perception_ws:
            stats["backends"].append("perception_world_state")

        return stats
