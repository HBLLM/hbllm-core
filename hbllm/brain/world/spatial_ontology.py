"""Spatial Ontology — qualitative spatial reasoning substrate for A13.

Provides a structured representation of spatial knowledge categorized by
relation type.  Feeds A12's SpatialOperator with richer relational substrate.

Relation categories::

    SpatialOntology
        ├── topological
        │     ├── TOUCHING
        │     └── OVERLAPPING
        │
        ├── directional
        │     ├── ABOVE / BELOW
        │     └── LEFT_OF / RIGHT_OF
        │
        ├── metric (approximate)
        │     ├── NEAR
        │     └── FAR (no edge; inferred as absence of NEAR)
        │
        └── containment
              └── LOCATED_IN

Semantic distinction (critical)::

    PART_OF      → physical/semantic composition (Wheel PART_OF Car)
    LOCATED_IN   → spatial containment (Car LOCATED_IN Garage)
    NEAR         → spatial proximity (Car NEAR Door)

These are NEVER interchangeable.

**HCIR invariant:** Spatial relations are represented as typed HCIREdge
entries.  Spatial regions are PhysicalEntityNode instances with
entity_type="spatial_region".  The ontology never owns state — it
provides query/inference services over HCIR edges.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hbllm.brain.world.event_chronicle import (
    ChronicleEvent,
    EventChronicle,
    WorldEventKind,
)
from hbllm.hcir.graph import (
    CognitiveGraph,
    HCIREdge,
    HCIREdgeType,
    PhysicalEntityNode,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Spatial Relation Category
# ═══════════════════════════════════════════════════════════════════════════


class SpatialCategory(StrEnum):
    """Categorization of spatial relation types."""

    TOPOLOGICAL = "topological"  # TOUCHING, OVERLAPPING
    DIRECTIONAL = "directional"  # ABOVE, BELOW, LEFT_OF, RIGHT_OF
    METRIC = "metric"  # NEAR
    CONTAINMENT = "containment"  # LOCATED_IN


# ═══════════════════════════════════════════════════════════════════════════
# Spatial Relation Descriptor
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SpatialRelation:
    """A spatial relation between two entities in the world model."""

    subject_id: str  # Entity A
    relation: HCIREdgeType  # e.g., ABOVE, NEAR, LOCATED_IN
    object_id: str  # Entity B
    category: SpatialCategory
    edge_id: str | None = None  # HCIR edge ID
    properties: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Inverse and Transitivity Tables
# ═══════════════════════════════════════════════════════════════════════════

# Inverse relations: if A rel B, then B inverse(rel) A
INVERSE_RELATIONS: dict[HCIREdgeType, HCIREdgeType] = {
    HCIREdgeType.ABOVE: HCIREdgeType.BELOW,
    HCIREdgeType.BELOW: HCIREdgeType.ABOVE,
    HCIREdgeType.LEFT_OF: HCIREdgeType.RIGHT_OF,
    HCIREdgeType.RIGHT_OF: HCIREdgeType.LEFT_OF,
    # Symmetric relations map to themselves
    HCIREdgeType.NEAR: HCIREdgeType.NEAR,
    HCIREdgeType.TOUCHING: HCIREdgeType.TOUCHING,
    HCIREdgeType.OVERLAPPING: HCIREdgeType.OVERLAPPING,
}

# Transitive relations: if A rel B and B rel C, then A rel C
TRANSITIVE_RELATIONS: set[HCIREdgeType] = {
    HCIREdgeType.ABOVE,
    HCIREdgeType.BELOW,
    HCIREdgeType.LEFT_OF,
    HCIREdgeType.RIGHT_OF,
    HCIREdgeType.LOCATED_IN,
}

# Map edge types to their spatial categories
EDGE_TO_CATEGORY: dict[HCIREdgeType, SpatialCategory] = {
    HCIREdgeType.TOUCHING: SpatialCategory.TOPOLOGICAL,
    HCIREdgeType.OVERLAPPING: SpatialCategory.TOPOLOGICAL,
    HCIREdgeType.ABOVE: SpatialCategory.DIRECTIONAL,
    HCIREdgeType.BELOW: SpatialCategory.DIRECTIONAL,
    HCIREdgeType.LEFT_OF: SpatialCategory.DIRECTIONAL,
    HCIREdgeType.RIGHT_OF: SpatialCategory.DIRECTIONAL,
    HCIREdgeType.NEAR: SpatialCategory.METRIC,
    HCIREdgeType.LOCATED_IN: SpatialCategory.CONTAINMENT,
}

# Contradictory pairs: cannot hold simultaneously for the same pair
CONTRADICTORY_PAIRS: list[tuple[HCIREdgeType, HCIREdgeType]] = [
    (HCIREdgeType.ABOVE, HCIREdgeType.BELOW),
    (HCIREdgeType.LEFT_OF, HCIREdgeType.RIGHT_OF),
]


# ═══════════════════════════════════════════════════════════════════════════
# Spatial Ontology
# ═══════════════════════════════════════════════════════════════════════════


class SpatialOntology:
    """Qualitative spatial reasoning substrate for the A13 world model.

    Manages spatial relations between entities as HCIR edges, provides
    transitivity inference, inverse computation, and consistency checking.

    **Responsibility:** "How are entities and regions spatially related?"

    **Does NOT:**
    - Own geometric coordinates (future extension)
    - Decide truth (produces evidence for epistemics)
    - Mutate entity properties (only manages spatial edges)

    Usage::

        ontology = SpatialOntology(graph, chronicle)

        # Assert a relation
        ontology.assert_relation("ball", HCIREdgeType.LOCATED_IN, "box")
        ontology.assert_relation("box", HCIREdgeType.LOCATED_IN, "room")

        # Query
        relations = ontology.relations_of("ball")
        containers = ontology.containers_of("ball")

        # Infer transitive closure
        inferred = ontology.infer_transitive("ball", HCIREdgeType.LOCATED_IN)
        # → ["box", "room"]

        # Check consistency
        issues = ontology.check_consistency()
    """

    def __init__(
        self,
        graph: CognitiveGraph,
        chronicle: EventChronicle,
    ) -> None:
        self._graph = graph
        self._chronicle = chronicle

    # ── Assertion ─────────────────────────────────────────────────────

    def assert_relation(
        self,
        subject_id: str,
        relation: HCIREdgeType,
        object_id: str,
        properties: dict[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> str:
        """Assert a spatial relation between two entities.

        Creates an HCIREdge in the graph.  Records the relation
        establishment in the event chronicle.

        Args:
            subject_id: The entity the relation is FROM (e.g., "ball").
            relation: The spatial relation type (e.g., LOCATED_IN).
            object_id: The entity the relation is TO (e.g., "box").
            properties: Optional edge properties.
            timestamp: Optional event timestamp.

        Returns:
            The ID of the created HCIREdge.

        Raises:
            ValueError: If the relation is not a recognized spatial edge type.
        """
        if relation not in EDGE_TO_CATEGORY:
            msg = f"Not a spatial relation: {relation}"
            raise ValueError(msg)

        props = properties or {}

        edge = HCIREdge(
            edge_type=relation,
            sources=[subject_id],
            targets=[object_id],
            properties=props,
        )
        self._graph.add_edge(edge)

        # Record in chronicle
        import time as _time

        now = timestamp if timestamp is not None else _time.time()
        self._chronicle.record(
            ChronicleEvent(
                event_kind=WorldEventKind.RELATION_ESTABLISHED,
                subject_entity_id=subject_id,
                timestamp=now,
                state_after={
                    "relation": str(relation),
                    "object_id": object_id,
                },
                metadata={"edge_id": edge.id},
            )
        )

        logger.debug(
            "SpatialOntology: %s %s %s (edge %s)",
            subject_id,
            relation,
            object_id,
            edge.id,
        )

        return edge.id

    def retract_relation(self, edge_id: str, timestamp: float | None = None) -> None:
        """Retract a spatial relation by removing its HCIR edge."""
        edge = self._graph.get_edge(edge_id)
        if edge is None:
            return

        import time as _time

        now = timestamp if timestamp is not None else _time.time()

        # Record retraction in chronicle
        if edge.sources:
            self._chronicle.record(
                ChronicleEvent(
                    event_kind=WorldEventKind.RELATION_REMOVED,
                    subject_entity_id=edge.sources[0],
                    timestamp=now,
                    state_before={
                        "relation": str(edge.edge_type),
                        "targets": edge.targets,
                    },
                    metadata={"edge_id": edge_id},
                )
            )

        self._graph.remove_edge(edge_id)

    # ── Queries ───────────────────────────────────────────────────────

    def relations_of(
        self,
        entity_id: str,
        category: SpatialCategory | None = None,
    ) -> list[SpatialRelation]:
        """Get all spatial relations involving an entity (as subject or object).

        Args:
            entity_id: The entity to query.
            category: Optional filter by spatial category.

        Returns:
            List of SpatialRelation descriptors.
        """
        results: list[SpatialRelation] = []

        # Relations where entity is subject (outgoing edges)
        for edge in self._graph.edges_from(entity_id):
            if edge.edge_type in EDGE_TO_CATEGORY:
                cat = EDGE_TO_CATEGORY[edge.edge_type]
                if category is not None and cat != category:
                    continue
                for target in edge.targets:
                    results.append(
                        SpatialRelation(
                            subject_id=entity_id,
                            relation=edge.edge_type,
                            object_id=target,
                            category=cat,
                            edge_id=edge.id,
                        )
                    )

        # Relations where entity is object (incoming edges)
        for edge in self._graph.edges_to(entity_id):
            if edge.edge_type in EDGE_TO_CATEGORY:
                cat = EDGE_TO_CATEGORY[edge.edge_type]
                if category is not None and cat != category:
                    continue
                for source in edge.sources:
                    results.append(
                        SpatialRelation(
                            subject_id=source,
                            relation=edge.edge_type,
                            object_id=entity_id,
                            category=cat,
                            edge_id=edge.id,
                        )
                    )

        return results

    def containers_of(self, entity_id: str) -> list[str]:
        """Return all entities/regions that contain this entity (LOCATED_IN)."""
        containers: list[str] = []
        for edge in self._graph.edges_from(entity_id):
            if edge.edge_type == HCIREdgeType.LOCATED_IN:
                containers.extend(edge.targets)
        return containers

    def contents_of(self, region_id: str) -> list[str]:
        """Return all entities located inside a region."""
        contents: list[str] = []
        for edge in self._graph.edges_to(region_id):
            if edge.edge_type == HCIREdgeType.LOCATED_IN:
                contents.extend(edge.sources)
        return contents

    def entities_near(self, entity_id: str) -> list[str]:
        """Return all entities that are NEAR this entity."""
        near: list[str] = []
        for edge in self._graph.edges_from(entity_id):
            if edge.edge_type == HCIREdgeType.NEAR:
                near.extend(edge.targets)
        for edge in self._graph.edges_to(entity_id):
            if edge.edge_type == HCIREdgeType.NEAR:
                near.extend(edge.sources)
        return near

    # ── Transitivity Inference ────────────────────────────────────────

    def infer_transitive(
        self,
        entity_id: str,
        relation: HCIREdgeType,
        max_depth: int = 10,
    ) -> list[str]:
        """Compute the transitive closure for a relation from an entity.

        E.g., if A LOCATED_IN B and B LOCATED_IN C,
        infer_transitive(A, LOCATED_IN) → [B, C].

        Only works for relations in TRANSITIVE_RELATIONS.

        Args:
            entity_id: Starting entity.
            relation: The relation to follow transitively.
            max_depth: Maximum depth to prevent infinite loops.

        Returns:
            List of entity IDs reachable via transitive closure.
        """
        if relation not in TRANSITIVE_RELATIONS:
            return []

        result: list[str] = []
        visited: set[str] = {entity_id}
        frontier = [entity_id]

        for _ in range(max_depth):
            next_frontier: list[str] = []
            for current_id in frontier:
                for edge in self._graph.edges_from(current_id):
                    if edge.edge_type == relation:
                        for target in edge.targets:
                            if target not in visited:
                                visited.add(target)
                                result.append(target)
                                next_frontier.append(target)
            if not next_frontier:
                break
            frontier = next_frontier

        return result

    def compute_inverse(
        self,
        subject_id: str,
        relation: HCIREdgeType,
        object_id: str,
    ) -> SpatialRelation | None:
        """Compute the inverse relation.

        If A ABOVE B, the inverse is B BELOW A.
        """
        inverse = INVERSE_RELATIONS.get(relation)
        if inverse is None:
            return None

        return SpatialRelation(
            subject_id=object_id,
            relation=inverse,
            object_id=subject_id,
            category=EDGE_TO_CATEGORY.get(inverse, SpatialCategory.TOPOLOGICAL),
        )

    # ── Consistency Checking ──────────────────────────────────────────

    def check_consistency(self) -> list[str]:
        """Check for spatial contradictions in the graph.

        Detects:
        - Mutual containment: A LOCATED_IN B AND B LOCATED_IN A
        - Contradictory directional: A ABOVE B AND A BELOW B
        - Self-containment: A LOCATED_IN A

        Returns:
            List of human-readable issue descriptions.
        """
        issues: list[str] = []

        # Collect all spatial edges
        spatial_edges: list[tuple[str, HCIREdgeType, str]] = []
        for edge in self._graph.all_edges():
            if edge.edge_type in EDGE_TO_CATEGORY:
                for source in edge.sources:
                    for target in edge.targets:
                        spatial_edges.append((source, edge.edge_type, target))

        # Check self-containment
        for subj, rel, obj in spatial_edges:
            if subj == obj and rel == HCIREdgeType.LOCATED_IN:
                issues.append(f"Self-containment: {subj} LOCATED_IN itself")

        # Check mutual containment
        containment_pairs: set[tuple[str, str]] = set()
        for subj, rel, obj in spatial_edges:
            if rel == HCIREdgeType.LOCATED_IN:
                if (obj, subj) in containment_pairs:
                    issues.append(
                        f"Mutual containment: {subj} LOCATED_IN {obj} AND {obj} LOCATED_IN {subj}"
                    )
                containment_pairs.add((subj, obj))

        # Check contradictory directional relations
        relation_set: set[tuple[str, str, HCIREdgeType]] = set()
        for subj, rel, obj in spatial_edges:
            relation_set.add((subj, obj, rel))

        for rel_a, rel_b in CONTRADICTORY_PAIRS:
            for subj, obj, rel in relation_set:
                if rel == rel_a and (subj, obj, rel_b) in relation_set:
                    issues.append(f"Contradiction: {subj} {rel_a} {obj} AND {subj} {rel_b} {obj}")

        return issues

    # ── Region Management ─────────────────────────────────────────────

    def create_region(
        self,
        name: str,
        parent_region_id: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Create a spatial region as a PhysicalEntityNode.

        Spatial regions are entities with entity_type="spatial_region".
        They can be nested via LOCATED_IN edges.

        Args:
            name: Human-readable name for the region.
            parent_region_id: Optional parent region for nesting.
            properties: Optional region properties.

        Returns:
            The ID of the created region node.
        """
        from hbllm.hcir.graph import EntityLifecycle

        region = PhysicalEntityNode(
            entity_name=name,
            entity_type="spatial_region",
            properties=properties or {},
            entity_lifecycle=EntityLifecycle.TRACKED,
        )
        self._graph.add_node(region)

        if parent_region_id is not None:
            self.assert_relation(
                region.id,
                HCIREdgeType.LOCATED_IN,
                parent_region_id,
            )

        return region.id
