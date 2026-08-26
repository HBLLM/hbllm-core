"""A13 — Persistent World Model: End-to-End Benchmark.

Proves that the A13 world model maintains a coherent, persistent hypothesis
about a world that continues to exist when sensors are not observing it.

8 scenarios demonstrating the complete A13 capability:

1. Entity lifecycle — discovery → tracking → occlusion → re-identification
2. Multi-dimensional permanence — independent prediction/decay per dimension
3. Spatial reasoning — categorized relations with transitive inference
4. Event chronicle — temporal history with causal chains
5. Identity hypothesis — evidence-based re-identification (not eager merge)
6. World reconciliation — observation vs belief producing structured deltas
7. Long-gap persistence — entity identity survives extended absence
8. Full integration — end-to-end perception → world model → scene graph

**Zero LLM invocation throughout.**
"""

from __future__ import annotations

import pytest

from hbllm.brain.world.entity_registry import EntityRegistry
from hbllm.brain.world.event_chronicle import (
    ChronicleEvent,
    EventChronicle,
    WorldEventKind,
)
from hbllm.brain.world.object_permanence import (
    ObjectPermanence,
    PersistenceDimension,
)
from hbllm.brain.world.reconciler import (
    DeltaType,
    WorldStateReconciler,
)
from hbllm.brain.world.scene_graph import SceneGraph
from hbllm.brain.world.spatial_ontology import (
    SpatialCategory,
    SpatialOntology,
)
from hbllm.hcir.graph import (
    CognitiveGraph,
    EntityLifecycle,
    HCIREdge,
    HCIREdgeType,
    ObservationNode,
)

# ═══════════════════════════════════════════════════════════════════════════
# Fixtures — shared world model setup
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def world():
    """Create a complete A13 world model stack."""
    graph = CognitiveGraph()
    chronicle = EventChronicle(graph)
    registry = EntityRegistry(graph, chronicle)
    ontology = SpatialOntology(graph, chronicle)
    permanence = ObjectPermanence(graph)
    reconciler = WorldStateReconciler(
        graph=graph,
        entity_registry=registry,
        spatial_ontology=ontology,
        event_chronicle=chronicle,
        object_permanence=permanence,
    )
    scene_graph = SceneGraph(
        graph=graph,
        entity_registry=registry,
        spatial_ontology=ontology,
        event_chronicle=chronicle,
        object_permanence=permanence,
    )

    return {
        "graph": graph,
        "chronicle": chronicle,
        "registry": registry,
        "ontology": ontology,
        "permanence": permanence,
        "reconciler": reconciler,
        "scene_graph": scene_graph,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 1: Entity Lifecycle
# ═══════════════════════════════════════════════════════════════════════════


class TestEntityLifecycle:
    """Perception → reconciler → entity discovered → tracked → occluded → re-identified → tracked."""

    def test_full_lifecycle(self, world):
        graph = world["graph"]
        registry = world["registry"]
        chronicle = world["chronicle"]

        t0 = 1000.0

        # Step 1: Discover entity
        entity_id = registry.discover(
            entity_name="red_ball",
            entity_type="toy",
            properties={"color": "red", "shape": "sphere"},
            timestamp=t0,
        )

        entity = registry.get_entity(entity_id)
        assert entity is not None
        assert entity.entity_lifecycle == EntityLifecycle.DISCOVERED
        assert entity.entity_name == "red_ball"
        assert entity.first_observed_at == t0

        # Step 2: Track
        registry.track(entity_id, timestamp=t0 + 1)
        entity = registry.get_entity(entity_id)
        assert entity is not None
        assert entity.entity_lifecycle == EntityLifecycle.TRACKED

        # Step 3: Occlude
        registry.occlude(entity_id, timestamp=t0 + 10)
        entity = registry.get_entity(entity_id)
        assert entity is not None
        assert entity.entity_lifecycle == EntityLifecycle.OCCLUDED

        # Step 4: Create a new observation for re-identification
        obs = ObservationNode(
            payload={"entity_name": "red_ball", "entity_type": "toy", "color": "red"},
            sensor_source="camera_1",
        )
        graph.add_node(obs)

        # Step 5: Propose and confirm re-identification
        candidate = registry.propose_reidentification(
            observation_id=obs.id,
            candidate_entity_id=entity_id,
            similarity_score=0.95,
        )
        assert candidate.edge_id is not None

        registry.confirm_reidentification(
            observation_id=obs.id,
            entity_id=entity_id,
            timestamp=t0 + 100,
        )

        entity = registry.get_entity(entity_id)
        assert entity is not None
        assert entity.entity_lifecycle == EntityLifecycle.TRACKED  # Back to tracked!

        # Step 6: Verify re-identification was recorded as an EVENT, not a state
        events = chronicle.events_for_entity(
            entity_id,
            event_kind=WorldEventKind.ENTITY_RE_IDENTIFIED,
        )
        assert len(events) >= 1

        # Step 7: Verify observation count
        obs_ids = registry.observations_for_entity(entity_id)
        assert len(obs_ids) >= 1  # At least the re-identification observation

        # Step 8: Forget
        registry.forget(entity_id, timestamp=t0 + 200)
        entity = registry.get_entity(entity_id)
        assert entity is not None
        assert entity.entity_lifecycle == EntityLifecycle.FORGOTTEN


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 2: Multi-Dimensional Permanence
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiDimensionalPermanence:
    """Occlude entity → verify 4 prediction types → partial invalidation."""

    def test_multi_dimensional_predictions(self, world):
        registry = world["registry"]
        permanence = world["permanence"]

        t0 = 1000.0

        # Create and track entity
        entity_id = registry.discover(
            entity_name="toy_car",
            entity_type="toy",
            properties={"color": "blue", "size": "small"},
            timestamp=t0,
        )
        registry.track(entity_id, timestamp=t0 + 1)
        registry.occlude(entity_id, timestamp=t0 + 10)

        # Generate permanence predictions
        predictions = permanence.generate_predictions(
            entity_id=entity_id,
            occlusion_time=t0 + 10,
            spatial_context={"container": "box", "region": "living_room"},
            relation_context=[("near", "table_01")],
        )

        # Must have all 4 dimensions
        dimensions = {p.dimension for p in predictions}
        assert PersistenceDimension.EXISTENCE in dimensions
        assert PersistenceDimension.LOCATION in dimensions
        assert PersistenceDimension.PROPERTY in dimensions
        assert PersistenceDimension.RELATION in dimensions

        # All predictions should start with high confidence
        for pred in predictions:
            assert pred.initial_confidence >= 0.8

        # Check decay after some time
        t_check = t0 + 110  # 100 seconds after occlusion

        existence_conf = permanence.current_confidence(
            entity_id,
            PersistenceDimension.EXISTENCE,
            t_check,
        )
        location_conf = permanence.current_confidence(
            entity_id,
            PersistenceDimension.LOCATION,
            t_check,
        )

        # Existence should decay slower than location
        assert existence_conf > location_conf

        # Both should be > 0
        assert existence_conf > 0.0
        assert location_conf > 0.0

    def test_partial_invalidation(self, world):
        """Location wrong, but existence still valid."""
        registry = world["registry"]
        permanence = world["permanence"]

        t0 = 1000.0

        entity_id = registry.discover(
            entity_name="keys",
            entity_type="object",
            properties={"material": "metal"},
            timestamp=t0,
        )
        registry.track(entity_id, timestamp=t0 + 1)
        registry.occlude(entity_id, timestamp=t0 + 10)

        permanence.generate_predictions(
            entity_id=entity_id,
            occlusion_time=t0 + 10,
            spatial_context={"container": "kitchen_counter"},
        )

        # Re-observe with different location
        errors = permanence.check_against_observation(
            entity_id=entity_id,
            observed_properties={"material": "metal"},  # Properties match
            observed_location={"container": "bedroom_table"},  # Location changed!
            observation_time=t0 + 100,
        )

        # Should have at least one error (location)
        assert len(errors) >= 1

        # Existence is confirmed by re-observation — no error for existence
        error_causes = [e.suspected_cause for e in errors]
        assert "location_changed" in error_causes
        assert "existence_changed" not in error_causes


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 3: Spatial Reasoning
# ═══════════════════════════════════════════════════════════════════════════


class TestSpatialReasoning:
    """Categorized relations with transitive inference."""

    def test_transitive_containment(self, world):
        """Ball LOCATED_IN box, box LOCATED_IN room → ball transitively in room."""
        ontology = world["ontology"]
        registry = world["registry"]

        t0 = 1000.0

        # Create entities
        ball_id = registry.discover("ball", "toy", timestamp=t0)
        box_id = registry.discover("box", "container", timestamp=t0)
        room_id = ontology.create_region("living_room")

        # Assert spatial relations
        ontology.assert_relation(ball_id, HCIREdgeType.LOCATED_IN, box_id)
        ontology.assert_relation(box_id, HCIREdgeType.LOCATED_IN, room_id)

        # Transitive inference
        containers = ontology.infer_transitive(ball_id, HCIREdgeType.LOCATED_IN)
        assert box_id in containers
        assert room_id in containers

    def test_categorized_relations(self, world):
        """Relations are properly categorized."""
        ontology = world["ontology"]
        registry = world["registry"]

        t0 = 1000.0
        cup_id = registry.discover("cup", "object", timestamp=t0)
        table_id = registry.discover("table", "furniture", timestamp=t0)

        ontology.assert_relation(cup_id, HCIREdgeType.ABOVE, table_id)
        ontology.assert_relation(cup_id, HCIREdgeType.NEAR, table_id)

        relations = ontology.relations_of(cup_id)
        categories = {r.category for r in relations}
        assert SpatialCategory.DIRECTIONAL in categories
        assert SpatialCategory.METRIC in categories

    def test_located_in_vs_part_of(self, world):
        """LOCATED_IN (spatial) is distinct from PART_OF (composition)."""
        graph = world["graph"]
        registry = world["registry"]
        ontology = world["ontology"]

        t0 = 1000.0
        wheel_id = registry.discover("wheel", "component", timestamp=t0)
        car_id = registry.discover("car", "vehicle", timestamp=t0)
        garage_id = ontology.create_region("garage")

        # Wheel PART_OF car (composition — NOT spatial containment)
        graph.add_edge(
            HCIREdge(
                edge_type=HCIREdgeType.PART_OF,
                sources=[wheel_id],
                targets=[car_id],
            )
        )

        # Car LOCATED_IN garage (spatial containment)
        ontology.assert_relation(car_id, HCIREdgeType.LOCATED_IN, garage_id)

        # PART_OF should NOT appear in spatial relations
        spatial_rels = ontology.relations_of(wheel_id)
        spatial_types = {r.relation for r in spatial_rels}
        assert HCIREdgeType.PART_OF not in spatial_types

        # Car's containers should include garage
        containers = ontology.containers_of(car_id)
        assert garage_id in containers

    def test_consistency_detection(self, world):
        """Detect contradictory spatial relations."""
        ontology = world["ontology"]
        registry = world["registry"]

        t0 = 1000.0
        a_id = registry.discover("A", "object", timestamp=t0)
        b_id = registry.discover("B", "object", timestamp=t0)

        ontology.assert_relation(a_id, HCIREdgeType.ABOVE, b_id)
        ontology.assert_relation(a_id, HCIREdgeType.BELOW, b_id)

        issues = ontology.check_consistency()
        assert len(issues) > 0
        assert any("Contradiction" in issue for issue in issues)


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 4: Event Chronicle
# ═══════════════════════════════════════════════════════════════════════════


class TestEventChronicle:
    """Temporal history with causal chains."""

    def test_event_recording_and_timeline(self, world):
        chronicle = world["chronicle"]
        registry = world["registry"]

        t0 = 1000.0

        entity_id = registry.discover("door", "fixture", timestamp=t0)

        # Record a sequence of events
        e1 = chronicle.record(
            ChronicleEvent(
                event_kind="door_opened",
                subject_entity_id=entity_id,
                timestamp=t0 + 10,
                state_before={"state": "closed"},
                state_after={"state": "open"},
            )
        )

        e2 = chronicle.record(
            ChronicleEvent(
                event_kind="airflow_changed",
                subject_entity_id=entity_id,
                timestamp=t0 + 11,
                state_before={"airflow": "still"},
                state_after={"airflow": "moving"},
                cause_event_id=e1,
            )
        )

        e3 = chronicle.record(
            ChronicleEvent(
                event_kind="temperature_dropped",
                subject_entity_id=entity_id,
                timestamp=t0 + 12,
                state_before={"temp": 22},
                state_after={"temp": 20},
                cause_event_id=e2,
            )
        )

        # Query timeline
        events = chronicle.events_for_entity(entity_id)
        # Should include discovery event + 3 recorded events
        assert len(events) >= 3

        # Causal chain: temp_dropped → airflow_changed → door_opened
        chain = chronicle.causal_chain(e3, direction="backward")
        assert len(chain) == 3
        assert chain[0].id == e3
        assert chain[1].id == e2
        assert chain[2].id == e1

    def test_sequence_detection(self, world):
        chronicle = world["chronicle"]
        registry = world["registry"]

        t0 = 1000.0
        entity_id = registry.discover("sensor", "device", timestamp=t0)

        # Create a repeating pattern: A → B → A → B → A → B
        for i in range(3):
            chronicle.record(
                ChronicleEvent(
                    event_kind="reading_high",
                    subject_entity_id=entity_id,
                    timestamp=t0 + (i * 20),
                )
            )
            chronicle.record(
                ChronicleEvent(
                    event_kind="reading_low",
                    subject_entity_id=entity_id,
                    timestamp=t0 + (i * 20) + 10,
                )
            )

        patterns = chronicle.detect_sequences(entity_id, min_occurrences=2, window_size=2)
        assert len(patterns) > 0

        # Should detect the high→low repeating pattern
        pattern_kinds = [p.event_kinds for p in patterns]
        assert ("reading_high", "reading_low") in pattern_kinds


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 5: Identity Hypothesis
# ═══════════════════════════════════════════════════════════════════════════


class TestIdentityHypothesis:
    """Evidence-based re-identification — POTENTIAL_SAME_AS → IDENTIFIES."""

    def test_evidence_based_reidentification(self, world):
        graph = world["graph"]
        registry = world["registry"]

        t0 = 1000.0

        # Discover entity
        entity_id = registry.discover(
            entity_name="person_A",
            entity_type="person",
            properties={"height": "tall", "shirt": "blue"},
            timestamp=t0,
        )
        registry.track(entity_id, timestamp=t0 + 1)
        registry.occlude(entity_id, timestamp=t0 + 50)

        # New ambiguous observation
        obs = ObservationNode(
            payload={"entity_name": "unknown", "entity_type": "person", "height": "tall"},
            sensor_source="camera_2",
        )
        graph.add_node(obs)

        # Propose as identity candidate (hypothesis, NOT merge)
        registry.propose_reidentification(
            observation_id=obs.id,
            candidate_entity_id=entity_id,
            similarity_score=0.65,
            evidence={"height_match": True, "location_proximity": True},
        )

        # Verify POTENTIAL_SAME_AS edge exists (not IDENTIFIES yet)
        potential_edges = [
            e for e in graph.edges_from(obs.id) if e.edge_type == HCIREdgeType.POTENTIAL_SAME_AS
        ]
        assert len(potential_edges) == 1

        # After more evidence, confirm identity
        registry.confirm_reidentification(obs.id, entity_id, timestamp=t0 + 100)

        # POTENTIAL_SAME_AS should be removed, IDENTIFIES should exist
        potential_edges_after = [
            e for e in graph.edges_from(obs.id) if e.edge_type == HCIREdgeType.POTENTIAL_SAME_AS
        ]
        assert len(potential_edges_after) == 0

        identifies_edges = [
            e for e in graph.edges_from(obs.id) if e.edge_type == HCIREdgeType.IDENTIFIES
        ]
        assert len(identifies_edges) == 1

    def test_three_observations_one_entity(self, world):
        """Three observations of same person → ONE entity with 3 IDENTIFIES edges."""
        graph = world["graph"]
        registry = world["registry"]

        t0 = 1000.0

        # First observation creates entity
        obs1 = ObservationNode(
            payload={"entity_name": "cat", "entity_type": "animal", "color": "black"},
        )
        graph.add_node(obs1)

        entity_id = registry.discover(
            entity_name="cat",
            entity_type="animal",
            observation_id=obs1.id,
            properties={"color": "black"},
            timestamp=t0,
        )
        registry.track(entity_id, timestamp=t0 + 1)

        # Second observation → track
        obs2 = ObservationNode(
            payload={"entity_name": "cat", "entity_type": "animal", "color": "black"},
        )
        graph.add_node(obs2)
        registry.track_observation(entity_id, obs2.id, timestamp=t0 + 10)

        # Third observation → track
        obs3 = ObservationNode(
            payload={"entity_name": "cat", "entity_type": "animal", "color": "black"},
        )
        graph.add_node(obs3)
        registry.track_observation(entity_id, obs3.id, timestamp=t0 + 20)

        # Verify: ONE entity, THREE observations linked via IDENTIFIES
        obs_ids = registry.observations_for_entity(entity_id)
        assert len(obs_ids) == 3
        assert obs1.id in obs_ids
        assert obs2.id in obs_ids
        assert obs3.id in obs_ids

        # Only 1 PhysicalEntityNode
        assert registry.total_entities == 1


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 6: World Reconciliation
# ═══════════════════════════════════════════════════════════════════════════


class TestWorldReconciliation:
    """Observation vs belief producing structured deltas."""

    def test_new_entity_discovery(self, world):
        """Unmatched observation creates a new entity."""
        graph = world["graph"]
        reconciler = world["reconciler"]

        obs = ObservationNode(
            payload={"entity_name": "lamp", "entity_type": "furniture", "brightness": 0.8},
        )
        graph.add_node(obs)

        result = reconciler.reconcile(obs)

        assert result.created_entity_id is not None
        assert any(d.delta_type == DeltaType.NEW_ENTITY for d in result.deltas)

    def test_state_transition_delta(self, world):
        """Property change produces STATE_TRANSITION delta."""
        graph = world["graph"]
        registry = world["registry"]
        reconciler = world["reconciler"]

        t0 = 1000.0

        # Create entity with known state
        entity_id = registry.discover(
            entity_name="light_switch",
            entity_type="switch",
            properties={"state": "off", "brightness": 0.0},
            timestamp=t0,
        )
        registry.track(entity_id, timestamp=t0 + 1)

        # New observation with changed state
        obs = ObservationNode(
            payload={
                "entity_name": "light_switch",
                "entity_type": "switch",
                "state": "on",
                "brightness": 0.8,
            },
        )
        graph.add_node(obs)

        result = reconciler.reconcile(obs, entity_hint=entity_id, timestamp=t0 + 100)

        assert result.matched_entity_id == entity_id

        # Should have a state transition delta
        transition_deltas = [d for d in result.deltas if d.delta_type == DeltaType.STATE_TRANSITION]
        assert len(transition_deltas) >= 1

    def test_permanence_error_detection(self, world):
        """Occluded entity re-observed in wrong location → prediction error."""
        graph = world["graph"]
        registry = world["registry"]
        permanence = world["permanence"]
        reconciler = world["reconciler"]

        t0 = 1000.0

        entity_id = registry.discover(
            entity_name="phone",
            entity_type="device",
            properties={"color": "black"},
            timestamp=t0,
        )
        registry.track(entity_id, timestamp=t0 + 1)
        registry.occlude(entity_id, timestamp=t0 + 10)

        # Generate permanence predictions
        permanence.generate_predictions(
            entity_id=entity_id,
            occlusion_time=t0 + 10,
            spatial_context={"container": "desk"},
        )

        # Re-observe with different location
        obs = ObservationNode(
            payload={
                "entity_name": "phone",
                "entity_type": "device",
                "color": "black",
                "location": {"container": "couch"},  # Different!
            },
        )
        graph.add_node(obs)

        result = reconciler.reconcile(obs, entity_hint=entity_id, timestamp=t0 + 100)

        # Should detect prediction error
        assert len(result.prediction_errors) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 7: Long-Gap Persistence
# ═══════════════════════════════════════════════════════════════════════════


class TestLongGapPersistence:
    """Entity identity survives extended absence without observations."""

    def test_entity_survives_long_gap(self, world):
        """
        t0:  observe E17 in kitchen
        t1:  reason
        t2:  sensor loses E17
        t3:  multiple events for OTHER entities
        t4:  E17 still unobserved
        t5:  query scene
        t6:  E17 remains represented with decayed predictions
        t7:  E17 reappears
        t8:  reconcile — identity survives
        """
        graph = world["graph"]
        registry = world["registry"]
        ontology = world["ontology"]
        permanence = world["permanence"]
        chronicle = world["chronicle"]
        scene_graph = world["scene_graph"]

        t0 = 1000.0

        # t0: Observe E17
        kitchen_id = ontology.create_region("kitchen")
        entity_id = registry.discover(
            entity_name="mug",
            entity_type="container",
            properties={"material": "ceramic", "color": "white"},
            timestamp=t0,
        )
        registry.track(entity_id, timestamp=t0 + 1)
        ontology.assert_relation(entity_id, HCIREdgeType.LOCATED_IN, kitchen_id)

        # t2: Sensor loses E17
        registry.occlude(entity_id, timestamp=t0 + 60)
        permanence.generate_predictions(
            entity_id=entity_id,
            occlusion_time=t0 + 60,
            spatial_context={"container": "kitchen_counter", "region": kitchen_id},
        )

        # t3: Multiple events for OTHER entities
        other_id = registry.discover("plate", "dishware", timestamp=t0 + 100)
        registry.track(other_id, timestamp=t0 + 101)
        chronicle.record(
            ChronicleEvent(
                event_kind="moved",
                subject_entity_id=other_id,
                timestamp=t0 + 200,
            )
        )
        chronicle.record(
            ChronicleEvent(
                event_kind="washed",
                subject_entity_id=other_id,
                timestamp=t0 + 300,
            )
        )

        # t5: Query scene — E17 should still be present (occluded)
        snapshot = scene_graph.snapshot(current_time=t0 + 500)
        mug = snapshot.entity_by_id(entity_id)
        assert mug is not None
        assert mug.lifecycle == EntityLifecycle.OCCLUDED
        assert mug.entity_name == "mug"

        # Permanence predictions should be decayed but present
        assert len(mug.permanence_confidences) > 0
        for dim, conf in mug.permanence_confidences.items():
            assert conf > 0.0  # Not zero — entity is believed to still exist

        # t7: E17 reappears
        obs = ObservationNode(
            payload={"entity_name": "mug", "entity_type": "container", "color": "white"},
        )
        graph.add_node(obs)

        registry.confirm_reidentification(
            observation_id=obs.id,
            entity_id=entity_id,
            timestamp=t0 + 600,
        )

        # t8: Identity survives — same entity, now tracked again
        entity = registry.get_entity(entity_id)
        assert entity is not None
        assert entity.entity_lifecycle == EntityLifecycle.TRACKED
        assert entity.entity_name == "mug"

        # The entity has a complete event history
        all_events = chronicle.events_for_entity(entity_id)
        event_kinds = {e.event_kind for e in all_events}
        assert WorldEventKind.ENTITY_DISCOVERED in event_kinds
        assert WorldEventKind.ENTITY_TRACKED in event_kinds
        assert WorldEventKind.ENTITY_OCCLUDED in event_kinds
        assert WorldEventKind.ENTITY_RE_IDENTIFIED in event_kinds


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 8: Full Integration
# ═══════════════════════════════════════════════════════════════════════════


class TestFullIntegration:
    """End-to-end: perception → reconciler → world model → scene graph → A12-ready."""

    def test_full_pipeline(self, world):
        graph = world["graph"]
        reconciler = world["reconciler"]
        ontology = world["ontology"]
        scene_graph = world["scene_graph"]

        t0 = 1000.0

        # Create spatial regions
        room_id = ontology.create_region("room_A")
        table_id = ontology.create_region("table_1", parent_region_id=room_id)

        # Observation 1: Discover a cup on the table
        obs1 = ObservationNode(
            payload={
                "entity_name": "coffee_cup",
                "entity_type": "cup",
                "temperature": "hot",
            },
        )
        graph.add_node(obs1)
        result1 = reconciler.reconcile(obs1, timestamp=t0)
        cup_id = result1.created_entity_id
        assert cup_id is not None

        # Place cup on table
        ontology.assert_relation(cup_id, HCIREdgeType.LOCATED_IN, table_id)

        # Observation 2: Discover a book nearby
        obs2 = ObservationNode(
            payload={
                "entity_name": "notebook",
                "entity_type": "book",
                "pages": 100,
            },
        )
        graph.add_node(obs2)
        result2 = reconciler.reconcile(obs2, timestamp=t0 + 5)
        book_id = result2.created_entity_id
        assert book_id is not None

        ontology.assert_relation(book_id, HCIREdgeType.NEAR, cup_id)
        ontology.assert_relation(book_id, HCIREdgeType.LOCATED_IN, table_id)

        # Take a scene snapshot
        snapshot = scene_graph.snapshot(current_time=t0 + 10)

        # Verify scene structure
        assert snapshot.entity_count >= 2  # At least cup + book (may include regions)

        # Check spatial queries
        table_contents = scene_graph.entities_in_region(table_id, current_time=t0 + 10)
        table_entity_ids = {e.entity_id for e in table_contents}
        assert cup_id in table_entity_ids
        assert book_id in table_entity_ids

        # Check nearby queries
        cup_scene = snapshot.entity_by_id(cup_id)
        assert cup_scene is not None
        assert book_id in cup_scene.nearby_entities

        # Verify FrozenGraphView bridge to A12
        frozen_view = scene_graph.as_frozen_view()
        assert frozen_view.node_count > 0

        # Verify summary
        summary = scene_graph.summary(current_time=t0 + 10)
        assert summary["tracked"] >= 2
        assert summary["total_chronicle_events"] > 0


# ═══════════════════════════════════════════════════════════════════════════
# Zero-LLM Verification
# ═══════════════════════════════════════════════════════════════════════════


class TestZeroLLM:
    """Verify that no LLM modules are imported or invoked."""

    def test_no_llm_imports(self, world):
        """The A13 world model stack must not import any LLM modules."""
        import sys

        # Modules that indicate LLM dependency
        llm_markers = [
            "hbllm.hcir.world.predictors.llm",
            "openai",
            "anthropic",
            "litellm",
            "langchain",
        ]

        loaded = set(sys.modules.keys())
        for marker in llm_markers:
            assert marker not in loaded, f"LLM module loaded: {marker}"
