from __future__ import annotations

import pytest

from hbllm.hcir.grounding.contracts.enums import ClaimOrigin, Polarity
from hbllm.hcir.grounding.contracts.evidence import EvidenceRecord
from hbllm.hcir.grounding.contracts.references import GroundedEntityRef
from hbllm.hcir.grounding.relations.dyads import KinematicDyadHistory, SpatialDyad
from hbllm.hcir.grounding.relations.relation_learner import (
    RelationLearner,
)
from hbllm.hcir.grounding.relations.topology import (
    TopologicalInvariant,
    TopologyClassifier,
)


class TestSpatialDyads:
    def test_dyad_initialization_and_validation(self) -> None:
        dyad = SpatialDyad(
            subject_id="e_cup",
            object_id="e_table",
            distance=0.05,
            dx=0.0,
            dy=-0.05,
            contact=True,
            support_ratio=1.0,
        )
        assert dyad.subject_id == "e_cup"
        assert dyad.object_id == "e_table"
        assert dyad.contact is True
        assert dyad.support_ratio == 1.0

        with pytest.raises(ValueError, match="cannot be identical"):
            SpatialDyad(subject_id="e_cup", object_id="e_cup", distance=0.0, dx=0.0, dy=0.0)

        with pytest.raises(ValueError, match="distance cannot be negative"):
            SpatialDyad(subject_id="e_cup", object_id="e_table", distance=-1.0, dx=0.0, dy=0.0)

    def test_kinematic_dyad_history(self) -> None:
        history = KinematicDyadHistory("e_robot", "e_cargo", max_history=10)
        assert history.pair_key == ("e_robot", "e_cargo")
        assert history.sample_count() == 0

        # Add sequential samples maintaining fixed distance
        for t in range(5):
            d = SpatialDyad(
                subject_id="e_robot",
                object_id="e_cargo",
                distance=1.0,
                dx=1.0,
                dy=0.0,
                relative_speed=0.0,
                centroid_speed=1.0,
                contact=True,
                timestamp=100.0 + t,
            )
            history.add_sample(d)

        assert history.sample_count() == 5
        assert history.mean_distance() == pytest.approx(1.0)
        assert history.distance_variance() == pytest.approx(0.0)
        assert history.is_stable_contact() is True
        assert history.is_synchronous_motion(min_centroid_speed=0.1) is True


class TestTopologyClassifier:
    def test_contained_and_contains(self) -> None:
        # Box inside another
        outer = (0.0, 0.0, 10.0, 10.0)
        inner = (2.0, 2.0, 3.0, 3.0)

        topo, ratio = TopologyClassifier.classify_2d_boxes(inner, outer)
        assert topo == TopologicalInvariant.CONTAINED
        assert ratio >= 0.95

        topo, ratio = TopologyClassifier.classify_2d_boxes(outer, inner)
        assert topo == TopologicalInvariant.CONTAINS
        assert ratio >= 0.95

    def test_overlapping(self) -> None:
        box1 = (0.0, 0.0, 5.0, 5.0)
        box2 = (3.0, 3.0, 5.0, 5.0)

        topo, ratio = TopologyClassifier.classify_2d_boxes(box1, box2)
        assert topo == TopologicalInvariant.OVERLAPPING
        assert 0.0 < ratio < 0.95

    def test_above_supported(self) -> None:
        # Table top at y=2.0 (box: 0, 0, 10, 2.0)
        # Cup on top at y=2.0 (box: 4.0, 2.0, 1.0, 1.0)
        table = (0.0, 0.0, 10.0, 2.0)
        cup = (4.0, 2.0, 1.0, 1.0)

        topo, ratio = TopologyClassifier.classify_2d_boxes(cup, table)
        assert topo == TopologicalInvariant.ABOVE_SUPPORTED
        assert ratio == pytest.approx(1.0)

    def test_disjoint(self) -> None:
        box1 = (0.0, 0.0, 2.0, 2.0)
        box2 = (10.0, 10.0, 2.0, 2.0)

        topo, dist = TopologyClassifier.classify_2d_boxes(box1, box2)
        assert topo == TopologicalInvariant.DISJOINT
        assert dist > 5.0


class TestRelationLearner:
    def test_seed_relations(self) -> None:
        learner = RelationLearner(proximity_threshold=5.0)
        rel_on = learner.get_relation("rel_on")
        assert rel_on is not None
        assert rel_on.invariant_type == TopologicalInvariant.ABOVE_SUPPORTED
        assert rel_on.primary_alias == "ON"

    def test_late_bound_linguistic_alias(self) -> None:
        learner = RelationLearner()
        rel_follows = learner.get_relation("rel_follows")
        assert rel_follows is not None

        learner.bind_linguistic_alias("rel_follows", "CHASING", 0.99)
        assert rel_follows.primary_alias == "CHASING"

    def test_observe_and_propose_claims(self) -> None:
        learner = RelationLearner(proximity_threshold=5.0)

        # Avatar and cup on table
        states = {
            "e_table": {"position": (5.0, 2.0), "bbox": (0.0, 0.0, 10.0, 2.0)},
            "e_cup": {"position": (5.0, 2.5), "bbox": (4.5, 2.0, 1.0, 1.0)},
        }

        dyads = learner.observe_entity_states(states, timestamp=10.0)
        assert len(dyads) == 2  # (e_table, e_cup) and (e_cup, e_table)

        claims, evidence = learner.propose_claims(
            source_channel_id="spatial_cam",
            observation_id="obs_frame_1",
        )

        assert len(claims) >= 1
        assert len(evidence) == len(claims)

        # Check claim properties
        cup_claims = [c for c in claims if c.subject_ref == GroundedEntityRef("e_cup")]
        assert len(cup_claims) == 1
        c = cup_claims[0]
        assert c.predicate_ref == "ON"
        assert c.object_ref == GroundedEntityRef("e_table")
        assert c.polarity == Polarity.AFFIRMATIVE
        assert c.origin == ClaimOrigin.DIRECT_SENSORY

        # Check evidence record
        ev = evidence[0]
        assert isinstance(ev, EvidenceRecord)
        assert ev.observation_id == "obs_frame_1"
        assert ev.perceptual_score > 0.8

    def test_emergent_synchronous_motion_induction(self) -> None:
        learner = RelationLearner(proximity_threshold=10.0)

        # Simulate entity B following entity A over 5 time steps at fixed distance 2.0
        for t in range(6):
            states = {
                "e_leader": {
                    "position": (float(t * 2), 0.0),
                    "velocity": (2.0, 0.0),
                },
                "e_follower": {
                    "position": (float(t * 2) - 2.0, 0.0),
                    "velocity": (2.0, 0.0),
                },
            }
            learner.observe_entity_states(states, timestamp=100.0 + t)

        induced = learner.induce_relations()
        assert len(induced) >= 1
        sync_rel = induced[0]
        assert sync_rel.invariant_type == TopologicalInvariant.SYNCHRONOUS_MOTION
        assert "FOLLOWS" in sync_rel.linguistic_aliases
