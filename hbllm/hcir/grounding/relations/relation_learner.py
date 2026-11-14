from __future__ import annotations

import hashlib
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.grounding.contracts.claims import GroundingClaim
from hbllm.hcir.grounding.contracts.enums import ClaimOrigin, ModalitySource, Polarity
from hbllm.hcir.grounding.contracts.evidence import EvidenceRecord
from hbllm.hcir.grounding.contracts.references import (
    EntityCandidateRef,
    GroundedEntityRef,
    GroundedRef,
)
from hbllm.hcir.grounding.relations.dyads import KinematicDyadHistory, SpatialDyad
from hbllm.hcir.grounding.relations.topology import TopologicalInvariant, TopologyClassifier


@dataclass
class LearnedRelation:
    """An emergent or canonical dyadic relationship pattern between entities."""

    relation_id: str
    invariant_type: TopologicalInvariant
    distance_bounds: tuple[float, float] = (0.0, float("inf"))
    velocity_correlation_min: float = 0.0
    linguistic_aliases: dict[str, float] = field(default_factory=dict)
    support_count: int = 1
    confidence: float = 0.8
    created_at: float = field(default_factory=time.time)

    @property
    def primary_alias(self) -> str:
        """Returns highest-confidence linguistic token, or relation_id if unaliased."""
        if not self.linguistic_aliases:
            return self.relation_id
        return max(self.linguistic_aliases.items(), key=lambda item: item[1])[0]


class RelationLearner:
    """Learns and induces dynamic spatial, kinematic, and topological relations.

    Adheres to SGE Section 8 & Stage 6:
    Instead of hardcoded dictionaries, relations are induced from relative kinematic,
    spatial, and topological invariants between tracked entities, then late-bound
    to linguistic tokens.
    """

    def __init__(self, proximity_threshold: float = 5.0) -> None:
        self._proximity_threshold = proximity_threshold
        self._relations: dict[str, LearnedRelation] = {}
        self._dyad_histories: dict[tuple[str, str], KinematicDyadHistory] = {}
        self._seed_canonical_relations()

    def _seed_canonical_relations(self) -> None:
        """Pre-seeds canonical spatial invariants while allowing emergent ones."""
        self._relations["rel_on"] = LearnedRelation(
            relation_id="rel_on",
            invariant_type=TopologicalInvariant.ABOVE_SUPPORTED,
            distance_bounds=(0.0, 0.5),
            linguistic_aliases={"ON": 1.0, "ABOVE": 0.85},
            confidence=0.95,
        )
        self._relations["rel_in"] = LearnedRelation(
            relation_id="rel_in",
            invariant_type=TopologicalInvariant.CONTAINED,
            distance_bounds=(0.0, 0.2),
            linguistic_aliases={"IN": 1.0, "INSIDE": 0.95},
            confidence=0.95,
        )
        self._relations["rel_touching"] = LearnedRelation(
            relation_id="rel_touching",
            invariant_type=TopologicalInvariant.TOUCHING,
            distance_bounds=(0.0, 0.1),
            linguistic_aliases={"TOUCHING": 1.0, "CONTACT": 0.9},
            confidence=0.9,
        )
        self._relations["rel_near"] = LearnedRelation(
            relation_id="rel_near",
            invariant_type=TopologicalInvariant.DISJOINT,
            distance_bounds=(0.1, self._proximity_threshold),
            linguistic_aliases={"NEAR": 1.0, "BESIDE": 0.8},
            confidence=0.85,
        )
        self._relations["rel_follows"] = LearnedRelation(
            relation_id="rel_follows",
            invariant_type=TopologicalInvariant.SYNCHRONOUS_MOTION,
            distance_bounds=(0.2, 4.0),
            velocity_correlation_min=0.8,
            linguistic_aliases={"FOLLOWS": 0.95, "CHASING": 0.8},
            confidence=0.9,
        )

    def observe_entity_states(
        self,
        spatial_states: dict[str, dict[str, Any]],
        timestamp: float = 0.0,
    ) -> list[SpatialDyad]:
        """Extracts dyadic spatial metrics for all interacting entity pairs."""
        now = timestamp if timestamp > 0.0 else time.time()
        dyads: list[SpatialDyad] = []
        entity_ids = list(spatial_states.keys())

        for i, sub_id in enumerate(entity_ids):
            sub_st = spatial_states[sub_id]
            sub_pos = sub_st.get("position", (0.0, 0.0))
            sub_vel = sub_st.get("velocity", (0.0, 0.0))
            sub_bbox = sub_st.get("bbox")

            for j, obj_id in enumerate(entity_ids):
                if i == j:
                    continue
                obj_st = spatial_states[obj_id]
                obj_pos = obj_st.get("position", (0.0, 0.0))
                obj_vel = obj_st.get("velocity", (0.0, 0.0))
                obj_bbox = obj_st.get("bbox")

                # Relative displacements
                dx = obj_pos[0] - sub_pos[0]
                dy = obj_pos[1] - sub_pos[1]
                dz = (obj_pos[2] - sub_pos[2]) if len(sub_pos) > 2 and len(obj_pos) > 2 else 0.0
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)

                if dist > self._proximity_threshold and not (sub_bbox and obj_bbox):
                    continue

                # Relative and centroid speeds
                dvx = obj_vel[0] - sub_vel[0]
                dvy = obj_vel[1] - sub_vel[1]
                dvz = (obj_vel[2] - sub_vel[2]) if len(sub_vel) > 2 and len(obj_vel) > 2 else 0.0
                rel_speed = math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)

                cvx = (obj_vel[0] + sub_vel[0]) / 2.0
                cvy = (obj_vel[1] + sub_vel[1]) / 2.0
                cvz = (
                    (obj_vel[2] + sub_vel[2]) / 2.0
                    if len(sub_vel) > 2 and len(obj_vel) > 2
                    else 0.0
                )
                centroid_speed = math.sqrt(cvx * cvx + cvy * cvy + cvz * cvz)

                # Topology metrics
                contact = False
                containment = 0.0
                support = 0.0
                if sub_bbox and obj_bbox:
                    topo, ratio = TopologyClassifier.classify_2d_boxes(sub_bbox, obj_bbox)
                    if topo in (
                        TopologicalInvariant.TOUCHING,
                        TopologicalInvariant.ABOVE_SUPPORTED,
                        TopologicalInvariant.BELOW_SUPPORTING,
                        TopologicalInvariant.OVERLAPPING,
                    ):
                        contact = True
                    if topo == TopologicalInvariant.CONTAINED:
                        containment = ratio
                    elif topo == TopologicalInvariant.ABOVE_SUPPORTED:
                        support = ratio

                bearing = math.atan2(dy, dx)

                dyad = SpatialDyad(
                    subject_id=sub_id,
                    object_id=obj_id,
                    distance=dist,
                    dx=dx,
                    dy=dy,
                    dz=dz,
                    relative_speed=rel_speed,
                    centroid_speed=centroid_speed,
                    contact=contact,
                    containment_ratio=containment,
                    support_ratio=support,
                    bearing_rad=bearing,
                    timestamp=now,
                )
                dyads.append(dyad)

                # Update history
                pair_key = (sub_id, obj_id)
                if pair_key not in self._dyad_histories:
                    self._dyad_histories[pair_key] = KinematicDyadHistory(sub_id, obj_id)
                self._dyad_histories[pair_key].add_sample(dyad)

        return dyads

    def induce_relations(self) -> list[LearnedRelation]:
        """Induces emergent relations from dyad histories (e.g. synchronous trajectory / chasing)."""
        induced: list[LearnedRelation] = []

        for pair_key, history in self._dyad_histories.items():
            if history.sample_count() < 4:
                continue

            # Check for synchronous motion
            if history.is_synchronous_motion(max_dist_variance=0.25, min_centroid_speed=0.1):
                mean_d = history.mean_distance()
                rel_id = f"rel_sync_{int(mean_d)}"
                if rel_id not in self._relations:
                    rel = LearnedRelation(
                        relation_id=rel_id,
                        invariant_type=TopologicalInvariant.SYNCHRONOUS_MOTION,
                        distance_bounds=(max(0.0, mean_d - 0.5), mean_d + 0.5),
                        velocity_correlation_min=0.85,
                        linguistic_aliases={"FOLLOWS": 0.9, "ACCOMPANIES": 0.8},
                        confidence=0.88,
                    )
                    self._relations[rel_id] = rel
                    induced.append(rel)
                else:
                    self._relations[rel_id].support_count += 1
                    self._relations[rel_id].confidence = min(
                        0.98, self._relations[rel_id].confidence + 0.02
                    )

        return induced

    def propose_claims(
        self,
        source_channel_id: str = "spatial_stream",
        observation_id: str = "obs_dyad",
        dependency_group: str = "spatial_tracker",
    ) -> tuple[list[GroundingClaim], list[EvidenceRecord]]:
        """Synthesizes uncommitted GroundingClaim proposals and calibrated EvidenceRecords."""
        claims: list[GroundingClaim] = []
        evidence_records: list[EvidenceRecord] = []
        now = time.time()

        for (sub_id, obj_id), history in self._dyad_histories.items():
            latest = history.latest()
            if not latest:
                continue

            sub_ref = self._to_ref(sub_id)
            obj_ref = self._to_ref(obj_id)

            # Match against known relations
            matched_rel = self._match_relation(latest, history)
            if not matched_rel:
                continue

            ev_id = f"ev_rel_{uuid.uuid4().hex[:8]}"
            claim_id = f"claim_rel_{uuid.uuid4().hex[:8]}"
            predicate = matched_rel.primary_alias

            ev = EvidenceRecord(
                evidence_id=ev_id,
                observation_id=observation_id,
                source_channel_id=source_channel_id,
                modality=ModalitySource.KINESTHETIC,
                model_name="KinematicTopologyInducer",
                model_version="1.0.0",
                dependency_group=dependency_group,
                perceptual_score=matched_rel.confidence,
                calibrated_likelihood=matched_rel.confidence,
                evidence_strength=min(1.0, 0.5 + 0.1 * history.sample_count()),
                provenance_hash=hashlib.sha256(f"{ev_id}:{now}".encode()).hexdigest(),
                timestamp=now,
            )
            evidence_records.append(ev)

            claim = GroundingClaim(
                claim_id=claim_id,
                origin=ClaimOrigin.DIRECT_SENSORY,
                subject_ref=sub_ref,
                predicate_ref=predicate,
                object_ref=obj_ref,
                polarity=Polarity.AFFIRMATIVE,
                evidence_ids=(ev_id,),
                temporal_scope={"start": latest.timestamp, "end": now},
                spatial_scope={"dx": latest.dx, "dy": latest.dy, "distance": latest.distance},
                generated_by="RelationLearner",
                created_at=now,
            )
            claims.append(claim)

        return claims, evidence_records

    def _match_relation(
        self,
        dyad: SpatialDyad,
        history: KinematicDyadHistory,
    ) -> LearnedRelation | None:
        """Determines best matching relation for an instantaneous dyad and history."""
        # 1. Contained
        if dyad.containment_ratio >= 0.8:
            return self._relations.get("rel_in")

        # 2. Resting support / On
        if dyad.support_ratio >= 0.5 or (dyad.contact and dyad.dy > 0 and abs(dyad.dx) < 0.5):
            return self._relations.get("rel_on")

        # 3. Synchronous motion
        if history.is_synchronous_motion():
            return self._relations.get("rel_follows")

        # 4. Contact
        if dyad.contact:
            return self._relations.get("rel_touching")

        # 5. Near
        if dyad.distance <= self._proximity_threshold:
            return self._relations.get("rel_near")

        return None

    def bind_linguistic_alias(self, relation_id: str, alias: str, score: float = 1.0) -> None:
        """Binds a late-bound linguistic token to a relation."""
        if relation_id not in self._relations:
            raise KeyError(f"Relation {relation_id} does not exist")
        self._relations[relation_id].linguistic_aliases[alias.upper()] = score

    def get_relation(self, relation_id: str) -> LearnedRelation | None:
        return self._relations.get(relation_id)

    def all_relations(self) -> list[LearnedRelation]:
        return list(self._relations.values())

    @staticmethod
    def _to_ref(entity_id: str) -> GroundedRef:
        if entity_id.startswith("e_"):
            return GroundedEntityRef(entity_id)
        elif entity_id.startswith("ecand_"):
            return EntityCandidateRef(entity_id)
        return GroundedEntityRef(f"e_{entity_id}")
