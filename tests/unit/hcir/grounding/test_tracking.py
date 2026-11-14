from __future__ import annotations

from hbllm.hcir.grounding.contracts import (
    ClaimOrigin,
    EntityCandidateRef,
    EvidenceRecord,
    GroundedConceptRef,
    GroundedEntityRef,
    ModalitySource,
    Polarity,
    TrackletRef,
    UnresolvedRef,
)
from hbllm.hcir.grounding.tracking import (
    CandidateGenerator,
    EntityAliasOp,
    EntityAssociationOp,
    EntityMergeOp,
    EntityReidentificationOp,
    EntitySplitOp,
    IdentityOperationType,
    ThreeTierEntityTracker,
)


def test_tracklet_to_candidate_association() -> None:
    """Verify tracklet registration, spatial updates, and explicit candidate association."""
    tracker = ThreeTierEntityTracker(spatial_distance_threshold=1.5)
    trk = TrackletRef(id="trk_cam1_001", channel_id="cam_front")

    # Ingest tracklet observation
    tracker.update_tracklet(
        tracklet_ref=trk,
        channel_id="cam_front",
        spatial_coords=(1.0, 2.0, 0.5),
        feature_vector=[0.1, 0.2, 0.3],
        is_active=True,
    )

    # Explicit candidate association
    cand_ref = EntityCandidateRef(id="ecand_cup_42")
    op = tracker.associate_tracklet_to_candidate(
        tracklet_ref=trk,
        candidate_ref=cand_ref,
        confidence=0.95,
        rationale="Visual detection assigned to cup candidate",
    )

    assert isinstance(op, EntityAssociationOp)
    assert op.op_type == IdentityOperationType.ASSOCIATION
    assert op.tracklet_ref == trk
    assert op.candidate_ref == cand_ref

    # Lookup verification
    resolved_cand = tracker.get_candidate_for_tracklet(trk.id)
    assert resolved_cand == cand_ref

    cand_state = tracker.get_candidate(cand_ref.id)
    assert cand_state is not None
    assert cand_state.associated_tracklet_ids == {trk.id}
    assert (1.0, 2.0, 0.5) in cand_state.spatial_trajectory


def test_auto_association_proximity_and_spawn() -> None:
    """Test auto-association via spatial proximity gating and spawning on distance exceedance."""
    tracker = ThreeTierEntityTracker(spatial_distance_threshold=2.0)

    # 1. Tracklet 1 arrives at (0.0, 0.0) -> spawns Candidate 1
    trk1 = TrackletRef(id="trk_vis_1", channel_id="cam1")
    tracker.update_tracklet(trk1, "cam1", spatial_coords=(0.0, 0.0))
    cand1 = tracker.auto_associate(trk1, spatial_coords=(0.0, 0.0))
    assert cand1.id.startswith("ecand_")

    # 2. Tracklet 2 arrives nearby at (0.5, 0.5) (dist ~0.71 < 2.0) -> associates with Candidate 1
    trk2 = TrackletRef(id="trk_vis_2", channel_id="cam1")
    tracker.update_tracklet(trk2, "cam1", spatial_coords=(0.5, 0.5))
    cand2 = tracker.auto_associate(trk2, spatial_coords=(0.5, 0.5))
    assert cand2.id == cand1.id  # Same persistent candidate

    # 3. Tracklet 3 arrives distant at (10.0, 10.0) (dist ~14.1 > 2.0) -> spawns Candidate 2
    trk3 = TrackletRef(id="trk_vis_3", channel_id="cam1")
    tracker.update_tracklet(trk3, "cam1", spatial_coords=(10.0, 10.0))
    cand3 = tracker.auto_associate(trk3, spatial_coords=(10.0, 10.0))
    assert cand3.id != cand1.id  # Distinct candidate entity


def test_reversible_entity_merge_and_split() -> None:
    """Verify that merging candidates preserves prior history and can be completely reversed."""
    tracker = ThreeTierEntityTracker()

    # Candidate A
    trkA = TrackletRef(id="trk_camA_1", channel_id="camA")
    tracker.update_tracklet(trkA, "camA", spatial_coords=(1.0, 1.0))
    candA = EntityCandidateRef(id="ecand_A")
    tracker.associate_tracklet_to_candidate(trkA, candA)

    # Candidate B
    trkB = TrackletRef(id="trk_camB_1", channel_id="camB")
    tracker.update_tracklet(trkB, "camB", spatial_coords=(1.2, 1.1))
    candB = EntityCandidateRef(id="ecand_B")
    tracker.associate_tracklet_to_candidate(trkB, candB)

    # Execute Merge
    merge_op = tracker.merge_candidates(
        source_candidates=(candA, candB),
        confidence=0.92,
        rationale="Cross-camera correspondence matched",
    )
    assert isinstance(merge_op, EntityMergeOp)
    assert merge_op.op_type == IdentityOperationType.MERGE
    merged_cand = merge_op.merged_candidate

    # Verify merged state
    assert tracker.get_candidate(candA.id) is None
    assert tracker.get_candidate(candB.id) is None
    merged_state = tracker.get_candidate(merged_cand.id)
    assert merged_state is not None
    assert merged_state.associated_tracklet_ids == {trkA.id, trkB.id}
    assert tracker.get_candidate_for_tracklet(trkA.id) == merged_cand
    assert tracker.get_candidate_for_tracklet(trkB.id) == merged_cand

    # Reversal: Split executed due to contradictory evidence (e.g. both appear at once)
    split_op = tracker.split_candidate(
        candidate_to_split=merged_cand,
        merge_operation_ref=merge_op.operation_id,
        rationale="Contradiction detected: two objects appeared simultaneously",
    )
    assert isinstance(split_op, EntitySplitOp)
    assert split_op.op_type == IdentityOperationType.SPLIT

    # Verify restored state
    assert tracker.get_candidate(merged_cand.id) is None
    restoredA = tracker.get_candidate(candA.id)
    restoredB = tracker.get_candidate(candB.id)
    assert restoredA is not None
    assert restoredB is not None
    assert restoredA.associated_tracklet_ids == {trkA.id}
    assert restoredB.associated_tracklet_ids == {trkB.id}
    assert tracker.get_candidate_for_tracklet(trkA.id) == candA
    assert tracker.get_candidate_for_tracklet(trkB.id) == candB


def test_candidate_reidentification_with_grounded_entity() -> None:
    """Test re-identifying an active candidate against a canonical GroundedEntityRef."""
    tracker = ThreeTierEntityTracker()
    cand = EntityCandidateRef(id="ecand_desk_cup")
    trk = TrackletRef(id="trk_desk_01", channel_id="cam_desk")

    tracker.update_tracklet(trk, "cam_desk", spatial_coords=(2.0, 3.0))
    tracker.associate_tracklet_to_candidate(trk, cand)

    grounded_entity = GroundedEntityRef(id="e_mug_dumith_42")

    reid_op = tracker.reidentify_entity(
        candidate_ref=cand,
        grounded_ref=grounded_entity,
        confidence=0.98,
        similarity_metrics={"embedding_cosine": 0.96, "color_hist": 0.99},
        rationale="Identified known favorite mug by visual embeddings",
    )
    assert isinstance(reid_op, EntityReidentificationOp)
    assert reid_op.grounded_ref == grounded_entity

    cand_state = tracker.get_candidate(cand.id)
    assert cand_state is not None
    assert cand_state.grounded_entity_ref == grounded_entity


def test_entity_alias_binding() -> None:
    """Test binding contextual alias labels to entities without graph duplication."""
    tracker = ThreeTierEntityTracker()
    cand = EntityCandidateRef(id="ecand_backpack")
    trk = TrackletRef(id="trk_bp_01", channel_id="cam_door")
    tracker.update_tracklet(trk, "cam_door")
    tracker.associate_tracklet_to_candidate(trk, cand)

    alias_op = tracker.add_alias(
        target_ref=cand,
        alias_label="red backpack",
        alias_context="spoken by user",
    )
    assert isinstance(alias_op, EntityAliasOp)
    assert alias_op.alias_label == "red backpack"

    cand_state = tracker.get_candidate(cand.id)
    assert cand_state is not None
    assert "red backpack" in cand_state.aliases


def test_candidate_generator_proposals() -> None:
    """Verify CandidateGenerator constructs typed, provenance-backed GroundingClaims."""
    generator = CandidateGenerator(generator_name="test_vision_generator")

    ev = EvidenceRecord(
        evidence_id="ev_test_101",
        observation_id="obs_frame_101",
        source_channel_id="cam_front",
        modality=ModalitySource.VISION,
        model_name="yolov8",
        model_version="8.0",
        dependency_group="dep_frame_101",
        perceptual_score=0.92,
        calibrated_likelihood=0.90,
        evidence_strength=0.95,
        provenance_hash="prov_101",
    )

    cand = EntityCandidateRef(id="ecand_object_1")

    # 1. Property claim: COLOR(ecand_1, RED)
    prop_claim = generator.generate_property_claim(
        subject_ref=cand,
        property_name="color",
        value="red",
        evidence_records=[ev],
        origin=ClaimOrigin.DIRECT_SENSORY,
        polarity=Polarity.AFFIRMATIVE,
    )
    assert prop_claim.predicate_ref == "COLOR"
    assert prop_claim.evidence_ids == ("ev_test_101",)
    assert prop_claim.polarity == Polarity.AFFIRMATIVE

    # 2. Relational claim: LOCATED_NEAR(ecand_1, e_sofa_1)
    sofa = GroundedEntityRef(id="e_sofa_1")
    rel_claim = generator.generate_relation_claim(
        subject_ref=cand,
        relation_name="located_near",
        object_ref=sofa,
        evidence_records=[ev],
    )
    assert rel_claim.predicate_ref == "LOCATED_NEAR"
    assert rel_claim.object_ref == sofa

    # 3. Relational claim with UnresolvedRef: LOCATED_NEAR(ecand_1, ?)
    unresolved_target = UnresolvedRef(
        placeholder_id="target_table",
        constraints={"category": "table"},
    )
    unres_claim = generator.generate_relation_claim(
        subject_ref=cand,
        relation_name="on_top_of",
        object_ref=unresolved_target,
        evidence_records=[ev],
    )
    assert unres_claim.predicate_ref == "ON_TOP_OF"
    assert unres_claim.object_ref == unresolved_target

    # 4. Concept membership claim: INSTANCE_OF(ecand_1, C-00387)
    concept = GroundedConceptRef(id="C-00387")
    concept_claim = generator.generate_concept_membership_claim(
        candidate_ref=cand,
        concept_ref=concept,
        evidence_records=[ev],
    )
    assert concept_claim.predicate_ref == "INSTANCE_OF"
    assert concept_claim.object_ref == concept
