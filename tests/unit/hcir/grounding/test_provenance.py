from __future__ import annotations

import hashlib

from hbllm.hcir.grounding.contracts import (
    BeliefState,
    ClaimOrigin,
    EntityCandidateRef,
    EvidenceRecord,
    GroundedEntityRef,
    GroundingClaim,
    ModalitySource,
    ObservationEnvelope,
    Polarity,
    TrafficClass,
)


def test_deterministic_provenance_hash_generation() -> None:
    """Verify that ObservationEnvelope creates deterministic provenance hashes."""
    env1 = ObservationEnvelope.create(
        observation_id="obs_001",
        source_modality=ModalitySource.VISION,
        source_channel_id="cam_front",
        traffic_class=TrafficClass.CLASS_HIGH_FREQ,
        sequence_number=1,
        raw_payload_ref="blob://frames/001.bin",
        wall_timestamp=1700000000.0,
        monotonic_timestamp=100.0,
    )
    env2 = ObservationEnvelope.create(
        observation_id="obs_001",
        source_modality=ModalitySource.VISION,
        source_channel_id="cam_front",
        traffic_class=TrafficClass.CLASS_HIGH_FREQ,
        sequence_number=1,
        raw_payload_ref="blob://frames/001.bin",
        wall_timestamp=1700000000.0,
        monotonic_timestamp=100.0,
    )

    assert env1.provenance_hash != ""
    assert env1.provenance_hash == env2.provenance_hash

    # Manual verification against expected SHA256
    expected_seed = "obs_001:cam_front:1:1700000000.0:blob://frames/001.bin"
    expected_hash = hashlib.sha256(expected_seed.encode("utf-8")).hexdigest()
    assert env1.provenance_hash == expected_hash


def test_tamper_detection_in_provenance_seed() -> None:
    """Modifying payload ref or sequence number alters provenance hash."""
    env_base = ObservationEnvelope.create(
        observation_id="obs_001",
        source_modality=ModalitySource.VISION,
        source_channel_id="cam_front",
        traffic_class=TrafficClass.CLASS_HIGH_FREQ,
        sequence_number=1,
        raw_payload_ref="blob://frames/001.bin",
        wall_timestamp=1700000000.0,
    )
    env_tampered = ObservationEnvelope.create(
        observation_id="obs_001",
        source_modality=ModalitySource.VISION,
        source_channel_id="cam_front",
        traffic_class=TrafficClass.CLASS_HIGH_FREQ,
        sequence_number=2,  # altered sequence
        raw_payload_ref="blob://frames/001.bin",
        wall_timestamp=1700000000.0,
    )

    assert env_base.provenance_hash != env_tampered.provenance_hash


def test_end_to_end_provenance_chain() -> None:
    """Ensure provenance flows continuously from Observation to Evidence to Claim to Belief."""
    # 1. Observation
    obs = ObservationEnvelope.create(
        observation_id="obs_frame_901",
        source_modality=ModalitySource.VISION,
        source_channel_id="cam_table",
        traffic_class=TrafficClass.CLASS_HIGH_FREQ,
        sequence_number=901,
        raw_payload_ref="blob://cam_table/901.png",
    )

    # 2. Evidence Record referencing Observation and preserving provenance hash
    ev = EvidenceRecord(
        evidence_id="ev_det_901",
        observation_id=obs.observation_id,
        source_channel_id=obs.source_channel_id,
        modality=obs.source_modality,
        model_name="yolov8_segment",
        model_version="8.2.0",
        dependency_group="dep_table_901",
        perceptual_score=0.91,
        calibrated_likelihood=0.94,
        evidence_strength=0.96,
        provenance_hash=obs.provenance_hash,
    )
    assert ev.observation_id == obs.observation_id
    assert ev.provenance_hash == obs.provenance_hash

    # 3. GroundingClaim referencing Evidence
    claim = GroundingClaim(
        claim_id="claim_cup_on_table",
        origin=ClaimOrigin.DIRECT_SENSORY,
        subject_ref=EntityCandidateRef(id="ecand_cup_1"),
        predicate_ref="ON",
        object_ref=GroundedEntityRef(id="e_table_1"),
        polarity=Polarity.AFFIRMATIVE,
        evidence_ids=(ev.evidence_id,),
        generated_by="spatial_topology_detector",
    )
    assert claim.evidence_ids == ("ev_det_901",)

    # 4. BeliefState referencing Provenance Claims and Evidence IDs
    belief = BeliefState(
        belief_id="belief_cup_on_table",
        subject_ref=claim.subject_ref,
        predicate_ref=claim.predicate_ref,
        object_ref=claim.object_ref,
        polarity=claim.polarity,
        posterior_probability=0.94,
        evidence_strength=0.96,
        supporting_evidence_ids=claim.evidence_ids,
        provenance_claims=(claim.claim_id,),
    )
    assert belief.supporting_evidence_ids == ("ev_det_901",)
    assert belief.provenance_claims == ("claim_cup_on_table",)
