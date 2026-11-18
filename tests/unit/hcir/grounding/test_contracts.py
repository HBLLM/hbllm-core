from __future__ import annotations

import dataclasses

import pytest

from hbllm.hcir.grounding.contracts import (
    ActionExecutionRecord,
    ActionLifecycleState,
    ActionValidationResult,
    ClaimOrigin,
    DialogueAct,
    DialogueInterpretationResult,
    EntityCandidateRef,
    EvidenceRecord,
    GroundedConceptRef,
    GroundedEntityRef,
    GroundingClaim,
    GroundingQuality,
    LiteralValueRef,
    ModalitySource,
    Polarity,
    QueryIntent,
    RenderedActionPayload,
    SymbolicActionIntent,
    TrackletRef,
    UnresolvedRef,
)


def test_reference_hierarchy_validation() -> None:
    """Verify that typed reference constructors strictly enforce prefixes and immutability."""
    # Valid references
    trk = TrackletRef(id="trk_vis_001", channel_id="cam_front")
    ecand = EntityCandidateRef(id="ecand_992")
    entity = GroundedEntityRef(id="e_table_42")
    concept = GroundedConceptRef(id="C-00387")

    assert trk.id == "trk_vis_001"
    assert trk.channel_id == "cam_front"
    assert ecand.id == "ecand_992"
    assert entity.id == "e_table_42"
    assert concept.id == "C-00387"

    # Invalidation on wrong prefixes
    with pytest.raises(ValueError, match="must start with 'trk_'"):
        TrackletRef(id="invalid_trk_1")

    with pytest.raises(ValueError, match="must start with 'ecand_'"):
        EntityCandidateRef(id="e_invalid")

    with pytest.raises(ValueError, match="must start with 'e_'"):
        GroundedEntityRef(id="entity_invalid")

    with pytest.raises(ValueError, match="must start with 'C-' or 'concept_'"):
        GroundedConceptRef(id="cup_concept")

    # Immutability check
    with pytest.raises(dataclasses.FrozenInstanceError):
        trk.id = "trk_vis_002"  # type: ignore[misc]


def test_literal_and_unresolved_refs() -> None:
    """Test typed primitive literal values and unresolved placeholder references."""
    lit_float = LiteralValueRef(value=42.5, unit="cm")
    lit_str = LiteralValueRef(value="red")
    lit_bool = LiteralValueRef(value=True)

    assert lit_float.value == 42.5
    assert lit_float.unit == "cm"
    assert lit_str.value == "red"
    assert lit_bool.value is True

    # Reject complex objects
    with pytest.raises(TypeError, match="must be primitive"):
        LiteralValueRef(value={"complex": "object"})  # type: ignore[arg-type]

    # Unresolved reference as a first-class citizen
    unres = UnresolvedRef(
        placeholder_id="target_?",
        constraints={"role": "cargo", "near": "e_table_42"},
    )
    assert unres.placeholder_id == "target_?"
    assert unres.constraints["role"] == "cargo"

    with pytest.raises(ValueError, match="placeholder_id cannot be empty"):
        UnresolvedRef(placeholder_id="")


def test_evidence_record_score_bounds() -> None:
    """Ensure EvidenceRecord strictly validates probability and score boundaries."""
    # Valid evidence
    ev = EvidenceRecord(
        evidence_id="ev_101",
        observation_id="obs_500",
        source_channel_id="cam_front",
        modality=ModalitySource.VISION,
        model_name="yolov8",
        model_version="1.0.0",
        dependency_group="dep_grp_cam1_frame500",
        perceptual_score=0.88,
        calibrated_likelihood=0.92,
        evidence_strength=0.95,
        provenance_hash="a" * 64,
    )
    assert ev.perceptual_score == 0.88
    assert ev.calibrated_likelihood == 0.92

    # Reject out of bounds perceptual score
    with pytest.raises(ValueError, match="perceptual_score must be in range"):
        EvidenceRecord(
            evidence_id="ev_102",
            observation_id="obs_500",
            source_channel_id="cam_front",
            modality=ModalitySource.VISION,
            model_name="yolov8",
            model_version="1.0.0",
            dependency_group="dep_grp_cam1_frame500",
            perceptual_score=1.5,
            calibrated_likelihood=0.92,
            evidence_strength=0.95,
            provenance_hash="a" * 64,
        )

    # Reject empty dependency group
    with pytest.raises(ValueError, match="dependency_group must be non-empty"):
        EvidenceRecord(
            evidence_id="ev_103",
            observation_id="obs_500",
            source_channel_id="cam_front",
            modality=ModalitySource.VISION,
            model_name="yolov8",
            model_version="1.0.0",
            dependency_group="",
            perceptual_score=0.8,
            calibrated_likelihood=0.8,
            evidence_strength=0.8,
            provenance_hash="a" * 64,
        )


def test_grounding_claim_structure() -> None:
    """Ensure GroundingClaim references multiple EvidenceRecord IDs and supports typed refs."""
    subject = EntityCandidateRef(id="ecand_101")
    target_unresolved = UnresolvedRef(
        placeholder_id="target_keys",
        constraints={"relation": "beside", "landmark": "e_sofa_1"},
    )

    claim = GroundingClaim(
        claim_id="claim_881",
        origin=ClaimOrigin.REPORTED_BY_AGENT,
        subject_ref=subject,
        predicate_ref="LOCATED_NEAR",
        object_ref=target_unresolved,
        polarity=Polarity.AFFIRMATIVE,
        evidence_ids=("ev_aud_01", "ev_txt_02"),
        temporal_scope={"valid_from": 100.0, "valid_until": 200.0},
        generated_by="dialogue_adapter_v2",
    )

    assert claim.claim_id == "claim_881"
    assert claim.subject_ref == subject
    assert claim.object_ref == target_unresolved
    assert claim.evidence_ids == ("ev_aud_01", "ev_txt_02")
    assert claim.polarity == Polarity.AFFIRMATIVE

    # Immutability
    with pytest.raises(dataclasses.FrozenInstanceError):
        claim.claim_id = "modified"  # type: ignore[misc]


def test_grounding_quality_7_dimensions() -> None:
    """Verify 7-dimensional GroundingQuality contract and boundary validation."""
    quality = GroundingQuality(
        perceptual_stability=0.9,
        temporal_persistence=0.85,
        relational_consistency=0.8,
        cross_modal_support=0.75,
        predictive_utility=0.92,
        interaction_support=0.88,
        boundary_stability=0.95,
    )
    assert quality.predictive_utility == 0.92

    # Zero and balanced constructors
    zero_q = GroundingQuality.zero()
    assert zero_q.perceptual_stability == 0.0

    balanced = GroundingQuality.balanced(0.7)
    assert balanced.interaction_support == 0.7

    # Reject out of bounds
    with pytest.raises(ValueError, match="must be in"):
        GroundingQuality(
            perceptual_stability=1.2,
            temporal_persistence=0.8,
            relational_consistency=0.8,
            cross_modal_support=0.8,
            predictive_utility=0.8,
            interaction_support=0.8,
            boundary_stability=0.8,
        )


def test_dialogue_act_separation() -> None:
    """Verify DialogueInterpretationResult cleanly separates DialogueAct from claims and queries."""
    res = DialogueInterpretationResult(
        utterance_id="utt_1",
        dialogue_act=DialogueAct.QUESTION,
        queries=(
            QueryIntent(
                query_id="q_1",
                target_variable="location",
                constraints=({"subject": "e_cup_1"},),
            ),
        ),
        confidence=0.98,
    )
    assert res.dialogue_act == DialogueAct.QUESTION
    assert len(res.queries) == 1
    assert len(res.claims) == 0
    assert len(res.goal_proposals) == 0


def test_action_contracts() -> None:
    """Test SymbolicActionIntent, RenderedActionPayload, and ActionValidationResult."""
    intent = SymbolicActionIntent(
        intent_id="intent_001",
        action_type="GRASP",
        target_ref=GroundedEntityRef(id="e_cup_42"),
        parameters={"grip_force": 5.0},
        preconditions=("hand_empty", "in_reach"),
        expected_outcomes=("grasped(e_cup_42)",),
    )
    assert intent.action_type == "GRASP"

    payload = RenderedActionPayload(
        payload_id="payload_001",
        intent_id=intent.intent_id,
        renderer_name="kinematic_renderer",
        execution_channel="arm_controller",
        payload_data={"joint_angles": [0.1, 0.4, 0.9]},
    )
    assert payload.renderer_name == "kinematic_renderer"

    val_res = ActionValidationResult(
        payload_id=payload.payload_id,
        is_authorized=True,
        safety_margin=0.98,
        requires_human_confirmation=False,
    )
    assert val_res.is_authorized is True

    record = ActionExecutionRecord(
        intent_id=intent.intent_id,
        payload_id=payload.payload_id,
        state=ActionLifecycleState.OUTCOME_VERIFIED,
        outcome_verified=True,
        surprise_detected=False,
    )
    assert record.state == ActionLifecycleState.OUTCOME_VERIFIED
