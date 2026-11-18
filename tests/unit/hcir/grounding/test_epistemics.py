from __future__ import annotations

import pytest

from hbllm.hcir.grounding.contracts import (
    BeliefState,
    ClaimOrigin,
    DecisionOutcome,
    EntityCandidateRef,
    EvidenceRecord,
    GroundedEntityRef,
    GroundingClaim,
    LiteralValueRef,
    ModalitySource,
    Polarity,
)
from hbllm.hcir.grounding.epistemics import (
    BeliefRevisionEngine,
    ContradictionType,
    DependencyGroupEvidenceFusion,
    EpistemicEvaluator,
    EvidenceCalibrator,
    RevisionActionType,
)


def test_platt_scaling_evidence_calibration() -> None:
    """Verify Platt scaling maps raw scores to calibrated probabilities and clamps extremes."""
    calibrator = EvidenceCalibrator()

    # Raw score 0.99 from an overconfident model maps to a lower calibrated likelihood
    cal_overconf = calibrator.calibrate(0.99, model_name="overconfident_vision")
    assert cal_overconf < 0.90
    assert 0.001 <= cal_overconf <= 0.999

    # Calibrate an EvidenceRecord
    ev = EvidenceRecord(
        evidence_id="ev_raw",
        observation_id="obs_raw",
        source_channel_id="cam_main",
        modality=ModalitySource.VISION,
        model_name="overconfident_vision",
        model_version="1.0",
        dependency_group="dep_1",
        perceptual_score=0.95,
        calibrated_likelihood=0.95,  # Uncalibrated placeholder
        evidence_strength=0.90,
        provenance_hash="hash",
    )
    calibrated_ev = calibrator.calibrate_record(ev)
    assert calibrated_ev.perceptual_score == 0.95
    assert calibrated_ev.calibrated_likelihood != 0.95
    assert pytest.approx(calibrated_ev.calibrated_likelihood, abs=0.05) == cal_overconf


def test_dependency_group_bayesian_fusion_inflation_prevention() -> None:
    """Verify that models sharing a dependency group do not artificially inflate posterior."""
    fusion = DependencyGroupEvidenceFusion(default_prior=0.5)

    def make_ev(ev_id: str, dep_grp: str, p: float) -> EvidenceRecord:
        return EvidenceRecord(
            evidence_id=ev_id,
            observation_id="obs_1",
            source_channel_id="cam_1",
            modality=ModalitySource.VISION,
            model_name="detector",
            model_version="1.0",
            dependency_group=dep_grp,
            perceptual_score=p,
            calibrated_likelihood=p,
            evidence_strength=0.9,
            provenance_hash="h",
        )

    # Case A: 3 models (YOLO, SAM, CLIP) all looking at the exact SAME image frame (dep_frame_100)
    ev_shared = [
        make_ev("ev_1", "dep_frame_100", 0.8),
        make_ev("ev_2", "dep_frame_100", 0.82),
        make_ev("ev_3", "dep_frame_100", 0.78),
    ]
    res_shared = fusion.fuse(ev_shared)
    assert res_shared.independent_group_count == 1
    assert res_shared.total_evidence_count == 3
    # Posterior should be ~0.80 (reflecting a single observation group), NOT compounded 3 times
    assert 0.75 <= res_shared.posterior_probability <= 0.85

    # Case B: 3 independent observations from 3 distinct camera frames/times
    ev_independent = [
        make_ev("ev_1", "dep_frame_100", 0.8),
        make_ev("ev_2", "dep_frame_101", 0.82),
        make_ev("ev_3", "dep_frame_102", 0.78),
    ]
    res_independent = fusion.fuse(ev_independent)
    assert res_independent.independent_group_count == 3
    # Independent observations compound Bayesian evidence: posterior will be much higher (~0.98+)
    assert res_independent.posterior_probability > 0.95

    # Proves dependency-group prevents false Bayesian compounding:
    assert res_shared.posterior_probability < res_independent.posterior_probability


def test_contradiction_detection_types() -> None:
    """Verify detection of polarity conflicts, property exclusivity, and spatial impossibility."""
    revision = BeliefRevisionEngine(max_physical_velocity_mps=30.0)
    entity = GroundedEntityRef(id="e_cup_42")

    # Active Belief: e_cup_42 has COLOR=RED
    active_belief_color = BeliefState(
        belief_id="b_color_red",
        subject_ref=entity,
        predicate_ref="COLOR",
        object_ref=LiteralValueRef(value="red"),
        polarity=Polarity.AFFIRMATIVE,
        posterior_probability=0.92,
        evidence_strength=0.95,
        supporting_evidence_ids=("ev_vis_1",),
        last_updated=100.0,
        temporal_validity={"coords": (0.0, 0.0)},
    )

    # 1. Polarity Conflict on COLOR=RED
    claim_polarity_conflict = GroundingClaim(
        claim_id="cl_neg_red",
        origin=ClaimOrigin.DIRECT_SENSORY,
        subject_ref=entity,
        predicate_ref="COLOR",
        object_ref=LiteralValueRef(value="red"),
        polarity=Polarity.NEGATIVE,  # Opposite polarity
        evidence_ids=("ev_vis_2",),
    )
    contra_pol = revision.detect_contradictions(claim_polarity_conflict, [active_belief_color])
    assert len(contra_pol) == 1
    assert contra_pol[0].contradiction_type == ContradictionType.POLARITY_CONFLICT

    # 2. Property Exclusivity: COLOR=BLUE vs COLOR=RED
    claim_blue = GroundingClaim(
        claim_id="cl_blue",
        origin=ClaimOrigin.DIRECT_SENSORY,
        subject_ref=entity,
        predicate_ref="COLOR",
        object_ref=LiteralValueRef(value="blue"),
        polarity=Polarity.AFFIRMATIVE,
        evidence_ids=("ev_vis_3",),
    )
    contra_prop = revision.detect_contradictions(claim_blue, [active_belief_color])
    assert len(contra_prop) == 1
    assert contra_prop[0].contradiction_type == ContradictionType.PROPERTY_EXCLUSIVITY

    # 3. Spatial Impossibility: e_cup_42 moved 100 meters in 0.5s -> 200 m/s > 30 m/s
    active_belief_loc = BeliefState(
        belief_id="b_loc_1",
        subject_ref=entity,
        predicate_ref="LOCATED_AT",
        object_ref=LiteralValueRef(value="kitchen"),
        polarity=Polarity.AFFIRMATIVE,
        posterior_probability=0.95,
        evidence_strength=0.95,
        supporting_evidence_ids=("ev_vis_loc1",),
        last_updated=100.0,
        temporal_validity={"coords": (0.0, 0.0)},
    )
    claim_teleport = GroundingClaim(
        claim_id="cl_teleport",
        origin=ClaimOrigin.DIRECT_SENSORY,
        subject_ref=entity,
        predicate_ref="LOCATED_AT",
        object_ref=LiteralValueRef(value="garden"),
        polarity=Polarity.AFFIRMATIVE,
        evidence_ids=("ev_vis_loc2",),
        created_at=100.5,  # dt = 0.5s
        spatial_scope={"coords": (100.0, 0.0)},  # dist = 100m
    )
    contra_spatial = revision.detect_contradictions(claim_teleport, [active_belief_loc])
    assert len(contra_spatial) == 1
    assert contra_spatial[0].contradiction_type == ContradictionType.SPATIAL_IMPOSSIBILITY


def test_belief_revision_lifecycle() -> None:
    """Verify belief confirmation, suspension under severe conflict, and withdrawal."""
    revision = BeliefRevisionEngine(confirmation_threshold=0.85, withdrawal_threshold=0.20)
    fusion = DependencyGroupEvidenceFusion()
    entity = GroundedEntityRef(id="e_door_1")

    base_belief = BeliefState(
        belief_id="b_door_open",
        subject_ref=entity,
        predicate_ref="OPEN",
        object_ref=None,
        polarity=Polarity.AFFIRMATIVE,
        posterior_probability=0.75,
        evidence_strength=0.80,
        supporting_evidence_ids=("ev_init",),
    )

    # 1. Corroborating evidence raises posterior -> CONFIRMED
    ev_corroborate = EvidenceRecord(
        evidence_id="ev_corr",
        observation_id="obs_corr",
        source_channel_id="cam_door",
        modality=ModalitySource.VISION,
        model_name="detector",
        model_version="1.0",
        dependency_group="dep_corr_1",
        perceptual_score=0.92,
        calibrated_likelihood=0.92,
        evidence_strength=0.90,
        provenance_hash="h1",
    )
    fused_corr = fusion.fuse([ev_corroborate], prior_probability=base_belief.posterior_probability)
    res_conf = revision.revise_belief(
        current_belief=base_belief,
        fusion_result=fused_corr,
        contradictions=[],
        new_evidence_ids=("ev_corr",),
    )
    assert res_conf.action_type == RevisionActionType.CONFIRMED
    assert res_conf.updated_belief is not None
    assert res_conf.updated_belief.posterior_probability >= 0.85
    assert "ev_corr" in res_conf.updated_belief.supporting_evidence_ids

    # 2. Severe contradiction with equal evidence strength -> SUSPENDED
    fake_contra = revision.detect_contradictions(
        GroundingClaim(
            claim_id="cl_closed",
            origin=ClaimOrigin.DIRECT_SENSORY,
            subject_ref=entity,
            predicate_ref="OPEN",
            object_ref=None,
            polarity=Polarity.NEGATIVE,
            evidence_ids=("ev_closed",),
        ),
        [base_belief],
    )
    res_susp = revision.revise_belief(
        current_belief=base_belief,
        fusion_result=fused_corr,
        contradictions=fake_contra,
        new_evidence_ids=("ev_closed",),
    )
    assert res_susp.action_type == RevisionActionType.SUSPENDED
    assert res_susp.updated_belief is not None
    assert res_susp.updated_belief.posterior_probability == 0.5  # Elevated uncertainty

    # 3. Falsifying evidence drops posterior below 0.20 -> WITHDRAWN
    ev_falsify = EvidenceRecord(
        evidence_id="ev_false",
        observation_id="obs_f",
        source_channel_id="cam_door",
        modality=ModalitySource.VISION,
        model_name="detector",
        model_version="1.0",
        dependency_group="dep_f_1",
        perceptual_score=0.05,
        calibrated_likelihood=0.05,
        evidence_strength=0.95,
        provenance_hash="h2",
    )
    fused_false = fusion.fuse([ev_falsify], prior_probability=base_belief.posterior_probability)
    res_withdrawn = revision.revise_belief(
        current_belief=base_belief,
        fusion_result=fused_false,
        contradictions=[],
    )
    assert res_withdrawn.action_type == RevisionActionType.WITHDRAWN
    assert res_withdrawn.updated_belief is None


def test_epistemic_evaluator_dispositions() -> None:
    """Verify EpistemicEvaluator emits ACCEPT, REJECT, DEFER, and REQUEST_EVIDENCE."""
    evaluator = EpistemicEvaluator(acceptance_threshold=0.85, rejection_threshold=0.20)
    cand = EntityCandidateRef(id="ecand_target")

    def make_ev(score: float, strength: float, dep: str) -> EvidenceRecord:
        return EvidenceRecord(
            evidence_id=f"ev_{dep}",
            observation_id=f"obs_{dep}",
            source_channel_id="cam_test",
            modality=ModalitySource.VISION,
            model_name="default",
            model_version="1.0",
            dependency_group=dep,
            perceptual_score=score,
            calibrated_likelihood=score,
            evidence_strength=strength,
            provenance_hash="h",
        )

    # 1. High confidence, multi-group evidence -> ACCEPT
    claim_accept = GroundingClaim(
        claim_id="cl_accept",
        origin=ClaimOrigin.DIRECT_SENSORY,
        subject_ref=cand,
        predicate_ref="TYPE",
        object_ref=LiteralValueRef(value="mug"),
        polarity=Polarity.AFFIRMATIVE,
        evidence_ids=("ev_dep1", "ev_dep2"),
    )
    dec_accept, belief = evaluator.evaluate_claim(
        claim=claim_accept,
        evidence_records=[make_ev(0.95, 0.95, "dep1"), make_ev(0.92, 0.95, "dep2")],
    )
    assert dec_accept.outcome == DecisionOutcome.ACCEPT
    assert dec_accept.transaction_proposal_id is not None
    assert belief is not None
    assert belief.posterior_probability >= 0.85

    # 2. Low confidence -> REJECT
    claim_reject = GroundingClaim(
        claim_id="cl_reject",
        origin=ClaimOrigin.DIRECT_SENSORY,
        subject_ref=cand,
        predicate_ref="TYPE",
        object_ref=LiteralValueRef(value="car"),
        polarity=Polarity.AFFIRMATIVE,
        evidence_ids=("ev_dep_low",),
    )
    dec_reject, _ = evaluator.evaluate_claim(
        claim=claim_reject,
        evidence_records=[make_ev(0.05, 0.9, "dep_low")],
    )
    assert dec_reject.outcome == DecisionOutcome.REJECT
    assert dec_reject.transaction_proposal_id is None

    # 3. Intermediate confidence -> DEFER
    claim_defer = GroundingClaim(
        claim_id="cl_defer",
        origin=ClaimOrigin.DIRECT_SENSORY,
        subject_ref=cand,
        predicate_ref="COLOR",
        object_ref=LiteralValueRef(value="cyan"),
        polarity=Polarity.AFFIRMATIVE,
        evidence_ids=("ev_dep_mid",),
    )
    dec_defer, _ = evaluator.evaluate_claim(
        claim=claim_defer,
        evidence_records=[make_ev(0.60, 0.7, "dep_mid")],
    )
    assert dec_defer.outcome == DecisionOutcome.DEFER

    # 4. Severe contradiction with active belief -> REQUEST_EVIDENCE
    active_belief_color = BeliefState(
        belief_id="b_color_red",
        subject_ref=cand,
        predicate_ref="COLOR",
        object_ref=LiteralValueRef(value="red"),
        polarity=Polarity.AFFIRMATIVE,
        posterior_probability=0.92,
        evidence_strength=0.95,
        supporting_evidence_ids=("ev_red",),
    )
    claim_contra = GroundingClaim(
        claim_id="cl_contra",
        origin=ClaimOrigin.DIRECT_SENSORY,
        subject_ref=cand,
        predicate_ref="COLOR",
        object_ref=LiteralValueRef(value="blue"),
        polarity=Polarity.AFFIRMATIVE,
        evidence_ids=("ev_blue",),
    )
    dec_req, _ = evaluator.evaluate_claim(
        claim=claim_contra,
        evidence_records=[make_ev(0.90, 0.9, "dep_blue")],
        active_beliefs=[active_belief_color],
    )
    assert dec_req.outcome == DecisionOutcome.REQUEST_EVIDENCE
    assert len(dec_req.contradictions) > 0
