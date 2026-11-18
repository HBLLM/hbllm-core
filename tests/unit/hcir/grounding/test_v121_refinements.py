from __future__ import annotations

from hbllm.hcir.grounding.actions.gateway import (
    CommunicationValidationGateway,
)
from hbllm.hcir.grounding.causal.schema import (
    CausalSchema,
    InterventionExperimentProposal,
)
from hbllm.hcir.grounding.committer.committer import HCIRCommitter
from hbllm.hcir.grounding.concepts.lifecycle import ConceptPromotionProposal
from hbllm.hcir.grounding.concepts.model import GroundedConcept
from hbllm.hcir.grounding.contracts.actions import (
    RenderedActionPayload,
    SymbolicActionIntent,
)
from hbllm.hcir.grounding.contracts.claims import GroundingClaim
from hbllm.hcir.grounding.contracts.enums import (
    CausalStatus,
    ClaimOrigin,
    ConceptLifecycle,
    EvidenceRelation,
    ModalitySource,
    Polarity,
    TrafficClass,
)
from hbllm.hcir.grounding.contracts.evidence import EvidenceRecord
from hbllm.hcir.grounding.contracts.observation import ObservationEnvelope
from hbllm.hcir.grounding.contracts.quality import (
    GroundingQuality,
)
from hbllm.hcir.grounding.contracts.references import (
    EntityCandidateRef,
    GroundedConceptRef,
    GroundedEntityRef,
    LatentRelationRef,
    PredicateRef,
)
from hbllm.hcir.grounding.epistemics.calibration import EvidenceCalibrator
from hbllm.hcir.grounding.epistemics.fusion import DependencyGroupEvidenceFusion
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.transactions import TransactionStatus
from hbllm.hcir.workspace import HCIRWorkspaceState


class TestSGEv121Refinements:
    """Verifies all seven technical refinements of SGE Architecture Specification v1.2.1."""

    def test_2_1_formal_observation_envelope_definition(self) -> None:
        """Refinement 2.1: ObservationEnvelope distinguishes payload_ref, payload_hash, and clock_domain."""
        env = ObservationEnvelope.create(
            observation_id="obs_cam1_001",
            source_modality=ModalitySource.VISION,
            source_channel_id="cam_front_left",
            traffic_class=TrafficClass.CLASS_HIGH_FREQ,
            sequence_number=42,
            payload_ref="blob://frames/20261106_1001.raw",
            clock_domain="ptp_camera_sync",
            metadata={"exposure_ms": 12.5, "resolution": (1920, 1080)},
        )

        assert env.observation_id == "obs_cam1_001"
        assert env.payload_ref == "blob://frames/20261106_1001.raw"
        assert env.raw_payload_ref == env.payload_ref
        assert len(env.payload_hash) == 64  # Deterministic SHA-256
        assert env.provenance_hash == env.payload_hash
        assert env.clock_domain == "ptp_camera_sync"
        assert env.metadata["exposure_ms"] == 12.5

    def test_2_2_and_2_3_likelihood_ratio_and_evidence_polarity(self) -> None:
        """Refinements 2.2 & 2.3: Platt calibrated posteriors, likelihood ratios, and evidence polarity."""
        calibrator = EvidenceCalibrator()

        # Evidence A: Vision detector supports hypothesis with score 0.85
        ev_supp = EvidenceRecord(
            evidence_id="ev_supp_1",
            observation_id="obs_1",
            source_channel_id="cam1",
            modality=ModalitySource.VISION,
            model_name="default",
            model_version="1.0",
            dependency_group="dep_img_1",
            perceptual_score=0.85,
            evidence_strength=0.9,
            relation=EvidenceRelation.SUPPORTS,
        )
        cal_supp = calibrator.calibrate_record(ev_supp)
        assert cal_supp.calibrated_posterior is not None
        assert cal_supp.likelihood_ratio is not None
        assert cal_supp.likelihood_ratio > 1.0  # Supports raises odds

        # Evidence B: Acoustic sensor refutes hypothesis with high confidence
        ev_ref = EvidenceRecord(
            evidence_id="ev_ref_1",
            observation_id="obs_2",
            source_channel_id="mic1",
            modality=ModalitySource.AUDIO,
            model_name="conservative_audio",
            model_version="1.0",
            dependency_group="dep_mic_1",
            perceptual_score=0.90,
            evidence_strength=0.95,
            relation=EvidenceRelation.REFUTES,
        )
        cal_ref = calibrator.calibrate_record(ev_ref)
        assert cal_ref.likelihood_ratio is not None
        assert cal_ref.likelihood_ratio < 1.0  # Refutes inverts and lowers odds

        # Fusion test: Evidence that refutes drops the posterior below prior (0.5)
        fusion = DependencyGroupEvidenceFusion(default_prior=0.5)
        res_refuted = fusion.fuse([cal_ref], prior_probability=0.5)
        assert res_refuted.posterior_probability < 0.20

    def test_2_4_concept_lifecycle_grounded_vs_canonical(self) -> None:
        """Refinement 2.4: Distinguishes GROUNDED in working store from CANONICAL in CognitiveGraph."""
        ws = HCIRWorkspaceState()
        tx_mgr = TransactionManager(ws)
        committer = HCIRCommitter(tx_mgr)

        concept = GroundedConcept(
            concept_ref=GroundedConceptRef("C-0099"),
            lifecycle_state=ConceptLifecycle.CONSOLIDATED,
            exemplar_entity_ids={"e_obj_1", "e_obj_2"},
            affordances={"GRASP", "LIFT"},
        )
        assert concept.lifecycle_state == ConceptLifecycle.CONSOLIDATED

        proposal = ConceptPromotionProposal(
            proposal_id="prop_c0099",
            concept_ref=concept.concept_ref,
            concept_version=concept.concept_version,
            quality_summary=concept.grounding_quality,
            exemplar_count=len(concept.exemplar_entity_ids),
            affordances=tuple(concept.affordances),
            linguistic_aliases={"mug": 0.95},
            rationale="High empirical stability across physical trials",
        )

        tx = committer.commit_concept_promotion(proposal, concept)
        assert tx.status == TransactionStatus.COMMITTED
        # Successfully promoted to CANONICAL state in CognitiveGraph
        assert concept.lifecycle_state == ConceptLifecycle.CANONICAL
        assert ws.graph.has_node("C-0099")

    def test_2_5_quality_measurement_untested_affordance_handling(self) -> None:
        """Refinement 2.5: Untested affordances retain UNTESTED status rather than false negative score 0."""
        # Visual concept with untested physical interaction
        q = GroundingQuality.with_untested(
            perceptual_stability=0.92,
            temporal_persistence=0.88,
            relational_consistency=0.75,
            cross_modal_support=0.5,
            predictive_utility=0.35,  # Delta over base rate
            interaction_support=0.0,
            boundary_stability=0.85,
            untested_dimensions={"interaction_support"},
        )

        meas_interaction = q.get_measurement("interaction_support")
        assert meas_interaction.measurement_status == "UNTESTED"
        assert meas_interaction.value is None
        assert meas_interaction.is_tested is False

        meas_perc = q.get_measurement("perceptual_stability")
        assert meas_perc.measurement_status == "MEASURED"
        assert meas_perc.value == 0.92
        assert meas_perc.is_tested is True

    def test_2_6_causal_status_and_intervention_gateway_gating(self) -> None:
        """Refinement 2.6: Expanded CausalStatus and intervention experiments as gated actions."""
        # 1. Observational Identification without physical intervention
        schema = CausalSchema(
            schema_id="schema_light_switch",
            action_type="FLIP_SWITCH",
            target_role="switch",
            causal_status=CausalStatus.CAUSAL_EFFECT_IDENTIFIED,
        )
        assert schema.is_actionable_cause is True

        # 2. Intervention Experiment Proposal synthesized as first-class action intent
        proposal = InterventionExperimentProposal(
            proposal_id="exp_switch_01",
            schema_id="schema_light_switch",
            target_entity_ref=GroundedEntityRef("e_switch_1"),
            action_intent="TOGGLE",
            confounder_controls={"room_illuminated": True},
            expected_outcome={"light_on": True},
        )
        symbolic_intent = proposal.to_symbolic_intent()
        assert isinstance(symbolic_intent, SymbolicActionIntent)
        assert symbolic_intent.action_type == "TOGGLE"
        assert symbolic_intent.target_ref == GroundedEntityRef("e_switch_1")
        assert symbolic_intent.context["is_intervention_experiment"] is True

    def test_typed_predicate_and_latent_relation_refs(self) -> None:
        """Refinement B: PredicateRef and LatentRelationRef replace raw unconstrained strings."""
        cand = EntityCandidateRef("ecand_box")
        pred = PredicateRef("CONTAINS")
        latent = LatentRelationRef("rel_sync_chase_1")

        claim_known = GroundingClaim(
            claim_id="cl_pred_1",
            origin=ClaimOrigin.DIRECT_SENSORY,
            subject_ref=cand,
            predicate_ref=pred,
            object_ref=EntityCandidateRef("ecand_ball"),
            polarity=Polarity.AFFIRMATIVE,
            evidence_ids=("ev_1",),
        )
        assert claim_known.predicate_ref == pred

        claim_dynamic = GroundingClaim(
            claim_id="cl_pred_2",
            origin=ClaimOrigin.INFERRED_BY_REASONING,
            subject_ref=cand,
            predicate_ref=latent,
            object_ref=EntityCandidateRef("ecand_agent"),
            polarity=Polarity.AFFIRMATIVE,
            evidence_ids=("ev_2",),
        )
        assert claim_dynamic.predicate_ref == latent

    def test_communication_validation_gateway(self) -> None:
        """Refinement C: Communication validation separated from physical actuator gateway."""
        comm_gateway = CommunicationValidationGateway(
            forbidden_phrases={"[CONFIDENTIAL_LEAK]"}, max_utterance_length=50
        )

        intent = SymbolicActionIntent(
            intent_id="intent_speak",
            action_type="SAY",
            target_ref=GroundedEntityRef("e_user"),
        )
        payload_ok = RenderedActionPayload(
            payload_id="p_ok",
            intent_id="intent_speak",
            renderer_name="DialogueRenderer",
            execution_channel="speech_tts",
            payload_data={"utterance_text": "The keys are beside the sofa."},
        )
        payload_leak = RenderedActionPayload(
            payload_id="p_bad",
            intent_id="intent_speak",
            renderer_name="DialogueRenderer",
            execution_channel="speech_tts",
            payload_data={"utterance_text": "Here is the [CONFIDENTIAL_LEAK] token."},
        )

        res_ok = comm_gateway.validate_communication(intent, payload_ok)
        assert res_ok.is_authorized is True
        assert res_ok.filtered_text == "The keys are beside the sofa."

        res_bad = comm_gateway.validate_communication(intent, payload_leak)
        assert res_bad.is_authorized is False
        assert len(res_bad.rejection_reasons) >= 1
