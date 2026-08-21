"""Unit tests for Perceptual Epistemics integration in HCIR."""

import time

import pytest

from hbllm.brain.epistemics.belief_manager import DiscoveryBeliefManager
from hbllm.brain.epistemics.contradiction_engine import ContradictionEngine
from hbllm.brain.epistemics.likelihood_evaluator import EpistemicLikelihoodEvaluator
from hbllm.brain.epistemics.perceptual_evaluator import PerceptualEvidenceEvaluator
from hbllm.hcir.graph import (
    AudioObservationNode,
    BeliefNode,
    BeliefTransitionNode,
    CognitiveGraph,
    EvidenceNode,
    HCIREdge,
    HCIREdgeType,
    VisualObservationNode,
)
from hbllm.hcir.types import (
    PerceptualContradictionLevel,
    PerceptualEpistemicProfile,
    PropositionLikelihood,
)
from hbllm.perception.perception_epistemic_bridge import PerceptionEpistemicBridge
from hbllm.perception.providers.audio_evidence import (
    AcousticObservation,
    AudioAssessment,
    AudioEpistemicProfile,
    SpeechEvidence,
)
from hbllm.perception.providers.audio_types import TemporalSpan
from hbllm.perception.providers.provider_provenance import ProviderProvenance


@pytest.fixture
def graph() -> CognitiveGraph:
    return CognitiveGraph()


class TestPerceptualEpistemicProfile:
    """Test multidimensional profile and derived reliability."""

    def test_derived_reliability(self) -> None:
        profile = PerceptualEpistemicProfile(
            sensory_clarity=0.9,
            model_confidence=0.8,
            temporal_stability=0.7,
        )
        # 0.3*0.9 + 0.4*0.8 + 0.3*0.7 = 0.27 + 0.32 + 0.21 = 0.80
        assert abs(profile.reliability - 0.80) < 1e-4

    def test_preserves_multidimensional_evidence(self) -> None:
        p1 = PerceptualEpistemicProfile(sensory_clarity=0.9, model_confidence=0.6, temporal_stability=0.9)
        p2 = PerceptualEpistemicProfile(sensory_clarity=0.6, model_confidence=0.9, temporal_stability=0.9)
        # Different underlying dimensions even if composite is close
        assert p1.sensory_clarity != p2.sensory_clarity
        assert p1.model_confidence != p2.model_confidence


class TestPerceptualEvidenceEvaluator:
    """Test general evidence quality evaluation."""

    def test_evaluate_sensory_evidence(self, graph: CognitiveGraph) -> None:
        evaluator = PerceptualEvidenceEvaluator(graph=graph)
        profile = PerceptualEpistemicProfile(
            sensory_clarity=0.85,
            model_confidence=0.90,
            temporal_stability=0.80,
        )
        evidence = EvidenceNode(
            id="evi_001",
            modality="audio",
            strength=0.88,
            epistemic_profile=profile,
            provider_provenance={"provider": "moonshine", "model": "base", "version": "1.0"},
        )
        graph.upsert_node(evidence)

        assessment = evaluator.evaluate(evidence)
        assert assessment.evidence_id == "evi_001"
        assert 0.75 <= assessment.reliability <= 0.99
        assert assessment.information_gain > 0.0
        assert assessment.uncertainty.confidence == assessment.reliability


class TestEpistemicLikelihoodEvaluator:
    """Test proposition-specific likelihood and likelihood ratio."""

    def test_supporting_evidence_likelihood(self, graph: CognitiveGraph) -> None:
        evaluator = EpistemicLikelihoodEvaluator(graph=graph)
        perceptual_eval = PerceptualEvidenceEvaluator(graph=graph)

        belief = BeliefNode(id="bel_001", claim="There is a dog barking outside")
        belief.uncertainty.confidence = 0.5
        graph.upsert_node(belief)

        evidence = EvidenceNode(
            id="evi_dog",
            modality="audio",
            strength=0.85,
            candidates=[{"label": "dog barking", "score": 0.88}],
            provider_provenance={"provider": "yamnet"},
        )
        graph.upsert_node(evidence)

        assessment = perceptual_eval.evaluate(evidence)
        prop_lik = evaluator.evaluate_likelihood(belief, evidence, assessment, direction="supporting")

        assert prop_lik.p_e_given_h > 0.8
        assert prop_lik.p_e_given_not_h < 0.2
        assert prop_lik.likelihood_ratio > 4.0
        assert prop_lik.status == "informative"

    def test_insufficient_evidence_status(self, graph: CognitiveGraph) -> None:
        evaluator = EpistemicLikelihoodEvaluator(graph=graph)
        perceptual_eval = PerceptualEvidenceEvaluator(graph=graph)

        belief = BeliefNode(id="bel_002", claim="Someone is typing")
        belief.uncertainty.confidence = 0.5
        graph.upsert_node(belief)

        low_qual_evidence = EvidenceNode(
            id="evi_faint",
            modality="audio",
            strength=0.2,
            epistemic_profile=PerceptualEpistemicProfile(
                sensory_clarity=0.1, model_confidence=0.2, temporal_stability=0.2
            ),
            provider_provenance={"provider": "mock"},
        )
        graph.upsert_node(low_qual_evidence)

        assessment = perceptual_eval.evaluate(low_qual_evidence)
        prop_lik = evaluator.evaluate_likelihood(belief, low_qual_evidence, assessment)

        assert prop_lik.status == "insufficient"


class TestDiscoveryBeliefManagerOddsSpace:
    """Test odds-space Bayesian revision and BeliefTransitionNode emission."""

    @pytest.mark.asyncio
    async def test_odds_space_bayesian_revision(self, graph: CognitiveGraph) -> None:
        mgr = DiscoveryBeliefManager(graph=graph)

        belief = BeliefNode(id="bel_room", claim="The conference room is occupied")
        belief.uncertainty.confidence = 0.5  # Prior odds = 1.0
        graph.upsert_node(belief)

        evidence = EvidenceNode(id="evi_speech", modality="audio", strength=0.9)
        graph.upsert_node(evidence)

        # LR = 4.0 -> Posterior odds = 4.0 -> Posterior prob = 4/5 = 0.80
        prop_lik = PropositionLikelihood(
            belief_id="bel_room",
            evidence_id="evi_speech",
            p_e_given_h=0.80,
            p_e_given_not_h=0.20,
            likelihood_ratio=4.0,
            status="informative",
        )

        transition = await mgr.revise(belief.id, prop_lik)

        assert transition.transition_id != ""
        assert transition.prior_confidence == 0.5
        assert abs(transition.posterior_confidence - 0.80) < 0.05
        assert transition.prior_revision == 1
        assert transition.posterior_revision == 2

        # Check BeliefNode updated
        updated_belief = graph.get_node("bel_room")
        assert isinstance(updated_belief, BeliefNode)
        assert abs(updated_belief.uncertainty.confidence - 0.80) < 0.05
        assert updated_belief.current_revision == 2
        assert updated_belief.latest_transition_id == transition.transition_id

        # Check event-sourced BeliefTransitionNode in graph
        trans_node = graph.get_node(transition.transition_id)
        assert isinstance(trans_node, BeliefTransitionNode)
        assert trans_node.belief_id == "bel_room"
        assert trans_node.likelihood_ratio == 4.0


class TestThreeLevelContradictions:
    """Test detection of Level 1, 2, and 3 contradictions."""

    @pytest.mark.asyncio
    async def test_level_1_classifier_disagreement(self, graph: CognitiveGraph) -> None:
        engine = ContradictionEngine(graph=graph)

        # Evidence with ambiguous competing candidates
        evidence = EvidenceNode(
            id="evi_ambig",
            modality="audio",
            strength=0.8,
            candidates=[
                {"label": "doorbell", "score": 0.82},
                {"label": "door_knock", "score": 0.78},
            ],
        )
        graph.upsert_node(evidence)

        reports = await engine.scan_for_perceptual_contradictions()
        l1_reports = [
            r for r in reports
            if r.contradiction_level == PerceptualContradictionLevel.LEVEL_1_CLASSIFIER_DISAGREEMENT
        ]
        assert len(l1_reports) >= 1
        assert "doorbell" in l1_reports[0].claim_a_id
        assert "door_knock" in l1_reports[0].claim_b_id

    @pytest.mark.asyncio
    async def test_level_2_cross_modal_conflict(self, graph: CognitiveGraph) -> None:
        engine = ContradictionEngine(graph=graph)

        # Correlated visual empty room vs audio applause
        vis = VisualObservationNode(id="vis_empty", caption="empty dark conference room")
        aud = AudioObservationNode(id="aud_applause", event_type="applause", transcript="")
        graph.upsert_node(vis)
        graph.upsert_node(aud)

        graph.add_edge(
            HCIREdge(
                edge_type=HCIREdgeType.CORRELATES_WITH,
                sources=[vis.id],
                targets=[aud.id],
                properties={"confidence": 0.88, "delta_time_ms": 50.0},
            )
        )

        reports = await engine.scan_for_perceptual_contradictions()
        l2_reports = [
            r for r in reports
            if r.contradiction_level == PerceptualContradictionLevel.LEVEL_2_CROSS_MODAL_CONFLICT
        ]
        assert len(l2_reports) >= 1
        assert l2_reports[0].claim_a_id == vis.id
        assert l2_reports[0].claim_b_id == aud.id

    @pytest.mark.asyncio
    async def test_level_3_belief_conflict(self, graph: CognitiveGraph) -> None:
        engine = ContradictionEngine(graph=graph)

        belief = BeliefNode(id="bel_locked", claim="Front entrance is locked")
        belief.uncertainty.confidence = 0.90
        graph.upsert_node(belief)

        evidence = EvidenceNode(id="evi_door_open", modality="audio", strength=0.85)
        graph.upsert_node(evidence)

        graph.add_edge(
            HCIREdge(
                edge_type=HCIREdgeType.WEAKENS,
                sources=[evidence.id],
                targets=[belief.id],
            )
        )

        reports = await engine.scan_for_perceptual_contradictions()
        l3_reports = [
            r for r in reports
            if r.contradiction_level == PerceptualContradictionLevel.LEVEL_3_BELIEF_CONFLICT
        ]
        assert len(l3_reports) >= 1
        assert l3_reports[0].claim_a_id == belief.id


class TestPerceptionEpistemicBridge:
    """Test pure structural bridge ingestion without cognitive pollution."""

    def test_bridge_ingests_audio_assessment(self, graph: CognitiveGraph) -> None:
        bridge = PerceptionEpistemicBridge(graph=graph)

        now = time.time()
        obs = AcousticObservation(
            observation_id="aud_123",
            temporal=TemporalSpan(start_time=now, end_time=now + 2.0, duration=2.0),
            provenance=ProviderProvenance(provider="moonshine", model="base", version="1.0", device="cpu"),
        )
        assessment = AudioAssessment(
            observation=obs,
            speech=SpeechEvidence(
                transcript="Antigravity online",
                confidence=0.94,
                provider_provenance=ProviderProvenance(provider="moonshine", model="base", version="1.0", device="cpu"),
            ),
            epistemic_profile=AudioEpistemicProfile(
                perceptual_confidence=0.92,
                classification_confidence=0.94,
                temporal_confidence=0.90,
            ),
        )

        committed = bridge.ingest_audio_assessment(assessment)
        assert "aud_123" in committed
        aud_node = graph.get_node("aud_123")
        assert isinstance(aud_node, AudioObservationNode)
        assert aud_node.transcript == "Antigravity online"

        # Check evidence node created and linked
        edges = graph.edges_from("aud_123")
        assert any(e.edge_type == HCIREdgeType.DERIVED_FROM for e in edges)

    def test_bridge_cross_modal_correlation(self, graph: CognitiveGraph) -> None:
        bridge = PerceptionEpistemicBridge(graph=graph)
        now = time.time()

        vis = VisualObservationNode(
            id="vis_person",
            caption="person standing by the door",
            temporal_span={"start_time": now, "end_time": now + 1.0, "duration": 1.0},
        )
        aud = AudioObservationNode(
            id="aud_voice",
            label="speech",
            start_time=now + 0.1,
            end_time=now + 1.1,
            duration=1.0,
            temporal_span={"start_time": now + 0.1, "end_time": now + 1.1, "duration": 1.0},
        )
        graph.upsert_node(vis)
        graph.upsert_node(aud)

        candidates = bridge.correlate_and_commit(window_seconds=5.0)
        assert len(candidates) >= 1
        cand = candidates[0]
        assert cand.source_obs_id == "vis_person"
        assert cand.target_obs_id == "aud_voice"
        assert cand.confidence > 0.5

        # Check CORRELATES_WITH edge in graph
        corr_edges = [
            e for e in graph.all_edges()
            if e.edge_type == HCIREdgeType.CORRELATES_WITH
        ]
        assert len(corr_edges) == 1
