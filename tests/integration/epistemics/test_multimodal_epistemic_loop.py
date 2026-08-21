"""End-to-End Multimodal (Audio + Vision) Epistemic Loop Integration Tests.

Validates the full perception-to-epistemics cycle:
1. Multimodal Ingestion: AudioAssessment + VisualAssessment -> HCIR via PerceptionEpistemicBridge.
2. Cross-Modal Correlation: Neutral CORRELATES_WITH hyperedge committed with geometric properties.
3. 3-Tier Contradiction Detection: Classifier, cross-modal, and belief contradictions.
4. Epistemic Loop Execution: Curiosity scans anomalies -> Hypotheses generated -> Bayesian BeliefTransitions executed in odds space.
5. Invariant Verification: Perception only creates evidence; Epistemics owns beliefs.
"""

from __future__ import annotations

import time

import pytest

from hbllm.brain.epistemics.epistemic_loop import EpistemicLoop
from hbllm.hcir.graph import (
    AudioObservationNode,
    BeliefNode,
    BeliefTransitionNode,
    CognitiveGraph,
    ContradictionNode,
    HCIREdgeType,
    HypothesisNode,
    VisualObservationNode,
)
from hbllm.hcir.types import (
    PerceptualContradictionLevel,
)
from hbllm.perception.perception_epistemic_bridge import PerceptionEpistemicBridge
from hbllm.perception.providers.audio_evidence import (
    AcousticObservation,
    AudioAssessment,
    AudioEpistemicProfile,
    SoundEventEvidence,
    SpeechEvidence,
)
from hbllm.perception.providers.audio_types import TemporalSpan
from hbllm.perception.providers.evidence import (
    ConceptCandidate,
    EpistemicEvidenceProfile,
    VisualAssessment,
    VisualEvidence,
)
from hbllm.perception.providers.provider_provenance import ProviderProvenance
from hbllm.perception.providers.types import VisualEmbedding


class MockLLM:
    """Mock LLM for deterministic hypothesis generation in epistemic loop."""

    async def generate(self, prompt: str, **kwargs: object) -> str:
        if "contradiction" in prompt.lower():
            return (
                "CLAIM: The sound source is located in an adjacent room | PLAUSIBILITY: 0.85 | REASONING: Acoustic bleeding through walls\n"
                "CLAIM: The visual detector suffered a temporary occlusion | PLAUSIBILITY: 0.75 | REASONING: Optical pathway blocked"
            )
        elif "unknown" in prompt.lower() or "gap" in prompt.lower():
            return (
                "CLAIM: Ambient acoustic reflections caused ghost sound events | PLAUSIBILITY: 0.80 | REASONING: Reverberation in hall\n"
                "CLAIM: Sensor synchronization delay created false pairing | PLAUSIBILITY: 0.70 | REASONING: Clock drift"
            )
        return "CLAIM: Candidate explanatory hypothesis for sensory observation | PLAUSIBILITY: 0.80 | REASONING: General inference"


@pytest.fixture
def graph() -> CognitiveGraph:
    return CognitiveGraph()


@pytest.fixture
def llm() -> MockLLM:
    return MockLLM()


class TestMultimodalEpistemicLoop:
    """Full End-to-End integration suite."""

    @pytest.mark.asyncio
    async def test_full_multimodal_epistemic_cycle(
        self,
        graph: CognitiveGraph,
        llm: MockLLM,
    ) -> None:
        """Step-by-step verification of audio+vision ingestion, correlation,

        contradiction discovery, curiosity triggering, and belief revision.
        """
        bridge = PerceptionEpistemicBridge(graph=graph)
        loop = EpistemicLoop(graph=graph, llm=llm, max_investigations_per_cycle=3)

        # ── Step 1: Initialize Prior Belief ─────────────────────────────
        prior_belief = BeliefNode(
            id="bel_room_state",
            claim="Conference room is empty and quiet",
        )
        prior_belief.uncertainty.confidence = 0.85
        graph.upsert_node(prior_belief)

        # ── Step 2: Ingest Multimodal Perception ────────────────────────
        now = time.time()

        # Visual Observation: Camera detects movement/person
        vis_assessment = VisualAssessment(
            evidence=VisualEvidence(
                embedding=VisualEmbedding(
                    vector=[0.1] * 128,
                    model_id="siglip",
                    space_id="siglip_v1",
                    embedding_type="semantic",
                    dimensions=128,
                ),
            ),
            candidate_concepts=[
                ConceptCandidate(
                    concept_node_id="concept_person_01",
                    label="person standing near podium with laptop",
                    mean_similarity=0.88,
                    best_similarity=0.92,
                    matching_observations=5,
                ),
            ],
            epistemic_profile=EpistemicEvidenceProfile(
                source_reliability=0.90,
                perceptual_similarity=0.88,
                evidence_strength=0.85,
            ),
        )
        vis_nodes = bridge.ingest_visual_assessment(vis_assessment)
        assert len(vis_nodes) >= 2  # VisualObservationNode + EvidenceNode

        # Audio Observation: Microphone detects speech
        aud_obs = AcousticObservation(
            observation_id="aud_utterance_01",
            temporal=TemporalSpan(start_time=now, end_time=now + 2.5, duration=2.5),
            provenance=ProviderProvenance(
                provider="moonshine", model="base", version="1.0", device="cpu"
            ),
        )
        aud_assessment = AudioAssessment(
            observation=aud_obs,
            speech=SpeechEvidence(
                transcript="Let us begin the system review",
                confidence=0.92,
                provider_provenance=ProviderProvenance(
                    provider="moonshine", model="base", version="1.0", device="cpu"
                ),
            ),
            events=[
                SoundEventEvidence(
                    event_type="speech",
                    confidence=0.95,
                    provider_provenance=ProviderProvenance(
                        provider="yamnet", model="v1", version="1.0", device="cpu"
                    ),
                )
            ],
            epistemic_profile=AudioEpistemicProfile(
                perceptual_confidence=0.90,
                classification_confidence=0.95,
                temporal_confidence=0.92,
            ),
        )
        aud_nodes = bridge.ingest_audio_assessment(aud_assessment)
        assert "aud_utterance_01" in aud_nodes

        # ── Step 3: Run Cross-Modal Correlation ─────────────────────────
        correlations = bridge.correlate_and_commit(window_seconds=5.0)
        assert len(correlations) >= 1
        cand = correlations[0]
        assert cand.source_modality == "visual"
        assert cand.target_modality == "audio"

        # Verify CORRELATES_WITH edge in HCIR
        corr_edges = [e for e in graph.all_edges() if e.edge_type == HCIREdgeType.CORRELATES_WITH]
        assert len(corr_edges) >= 1

        # ── Step 4: Run Epistemic Loop ──────────────────────────────────
        results = await loop.run_cycle()
        assert results is not None
        assert len(results) >= 1

        # ── Step 5: Verify Odds-Space Belief Revision ───────────────────
        updated_belief = graph.get_node("bel_room_state")
        assert isinstance(updated_belief, BeliefNode)
        # Because speech evidence contradicted "empty and quiet", confidence dropped
        assert updated_belief.uncertainty.confidence < 0.85
        assert updated_belief.current_revision >= 2
        assert updated_belief.latest_transition_id != ""

        # Verify event-sourced BeliefTransitionNode exists
        transition_node = graph.get_node(updated_belief.latest_transition_id)
        assert isinstance(transition_node, BeliefTransitionNode)
        assert transition_node.belief_id == "bel_room_state"
        assert transition_node.prior_confidence > transition_node.posterior_confidence
        assert transition_node.posterior_confidence == updated_belief.uncertainty.confidence
        assert transition_node.delta < 0.0
        assert updated_belief.uncertainty.confidence < 0.20

    @pytest.mark.asyncio
    async def test_cross_modal_contradiction_investigation(
        self,
        graph: CognitiveGraph,
        llm: MockLLM,
    ) -> None:
        """Verify that a Level 2 cross-modal contradiction produces a

        ContradictionNode, triggers Curiosity, and generates explanatory Hypotheses.
        """
        bridge = PerceptionEpistemicBridge(graph=graph)
        loop = EpistemicLoop(graph=graph, llm=llm, max_investigations_per_cycle=3)

        now = time.time()

        # 1. Ingest conflicting visual and audio observations
        vis = VisualObservationNode(
            id="vis_empty_hall",
            caption="empty dark lecture hall with no person present",
            temporal_span={"start_time": now, "end_time": now + 1.0, "duration": 1.0},
        )
        aud = AudioObservationNode(
            id="aud_screaming_crowd",
            label="applause",
            event_type="applause",
            transcript="Encore! Bravo!",
            start_time=now + 0.05,
            end_time=now + 1.05,
            duration=1.0,
            temporal_span={"start_time": now + 0.05, "end_time": now + 1.05, "duration": 1.0},
        )
        graph.upsert_node(vis)
        graph.upsert_node(aud)

        # 2. Correlate them
        bridge.correlate_and_commit(window_seconds=2.0)

        # 3. Run Epistemic Loop
        results = await loop.run_cycle()
        assert results is not None

        # 4. Verify Level 2 Contradiction Node created
        contradictions = [
            n
            for _n in graph.all_nodes()
            if isinstance((n := graph.get_node(_n.id)), ContradictionNode)
        ]
        assert len(contradictions) >= 1
        l2_contra = contradictions[0]
        assert (
            l2_contra.contradiction_level
            == PerceptualContradictionLevel.LEVEL_2_CROSS_MODAL_CONFLICT
        )

        # 5. Verify Hypotheses were generated to explain the sensory mismatch
        hypotheses = [
            n
            for _n in graph.all_nodes()
            if isinstance((n := graph.get_node(_n.id)), HypothesisNode)
        ]
        assert len(hypotheses) >= 1
