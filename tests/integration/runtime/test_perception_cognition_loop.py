"""End-to-End Integration Test: Perception → Normalization → HCIR → Epistemics → Cognition → Action.

Validates the full cognitive runtime loop:
1. Perception: Providers generate raw observations (Audio speech, Visual detection, Sensor IMU).
2. Normalization: EvidenceNormalizer converts them into modality-neutral PerceptualEvidenceNodes.
3. Grounding / Ingestion: PerceptionEpistemicBridge ingests nodes into CognitiveGraph.
4. Epistemics: DiscoveryBeliefManager revises beliefs using PerceptualEvidenceNode via EvidenceIntegrationMixin.
5. Cognitive Budget: CognitiveBudgetEngine evaluates epistemic state & provider capabilities to generate CognitiveDispatchPlan.
6. Cognition: CognitionProvider receives structured CognitionRequest and reasons over HCIR evidence.
7. Action: Cognition emits ActionIntent with safety constraints executed by ActionProvider.
"""

from __future__ import annotations

import time

import pytest

from hbllm.brain.epistemics.belief_manager import DiscoveryBeliefManager
from hbllm.hcir.graph import (
    BeliefNode,
    CognitiveGraph,
    PerceptualEvidenceNode,
)
from hbllm.hcir.proposition import BoundingBox
from hbllm.hcir.types import (
    BeliefConfidence,
    PropositionLikelihood,
    Provenance,
    UncertaintyVector,
)
from hbllm.perception.evidence_normalizer import EvidenceNormalizer
from hbllm.perception.perception_epistemic_bridge import PerceptionEpistemicBridge
from hbllm.perception.providers.audio_evidence import (
    AcousticObservation,
    SpeechEvidence,
)
from hbllm.perception.providers.audio_types import TemporalSpan
from hbllm.perception.providers.evidence import (
    CandidateRanking,
    ConceptCandidate,
    EpistemicEvidenceProfile,
    VisualAssessment,
    VisualEvidence,
)
from hbllm.perception.providers.provider_provenance import ProviderProvenance
from hbllm.perception.providers.types import VisualEmbedding
from hbllm.runtime.cognitive_budget import CognitiveBudgetEngine
from hbllm.runtime.providers.action import ActionIntent, ExecutionResult
from hbllm.runtime.providers.capability import ProviderCapability
from hbllm.runtime.providers.cognition import CognitionRequest, ThoughtResult
from hbllm.runtime.providers.registry import ProviderRegistry


class MockLocalLLMCognition:
    """Mock Local LLM Cognition Provider implementing CognitionProvider protocol."""

    def __init__(self, provider_id: str = "qwen2.5_3b") -> None:
        self.provider_id = provider_id

    @property
    def capability(self) -> ProviderCapability:
        return ProviderCapability(
            provider_id=self.provider_id,
            provider_type="cognition",
            capabilities=["text_reasoning", "planning", "intent_resolution"],
            modalities=["text"],
            latency_profile="low",
            quality_profile="high",
            requires_network=False,
        )

    async def reason(self, request: CognitionRequest) -> ThoughtResult:
        trace = [
            f"Analyzed intent: {request.intent}",
            f"Evaluated {len(request.evidence_refs)} evidence nodes",
            "Synthesized action requirements with safety constraints",
        ]
        return ThoughtResult(
            conclusion="User requested to locate screwdriver and verify position; action planned.",
            confidence=0.92,
            reasoning_trace=trace,
            evidence_produced=[],
            tokens_used=120,
            latency_ms=45.0,
            provider_id=self.provider_id,
        )

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass


class MockTTSActionProvider:
    """Mock TTS Action Provider implementing ActionProvider protocol."""

    def __init__(self, provider_id: str = "piper_tts") -> None:
        self.provider_id = provider_id
        self.executed_actions: list[ActionIntent] = []

    @property
    def capability(self) -> ProviderCapability:
        return ProviderCapability(
            provider_id=self.provider_id,
            provider_type="action",
            capabilities=["speak", "audio_feedback"],
            modalities=["audio"],
            latency_profile="very_low",
            quality_profile="high",
            risk_profile="none",
        )

    async def execute(self, intent: ActionIntent) -> ExecutionResult:
        self.executed_actions.append(intent)
        return ExecutionResult(
            success=True,
            action_type=intent.action_type,
            actual_effect=f"Spoke utterance: {intent.parameters.get('text', '')}",
            duration_ms=150.0,
            provider_id=self.provider_id,
        )

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass


@pytest.mark.asyncio
async def test_full_cognitive_runtime_loop() -> None:
    # 1. Initialize Substrates
    graph = CognitiveGraph()
    bridge = PerceptionEpistemicBridge(graph=graph)
    normalizer = EvidenceNormalizer()
    registry = ProviderRegistry()
    budget_engine = CognitiveBudgetEngine()
    belief_manager = DiscoveryBeliefManager(graph=graph)

    # 2. Register Providers
    llm_provider = MockLocalLLMCognition()
    tts_provider = MockTTSActionProvider()
    registry.register(llm_provider, llm_provider.capability)
    registry.register(tts_provider, tts_provider.capability)

    # 3. Perception Phase (Audio + Vision + Sensor)
    # 3a. Audio Speech: "Where is my screwdriver?"
    speech_obs = SpeechEvidence(
        observation=AcousticObservation(
            observation_id="aobs_user_cmd",
            temporal=TemporalSpan(start_time=time.time(), duration=2.0),
        ),
        transcript="Where is my screwdriver?",
        confidence=0.95,
        provider_provenance=ProviderProvenance(provider="whisper", model="small"),
    )
    speech_node = normalizer.normalize_speech(speech_obs)

    # 3b. Visual Detection: Screwdriver located in toolbox
    vis_obs = VisualAssessment(
        evidence=VisualEvidence(
            embedding=VisualEmbedding(
                vector=[0.1, 0.2],
                model_id="siglip",
                space_id="siglip-base",
                embedding_type="semantic",
                dimensions=2,
            ),
            image_hash="hash_cam_01",
        ),
        candidate_concepts=[
            ConceptCandidate(
                concept_node_id="vc_screwdriver",
                label="screwdriver",
                mean_similarity=0.88,
                best_similarity=0.94,
                matching_observations=5,
            )
        ],
        ranking=CandidateRanking(best_score=0.94, second_score=0.3, margin=0.64),
        epistemic_profile=EpistemicEvidenceProfile(
            perceptual_similarity=0.94,
            evidence_strength=0.9,
            source_reliability=1.0,
        ),
    )
    vis_node = normalizer.normalize_visual(
        vis_obs,
        frame_id="workbench_cam",
        bounding_box=BoundingBox(x1=0.2, y1=0.3, x2=0.4, y2=0.6),
        depth_meters=0.8,
    )

    # 3c. Sensor: Workbench light sensor
    sensor_node = normalizer.normalize_sensor(
        sensor_id="light_sensor_01",
        predicate="illuminance_lux",
        value=450.0,
        value_type="lux",
        confidence=0.99,
    )

    # 4. Grounding / HCIR Ingestion
    ingested_speech_ids = bridge.ingest_perceptual_evidence(speech_node)
    ingested_vis_ids = bridge.ingest_perceptual_evidence(vis_node)
    ingested_sensor_ids = bridge.ingest_perceptual_evidence(sensor_node)

    assert len(ingested_speech_ids) == 1
    assert len(ingested_vis_ids) == 1
    assert len(ingested_sensor_ids) == 1

    # Verify nodes in graph
    retrieved_vis = graph.get_node(vis_node.id)
    assert isinstance(retrieved_vis, PerceptualEvidenceNode)
    assert retrieved_vis.proposition.object_value == "screwdriver"

    # 5. Epistemic Belief Revision
    # Create a prior belief that screwdriver is in the workshop
    initial_belief = BeliefNode(
        id="belief_screwdriver_location",
        claim="The screwdriver is on the workbench",
        uncertainty=UncertaintyVector(confidence=0.5),
        belief_confidence=BeliefConfidence(stability=0.5),
        provenance=Provenance(source_node="initial_prior"),
    )
    graph.upsert_node(initial_belief)

    # Revise belief with the visual PerceptualEvidenceNode
    transition = await belief_manager.revise(
        belief_id=initial_belief.id,
        proposition_likelihood=PropositionLikelihood(
            belief_id=initial_belief.id,
            evidence_id=vis_node.id,
            likelihood_ratio=4.0,
            effective_likelihood_ratio=3.5,
            novelty_discount=0.9,
            status="supported",
        ),
        rationale="Grounded visual identification on workbench",
    )

    assert transition.posterior_confidence > 0.5
    updated_belief = graph.get_node(initial_belief.id)
    assert updated_belief.uncertainty.confidence > 0.5
    assert vis_node.id in updated_belief.evidence_sources

    # 6. Cognitive Budget Planning
    dispatch_plan = budget_engine.plan(
        task_intent="Answer user question about screwdriver location",
        hcir_state=graph,
        registry=registry,
        latency_budget_ms=1000,
    )

    assert len(dispatch_plan.steps) > 0
    assert dispatch_plan.stopping_condition == "sufficient_confidence"

    # 7. Cognition Execution
    cognition_req = CognitionRequest(
        intent="answer_user_query",
        cognitive_state_summary={
            "query": speech_node.proposition.object_value,
            "belief": updated_belief.claim,
            "belief_confidence": updated_belief.uncertainty.confidence,
            "spatial_location": f"{retrieved_vis.spatial.frame_id} (depth {retrieved_vis.spatial.depth_meters}m)",
        },
        evidence_refs=[speech_node.id, vis_node.id],
        goals=["respond_to_user_coherently"],
        constraints=["truthful", "succinct"],
    )

    thought_result = await llm_provider.reason(cognition_req)
    assert thought_result.confidence > 0.8
    assert len(thought_result.reasoning_trace) > 0

    # 8. Action Execution via TTS
    spoken_text = "Your screwdriver is on the workbench, about 0.8 meters in front of the camera."
    action_intent = ActionIntent(
        action_type="speak",
        target="user",
        parameters={"text": spoken_text, "voice": "default"},
        preconditions=["audio_output_available"],
        expected_effect="User informed of screwdriver position",
        safety_constraints=["volume_below_80db"],
        authorization="user",
    )

    exec_result = await tts_provider.execute(action_intent)
    assert exec_result.success is True
    assert len(tts_provider.executed_actions) == 1
    assert tts_provider.executed_actions[0].parameters["text"] == spoken_text
