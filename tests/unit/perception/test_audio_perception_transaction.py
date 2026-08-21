"""Tests for Audio Perception Transaction — Wave A4.

Tests HCIR commitment: speech, events, learning, and full pipeline.
"""

from __future__ import annotations

import pytest

from hbllm.hcir.graph import (
    AcousticConceptNode,
    AudioObservationNode,
    CognitiveGraph,
    HCIREdgeType,
    HCIRNodeType,
)
from hbllm.perception.audio_memory import AudioMemory
from hbllm.perception.audio_perception import AudioPerception
from hbllm.perception.audio_perception_runtime import AudioPerceptionRuntime
from hbllm.perception.audio_perception_transaction import AudioPerceptionTransaction
from hbllm.perception.providers.mock_audio_provider import MockAudioProvider


@pytest.fixture
def provider() -> MockAudioProvider:
    return MockAudioProvider()


@pytest.fixture
def graph() -> CognitiveGraph:
    return CognitiveGraph()


@pytest.fixture
def memory() -> AudioMemory:
    return AudioMemory()


@pytest.fixture
def runtime(provider: MockAudioProvider) -> AudioPerceptionRuntime:
    return AudioPerceptionRuntime(
        speech=provider,
        events=provider,
        scene=provider,
        speaker=provider,
    )


@pytest.fixture
def transaction(
    graph: CognitiveGraph,
    memory: AudioMemory,
) -> AudioPerceptionTransaction:
    return AudioPerceptionTransaction(
        graph=graph,
        memory=memory,
        provider_id="mock-audio-v1",
    )


@pytest.fixture
def perception(
    runtime: AudioPerceptionRuntime,
    transaction: AudioPerceptionTransaction,
) -> AudioPerception:
    return AudioPerception(runtime, transaction)


# ═══════════════════════════════════════════════════════════════════════════
# Commit Speech
# ═══════════════════════════════════════════════════════════════════════════


class TestCommitSpeech:
    """Tests for committing speech evidence to HCIR."""

    @pytest.mark.asyncio
    async def test_commit_creates_node(
        self,
        runtime: AudioPerceptionRuntime,
        transaction: AudioPerceptionTransaction,
        graph: CognitiveGraph,
    ) -> None:
        assessment = await runtime.perceive(b"speech test")
        node = transaction.commit_speech(assessment)
        assert node is not None
        assert isinstance(node, AudioObservationNode)
        assert node.event_type == "speech"
        assert node.transcript != ""

    @pytest.mark.asyncio
    async def test_node_in_graph(
        self,
        runtime: AudioPerceptionRuntime,
        transaction: AudioPerceptionTransaction,
        graph: CognitiveGraph,
    ) -> None:
        assessment = await runtime.perceive(b"in graph")
        node = transaction.commit_speech(assessment)
        assert node is not None
        audio_nodes = list(graph.nodes_by_type(HCIRNodeType.AUDIO_OBSERVATION))
        assert len(audio_nodes) == 1
        assert audio_nodes[0].id == node.id

    @pytest.mark.asyncio
    async def test_no_speech_returns_none(
        self,
        transaction: AudioPerceptionTransaction,
    ) -> None:
        from hbllm.perception.providers.audio_evidence import AudioAssessment

        assessment = AudioAssessment()  # No speech
        node = transaction.commit_speech(assessment)
        assert node is None

    @pytest.mark.asyncio
    async def test_speaker_ref_stored(
        self,
        runtime: AudioPerceptionRuntime,
        transaction: AudioPerceptionTransaction,
    ) -> None:
        assessment = await runtime.perceive(b"speaker test")
        node = transaction.commit_speech(assessment)
        assert node is not None
        # Speaker ref should be set (mock always provides one)
        assert node.speaker_ref != ""


# ═══════════════════════════════════════════════════════════════════════════
# Commit Event
# ═══════════════════════════════════════════════════════════════════════════


class TestCommitEvent:
    """Tests for committing sound events to HCIR."""

    @pytest.mark.asyncio
    async def test_commit_creates_node(
        self,
        runtime: AudioPerceptionRuntime,
        transaction: AudioPerceptionTransaction,
        graph: CognitiveGraph,
    ) -> None:
        assessment = await runtime.perceive(b"event test")
        node = transaction.commit_event(assessment)
        assert node is not None
        assert isinstance(node, AudioObservationNode)
        assert node.event_type != ""

    @pytest.mark.asyncio
    async def test_no_events_returns_none(
        self,
        transaction: AudioPerceptionTransaction,
    ) -> None:
        from hbllm.perception.providers.audio_evidence import AudioAssessment

        assessment = AudioAssessment()  # No events
        node = transaction.commit_event(assessment)
        assert node is None


# ═══════════════════════════════════════════════════════════════════════════
# Commit Learning
# ═══════════════════════════════════════════════════════════════════════════


class TestCommitLearning:
    """Tests for learning — creates cognitive artifact, NOT model training."""

    @pytest.mark.asyncio
    async def test_creates_concept(
        self,
        runtime: AudioPerceptionRuntime,
        transaction: AudioPerceptionTransaction,
        graph: CognitiveGraph,
    ) -> None:
        assessment = await runtime.perceive(b"doorbell sound", label="my_doorbell")
        concept = transaction.commit_learning(assessment)
        assert concept is not None
        assert isinstance(concept, AcousticConceptNode)
        assert concept.label == "my_doorbell"
        assert concept.observation_count == 1

    @pytest.mark.asyncio
    async def test_concept_in_graph(
        self,
        runtime: AudioPerceptionRuntime,
        transaction: AudioPerceptionTransaction,
        graph: CognitiveGraph,
    ) -> None:
        assessment = await runtime.perceive(b"learn", label="alarm")
        transaction.commit_learning(assessment)
        concepts = list(graph.nodes_by_type(HCIRNodeType.ACOUSTIC_CONCEPT))
        assert len(concepts) == 1
        assert concepts[0].label == "alarm"

    @pytest.mark.asyncio
    async def test_supports_edge_created(
        self,
        runtime: AudioPerceptionRuntime,
        transaction: AudioPerceptionTransaction,
        graph: CognitiveGraph,
    ) -> None:
        assessment = await runtime.perceive(b"edge", label="siren")
        transaction.commit_learning(assessment)

        edges = [e for e in graph.all_edges() if e.edge_type == HCIREdgeType.SUPPORTS]
        assert len(edges) >= 1

    @pytest.mark.asyncio
    async def test_update_existing_concept(
        self,
        runtime: AudioPerceptionRuntime,
        transaction: AudioPerceptionTransaction,
    ) -> None:
        a1 = await runtime.perceive(b"sample1", label="kettle")
        c1 = transaction.commit_learning(a1)
        assert c1 is not None
        assert c1.observation_count == 1

        a2 = await runtime.perceive(b"sample2", label="kettle")
        c2 = transaction.commit_learning(a2)
        assert c2 is not None
        assert c2.observation_count == 2
        # Same concept instance
        assert c1.id == c2.id

    @pytest.mark.asyncio
    async def test_no_label_returns_none(
        self,
        runtime: AudioPerceptionRuntime,
        transaction: AudioPerceptionTransaction,
    ) -> None:
        assessment = await runtime.perceive(b"no label")
        concept = transaction.commit_learning(assessment)
        assert concept is None


# ═══════════════════════════════════════════════════════════════════════════
# Full Pipeline (Facade)
# ═══════════════════════════════════════════════════════════════════════════


class TestAudioPerceptionFacade:
    """Tests for AudioPerception facade — end-to-end pipeline."""

    @pytest.mark.asyncio
    async def test_listen_returns_assessment(
        self,
        perception: AudioPerception,
    ) -> None:
        assessment = await perception.listen(b"listen test")
        assert assessment.speech is not None
        assert len(assessment.events) >= 1

    @pytest.mark.asyncio
    async def test_learn_sound(
        self,
        perception: AudioPerception,
        graph: CognitiveGraph,
    ) -> None:
        concept = await perception.learn_sound(b"doorbell", "my_doorbell")
        assert concept is not None
        assert concept.label == "my_doorbell"

        # Should be in HCIR
        concepts = list(graph.nodes_by_type(HCIRNodeType.ACOUSTIC_CONCEPT))
        assert len(concepts) == 1

    @pytest.mark.asyncio
    async def test_recognize_speech(
        self,
        perception: AudioPerception,
        graph: CognitiveGraph,
    ) -> None:
        node = await perception.recognize_speech(b"hello world")
        assert node is not None
        assert node.event_type == "speech"

        # Should be in HCIR
        obs = list(graph.nodes_by_type(HCIRNodeType.AUDIO_OBSERVATION))
        assert len(obs) == 1

    @pytest.mark.asyncio
    async def test_recognize_event(
        self,
        perception: AudioPerception,
        graph: CognitiveGraph,
    ) -> None:
        node = await perception.recognize_event(b"alarm sound")
        assert node is not None
        assert node.event_type != ""

    @pytest.mark.asyncio
    async def test_learn_five_recognize(
        self,
        perception: AudioPerception,
        graph: CognitiveGraph,
    ) -> None:
        """Learn 5 sounds and verify all in HCIR."""
        labels = ["doorbell", "alarm", "dog_bark", "kettle", "phone"]
        for i, label in enumerate(labels):
            await perception.learn_sound(f"sample_{i}".encode(), label)

        concepts = list(graph.nodes_by_type(HCIRNodeType.ACOUSTIC_CONCEPT))
        assert len(concepts) == 5
        concept_labels = {c.label for c in concepts}
        assert concept_labels == set(labels)
