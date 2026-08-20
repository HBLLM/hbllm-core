"""Tests for Visual Perception Transaction — V2."""

from __future__ import annotations

import pytest

from hbllm.hcir.graph import (
    CognitiveGraph,
    HCIREdgeType,
    HCIRNodeType,
    VisualConceptNode,
)
from hbllm.memory.belief_graph import BeliefGraph
from hbllm.perception.providers.mock_provider import MockVisionProvider
from hbllm.perception.visual_memory import VisualMemory
from hbllm.perception.visual_perception_runtime import VisualPerceptionRuntime
from hbllm.perception.visual_perception_transaction import (
    VisualPerceptionTransaction,
)

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def graph() -> CognitiveGraph:
    return CognitiveGraph()


@pytest.fixture
def memory() -> VisualMemory:
    return VisualMemory()


@pytest.fixture
def belief_graph() -> BeliefGraph:
    return BeliefGraph()


@pytest.fixture
def provider() -> MockVisionProvider:
    return MockVisionProvider()


@pytest.fixture
def runtime(provider: MockVisionProvider, memory: VisualMemory) -> VisualPerceptionRuntime:
    return VisualPerceptionRuntime(provider, memory)


@pytest.fixture
def transaction(
    graph: CognitiveGraph,
    memory: VisualMemory,
    belief_graph: BeliefGraph,
) -> VisualPerceptionTransaction:
    return VisualPerceptionTransaction(
        graph=graph,
        memory=memory,
        belief_graph=belief_graph,
    )


# ═══════════════════════════════════════════════════════════════════════════
# commit_learning Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCommitLearning:
    @pytest.mark.asyncio
    async def test_create_new_concept(
        self,
        runtime: VisualPerceptionRuntime,
        transaction: VisualPerceptionTransaction,
        graph: CognitiveGraph,
        memory: VisualMemory,
    ) -> None:
        """Learning a new label creates concept + observation + edge + belief."""
        assessment = await runtime.perceive_with_label(b"cup_image", "cup")
        concept = await transaction.commit_learning(assessment)

        # Concept created
        assert isinstance(concept, VisualConceptNode)
        assert concept.label == "cup"
        assert concept.observation_count == 1
        assert len(concept.exemplar_refs) == 1

        # HCIR nodes created
        concept_nodes = graph.nodes_by_type(HCIRNodeType.VISUAL_CONCEPT)
        assert len(concept_nodes) == 1

        obs_nodes = graph.nodes_by_type(HCIRNodeType.VISUAL_OBSERVATION)
        assert len(obs_nodes) == 1

        # SUPPORTS edge created
        edges = [e for e in graph.all_edges() if e.edge_type == HCIREdgeType.SUPPORTS]
        assert len(edges) == 1
        assert obs_nodes[0].id in edges[0].sources
        assert concept.id in edges[0].targets

        # Memory updated
        assert memory.observation_count == 1
        assert memory.prototype_count == 1

    @pytest.mark.asyncio
    async def test_update_existing_concept(
        self,
        runtime: VisualPerceptionRuntime,
        transaction: VisualPerceptionTransaction,
        graph: CognitiveGraph,
        memory: VisualMemory,
    ) -> None:
        """Learning same label with same/similar image updates existing concept."""
        # First learning
        a1 = await runtime.perceive_with_label(b"cup_image", "cup")
        concept1 = await transaction.commit_learning(a1)

        # Second learning with same label AND same image
        # (mock provider produces identical embedding → update path)
        a2 = await runtime.perceive_with_label(b"cup_image", "cup")
        concept2 = await transaction.commit_learning(a2)

        # Should be the same concept (by ID)
        assert concept2.id == concept1.id

        # Concept nodes in graph — still just one
        concepts = graph.nodes_by_type(HCIRNodeType.VISUAL_CONCEPT)
        assert len(concepts) == 1

    @pytest.mark.asyncio
    async def test_different_label_similar_creates_new_with_edge(
        self,
        runtime: VisualPerceptionRuntime,
        transaction: VisualPerceptionTransaction,
        graph: CognitiveGraph,
    ) -> None:
        """Learning different label with similar embedding creates new concept + SIMILAR_TO."""
        # Learn "cup"
        a1 = await runtime.perceive_with_label(b"cup_image", "cup")
        c1 = await transaction.commit_learning(a1)

        # Learn "mug" with same image (will be similar) but different label
        a2 = await runtime.perceive_with_label(b"cup_image", "mug")
        c2 = await transaction.commit_learning(a2)

        # Should create two different concepts
        concepts = graph.nodes_by_type(HCIRNodeType.VISUAL_CONCEPT)
        assert len(concepts) == 2

        # SIMILAR_TO edge should exist
        similar_edges = [e for e in graph.all_edges() if e.edge_type == HCIREdgeType.SIMILAR_TO]
        assert len(similar_edges) == 1

    @pytest.mark.asyncio
    async def test_requires_label(
        self,
        runtime: VisualPerceptionRuntime,
        transaction: VisualPerceptionTransaction,
    ) -> None:
        """commit_learning without label should raise."""
        assessment = await runtime.perceive(b"test")
        with pytest.raises(ValueError, match="proposed_label"):
            await transaction.commit_learning(assessment)

    @pytest.mark.asyncio
    async def test_belief_recorded(
        self,
        runtime: VisualPerceptionRuntime,
        transaction: VisualPerceptionTransaction,
        belief_graph: BeliefGraph,
    ) -> None:
        """Learning should record a belief."""
        assessment = await runtime.perceive_with_label(b"cup", "cup")
        concept = await transaction.commit_learning(assessment)

        belief = await belief_graph.get_belief(concept.id)
        assert belief is not None
        assert belief.confidence > 0
        assert "cup" in belief.reason

    @pytest.mark.asyncio
    async def test_provenance_chain(
        self,
        runtime: VisualPerceptionRuntime,
        transaction: VisualPerceptionTransaction,
        graph: CognitiveGraph,
    ) -> None:
        """Concept and observation should have traceable provenance."""
        assessment = await runtime.perceive_with_label(b"cup", "cup")
        concept = await transaction.commit_learning(assessment)

        assert "visual_perception_transaction" in concept.provenance.created_by
        assert concept.provenance.source_type == "observed"

        obs = graph.nodes_by_type(HCIRNodeType.VISUAL_OBSERVATION)[0]
        assert "visual_perception:" in obs.provenance.created_by


# ═══════════════════════════════════════════════════════════════════════════
# commit_recognition Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCommitRecognition:
    @pytest.mark.asyncio
    async def test_novel_when_empty(
        self,
        runtime: VisualPerceptionRuntime,
        transaction: VisualPerceptionTransaction,
    ) -> None:
        """Recognition with empty memory → novel."""
        assessment = await runtime.perceive(b"unknown_thing")
        result = await transaction.commit_recognition(assessment)

        assert not result.matched
        assert result.is_novel
        assert result.observation_node_id is not None

    @pytest.mark.asyncio
    async def test_recognition_after_learning(
        self,
        runtime: VisualPerceptionRuntime,
        transaction: VisualPerceptionTransaction,
    ) -> None:
        """Learn a concept, then recognize the same image."""
        # Learn
        a_learn = await runtime.perceive_with_label(b"cup_image", "cup")
        concept = await transaction.commit_learning(a_learn)

        # Recognize same image
        a_recog = await runtime.perceive(b"cup_image")
        result = await transaction.commit_recognition(a_recog)

        assert result.matched
        assert result.concept_node_id == concept.id
        assert result.label == "cup"


# ═══════════════════════════════════════════════════════════════════════════
# Integration: Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_learn_five_recognize_five(self) -> None:
        """End-to-end: learn 5 objects, recognize each."""
        from hbllm.perception.visual_perception import VisualPerception

        provider = MockVisionProvider()
        memory = VisualMemory()
        graph = CognitiveGraph()
        belief_graph = BeliefGraph()

        runtime = VisualPerceptionRuntime(provider, memory)
        transaction = VisualPerceptionTransaction(
            graph=graph,
            memory=memory,
            belief_graph=belief_graph,
        )
        perception = VisualPerception(runtime, transaction)

        objects = ["cup", "bottle", "phone", "keys", "book"]
        concepts = {}

        # Learn each
        for obj in objects:
            concept = await perception.learn(f"{obj}_image_data".encode(), obj)
            concepts[obj] = concept
            assert concept.label == obj

        # Recognize each
        for obj in objects:
            result = await perception.recognize(f"{obj}_image_data".encode())
            assert result.matched, f"Failed to recognize {obj}"
            assert result.label == obj

        # Graph state
        concept_nodes = graph.nodes_by_type(HCIRNodeType.VISUAL_CONCEPT)
        assert len(concept_nodes) == 5

        obs_nodes = graph.nodes_by_type(HCIRNodeType.VISUAL_OBSERVATION)
        assert len(obs_nodes) >= 5  # At least 5 from learning

        # Beliefs recorded
        for obj in objects:
            belief = await belief_graph.get_belief(concepts[obj].id)
            assert belief is not None
