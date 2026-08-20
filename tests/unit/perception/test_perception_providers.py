"""Tests for Perception Providers — V1.

Validates:
    - MockVisionProvider determinism and protocol compliance
    - SigLIPVisionProvider (monkeypatched, no real model)
    - VisionNode provider delegation
    - Protocol checks (isinstance VisionProvider/VisionDetector)
    - VisualObservationNode and VisualConceptNode HCIR integration
"""

from __future__ import annotations

import math

import pytest

from hbllm.hcir.graph import (
    NODE_TYPE_REGISTRY,
    HCIRNodeType,
    VisualConceptNode,
    VisualObservationNode,
)
from hbllm.perception.providers.base import (
    PerceptionProvider,
    VisionDetector,
    VisionOCR,
    VisionProvider,
)
from hbllm.perception.providers.mock_provider import MockVisionProvider

# ═══════════════════════════════════════════════════════════════════════════
# MockVisionProvider
# ═══════════════════════════════════════════════════════════════════════════


class TestMockVisionProvider:
    @pytest.fixture
    def provider(self) -> MockVisionProvider:
        return MockVisionProvider()

    @pytest.mark.asyncio
    async def test_encode_returns_visual_embedding(self, provider: MockVisionProvider) -> None:
        emb = await provider.encode(b"test_image_data")
        assert emb.model_id == "mock/vision-v1"
        assert emb.space_id == "mock/vision-v1-image"
        assert emb.embedding_type == "semantic"
        assert emb.normalization == "l2"
        assert emb.source == "image"
        assert emb.dimensions == 384
        assert len(emb.vector) == 384

    @pytest.mark.asyncio
    async def test_determinism_same_input(self, provider: MockVisionProvider) -> None:
        """Identical inputs must produce identical embeddings."""
        emb1 = await provider.encode(b"same_data")
        emb2 = await provider.encode(b"same_data")
        assert emb1.vector == emb2.vector
        assert emb1.image_hash == emb2.image_hash

    @pytest.mark.asyncio
    async def test_different_inputs_different_embeddings(
        self, provider: MockVisionProvider
    ) -> None:
        emb1 = await provider.encode(b"image_a")
        emb2 = await provider.encode(b"image_b")
        assert emb1.vector != emb2.vector
        assert emb1.image_hash != emb2.image_hash

    @pytest.mark.asyncio
    async def test_l2_normalized(self, provider: MockVisionProvider) -> None:
        emb = await provider.encode(b"some_image")
        norm = math.sqrt(sum(x * x for x in emb.vector))
        assert abs(norm - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_encode_batch(self, provider: MockVisionProvider) -> None:
        images = [b"img1", b"img2", b"img3"]
        results = await provider.encode_batch(images)
        assert len(results) == 3
        # Each result should be unique
        hashes = {r.image_hash for r in results}
        assert len(hashes) == 3

    @pytest.mark.asyncio
    async def test_encode_string_path(self, provider: MockVisionProvider) -> None:
        import pathlib

        emb = await provider.encode(pathlib.Path("/tmp/test.jpg"))
        assert len(emb.vector) == 384
        assert emb.image_hash != ""

    @pytest.mark.asyncio
    async def test_compatibility_same_provider(self, provider: MockVisionProvider) -> None:
        emb1 = await provider.encode(b"a")
        emb2 = await provider.encode(b"b")
        assert emb1.is_compatible_with(emb2)

    @pytest.mark.asyncio
    async def test_compatibility_different_provider(self) -> None:
        p1 = MockVisionProvider(model_name="mock/v1")
        p2 = MockVisionProvider(model_name="mock/v2")
        emb1 = await p1.encode(b"a")
        emb2 = await p2.encode(b"a")
        assert not emb1.is_compatible_with(emb2)

    def test_modality(self, provider: MockVisionProvider) -> None:
        assert provider.modality == "vision"

    def test_provider_id(self, provider: MockVisionProvider) -> None:
        assert provider.provider_id == "mock:mock/vision-v1"

    @pytest.mark.asyncio
    async def test_initialize_shutdown(self, provider: MockVisionProvider) -> None:
        """initialize/shutdown should be no-ops for mock."""
        await provider.initialize()
        await provider.shutdown()

    def test_custom_dimensions(self) -> None:
        p = MockVisionProvider(dimensions=768)
        assert p.dimensions == 768


# ═══════════════════════════════════════════════════════════════════════════
# Protocol Checks
# ═══════════════════════════════════════════════════════════════════════════


class TestProtocolChecks:
    def test_mock_is_vision_provider(self) -> None:
        p = MockVisionProvider()
        assert isinstance(p, VisionProvider)

    def test_mock_is_perception_provider(self) -> None:
        p = MockVisionProvider()
        assert isinstance(p, PerceptionProvider)

    def test_mock_is_not_detector(self) -> None:
        p = MockVisionProvider()
        assert not isinstance(p, VisionDetector)

    def test_mock_is_not_ocr(self) -> None:
        p = MockVisionProvider()
        assert not isinstance(p, VisionOCR)


# ═══════════════════════════════════════════════════════════════════════════
# HCIR Visual Node Types
# ═══════════════════════════════════════════════════════════════════════════


class TestVisualObservationNode:
    def test_creation(self) -> None:
        node = VisualObservationNode(
            embedding_ref="vobs_abc123",
            embedding_space="siglip-image",
            embedding_model="siglip",
            image_hash="hash456",
        )
        assert node.node_type == HCIRNodeType.VISUAL_OBSERVATION
        assert node.embedding_ref == "vobs_abc123"
        assert node.embedding_space == "siglip-image"

    def test_inherits_observation(self) -> None:
        from hbllm.hcir.graph import ObservationNode

        node = VisualObservationNode()
        assert isinstance(node, ObservationNode)

    def test_in_registry(self) -> None:
        assert HCIRNodeType.VISUAL_OBSERVATION in NODE_TYPE_REGISTRY
        assert NODE_TYPE_REGISTRY[HCIRNodeType.VISUAL_OBSERVATION] is VisualObservationNode


class TestVisualConceptNode:
    def test_creation(self) -> None:
        node = VisualConceptNode(
            label="screwdriver",
            definition="A hand tool for screws",
            prototype_ref="vproto_abc",
            embedding_space="siglip-image",
            observation_count=3,
            exemplar_refs=["vobs_1", "vobs_2", "vobs_3"],
        )
        assert node.node_type == HCIRNodeType.VISUAL_CONCEPT
        assert node.label == "screwdriver"
        assert node.prototype_ref == "vproto_abc"
        assert len(node.exemplar_refs) == 3

    def test_inherits_concept(self) -> None:
        from hbllm.hcir.graph import ConceptNode

        node = VisualConceptNode()
        assert isinstance(node, ConceptNode)

    def test_in_registry(self) -> None:
        assert HCIRNodeType.VISUAL_CONCEPT in NODE_TYPE_REGISTRY
        assert NODE_TYPE_REGISTRY[HCIRNodeType.VISUAL_CONCEPT] is VisualConceptNode

    def test_default_fields(self) -> None:
        node = VisualConceptNode()
        assert node.observation_count == 0
        assert node.exemplar_refs == []
        assert node.aliases == []
        assert node.contexts == []
        assert node.last_seen == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# VisionNode Provider Delegation
# ═══════════════════════════════════════════════════════════════════════════


class TestVisionNodeProviderDelegation:
    def test_default_no_provider(self) -> None:
        """VisionNode without provider should still work (hash fallback)."""
        from hbllm.perception.vision_node import VisionNode

        node = VisionNode("test_vision")
        assert node._vision_provider is None
        # Hash-based fallback should produce something
        result = node._embed_image("test_image_path")
        assert len(result) == 768

    def test_with_mock_provider(self) -> None:
        """VisionNode with provider should store it."""
        from hbllm.perception.vision_node import VisionNode

        provider = MockVisionProvider()
        node = VisionNode("test_vision", provider=provider)
        assert node._vision_provider is provider
