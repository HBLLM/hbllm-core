"""Unit Tests for ProviderRegistry and ProviderCapability."""

from __future__ import annotations

import pytest

from hbllm.runtime.providers.capability import ProviderCapability
from hbllm.runtime.providers.registry import ProviderRegistry


class MockPerceptionProvider:
    def __init__(self, name: str = "whisper_provider"):
        self.name = name


class MockCognitionProvider:
    def __init__(self, name: str = "qwen_provider"):
        self.name = name


class MockActionProvider:
    def __init__(self, name: str = "tts_provider"):
        self.name = name


@pytest.fixture
def registry() -> ProviderRegistry:
    return ProviderRegistry()


@pytest.fixture
def whisper_cap() -> ProviderCapability:
    return ProviderCapability(
        provider_id="whisper_small",
        provider_type="perception",
        capabilities=["transcribe_speech", "detect_language"],
        modalities=["audio"],
        latency_profile="low",
        quality_profile="medium",
        memory_requirement_mb=500,
        hardware_requirements=["cpu"],
        requires_network=False,
    )


@pytest.fixture
def yolo_cap() -> ProviderCapability:
    return ProviderCapability(
        provider_id="yolo11n",
        provider_type="perception",
        capabilities=["detect_objects", "track_objects"],
        modalities=["visual"],
        latency_profile="very_low",
        quality_profile="medium",
        memory_requirement_mb=250,
        hardware_requirements=["cpu"],
        requires_network=False,
    )


@pytest.fixture
def qwen_cap() -> ProviderCapability:
    return ProviderCapability(
        provider_id="qwen2.5_3b",
        provider_type="cognition",
        capabilities=["text_reasoning", "planning"],
        modalities=["text"],
        latency_profile="medium",
        quality_profile="high",
        memory_requirement_mb=3000,
        hardware_requirements=["gpu"],
        requires_network=False,
    )


@pytest.fixture
def cloud_llm_cap() -> ProviderCapability:
    return ProviderCapability(
        provider_id="cloud_gemini",
        provider_type="cognition",
        capabilities=["text_reasoning", "visual_question_answering", "complex_planning"],
        modalities=["text", "visual"],
        latency_profile="high",
        quality_profile="very_high",
        requires_network=True,
    )


class TestProviderRegistry:
    def test_register_and_lookup_by_id(
        self,
        registry: ProviderRegistry,
        whisper_cap: ProviderCapability,
    ) -> None:
        p = MockPerceptionProvider()
        registry.register(p, whisper_cap)

        assert registry.provider_count == 1
        assert registry.get_provider("whisper_small") is p
        assert registry.get_capability("whisper_small") == whisper_cap

    def test_duplicate_registration_raises(
        self,
        registry: ProviderRegistry,
        whisper_cap: ProviderCapability,
    ) -> None:
        p = MockPerceptionProvider()
        registry.register(p, whisper_cap)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(p, whisper_cap)

    def test_unregister(
        self,
        registry: ProviderRegistry,
        whisper_cap: ProviderCapability,
    ) -> None:
        p = MockPerceptionProvider()
        registry.register(p, whisper_cap)
        assert registry.is_available("transcribe_speech")

        registry.unregister("whisper_small")
        assert registry.provider_count == 0
        assert not registry.is_available("transcribe_speech")
        with pytest.raises(KeyError):
            registry.get_provider("whisper_small")

    def test_find_by_capability(
        self,
        registry: ProviderRegistry,
        whisper_cap: ProviderCapability,
        qwen_cap: ProviderCapability,
        cloud_llm_cap: ProviderCapability,
    ) -> None:
        p_whisper = MockPerceptionProvider()
        p_qwen = MockCognitionProvider()
        p_cloud = MockCognitionProvider("cloud")

        registry.register(p_whisper, whisper_cap)
        registry.register(p_qwen, qwen_cap)
        registry.register(p_cloud, cloud_llm_cap)

        # Look up text reasoning
        reasoners = registry.find_capability("text_reasoning")
        assert len(reasoners) == 2
        assert p_qwen in reasoners
        assert p_cloud in reasoners

        # Look up transcribe
        transcribers = registry.find_capability("transcribe_speech")
        assert transcribers == [p_whisper]

        # Non-existent
        assert registry.find_capability("teleportation") == []

    def test_find_by_modality(
        self,
        registry: ProviderRegistry,
        whisper_cap: ProviderCapability,
        yolo_cap: ProviderCapability,
        cloud_llm_cap: ProviderCapability,
    ) -> None:
        p_whisper = MockPerceptionProvider()
        p_yolo = MockPerceptionProvider("yolo")
        p_cloud = MockCognitionProvider("cloud")

        registry.register(p_whisper, whisper_cap)
        registry.register(p_yolo, yolo_cap)
        registry.register(p_cloud, cloud_llm_cap)

        visual_providers = registry.find_by_modality("visual")
        assert len(visual_providers) == 2
        assert p_yolo in visual_providers
        assert p_cloud in visual_providers

    def test_find_best_for_preferences(
        self,
        registry: ProviderRegistry,
        qwen_cap: ProviderCapability,
        cloud_llm_cap: ProviderCapability,
    ) -> None:
        p_qwen = MockCognitionProvider("local")
        p_cloud = MockCognitionProvider("cloud")

        registry.register(p_qwen, qwen_cap)
        registry.register(p_cloud, cloud_llm_cap)

        # Require local
        best_local = registry.find_best_for("text_reasoning", require_local=True)
        assert best_local is p_qwen

        # Prefer low latency (local is medium, cloud is high)
        best_fast = registry.find_best_for("text_reasoning", prefer_low_latency=True)
        assert best_fast is p_qwen

        # Prefer high quality (cloud is very_high, local is high)
        best_quality = registry.find_best_for("text_reasoning", prefer_high_quality=True)
        assert best_quality is p_cloud
