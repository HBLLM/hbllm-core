"""Unit Tests for Concrete Provider Adapters."""

from __future__ import annotations

import pytest

from hbllm.perception.providers.audio_types import SpeechResult, TemporalSpan
from hbllm.perception.providers.types import VisualEmbedding
from hbllm.runtime.adapters.action.system_adapter import SystemActionAdapter
from hbllm.runtime.adapters.action.tts_adapter import TTSActionAdapter
from hbllm.runtime.adapters.cognition.llm_adapter import LLMCognitionAdapter
from hbllm.runtime.adapters.cognition.symbolic_adapter import SymbolicCognitionAdapter
from hbllm.runtime.adapters.perception.audio_event_adapter import AudioEventPerceptionAdapter
from hbllm.runtime.adapters.perception.sensor_adapter import TelemetrySensorAdapter
from hbllm.runtime.adapters.perception.speech_adapter import SpeechPerceptionAdapter
from hbllm.runtime.adapters.perception.vision_adapter import VisionPerceptionAdapter
from hbllm.runtime.providers.action import ActionIntent
from hbllm.runtime.providers.cognition import CognitionRequest
from hbllm.serving.provider import LLMProvider, LLMResponse


class StubSiglipProvider:
    def __init__(self) -> None:
        self._model = "loaded"
        self._processor = "loaded"

    async def encode(self, image_data: bytes) -> VisualEmbedding:
        return VisualEmbedding(
            vector=[0.1, 0.2, 0.3],
            model_id="google/siglip-base-patch16-224",
            space_id="siglip-base",
            embedding_type="semantic",
            dimensions=3,
        )


class StubMoonshineProvider:
    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def transcribe(self, audio_data: bytes) -> SpeechResult:
        return SpeechResult(
            transcript="Turn on the desk lamp",
            language="en",
            confidence=0.96,
            temporal=TemporalSpan(start_time=100.0, end_time=102.0, duration=2.0),
        )


class StubLLMProvider(LLMProvider):
    @property
    def name(self) -> str:
        return "mock-qwen-3b"

    async def generate(self, messages: list[dict[str, str]], **kwargs: object) -> LLMResponse:
        return LLMResponse(
            content="Plan: 1. Confirm lamp location. 2. Send power toggle intent.",
            model="mock-qwen-3b",
            usage={"total_tokens": 85},
        )

    async def stream(self, messages: list[dict[str, str]], **kwargs: object):
        yield "Plan: "
        yield "1. Confirm lamp location."


@pytest.mark.asyncio
async def test_vision_perception_adapter() -> None:
    stub = StubSiglipProvider()
    adapter = VisionPerceptionAdapter(underlying_provider=stub)  # type: ignore[arg-type]
    await adapter.initialize()

    assert adapter.capability.provider_type == "perception"
    assert "visual" in adapter.capability.modalities

    assessments = await adapter.observe(b"fake_image_bytes")
    assert len(assessments) == 1
    assert assessments[0].evidence.embedding.model_id == "google/siglip-base-patch16-224"

    await adapter.shutdown()


@pytest.mark.asyncio
async def test_speech_perception_adapter() -> None:
    stub = StubMoonshineProvider()
    adapter = SpeechPerceptionAdapter(underlying_provider=stub)  # type: ignore[arg-type]
    await adapter.initialize()

    assert adapter.capability.supports_capability("transcribe_speech")

    evidence_list = await adapter.observe(b"fake_audio_pcm")
    assert len(evidence_list) == 1
    assert evidence_list[0].transcript == "Turn on the desk lamp"
    assert evidence_list[0].confidence == 0.96

    await adapter.shutdown()


@pytest.mark.asyncio
async def test_audio_event_perception_adapter() -> None:
    adapter = AudioEventPerceptionAdapter()
    await adapter.initialize()

    assert adapter.capability.supports_capability("classify_sound_events")
    await adapter.shutdown()


@pytest.mark.asyncio
async def test_telemetry_sensor_adapter() -> None:
    adapter = TelemetrySensorAdapter()
    await adapter.initialize()

    assert adapter.capability.provider_type == "perception"
    readings = await adapter.observe()
    assert len(readings) >= 1
    assert any("system_os" in r["predicate"] for r in readings)

    await adapter.shutdown()


@pytest.mark.asyncio
async def test_llm_cognition_adapter() -> None:
    stub_llm = StubLLMProvider()
    adapter = LLMCognitionAdapter(underlying_provider=stub_llm)
    await adapter.initialize()

    assert adapter.capability.provider_type == "cognition"

    req = CognitionRequest(
        intent="plan_device_action",
        cognitive_state_summary={"device": "lamp", "state": "off"},
        goals=["toggle_lamp"],
        constraints=["safety_check"],
    )

    thought = await adapter.reason(req)
    assert thought.confidence >= 0.8
    assert "Plan:" in thought.conclusion
    assert thought.tokens_used == 85
    assert len(thought.reasoning_trace) > 0

    await adapter.shutdown()


@pytest.mark.asyncio
async def test_symbolic_cognition_adapter() -> None:
    adapter = SymbolicCognitionAdapter()
    await adapter.initialize()

    assert adapter.capability.supports_capability("symbolic_reasoning")

    req = CognitionRequest(
        intent="verify_spatial_state",
        cognitive_state_summary={
            "belief": "Screwdriver on workbench",
            "spatial_location": "workbench_cam (depth 0.8m)",
            "query": "Where is the tool?",
        },
        evidence_refs=["ev_001"],
    )

    thought = await adapter.reason(req)
    assert thought.confidence == 0.98
    assert thought.tokens_used == 0
    assert "Grounded belief validated" in thought.conclusion

    await adapter.shutdown()


@pytest.mark.asyncio
async def test_tts_action_adapter() -> None:
    adapter = TTSActionAdapter()
    await adapter.initialize()

    assert adapter.capability.supports_capability("speak")

    intent = ActionIntent(
        action_type="speak",
        target="user",
        parameters={"text": "Task completed successfully."},
        safety_constraints=["max_chars=100"],
    )

    result = await adapter.execute(intent)
    assert result.success is True
    assert "Task completed" in result.actual_effect
    assert len(adapter.speech_history) == 1

    await adapter.shutdown()


@pytest.mark.asyncio
async def test_system_action_adapter() -> None:
    adapter = SystemActionAdapter()
    await adapter.initialize()

    assert adapter.capability.supports_capability("notify")

    intent = ActionIntent(
        action_type="notify",
        target="dashboard",
        parameters={"level": "info", "message": "Loop active"},
        authorization="user",
    )

    result = await adapter.execute(intent)
    assert result.success is True
    assert "Executed notify" in result.actual_effect

    # Unauthorized dangerous action test
    unauth_intent = ActionIntent(
        action_type="run_command",
        target="host",
        parameters={"cmd": "reboot"},
        authorization="untrusted",
    )
    unauth_res = await adapter.execute(unauth_intent)
    assert unauth_res.success is False
    assert "Unauthorized" in unauth_res.error

    await adapter.shutdown()
