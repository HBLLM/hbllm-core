"""Speech Perception Adapter — concrete UnifiedPerceptionProvider wrapping Moonshine / Whisper.

Converts speech audio inputs into typed SpeechEvidence observations without coupling
the ASR models directly to HCIR.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from hbllm.hcir.types import Provenance
from hbllm.perception.providers.audio_evidence import (
    AcousticObservation,
    SpeechEvidence,
)
from hbllm.perception.providers.audio_types import TemporalSpan
from hbllm.perception.providers.moonshine_speech_provider import MoonshineSpeechProvider
from hbllm.perception.providers.provider_provenance import ProviderProvenance
from hbllm.runtime.providers.capability import ProviderCapability
from hbllm.runtime.providers.perception import UnifiedPerceptionProvider

logger = logging.getLogger(__name__)


class SpeechPerceptionAdapter:
    """Concrete perception adapter wrapping MoonshineSpeechProvider / Whisper.

    Conforms to ``UnifiedPerceptionProvider``.
    Produces ``SpeechEvidence`` instances that ``EvidenceNormalizer``
    converts to canonical ``PerceptualEvidenceNode`` objects.

    Usage::

        adapter = SpeechPerceptionAdapter()
        await adapter.initialize()
        evidence_list = await adapter.observe(pcm_audio_bytes)
    """

    def __init__(
        self,
        provider_id: str = "moonshine_speech",
        model_size: str = "base",
        target_sample_rate: int = 16000,
        underlying_provider: MoonshineSpeechProvider | None = None,
    ) -> None:
        self._provider_id = provider_id
        self._model_size = model_size
        self._provider = underlying_provider or MoonshineSpeechProvider(
            model_size=model_size,
            target_sample_rate=target_sample_rate,
        )

    @property
    def capability(self) -> ProviderCapability:
        """Declarative capability manifest for speech transcription."""
        return ProviderCapability(
            provider_id=self._provider_id,
            provider_type="perception",
            capabilities=["transcribe_speech", "detect_language", "speaker_identification"],
            modalities=["audio"],
            latency_profile="low",
            quality_profile="high",
            memory_requirement_mb=350,
            hardware_requirements=["cpu"],
            requires_network=False,
        )

    async def initialize(self) -> None:
        """Initialize speech model resources."""
        await self._provider.initialize()
        logger.info("Initialized SpeechPerceptionAdapter (%s)", self._provider_id)

    async def shutdown(self) -> None:
        """Release speech model resources."""
        await self._provider.shutdown()
        logger.info("Shutdown SpeechPerceptionAdapter (%s)", self._provider_id)

    async def observe(self, input_data: Any) -> list[SpeechEvidence]:
        """Produce typed speech observations from audio input.

        Args:
            input_data: Audio data (bytes, numpy array, or file path).

        Returns:
            List containing SpeechEvidence.
        """
        try:
            result = await self._provider.transcribe(input_data)
            obs = AcousticObservation(
                temporal=result.temporal if result.temporal else TemporalSpan(start_time=time.time()),
                provenance=Provenance(
                    created_by=self._provider_id,
                    engine=f"moonshine-{self._model_size}",
                    source_type="observed",
                ),
            )

            speech_ev = SpeechEvidence(
                observation=obs,
                transcript=result.transcript,
                language=result.language,
                confidence=result.confidence,
                speaker_ref=result.speaker,
                paralinguistic=result.paralinguistic,
                is_partial=False,
                provider_provenance=ProviderProvenance(
                    provider="moonshine",
                    model=f"moonshine-{self._model_size}",
                    version="1.0",
                ),
            )
            return [speech_ev]
        except Exception as e:
            logger.error("SpeechPerceptionAdapter observation failed: %s", e)
            return []
