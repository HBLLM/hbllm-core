"""Audio Event Perception Adapter — concrete UnifiedPerceptionProvider wrapping AmbientEventProvider (YAMNet).

Converts ambient sound classifications and scene characterizations into typed
SoundEventEvidence / AcousticSceneEvidence without coupling to HCIR.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from hbllm.hcir.types import Provenance
from hbllm.perception.providers.ambient_event_provider import AmbientEventProvider
from hbllm.perception.providers.audio_evidence import (
    AcousticObservation,
    AcousticSceneEvidence,
    SoundEventEvidence,
)
from hbllm.perception.providers.audio_types import (
    AudioEventState,
    TemporalSpan,
)
from hbllm.perception.providers.provider_provenance import ProviderProvenance
from hbllm.runtime.providers.capability import ProviderCapability

logger = logging.getLogger(__name__)


class AudioEventPerceptionAdapter:
    """Concrete perception adapter wrapping AmbientEventProvider (YAMNet/heuristic).

    Conforms to ``UnifiedPerceptionProvider``.
    Produces ``SoundEventEvidence`` and ``AcousticSceneEvidence`` instances
    that ``EvidenceNormalizer`` converts to canonical ``PerceptualEvidenceNode`` objects.

    Usage::

        adapter = AudioEventPerceptionAdapter()
        await adapter.initialize()
        evidence_list = await adapter.observe(pcm_audio_bytes)
    """

    def __init__(
        self,
        provider_id: str = "yamnet_event",
        model_path: str | None = None,
        min_energy_db: float = -40.0,
        underlying_provider: AmbientEventProvider | None = None,
    ) -> None:
        self._provider_id = provider_id
        self._provider = underlying_provider or AmbientEventProvider(
            model_path=model_path,
            min_energy_db=min_energy_db,
        )

    @property
    def capability(self) -> ProviderCapability:
        """Declarative capability manifest for acoustic event detection."""
        return ProviderCapability(
            provider_id=self._provider_id,
            provider_type="perception",
            capabilities=[
                "classify_sound_events",
                "detect_acoustic_scene",
                "audio_event_detection",
            ],
            modalities=["audio"],
            latency_profile="very_low",
            quality_profile="high",
            memory_requirement_mb=120,
            hardware_requirements=["cpu"],
            requires_network=False,
        )

    async def initialize(self) -> None:
        """Initialize ambient audio classifier resources."""
        await self._provider.initialize()
        logger.info("Initialized AudioEventPerceptionAdapter (%s)", self._provider_id)

    async def shutdown(self) -> None:
        """Release audio classifier resources."""
        await self._provider.shutdown()
        logger.info("Shutdown AudioEventPerceptionAdapter (%s)", self._provider_id)

    async def observe(self, input_data: Any) -> list[SoundEventEvidence | AcousticSceneEvidence]:
        """Produce typed acoustic event observations from audio input.

        Args:
            input_data: Audio input data (bytes, array, path).

        Returns:
            List of SoundEventEvidence and AcousticSceneEvidence objects.
        """
        observations: list[SoundEventEvidence | AcousticSceneEvidence] = []
        try:
            # 1. Classify sound events
            events = await self._provider.classify(input_data)
            for ev in events:
                obs = AcousticObservation(
                    temporal=ev.temporal if ev.temporal else TemporalSpan(start_time=time.time()),
                    provenance=Provenance(
                        created_by=self._provider_id,
                        engine="yamnet",
                        source_type="observed",
                    ),
                )
                evidence = SoundEventEvidence(
                    observation=obs,
                    event_type=ev.event_type,
                    confidence=ev.confidence,
                    is_critical=ev.is_critical,
                    event_state=AudioEventState.INSTANTANEOUS,
                    top_classes=ev.top_classes,
                    provider_provenance=ProviderProvenance(
                        provider="yamnet",
                        model="yamnet-onnx",
                        version="1.0",
                    ),
                )
                observations.append(evidence)

            # 2. Analyze acoustic scene
            scene = await self._provider.analyze_scene(input_data)
            if scene is not None:
                scene_obs = AcousticObservation(
                    temporal=scene.temporal
                    if scene.temporal
                    else TemporalSpan(start_time=time.time()),
                    provenance=Provenance(
                        created_by=self._provider_id,
                        engine="yamnet-scene",
                        source_type="observed",
                    ),
                )
                scene_evidence = AcousticSceneEvidence(
                    observation=scene_obs,
                    indoor=scene.indoor,
                    speech_present=scene.speech_present,
                    noise_level=scene.noise_level,
                    estimated_activity=scene.estimated_activity,
                    scene_tags=scene.scene_tags,
                    confidence=0.85,
                    provider_provenance=ProviderProvenance(
                        provider="yamnet",
                        model="yamnet-scene",
                        version="1.0",
                    ),
                )
                observations.append(scene_evidence)

            return observations
        except Exception as e:
            logger.error("AudioEventPerceptionAdapter observation failed: %s", e)
            return []
