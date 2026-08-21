"""Audio Perception Runtime — evidence-only layer.

Produces AudioAssessment from audio input. NEVER mutates HCIR.

Architecture:
    AudioInput → providers → raw results → normalize → AudioAssessment

    The runtime coordinates multiple providers (speech, event, scene,
    speaker) and normalizes their outputs into a single AudioAssessment
    with epistemic profile.

Invariant:
    Providers produce raw perception results.
    Runtime normalizes them into evidence.
    HCIR transaction commits observations.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from hbllm.perception.audio_memory import AudioMemory
from hbllm.perception.providers.audio_base import (
    AcousticEventProvider,
    AcousticSceneProvider,
    SpeakerProvider,
    SpeechProvider,
)
from hbllm.perception.providers.audio_evidence import (
    AcousticObservation,
    AcousticSceneEvidence,
    AudioAssessment,
    AudioEpistemicProfile,
    SoundEventEvidence,
    SpeechEvidence,
)
from hbllm.perception.providers.audio_types import (
    AudioInput,
    TemporalSpan,
)
from hbllm.perception.providers.provider_provenance import ProviderProvenance

logger = logging.getLogger(__name__)


class AudioPerceptionRuntime:
    """Evidence-only audio perception runtime.

    Coordinates providers, normalizes results into evidence, and
    builds an AudioAssessment. NEVER mutates HCIR or memory.

    Usage::

        runtime = AudioPerceptionRuntime(
            speech=moonshine_provider,
            events=ambient_provider,
        )
        assessment = await runtime.perceive(audio_bytes)

    """

    def __init__(
        self,
        speech: SpeechProvider | None = None,
        events: AcousticEventProvider | None = None,
        scene: AcousticSceneProvider | None = None,
        speaker: SpeakerProvider | None = None,
        memory: AudioMemory | None = None,
    ) -> None:
        self._speech = speech
        self._events = events
        self._scene = scene
        self._speaker = speaker
        self._memory = memory or AudioMemory()

    def _get_provenance(self, provider: Any) -> ProviderProvenance:
        """Extract provenance metadata from a provider."""
        provider_id = getattr(provider, "provider_id", "unknown")
        # Parse provider_id format "name:model" if available
        parts = provider_id.split(":", 1)
        return ProviderProvenance(
            provider=parts[0],
            model=parts[1] if len(parts) > 1 else "",
        )

    async def perceive(
        self,
        audio: AudioInput,
        label: str | None = None,
    ) -> AudioAssessment:
        """Full perception — run all available providers.

        Args:
            audio: Raw audio input.
            label: Optional user-provided label (for learning).

        Returns:
            AudioAssessment with all evidence and epistemic profile.

        """
        now = time.time()
        observation = AcousticObservation(
            temporal=TemporalSpan(start_time=now),
        )

        speech_evidence = None
        event_evidence: list[SoundEventEvidence] = []
        scene_evidence = None

        # ── Speech ──
        if self._speech is not None:
            try:
                result = await self._speech.transcribe(audio)
                speech_evidence = SpeechEvidence(
                    observation=observation,
                    transcript=result.transcript,
                    language=result.language,
                    confidence=result.confidence,
                    speaker_ref=result.speaker,
                    paralinguistic=result.paralinguistic,
                    provider_provenance=self._get_provenance(self._speech),
                )
            except Exception:
                logger.exception("Speech provider failed")

        # ── Events ──
        if self._events is not None:
            try:
                results = await self._events.classify(audio)
                event_prov = self._get_provenance(self._events)
                for r in results:
                    event_evidence.append(
                        SoundEventEvidence(
                            observation=observation,
                            event_type=r.event_type,
                            confidence=r.confidence,
                            is_critical=r.is_critical,
                            event_state=r.temporal.state,
                            top_classes=r.top_classes,
                            provider_provenance=event_prov,
                        ),
                    )
            except Exception:
                logger.exception("Event provider failed")

        # ── Scene ──
        if self._scene is not None:
            try:
                result = await self._scene.analyze_scene(audio)
                scene_evidence = AcousticSceneEvidence(
                    observation=observation,
                    indoor=result.indoor,
                    speech_present=result.speech_present,
                    noise_level=result.noise_level,
                    estimated_activity=result.estimated_activity,
                    scene_tags=result.scene_tags,
                    provider_provenance=self._get_provenance(self._scene),
                )
            except Exception:
                logger.exception("Scene provider failed")

        # ── Build epistemic profile ──
        profile = self._build_profile(
            speech_evidence,
            event_evidence,
            scene_evidence,
        )

        return AudioAssessment(
            observation=observation,
            speech=speech_evidence,
            events=event_evidence,
            scene=scene_evidence,
            epistemic_profile=profile,
            proposed_label=label,
        )

    async def perceive_speech(self, audio: AudioInput) -> SpeechEvidence:
        """Speech-only perception.

        Args:
            audio: Raw audio input.

        Returns:
            SpeechEvidence from the speech provider.

        Raises:
            RuntimeError: If no speech provider is configured.

        """
        if self._speech is None:
            msg = "No speech provider configured"
            raise RuntimeError(msg)

        result = await self._speech.transcribe(audio)
        observation = AcousticObservation(
            temporal=TemporalSpan(start_time=time.time()),
        )
        return SpeechEvidence(
            observation=observation,
            transcript=result.transcript,
            language=result.language,
            confidence=result.confidence,
            speaker_ref=result.speaker,
            paralinguistic=result.paralinguistic,
            provider_provenance=self._get_provenance(self._speech),
        )

    def _build_profile(
        self,
        speech: SpeechEvidence | None,
        events: list[SoundEventEvidence],
        scene: AcousticSceneEvidence | None,
    ) -> AudioEpistemicProfile:
        """Build epistemic profile from available evidence."""
        perceptual = 0.0
        classification = 0.0
        temporal = 0.5  # Moderate default

        if speech is not None:
            perceptual = max(perceptual, speech.confidence)

        if events:
            best_event = max(events, key=lambda e: e.confidence)
            classification = best_event.confidence

        if scene is not None:
            classification = max(classification, scene.confidence)

        return AudioEpistemicProfile(
            perceptual_confidence=perceptual,
            classification_confidence=classification,
            temporal_confidence=temporal,
        )
