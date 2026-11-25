"""Audio Provider Protocols — HBLLM Audio Perception §A1.

Defines the typed provider protocols for audio perception.
Each protocol produces raw/typed perception results; the
AudioPerceptionRuntime normalizes them into evidence.

Providers NEVER know about HCIR.

Protocols:
    AudioProvider        — base for all audio providers
    SpeechProvider       — speech-to-text
    AcousticEventProvider — ambient/environmental sound detection
    AcousticSceneProvider — scene-level acoustic analysis
    SpeakerProvider      — speaker identification/verification
    SoundLocalizationProvider — source direction/distance
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from hbllm.perception.providers.base import PerceptionProvider

if TYPE_CHECKING:
    from hbllm.perception.providers.audio_types import (
        AcousticSceneResult,
        AudioInput,
        SoundEventResult,
        SoundLocalizationResult,
        SpeakerIdentification,
        SpeechResult,
    )


@runtime_checkable
class AudioProvider(PerceptionProvider, Protocol):
    """Base protocol for all audio perception providers.

    All audio providers share:
        - modality = "audio"
        - provider_id for provenance
        - initialize/shutdown lifecycle
    """

    @property
    def sample_rate(self) -> int:
        """Expected input sample rate in Hz."""
        ...


@runtime_checkable
class SpeechProvider(AudioProvider, Protocol):
    """Speech-to-text provider.

    Produces raw transcription results. The runtime normalizes
    these into SpeechEvidence with temporal identity and
    epistemic profile.
    """

    async def transcribe(self, audio: AudioInput) -> SpeechResult:
        """Transcribe audio to text.

        Args:
            audio: Raw audio input (bytes, path, or numpy array).

        Returns:
            SpeechResult with transcript, language, confidence.

        """
        ...

    async def transcribe_streaming(
        self,
        audio_chunks: Sequence[AudioInput],
    ) -> SpeechResult:
        """Transcribe from streaming audio chunks.

        Args:
            audio_chunks: Ordered list of audio chunks.

        Returns:
            SpeechResult for the accumulated utterance.

        """
        ...


@runtime_checkable
class AcousticEventProvider(AudioProvider, Protocol):
    """Ambient/environmental sound detector.

    Classifies audio segments into acoustic event categories
    (doorbell, alarm, glass breaking, etc.). Produces raw
    classification results, NOT HCIR observations.
    """

    async def classify(self, audio: AudioInput) -> list[SoundEventResult]:
        """Classify acoustic events in an audio segment.

        Args:
            audio: Raw audio input.

        Returns:
            List of detected sound events with confidence scores.

        """
        ...


@runtime_checkable
class AcousticSceneProvider(AudioProvider, Protocol):
    """Scene-level acoustic analysis.

    Produces an overall characterization of the acoustic
    environment (indoor/outdoor, noise level, activity level).
    """

    async def analyze_scene(self, audio: AudioInput) -> AcousticSceneResult:
        """Analyze the acoustic scene.

        Args:
            audio: Raw audio input.

        Returns:
            AcousticSceneResult with environment characterization.

        """
        ...


@runtime_checkable
class SpeakerProvider(AudioProvider, Protocol):
    """Speaker identification and verification.

    Identifies who is speaking from audio. Returns a structured
    SpeakerIdentification, not a plain string ID.
    """

    async def identify(self, audio: AudioInput) -> SpeakerIdentification:
        """Identify the speaker from an audio segment.

        Args:
            audio: Raw audio input.

        Returns:
            SpeakerIdentification with embedding, confidence, enrolled status.

        """
        ...

    async def enroll(self, speaker_id: str, audio: AudioInput) -> bool:
        """Enroll a new speaker from audio sample.

        Args:
            speaker_id: Label for the speaker.
            audio: Audio sample for enrollment.

        Returns:
            True if enrollment succeeded.

        """
        ...


@runtime_checkable
class SoundLocalizationProvider(AudioProvider, Protocol):
    """Sound source localization (direction/distance).

    For multi-microphone setups. Not implemented in A2,
    but the protocol is defined here for future extensibility.
    """

    async def localize(self, audio: AudioInput) -> SoundLocalizationResult:
        """Estimate the direction and distance of a sound source.

        Args:
            audio: Multi-channel audio input.

        Returns:
            SoundLocalizationResult with direction and distance estimates.

        """
        ...
