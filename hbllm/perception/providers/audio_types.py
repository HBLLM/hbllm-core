"""Audio Perception Types — HBLLM Audio Perception §A1.

Raw types for audio perception. These are provider-level outputs,
NOT evidence or HCIR observations.

Hierarchy (composition, not inheritance):
    AudioInput              — input format union
    AudioEmbedding          — embedding vector + metadata
    SpeechResult            — raw STT output
    SoundEventResult        — raw acoustic event classification
    AcousticSceneResult     — raw scene characterization
    SpeakerIdentification   — structured speaker identity
    SoundLocalizationResult — direction/distance estimate
    ParalinguisticProfile   — probabilistic tone/emotion

Identity model:
    observation_id — unique per sensor reading
    event_id       — groups related observations (start/continue/end)
    segment_id     — identifies the audio segment that was analyzed
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# Input Types
# ═══════════════════════════════════════════════════════════════════════════

AudioInput = bytes | str | Path | np.ndarray
"""Raw audio input: bytes, file path, or numpy array (float32, mono)."""


# ═══════════════════════════════════════════════════════════════════════════
# Embedding
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AudioEmbedding:
    """Audio embedding vector with metadata.

    Analogous to VisualEmbedding. L2-normalized.

    Attributes:
        vector: The embedding vector (L2-normalized).
        model_id: Which model produced this embedding.
        space_id: Embedding space identifier for compatibility checks.
        dimensions: Vector dimensionality.
        sample_rate: Sample rate of the input audio.

    """

    vector: list[float]
    model_id: str
    space_id: str
    dimensions: int
    sample_rate: int = 16000


# ═══════════════════════════════════════════════════════════════════════════
# Temporal Identity
# ═══════════════════════════════════════════════════════════════════════════


def _new_observation_id() -> str:
    return f"aobs_{uuid.uuid4().hex[:12]}"


def _new_event_id() -> str:
    return f"aevt_{uuid.uuid4().hex[:12]}"


def _new_segment_id() -> str:
    return f"aseg_{uuid.uuid4().hex[:12]}"


class AudioEventState(StrEnum):
    """Temporal state of an acoustic event."""

    STARTED = "started"
    CONTINUED = "continued"
    ENDED = "ended"
    INSTANTANEOUS = "instantaneous"  # Single-shot events (clap, knock)


@dataclass
class TemporalSpan:
    """Temporal identity for audio evidence.

    Separates three kinds of identity:
        observation_id — unique per sensor reading
        event_id       — groups related observations (start/continue/end)
        segment_id     — identifies the audio segment analyzed

    Attributes:
        observation_id: Unique ID for this specific observation.
        event_id: Groups related observations of the same event.
        segment_id: Identifies the audio segment.
        start_time: When this observation/event started.
        end_time: When this observation/event ended.
        duration: Duration in seconds.
        state: Temporal state (started/continued/ended/instantaneous).

    """

    observation_id: str = field(default_factory=_new_observation_id)
    event_id: str = field(default_factory=_new_event_id)
    segment_id: str = field(default_factory=_new_segment_id)
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    state: AudioEventState = AudioEventState.INSTANTANEOUS


# ═══════════════════════════════════════════════════════════════════════════
# Speaker
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SpeakerIdentification:
    """Structured speaker identity — NOT a plain string.

    Can represent identified, unidentified, or partially matched speakers.
    Designed to later become an independent SpeakerEvidence type.

    Attributes:
        speaker_id: Enrolled speaker ID, or None if unknown.
        embedding_ref: Reference to stored voice embedding.
        confidence: Identification confidence (0.0-1.0).
        is_enrolled: Whether this speaker is in the enrollment database.
        voice_characteristics: Optional acoustic features (pitch, timbre).

    """

    speaker_id: str | None = None
    embedding_ref: str = ""
    confidence: float = 0.0
    is_enrolled: bool = False
    voice_characteristics: dict[str, float] = field(default_factory=dict)

    @property
    def is_identified(self) -> bool:
        """Whether speaker was positively identified."""
        return self.speaker_id is not None and self.confidence > 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Paralinguistic
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ParalinguisticProfile:
    """Probabilistic tone/emotion profile.

    This is a probabilistic estimate, NOT a fact about someone's
    internal state. Should be treated as uncertain evidence.

    Attributes:
        tone: Estimated tone ("urgent", "calm", "excited", "hesitant").
        confidence: How confident the estimate is.
        pitch_mean: Average pitch in Hz.
        pitch_variance: Pitch variation.
        speech_rate: Words per minute estimate.
        energy_level: Normalized energy (0.0-1.0).

    """

    tone: str = "neutral"
    confidence: float = 0.0
    pitch_mean: float = 0.0
    pitch_variance: float = 0.0
    speech_rate: float = 0.0
    energy_level: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Provider Results (raw, pre-evidence)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SpeechResult:
    """Raw STT provider output — pre-evidence.

    Attributes:
        transcript: The transcribed text.
        language: Detected language code.
        confidence: Transcription confidence (0.0-1.0).
        speaker: Optional speaker identification.
        paralinguistic: Optional paralinguistic analysis.
        temporal: Temporal span of the speech.
        word_timestamps: Per-word timing (if available).

    """

    transcript: str = ""
    language: str = "en"
    confidence: float = 0.0
    speaker: SpeakerIdentification | None = None
    paralinguistic: ParalinguisticProfile | None = None
    temporal: TemporalSpan = field(default_factory=TemporalSpan)
    word_timestamps: list[tuple[str, float, float]] = field(default_factory=list)


@dataclass
class SoundEventResult:
    """Raw acoustic event classification — pre-evidence.

    Attributes:
        event_type: Classified event (DOORBELL, ALARM, etc.).
        confidence: Classification confidence (0.0-1.0).
        is_critical: Whether this is a safety-critical sound.
        temporal: Temporal span of the event.
        top_classes: Top-k classification scores.

    """

    event_type: str = "unknown"
    confidence: float = 0.0
    is_critical: bool = False
    temporal: TemporalSpan = field(default_factory=TemporalSpan)
    top_classes: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class AcousticSceneResult:
    """Raw scene characterization — pre-evidence.

    Attributes:
        indoor: Whether the environment is indoor.
        speech_present: Whether speech is detected.
        noise_level: Normalized noise level (0.0-1.0).
        estimated_activity: Estimated activity level (0.0-1.0).
        scene_tags: Descriptive tags for the scene.
        temporal: Temporal span of the analysis.

    """

    indoor: bool = True
    speech_present: bool = False
    noise_level: float = 0.0
    estimated_activity: float = 0.0
    scene_tags: list[str] = field(default_factory=list)
    temporal: TemporalSpan = field(default_factory=TemporalSpan)


@dataclass
class SoundLocalizationResult:
    """Raw sound localization — pre-evidence.

    Attributes:
        direction_degrees: Estimated direction in degrees (0-360).
        distance_estimate: Estimated distance (meters), or None.
        confidence: Localization confidence (0.0-1.0).
        temporal: Temporal span.

    """

    direction_degrees: float = 0.0
    distance_estimate: float | None = None
    confidence: float = 0.0
    temporal: TemporalSpan = field(default_factory=TemporalSpan)
