"""Audio Evidence Types — HBLLM Audio Perception §A1.

Evidence is the NORMALIZED interpretation of provider results.
Evidence ≠ observation ≠ belief.

Architecture:
    Provider Result  → raw STT/classification output
    Audio Evidence   → normalized, with epistemic profile
    HCIR Observation → "something happened in the world"
    Belief           → "I believe X because of Y"

Composition model (NOT inheritance):
    AcousticObservation contains the raw "microphone received this"
    Specialized evidence types are separate, linked by observation_ref.

    AcousticObservation ─┬── SpeechEvidence
                         ├── SoundEventEvidence
                         ├── SoundSourceEvidence
                         └── AcousticSceneEvidence

    Each is a different interpretation of the same observation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from hbllm.hcir.types import Provenance
from hbllm.perception.providers.audio_types import (
    AudioEventState,
    ParalinguisticProfile,
    SpeakerIdentification,
    TemporalSpan,
)
from hbllm.perception.providers.provider_provenance import ProviderProvenance

# ═══════════════════════════════════════════════════════════════════════════
# Acoustic Observation — "the microphone received this"
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AcousticObservation:
    """Raw acoustic observation — the microphone received this signal.

    This is NOT an interpretation. It is the fact that an acoustic
    pattern was observed. Multiple evidence types can reference the
    same observation.

    Attributes:
        observation_id: Unique ID for this observation.
        embedding_ref: Reference to stored audio embedding (not the vector).
        embedding_space: Embedding space ID for compatibility.
        temporal: Temporal identity (start/end/duration/event/segment).
        energy_db: RMS energy in decibels.
        provenance: How this observation was created.

    """

    observation_id: str = field(
        default_factory=lambda: f"aobs_{uuid.uuid4().hex[:12]}",
    )
    embedding_ref: str | None = None
    embedding_space: str = ""
    temporal: TemporalSpan = field(default_factory=TemporalSpan)
    energy_db: float = 0.0
    provenance: Provenance = field(default_factory=Provenance)


# ═══════════════════════════════════════════════════════════════════════════
# Evidence Types — interpretations of observations
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SpeechEvidence:
    """Speech evidence — one interpretation of an acoustic observation.

    Attributes:
        observation: The underlying acoustic observation.
        transcript: Transcribed text.
        language: Detected language code.
        confidence: Transcription confidence.
        speaker_ref: Structured speaker identity (NOT a plain string).
        paralinguistic: Probabilistic tone/emotion profile.
        is_partial: Whether this is a partial/streaming result.

    """

    observation: AcousticObservation = field(default_factory=AcousticObservation)
    transcript: str = ""
    language: str = "en"
    confidence: float = 0.0
    speaker_ref: SpeakerIdentification | None = None
    paralinguistic: ParalinguisticProfile | None = None
    is_partial: bool = False
    provider_provenance: ProviderProvenance = field(default_factory=ProviderProvenance)


@dataclass
class SoundEventEvidence:
    """Sound event evidence — one interpretation of an acoustic observation.

    Attributes:
        observation: The underlying acoustic observation.
        event_type: Classified event (DOORBELL, ALARM, KNOCK, etc.).
        confidence: Classification confidence.
        is_critical: Safety-critical sound (smoke detector, glass breaking).
        event_state: Temporal state (started/continued/ended/instantaneous).
        top_classes: Top-k classification alternatives.

    """

    observation: AcousticObservation = field(default_factory=AcousticObservation)
    event_type: str = "unknown"
    confidence: float = 0.0
    is_critical: bool = False
    event_state: AudioEventState = AudioEventState.INSTANTANEOUS
    top_classes: list[tuple[str, float]] = field(default_factory=list)
    provider_provenance: ProviderProvenance = field(default_factory=ProviderProvenance)


@dataclass
class SoundSourceEvidence:
    """Sound source evidence — what produced the sound and where.

    Attributes:
        observation: The underlying acoustic observation.
        source_class: What kind of source (DOG, VEHICLE, APPLIANCE, HUMAN).
        direction_degrees: Estimated direction (0-360), or None.
        distance_estimate: Estimated distance in meters, or None.
        confidence: Classification/localization confidence.

    """

    observation: AcousticObservation = field(default_factory=AcousticObservation)
    source_class: str = "unknown"
    direction_degrees: float | None = None
    distance_estimate: float | None = None
    confidence: float = 0.0
    provider_provenance: ProviderProvenance = field(default_factory=ProviderProvenance)


@dataclass
class AcousticSceneEvidence:
    """Acoustic scene evidence — overall environment characterization.

    NOT itself an acoustic event. A scene assessment is a separate
    interpretation of the same observation.

    Attributes:
        observation: The underlying acoustic observation.
        indoor: Whether the environment appears indoor.
        speech_present: Whether speech is detected.
        noise_level: Normalized noise level (0.0-1.0).
        estimated_activity: Activity level (0.0-1.0).
        scene_tags: Descriptive tags.
        confidence: Overall scene assessment confidence.

    """

    observation: AcousticObservation = field(default_factory=AcousticObservation)
    indoor: bool = True
    speech_present: bool = False
    noise_level: float = 0.0
    estimated_activity: float = 0.0
    scene_tags: list[str] = field(default_factory=list)
    confidence: float = 0.0
    provider_provenance: ProviderProvenance = field(default_factory=ProviderProvenance)


# ═══════════════════════════════════════════════════════════════════════════
# Epistemic Profile
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AudioEpistemicProfile:
    """Multi-dimensional confidence for audio evidence.

    Mirrors VisualEpistemicProfile. Each dimension is independent.

    Attributes:
        perceptual_confidence: How clear was the audio signal?
        classification_confidence: How confident is the classifier?
        source_reliability: How reliable is this audio source?
        label_provenance: Was the label user-provided (1.0) or inferred (0.0)?
        temporal_confidence: How precise is the temporal localization?

    """

    perceptual_confidence: float = 0.0
    classification_confidence: float = 0.0
    source_reliability: float = 1.0
    label_provenance: float = 0.0
    temporal_confidence: float = 0.0

    @property
    def combined(self) -> float:
        """Weighted combination of all dimensions."""
        weights = [0.25, 0.30, 0.20, 0.15, 0.10]
        values = [
            self.perceptual_confidence,
            self.classification_confidence,
            self.source_reliability,
            self.label_provenance,
            self.temporal_confidence,
        ]
        return sum(w * v for w, v in zip(weights, values, strict=True))


# ═══════════════════════════════════════════════════════════════════════════
# Assessment — the full perception output
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AudioAssessment:
    """Full audio perception assessment — analogous to VisualAssessment.

    Contains the observation, all evidence interpretations, and
    epistemic profile. This is what the HCIR transaction receives.

    Attributes:
        observation: The base acoustic observation.
        speech: Speech evidence (if speech detected).
        events: Detected sound events.
        scene: Scene characterization.
        source: Sound source evidence.
        epistemic_profile: Multi-dimensional confidence.
        proposed_label: User-provided label (for learning).
        proposed_context: User-provided context.

    """

    observation: AcousticObservation = field(default_factory=AcousticObservation)
    speech: SpeechEvidence | None = None
    events: list[SoundEventEvidence] = field(default_factory=list)
    scene: AcousticSceneEvidence | None = None
    source: SoundSourceEvidence | None = None
    epistemic_profile: AudioEpistemicProfile = field(
        default_factory=AudioEpistemicProfile,
    )
    proposed_label: str | None = None
    proposed_context: str | None = None
