"""Perception Providers — typed protocols and adapters for grounded perception.

Provider protocols define the contracts for how HBLLM acquires perceptual
evidence from the physical world.  Each protocol represents a single
capability (encode, detect, extract_text, segment) — a provider can
implement multiple capabilities.

Architecture invariant:
    Providers NEVER mutate HCIR.
    Providers produce evidence (visual or audio).
    HCIR transactions own state mutation.
"""

from hbllm.perception.providers.audio_base import (
    AcousticEventProvider,
    AcousticSceneProvider,
    AudioProvider,
    SoundLocalizationProvider,
    SpeakerProvider,
    SpeechProvider,
)
from hbllm.perception.providers.audio_evidence import (
    AcousticObservation,
    AcousticSceneEvidence,
    AudioAssessment,
    AudioEpistemicProfile,
    SoundEventEvidence,
    SoundSourceEvidence,
    SpeechEvidence,
)
from hbllm.perception.providers.audio_policy import AudioRecognitionPolicy
from hbllm.perception.providers.audio_types import (
    AcousticSceneResult,
    AudioEmbedding,
    AudioEventState,
    AudioInput,
    ParalinguisticProfile,
    SoundEventResult,
    SoundLocalizationResult,
    SpeakerIdentification,
    SpeechResult,
    TemporalSpan,
)
from hbllm.perception.providers.base import (
    ImageInput,
    PerceptionProvider,
    VisionDetector,
    VisionOCR,
    VisionProvider,
)
from hbllm.perception.providers.evidence import (
    CandidateRanking,
    ConceptCandidate,
    EpistemicEvidenceProfile,
    ObservationMatch,
    VisualAssessment,
    VisualEvidence,
)
from hbllm.perception.providers.policy import RecognitionPolicy
from hbllm.perception.providers.types import (
    EmbeddingRef,
    VisualEmbedding,
    VisualRegion,
)

__all__ = [
    # ── Vision Protocols ──
    "PerceptionProvider",
    "VisionDetector",
    "VisionOCR",
    "VisionProvider",
    # ── Audio Protocols ──
    "AcousticEventProvider",
    "AcousticSceneProvider",
    "AudioProvider",
    "SoundLocalizationProvider",
    "SpeakerProvider",
    "SpeechProvider",
    # ── Vision Types ──
    "EmbeddingRef",
    "ImageInput",
    "VisualEmbedding",
    "VisualRegion",
    # ── Audio Types ──
    "AcousticSceneResult",
    "AudioEmbedding",
    "AudioEventState",
    "AudioInput",
    "ParalinguisticProfile",
    "SoundEventResult",
    "SoundLocalizationResult",
    "SpeakerIdentification",
    "SpeechResult",
    "TemporalSpan",
    # ── Vision Evidence ──
    "CandidateRanking",
    "ConceptCandidate",
    "EpistemicEvidenceProfile",
    "ObservationMatch",
    "VisualAssessment",
    "VisualEvidence",
    # ── Audio Evidence ──
    "AcousticObservation",
    "AcousticSceneEvidence",
    "AudioAssessment",
    "AudioEpistemicProfile",
    "SoundEventEvidence",
    "SoundSourceEvidence",
    "SpeechEvidence",
    # ── Policies ──
    "AudioRecognitionPolicy",
    "RecognitionPolicy",
]
