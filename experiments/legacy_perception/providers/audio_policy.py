"""Audio Recognition Policy — HBLLM Audio Perception §A1.

Configurable thresholds for audio perception decisions.
No magic numbers — all thresholds are explicit and tunable.

Usage::

    policy = AudioRecognitionPolicy()
    if evidence.confidence >= policy.speech_confidence_threshold:
        # Accept transcription

    # Or use a strict policy
    policy = AudioRecognitionPolicy.strict()
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioRecognitionPolicy:
    """Configurable thresholds for audio perception decisions.

    Attributes:
        speech_confidence_threshold: Minimum confidence to accept transcription.
        event_confidence_threshold: Minimum confidence to accept event classification.
        scene_confidence_threshold: Minimum confidence for scene characterization.
        speaker_confidence_threshold: Minimum confidence for speaker identification.
        critical_event_threshold: Lower threshold for safety-critical events.
        novelty_threshold: Below this similarity, sound is considered novel.
        ambiguity_margin: If top-2 candidates are within this margin, mark ambiguous.
        min_energy_db: Minimum energy to process (below = silence).

    """

    speech_confidence_threshold: float = 0.5
    event_confidence_threshold: float = 0.6
    scene_confidence_threshold: float = 0.5
    speaker_confidence_threshold: float = 0.7
    critical_event_threshold: float = 0.4
    novelty_threshold: float = 0.3
    ambiguity_margin: float = 0.15
    min_energy_db: float = -40.0

    @classmethod
    def strict(cls) -> AudioRecognitionPolicy:
        """High-confidence policy — fewer false positives."""
        return cls(
            speech_confidence_threshold=0.7,
            event_confidence_threshold=0.8,
            scene_confidence_threshold=0.7,
            speaker_confidence_threshold=0.85,
            critical_event_threshold=0.6,
            novelty_threshold=0.2,
            ambiguity_margin=0.10,
            min_energy_db=-35.0,
        )

    @classmethod
    def permissive(cls) -> AudioRecognitionPolicy:
        """Low-threshold policy — more detections, more noise."""
        return cls(
            speech_confidence_threshold=0.3,
            event_confidence_threshold=0.4,
            scene_confidence_threshold=0.3,
            speaker_confidence_threshold=0.5,
            critical_event_threshold=0.25,
            novelty_threshold=0.4,
            ambiguity_margin=0.20,
            min_energy_db=-50.0,
        )
