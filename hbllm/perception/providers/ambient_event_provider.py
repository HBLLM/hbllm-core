"""Ambient Event Provider — HBLLM Audio Perception §A7.

Implements AcousticEventProvider and AcousticSceneProvider protocols
by wrapping the existing AmbientAudioClassifier (YAMNet ONNX + heuristic).

The provider returns raw classification results. The runtime converts
them into evidence. No HCIR awareness.

Multiple classifier results are preserved as ranked candidates —
no arbitration happens at the provider level.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from hbllm.perception.providers.audio_types import (
    AcousticSceneResult,
    AudioInput,
    SoundEventResult,
    TemporalSpan,
)

logger = logging.getLogger(__name__)


class AmbientEventProvider:
    """AcousticEventProvider + AcousticSceneProvider wrapping AmbientAudioClassifier.

    Implements both protocols from audio_base.py.

    Usage::

        provider = AmbientEventProvider()
        await provider.initialize()
        events = await provider.classify(audio_bytes)
        scene = await provider.analyze_scene(audio_bytes)

    """

    def __init__(
        self,
        model_path: str | None = None,
        min_energy_db: float = -40.0,
    ) -> None:
        self._model_path = model_path
        self._min_energy_db = min_energy_db
        self._classifier: Any = None
        self._target_sr = 16000

    # ── PerceptionProvider protocol ──────────────────────────────────────

    @property
    def modality(self) -> str:
        return "audio"

    @property
    def provider_id(self) -> str:
        mode = "onnx" if (self._classifier and self._classifier._use_onnx) else "heuristic"
        return f"ambient:{mode}"

    @property
    def sample_rate(self) -> int:
        return self._target_sr

    async def initialize(self) -> None:
        """Initialize the ambient audio classifier."""
        from hbllm.perception.ambient_audio_classifier import AmbientAudioClassifier

        self._classifier = AmbientAudioClassifier(
            model_path=self._model_path,
            min_energy_db=self._min_energy_db,
        )

    async def shutdown(self) -> None:
        """Release classifier resources."""
        self._classifier = None

    # ── AcousticEventProvider protocol ───────────────────────────────────

    async def classify(self, audio: AudioInput) -> list[SoundEventResult]:
        """Classify audio into acoustic events.

        Returns ALL candidate events with their confidences —
        no arbitration. The runtime decides what to keep.
        """
        if self._classifier is None:
            await self.initialize()

        samples = self._to_float32(audio)
        if len(samples) == 0:
            return []

        result = self._classifier.classify(samples, self._target_sr)

        now = time.time()
        events: list[SoundEventResult] = []

        # Primary classification
        if result.sound_class.value != "silence":
            events.append(SoundEventResult(
                event_type=result.sound_class.value,
                confidence=result.confidence,
                is_critical=result.is_critical,
                temporal=TemporalSpan(start_time=now),
                top_classes=result.top_classes,
            ))

        # Add runner-up candidates above threshold
        for class_name, score in result.top_classes[1:]:
            if score >= 0.3:
                events.append(SoundEventResult(
                    event_type=class_name,
                    confidence=score,
                    is_critical=False,
                    temporal=TemporalSpan(start_time=now),
                    top_classes=[],
                ))

        return events

    # ── AcousticSceneProvider protocol ───────────────────────────────────

    async def analyze_scene(self, audio: AudioInput) -> AcousticSceneResult:
        """Analyze the acoustic scene from audio.

        Produces scene-level characterization: indoor/outdoor,
        noise level, activity level, and descriptive tags.
        """
        if self._classifier is None:
            await self.initialize()

        samples = self._to_float32(audio)
        now = time.time()

        if len(samples) == 0:
            return AcousticSceneResult(temporal=TemporalSpan(start_time=now))

        result = self._classifier.classify(samples, self._target_sr)

        # Infer scene properties from classification
        speech_present = result.sound_class.value in ("speech",)
        scene_tags = [result.sound_class.value]
        if result.is_critical:
            scene_tags.append("alert")

        # Estimate noise level from energy
        noise_level = max(0.0, min(1.0, (result.energy_db + 60.0) / 60.0))

        # Estimate activity from energy and classification
        activity = 0.0
        if result.sound_class.value != "silence":
            activity = min(1.0, result.confidence * 0.5 + noise_level * 0.5)

        return AcousticSceneResult(
            indoor=True,  # Default — requires environmental model to determine
            speech_present=speech_present,
            noise_level=noise_level,
            estimated_activity=activity,
            scene_tags=scene_tags,
            temporal=TemporalSpan(start_time=now),
        )

    # ── Audio Conversion ─────────────────────────────────────────────────

    def _to_float32(self, audio: AudioInput) -> np.ndarray:
        """Convert AudioInput to float32 numpy array."""
        if isinstance(audio, np.ndarray):
            if audio.dtype == np.float32:
                return audio
            if audio.dtype == np.int16:
                return audio.astype(np.float32) / 32768.0
            return audio.astype(np.float32)

        if isinstance(audio, bytes):
            if len(audio) == 0:
                return np.zeros(0, dtype=np.float32)
            return np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        return np.zeros(0, dtype=np.float32)
