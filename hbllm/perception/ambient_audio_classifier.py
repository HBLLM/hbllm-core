"""Ambient Audio Classifier — environmental sound recognition.

Classifies ambient sounds into semantic categories:
    - doorbell, alarm, music, appliance, vehicle, pet, glass_breaking,
      smoke_detector, speech, silence, unknown

Architecture:
    Uses a lightweight ONNX model (YAMNet-style) for audio classification.
    Falls back to a simple energy + spectral feature classifier when the
    ONNX model is not available.

Integration:
    Publishes classified events to `perception.audio.ambient` on the
    MessageBus for consumption by autonomy reflexes and the WorldStateEngine.

Usage::

    classifier = AmbientAudioClassifier(bus=message_bus)
    await classifier.start()
    # Feeds from AudioInputNode or directly from microphone
    result = classifier.classify(audio_chunk, sample_rate=16000)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class AmbientSoundClass(str, Enum):
    """Recognized ambient sound categories."""

    SILENCE = "silence"
    SPEECH = "speech"
    MUSIC = "music"
    DOORBELL = "doorbell"
    KNOCK = "knock"
    ALARM = "alarm"
    SMOKE_DETECTOR = "smoke_detector"
    GLASS_BREAKING = "glass_breaking"
    DOG_BARK = "dog_bark"
    CAT_MEOW = "cat_meow"
    APPLIANCE = "appliance"
    VEHICLE = "vehicle"
    SIREN = "siren"
    WATER_RUNNING = "water_running"
    PHONE_RINGING = "phone_ringing"
    TYPING = "typing"
    FOOTSTEPS = "footsteps"
    UNKNOWN = "unknown"


# Critical sounds that should trigger immediate reflexes
CRITICAL_SOUNDS = {
    AmbientSoundClass.SMOKE_DETECTOR,
    AmbientSoundClass.GLASS_BREAKING,
    AmbientSoundClass.ALARM,
    AmbientSoundClass.SIREN,
}

# Sounds that indicate human presence
PRESENCE_SOUNDS = {
    AmbientSoundClass.SPEECH,
    AmbientSoundClass.FOOTSTEPS,
    AmbientSoundClass.TYPING,
    AmbientSoundClass.KNOCK,
}


@dataclass
class ClassificationResult:
    """Result of ambient sound classification."""

    sound_class: AmbientSoundClass
    confidence: float  # 0.0 - 1.0
    energy_db: float  # RMS energy in decibels
    timestamp: float = field(default_factory=time.time)
    is_critical: bool = False
    is_presence: bool = False
    top_classes: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sound_class": self.sound_class.value,
            "confidence": round(self.confidence, 3),
            "energy_db": round(self.energy_db, 1),
            "timestamp": self.timestamp,
            "is_critical": self.is_critical,
            "is_presence": self.is_presence,
            "top_classes": [(c, round(s, 3)) for c, s in self.top_classes[:5]],
        }


class AmbientAudioClassifier:
    """Ambient sound classification engine.

    Classifies audio chunks into semantic categories using either:
    1. An ONNX model (YAMNet-style, if available)
    2. A spectral feature-based heuristic classifier (fallback)

    Args:
        model_path: Path to ONNX classification model. None for heuristic mode.
        min_energy_db: Minimum energy threshold to trigger classification.
            Below this, the audio is classified as SILENCE.
        cooldown_s: Minimum seconds between publishing the same sound class
            to prevent notification storms.
        bus: MessageBus for publishing classification events.
    """

    def __init__(
        self,
        model_path: str | None = None,
        min_energy_db: float = -40.0,
        cooldown_s: float = 3.0,
        bus: Any | None = None,
    ) -> None:
        self.min_energy_db = min_energy_db
        self.cooldown_s = cooldown_s
        self.bus = bus

        self._onnx_session: Any = None
        self._use_onnx = False
        self._last_published: dict[str, float] = {}  # class → timestamp
        self._classifications_count = 0

        # Try to load ONNX model
        if model_path:
            try:
                import onnxruntime as ort

                self._onnx_session = ort.InferenceSession(
                    model_path,
                    providers=["CPUExecutionProvider"],
                )
                self._use_onnx = True
                logger.info("AmbientAudioClassifier: ONNX model loaded from %s", model_path)
            except Exception as e:
                logger.warning(
                    "AmbientAudioClassifier: ONNX model load failed (%s), "
                    "using heuristic classifier",
                    e,
                )

        if not self._use_onnx:
            logger.info(
                "AmbientAudioClassifier: Using heuristic spectral classifier "
                "(install ONNX model for better accuracy)"
            )

    def classify(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000,
    ) -> ClassificationResult:
        """Classify an audio chunk.

        Args:
            audio_chunk: Audio samples as float32 numpy array, shape (N,).
            sample_rate: Sample rate of the audio.

        Returns:
            ClassificationResult with the detected sound class and confidence.
        """
        self._classifications_count += 1

        # Compute RMS energy
        rms = float(np.sqrt(np.mean(audio_chunk**2))) if len(audio_chunk) > 0 else 0.0
        energy_db = 20 * np.log10(max(rms, 1e-10))

        # Below energy threshold → silence
        if energy_db < self.min_energy_db:
            return ClassificationResult(
                sound_class=AmbientSoundClass.SILENCE,
                confidence=0.95,
                energy_db=energy_db,
            )

        # Classify using ONNX or heuristic
        if self._use_onnx and self._onnx_session is not None:
            result = self._classify_onnx(audio_chunk, sample_rate, energy_db)
        else:
            result = self._classify_heuristic(audio_chunk, sample_rate, energy_db)

        # Set critical/presence flags
        result.is_critical = result.sound_class in CRITICAL_SOUNDS
        result.is_presence = result.sound_class in PRESENCE_SOUNDS

        return result

    async def classify_and_publish(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000,
    ) -> ClassificationResult:
        """Classify and publish result to MessageBus if not on cooldown.

        Deduplicates rapid-fire detections of the same sound class.
        """
        result = self.classify(audio_chunk, sample_rate)

        # Skip publishing silence
        if result.sound_class == AmbientSoundClass.SILENCE:
            return result

        # Check cooldown
        now = time.time()
        last = self._last_published.get(result.sound_class.value, 0.0)
        if now - last < self.cooldown_s and not result.is_critical:
            return result

        # Publish to bus
        if self.bus is not None:
            from hbllm.network.messages import Message, MessageType

            topic = "perception.audio.ambient"
            if result.is_critical:
                topic = "perception.audio.ambient.critical"

            await self.bus.publish(
                topic,
                Message(
                    type=MessageType.EVENT,
                    source_node_id="ambient_audio_classifier",
                    topic=topic,
                    payload=result.to_dict(),
                ),
            )
            self._last_published[result.sound_class.value] = now

        return result

    def _classify_onnx(
        self,
        audio: np.ndarray,
        sample_rate: int,
        energy_db: float,
    ) -> ClassificationResult:
        """Classify using ONNX model (YAMNet-style)."""
        try:
            # Resample to model's expected rate if needed (typically 16kHz)
            if sample_rate != 16000:
                # Simple linear interpolation resample
                ratio = 16000 / sample_rate
                new_length = int(len(audio) * ratio)
                indices = np.linspace(0, len(audio) - 1, new_length)
                audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

            # Run inference
            input_name = self._onnx_session.get_inputs()[0].name
            outputs = self._onnx_session.run(None, {input_name: audio.reshape(1, -1)})

            # Parse scores
            scores = outputs[0][0]  # Assuming shape (1, num_classes)
            top_indices = np.argsort(scores)[::-1][:5]

            # Map model output indices to our categories
            # (This mapping depends on the specific model used)
            sound_class = self._map_model_class(top_indices[0])
            confidence = float(scores[top_indices[0]])

            top_classes = [
                (self._map_model_class(idx).value, float(scores[idx])) for idx in top_indices
            ]

            return ClassificationResult(
                sound_class=sound_class,
                confidence=confidence,
                energy_db=energy_db,
                top_classes=top_classes,
            )
        except Exception as e:
            logger.debug("ONNX classification failed: %s, falling back to heuristic", e)
            return self._classify_heuristic(audio, sample_rate, energy_db)

    def _classify_heuristic(
        self,
        audio: np.ndarray,
        sample_rate: int,
        energy_db: float,
    ) -> ClassificationResult:
        """Classify using spectral features (no ML model required).

        Uses:
            - Spectral centroid (brightness)
            - Zero-crossing rate (noisiness)
            - Energy envelope variance (impulsiveness)
            - Fundamental frequency estimate (pitch)
        """
        n = len(audio)
        if n < 256:
            return ClassificationResult(
                sound_class=AmbientSoundClass.UNKNOWN,
                confidence=0.3,
                energy_db=energy_db,
            )

        # Spectral centroid — high = bright/sharp, low = bass/rumble
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
        centroid = float(np.sum(freqs * fft) / (np.sum(fft) + 1e-10))

        # Zero-crossing rate — high = noise/speech, low = tonal
        zcr = float(np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * n))

        # Energy variance — high = impulsive (knock, break), low = steady
        frame_size = min(512, n // 4)
        if frame_size > 0:
            frames = n // frame_size
            energies = [
                float(np.sqrt(np.mean(audio[i * frame_size : (i + 1) * frame_size] ** 2)))
                for i in range(frames)
            ]
            energy_var = float(np.var(energies)) if energies else 0.0
        else:
            energy_var = 0.0

        # Heuristic classification rules
        scores: dict[AmbientSoundClass, float] = {}

        # High pitch + impulsive = alarm/doorbell/smoke
        if centroid > 3000 and energy_var > 0.01:
            scores[AmbientSoundClass.ALARM] = 0.6
            if centroid > 4000:
                scores[AmbientSoundClass.SMOKE_DETECTOR] = 0.7

        # High pitch + tonal (low zcr) = phone/doorbell
        if centroid > 2000 and zcr < 0.2:
            scores[AmbientSoundClass.DOORBELL] = 0.5
            scores[AmbientSoundClass.PHONE_RINGING] = 0.4

        # Very impulsive + high energy = glass breaking / knock
        if energy_var > 0.05 and energy_db > -20:
            scores[AmbientSoundClass.GLASS_BREAKING] = 0.4
            scores[AmbientSoundClass.KNOCK] = 0.45

        # Medium ZCR + medium centroid = speech
        if 0.05 < zcr < 0.3 and 200 < centroid < 3000:
            scores[AmbientSoundClass.SPEECH] = 0.55

        # Low centroid + steady energy = music / appliance
        if centroid < 1000 and energy_var < 0.01:
            scores[AmbientSoundClass.MUSIC] = 0.4
            scores[AmbientSoundClass.APPLIANCE] = 0.35

        # Very low centroid + steady = vehicle
        if centroid < 500 and energy_var < 0.005:
            scores[AmbientSoundClass.VEHICLE] = 0.4

        # Medium-high centroid + periodic = animal
        if 1000 < centroid < 4000 and zcr > 0.1:
            scores[AmbientSoundClass.DOG_BARK] = 0.3

        # Low energy + rhythmic = footsteps/typing
        if energy_db < -20 and energy_var > 0.002:
            scores[AmbientSoundClass.FOOTSTEPS] = 0.35
            scores[AmbientSoundClass.TYPING] = 0.3

        # Steady mid-range = water
        if 500 < centroid < 3000 and zcr > 0.3 and energy_var < 0.005:
            scores[AmbientSoundClass.WATER_RUNNING] = 0.4

        if not scores:
            return ClassificationResult(
                sound_class=AmbientSoundClass.UNKNOWN,
                confidence=0.3,
                energy_db=energy_db,
            )

        # Pick best
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_class, best_score = sorted_scores[0]

        return ClassificationResult(
            sound_class=best_class,
            confidence=best_score,
            energy_db=energy_db,
            top_classes=[(c.value, s) for c, s in sorted_scores[:5]],
        )

    def _map_model_class(self, class_idx: int) -> AmbientSoundClass:
        """Map ONNX model output index to AmbientSoundClass.

        Override this method for different model architectures.
        Default mapping is for YAMNet-style 521-class output.
        """
        # Simplified mapping — extend based on actual model
        yamnet_mappings: dict[int, AmbientSoundClass] = {
            0: AmbientSoundClass.SPEECH,
            1: AmbientSoundClass.SPEECH,
            # Music
            137: AmbientSoundClass.MUSIC,
            138: AmbientSoundClass.MUSIC,
            # Doorbell
            390: AmbientSoundClass.DOORBELL,
            # Alarm
            400: AmbientSoundClass.ALARM,
            401: AmbientSoundClass.SMOKE_DETECTOR,
            402: AmbientSoundClass.SIREN,
            # Animals
            67: AmbientSoundClass.DOG_BARK,
            80: AmbientSoundClass.CAT_MEOW,
            # Glass
            447: AmbientSoundClass.GLASS_BREAKING,
            # Vehicle
            300: AmbientSoundClass.VEHICLE,
        }
        return yamnet_mappings.get(class_idx, AmbientSoundClass.UNKNOWN)

    def stats(self) -> dict[str, Any]:
        """Classification engine statistics."""
        return {
            "mode": "onnx" if self._use_onnx else "heuristic",
            "classifications": self._classifications_count,
            "cooldown_s": self.cooldown_s,
            "min_energy_db": self.min_energy_db,
        }
