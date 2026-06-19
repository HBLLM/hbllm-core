"""
Wake Word Detector — Hands-free voice activation.

Provides always-on keyword spotting for triggering the audio pipeline
without explicit activation. Supports multiple backends:

- **OpenWakeWord** (default): Open-source, lightweight (~5MB), supports
  custom wake words. Runs on CPU.
- **Porcupine**: Picovoice commercial engine, high accuracy, requires
  API key.
- **Energy-based**: Simple energy threshold fallback — no ML required.

Bus Topics:
    perception.wake_word.detected  → Published when wake word is detected
    sensory.audio.stream           → Subscribed (raw audio chunks)

Usage::

    detector = WakeWordDetector(
        wake_words=["hey sentra"],
        backend="openwakeword",
    )
    await detector.start(bus)

    # When "hey sentra" is detected, publishes:
    #   topic=perception.wake_word.detected
    #   payload={"wake_word": "hey sentra", "confidence": 0.92}
"""

from __future__ import annotations

import logging
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────


class WakeWordBackend(StrEnum):
    """Supported wake word detection backends."""

    OPENWAKEWORD = "openwakeword"
    PORCUPINE = "porcupine"
    ENERGY = "energy"  # Simple energy-threshold fallback


@dataclass
class WakeWordConfig:
    """Wake word detector configuration."""

    wake_words: list[str] = field(default_factory=lambda: ["hey sentra"])
    backend: WakeWordBackend = WakeWordBackend.OPENWAKEWORD
    confidence_threshold: float = 0.7
    cooldown_seconds: float = 2.0  # Minimum time between activations
    sample_rate: int = 16000
    frame_length_ms: int = 80  # Frame size for processing (ms)

    # Porcupine-specific
    porcupine_access_key: str = ""
    porcupine_keyword_paths: list[str] = field(default_factory=list)

    # OpenWakeWord-specific
    oww_model_paths: list[str] = field(default_factory=list)
    oww_inference_framework: str = "onnx"

    # Energy-based fallback
    energy_threshold: float = 500.0  # RMS energy threshold
    energy_min_duration_ms: int = 300  # Minimum speech duration


# ── Backend Protocols ────────────────────────────────────────────────────────


class WakeWordEngine(ABC):
    """Abstract base for wake word detection engines."""

    @abstractmethod
    def process_audio(self, pcm_int16: bytes) -> list[WakeWordEvent]:
        """Process a frame of 16-bit PCM audio.

        Returns a list of detection events (empty if none).
        """
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Release resources."""
        ...


@dataclass
class WakeWordEvent:
    """A detected wake word event."""

    wake_word: str
    confidence: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "wake_word": self.wake_word,
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp,
        }


# ── OpenWakeWord Backend ─────────────────────────────────────────────────────


class OpenWakeWordEngine(WakeWordEngine):
    """Wake word detection using OpenWakeWord (ONNX-based).

    Requires: ``pip install openwakeword``
    """

    def __init__(self, config: WakeWordConfig) -> None:
        self._config = config
        self._model: Any | None = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            import openwakeword  # type: ignore[import-untyped]
            from openwakeword.model import Model  # type: ignore[import-untyped]

            # Download default models if no custom paths
            if not self._config.oww_model_paths:
                openwakeword.utils.download_models()

            self._model = Model(
                wakeword_models=self._config.oww_model_paths or None,
                inference_framework=self._config.oww_inference_framework,
            )
            self._loaded = True
            logger.info("OpenWakeWord loaded (words=%s)", self._config.wake_words)
        except ImportError:
            logger.warning("openwakeword not installed. Install with: pip install openwakeword")
        except Exception as e:
            logger.error("Failed to load OpenWakeWord: %s", e)

    def process_audio(self, pcm_int16: bytes) -> list[WakeWordEvent]:
        self._load()
        if self._model is None:
            return []

        try:
            import numpy as np

            # Convert 16-bit PCM to int16 numpy array
            audio = np.frombuffer(pcm_int16, dtype=np.int16)

            # Run prediction
            self._model.predict(audio)

            events: list[WakeWordEvent] = []
            for mdl_name in self._model.prediction_buffer:
                scores = self._model.prediction_buffer[mdl_name]
                if scores and scores[-1] >= self._config.confidence_threshold:
                    # Map model name to wake word
                    wake_word = self._match_wake_word(mdl_name)
                    events.append(
                        WakeWordEvent(
                            wake_word=wake_word,
                            confidence=float(scores[-1]),
                        )
                    )
            return events

        except Exception as e:
            logger.debug("OpenWakeWord processing error: %s", e)
            return []

    def _match_wake_word(self, model_name: str) -> str:
        """Map an OWW model name to a configured wake word."""
        model_lower = model_name.lower().replace("_", " ")
        for ww in self._config.wake_words:
            if ww.lower() in model_lower or model_lower in ww.lower():
                return ww
        return model_name

    def cleanup(self) -> None:
        self._model = None
        self._loaded = False


# ── Porcupine Backend ────────────────────────────────────────────────────────


class PorcupineEngine(WakeWordEngine):
    """Wake word detection using Picovoice Porcupine.

    Requires: ``pip install pvporcupine``
    """

    def __init__(self, config: WakeWordConfig) -> None:
        self._config = config
        self._porcupine: Any | None = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            import pvporcupine  # type: ignore[import-untyped]

            if self._config.porcupine_keyword_paths:
                self._porcupine = pvporcupine.create(
                    access_key=self._config.porcupine_access_key,
                    keyword_paths=self._config.porcupine_keyword_paths,
                    sensitivities=[self._config.confidence_threshold]
                    * len(self._config.porcupine_keyword_paths),
                )
            else:
                # Use built-in keywords
                keywords = [
                    w.lower().replace(" ", "_")
                    for w in self._config.wake_words
                    if w.lower().replace(" ", "_") in pvporcupine.KEYWORDS
                ]
                if not keywords:
                    keywords = ["porcupine"]  # Default keyword
                self._porcupine = pvporcupine.create(
                    access_key=self._config.porcupine_access_key,
                    keywords=keywords,
                    sensitivities=[self._config.confidence_threshold] * len(keywords),
                )
            self._loaded = True
            logger.info("Porcupine loaded (words=%s)", self._config.wake_words)
        except ImportError:
            logger.warning("pvporcupine not installed. Install with: pip install pvporcupine")
        except Exception as e:
            logger.error("Failed to load Porcupine: %s", e)

    def process_audio(self, pcm_int16: bytes) -> list[WakeWordEvent]:
        self._load()
        if self._porcupine is None:
            return []

        try:
            # Porcupine expects a list of int16 samples
            n_samples = len(pcm_int16) // 2
            samples = list(struct.unpack(f"<{n_samples}h", pcm_int16))

            # Process in frame-sized chunks
            frame_length = self._porcupine.frame_length
            events: list[WakeWordEvent] = []

            for i in range(0, len(samples) - frame_length + 1, frame_length):
                frame = samples[i : i + frame_length]
                keyword_index = self._porcupine.process(frame)
                if keyword_index >= 0:
                    ww = (
                        self._config.wake_words[keyword_index]
                        if keyword_index < len(self._config.wake_words)
                        else f"keyword_{keyword_index}"
                    )
                    events.append(
                        WakeWordEvent(
                            wake_word=ww,
                            confidence=1.0,  # Porcupine is binary
                        )
                    )
            return events

        except Exception as e:
            logger.debug("Porcupine processing error: %s", e)
            return []

    def cleanup(self) -> None:
        if self._porcupine is not None:
            try:
                self._porcupine.delete()
            except Exception:
                pass
        self._porcupine = None
        self._loaded = False


# ── Energy-Based Fallback ────────────────────────────────────────────────────


class EnergyWakeWordEngine(WakeWordEngine):
    """Simple energy-threshold wake word detector.

    Uses RMS energy of audio frames to detect speech presence.
    No ML required — useful as a lightweight fallback or for testing.
    """

    def __init__(self, config: WakeWordConfig) -> None:
        self._config = config
        self._speech_frames = 0
        self._min_frames = max(
            1,
            int(config.energy_min_duration_ms / config.frame_length_ms),
        )

    def process_audio(self, pcm_int16: bytes) -> list[WakeWordEvent]:
        import math

        n_samples = len(pcm_int16) // 2
        if n_samples == 0:
            return []

        samples = struct.unpack(f"<{n_samples}h", pcm_int16)
        rms = math.sqrt(sum(s * s for s in samples) / n_samples)

        if rms >= self._config.energy_threshold:
            self._speech_frames += 1
            if self._speech_frames >= self._min_frames:
                self._speech_frames = 0
                confidence = min(1.0, rms / (self._config.energy_threshold * 2))
                return [
                    WakeWordEvent(
                        wake_word=self._config.wake_words[0]
                        if self._config.wake_words
                        else "activation",
                        confidence=confidence,
                    )
                ]
        else:
            self._speech_frames = 0

        return []

    def cleanup(self) -> None:
        self._speech_frames = 0


# ── Wake Word Detector Node ─────────────────────────────────────────────────


class WakeWordDetector(Node):
    """Always-on wake word detection node.

    Subscribes to raw audio stream and publishes activation events
    when a configured wake word is detected.

    Usage::

        detector = WakeWordDetector(
            config=WakeWordConfig(
                wake_words=["hey sentra"],
                backend=WakeWordBackend.OPENWAKEWORD,
            )
        )
        await detector.start(bus)
    """

    def __init__(
        self,
        node_id: str = "wake_word_detector",
        config: WakeWordConfig | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.PERCEPTION,
            capabilities=["wake_word_detection", "hands_free_activation"],
        )
        self.config = config or WakeWordConfig()
        self._engine: WakeWordEngine | None = None
        self._last_activation: float = 0.0
        self._total_detections: int = 0
        self._active = True

    def _create_engine(self) -> WakeWordEngine:
        """Create the appropriate wake word engine."""
        if self.config.backend == WakeWordBackend.OPENWAKEWORD:
            return OpenWakeWordEngine(self.config)
        elif self.config.backend == WakeWordBackend.PORCUPINE:
            return PorcupineEngine(self.config)
        else:
            return EnergyWakeWordEngine(self.config)

    async def on_start(self) -> None:
        """Subscribe to audio stream for wake word detection."""
        self._engine = self._create_engine()
        await self.bus.subscribe("sensory.audio.stream", self._on_audio_stream)
        logger.info(
            "WakeWordDetector started (backend=%s, words=%s)",
            self.config.backend.value,
            self.config.wake_words,
        )

    async def on_stop(self) -> None:
        """Release wake word engine resources."""
        if self._engine:
            self._engine.cleanup()
        self._active = False
        logger.info(
            "WakeWordDetector stopped (total_detections=%d)",
            self._total_detections,
        )

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def _on_audio_stream(self, message: Message) -> None:
        """Process audio stream for wake word detection."""
        if not self._active or self._engine is None:
            return

        payload = message.payload
        chunk_hex = str(payload.get("chunk", ""))
        if not chunk_hex:
            return

        try:
            pcm_bytes = bytes.fromhex(chunk_hex)
        except ValueError:
            return

        # Cooldown check
        now = time.time()
        if now - self._last_activation < self.config.cooldown_seconds:
            return

        # Run detection
        events = self._engine.process_audio(pcm_bytes)

        for event in events:
            self._last_activation = now
            self._total_detections += 1

            logger.info(
                "🎤 Wake word detected: '%s' (confidence=%.2f)",
                event.wake_word,
                event.confidence,
            )

            # Publish activation event
            await self.bus.publish(
                "perception.wake_word.detected",
                Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    tenant_id=message.tenant_id,
                    session_id=message.session_id,
                    topic="perception.wake_word.detected",
                    payload=event.to_dict(),
                ),
            )
            break  # Only one activation per frame

    def set_active(self, active: bool) -> None:
        """Enable/disable wake word detection at runtime."""
        self._active = active
        logger.info("WakeWordDetector active=%s", active)

    def stats(self) -> dict[str, Any]:
        """Return detection statistics."""
        return {
            "backend": self.config.backend.value,
            "wake_words": self.config.wake_words,
            "total_detections": self._total_detections,
            "active": self._active,
            "last_activation": self._last_activation,
        }
