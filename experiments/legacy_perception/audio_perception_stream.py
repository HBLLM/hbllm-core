"""Audio Perception Stream — SNN-gated continuous audio perception.

Processes incoming audio chunks through a lightweight SNN gate
before spending expensive compute on speech-to-text, sound event
classification, or scene analysis.

Architecture:
    AudioChunk → AudioSignals → AudioPerceptionEnsemble
                                        │
                         ┌──────────────┴──────────────┐
                         ▼ (should_process=False)      ▼ (should_process=True)
                        Skip                        AudioPerceptionRuntime
                                                               │
                                                               ▼
                                                        AudioAssessment
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from hbllm.brain.snn.perception.audio_ensemble import AudioPerceptionEnsemble
from hbllm.brain.snn.perception.audio_signals import extract_audio_signals
from hbllm.brain.snn.perception.gate import PerceptionGateDecision
from hbllm.perception.audio_perception_runtime import AudioPerceptionRuntime
from hbllm.perception.providers.audio_evidence import AudioAssessment

logger = logging.getLogger(__name__)


class AudioPerceptionStream:
    """SNN-gated audio perception stream processor.

    Usage::

        stream = AudioPerceptionStream(
            runtime=runtime,
            sample_rate=16000,
        )

        for chunk in audio_chunks:
            decision, assessment = await stream.process_chunk(chunk)
            if assessment:
                print(f"Perception result: {assessment}")
    """

    def __init__(
        self,
        runtime: AudioPerceptionRuntime | None = None,
        ensemble: AudioPerceptionEnsemble | None = None,
        sample_rate: int = 16000,
        on_assessment: Callable[[AudioAssessment, PerceptionGateDecision], Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self.ensemble = ensemble or AudioPerceptionEnsemble()
        self.sample_rate = sample_rate
        self.on_assessment = on_assessment

        self._chunk_index = 0
        self._prev_spectrum: np.ndarray | None = None
        self._chunks_processed = 0
        self._chunks_analyzed = 0

    async def process_chunk(
        self,
        audio: np.ndarray | bytes,
    ) -> tuple[PerceptionGateDecision, AudioAssessment | None]:
        """Process an audio chunk through SNN gate and optional runtime.

        Args:
            audio: Audio buffer as float32 numpy array or bytes.

        Returns:
            Tuple of (gate decision, optional AudioAssessment).
        """
        self._chunk_index += 1
        self._chunks_processed += 1

        # Convert bytes to numpy if needed
        if isinstance(audio, bytes):
            audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio_array = audio.astype(np.float32)

        # 1. Extract cheap features
        signals = extract_audio_signals(
            audio_array,
            sample_rate=self.sample_rate,
            prev_spectrum=self._prev_spectrum,
        )

        # Update previous spectrum
        n_fft = min(2048, audio_array.size)
        if n_fft > 0:
            self._prev_spectrum = np.abs(np.fft.rfft(audio_array[:n_fft] * np.hanning(n_fft)))

        # 2. Step SNN ensemble
        decision = self.ensemble.step(
            signals,
            dt_s=len(audio_array) / self.sample_rate if len(audio_array) > 0 else 0.05,
            sample_index=self._chunk_index,
        )

        assessment: AudioAssessment | None = None

        # 3. If SNN gates open, execute perception
        if decision.should_process and self.runtime is not None:
            self._chunks_analyzed += 1
            assessment = await self.runtime.perceive(audio_array)

            if self.on_assessment is not None:
                try:
                    res = self.on_assessment(assessment, decision)
                    if hasattr(res, "__await__"):
                        await res
                except Exception:
                    logger.exception("Error in on_assessment callback")

        return decision, assessment

    @property
    def stats(self) -> dict[str, Any]:
        """Processing statistics."""
        return {
            "chunks_processed": self._chunks_processed,
            "chunks_analyzed": self._chunks_analyzed,
            "skip_rate": (
                1.0 - (self._chunks_analyzed / self._chunks_processed)
                if self._chunks_processed > 0
                else 0.0
            ),
        }

    def reset(self) -> None:
        """Reset internal stream state."""
        self.ensemble.reset()
        self._chunk_index = 0
        self._prev_spectrum = None
        self._chunks_processed = 0
        self._chunks_analyzed = 0
