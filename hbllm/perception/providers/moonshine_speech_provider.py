"""Moonshine Speech Provider — HBLLM Audio Perception §A7.

Implements SpeechProvider protocol wrapping Moonshine ONNX (primary),
Whisper (fallback), and NVIDIA Cloud ASR (cloud fallback).

The provider is responsible ONLY for:
    - Model loading (lazy, thread-safe)
    - PCM normalization / resampling
    - Transcription
    - Producing typed SpeechResult

The provider does NOT:
    - Know about HCIR
    - Manage VAD, sessions, or message bus
    - Construct evidence (that's the runtime's job)
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import Sequence
from typing import Any

import numpy as np

from hbllm.perception.providers.audio_types import (
    AudioInput,
    SpeechResult,
    TemporalSpan,
)

logger = logging.getLogger(__name__)


class MoonshineSpeechProvider:
    """SpeechProvider wrapping Moonshine ONNX + Whisper fallback.

    Implements the SpeechProvider protocol from audio_base.py.

    Usage::

        provider = MoonshineSpeechProvider(model_size="base")
        await provider.initialize()
        result = await provider.transcribe(pcm_bytes)

    """

    def __init__(
        self,
        model_size: str = "base",
        target_sample_rate: int = 16000,
    ) -> None:
        self._model_size = model_size
        self._target_sr = target_sample_rate

        # Lazy-loaded models
        self._moonshine_model: Any | None = None
        self._whisper_model: Any | None = None
        self._model_lock = threading.Lock()
        self._initialized = False

    # ── PerceptionProvider protocol ──────────────────────────────────────

    @property
    def modality(self) -> str:
        return "audio"

    @property
    def provider_id(self) -> str:
        return f"moonshine:{self._model_size}"

    @property
    def sample_rate(self) -> int:
        return self._target_sr

    async def initialize(self) -> None:
        """Pre-load the Moonshine model."""
        await asyncio.to_thread(self._load_moonshine)
        self._initialized = True

    async def shutdown(self) -> None:
        """Release model resources."""
        self._moonshine_model = None
        self._whisper_model = None
        self._initialized = False

    # ── SpeechProvider protocol ──────────────────────────────────────────

    async def transcribe(self, audio: AudioInput) -> SpeechResult:
        """Transcribe audio input into a SpeechResult.

        Args:
            audio: Raw audio as bytes (16-bit PCM), file path, or numpy array.

        Returns:
            SpeechResult with transcript and confidence.

        """
        import time

        t0 = time.monotonic()

        # Convert to float32 numpy array
        samples = self._to_float32(audio)

        # Resample if needed
        if hasattr(audio, "__len__") and isinstance(audio, bytes):
            # PCM bytes assume target sample rate
            pass

        # Normalize
        samples = self._normalize_audio(samples)

        if len(samples) < 1600:  # < 0.1s at 16kHz
            return SpeechResult(transcript="", confidence=0.0)

        # Transcribe with fallback chain
        transcript = await self._transcribe_samples(samples)

        elapsed = time.monotonic() - t0
        # Estimate confidence from transcript quality
        confidence = self._estimate_confidence(transcript, samples, elapsed)

        return SpeechResult(
            transcript=transcript,
            language="en",
            confidence=confidence,
            temporal=TemporalSpan(
                start_time=time.time() - elapsed,
                end_time=time.time(),
                duration=len(samples) / self._target_sr,
            ),
        )

    async def transcribe_streaming(
        self,
        audio_chunks: Sequence[AudioInput],
    ) -> SpeechResult:
        """Transcribe an accumulated sequence of audio chunks."""
        if not audio_chunks:
            return SpeechResult(transcript="", confidence=0.0)

        # Convert each chunk and concatenate
        arrays = [self._to_float32(c) for c in audio_chunks]
        non_empty = [a for a in arrays if len(a) > 0]
        if not non_empty:
            return SpeechResult(transcript="", confidence=0.0)

        combined = np.concatenate(non_empty)
        return await self.transcribe(combined)

    # ── Model Loading ────────────────────────────────────────────────────

    def _load_moonshine(self) -> None:
        """Load Moonshine ONNX model (thread-safe)."""
        if self._moonshine_model is not None:
            return
        with self._model_lock:
            if self._moonshine_model is not None:
                return
            try:
                from moonshine_onnx import MoonshineOnnxModel  # type: ignore[import-untyped]

                logger.info("Loading Moonshine %s ASR model...", self._model_size)
                self._moonshine_model = MoonshineOnnxModel(model_name=self._model_size)
                logger.info("Moonshine ASR loaded successfully")
            except ImportError:
                logger.warning(
                    "moonshine-onnx not installed. Install with: pip install useful-moonshine-onnx"
                )
            except Exception:
                logger.exception("Failed to load Moonshine model")

    def _load_whisper(self) -> None:
        """Load Whisper model (deprecated fallback)."""
        if self._whisper_model is not None:
            return
        try:
            import whisper  # type: ignore[import-untyped]

            logger.info("Loading Whisper %s model (fallback)...", self._model_size)
            self._whisper_model = whisper.load_model(self._model_size)
        except ImportError:
            logger.warning("Whisper library not found.")

    # ── Audio Processing ─────────────────────────────────────────────────

    def _to_float32(self, audio: AudioInput) -> np.ndarray:
        """Convert any AudioInput to float32 numpy array."""
        if isinstance(audio, np.ndarray):
            if audio.dtype == np.float32:
                return audio
            if audio.dtype == np.int16:
                return audio.astype(np.float32) / 32768.0
            return audio.astype(np.float32)

        if isinstance(audio, bytes):
            return np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        if isinstance(audio, (str, os.PathLike)):
            try:
                import soundfile as sf  # type: ignore[import-not-found]

                data, sr = sf.read(str(audio), dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                if sr != self._target_sr:
                    data = self._resample(data, sr)
                return data.astype(np.float32)
            except ImportError:
                logger.warning("soundfile not installed for file-based transcription")
                return np.zeros(0, dtype=np.float32)

        return np.zeros(0, dtype=np.float32)

    def _normalize_audio(self, samples: np.ndarray) -> np.ndarray:
        """Trim silence and normalize peak amplitude."""
        if len(samples) == 0:
            return samples

        # Trim leading/trailing silence (below 2% amplitude)
        threshold = 0.02
        above = np.where(np.abs(samples) > threshold)[0]
        if len(above) > 0:
            pad = int(0.1 * self._target_sr)  # 0.1s padding
            start = max(0, above[0] - pad)
            end = min(len(samples), above[-1] + pad)
            samples = samples[start:end]

        # Normalize to peak=0.95
        peak = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0
        if peak > 0.001:
            samples = samples * (0.95 / peak)

        return samples

    def _resample(self, samples: np.ndarray, from_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if from_sr == self._target_sr:
            return samples
        try:
            import scipy.signal  # type: ignore[import-untyped]

            new_length = int(len(samples) * self._target_sr / from_sr)
            return scipy.signal.resample(samples, new_length).astype(np.float32)
        except ImportError:
            # Linear interpolation fallback
            ratio = self._target_sr / from_sr
            new_length = int(len(samples) * ratio)
            indices = np.linspace(0, len(samples) - 1, new_length)
            return np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)

    # ── Transcription Backends ───────────────────────────────────────────

    async def _transcribe_samples(self, samples: np.ndarray) -> str:
        """Transcribe float32 samples using fallback chain."""
        # Try Moonshine first
        result = await asyncio.to_thread(self._transcribe_moonshine, samples)
        if result:
            return result

        # Fallback to NVIDIA Cloud
        nvidia_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NVIDIA_NIM_API_KEY")
        if nvidia_key:
            try:
                cloud_result = await self._transcribe_nvidia_pcm(samples, nvidia_key)
                if cloud_result:
                    return cloud_result
            except Exception:
                logger.warning("NVIDIA Cloud ASR fallback failed", exc_info=True)

        # Fallback to Whisper
        result = await asyncio.to_thread(self._transcribe_whisper, samples)
        return result or ""

    def _transcribe_moonshine(self, samples: np.ndarray) -> str:
        """Transcribe with Moonshine ONNX."""
        self._load_moonshine()
        if self._moonshine_model is None:
            return ""

        samples_2d = samples.reshape(1, -1)
        token_ids = self._moonshine_model.generate(samples_2d)

        from moonshine_onnx import load_tokenizer  # type: ignore[import-untyped]

        tokenizer = load_tokenizer()
        texts = tokenizer.decode_batch(token_ids)
        return texts[0].strip() if texts else ""

    def _transcribe_whisper(self, samples: np.ndarray) -> str:
        """Transcribe with Whisper (deprecated fallback)."""
        self._load_whisper()
        if self._whisper_model is None:
            return ""

        import tempfile
        import wave

        # Write to temp WAV file
        pcm_bytes = (samples * 32768.0).astype(np.int16).tobytes()
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            with wave.open(tmp.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._target_sr)
                wf.writeframes(pcm_bytes)
            result = self._whisper_model.transcribe(tmp.name, fp16=False)
            return str(result["text"]).strip()
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    async def _transcribe_nvidia_pcm(
        self,
        samples: np.ndarray,
        api_key: str,
    ) -> str | None:
        """Transcribe via NVIDIA Cloud Whisper API."""
        import tempfile
        import wave

        import httpx

        nvidia_url = os.getenv(
            "NVIDIA_ASR_URL",
            "https://integrate.api.nvidia.com/v1/audio/transcriptions",
        )

        pcm_bytes = (samples * 32768.0).astype(np.int16).tobytes()
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            with wave.open(tmp.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._target_sr)
                wf.writeframes(pcm_bytes)

            async with httpx.AsyncClient(timeout=60) as client:
                headers = {"Authorization": f"Bearer {api_key}"}
                with open(tmp.name, "rb") as f:
                    files = {"file": ("audio.wav", f, "audio/wav")}
                    data = {"model": "openai/whisper-large-v3"}
                    resp = await client.post(nvidia_url, headers=headers, files=files, data=data)
                    resp.raise_for_status()
                    return str(resp.json().get("text", "")).strip()
        except Exception as e:
            logger.warning("NVIDIA Cloud ASR failed: %s", e)
            return None
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    # ── Confidence Estimation ────────────────────────────────────────────

    def _estimate_confidence(
        self,
        transcript: str,
        samples: np.ndarray,
        elapsed: float,
    ) -> float:
        """Estimate transcription confidence from heuristics."""
        if not transcript:
            return 0.0

        confidence = 0.7  # Base confidence for successful transcription

        # Boost for longer transcripts (more context = more reliable)
        word_count = len(transcript.split())
        if word_count >= 5:
            confidence += 0.1
        elif word_count <= 1:
            confidence -= 0.1

        # Penalize if transcription took very long relative to audio
        audio_duration = len(samples) / self._target_sr
        if elapsed > audio_duration * 2:
            confidence -= 0.1

        # Penalize for very low energy audio
        rms = float(np.sqrt(np.mean(samples**2)))
        if rms < 0.01:
            confidence -= 0.2

        return max(0.0, min(1.0, confidence))
