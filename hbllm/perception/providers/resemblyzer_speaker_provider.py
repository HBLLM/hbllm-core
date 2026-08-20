"""Resemblyzer Speaker Provider — HBLLM Audio Perception §A7.

Implements SpeakerProvider protocol wrapping the Resemblyzer GE2E
voice encoder for speaker identification and verification.

The provider produces:
    - SpeakerIdentification (structured, with embedding_ref, confidence)
    - AudioEmbedding (256-dim L2-normalized voice embedding)

The provider does NOT:
    - Manage voice profile storage (that's VoiceProfileStore)
    - Know about HCIR or message bus
    - Arbitrate between speaker hypotheses
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

import numpy as np

from hbllm.perception.providers.audio_types import (
    AudioEmbedding,
    AudioInput,
    SpeakerIdentification,
)

logger = logging.getLogger(__name__)


class ResemblyzerSpeakerProvider:
    """SpeakerProvider wrapping Resemblyzer GE2E encoder.

    Implements the SpeakerProvider protocol from audio_base.py.

    Usage::

        provider = ResemblyzerSpeakerProvider()
        await provider.initialize()
        speaker = await provider.identify(audio_bytes)
        embedding = await provider.embed_voice(audio_bytes)

    """

    # Cosine similarity threshold for positive identification
    DEFAULT_THRESHOLD = 0.75
    EMBEDDING_DIM = 256

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        target_sample_rate: int = 16000,
    ) -> None:
        self._threshold = threshold
        self._target_sr = target_sample_rate
        self._encoder: Any | None = None
        self._encoder_lock = asyncio.Lock()

        # In-memory enrolled profiles: speaker_id → normalized embedding
        self._enrolled: dict[str, np.ndarray] = {}

    # ── PerceptionProvider protocol ──────────────────────────────────────

    @property
    def modality(self) -> str:
        return "audio"

    @property
    def provider_id(self) -> str:
        return "resemblyzer:ge2e-256"

    @property
    def sample_rate(self) -> int:
        return self._target_sr

    async def initialize(self) -> None:
        """Load the Resemblyzer encoder (~50MB)."""
        async with self._encoder_lock:
            if self._encoder is not None:
                return
            try:
                from resemblyzer import VoiceEncoder  # type: ignore[import-untyped]

                self._encoder = await asyncio.to_thread(
                    VoiceEncoder, "cpu",
                )
                logger.info("Resemblyzer GE2E encoder loaded")
            except ImportError:
                logger.warning(
                    "resemblyzer not installed. Speaker identification disabled. "
                    "Install with: pip install resemblyzer"
                )
            except Exception:
                logger.exception("Failed to load Resemblyzer encoder")

    async def shutdown(self) -> None:
        """Release encoder resources."""
        self._encoder = None

    # ── SpeakerProvider protocol ─────────────────────────────────────────

    async def identify(
        self,
        audio: AudioInput,
    ) -> SpeakerIdentification:
        """Identify the speaker from audio.

        Returns a structured SpeakerIdentification with the best match
        from enrolled profiles, or an unknown speaker with embedding ref.
        """
        embedding = await self.embed_voice(audio)

        if not self._enrolled:
            return SpeakerIdentification(
                speaker_id=None,
                embedding_ref=self._embedding_ref(embedding),
                confidence=0.0,
                is_enrolled=False,
            )

        # Compare against all enrolled speakers
        query = np.array(embedding.vector, dtype=np.float32)
        best_id: str | None = None
        best_score = 0.0

        for speaker_id, enrolled_emb in self._enrolled.items():
            score = float(np.dot(query, enrolled_emb))
            if score > best_score:
                best_score = score
                best_id = speaker_id

        if best_id is not None and best_score >= self._threshold:
            return SpeakerIdentification(
                speaker_id=best_id,
                embedding_ref=self._embedding_ref(embedding),
                confidence=best_score,
                is_enrolled=True,
            )

        return SpeakerIdentification(
            speaker_id=None,
            embedding_ref=self._embedding_ref(embedding),
            confidence=best_score,
            is_enrolled=False,
        )

    async def embed_voice(self, audio: AudioInput) -> AudioEmbedding:
        """Extract a 256-dim L2-normalized voice embedding.

        If the encoder is not available, returns a deterministic
        hash-based embedding for graceful degradation.
        """
        samples = self._to_float32(audio)

        if self._encoder is not None and len(samples) > 0:
            embedding = await asyncio.to_thread(
                self._encoder.embed_utterance, samples,
            )
            # L2 normalize
            norm = float(np.linalg.norm(embedding))
            if norm > 0:
                embedding = embedding / norm
            return AudioEmbedding(
                vector=embedding.tolist(),
                model_id="resemblyzer-ge2e",
                space_id="ge2e-256",
                dimensions=self.EMBEDDING_DIM,
                sample_rate=self._target_sr,
            )

        # Fallback: deterministic hash-based embedding
        return self._hash_embedding(samples)

    async def enroll(
        self,
        speaker_id: str,
        audio_samples: list[AudioInput],
    ) -> SpeakerIdentification:
        """Enroll a speaker from multiple audio samples.

        Averages embeddings from all samples for robustness.
        """
        embeddings: list[np.ndarray] = []
        for sample in audio_samples:
            emb = await self.embed_voice(sample)
            embeddings.append(np.array(emb.vector, dtype=np.float32))

        if not embeddings:
            return SpeakerIdentification(speaker_id=speaker_id, confidence=0.0)

        # Average and L2-normalize
        avg = np.mean(embeddings, axis=0)
        norm = float(np.linalg.norm(avg))
        if norm > 0:
            avg = avg / norm

        self._enrolled[speaker_id] = avg.astype(np.float32)

        return SpeakerIdentification(
            speaker_id=speaker_id,
            embedding_ref=f"enrolled:{speaker_id}",
            confidence=1.0,
            is_enrolled=True,
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

    def _hash_embedding(self, samples: np.ndarray) -> AudioEmbedding:
        """Deterministic hash-based embedding fallback."""
        data = samples.tobytes() if len(samples) > 0 else b"empty"
        h = hashlib.sha256(data).digest()
        vector = [float(b) / 255.0 for b in h[:self.EMBEDDING_DIM]]
        # Pad to full dimension
        while len(vector) < self.EMBEDDING_DIM:
            vector.append(0.0)
        # L2 normalize
        norm = sum(v * v for v in vector) ** 0.5
        if norm > 0:
            vector = [v / norm for v in vector]
        return AudioEmbedding(
            vector=vector[:self.EMBEDDING_DIM],
            model_id="hash-fallback",
            space_id="hash-256",
            dimensions=self.EMBEDDING_DIM,
            sample_rate=self._target_sr,
        )

    @staticmethod
    def _embedding_ref(embedding: AudioEmbedding) -> str:
        """Generate a reference ID for an embedding."""
        data = str(embedding.vector[:8]).encode()
        return f"voice_{hashlib.md5(data).hexdigest()[:12]}"  # noqa: S324
