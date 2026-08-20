"""Mock Audio Provider — deterministic testing provider for audio perception.

Produces deterministic, repeatable results for unit testing.
Same pattern as MockVisionProvider.

Usage::

    provider = MockAudioProvider()
    result = await provider.transcribe(audio_bytes)
    # result.transcript = hash-based deterministic text
    # result.confidence = hash-based deterministic value
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

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

# Deterministic word pool for transcript generation
_WORD_POOL = [
    "hello", "world", "turn", "on", "off", "the", "light",
    "what", "is", "time", "play", "music", "stop", "open",
    "door", "set", "alarm", "call", "home", "check",
]

# Deterministic event pool
_EVENT_POOL = [
    "silence", "speech", "doorbell", "knock", "alarm",
    "dog_bark", "vehicle", "footsteps", "music", "appliance",
]

# Deterministic scene tags
_SCENE_TAGS = [
    "indoor", "outdoor", "quiet", "noisy", "residential",
    "urban", "natural", "mechanical",
]


def _hash_audio(audio: AudioInput) -> str:
    """Generate deterministic hash from audio input."""
    if isinstance(audio, bytes):
        data = audio
    elif isinstance(audio, (str, Path)):
        data = str(audio).encode("utf-8")
    elif isinstance(audio, np.ndarray):
        data = audio.tobytes()
    else:
        data = str(audio).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


class MockAudioProvider:
    """Deterministic audio provider for testing.

    All outputs are derived from a SHA-256 hash of the input,
    ensuring deterministic, repeatable results.

    Implements SpeechProvider, AcousticEventProvider,
    AcousticSceneProvider, and SpeakerProvider protocols.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        sample_rate: int = 16000,
    ) -> None:
        self._embedding_dim = embedding_dim
        self._sample_rate = sample_rate

    # ── PerceptionProvider ──────────────────────────────────────────────

    @property
    def provider_id(self) -> str:
        """Unique provider identifier."""
        return "mock-audio-v1"

    @property
    def modality(self) -> str:
        """Perception modality."""
        return "audio"

    @property
    def sample_rate(self) -> int:
        """Expected sample rate."""
        return self._sample_rate

    async def initialize(self) -> None:
        """No-op for mock."""

    async def shutdown(self) -> None:
        """No-op for mock."""

    # ── Embedding ───────────────────────────────────────────────────────

    def _hash_to_embedding(self, h: str) -> AudioEmbedding:
        """Deterministic embedding from hash."""
        seed = int(h[:8], 16)
        rng = np.random.RandomState(seed)  # noqa: NPY002
        vec = rng.randn(self._embedding_dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return AudioEmbedding(
            vector=vec.tolist(),
            model_id=self.provider_id,
            space_id="mock-audio-space",
            dimensions=self._embedding_dim,
            sample_rate=self._sample_rate,
        )

    # ── SpeechProvider ──────────────────────────────────────────────────

    async def transcribe(self, audio: AudioInput) -> SpeechResult:
        """Deterministic transcription from audio hash."""
        h = _hash_audio(audio)
        seed = int(h[:8], 16)
        rng = np.random.RandomState(seed)  # noqa: NPY002

        # Generate deterministic transcript
        n_words = 3 + (seed % 5)
        words = [_WORD_POOL[rng.randint(len(_WORD_POOL))] for _ in range(n_words)]
        transcript = " ".join(words)

        now = time.time()
        duration = 0.5 + (seed % 30) / 10.0

        return SpeechResult(
            transcript=transcript,
            language="en",
            confidence=0.5 + (seed % 50) / 100.0,
            speaker=SpeakerIdentification(
                speaker_id=f"speaker_{seed % 5}",
                confidence=0.6 + (seed % 40) / 100.0,
                is_enrolled=seed % 3 == 0,
                embedding_ref=f"spk_{h[:8]}",
            ),
            paralinguistic=ParalinguisticProfile(
                tone=["neutral", "urgent", "calm", "excited"][seed % 4],
                confidence=0.4 + (seed % 60) / 100.0,
                pitch_mean=100.0 + seed % 200,
                speech_rate=100.0 + seed % 100,
            ),
            temporal=TemporalSpan(
                start_time=now,
                end_time=now + duration,
                duration=duration,
                state=AudioEventState.INSTANTANEOUS,
            ),
        )

    async def transcribe_streaming(
        self,
        audio_chunks: Sequence[AudioInput],
    ) -> SpeechResult:
        """Transcribe by combining chunk hashes."""
        combined = b"".join(
            _hash_audio(chunk).encode() for chunk in audio_chunks
        )
        return await self.transcribe(combined)

    # ── AcousticEventProvider ───────────────────────────────────────────

    async def classify(self, audio: AudioInput) -> list[SoundEventResult]:
        """Deterministic event classification."""
        h = _hash_audio(audio)
        seed = int(h[:8], 16)

        event_idx = seed % len(_EVENT_POOL)
        event_type = _EVENT_POOL[event_idx]
        confidence = 0.5 + (seed % 50) / 100.0

        now = time.time()
        return [
            SoundEventResult(
                event_type=event_type,
                confidence=confidence,
                is_critical=event_type in ("alarm", "glass_breaking"),
                temporal=TemporalSpan(
                    start_time=now,
                    end_time=now + 0.5,
                    duration=0.5,
                ),
                top_classes=[
                    (event_type, confidence),
                    (_EVENT_POOL[(event_idx + 1) % len(_EVENT_POOL)], 0.1),
                ],
            ),
        ]

    # ── AcousticSceneProvider ───────────────────────────────────────────

    async def analyze_scene(self, audio: AudioInput) -> AcousticSceneResult:
        """Deterministic scene analysis."""
        h = _hash_audio(audio)
        seed = int(h[:8], 16)

        return AcousticSceneResult(
            indoor=seed % 2 == 0,
            speech_present=seed % 3 == 0,
            noise_level=(seed % 100) / 100.0,
            estimated_activity=(seed % 80) / 100.0,
            scene_tags=[
                _SCENE_TAGS[seed % len(_SCENE_TAGS)],
                _SCENE_TAGS[(seed + 1) % len(_SCENE_TAGS)],
            ],
        )

    # ── SpeakerProvider ─────────────────────────────────────────────────

    async def identify(self, audio: AudioInput) -> SpeakerIdentification:
        """Deterministic speaker identification."""
        h = _hash_audio(audio)
        seed = int(h[:8], 16)

        return SpeakerIdentification(
            speaker_id=f"speaker_{seed % 5}",
            confidence=0.6 + (seed % 40) / 100.0,
            is_enrolled=seed % 3 == 0,
            embedding_ref=f"spk_{h[:8]}",
            voice_characteristics={
                "pitch_hz": 100.0 + seed % 200,
            },
        )

    async def enroll(self, speaker_id: str, audio: AudioInput) -> bool:
        """Always succeeds for mock."""
        return True

    # ── SoundLocalizationProvider ───────────────────────────────────────

    async def localize(self, audio: AudioInput) -> SoundLocalizationResult:
        """Deterministic localization."""
        h = _hash_audio(audio)
        seed = int(h[:8], 16)

        return SoundLocalizationResult(
            direction_degrees=float(seed % 360),
            distance_estimate=1.0 + (seed % 100) / 10.0,
            confidence=0.5 + (seed % 50) / 100.0,
        )

    # ── Compatibility ───────────────────────────────────────────────────

    def is_compatible(self, other_provider_id: str) -> bool:
        """Only compatible with other mock providers."""
        return other_provider_id == self.provider_id

    def get_embedding(self, audio: AudioInput) -> AudioEmbedding:
        """Get embedding synchronously (for testing)."""
        return self._hash_to_embedding(_hash_audio(audio))
