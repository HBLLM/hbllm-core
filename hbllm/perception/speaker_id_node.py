"""
Speaker Identification Node — Real-time voice recognition using Resemblyzer.

Extracts 256-dim GE2E speaker embeddings from audio and identifies speakers
against enrolled voice profiles. Supports enrollment, identification, and
profile management.

When an unknown speaker is detected, their embedding is cached. If that
speaker is later enrolled, cached unknowns are retroactively matched and
a `speaker.retroactive_update` event is published to re-tag old messages.

Topics:
    speaker.identify            — Identify who is speaking from PCM audio
    speaker.enroll              — Enroll a new speaker from audio samples
    speaker.list                — List enrolled speakers for a tenant
    speaker.delete              — Delete a speaker profile
    speaker.retroactive_update  — (published) when old unknowns match a new enrollment
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node
from hbllm.perception.voice_profile_store import VoiceProfileStore

logger = logging.getLogger(__name__)


@dataclass
class UnknownEmbedding:
    """Cached embedding for an unidentified speaker, for retroactive matching."""

    embedding: np.ndarray
    session_id: str = ""
    timestamp: float = 0.0
    text: str = ""


class SpeakerIdNode(Node):
    """
    Real-time speaker identification using Resemblyzer GE2E encoder.

    Lazily loads the Resemblyzer encoder (~50MB) on first use.
    Embeddings are 256-dim float32 vectors compared via cosine similarity.

    The node maintains a per-tenant cache of embeddings for fast lookup.
    """

    node_id = "speaker_id"
    description = "Speaker identification and enrollment via voice embeddings"

    # Max unknown embeddings to cache per tenant before evicting oldest
    MAX_UNKNOWN_CACHE = 200
    # How long to keep unknown embeddings (seconds) — 1 hour
    UNKNOWN_TTL = 3600.0

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._encoder: Any | None = None
        self._encoder_lock = asyncio.Lock()
        self._store: VoiceProfileStore | None = None
        # Cache: tenant_id → {speaker_id: embedding}
        self._embedding_cache: dict[str, dict[str, np.ndarray]] = {}
        # Cache of unknown speaker embeddings for retroactive matching
        # tenant_id → list of UnknownEmbedding
        self._unknown_cache: dict[str, list[UnknownEmbedding]] = {}

    def _get_store(self) -> VoiceProfileStore:
        """Lazy-init the voice profile store."""
        if self._store is None:
            data_dir = os.environ.get("HBLLM_DATA_DIR", "./data")
            db_path = os.path.join(data_dir, "voice_profiles.db")
            self._store = VoiceProfileStore(db_path)
        return self._store

    def _load_encoder(self) -> None:
        """Load the Resemblyzer encoder (lazy, thread-safe)."""
        if self._encoder is not None:
            return

        try:
            from resemblyzer import VoiceEncoder  # type: ignore[import-untyped]

            self._encoder = VoiceEncoder("cpu")
            logger.info("Resemblyzer VoiceEncoder loaded (CPU)")
        except ImportError:
            logger.error("resemblyzer not installed. Run: pip install resemblyzer")
            raise
        except Exception:
            logger.exception("Failed to load Resemblyzer encoder")
            raise

    def _extract_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract a 256-dim speaker embedding from audio.

        Args:
            audio: 1D float32 numpy array (mono, any sample rate)
            sample_rate: Sample rate of the audio

        Returns:
            256-dim float32 numpy array
        """
        self._load_encoder()

        # Resemblyzer expects 16kHz audio
        if sample_rate != 16000:
            from scipy.signal import resample  # type: ignore[import-untyped]

            audio = resample(audio, int(len(audio) * 16000 / sample_rate)).astype(np.float32)

        # Preprocess: remove silence, normalize
        from resemblyzer import preprocess_wav  # type: ignore[import-untyped]

        processed = preprocess_wav(audio, source_sr=16000)

        if len(processed) < 1600:  # Less than 0.1s
            raise ValueError("Audio too short for speaker embedding")

        # Extract embedding
        embedding = self._encoder.embed_utterance(processed)
        return embedding.astype(np.float32)

    async def start(self) -> None:
        """Subscribe to speaker identification topics."""
        await self.bus.subscribe("speaker.identify", self.handle_identify)
        await self.bus.subscribe("speaker.enroll", self.handle_enroll)
        await self.bus.subscribe("speaker.list", self.handle_list)
        await self.bus.subscribe("speaker.delete", self.handle_delete)
        logger.info("SpeakerIdNode started")

    async def handle_message(self, message: Message) -> Message | None:
        """Route messages to appropriate handlers."""
        topic = message.topic
        if topic == "speaker.identify":
            return await self.handle_identify(message)
        if topic == "speaker.enroll":
            return await self.handle_enroll(message)
        if topic == "speaker.list":
            return await self.handle_list(message)
        if topic == "speaker.delete":
            return await self.handle_delete(message)
        return None

    async def handle_identify(self, message: Message) -> Message | None:
        """
        Identify a speaker from PCM audio.

        Payload:
            pcm_bytes: bytes — raw 16-bit PCM audio
            sample_rate: int — audio sample rate (default 16000)
            audio_float32: list[float] — alternative: pre-converted float32 samples

        Returns:
            speaker_id, speaker_name, confidence
        """
        payload = message.payload
        tenant_id = message.tenant_id or "default"

        try:
            # Get audio data
            audio = self._payload_to_audio(payload)
            if audio is None or len(audio) < 1600:
                return message.create_response(
                    {"speaker_id": "unknown", "speaker_name": "", "confidence": 0.0}
                )

            sample_rate = int(payload.get("sample_rate", 16000))

            # Extract embedding in thread (CPU-bound)
            embedding = await asyncio.to_thread(self._extract_embedding, audio, sample_rate)

            # Identify against enrolled profiles
            store = self._get_store()
            speaker_id, speaker_name, confidence = await asyncio.to_thread(
                store.identify, tenant_id, embedding
            )

            # If identified with high confidence, incrementally update profile
            if speaker_id != "unknown" and confidence > 0.85:
                await asyncio.to_thread(
                    store.update_embedding, tenant_id, speaker_id, embedding, weight=0.05
                )

            # Cache unknown embeddings for retroactive matching on future enrollment
            if speaker_id == "unknown":
                self._cache_unknown(
                    tenant_id,
                    embedding,
                    session_id=message.session_id or "",
                    timestamp=time.time(),
                    text=payload.get("transcription_text", ""),
                )

            logger.info(
                "Speaker identified: %s (%s) confidence=%.3f tenant=%s",
                speaker_id,
                speaker_name,
                confidence,
                tenant_id,
            )

            return message.create_response(
                {
                    "speaker_id": speaker_id,
                    "speaker_name": speaker_name,
                    "confidence": confidence,
                }
            )

        except Exception as e:
            logger.warning("Speaker identification failed: %s", e)
            return message.create_response(
                {"speaker_id": "unknown", "speaker_name": "", "confidence": 0.0}
            )

    async def handle_enroll(self, message: Message) -> Message | None:
        """
        Enroll a new speaker from audio samples.

        Payload:
            speaker_id: str — unique ID for the speaker
            speaker_name: str — display name
            pcm_bytes: bytes — raw 16-bit PCM audio (at least 3s recommended)
            sample_rate: int — audio sample rate (default 16000)
            audio_samples: list[bytes] — alternative: multiple PCM samples

        Returns:
            success: bool, speaker_id: str, enrollment_samples: int
        """
        payload = message.payload
        tenant_id = message.tenant_id or "default"
        speaker_id = str(payload.get("speaker_id", ""))
        speaker_name = str(payload.get("speaker_name", speaker_id))

        if not speaker_id:
            return message.create_error("speaker_id is required")

        try:
            sample_rate = int(payload.get("sample_rate", 16000))
            embeddings: list[np.ndarray] = []

            # Support single audio or multiple samples
            audio_samples = payload.get("audio_samples", [])
            if audio_samples:
                for sample_bytes in audio_samples:
                    if isinstance(sample_bytes, str):
                        sample_bytes = bytes.fromhex(sample_bytes)
                    audio = np.frombuffer(sample_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    emb = await asyncio.to_thread(self._extract_embedding, audio, sample_rate)
                    embeddings.append(emb)
            else:
                audio = self._payload_to_audio(payload)
                if audio is None or len(audio) < 16000:  # At least 1 second
                    return message.create_error("Need at least 1 second of audio for enrollment")
                emb = await asyncio.to_thread(self._extract_embedding, audio, sample_rate)
                embeddings.append(emb)

            if not embeddings:
                return message.create_error("No valid audio samples for enrollment")

            # Average all embeddings and normalize
            avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)
            avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)

            # Store in database
            store = self._get_store()
            await asyncio.to_thread(
                store.save_profile,
                tenant_id,
                speaker_id,
                speaker_name,
                avg_embedding,
                len(embeddings),
            )

            # Invalidate cache
            self._embedding_cache.pop(tenant_id, None)

            logger.info(
                "Speaker enrolled: %s (%s) with %d samples, tenant=%s",
                speaker_id,
                speaker_name,
                len(embeddings),
                tenant_id,
            )

            # Retroactively match cached unknown embeddings
            retro_matches = await self._retroactive_match(
                tenant_id, speaker_id, speaker_name, avg_embedding
            )

            return message.create_response(
                {
                    "success": True,
                    "speaker_id": speaker_id,
                    "speaker_name": speaker_name,
                    "enrollment_samples": len(embeddings),
                    "retroactive_matches": retro_matches,
                }
            )

        except Exception as e:
            logger.exception("Speaker enrollment failed: %s", e)
            return message.create_error(f"Enrollment failed: {e}")

    async def handle_list(self, message: Message) -> Message | None:
        """List all enrolled speakers for a tenant."""
        tenant_id = message.tenant_id or "default"
        store = self._get_store()
        profiles = await asyncio.to_thread(store.list_profiles, tenant_id)
        return message.create_response({"speakers": profiles})

    async def handle_delete(self, message: Message) -> Message | None:
        """Delete a speaker profile."""
        payload = message.payload
        tenant_id = message.tenant_id or "default"
        speaker_id = str(payload.get("speaker_id", ""))

        if not speaker_id:
            return message.create_error("speaker_id is required")

        store = self._get_store()
        deleted = await asyncio.to_thread(store.delete_profile, tenant_id, speaker_id)

        # Invalidate cache
        self._embedding_cache.pop(tenant_id, None)

        return message.create_response({"success": deleted, "speaker_id": speaker_id})

    def _payload_to_audio(self, payload: dict[str, Any]) -> np.ndarray | None:
        """Extract audio from a message payload."""
        # Option 1: raw PCM bytes (hex-encoded or raw)
        pcm_bytes = payload.get("pcm_bytes")
        if pcm_bytes:
            if isinstance(pcm_bytes, str):
                pcm_bytes = bytes.fromhex(pcm_bytes)
            return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Option 2: pre-converted float32
        audio_float = payload.get("audio_float32")
        if audio_float is not None:
            return np.array(audio_float, dtype=np.float32)

        return None

    def _cache_unknown(
        self,
        tenant_id: str,
        embedding: np.ndarray,
        session_id: str = "",
        timestamp: float = 0.0,
        text: str = "",
    ) -> None:
        """Cache an unknown speaker embedding for retroactive matching."""
        if tenant_id not in self._unknown_cache:
            self._unknown_cache[tenant_id] = []

        cache = self._unknown_cache[tenant_id]

        # Evict expired entries
        now = time.time()
        cache[:] = [e for e in cache if (now - e.timestamp) < self.UNKNOWN_TTL]

        # Evict oldest if over limit
        while len(cache) >= self.MAX_UNKNOWN_CACHE:
            cache.pop(0)

        cache.append(
            UnknownEmbedding(
                embedding=embedding,
                session_id=session_id,
                timestamp=timestamp or now,
                text=text,
            )
        )
        logger.debug(
            "Cached unknown embedding: tenant=%s session=%s (cache size=%d)",
            tenant_id,
            session_id[:8],
            len(cache),
        )

    async def _retroactive_match(
        self,
        tenant_id: str,
        speaker_id: str,
        speaker_name: str,
        enrollment_embedding: np.ndarray,
        threshold: float = 0.72,
    ) -> int:
        """
        Compare a newly enrolled speaker against cached unknown embeddings.

        For each match, publishes a `speaker.retroactive_update` event so
        downstream consumers (WebSocket, memory) can re-tag old messages.

        Returns the number of retroactive matches found.
        """
        cache = self._unknown_cache.get(tenant_id, [])
        if not cache:
            return 0

        matches: list[UnknownEmbedding] = []
        remaining: list[UnknownEmbedding] = []

        enroll_norm = enrollment_embedding / (np.linalg.norm(enrollment_embedding) + 1e-8)

        for entry in cache:
            entry_norm = entry.embedding / (np.linalg.norm(entry.embedding) + 1e-8)
            similarity = float(np.dot(enroll_norm, entry_norm))
            if similarity >= threshold:
                matches.append(entry)
            else:
                remaining.append(entry)

        # Update the cache — remove matched entries
        self._unknown_cache[tenant_id] = remaining

        if matches:
            logger.info(
                "Retroactive speaker match: %d unknown utterances now attributed "
                "to '%s' (%s), tenant=%s",
                len(matches),
                speaker_id,
                speaker_name,
                tenant_id,
            )

            # Publish retroactive update event
            if self.bus:
                update_msg = Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    tenant_id=tenant_id,
                    topic="speaker.retroactive_update",
                    payload={
                        "speaker_id": speaker_id,
                        "speaker_name": speaker_name,
                        "matched_count": len(matches),
                        "matched_timestamps": [m.timestamp for m in matches],
                        "matched_sessions": list({m.session_id for m in matches if m.session_id}),
                        "matched_texts": [m.text for m in matches if m.text],
                    },
                )
                await self.bus.publish("speaker.retroactive_update", update_msg)

        return len(matches)

    async def stop(self) -> None:
        """Cleanup resources."""
        if self._store:
            self._store.close()
        self._encoder = None
        self._unknown_cache.clear()
        logger.info("SpeakerIdNode stopped")
