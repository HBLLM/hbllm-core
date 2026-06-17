"""
Voice configuration and per-tenant voice registry.

Manages voice preferences for TTS output, supporting multiple backends
(Kokoro, Orpheus, SpeechT5) with per-tenant voice customization.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TTSBackend(StrEnum):
    """Supported TTS engine backends."""

    KOKORO = "kokoro"
    ORPHEUS = "orpheus"
    SPEECHT5 = "speecht5"  # Deprecated: will be removed in a future release
    NVIDIA = "nvidia"


class ASRBackend(StrEnum):
    """Supported ASR engine backends."""

    MOONSHINE = "moonshine"
    WHISPER = "whisper"  # Deprecated: will be removed in a future release
    NVIDIA = "nvidia"


# ── Kokoro voice presets ────────────────────────────────────────────────────
# Format: {voice_id: (language, gender, description)}
KOKORO_VOICES: dict[str, tuple[str, str, str]] = {
    # American English - Female
    "af_heart": ("en-us", "female", "Heart — warm, expressive"),
    "af_bella": ("en-us", "female", "Bella — clear, professional"),
    "af_nicole": ("en-us", "female", "Nicole — confident, articulate"),
    "af_sarah": ("en-us", "female", "Sarah — friendly, natural"),
    "af_sky": ("en-us", "female", "Sky — bright, energetic"),
    # American English - Male
    "am_adam": ("en-us", "male", "Adam — deep, authoritative"),
    "am_michael": ("en-us", "male", "Michael — warm, conversational"),
    # British English - Female
    "bf_emma": ("en-gb", "female", "Emma — refined, clear"),
    "bf_isabella": ("en-gb", "female", "Isabella — elegant, composed"),
    # British English - Male
    "bm_george": ("en-gb", "male", "George — measured, professional"),
    "bm_lewis": ("en-gb", "male", "Lewis — warm, natural"),
}

DEFAULT_VOICE_ID = "af_heart"
DEFAULT_KOKORO_SPEED = 1.0


@dataclass
class VoiceConfig:
    """Voice configuration for a single tenant or session."""

    voice_id: str = DEFAULT_VOICE_ID
    speed: float = DEFAULT_KOKORO_SPEED
    backend: TTSBackend = TTSBackend.KOKORO
    language: str = "en-us"
    # Per-tenant custom speaker embedding (for SpeechT5 fallback)
    custom_embedding: list[float] | None = None
    # Orpheus-specific emotional tags
    orpheus_emotion: str | None = None  # e.g. "happy", "sad", "whisper"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioPipelineConfig:
    """Configuration for the full audio pipeline."""

    # ASR settings
    asr_backend: ASRBackend = ASRBackend.MOONSHINE
    asr_model_size: str = "base"  # moonshine: tiny/base, whisper: tiny.en/base.en/small
    asr_language: str = "en"

    # TTS settings
    tts_backend: TTSBackend = TTSBackend.KOKORO
    default_voice: VoiceConfig = field(default_factory=VoiceConfig)

    # VAD settings
    vad_threshold: float = 0.5  # Silero VAD speech probability threshold
    vad_min_speech_ms: int = 250  # Minimum speech duration to trigger ASR
    vad_max_silence_ms: int = 1500  # Silence duration to end utterance (1.5s for natural pauses)
    vad_padding_ms: int = 300  # Padding around speech segments

    # Streaming settings
    stream_sample_rate: int = 16000
    stream_max_buffer_seconds: float = 15.0


class VoiceRegistry:
    """Per-tenant voice preferences backed by SQLite."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        # In-memory cache for hot path
        self._cache: dict[str, VoiceConfig] = {}

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS voice_preferences (
                    tenant_id TEXT PRIMARY KEY,
                    voice_id TEXT NOT NULL DEFAULT 'af_heart',
                    speed REAL NOT NULL DEFAULT 1.0,
                    backend TEXT NOT NULL DEFAULT 'kokoro',
                    language TEXT NOT NULL DEFAULT 'en-us',
                    custom_embedding TEXT,
                    orpheus_emotion TEXT,
                    extra TEXT,
                    updated_at REAL
                )
                """
            )

    def get(self, tenant_id: str) -> VoiceConfig:
        """Get voice config for a tenant (cached)."""
        if tenant_id in self._cache:
            return self._cache[tenant_id]

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM voice_preferences WHERE tenant_id = ?",
                    (tenant_id,),
                ).fetchone()

            if row:
                import json

                config = VoiceConfig(
                    voice_id=row["voice_id"],
                    speed=row["speed"],
                    backend=TTSBackend(row["backend"]),
                    language=row["language"],
                    custom_embedding=(
                        json.loads(row["custom_embedding"]) if row["custom_embedding"] else None
                    ),
                    orpheus_emotion=row["orpheus_emotion"],
                    extra=json.loads(row["extra"]) if row["extra"] else {},
                )
                self._cache[tenant_id] = config
                return config
        except Exception as e:
            logger.warning("Failed to load voice config for %s: %s", tenant_id, e)

        return VoiceConfig()  # Return defaults

    def set(self, tenant_id: str, config: VoiceConfig) -> None:
        """Save voice config for a tenant."""
        import json
        import time

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO voice_preferences
                (tenant_id, voice_id, speed, backend, language,
                 custom_embedding, orpheus_emotion, extra, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant_id,
                    config.voice_id,
                    config.speed,
                    config.backend.value,
                    config.language,
                    json.dumps(config.custom_embedding) if config.custom_embedding else None,
                    config.orpheus_emotion,
                    json.dumps(config.extra) if config.extra else None,
                    time.time(),
                ),
            )
        self._cache[tenant_id] = config
        logger.info("Updated voice config for tenant %s: voice=%s", tenant_id, config.voice_id)

    def list_voices(self, backend: TTSBackend = TTSBackend.KOKORO) -> list[dict[str, str]]:
        """List available voices for a backend."""
        if backend == TTSBackend.KOKORO:
            return [
                {
                    "voice_id": vid,
                    "language": lang,
                    "gender": gender,
                    "description": desc,
                }
                for vid, (lang, gender, desc) in KOKORO_VOICES.items()
            ]
        return []
