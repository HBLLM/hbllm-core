"""
Studio Voice Pipeline Endpoints.

Exposes voice/audio pipeline configuration: ASR/TTS status, voice listing,
per-tenant voice config management, and voice test synthesis.

Extracted from ``_legacy.py`` — see Work Stream 1.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from hbllm.network.messages import Message, MessageType
from hbllm.serving.studio.helpers import get_brain, get_node_map, get_tenant_id

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/studio/voice")
async def studio_voice(request: Request) -> Any:
    """Voice pipeline status: ASR backend, TTS backend, VAD, per-tenant voice config."""
    tenant_id = get_tenant_id(request)
    node_map = get_node_map()

    result: dict[str, Any] = {
        "status": "not_loaded",
        "asr": {"backend": "moonshine", "model_loaded": False},
        "tts": {"backend": "kokoro", "model_loaded": False},
        "vad": {"loaded": False},
        "streaming": {"active_sessions": 0},
        "voice_config": None,
    }

    # Check AudioInputNode
    audio_in = node_map.get("AudioInputNode")
    if audio_in:
        result["status"] = "active"
        config = getattr(audio_in, "config", None)
        if config:
            result["asr"]["backend"] = config.asr_backend.value
            result["asr"]["model_size"] = config.asr_model_size
            result["asr"]["sample_rate"] = config.stream_sample_rate
        result["asr"]["model_loaded"] = getattr(audio_in, "_moonshine_model", None) is not None
        result["vad"]["loaded"] = getattr(audio_in, "_vad_model", None) is not None
        result["streaming"]["active_sessions"] = len(getattr(audio_in, "_stream_buffers", {}))

    # Check AudioOutputNode
    audio_out = node_map.get("AudioOutputNode")
    if audio_out:
        result["status"] = "active"
        config = getattr(audio_out, "config", None)
        if config:
            result["tts"]["backend"] = config.tts_backend.value
        result["tts"]["model_loaded"] = (
            getattr(audio_out, "_kokoro_pipeline", None) is not None
            or getattr(audio_out, "_orpheus_engine", None) is not None
        )
        # Get tenant's voice config
        registry = getattr(audio_out, "_voice_registry", None)
        if registry:
            voice = registry.get(tenant_id)
            result["voice_config"] = {
                "voice_id": voice.voice_id,
                "speed": voice.speed,
                "backend": voice.backend.value,
                "language": voice.language,
                "orpheus_emotion": voice.orpheus_emotion,
            }

    return result


@router.get("/studio/voice/voices")
async def studio_voice_list(backend: str = "kokoro") -> Any:
    """List available TTS voices for a backend."""
    from hbllm.perception.voice_config import TTSBackend, VoiceRegistry

    data_dir = os.environ.get("HBLLM_DATA_DIR", "data")
    db_path = os.path.join(data_dir, "voice_preferences.db")
    registry = VoiceRegistry(db_path)

    try:
        voices = registry.list_voices(TTSBackend(backend))
        return {"voices": voices, "backend": backend}
    except Exception as e:
        return {"voices": [], "backend": backend, "error": str(e)}


@router.get("/studio/voice/config/{tenant_id}")
async def studio_voice_get_config(tenant_id: str) -> Any:
    """Get voice config for a specific tenant."""
    from hbllm.perception.voice_config import VoiceRegistry

    data_dir = os.environ.get("HBLLM_DATA_DIR", "data")
    db_path = os.path.join(data_dir, "voice_preferences.db")
    registry = VoiceRegistry(db_path)

    voice = registry.get(tenant_id)
    return {
        "tenant_id": tenant_id,
        "voice_id": voice.voice_id,
        "speed": voice.speed,
        "backend": voice.backend.value,
        "language": voice.language,
        "orpheus_emotion": voice.orpheus_emotion,
    }


@router.put("/studio/voice/config")
async def studio_voice_update_config(request: Request) -> Any:
    """Update voice config for a tenant.

    Body:
        {
            "tenant_id": "my-tenant",
            "voice_id": "am_adam",
            "speed": 1.2,
            "backend": "kokoro",
            "emotion": "happy"
        }
    """
    from hbllm.perception.voice_config import TTSBackend, VoiceConfig, VoiceRegistry

    body = await request.json()
    tenant_id = body.get("tenant_id") or get_tenant_id(request)

    data_dir = os.environ.get("HBLLM_DATA_DIR", "data")
    db_path = os.path.join(data_dir, "voice_preferences.db")
    registry = VoiceRegistry(db_path)

    voice = VoiceConfig(
        voice_id=body.get("voice_id", "af_heart"),
        speed=float(body.get("speed", 1.0)),
        backend=TTSBackend(body.get("backend", "kokoro")),
        language=body.get("language", "en-us"),
        orpheus_emotion=body.get("emotion"),
    )
    registry.set(tenant_id, voice)

    # Also publish to live bus if available
    bus = getattr(get_brain(), "bus", None)
    if bus:
        config_msg = Message(
            type=MessageType.EVENT,
            source_node_id="studio",
            tenant_id=tenant_id,
            topic="voice.config",
            payload={
                "tenant_id": tenant_id,
                "voice_id": voice.voice_id,
                "speed": voice.speed,
                "backend": voice.backend.value,
                "emotion": voice.orpheus_emotion,
            },
        )
        await bus.publish("voice.config", config_msg)

    return {"status": "updated", "tenant_id": tenant_id, "voice_id": voice.voice_id}


@router.post("/studio/voice/test")
async def studio_voice_test(request: Request) -> Any:
    """Test TTS by synthesizing a short phrase.

    Body:
        {
            "text": "Hello, how are you today?",
            "voice_id": "af_heart",
            "backend": "kokoro"
        }
    """
    body = await request.json()
    text = body.get("text", "Hello, this is a voice test.")
    tenant_id = get_tenant_id(request)

    bus = getattr(get_brain(), "bus", None)
    if not bus:
        raise HTTPException(status_code=503, detail="Brain pipeline not initialized")

    msg = Message(
        type=MessageType.QUERY,
        source_node_id="studio",
        tenant_id=tenant_id,
        topic="sensory.audio.out",
        payload={
            "text": text,
            "voice_id": body.get("voice_id"),
            "backend": body.get("backend"),
        },
    )
    try:
        resp = await asyncio.wait_for(
            bus.request("sensory.audio.out", msg, timeout=15.0), timeout=15.0
        )
        if resp and resp.type != MessageType.ERROR:
            return {
                "status": "success",
                "audio_path": resp.payload.get("audio_path"),
                "voice": resp.payload.get("voice"),
            }
        return {"status": "error", "error": resp.payload.get("error", "Synthesis failed")}
    except Exception as e:
        return {"status": "error", "error": str(e)}
