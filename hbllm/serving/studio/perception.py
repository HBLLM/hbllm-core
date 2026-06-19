"""
Studio Perception Endpoints — Wake Word, Location, Voice Stream Bridge.

Provides observability and control for the perception-layer nodes
added to the cognitive pipeline.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from hbllm.serving.studio.helpers import get_brain, get_node, get_tenant_id

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Wake Word ────────────────────────────────────────────────────────────────


@router.get("/studio/wake-word")
async def studio_wake_word_status():
    """Wake word detector status and detection stats."""
    node = get_node("WakeWordDetector")
    if not node:
        return {
            "status": "not_loaded",
            "wake_words": [],
            "backend": "unknown",
            "total_detections": 0,
            "active": False,
        }

    stats = node.stats() if hasattr(node, "stats") else {}
    config = getattr(node, "config", None)

    return {
        "status": "active",
        "wake_words": stats.get("wake_words", []),
        "backend": stats.get("backend", "unknown"),
        "total_detections": stats.get("total_detections", 0),
        "active": stats.get("active", False),
        "last_activation": stats.get("last_activation", 0),
        "config": {
            "confidence_threshold": getattr(config, "confidence_threshold", 0.7),
            "cooldown_seconds": getattr(config, "cooldown_seconds", 2.0),
            "sample_rate": getattr(config, "sample_rate", 16000),
            "frame_length_ms": getattr(config, "frame_length_ms", 80),
        }
        if config
        else {},
    }


@router.put("/studio/wake-word/config")
async def studio_wake_word_update_config(request: Request):
    """Update wake word detector configuration.

    Body::

        {
            "wake_words": ["hey sentra", "hello computer"],
            "confidence_threshold": 0.8,
            "cooldown_seconds": 3.0
        }
    """
    node = get_node("WakeWordDetector")
    if not node:
        raise HTTPException(status_code=503, detail="WakeWordDetector not loaded")

    body = await request.json()
    config = node.config

    if "wake_words" in body:
        config.wake_words = body["wake_words"]
    if "confidence_threshold" in body:
        config.confidence_threshold = float(body["confidence_threshold"])
    if "cooldown_seconds" in body:
        config.cooldown_seconds = float(body["cooldown_seconds"])

    return {"status": "updated", "wake_words": config.wake_words}


@router.post("/studio/wake-word/toggle")
async def studio_wake_word_toggle():
    """Enable or disable wake word detection at runtime."""
    node = get_node("WakeWordDetector")
    if not node:
        raise HTTPException(status_code=503, detail="WakeWordDetector not loaded")

    new_state = not node._active
    node.set_active(new_state)
    return {"status": "toggled", "active": new_state}


# ── Location Adapter ─────────────────────────────────────────────────────────


@router.get("/studio/location")
async def studio_location_status():
    """Location adapter status and tracking stats."""
    node = get_node("LocationAdapter")
    if not node:
        return {
            "status": "not_loaded",
            "tracked_tenants": 0,
            "total_updates": 0,
            "active_geofences": 0,
        }

    stats = node.stats() if hasattr(node, "stats") else {}
    return {
        "status": "active",
        **stats,
    }


@router.get("/studio/location/current")
async def studio_location_current(request: Request):
    """Get the current location for a tenant."""
    node = get_node("LocationAdapter")
    if not node:
        raise HTTPException(status_code=503, detail="LocationAdapter not loaded")

    tenant_id = get_tenant_id(request)
    loc = node.get_location(tenant_id)

    if loc is None:
        return {"tenant_id": tenant_id, "location": None}

    return {"tenant_id": tenant_id, "location": loc.to_dict()}


@router.get("/studio/location/geofences")
async def studio_location_geofences(request: Request):
    """List active geofences for a tenant."""
    node = get_node("LocationAdapter")
    if not node:
        raise HTTPException(status_code=503, detail="LocationAdapter not loaded")

    tenant_id = get_tenant_id(request)
    fences = node.list_geofences(tenant_id)
    return {"tenant_id": tenant_id, "geofences": fences, "count": len(fences)}


@router.post("/studio/location/geofence")
async def studio_location_add_geofence(request: Request):
    """Add a geofence.

    Body::

        {
            "id": "home",
            "name": "Home",
            "latitude": 37.7749,
            "longitude": -122.4194,
            "radius_meters": 100,
            "trigger_on_enter": true,
            "trigger_on_exit": true
        }
    """
    node = get_node("LocationAdapter")
    if not node:
        raise HTTPException(status_code=503, detail="LocationAdapter not loaded")

    body = await request.json()
    tenant_id = get_tenant_id(request)

    if "latitude" not in body or "longitude" not in body:
        raise HTTPException(status_code=400, detail="latitude and longitude are required")

    from hbllm.perception.location_adapter import Geofence

    fence = Geofence(
        id=body.get("id", f"fence_{len(node.list_geofences(tenant_id))}"),
        name=body.get("name", "Unnamed"),
        latitude=float(body["latitude"]),
        longitude=float(body["longitude"]),
        radius_meters=float(body.get("radius_meters", 100)),
        tenant_id=tenant_id,
        trigger_on_enter=body.get("trigger_on_enter", True),
        trigger_on_exit=body.get("trigger_on_exit", True),
        metadata=body.get("metadata", {}),
    )
    node.add_geofence(fence)

    return {"status": "created", "geofence": fence.to_dict()}


@router.delete("/studio/location/geofence/{fence_id}")
async def studio_location_remove_geofence(fence_id: str, request: Request):
    """Remove a geofence by ID."""
    node = get_node("LocationAdapter")
    if not node:
        raise HTTPException(status_code=503, detail="LocationAdapter not loaded")

    tenant_id = get_tenant_id(request)
    removed = node.remove_geofence(tenant_id, fence_id)

    if not removed:
        raise HTTPException(status_code=404, detail=f"Geofence '{fence_id}' not found")

    return {"status": "removed", "id": fence_id}


# ── Voice Stream Bridge ─────────────────────────────────────────────────────


@router.get("/studio/voice-stream")
async def studio_voice_stream_status():
    """Voice stream bridge status and stats."""
    brain = get_brain()
    bridge = getattr(brain, "voice_bridge", None) if brain else None

    if not bridge:
        # Fall back to node map lookup
        bridge = get_node("VoiceStreamBridge")

    if not bridge:
        return {
            "status": "not_loaded",
            "active_sessions": 0,
            "total_sentences_dispatched": 0,
            "total_interruptions": 0,
            "barge_in_enabled": False,
        }

    stats = bridge.stats() if hasattr(bridge, "stats") else {}
    config = getattr(bridge, "config", None)

    return {
        "status": "active",
        **stats,
        "config": {
            "min_sentence_length": getattr(config, "min_sentence_length", 10),
            "max_sentence_length": getattr(config, "max_sentence_length", 500),
            "default_voice_id": getattr(config, "default_voice_id", "af_heart"),
            "speed": getattr(config, "speed", 1.0),
            "skip_code_blocks": getattr(config, "skip_code_blocks", True),
            "skip_urls": getattr(config, "skip_urls", True),
        }
        if config
        else {},
    }


@router.put("/studio/voice-stream/config")
async def studio_voice_stream_update_config(request: Request):
    """Update voice stream bridge configuration.

    Body::

        {
            "allow_barge_in": true,
            "default_voice_id": "am_adam",
            "speed": 1.2,
            "min_sentence_length": 15
        }
    """
    brain = get_brain()
    bridge = getattr(brain, "voice_bridge", None) if brain else None
    if not bridge:
        bridge = get_node("VoiceStreamBridge")
    if not bridge:
        raise HTTPException(status_code=503, detail="VoiceStreamBridge not loaded")

    body = await request.json()
    config = bridge.config

    if "allow_barge_in" in body:
        config.allow_barge_in = body["allow_barge_in"]
    if "default_voice_id" in body:
        config.default_voice_id = body["default_voice_id"]
    if "speed" in body:
        config.speed = float(body["speed"])
    if "min_sentence_length" in body:
        config.min_sentence_length = int(body["min_sentence_length"])
    if "max_sentence_length" in body:
        config.max_sentence_length = int(body["max_sentence_length"])

    return {"status": "updated"}
