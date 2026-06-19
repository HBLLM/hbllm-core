"""
Studio Persona Endpoints — Routes through PersonaEngine.

Instead of directly hitting identity.db with raw SQL, these endpoints
now query the PersonaEngine node via the message bus.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3

from fastapi import APIRouter, Request

from hbllm.serving.studio.helpers import get_data_dir, get_node, get_tenant_id

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_persona_from_engine(tenant_id: str) -> dict | None:
    """Try to get persona traits from PersonaEngine node."""
    node = get_node("PersonaEngine")
    if not node:
        return None
    try:
        if hasattr(node, "get_traits"):
            traits = node.get_traits(tenant_id)
            if traits:
                return traits if isinstance(traits, dict) else traits.to_dict()
    except Exception as e:
        logger.debug("PersonaEngine query failed: %s", e)
    return None


def _get_persona_from_db(tenant_id: str) -> dict:
    """Fallback: read traits directly from identity.db."""
    db_path = os.path.join(get_data_dir(), "identity.db")
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT traits_json FROM identities WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()
            if row:
                return json.loads(row["traits_json"])
    except Exception:
        pass
    return {}


_PERSONA_DEFAULTS = {
    "verbosity": "balanced",
    "tone": "neutral",
    "emoji_preference": "minimal",
    "interaction_count": 5,
    "topics_of_interest": ["AI", "cognitive architecture"],
}


@router.get("/api/persona/profile")
async def get_persona_profile(request: Request):
    """Get persona profile — tries PersonaEngine first, falls back to DB."""
    tenant_id = get_tenant_id(request)

    # Try engine
    traits = _get_persona_from_engine(tenant_id)
    if traits:
        merged = {**_PERSONA_DEFAULTS, **traits}
        merged["source"] = "engine"
        return merged

    # Fallback to DB
    traits = _get_persona_from_db(tenant_id)
    if traits:
        merged = {**_PERSONA_DEFAULTS, **traits}
        merged["source"] = "database"
        return merged

    return {**_PERSONA_DEFAULTS, "source": "defaults"}


@router.put("/api/persona/override")
async def override_persona(request: Request):
    """Override persona traits for a tenant."""
    body = await request.json()
    tenant_id = get_tenant_id(request)

    # Try engine first
    node = get_node("PersonaEngine")
    if node and hasattr(node, "update_traits"):
        try:
            node.update_traits(tenant_id, body)
            return {"status": "ok", "source": "engine"}
        except Exception as e:
            logger.debug("PersonaEngine update failed: %s", e)

    # Fallback: write to DB
    db_path = os.path.join(get_data_dir(), "identity.db")
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT traits_json FROM identities WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()
            traits = {}
            if row and row["traits_json"]:
                traits = json.loads(row["traits_json"])
            traits.update(body)
            conn.execute(
                "INSERT INTO identities (tenant_id, traits_json, created_at, updated_at) "
                "VALUES (?, ?, datetime('now'), datetime('now')) "
                "ON CONFLICT(tenant_id) DO UPDATE SET "
                "traits_json = excluded.traits_json, updated_at = excluded.updated_at",
                (tenant_id, json.dumps(traits)),
            )
            conn.commit()
    except Exception as e:
        logger.error("Failed to override persona: %s", e)
    return {"status": "ok", "source": "database"}


@router.post("/api/persona/reset")
async def reset_persona(request: Request):
    """Reset persona traits to defaults."""
    tenant_id = get_tenant_id(request)

    # Try engine
    node = get_node("PersonaEngine")
    if node and hasattr(node, "reset_traits"):
        try:
            node.reset_traits(tenant_id)
            return {"status": "ok", "source": "engine"}
        except Exception:
            pass

    # Fallback to DB
    db_path = os.path.join(get_data_dir(), "identity.db")
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE identities SET traits_json = '{}' WHERE tenant_id = ?",
                (tenant_id,),
            )
            conn.commit()
    except Exception as e:
        logger.error("Failed to reset persona: %s", e)
    return {"status": "ok", "source": "database"}


@router.delete("/api/persona/override")
async def clear_persona_overrides(request: Request):
    """Clear all persona overrides."""
    return await reset_persona(request)
