"""
Studio Emotion Endpoint — Uses core EmotionEngine.

Replaces the legacy lookup that searched for 'EmotionNode' by attribute
sniffing. Now directly queries the core EmotionEngine node.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from hbllm.serving.studio.helpers import get_node_map

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/emotion/state")
async def get_emotion_state(agent_name: str = "assistant", tenant_id: str = "default"):
    """Get current emotional state from the core EmotionEngine."""
    # Try core EmotionEngine first (new path)
    node_map = get_node_map()
    emotion_engine = node_map.get("EmotionEngine")

    if emotion_engine and hasattr(emotion_engine, "get_state"):
        state = emotion_engine.get_state()
        if isinstance(state, dict):
            return {
                "valence": state.get("valence", 0.0),
                "arousal": state.get("arousal", 0.0),
                "emotion_label": state.get("label", "neutral"),
                "dominant_emotion": state.get("dominant", "neutral"),
                "status": "active",
                "source": "core_engine",
            }

    # Fallback: try legacy EmotionNode or plugin node
    emotion_node = node_map.get("EmotionNode")

    # Search plugins if not found by class name
    if not emotion_node:
        from hbllm.serving.state import _state

        pm = _state.get("plugin_manager")
        if pm:
            for node in getattr(pm, "_loaded_nodes", []):
                if hasattr(node, "current_valence") and hasattr(node, "current_arousal"):
                    emotion_node = node
                    break

    if emotion_node:
        valence = getattr(emotion_node, "current_valence", 0.0)
        arousal = getattr(emotion_node, "current_arousal", 0.0)
        label = "neutral"
        if valence > 0.3:
            label = "happy"
        elif valence > 0.0:
            label = "content"
        elif valence < -0.3:
            label = "sad"
        elif valence < 0.0:
            label = "uneasy"
        return {
            "valence": valence,
            "arousal": arousal,
            "emotion_label": label,
            "status": "active",
            "source": "legacy_node",
        }

    return {
        "valence": 0.0,
        "arousal": 0.0,
        "emotion_label": "neutral",
        "status": "not_loaded",
        "source": "none",
    }
