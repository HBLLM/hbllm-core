"""
Emotion Modeling Plugin — DEPRECATED.

This module has been moved to ``hbllm.brain.emotion_engine``.
This file re-exports the classes for backward compatibility with
existing plugin configurations.
"""

# Re-export from core location for backward compatibility
from hbllm.brain.emotion_engine import (  # noqa: F401
    _EMOTION_LEXICON,
    _EMOTION_MAP,
    EmotionEngine,
    EmotionState,
)

__all__ = ["EmotionEngine", "EmotionState", "_EMOTION_LEXICON", "_EMOTION_MAP"]
