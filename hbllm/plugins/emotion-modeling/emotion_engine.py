"""
Emotion Modeling Plugin — Tracks emotional valence and adapts response tone.

Subscribes to system.experience and system.evaluation to build an emotional
state model. Publishes emotion.state updates and emotion.adaptation hints
that downstream nodes can use to adjust tone, empathy, and vocabulary.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.plugin.sdk import HBLLMPlugin, subscribe

logger = logging.getLogger(__name__)


# ── Emotion Data Structures ───────────────────────────────────────────────────


@dataclass
class EmotionState:
    """Current emotional state of the conversation."""

    valence: float = 0.0  # -1.0 (negative) to 1.0 (positive)
    arousal: float = 0.0  # 0.0 (calm) to 1.0 (excited)
    dominance: float = 0.5  # 0.0 (submissive) to 1.0 (dominant)
    confidence: float = 0.5  # How confident we are in this assessment
    primary_emotion: str = "neutral"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "dominance": round(self.dominance, 3),
            "confidence": round(self.confidence, 3),
            "primary_emotion": self.primary_emotion,
            "timestamp": self.timestamp,
        }


# ── Emotion Lexicon ───────────────────────────────────────────────────────────

# Compact VAD (valence-arousal-dominance) lexicon for common emotion words
_EMOTION_LEXICON: dict[str, tuple[float, float, float]] = {
    # Positive emotions
    "happy": (0.8, 0.5, 0.6),
    "great": (0.8, 0.5, 0.6),
    "excellent": (0.9, 0.6, 0.7),
    "amazing": (0.9, 0.7, 0.6),
    "love": (0.9, 0.6, 0.5),
    "wonderful": (0.9, 0.5, 0.6),
    "thank": (0.7, 0.3, 0.4),
    "thanks": (0.7, 0.3, 0.4),
    "good": (0.6, 0.3, 0.5),
    "nice": (0.6, 0.3, 0.5),
    "perfect": (0.9, 0.5, 0.7),
    "awesome": (0.8, 0.7, 0.6),
    "excited": (0.8, 0.8, 0.6),
    "glad": (0.7, 0.4, 0.5),
    # Negative emotions
    "sad": (-0.7, 0.3, 0.2),
    "angry": (-0.7, 0.8, 0.7),
    "frustrated": (-0.6, 0.7, 0.4),
    "annoyed": (-0.5, 0.6, 0.5),
    "confused": (-0.3, 0.5, 0.3),
    "worried": (-0.5, 0.6, 0.3),
    "disappointed": (-0.6, 0.4, 0.3),
    "terrible": (-0.8, 0.6, 0.4),
    "hate": (-0.9, 0.8, 0.7),
    "awful": (-0.8, 0.5, 0.4),
    "upset": (-0.6, 0.6, 0.4),
    "stuck": (-0.4, 0.5, 0.2),
    "broken": (-0.5, 0.4, 0.3),
    "failed": (-0.6, 0.5, 0.3),
    "error": (-0.4, 0.5, 0.3),
    "bug": (-0.3, 0.5, 0.4),
    # Neutral / task-oriented
    "help": (-0.1, 0.4, 0.3),
    "please": (0.1, 0.3, 0.3),
    "how": (0.0, 0.3, 0.4),
    "what": (0.0, 0.3, 0.4),
    "why": (-0.1, 0.4, 0.4),
    "urgent": (-0.3, 0.8, 0.5),
}

# Map VAD ranges to primary emotions
_EMOTION_MAP = [
    ("joy", lambda v, a, d: v > 0.5 and a > 0.3),
    ("excitement", lambda v, a, d: v > 0.3 and a > 0.7),
    ("gratitude", lambda v, a, d: v > 0.5 and a < 0.4 and d < 0.5),
    ("frustration", lambda v, a, d: v < -0.3 and a > 0.5),
    ("sadness", lambda v, a, d: v < -0.4 and a < 0.4),
    ("confusion", lambda v, a, d: -0.4 < v < 0.1 and 0.3 < a < 0.6),
    ("urgency", lambda v, a, d: a > 0.7 and d > 0.4),
    ("neutral", lambda v, a, d: True),  # fallback
]


# ── Emotion Engine Plugin ─────────────────────────────────────────────────────


class EmotionEngine(HBLLMPlugin):
    """Tracks emotional valence across conversations and adapts response tone."""

    def __init__(
        self,
        node_id: str = "emotion_engine",
        decay_rate: float = 0.1,
        history_size: int = 100,
    ) -> None:
        super().__init__(node_id=node_id, capabilities=["emotion_tracking", "tone_adaptation"])
        self.decay_rate = decay_rate
        self.state = EmotionState()
        self._history: deque[EmotionState] = deque(maxlen=history_size)
        self._word_count = 0

    @subscribe("system.experience")
    async def on_experience(self, message: Message) -> None:
        """Analyze experience messages for emotional content."""
        text = message.payload.get("text", "")
        query = message.payload.get("query", "")
        combined = f"{query} {text}".lower()

        self._update_from_text(combined)
        await self._publish_state()

    @subscribe("system.evaluation")
    async def on_evaluation(self, message: Message) -> None:
        """Adjust emotional state based on interaction quality."""
        score = message.payload.get("overall_score", 0.5)
        flags = message.payload.get("flags", [])

        # Low scores suggest user frustration
        if score < 0.4:
            self.state.valence = max(-1.0, self.state.valence - 0.15)
            self.state.arousal = min(1.0, self.state.arousal + 0.1)
        elif score > 0.8:
            self.state.valence = min(1.0, self.state.valence + 0.05)

        if "high_uncertainty" in flags:
            self.state.valence = max(-1.0, self.state.valence - 0.05)

        self._classify_emotion()
        await self._publish_state()

    def _update_from_text(self, text: str) -> None:
        """Update emotional state from text using lexicon matching."""
        words = text.split()
        val_sum = 0.0
        aro_sum = 0.0
        dom_sum = 0.0
        matches = 0

        for word in words:
            clean = word.strip(".,!?;:'\"()[]{}").lower()
            if clean in _EMOTION_LEXICON:
                v, a, d = _EMOTION_LEXICON[clean]
                val_sum += v
                aro_sum += a
                dom_sum += d
                matches += 1

        if matches > 0:
            # Blend with current state using exponential moving average
            alpha = min(0.5, matches / 10)
            self.state.valence = (1 - alpha) * self.state.valence + alpha * (val_sum / matches)
            self.state.arousal = (1 - alpha) * self.state.arousal + alpha * (aro_sum / matches)
            self.state.dominance = (1 - alpha) * self.state.dominance + alpha * (dom_sum / matches)
            self.state.confidence = min(1.0, 0.3 + matches * 0.1)
        else:
            # Decay toward neutral over time
            self.state.valence *= 1 - self.decay_rate
            self.state.arousal *= 1 - self.decay_rate
            self.state.dominance = 0.5 + (self.state.dominance - 0.5) * (1 - self.decay_rate)
            self.state.confidence *= 0.95

        # Clamp values
        self.state.valence = max(-1.0, min(1.0, self.state.valence))
        self.state.arousal = max(0.0, min(1.0, self.state.arousal))
        self.state.dominance = max(0.0, min(1.0, self.state.dominance))

        self._classify_emotion()
        self.state.timestamp = time.time()
        self._history.append(
            EmotionState(**{k: getattr(self.state, k) for k in EmotionState.__dataclass_fields__})
        )

    def _classify_emotion(self) -> None:
        """Classify the primary emotion from VAD values."""
        v, a, d = self.state.valence, self.state.arousal, self.state.dominance
        for name, predicate in _EMOTION_MAP:
            if predicate(v, a, d):
                self.state.primary_emotion = name
                return

    async def _publish_state(self) -> None:
        """Publish current emotional state on the bus."""
        if self.bus:
            await self.bus.publish(
                "emotion.state",
                Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    topic="emotion.state",
                    payload=self.state.to_dict(),
                ),
            )

    def get_adaptation_hints(self) -> dict[str, Any]:
        """Get tone adaptation hints based on current emotional state."""
        hints: dict[str, Any] = {
            "tone": "neutral",
            "empathy_level": "normal",
            "formality": "standard",
            "encouragement": False,
        }

        if self.state.valence < -0.3:
            hints["tone"] = "empathetic"
            hints["empathy_level"] = "high"
            hints["encouragement"] = True
        elif self.state.valence > 0.5:
            hints["tone"] = "enthusiastic"
        elif self.state.arousal > 0.7:
            hints["tone"] = "focused"
            hints["formality"] = "concise"

        if self.state.primary_emotion == "frustration":
            hints["tone"] = "patient"
            hints["empathy_level"] = "high"
            hints["encouragement"] = True
        elif self.state.primary_emotion == "confusion":
            hints["tone"] = "clarifying"
            hints["formality"] = "simple"

        return hints

    def stats(self) -> dict[str, Any]:
        """Return emotional statistics."""
        return {
            "current_state": self.state.to_dict(),
            "adaptation_hints": self.get_adaptation_hints(),
            "history_size": len(self._history),
            "trend": self._compute_trend(),
        }

    def _compute_trend(self) -> str:
        """Compute the emotional trend from recent history."""
        if len(self._history) < 3:
            return "insufficient_data"
        recent = list(self._history)[-5:]
        avg_val = sum(s.valence for s in recent) / len(recent)
        if avg_val > 0.3:
            return "positive"
        elif avg_val < -0.3:
            return "negative"
        return "stable"
