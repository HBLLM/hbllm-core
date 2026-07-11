"""
Emotion Engine — Tracks emotional valence and adapts response tone.

Subscribes to ``system.experience`` and ``system.evaluation`` to build an
emotional state model.  Publishes ``emotion.state`` updates and provides
adaptation hints that downstream nodes (PersonaEngine, StyleAdapter) use
to adjust tone, empathy, and vocabulary.

Moved from the ``emotion-modeling`` plugin to core because emotion tracking
is a fundamental cognitive capability needed by PersonaEngine, Awareness,
and the expression pipeline.

Bus Topics:
    emotion.state      → Published on every state update
    system.experience  → Subscribed (text-based emotion analysis)
    system.evaluation  → Subscribed (score-based adjustment)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

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


# ── Emotion Engine Node ──────────────────────────────────────────────────────


class EmotionEngine(Node):
    """Tracks emotional valence across conversations and adapts response tone.

    Usage::

        engine = EmotionEngine()
        await engine.start(bus)

        # Engine automatically subscribes to system.experience and
        # system.evaluation, publishes to emotion.state

        hints = engine.get_adaptation_hints()
        # {"tone": "empathetic", "empathy_level": "high", ...}
    """

    def __init__(
        self,
        node_id: str = "emotion_engine",
        decay_rate: float = 0.1,
        history_size: int = 100,
        llm: Any | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.META,
            capabilities=["emotion_tracking", "tone_adaptation", "contextual_inference"],
        )
        self.decay_rate = decay_rate
        self.state = EmotionState()
        self._history: deque[EmotionState] = deque(maxlen=history_size)
        self._word_count = 0
        self._llm = llm  # For contextual emotion inference

        # Behavioral pattern tracking
        self._response_times: deque[float] = deque(maxlen=20)
        self._error_count = 0
        self._message_lengths: deque[int] = deque(maxlen=20)
        self._last_interaction: float = 0.0

        # Per-tenant state cache
        self._tenant_states: dict[str, EmotionState] = {}

    async def on_start(self) -> None:
        """Subscribe to experience, evaluation, and behavioral topics."""
        await self.bus.subscribe("system.experience", self._on_experience)
        await self.bus.subscribe("system.evaluation", self._on_evaluation)
        await self.bus.subscribe("router.query", self._on_user_message)
        await self.bus.subscribe("pipeline.error", self._on_pipeline_error)
        logger.info("EmotionEngine started (llm_inference=%s)", self._llm is not None)

    async def on_stop(self) -> None:
        """Cleanup on shutdown."""
        logger.info("EmotionEngine stopped")

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def _on_experience(self, message: Message) -> None:
        """Analyze experience messages for emotional content."""
        text = message.payload.get("text", "")
        query = message.payload.get("query", "")
        combined = f"{query} {text}".lower()

        self._update_from_text(combined)
        await self._publish_state()

    async def _on_evaluation(self, message: Message) -> None:
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

        # Behavioral signals
        if self._error_count > 3:
            hints["encouragement"] = True
            hints["empathy_level"] = "high"

        if self._response_times and len(self._response_times) >= 3:
            avg_gap = sum(self._response_times) / len(self._response_times)
            if avg_gap < 2.0:  # Rapid-fire messages → user is impatient
                hints["formality"] = "concise"

        return hints

    def get_state(self, tenant_id: str = "default") -> dict[str, Any]:
        """Get emotional state for a specific tenant."""
        state = self._tenant_states.get(tenant_id, self.state)
        return {
            "dominant_emotion": state.primary_emotion,
            "valence": state.valence,
            "arousal": state.arousal,
            "dominance": state.dominance,
            "confidence": state.confidence,
        }

    # ── Behavioral Pattern Tracking ──────────────────────────────────────

    async def _on_user_message(self, message: Message) -> None:
        """Track behavioral signals from user messages."""
        now = time.time()
        text = message.payload.get("text", "")

        # Response time pattern
        if self._last_interaction > 0:
            gap = now - self._last_interaction
            self._response_times.append(gap)
        self._last_interaction = now

        # Message length pattern
        self._message_lengths.append(len(text))

        # Behavioral emotional signals
        self._apply_behavioral_signals()

        # LLM-based contextual inference (if available and text is substantial)
        if self._llm and len(text) > 20:
            await self._infer_emotion_from_context(text, message.tenant_id)

    async def _on_pipeline_error(self, message: Message) -> None:
        """Track errors that may frustrate the user."""
        self._error_count += 1
        # Multiple errors in succession strongly indicate frustration
        if self._error_count >= 2:
            self.state.valence = max(-1.0, self.state.valence - 0.1)
            self.state.arousal = min(1.0, self.state.arousal + 0.1)
            self._classify_emotion()
            await self._publish_state()

    def _apply_behavioral_signals(self) -> None:
        """Adjust emotional state based on behavioral patterns."""
        # Short rapid messages → possible impatience/frustration
        if len(self._message_lengths) >= 3:
            recent = list(self._message_lengths)[-3:]
            avg_len = sum(recent) / len(recent)
            if avg_len < 15 and len(self._response_times) >= 3:
                recent_gaps = list(self._response_times)[-3:]
                avg_gap = sum(recent_gaps) / len(recent_gaps)
                if avg_gap < 3.0:  # Very rapid short messages
                    self.state.arousal = min(1.0, self.state.arousal + 0.05)

        # Long detailed messages → user is engaged
        if self._message_lengths and list(self._message_lengths)[-1] > 200:
            self.state.valence = min(1.0, self.state.valence + 0.02)

    # ── LLM-Based Contextual Inference ───────────────────────────────────

    async def _infer_emotion_from_context(self, text: str, tenant_id: str | None = None) -> None:
        """Use LLM to infer emotional state from full message context."""
        try:
            import asyncio

            prompt = [
                {
                    "role": "system",
                    "content": (
                        "Analyze the emotional state of the user from their message. "
                        "Respond with ONLY a JSON object: "
                        '{"valence": float(-1 to 1), "arousal": float(0 to 1), '
                        '"emotion": "word", "confidence": float(0 to 1)}'
                    ),
                },
                {"role": "user", "content": text[:500]},
            ]

            # Non-blocking with timeout so we don't slow down the pipeline
            if hasattr(self._llm, "generate"):
                result = await asyncio.wait_for(self._llm.generate(prompt), timeout=3.0)
            elif hasattr(self._llm, "chat"):
                result = await asyncio.wait_for(self._llm.chat(prompt), timeout=3.0)
            else:
                return

            # Parse the JSON response
            import json

            result_str = str(result)
            # Extract JSON from response
            start = result_str.find("{")
            end = result_str.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(result_str[start:end])
                llm_valence = float(data.get("valence", 0))
                llm_arousal = float(data.get("arousal", 0.5))
                llm_confidence = float(data.get("confidence", 0.5))

                # Blend LLM inference with lexicon-based state
                # LLM gets higher weight because it understands context
                alpha = 0.6 * llm_confidence
                self.state.valence = (1 - alpha) * self.state.valence + alpha * llm_valence
                self.state.arousal = (1 - alpha) * self.state.arousal + alpha * llm_arousal
                self.state.confidence = max(self.state.confidence, llm_confidence)

                emotion = data.get("emotion", "")
                if emotion and llm_confidence > 0.6:
                    self.state.primary_emotion = emotion

                # Cache per-tenant
                if tenant_id:
                    self._tenant_states[tenant_id] = EmotionState(
                        valence=self.state.valence,
                        arousal=self.state.arousal,
                        dominance=self.state.dominance,
                        confidence=self.state.confidence,
                        primary_emotion=self.state.primary_emotion,
                    )

                logger.debug(
                    "LLM emotion inference: %s (v=%.2f, a=%.2f, conf=%.2f)",
                    emotion,
                    llm_valence,
                    llm_arousal,
                    llm_confidence,
                )
        except (TimeoutError, asyncio.TimeoutError):
            logger.debug("LLM emotion inference timed out")
        except Exception as e:
            logger.debug("LLM emotion inference failed: %s", e)

    def stats(self) -> dict[str, Any]:
        """Return emotional statistics."""
        return {
            "current_state": self.state.to_dict(),
            "adaptation_hints": self.get_adaptation_hints(),
            "history_size": len(self._history),
            "trend": self._compute_trend(),
            "behavioral": {
                "error_count": self._error_count,
                "avg_response_time": (
                    round(sum(self._response_times) / len(self._response_times), 1)
                    if self._response_times
                    else None
                ),
                "avg_message_length": (
                    round(sum(self._message_lengths) / len(self._message_lengths))
                    if self._message_lengths
                    else None
                ),
            },
            "llm_inference": self._llm is not None,
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
