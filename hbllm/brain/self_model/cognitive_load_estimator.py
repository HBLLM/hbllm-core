"""Cognitive Load Estimator — adaptive response complexity.

Estimates the user's current cognitive load to modulate:
    - Response verbosity (exhausted user → shorter answers)
    - Information density (high load → simpler explanations)
    - Proactive suggestion timing (high load → fewer interruptions)
    - Tone (stressed → more empathetic)

Signals used:
    1. Time of day (late night = higher cognitive load)
    2. Conversation pace (rapid questions = task-focused)
    3. Message complexity (long questions = deep thought)
    4. Error rate in user input (typos = fatigue)
    5. Session duration (long sessions = increasing load)
    6. Engagement state (from InterruptDetector)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CognitiveLoadEstimate:
    """Estimated cognitive load and recommendations."""

    load_level: float = 0.5  # 0.0 (relaxed) → 1.0 (overloaded)
    load_label: str = "normal"  # "relaxed", "normal", "focused", "high", "overloaded"
    recommended_verbosity: str = "normal"  # "concise", "normal", "detailed"
    recommended_tone: str = "neutral"  # "casual", "neutral", "professional", "empathetic"
    suppress_proactive: bool = False
    signals: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "load_level": round(self.load_level, 2),
            "load_label": self.load_label,
            "recommended_verbosity": self.recommended_verbosity,
            "recommended_tone": self.recommended_tone,
            "suppress_proactive": self.suppress_proactive,
            "signals": {k: round(v, 2) for k, v in self.signals.items()},
        }


class CognitiveLoadEstimator:
    """Estimates user cognitive load from interaction signals.

    Usage::

        estimator = CognitiveLoadEstimator()

        # Record interactions
        estimator.record_message("How do I fix this bug in the auth module?")
        estimator.record_message("nvm found it")

        # Get current estimate
        estimate = estimator.estimate()
        # → load_level=0.6, recommended_verbosity="concise"
    """

    def __init__(
        self,
        session_start: float | None = None,
    ) -> None:
        self._session_start = session_start or time.time()
        self._messages: list[dict[str, Any]] = []
        self._max_messages = 100

        # Weights for combining signals
        self._weights = {
            "time_of_day": 0.15,
            "session_duration": 0.15,
            "message_pace": 0.20,
            "message_complexity": 0.15,
            "typo_rate": 0.15,
            "engagement": 0.20,
        }

        # External state
        self._engagement_state: str = "present"

    def record_message(self, content: str, role: str = "user") -> None:
        """Record a message for cognitive load analysis."""
        self._messages.append(
            {
                "content": content,
                "role": role,
                "timestamp": time.time(),
                "length": len(content),
                "word_count": len(content.split()),
            }
        )

        if len(self._messages) > self._max_messages:
            self._messages = self._messages[-self._max_messages :]

    def update_engagement(self, state: str) -> None:
        """Update engagement state from InterruptDetector."""
        self._engagement_state = state

    def estimate(self) -> CognitiveLoadEstimate:
        """Compute the current cognitive load estimate."""
        signals: dict[str, float] = {}

        # Signal 1: Time of day
        signals["time_of_day"] = self._time_of_day_load()

        # Signal 2: Session duration
        signals["session_duration"] = self._session_duration_load()

        # Signal 3: Message pace (messages per minute)
        signals["message_pace"] = self._message_pace_load()

        # Signal 4: Message complexity
        signals["message_complexity"] = self._message_complexity_load()

        # Signal 5: Typo rate
        signals["typo_rate"] = self._typo_rate_load()

        # Signal 6: Engagement state
        signals["engagement"] = self._engagement_load()

        # Weighted combination
        total_load = sum(signals[key] * self._weights.get(key, 0) for key in signals)
        total_load = max(0.0, min(1.0, total_load))

        # Determine labels
        if total_load < 0.2:
            label = "relaxed"
        elif total_load < 0.4:
            label = "normal"
        elif total_load < 0.6:
            label = "focused"
        elif total_load < 0.8:
            label = "high"
        else:
            label = "overloaded"

        # Recommendations
        if total_load < 0.3:
            verbosity = "detailed"
            tone = "casual"
        elif total_load < 0.6:
            verbosity = "normal"
            tone = "neutral"
        elif total_load < 0.8:
            verbosity = "concise"
            tone = "professional"
        else:
            verbosity = "concise"
            tone = "empathetic"

        return CognitiveLoadEstimate(
            load_level=total_load,
            load_label=label,
            recommended_verbosity=verbosity,
            recommended_tone=tone,
            suppress_proactive=total_load > 0.7,
            signals=signals,
        )

    def get_system_prompt_modifier(self) -> str:
        """Generate a system prompt modifier based on current load.

        This string is injected into the LLM's system prompt to
        modulate response style.
        """
        est = self.estimate()

        modifiers: list[str] = []

        if est.recommended_verbosity == "concise":
            modifiers.append("Keep responses SHORT and to the point.")
        elif est.recommended_verbosity == "detailed":
            modifiers.append("You can be thorough and detailed in your responses.")

        if est.recommended_tone == "empathetic":
            modifiers.append("The user seems tired — be warm and supportive.")
        elif est.recommended_tone == "casual":
            modifiers.append("Keep a relaxed, conversational tone.")

        if est.suppress_proactive:
            modifiers.append("Avoid unsolicited suggestions — only answer what's asked.")

        return " ".join(modifiers) if modifiers else ""

    # ── Signal Computation ─────────────────────────────────────────────

    def _time_of_day_load(self) -> float:
        """Late night / early morning = higher load."""
        hour = datetime.now(timezone.utc).hour
        if 0 <= hour < 6:
            return 0.8  # Deep night
        elif 6 <= hour < 9:
            return 0.4  # Early morning
        elif 9 <= hour < 17:
            return 0.3  # Work hours
        elif 17 <= hour < 21:
            return 0.4  # Evening
        elif 21 <= hour < 23:
            return 0.5  # Night
        else:
            return 0.7  # Late night

    def _session_duration_load(self) -> float:
        """Longer sessions = higher load (fatigue)."""
        duration_min = (time.time() - self._session_start) / 60

        if duration_min < 15:
            return 0.1  # Fresh
        elif duration_min < 60:
            return 0.3  # Normal
        elif duration_min < 120:
            return 0.5  # Getting tired
        elif duration_min < 240:
            return 0.7  # Extended session
        else:
            return 0.9  # Marathon session

    def _message_pace_load(self) -> float:
        """Rapid messages = task-focused (higher load)."""
        user_msgs = [m for m in self._messages if m["role"] == "user"]
        if len(user_msgs) < 2:
            return 0.3

        # Messages per minute over last 10 messages
        recent = user_msgs[-10:]
        if len(recent) < 2:
            return 0.3

        span = recent[-1]["timestamp"] - recent[0]["timestamp"]
        if span <= 0:
            return 0.5

        mpm = len(recent) / (span / 60)

        if mpm < 0.5:
            return 0.2  # Leisurely
        elif mpm < 2:
            return 0.4  # Normal
        elif mpm < 5:
            return 0.6  # Fast
        else:
            return 0.8  # Rapid-fire

    def _message_complexity_load(self) -> float:
        """Longer, more complex messages = deeper thought required."""
        user_msgs = [m for m in self._messages[-10:] if m["role"] == "user"]
        if not user_msgs:
            return 0.3

        avg_words = sum(m["word_count"] for m in user_msgs) / len(user_msgs)

        if avg_words < 5:
            return 0.6  # Very short = possibly frustrated/hurried
        elif avg_words < 15:
            return 0.3  # Normal
        elif avg_words < 50:
            return 0.5  # Detailed
        else:
            return 0.7  # Complex

    def _typo_rate_load(self) -> float:
        """More typos/corrections = fatigue indicator."""
        user_msgs = [m for m in self._messages[-10:] if m["role"] == "user"]
        if len(user_msgs) < 3:
            return 0.2

        # Check for correction indicators
        corrections = 0
        for m in user_msgs:
            content = m["content"].lower()
            if any(w in content for w in ["*", "i mean", "sorry", "nvm", "actually", "typo"]):
                corrections += 1

        correction_rate = corrections / len(user_msgs)

        if correction_rate < 0.1:
            return 0.1
        elif correction_rate < 0.3:
            return 0.4
        else:
            return 0.7

    def _engagement_load(self) -> float:
        """Map engagement state to cognitive load."""
        engagement_map = {
            "engaged": 0.5,  # Actively working
            "listening": 0.4,  # Receptive
            "present": 0.3,  # Available
            "idle": 0.2,  # Low load
            "deep_idle": 0.1,  # Very low
        }
        return engagement_map.get(self._engagement_state, 0.3)

    def stats(self) -> dict[str, Any]:
        est = self.estimate()
        return {
            "current_load": est.load_level,
            "load_label": est.load_label,
            "session_duration_min": (time.time() - self._session_start) / 60,
            "messages_recorded": len(self._messages),
            "engagement_state": self._engagement_state,
        }
