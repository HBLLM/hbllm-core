"""Style Adapter — maps EmotionState (VAD) to rendering style hints.

Translates the Valence-Arousal-Dominance emotional state into concrete
directives for the ExpressionStream pipeline: tone, verbosity, formality.

This bridges the emotion-modeling plugin's affect detection with the
expression pipeline's text generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StyleHints:
    """Concrete rendering directives derived from emotional state.

    Attributes:
        tone: Emotional tone for the response (e.g. "empathetic", "enthusiastic").
        verbosity: How verbose the response should be ("concise", "moderate", "detailed").
        formality: Formality level ("casual", "neutral", "formal").
        energy: Response energy level (0.0 = calm, 1.0 = high energy).
        hedging: How much hedging/uncertainty language to use (0.0 = direct, 1.0 = cautious).
        empathy_level: How much empathetic acknowledgment to include (0.0-1.0).
        prompt_prefix: Optional text to prepend to generation prompts.
    """

    tone: str = "neutral"
    verbosity: str = "moderate"
    formality: str = "neutral"
    energy: float = 0.5
    hedging: float = 0.3
    empathy_level: float = 0.2
    prompt_prefix: str = ""

    def to_prompt_instruction(self) -> str:
        """Convert style hints to a natural-language instruction for the LLM."""
        parts: list[str] = []

        if self.tone != "neutral":
            parts.append(f"Use a {self.tone} tone.")

        if self.verbosity == "concise":
            parts.append("Be concise and direct — skip unnecessary details.")
        elif self.verbosity == "detailed":
            parts.append("Be thorough and detailed in your explanation.")

        if self.formality == "casual":
            parts.append("Keep the language casual and conversational.")
        elif self.formality == "formal":
            parts.append("Maintain a professional and formal register.")

        if self.energy > 0.7:
            parts.append("Match the user's enthusiasm and energy.")
        elif self.energy < 0.3:
            parts.append("Keep a calm, measured pace.")

        if self.hedging < 0.15:
            parts.append("Be assertive — avoid hedge words like 'maybe' or 'perhaps'.")

        if self.empathy_level > 0.6:
            parts.append("Acknowledge the user's feelings before addressing their question.")

        if not parts:
            return ""

        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tone": self.tone,
            "verbosity": self.verbosity,
            "formality": self.formality,
            "energy": self.energy,
            "hedging": self.hedging,
            "empathy_level": self.empathy_level,
        }


# ── Tone Mapping ─────────────────────────────────────────────────────────

# VAD (Valence-Arousal-Dominance) ranges → tone descriptors
_TONE_MAP: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float], str]] = [
    # (valence_range, arousal_range, dominance_range, tone)
    ((-1.0, -0.3), (0.3, 1.0), (-1.0, 1.0), "empathetic"),  # Frustrated/angry
    ((-1.0, -0.3), (-1.0, 0.3), (-1.0, 1.0), "supportive"),  # Sad/down
    ((0.3, 1.0), (0.3, 1.0), (-1.0, 1.0), "enthusiastic"),  # Excited/happy
    ((0.3, 1.0), (-1.0, 0.3), (-1.0, 1.0), "warm"),  # Content/calm
    ((-0.3, 0.3), (0.3, 1.0), (0.3, 1.0), "focused"),  # Intense/engaged
    ((-0.3, 0.3), (-1.0, 0.3), (-1.0, 1.0), "neutral"),  # Neutral
]


def _in_range(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1]


class StyleAdapter:
    """Maps EmotionState VAD values to StyleHints for ExpressionStream.

    Usage::

        adapter = StyleAdapter()

        # From EmotionEngine's current state
        hints = adapter.adapt(valence=-0.5, arousal=0.7, dominance=0.3)
        # → StyleHints(tone="empathetic", verbosity="concise", ...)

        # Pass to ExpressionStream
        result = await expression_stream.express(
            understanding=...,
            base_thought=...,
            original_query=...,
            style_hints=hints,
        )
    """

    def adapt(
        self,
        valence: float = 0.0,
        arousal: float = 0.0,
        dominance: float = 0.0,
    ) -> StyleHints:
        """Convert VAD dimensions to concrete style directives.

        Args:
            valence: Pleasure dimension (-1.0 to 1.0). Negative = unhappy.
            arousal: Activation dimension (-1.0 to 1.0). High = energetic.
            dominance: Control dimension (-1.0 to 1.0). High = assertive.

        Returns:
            StyleHints with tone, verbosity, formality, etc.
        """
        # Clamp inputs
        valence = max(-1.0, min(1.0, valence))
        arousal = max(-1.0, min(1.0, arousal))
        dominance = max(-1.0, min(1.0, dominance))

        # 1. Determine tone from VAD ranges
        tone = "neutral"
        for v_range, a_range, d_range, tone_name in _TONE_MAP:
            if (
                _in_range(valence, v_range)
                and _in_range(arousal, a_range)
                and _in_range(dominance, d_range)
            ):
                tone = tone_name
                break

        # 2. Verbosity: frustrated/high-arousal users want concise answers
        if valence < -0.3 or arousal > 0.6:
            verbosity = "concise"
        elif arousal < -0.3 and valence > 0.0:
            verbosity = "detailed"  # Calm and receptive
        else:
            verbosity = "moderate"

        # 3. Formality: high dominance → more formal, low → casual
        if dominance > 0.4:
            formality = "formal"
        elif dominance < -0.3 and valence > 0.0:
            formality = "casual"
        else:
            formality = "neutral"

        # 4. Energy: mirrors arousal
        energy = (arousal + 1.0) / 2.0  # Map [-1, 1] → [0, 1]

        # 5. Hedging: assertive users hate hedging
        hedging = max(0.0, 0.5 - dominance * 0.4)

        # 6. Empathy: required when valence is negative
        empathy_level = max(0.0, -valence * 0.8) if valence < 0 else 0.1

        hints = StyleHints(
            tone=tone,
            verbosity=verbosity,
            formality=formality,
            energy=round(energy, 2),
            hedging=round(hedging, 2),
            empathy_level=round(empathy_level, 2),
        )

        # Build prompt prefix
        instruction = hints.to_prompt_instruction()
        if instruction:
            hints.prompt_prefix = f"[Style: {instruction}]\n\n"

        logger.debug(
            "[StyleAdapter] VAD(%.2f, %.2f, %.2f) → %s",
            valence,
            arousal,
            dominance,
            hints.to_dict(),
        )

        return hints

    def adapt_from_emotion_state(self, emotion_state: Any) -> StyleHints:
        """Convenience: adapt directly from an EmotionState object.

        Args:
            emotion_state: An EmotionState from the emotion-modeling plugin
                           (must have .valence, .arousal, .dominance attributes).
        """
        return self.adapt(
            valence=getattr(emotion_state, "valence", 0.0),
            arousal=getattr(emotion_state, "arousal", 0.0),
            dominance=getattr(emotion_state, "dominance", 0.0),
        )
