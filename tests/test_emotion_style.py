"""Tests for StyleAdapter — VAD → StyleHints mapping."""

from __future__ import annotations

from hbllm.brain.snn.expression.style_adapter import StyleAdapter, StyleHints

# ── StyleHints ───────────────────────────────────────────────────────────


def test_style_hints_defaults():
    """Default hints should be neutral."""
    hints = StyleHints()
    assert hints.tone == "neutral"
    assert hints.verbosity == "moderate"
    assert hints.formality == "neutral"


def test_style_hints_to_dict():
    """to_dict should include all key fields."""
    hints = StyleHints(tone="empathetic", verbosity="concise")
    d = hints.to_dict()
    assert d["tone"] == "empathetic"
    assert d["verbosity"] == "concise"
    assert "energy" in d
    assert "hedging" in d


def test_style_hints_prompt_instruction_neutral():
    """Neutral hints should produce empty instruction."""
    hints = StyleHints()
    assert hints.to_prompt_instruction() == ""


def test_style_hints_prompt_instruction_empathetic():
    """Empathetic tone should produce tone instruction."""
    hints = StyleHints(tone="empathetic", verbosity="concise", empathy_level=0.8)
    instruction = hints.to_prompt_instruction()
    assert "empathetic" in instruction
    assert "concise" in instruction.lower() or "direct" in instruction.lower()
    assert "feelings" in instruction.lower() or "acknowledge" in instruction.lower()


def test_style_hints_prompt_instruction_formal():
    """Formal hints should mention professional register."""
    hints = StyleHints(formality="formal")
    instruction = hints.to_prompt_instruction()
    assert "professional" in instruction.lower() or "formal" in instruction.lower()


# ── StyleAdapter ─────────────────────────────────────────────────────────


class TestStyleAdapter:
    """Test VAD → StyleHints mapping."""

    def setup_method(self):
        self.adapter = StyleAdapter()

    def test_neutral_input(self):
        """Neutral VAD should produce neutral hints."""
        hints = self.adapter.adapt(valence=0.0, arousal=0.0, dominance=0.0)
        assert hints.tone == "neutral"
        assert hints.verbosity == "moderate"

    def test_frustrated_user(self):
        """Negative valence + high arousal = empathetic + concise."""
        hints = self.adapter.adapt(valence=-0.6, arousal=0.7, dominance=0.2)
        assert hints.tone == "empathetic"
        assert hints.verbosity == "concise"
        assert hints.empathy_level > 0.3

    def test_excited_user(self):
        """Positive valence + high arousal = enthusiastic."""
        hints = self.adapter.adapt(valence=0.7, arousal=0.8, dominance=0.5)
        assert hints.tone == "enthusiastic"
        assert hints.energy > 0.7

    def test_sad_user(self):
        """Negative valence + low arousal = supportive."""
        hints = self.adapter.adapt(valence=-0.5, arousal=-0.2, dominance=-0.3)
        assert hints.tone == "supportive"
        assert hints.empathy_level > 0.2

    def test_calm_content_user(self):
        """Positive valence + low arousal = warm + detailed."""
        hints = self.adapter.adapt(valence=0.5, arousal=-0.5, dominance=0.0)
        assert hints.tone == "warm"
        assert hints.verbosity == "detailed"

    def test_assertive_user(self):
        """High dominance reduces hedging."""
        hints = self.adapter.adapt(valence=0.0, arousal=0.5, dominance=0.8)
        assert hints.hedging < 0.2
        assert hints.formality == "formal"

    def test_casual_low_dominance(self):
        """Low dominance + positive valence = casual."""
        hints = self.adapter.adapt(valence=0.3, arousal=0.0, dominance=-0.5)
        assert hints.formality == "casual"

    def test_clamping(self):
        """Values outside [-1, 1] should be clamped."""
        hints = self.adapter.adapt(valence=-5.0, arousal=5.0, dominance=3.0)
        assert hints.energy <= 1.0
        assert hints.hedging >= 0.0

    def test_prompt_prefix_generated(self):
        """Non-neutral states should produce a prompt prefix."""
        hints = self.adapter.adapt(valence=-0.7, arousal=0.8, dominance=0.2)
        assert hints.prompt_prefix  # Should be non-empty
        assert "[Style:" in hints.prompt_prefix

    def test_prompt_prefix_empty_for_neutral(self):
        """Neutral state should produce empty prompt prefix."""
        hints = self.adapter.adapt(valence=0.0, arousal=0.0, dominance=0.0)
        assert hints.prompt_prefix == ""

    def test_adapt_from_emotion_state(self):
        """Should work with an emotion state object."""

        class FakeEmotionState:
            valence = -0.4
            arousal = 0.6
            dominance = 0.1

        hints = self.adapter.adapt_from_emotion_state(FakeEmotionState())
        assert hints.tone != ""
        assert isinstance(hints.empathy_level, float)

    def test_adapt_from_missing_attributes(self):
        """Should handle objects without VAD attributes gracefully."""

        class Empty:
            pass

        hints = self.adapter.adapt_from_emotion_state(Empty())
        assert hints.tone == "neutral"
