"""Tests for EmotionEngine — emotional valence tracking and adaptation."""

from hbllm.brain.emotion_engine import (
    _EMOTION_LEXICON,
    _EMOTION_MAP,
    EmotionEngine,
    EmotionState,
)


class TestEmotionState:
    def test_defaults(self):
        s = EmotionState()
        assert s.valence == 0.0
        assert s.primary_emotion == "neutral"

    def test_to_dict(self):
        s = EmotionState(valence=0.5, arousal=0.3)
        d = s.to_dict()
        assert d["valence"] == 0.5
        assert "primary_emotion" in d

    def test_timestamp(self):
        s = EmotionState()
        assert s.timestamp > 0


class TestEmotionLexicon:
    def test_positive_words(self):
        assert "happy" in _EMOTION_LEXICON
        assert _EMOTION_LEXICON["happy"][0] > 0

    def test_negative_words(self):
        assert "frustrated" in _EMOTION_LEXICON
        assert _EMOTION_LEXICON["frustrated"][0] < 0

    def test_lexicon_values_are_tuples(self):
        for word, vad in _EMOTION_LEXICON.items():
            assert len(vad) == 3, f"Word '{word}' has {len(vad)} values"

    def test_emotion_map_has_fallback(self):
        # Last entry should be "neutral" (always True)
        name, predicate = _EMOTION_MAP[-1]
        assert name == "neutral"
        assert predicate(0, 0, 0)


class TestEmotionEngine:
    def test_initial_state(self):
        engine = EmotionEngine(node_id="test_emotion")
        assert engine.state.primary_emotion == "neutral"

    def test_update_from_positive_text(self):
        engine = EmotionEngine(node_id="test_emotion")
        engine._update_from_text("I am so happy and excited about this!")
        assert engine.state.valence > 0

    def test_update_from_negative_text(self):
        engine = EmotionEngine(node_id="test_emotion")
        engine._update_from_text("This is frustrating and terrible")
        assert engine.state.valence < 0

    def test_decay_on_neutral_text(self):
        engine = EmotionEngine(node_id="test_emotion")
        engine.state.valence = 0.5
        engine._update_from_text("the cat sat on the mat")  # No emotion words
        assert engine.state.valence < 0.5  # Should decay

    def test_history_tracking(self):
        engine = EmotionEngine(node_id="test_emotion")
        engine._update_from_text("happy")
        engine._update_from_text("sad")
        assert len(engine._history) == 2

    def test_adaptation_hints_neutral(self):
        engine = EmotionEngine(node_id="test_emotion")
        hints = engine.get_adaptation_hints()
        assert hints["tone"] == "neutral"

    def test_adaptation_hints_empathetic(self):
        engine = EmotionEngine(node_id="test_emotion")
        engine.state.valence = -0.5
        hints = engine.get_adaptation_hints()
        assert hints["tone"] == "empathetic"
        assert hints["empathy_level"] == "high"

    def test_adaptation_hints_enthusiastic(self):
        engine = EmotionEngine(node_id="test_emotion")
        engine.state.valence = 0.7
        hints = engine.get_adaptation_hints()
        assert hints["tone"] == "enthusiastic"

    def test_adaptation_hints_frustration(self):
        engine = EmotionEngine(node_id="test_emotion")
        engine.state.primary_emotion = "frustration"
        hints = engine.get_adaptation_hints()
        assert hints["tone"] == "patient"
        assert hints["encouragement"]

    def test_adaptation_hints_confusion(self):
        engine = EmotionEngine(node_id="test_emotion")
        engine.state.primary_emotion = "confusion"
        hints = engine.get_adaptation_hints()
        assert hints["tone"] == "clarifying"

    def test_stats(self):
        engine = EmotionEngine(node_id="test_emotion")
        stats = engine.stats()
        assert "current_state" in stats
        assert "adaptation_hints" in stats
        assert "trend" in stats

    def test_trend_insufficient_data(self):
        engine = EmotionEngine(node_id="test_emotion")
        assert engine._compute_trend() == "insufficient_data"

    def test_trend_positive(self):
        engine = EmotionEngine(node_id="test_emotion")
        for _ in range(5):
            engine._update_from_text("happy amazing excellent")
        assert engine._compute_trend() == "positive"

    def test_trend_negative(self):
        engine = EmotionEngine(node_id="test_emotion")
        for _ in range(5):
            engine._update_from_text("terrible frustrated angry")
        assert engine._compute_trend() == "negative"

    def test_emotion_classification_joy(self):
        engine = EmotionEngine(node_id="test_emotion")
        engine.state.valence = 0.7
        engine.state.arousal = 0.5
        engine._classify_emotion()
        assert engine.state.primary_emotion == "joy"

    def test_emotion_classification_frustration(self):
        engine = EmotionEngine(node_id="test_emotion")
        engine.state.valence = -0.5
        engine.state.arousal = 0.7
        engine._classify_emotion()
        assert engine.state.primary_emotion == "frustration"

    def test_clamping(self):
        engine = EmotionEngine(node_id="test_emotion")
        # Feed many strongly negative words
        for _ in range(20):
            engine._update_from_text("hate terrible awful angry")
        assert engine.state.valence >= -1.0
        assert engine.state.arousal <= 1.0
        assert engine.state.dominance >= 0.0
