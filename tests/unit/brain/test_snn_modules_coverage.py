"""Unit tests for brain SNN modules — style_adapter, reasoning_network, prm_trainer."""

from unittest.mock import MagicMock

from hbllm.brain.snn.expression.style_adapter import StyleAdapter, StyleHints, _in_range


class TestStyleHints:
    def test_default_hints(self):
        hints = StyleHints()
        assert hints.tone == "neutral"
        assert hints.verbosity == "moderate"
        assert hints.formality == "neutral"

    def test_to_prompt_instruction(self):
        hints = StyleHints(formality="formal", verbosity="concise")
        instruction = hints.to_prompt_instruction()
        assert isinstance(instruction, str)

    def test_to_dict(self):
        hints = StyleHints()
        d = hints.to_dict()
        assert "tone" in d
        assert "verbosity" in d


class TestInRange:
    def test_in_range_true(self):
        assert _in_range(0.5, (0.0, 1.0)) is True

    def test_in_range_false(self):
        assert _in_range(1.5, (0.0, 1.0)) is False

    def test_in_range_boundary(self):
        assert _in_range(0.0, (0.0, 1.0)) is True
        assert _in_range(1.0, (0.0, 1.0)) is True


class TestStyleAdapter:
    def test_adapt_default(self):
        adapter = StyleAdapter()
        hints = adapter.adapt()
        assert isinstance(hints, StyleHints)

    def test_adapt_positive_valence(self):
        adapter = StyleAdapter()
        hints = adapter.adapt(valence=0.8, arousal=0.3, dominance=0.5)
        assert isinstance(hints, StyleHints)

    def test_adapt_negative_valence(self):
        adapter = StyleAdapter()
        hints = adapter.adapt(valence=-0.5, arousal=0.7, dominance=0.2)
        assert isinstance(hints, StyleHints)

    def test_adapt_from_emotion_state(self):
        adapter = StyleAdapter()
        emotion = MagicMock()
        emotion.valence = 0.8
        emotion.arousal = 0.3
        emotion.dominance = 0.5
        hints = adapter.adapt_from_emotion_state(emotion)
        assert isinstance(hints, StyleHints)


# ── Reasoning Network ────────────────────────────────────────────────────────

from hbllm.brain.snn.reasoning.reasoning_network import ReasoningNetwork


class TestReasoningNetwork:
    def test_init(self):
        rn = ReasoningNetwork()
        assert rn is not None

    def test_evaluate_returns_float(self):
        rn = ReasoningNetwork()
        score = rn.evaluate({"coherence": 0.8, "relevance": 0.7, "confidence": 0.9})
        assert isinstance(score, float)

    def test_evaluate_all_zeros(self):
        rn = ReasoningNetwork()
        score = rn.evaluate({"coherence": 0.0, "relevance": 0.0, "confidence": 0.0})
        assert isinstance(score, float)

    def test_network_property(self):
        rn = ReasoningNetwork()
        net = rn.network
        assert net is not None

    def test_reset(self):
        rn = ReasoningNetwork()
        rn.evaluate({"coherence": 0.5, "relevance": 0.5, "confidence": 0.5})
        rn.reset()


# ── PRM Trainer ──────────────────────────────────────────────────────────────

from hbllm.brain.snn.expression.prm_trainer import PRMTrainer, TrainingMetrics


class TestTrainingMetrics:
    def test_creation(self):
        metrics = TrainingMetrics(examples_trained=10, epochs_completed=1)
        assert metrics.examples_trained == 10

    def test_to_dict(self):
        metrics = TrainingMetrics(examples_trained=5, pre_accuracy=0.5, post_accuracy=0.8)
        d = metrics.to_dict()
        assert "examples_trained" in d
        assert "pre_accuracy" in d


class TestPRMTrainer:
    def test_init(self):
        network = MagicMock()
        trainer = PRMTrainer(trained_prm=network)
        assert trainer is not None

    def test_should_train_no_examples(self):
        network = MagicMock()
        network.collector.count = 0
        trainer = PRMTrainer(trained_prm=network)
        result = trainer.should_train()
        assert result is False

    def test_total_sweeps_initial(self):
        network = MagicMock()
        trainer = PRMTrainer(trained_prm=network)
        assert trainer.total_sweeps == 0

    def test_get_metrics_empty(self):
        network = MagicMock()
        trainer = PRMTrainer(trained_prm=network)
        assert trainer.get_metrics() == []

    def test_last_metrics_none(self):
        network = MagicMock()
        trainer = PRMTrainer(trained_prm=network)
        assert trainer.last_metrics is None
