"""
Tests for Trained Process Reward Model (v3 Step 4).

Tests cover:
  - RewardNetwork: feature encoding, scoring, layer structure
  - TrainedPRM: evaluation, outcome recording, blend, fallback
  - TrainingCollector: record, buffer, persistence
  - Integration: ExpressionStream with TrainedPRM
"""

from __future__ import annotations

import time

import pytest

from hbllm.brain.snn.expression.models import ThoughtFragment, ThoughtGoal
from hbllm.brain.snn.expression.reward_evaluator import RewardEvaluator
from hbllm.brain.snn.expression.trained_prm import (
    RewardNetwork,
    TrainedPRM,
    TrainingCollector,
    TrainingExample,
)
from hbllm.brain.snn.network import SpikingNetwork


# ═══════════════════════════════════════════════════════════════════════════
# RewardNetwork Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRewardNetwork:
    """Test the 4-layer SNN reward scorer."""

    def test_score_returns_valid_keys(self) -> None:
        net = RewardNetwork()
        result = net.score({
            "heuristic_relevance": 0.8,
            "heuristic_coherence": 0.7,
            "heuristic_completeness": 0.6,
            "heuristic_conciseness": 0.5,
            "goal_salience": 0.9,
            "text_length_ratio": 0.5,
        })

        assert "accept_score" in result
        assert "revise_score" in result
        assert "reward" in result
        assert 0.0 <= result["reward"] <= 1.0

    def test_high_features_produce_reward(self) -> None:
        net = RewardNetwork()
        result = net.score({
            "heuristic_relevance": 1.0,
            "heuristic_coherence": 1.0,
            "heuristic_completeness": 1.0,
            "heuristic_conciseness": 1.0,
            "goal_salience": 1.0,
            "text_length_ratio": 0.5,
        })
        assert result["reward"] > 0.3

    def test_empty_features_valid(self) -> None:
        net = RewardNetwork()
        result = net.score({})
        assert 0.0 <= result["reward"] <= 1.0

    def test_network_structure(self) -> None:
        net = RewardNetwork()
        snn = net.network

        assert isinstance(snn, SpikingNetwork)
        assert "input" in snn.layer_names
        assert "hidden" in snn.layer_names
        assert "quality" in snn.layer_names
        assert "output" in snn.layer_names

    def test_reset(self) -> None:
        net = RewardNetwork()
        net.score({"heuristic_relevance": 0.9})
        net.reset()
        # Should not crash after reset
        result = net.score({"heuristic_relevance": 0.5})
        assert 0.0 <= result["reward"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# TrainingCollector Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainingCollector:
    """Test the circular buffer for training examples."""

    def test_record_and_count(self) -> None:
        collector = TrainingCollector()
        collector.record({"relevance": 0.8}, accepted=True, reward_score=0.75)
        collector.record({"relevance": 0.3}, accepted=False, reward_score=0.25)
        assert collector.count == 2

    def test_circular_buffer(self) -> None:
        collector = TrainingCollector(max_size=3)
        for i in range(5):
            collector.record({"step": float(i)}, accepted=True)
        assert collector.count == 3
        # Should keep the last 3
        examples = collector.get_all()
        assert examples[0].features["step"] == 2.0

    def test_accept_rate(self) -> None:
        collector = TrainingCollector()
        collector.record({}, accepted=True)
        collector.record({}, accepted=True)
        collector.record({}, accepted=False)
        assert collector.accept_rate == pytest.approx(2.0 / 3.0)

    def test_get_recent(self) -> None:
        collector = TrainingCollector()
        for i in range(10):
            collector.record({"i": float(i)}, accepted=True)
        recent = collector.get_recent(3)
        assert len(recent) == 3
        assert recent[0].features["i"] == 7.0

    def test_persistence(self, tmp_path) -> None:
        collector = TrainingCollector()
        collector.record({"a": 1.0}, accepted=True, reward_score=0.9)
        collector.record({"b": 2.0}, accepted=False, reward_score=0.3)

        path = tmp_path / "training.json"
        collector.save(path)
        assert path.exists()

        loaded = TrainingCollector()
        loaded.load(path)
        assert loaded.count == 2
        assert loaded.get_all()[0].features["a"] == 1.0

    def test_example_serialization(self) -> None:
        ex = TrainingExample(
            features={"test": 0.5},
            accepted=True,
            reward_score=0.8,
            timestamp=time.time(),
        )
        d = ex.to_dict()
        restored = TrainingExample.from_dict(d)
        assert restored.features["test"] == 0.5
        assert restored.accepted is True


# ═══════════════════════════════════════════════════════════════════════════
# TrainedPRM Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainedPRM:
    """Test the hybrid heuristic + SNN reward model."""

    @pytest.fixture
    def prm(self):
        evaluator = RewardEvaluator(min_acceptable_reward=0.4)
        return TrainedPRM(
            reward_evaluator=evaluator,
            fallback_threshold=10,
            snn_blend_weight=0.6,
        )

    def test_evaluate_returns_fragment(self, prm) -> None:
        goal = ThoughtGoal(
            id="test_1",
            text="Explain SNN architecture",
            source_concept_text="spiking neural networks",
            max_tokens=200,
        )
        fragment = prm.evaluate(
            "Spiking neural networks use LIF neurons for signal processing",
            goal,
        )

        assert isinstance(fragment, ThoughtFragment)
        assert 0.0 <= fragment.reward_score <= 1.0
        assert "prm_snn_reward" in fragment.metadata
        assert "prm_blend_weight" in fragment.metadata

    def test_initial_blend_is_heuristic(self, prm) -> None:
        """With no training data, blend weight should be ~0.0."""
        goal = ThoughtGoal(
            text="test goal",
            source_concept_text="test",
            max_tokens=200,
        )
        fragment = prm.evaluate("test text about testing", goal)

        # blend weight should be 0 with no training
        assert fragment.metadata["prm_blend_weight"] == pytest.approx(0.0)
        # So the reward should equal the heuristic reward
        assert fragment.reward_score == pytest.approx(
            fragment.metadata["prm_heuristic_reward"]
        )

    def test_blend_ramps_with_training(self, prm) -> None:
        """Blend weight ramps up as training examples accumulate."""
        goal = ThoughtGoal(
            text="test", source_concept_text="test", max_tokens=200
        )

        # Record some training data
        for _ in range(5):
            frag = prm.evaluate("some test text content", goal)
            prm.record_outcome(frag, accepted=True)

        frag = prm.evaluate("another test text", goal)
        # 5 out of 10 threshold = 50% * 0.6 blend = 0.3
        assert frag.metadata["prm_blend_weight"] == pytest.approx(0.3)

    def test_record_outcome_accepted(self, prm) -> None:
        goal = ThoughtGoal(
            text="test", source_concept_text="test", max_tokens=200
        )
        fragment = prm.evaluate("good text", goal)
        prm.record_outcome(fragment, accepted=True)

        assert prm.collector.count == 1
        assert prm.collector.get_all()[0].accepted is True

    def test_record_outcome_revised(self, prm) -> None:
        goal = ThoughtGoal(
            text="test", source_concept_text="test", max_tokens=200
        )
        fragment = prm.evaluate("bad text", goal)
        prm.record_outcome(fragment, accepted=False)

        assert prm.collector.count == 1
        assert prm.collector.get_all()[0].accepted is False

    def test_should_revise_low_reward(self, prm) -> None:
        """Fragments below threshold should be revised."""
        fragment = ThoughtFragment(reward_score=0.2)
        assert prm.should_revise(fragment) is True

    def test_should_not_revise_high_reward(self, prm) -> None:
        fragment = ThoughtFragment(reward_score=0.8)
        assert prm.should_revise(fragment) is False

    def test_properties_accessible(self, prm) -> None:
        assert isinstance(prm.reward_network, RewardNetwork)
        assert isinstance(prm.collector, TrainingCollector)
        assert isinstance(prm.heuristic, RewardEvaluator)


# ═══════════════════════════════════════════════════════════════════════════
# Integration: ExpressionStream + TrainedPRM
# ═══════════════════════════════════════════════════════════════════════════


class TestExpressionStreamPRMIntegration:
    """Test ExpressionStream with TrainedPRM wired in."""

    @pytest.mark.asyncio
    async def test_stream_with_trained_prm(self) -> None:
        from hbllm.brain.snn.expression import (
            ExpressionStream,
            RewardEvaluator,
            ThoughtController,
            ThoughtPlanner,
        )
        from hbllm.brain.snn.comprehension.models import (
            ComprehensionUnit,
            UnderstandingState,
        )
        import numpy as np

        evaluator = RewardEvaluator(min_acceptable_reward=0.4)
        prm = TrainedPRM(reward_evaluator=evaluator)

        stream = ExpressionStream(
            planner=ThoughtPlanner(),
            controller=ThoughtController(),
            evaluator=evaluator,
            trained_prm=prm,
        )

        state = UnderstandingState(
            concepts=[
                ComprehensionUnit(
                    text="SNN architecture",
                    embedding=np.zeros(384),
                    salience=1.2,
                ),
            ],
            salience_map=[1.2],
        )

        result = await stream.express(
            understanding=state,
            base_thought="Spiking neural networks use LIF neurons for temporal coding.",
            original_query="How do SNNs work?",
        )

        assert result.text
        assert result.thought_count >= 0

    @pytest.mark.asyncio
    async def test_stream_without_prm_still_works(self) -> None:
        from hbllm.brain.snn.expression import (
            ExpressionStream,
            RewardEvaluator,
            ThoughtController,
            ThoughtPlanner,
        )
        from hbllm.brain.snn.comprehension.models import (
            ComprehensionUnit,
            UnderstandingState,
        )
        import numpy as np

        stream = ExpressionStream(
            planner=ThoughtPlanner(),
            controller=ThoughtController(),
            evaluator=RewardEvaluator(),
        )

        state = UnderstandingState(
            concepts=[
                ComprehensionUnit(
                    text="test concept",
                    embedding=np.zeros(384),
                ),
            ],
            salience_map=[1.0],
        )

        result = await stream.express(
            understanding=state,
            base_thought="Test response.",
            original_query="test",
        )

        assert result.text
