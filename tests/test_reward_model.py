"""Tests for RewardModel — preference learning and implicit feedback."""

import os
import tempfile
import pytest
from hbllm.training.reward_model import RewardModel, RewardSignal, PreferencePair


@pytest.fixture
def reward_model(tmp_path):
    return RewardModel(data_dir=str(tmp_path))


class TestRewardSignalRecording:
    def test_record_explicit_reward(self, reward_model):
        signal = RewardSignal(query="What is Python?", response="A programming language.", reward=0.9)
        reward_model.record_reward(signal)
        stats = reward_model.stats()
        assert stats["total_rewards"] == 1
        assert stats["avg_reward"] == 0.9

    def test_record_multiple_rewards(self, reward_model):
        for i in range(5):
            reward_model.record_reward(RewardSignal(
                query=f"query {i}", response=f"response {i}", reward=0.2 * i,
            ))
        stats = reward_model.stats()
        assert stats["total_rewards"] == 5

    def test_record_preference_pair(self, reward_model):
        pair = PreferencePair(
            query="Explain AI", chosen="AI is a field of...", rejected="I don't know",
        )
        reward_model.record_preference(pair)
        stats = reward_model.stats()
        assert stats["total_preferences"] == 1


class TestImplicitFeedback:
    def test_thumbs_up(self, reward_model):
        r = reward_model.infer_implicit_reward("q", "r", {"thumbs": "up"})
        assert r > 0.5

    def test_thumbs_down(self, reward_model):
        r = reward_model.infer_implicit_reward("q", "r", {"thumbs": "down"})
        assert r < 0

    def test_regenerated(self, reward_model):
        r = reward_model.infer_implicit_reward("q", "r", {"regenerated": True})
        assert r < 0

    def test_copied(self, reward_model):
        r = reward_model.infer_implicit_reward("q", "r", {"copied": True})
        assert r > 0

    def test_reading_time_positive(self, reward_model):
        r = reward_model.infer_implicit_reward("q", "r", {"time_on_response": 30})
        assert r > 0


class TestScoring:
    def test_score_response(self, reward_model):
        score = reward_model.score_response("What is Python?", "Python is a programming language.")
        assert 0.0 <= score <= 1.0

    def test_very_short_response_penalized(self, reward_model):
        short = reward_model.score_response("Explain AI", "No")
        normal = reward_model.score_response("Explain AI", "Artificial intelligence is a field of computer science focused on creating intelligent machines.")
        assert normal > short


class TestExport:
    def test_export_preferences(self, reward_model):
        for i in range(3):
            reward_model.record_preference(PreferencePair(
                query=f"q{i}", chosen=f"good{i}", rejected=f"bad{i}", margin=0.8,
            ))
        data = reward_model.export_preferences(min_margin=0.5)
        assert len(data) == 3
        assert "query" in data[0]

    def test_export_rewards(self, reward_model):
        for i in range(3):
            reward_model.record_reward(RewardSignal(
                query=f"q{i}", response=f"r{i}", reward=0.7 + 0.1 * i,
            ))
        data = reward_model.export_rewards(min_reward=0.7)
        assert len(data) == 3
