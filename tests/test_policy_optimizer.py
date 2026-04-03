"""Tests for PolicyOptimizer — PPO, DPO, GAE, best-of-N."""

import pytest

from hbllm.training.policy_optimizer import PolicyOptimizer


@pytest.fixture
def optimizer():
    return PolicyOptimizer()


class TestBestOfN:
    def test_selects_highest_score(self, optimizer):
        candidates = ["bad", "good", "best"]
        scores = [0.2, 0.7, 0.95]
        result = optimizer.best_of_n(candidates, scores, n=1)
        assert result[0][0] == "best"

    def test_returns_n_results(self, optimizer):
        candidates = ["a", "b", "c", "d"]
        scores = [0.1, 0.9, 0.5, 0.3]
        result = optimizer.best_of_n(candidates, scores, n=2)
        assert len(result) == 2


class TestPPOLoss:
    def test_computes_loss(self, optimizer):
        result = optimizer.compute_ppo_loss(
            log_probs=[-1.0, -0.5, -2.0],
            old_log_probs=[-1.1, -0.6, -1.9],
            advantages=[0.5, -0.3, 0.8],
            values=[0.3, 0.5, 0.2],
            returns=[0.4, 0.4, 0.6],
        )
        assert "policy_loss" in result
        assert "value_loss" in result
        assert "total_loss" in result

    def test_loss_is_finite(self, optimizer):
        result = optimizer.compute_ppo_loss(
            log_probs=[-1.0], old_log_probs=[-1.0],
            advantages=[0.5], values=[0.3], returns=[0.4],
        )
        assert result["total_loss"] != float("inf")


class TestDPOLoss:
    def test_computes_dpo_loss(self, optimizer):
        loss = optimizer.compute_dpo_loss(
            chosen_logprobs=[-0.5, -0.3],
            rejected_logprobs=[-1.5, -2.0],
            ref_chosen_logprobs=[-0.6, -0.4],
            ref_rejected_logprobs=[-1.4, -1.9],
        )
        assert isinstance(loss, float)
        assert loss >= 0


class TestGAE:
    def test_computes_advantages(self, optimizer):
        advantages, returns = optimizer.compute_advantages(
            rewards=[1.0, 0.5, 0.8],
            values=[0.3, 0.4, 0.5, 0.0],
        )
        assert len(advantages) == 3
        assert len(returns) == 3

    def test_advantages_normalized(self, optimizer):
        advantages, _ = optimizer.compute_advantages(
            rewards=[1.0, 1.0, 1.0],
            values=[0.5, 0.5, 0.5, 0.0],
        )
        mean = sum(advantages) / len(advantages)
        assert abs(mean) < 0.1  # roughly zero-centered


class TestKL:
    def test_kl_same_distribution(self, optimizer):
        kl = optimizer.compute_kl_penalty([-1.0, -2.0], [-1.0, -2.0])
        assert kl < 0.01

    def test_kl_different_distribution(self, optimizer):
        kl = optimizer.compute_kl_penalty([-0.5, -1.0], [-2.0, -3.0])
        assert kl > 0


class TestStep:
    def test_records_step(self, optimizer):
        update = optimizer.step(reward_mean=0.7, loss=0.3)
        assert update.step == 1
        stats = optimizer.stats()
        assert stats["steps"] == 1
