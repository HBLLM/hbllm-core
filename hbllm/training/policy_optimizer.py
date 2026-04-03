"""
Policy Optimizer — PPO/DPO-based policy updates from reward signals.

Integrates with RewardModel to continuously improve response quality
through reinforcement learning from human (and implicit) feedback.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PolicyUpdate:
    """Record of a policy optimization step."""

    step: int
    loss: float
    reward_mean: float
    kl_divergence: float
    timestamp: float = field(default_factory=time.time)


class PolicyOptimizer:
    """
    Optimizes the LLM's response policy using reward signals.

    Supports:
    - PPO (Proximal Policy Optimization) — clip-based updates
    - DPO (Direct Preference Optimization) — preference pairs
    - Best-of-N sampling — simple but effective reranking
    - KL penalty — prevents catastrophic forgetting
    """

    def __init__(
        self,
        learning_rate: float = 1e-5,
        kl_coeff: float = 0.1,
        clip_range: float = 0.2,
        max_grad_norm: float = 1.0,
        temperature: float = 1.0,
    ):
        self.lr = learning_rate
        self.kl_coeff = kl_coeff
        self.clip_range = clip_range
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature
        self._step = 0
        self._history: list[PolicyUpdate] = []

    # ─── Best-of-N Sampling ──────────────────────────────────────────

    def best_of_n(
        self,
        candidates: list[str],
        scores: list[float],
        n: int = 1,
    ) -> list[tuple[str, float]]:
        """
        Select the best N responses based on reward scores.

        This is the simplest policy optimization: generate multiple
        candidates, score them, and return the highest-scoring ones.
        """
        paired = list(zip(candidates, scores))
        paired.sort(key=lambda x: x[1], reverse=True)
        return paired[:n]

    # ─── PPO Loss Computation ────────────────────────────────────────

    def compute_ppo_loss(
        self,
        log_probs: list[float],
        old_log_probs: list[float],
        advantages: list[float],
        values: list[float],
        returns: list[float],
    ) -> dict[str, float]:
        """
        Compute PPO clipped objective loss.

        Returns policy_loss, value_loss, entropy, and total_loss.
        """
        policy_loss = 0.0
        value_loss = 0.0
        entropy = 0.0

        for lp, olp, adv, val, ret in zip(log_probs, old_log_probs, advantages, values, returns):
            ratio = math.exp(lp - olp)
            clipped = max(
                min(ratio, 1.0 + self.clip_range),
                1.0 - self.clip_range,
            )
            policy_loss -= min(ratio * adv, clipped * adv)
            value_loss += (val - ret) ** 2
            entropy -= lp * math.exp(lp) if lp > -20 else 0

        n = max(len(log_probs), 1)
        policy_loss /= n
        value_loss /= 2 * n
        entropy /= n

        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "total_loss": total_loss,
        }

    # ─── DPO Loss Computation ────────────────────────────────────────

    def compute_dpo_loss(
        self,
        chosen_logprobs: list[float],
        rejected_logprobs: list[float],
        ref_chosen_logprobs: list[float],
        ref_rejected_logprobs: list[float],
        beta: float = 0.1,
    ) -> float:
        """
        Compute Direct Preference Optimization loss.

        DPO directly optimizes the policy from preferences without
        needing a separate reward model.
        """
        total_loss = 0.0
        for c, r, rc, rr in zip(
            chosen_logprobs,
            rejected_logprobs,
            ref_chosen_logprobs,
            ref_rejected_logprobs,
        ):
            log_ratio_chosen = c - rc
            log_ratio_rejected = r - rr
            logit = beta * (log_ratio_chosen - log_ratio_rejected)
            # Sigmoid loss
            loss = -math.log(1.0 / (1.0 + math.exp(-logit)) + 1e-10)
            total_loss += loss

        return total_loss / max(len(chosen_logprobs), 1)

    # ─── KL Divergence ───────────────────────────────────────────────

    def compute_kl_penalty(
        self,
        current_logprobs: list[float],
        reference_logprobs: list[float],
    ) -> float:
        """
        Compute KL divergence penalty to prevent catastrophic forgetting.

        KL(current || reference) — penalizes drifting too far from base model.
        """
        kl = 0.0
        for cp, rp in zip(current_logprobs, reference_logprobs):
            kl += math.exp(cp) * (cp - rp)
        return max(0.0, kl / max(len(current_logprobs), 1))

    # ─── Advantage Estimation ────────────────────────────────────────

    def compute_advantages(
        self,
        rewards: list[float],
        values: list[float],
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> tuple[list[float], list[float]]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Returns (advantages, returns).
        """
        advantages = []
        returns_list = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
            returns_list.insert(0, gae + values[t])

        # Normalize advantages
        if advantages:
            mean_adv = sum(advantages) / len(advantages)
            std_adv = (sum((a - mean_adv) ** 2 for a in advantages) / len(advantages)) ** 0.5 + 1e-8
            advantages = [(a - mean_adv) / std_adv for a in advantages]

        return advantages, returns_list

    # ─── Training Step ───────────────────────────────────────────────

    def step(
        self,
        reward_mean: float,
        loss: float,
        kl_divergence: float = 0.0,
    ) -> PolicyUpdate:
        """Record a policy optimization step."""
        self._step += 1
        update = PolicyUpdate(
            step=self._step,
            loss=loss,
            reward_mean=reward_mean,
            kl_divergence=kl_divergence,
        )
        self._history.append(update)
        logger.info(
            "Policy step %d: loss=%.4f reward=%.3f kl=%.4f",
            self._step,
            loss,
            reward_mean,
            kl_divergence,
        )
        return update

    def stats(self) -> dict[str, Any]:
        """Return optimizer statistics."""
        if not self._history:
            return {"steps": 0}
        recent = self._history[-10:]
        return {
            "steps": self._step,
            "recent_avg_loss": sum(u.loss for u in recent) / len(recent),
            "recent_avg_reward": sum(u.reward_mean for u in recent) / len(recent),
            "recent_avg_kl": sum(u.kl_divergence for u in recent) / len(recent),
        }
