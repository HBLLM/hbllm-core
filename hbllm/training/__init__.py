"""Training — pre-training loop, optimizer, data loading, and checkpointing."""

from __future__ import annotations

from pathlib import Path

_exp_path = Path(__file__).resolve().parents[2] / "experiments" / "legacy_llm" / "training"
if _exp_path.exists() and str(_exp_path) not in __path__:
    __path__.append(str(_exp_path))


from hbllm.training.policy_optimizer import PolicyOptimizer
from hbllm.training.reward_model import PreferencePair, RewardModel, RewardSignal

__all__ = ["RewardModel", "RewardSignal", "PreferencePair", "PolicyOptimizer"]
