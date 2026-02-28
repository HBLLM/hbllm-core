"""
Comprehensive tests for the HBLLM training components.

Covers: CosineWarmupScheduler, CheckpointManager, Trainer, and model training.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest
import torch

from hbllm.model.config import ModelConfig
from hbllm.model.transformer import HBLLMForCausalLM
from hbllm.training.trainer import (
    CheckpointManager,
    CosineWarmupScheduler,
    Trainer,
    TrainingConfig,
)


# Use a small model for fast tests
def _small_config() -> ModelConfig:
    return ModelConfig(
        name="test-tiny",
        vocab_size=256,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
    )


def _small_model() -> HBLLMForCausalLM:
    return HBLLMForCausalLM(_small_config())


# ──────────────────────────────────────────────
# Learning Rate Schedule tests
# ──────────────────────────────────────────────

class TestCosineWarmupScheduler:
    def test_warmup_starts_at_zero(self):
        sched = CosineWarmupScheduler(warmup_steps=100, max_steps=1000, max_lr=1e-3)
        assert sched.get_lr(0) == 0.0

    def test_warmup_linear_increase(self):
        sched = CosineWarmupScheduler(warmup_steps=100, max_steps=1000, max_lr=1e-3)
        lr_50 = sched.get_lr(50)
        lr_100 = sched.get_lr(99)
        assert lr_50 < lr_100

    def test_peak_at_warmup_end(self):
        sched = CosineWarmupScheduler(warmup_steps=100, max_steps=1000, max_lr=1e-3)
        lr_at_peak = sched.get_lr(100)
        assert abs(lr_at_peak - 1e-3) < 1e-6

    def test_cosine_decay(self):
        sched = CosineWarmupScheduler(warmup_steps=100, max_steps=1000, max_lr=1e-3)
        lr_200 = sched.get_lr(200)
        lr_500 = sched.get_lr(500)
        lr_900 = sched.get_lr(900)
        assert lr_200 > lr_500 > lr_900

    def test_minimum_lr(self):
        sched = CosineWarmupScheduler(
            warmup_steps=100, max_steps=1000, max_lr=1e-3, min_lr_ratio=0.1
        )
        lr_end = sched.get_lr(1000)
        assert abs(lr_end - 1e-4) < 1e-6  # 0.1 * 1e-3

    def test_past_max_steps(self):
        sched = CosineWarmupScheduler(warmup_steps=100, max_steps=1000, max_lr=1e-3)
        lr_past = sched.get_lr(2000)
        assert lr_past == sched.get_lr(1000)  # Stays at min

    def test_callable_for_lambda_lr(self):
        sched = CosineWarmupScheduler(warmup_steps=100, max_steps=1000, max_lr=1e-3)
        multiplier = sched(100)
        assert abs(multiplier - 1.0) < 1e-6  # Peak = multiplier 1.0


# ──────────────────────────────────────────────
# Checkpoint Manager tests
# ──────────────────────────────────────────────

class TestCheckpointManager:
    def test_save_checkpoint(self, tmp_path: Path):
        mgr = CheckpointManager(tmp_path, max_checkpoints=3)
        model = _small_model()
        optimizer = torch.optim.Adam(model.parameters())

        path = mgr.save(model, optimizer, None, step=100, loss=2.5, config={"lr": 1e-3})
        assert path.exists()
        assert (path / "model.pt").exists()
        assert (path / "optimizer.pt").exists()
        assert (path / "training_state.json").exists()

    def test_checkpoint_rotation(self, tmp_path: Path):
        mgr = CheckpointManager(tmp_path, max_checkpoints=2)
        model = _small_model()
        optimizer = torch.optim.Adam(model.parameters())

        p1 = mgr.save(model, optimizer, None, step=1, loss=3.0, config={})
        p2 = mgr.save(model, optimizer, None, step=2, loss=2.5, config={})
        p3 = mgr.save(model, optimizer, None, step=3, loss=2.0, config={})

        # Oldest should be deleted
        assert not p1.exists()
        assert p2.exists()
        assert p3.exists()

    def test_load_latest(self, tmp_path: Path):
        mgr = CheckpointManager(tmp_path, max_checkpoints=5)
        model = _small_model()
        optimizer = torch.optim.Adam(model.parameters())

        mgr.save(model, optimizer, None, step=100, loss=2.5, config={})
        mgr.save(model, optimizer, None, step=200, loss=2.0, config={})

        latest = mgr.load_latest()
        assert latest is not None
        assert "200" in latest.name

    def test_load_latest_empty(self, tmp_path: Path):
        mgr = CheckpointManager(tmp_path / "empty")
        assert mgr.load_latest() is None

    def test_training_state_content(self, tmp_path: Path):
        mgr = CheckpointManager(tmp_path)
        model = _small_model()
        optimizer = torch.optim.Adam(model.parameters())

        path = mgr.save(model, optimizer, None, step=42, loss=1.5, config={"lr": 3e-4})
        with open(path / "training_state.json") as f:
            state = json.load(f)

        assert state["step"] == 42
        assert state["loss"] == 1.5
        assert state["config"]["lr"] == 3e-4


# ──────────────────────────────────────────────
# Trainer tests
# ──────────────────────────────────────────────

class TestTrainer:
    def test_trainer_creation(self):
        model = _small_model()
        config = TrainingConfig(max_steps=10, checkpoint_dir="/tmp/hbllm_test_ckpt")
        trainer = Trainer(model, config, device=torch.device("cpu"))
        assert trainer.global_step == 0

    def test_optimizer_param_groups(self):
        model = _small_model()
        config = TrainingConfig()
        trainer = Trainer(model, config, device=torch.device("cpu"))

        # Should have 2 param groups: decay and no_decay
        assert len(trainer.optimizer.param_groups) == 2
        assert trainer.optimizer.param_groups[0]["weight_decay"] == config.weight_decay
        assert trainer.optimizer.param_groups[1]["weight_decay"] == 0.0

    def test_train_step(self, tmp_path: Path):
        model = _small_model()
        config = TrainingConfig(checkpoint_dir=str(tmp_path))
        trainer = Trainer(model, config, device=torch.device("cpu"))

        batch = {
            "input_ids": torch.randint(0, 256, (2, 32)),
            "labels": torch.randint(0, 256, (2, 32)),
        }

        metrics = trainer.train_step(batch)
        assert "loss" in metrics
        assert metrics["loss"] > 0

    def test_optimizer_step(self, tmp_path: Path):
        model = _small_model()
        config = TrainingConfig(checkpoint_dir=str(tmp_path))
        trainer = Trainer(model, config, device=torch.device("cpu"))

        batch = {
            "input_ids": torch.randint(0, 256, (2, 32)),
            "labels": torch.randint(0, 256, (2, 32)),
        }

        trainer.train_step(batch)
        step_metrics = trainer.step()

        assert "grad_norm" in step_metrics
        assert "lr" in step_metrics
        assert trainer.global_step == 1

    def test_multiple_steps_reduce_loss(self, tmp_path: Path):
        model = _small_model()
        config = TrainingConfig(
            learning_rate=1e-3,
            warmup_steps=0,
            checkpoint_dir=str(tmp_path),
        )
        trainer = Trainer(model, config, device=torch.device("cpu"))

        # Fixed batch for deterministic training
        torch.manual_seed(42)
        batch = {
            "input_ids": torch.randint(0, 256, (4, 32)),
            "labels": torch.randint(0, 256, (4, 32)),
        }

        losses = []
        for _ in range(10):
            metrics = trainer.train_step(batch)
            trainer.step()
            losses.append(metrics["loss"])

        # Loss should generally decrease over multiple steps
        assert losses[-1] < losses[0]

    def test_save_and_load_checkpoint(self, tmp_path: Path):
        model = _small_model()
        config = TrainingConfig(checkpoint_dir=str(tmp_path))
        trainer = Trainer(model, config, device=torch.device("cpu"))

        batch = {
            "input_ids": torch.randint(0, 256, (2, 32)),
            "labels": torch.randint(0, 256, (2, 32)),
        }
        trainer.train_step(batch)
        trainer.step()

        # Save
        ckpt_path = trainer.save_checkpoint(loss=2.0)
        assert ckpt_path.exists()

        # Load into new trainer
        model2 = _small_model()
        trainer2 = Trainer(model2, config, device=torch.device("cpu"))
        trainer2.load_checkpoint(ckpt_path)
        assert trainer2.global_step == 1

    def test_effective_batch_size(self):
        config = TrainingConfig(micro_batch_size=4, gradient_accumulation_steps=8)
        assert config.effective_batch_size == 32

    def test_training_config_defaults(self):
        config = TrainingConfig()
        assert config.learning_rate == 3e-4
        assert config.weight_decay == 0.1
        assert config.precision == "bf16"
        assert config.max_grad_norm == 1.0
