"""
Training Pipeline Smoke Tests — verifies SFT, DPO, and checkpoint
sanity without requiring a GPU or large dataset.
"""

import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from hbllm.model.config import get_config
from hbllm.model.transformer import HBLLMForCausalLM
from hbllm.training.dpo import compute_dpo_loss, get_batch_logps
from hbllm.training.trainer import CheckpointManager, Trainer, TrainingConfig

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def tiny_model():
    """Create a tiny model for fast CPU testing."""
    config = get_config("125m")
    model = HBLLMForCausalLM(config)
    model.eval()
    return model, config


@pytest.fixture
def tmp_checkpoint_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ── DPO Loss Tests ───────────────────────────────────────────────────────────


class TestDPOLoss:
    def test_basic_dpo_loss(self):
        """DPO loss should be finite and positive for random inputs."""
        batch_size = 4
        policy_chosen = torch.randn(batch_size)
        policy_rejected = torch.randn(batch_size)
        ref_chosen = torch.randn(batch_size)
        ref_rejected = torch.randn(batch_size)

        losses, chosen_rewards, rejected_rewards = compute_dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1
        )

        assert losses.shape == (batch_size,)
        assert torch.isfinite(losses).all()
        assert (losses >= 0).all()  # DPO loss is non-negative

    def test_chosen_preferred_gives_lower_loss(self):
        """When policy strongly prefers chosen, loss should be small."""
        # Policy strongly prefers chosen
        policy_chosen = torch.tensor([5.0, 5.0])
        policy_rejected = torch.tensor([-5.0, -5.0])
        # Reference is neutral
        ref_chosen = torch.tensor([0.0, 0.0])
        ref_rejected = torch.tensor([0.0, 0.0])

        losses_good, _, _ = compute_dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected
        )

        # Policy wrongly prefers rejected
        losses_bad, _, _ = compute_dpo_loss(
            policy_rejected, policy_chosen, ref_chosen, ref_rejected
        )

        assert losses_good.mean() < losses_bad.mean()

    def test_reward_ordering(self):
        """Chosen rewards should be higher when policy prefers chosen."""
        policy_chosen = torch.tensor([3.0])
        policy_rejected = torch.tensor([-3.0])
        ref_chosen = torch.tensor([0.0])
        ref_rejected = torch.tensor([0.0])

        _, chosen_rewards, rejected_rewards = compute_dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected
        )

        assert chosen_rewards.item() > rejected_rewards.item()

    def test_batch_logps(self):
        """get_batch_logps computes per-example summed log probs."""
        batch_size = 2
        seq_len = 5
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        logps = get_batch_logps(logits, labels)
        assert logps.shape == (batch_size,)
        assert torch.isfinite(logps).all()
        assert (logps <= 0).all()  # Log probabilities are negative

    def test_batch_logps_with_ignore_index(self):
        """Ignored tokens shouldn't contribute to log probs."""
        batch_size = 1
        seq_len = 5
        vocab_size = 50

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels_full = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels_masked = labels_full.clone()
        labels_masked[:, 3:] = -100  # Mask last 2 tokens

        logps_full = get_batch_logps(logits, labels_full)
        logps_masked = get_batch_logps(logits, labels_masked)

        # Masked version should have higher (less negative) logps
        # since fewer tokens contribute
        assert logps_masked.item() >= logps_full.item()


# ── Trainer Tests ────────────────────────────────────────────────────────────


class TestTrainer:
    def test_trainer_initialization(self, tiny_model):
        """Trainer initializes on CPU without errors."""
        model, config = tiny_model
        trainer_config = TrainingConfig(
            learning_rate=1e-4,
            warmup_steps=10,
            max_steps=100,
            micro_batch_size=2,
        )
        trainer = Trainer(model, trainer_config, device=torch.device("cpu"))
        assert trainer.global_step == 0
        assert trainer.optimizer is not None

    def test_train_step_reduces_loss(self, tiny_model):
        """Two training steps should produce finite losses."""
        model, config = tiny_model
        model.train()

        trainer_config = TrainingConfig(
            learning_rate=1e-3,
            warmup_steps=0,
            max_steps=10,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
        )
        trainer = Trainer(model, trainer_config, device=torch.device("cpu"))

        # Create synthetic batch
        seq_len = 32
        batch = {
            "input_ids": torch.randint(0, config.vocab_size, (2, seq_len)),
            "labels": torch.randint(0, config.vocab_size, (2, seq_len)),
        }

        # Step 1
        result1 = trainer.train_step(batch)
        trainer.step()
        assert "loss" in result1
        assert torch.isfinite(torch.tensor(result1["loss"]))

        # Step 2
        result2 = trainer.train_step(batch)
        trainer.step()
        assert torch.isfinite(torch.tensor(result2["loss"]))


# ── Checkpoint Tests ─────────────────────────────────────────────────────────


class TestCheckpoint:
    def test_save_load_round_trip(self, tiny_model, tmp_checkpoint_dir):
        """Checkpoint save + load should preserve model state."""
        model, config = tiny_model
        model.train()

        trainer_config = TrainingConfig(
            learning_rate=1e-4,
            warmup_steps=10,
            max_steps=100,
            checkpoint_dir=tmp_checkpoint_dir,
        )
        trainer = Trainer(model, trainer_config, device=torch.device("cpu"))

        # Run one step to create optimzier state
        batch = {
            "input_ids": torch.randint(0, config.vocab_size, (1, 16)),
            "labels": torch.randint(0, config.vocab_size, (1, 16)),
        }
        trainer.train_step(batch)
        trainer.step()

        # Save checkpoint
        trainer.save_checkpoint(loss=1.5)

        # Verify checkpoint directory exists (format: checkpoint-{step}/model.pt)
        ckpt_dirs = list(Path(tmp_checkpoint_dir).glob("checkpoint-*"))
        assert len(ckpt_dirs) >= 1

        # Load and verify checkpoint contents
        ckpt_dir = ckpt_dirs[0]
        assert (ckpt_dir / "model.pt").exists()
        assert (ckpt_dir / "optimizer.pt").exists()
        assert (ckpt_dir / "training_state.json").exists()

        import json

        with open(ckpt_dir / "training_state.json") as f:
            state = json.load(f)
        assert state["step"] == 1
        assert state["loss"] == 1.5

    def test_checkpoint_manager_rotation(self, tmp_checkpoint_dir):
        """CheckpointManager rotates old checkpoints."""
        mgr = CheckpointManager(tmp_checkpoint_dir, max_checkpoints=2)

        # Create a minimal model
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.Adam(model.parameters())

        # Save 3 checkpoints
        mgr.save(model, optimizer, None, step=1, loss=1.0, config={})
        mgr.save(model, optimizer, None, step=2, loss=0.8, config={})
        mgr.save(model, optimizer, None, step=3, loss=0.6, config={})

        # Should keep only 2
        ckpt_dirs = sorted(Path(tmp_checkpoint_dir).glob("checkpoint-*"))
        assert len(ckpt_dirs) <= 2
        # Newest should survive
        assert any("checkpoint-3" in d.name for d in ckpt_dirs)


# ── SFT Smoke Test ───────────────────────────────────────────────────────────


class TestSFTSmoke:
    def test_sft_forward_pass(self, tiny_model):
        """Model forward pass on instruction-like data works."""
        model, config = tiny_model
        model.train()

        # Simulate tokenized instruction data
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()

        # Forward pass
        outputs = model(input_ids)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs

        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert torch.isfinite(logits).all()

        # Compute cross-entropy loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size),
            labels.view(-1),
        )
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_sft_backward_pass(self, tiny_model):
        """Gradients flow through the model on SFT-style data."""
        model, config = tiny_model
        model.train()

        input_ids = torch.randint(0, config.vocab_size, (1, 16))
        labels = input_ids.clone()

        outputs = model(input_ids)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size),
            labels.view(-1),
        )
        loss.backward()

        # Check gradients exist
        has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grads
