"""
Pre-training loop — trains the HBLLM transformer from scratch.

Implements:
- AdamW optimizer with weight decay
- Cosine learning rate schedule with warmup
- BF16 mixed precision training
- Gradient accumulation
- Gradient clipping
- Periodic checkpointing
- Periodic evaluation
- Weights & Biases logging
- PyTorch FSDP support for multi-GPU
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from hbllm.model.config import ModelConfig
from hbllm.model.transformer import HBLLMForCausalLM

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for the pre-training loop."""

    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    # LR Schedule
    warmup_steps: int = 2000
    max_steps: int = 100_000
    min_lr_ratio: float = 0.1

    # Batch
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 4

    # Precision
    precision: str = "bf16"

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Checkpointing
    save_interval_steps: int = 1000
    eval_interval_steps: int = 500
    checkpoint_dir: str = "./checkpoints"
    max_checkpoints: int = 5

    # Logging
    log_interval_steps: int = 10
    wandb_project: str | None = None
    wandb_entity: str | None = None

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps


class CosineWarmupScheduler:
    """
    Learning rate scheduler: linear warmup then cosine decay.

    LR = warmup_lr during [0, warmup_steps)
    LR = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress)) during [warmup_steps, max_steps)
    """

    def __init__(
        self,
        warmup_steps: int,
        max_steps: int,
        max_lr: float,
        min_lr_ratio: float = 0.1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = max_lr * min_lr_ratio

    def get_lr(self, step: int) -> float:
        """Calculate the learning rate for the given step."""
        if step < self.warmup_steps:
            # Linear warmup
            return self.max_lr * step / max(1, self.warmup_steps)
        elif step >= self.max_steps:
            return self.min_lr
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def __call__(self, step: int) -> float:
        """For use with LambdaLR — returns multiplier."""
        return self.get_lr(step) / self.max_lr


class CheckpointManager:
    """Manages model checkpoints with rotation."""

    def __init__(self, checkpoint_dir: str | Path, max_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self._saved: list[Path] = []

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step: int,
        loss: float,
        config: dict[str, Any],
    ) -> Path:
        """Save a checkpoint and rotate old ones."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint-{step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save(model.state_dict(), checkpoint_path / "model.pt")

        # Save optimizer
        torch.save(optimizer.state_dict(), checkpoint_path / "optimizer.pt")

        # Save training state
        training_state = {
            "step": step,
            "loss": loss,
            "config": config,
        }
        with open(checkpoint_path / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)

        self._saved.append(checkpoint_path)
        logger.info("Saved checkpoint at step %d to %s", step, checkpoint_path)

        # Rotate old checkpoints
        while len(self._saved) > self.max_checkpoints:
            old = self._saved.pop(0)
            if old.exists():
                import shutil
                shutil.rmtree(old)
                logger.info("Removed old checkpoint: %s", old)

        return checkpoint_path

    def load_latest(self) -> Path | None:
        """Find the latest checkpoint."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint-*"))
        return checkpoints[-1] if checkpoints else None


class Trainer:
    """
    Pre-training loop for HBLLM.

    Handles the full training cycle: optimizer setup, LR scheduling,
    mixed precision, gradient accumulation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: HBLLMForCausalLM,
        config: TrainingConfig,
        device: torch.device | None = None,
    ):
        self.model = model
        self.config = config
        self.device = device or self._auto_device()

        # Move model to device
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler
        self.lr_scheduler = CosineWarmupScheduler(
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            max_lr=config.learning_rate,
            min_lr_ratio=config.min_lr_ratio,
        )
        self.torch_scheduler = LambdaLR(self.optimizer, self.lr_scheduler)

        # Mixed precision
        self.scaler = None
        if config.precision == "fp16" and self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler()

        # Checkpoint manager
        self.checkpoint_mgr = CheckpointManager(
            config.checkpoint_dir, config.max_checkpoints
        )

        # Training state
        self.global_step = 0
        self.tokens_seen = 0

        # W&B
        self._wandb_run = None
        if config.wandb_project:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    config=config.__dict__
                )
            except ImportError:
                logger.warning("wandb is not installed. Run `pip install wandb` to enable W&B logging.")

    def _auto_device(self) -> torch.device:
        """Automatically select the best device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer with weight decay on non-bias/norm params."""
        # Separate params: apply weight decay only to 2D+ params (not biases/norms)
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or "bias" in name or "layernorm" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )

    def _get_amp_context(self):
        """Get the appropriate autocast context for mixed precision."""
        if self.config.precision == "bf16" and self.device.type == "cuda":
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        elif self.config.precision == "fp16" and self.device.type == "cuda":
            return torch.amp.autocast("cuda", dtype=torch.float16)
        else:
            import contextlib
            return contextlib.nullcontext()

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Execute a single training step (may span multiple micro-batches).

        Args:
            batch: Dict with 'input_ids' and 'labels' tensors

        Returns:
            Dict with 'loss', 'lr', 'grad_norm'
        """
        self.model.train()

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        with self._get_amp_context():
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

        # Backward
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        metrics = {"loss": loss.item() * self.config.gradient_accumulation_steps}

        return metrics

    def step(self) -> dict[str, float]:
        """
        Execute optimizer step after gradient accumulation.

        Returns:
            Dict with 'grad_norm', 'lr'
        """
        # Gradient clipping
        if self.scaler:
            self.scaler.unscale_(self.optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )

        # Optimizer step
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.torch_scheduler.step()
        self.optimizer.zero_grad()

        self.global_step += 1

        metrics = {
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=self.global_step)
            
        return metrics

    def save_checkpoint(self, loss: float = 0.0) -> Path:
        """Save a training checkpoint."""
        return self.checkpoint_mgr.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.torch_scheduler,
            step=self.global_step,
            loss=loss,
            config=self.config.__dict__,
        )

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Resume training from a checkpoint."""
        # Load model
        model_state = torch.load(
            checkpoint_path / "model.pt", map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(model_state)

        # Load optimizer
        opt_state = torch.load(
            checkpoint_path / "optimizer.pt", map_location=self.device, weights_only=True
        )
        self.optimizer.load_state_dict(opt_state)

        # Load training state
        with open(checkpoint_path / "training_state.json") as f:
            state = json.load(f)
            self.global_step = state["step"]

        logger.info("Resumed from checkpoint at step %d", self.global_step)
