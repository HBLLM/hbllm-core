"""
Supervised Fine-Tuning (SFT) pipeline for HBLLM.

Trains the model on instruction-following data (Alpaca/ShareGPT format).
Supports full fine-tuning or LoRA-only fine-tuning.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# ─── Instruction Dataset ─────────────────────────────────────────────────────

class InstructionDataset(Dataset):
    """
    Dataset for instruction fine-tuning.

    Supports Alpaca format:
        {"instruction": ..., "input": ..., "output": ...}

    And ShareGPT format:
        {"conversations": [{"from": "human", "value": ...}, {"from": "gpt", "value": ...}]}
    """

    def __init__(
        self,
        data: list[dict],
        tokenizer: Any,
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._prepare(data)

    def _prepare(self, data: list[dict]) -> list[dict]:
        """Convert raw data to tokenized examples."""
        examples = []
        for item in data:
            messages = self._to_messages(item)
            if not messages:
                continue

            # Encode full conversation
            full_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False)
            full_ids = self.tokenizer.encode(full_text, add_bos=True, add_eos=True)

            # Find where the assistant response starts (for loss masking)
            # We compute loss only on assistant tokens
            prompt_messages = [m for m in messages if m["role"] != "assistant"]
            prompt_text = self.tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True)
            prompt_ids = self.tokenizer.encode(prompt_text, add_bos=True)
            prompt_len = len(prompt_ids)

            # Truncate if needed
            if len(full_ids) > self.max_length:
                full_ids = full_ids[:self.max_length]

            examples.append({
                "input_ids": full_ids,
                "prompt_len": prompt_len,
            })

        logger.info("Prepared %d SFT examples (from %d raw)", len(examples), len(data))
        return examples

    def _to_messages(self, item: dict) -> list[dict[str, str]]:
        """Convert item to messages format."""
        # Alpaca format
        if "instruction" in item:
            messages = []
            prompt = item["instruction"]
            if item.get("input"):
                prompt += "\n\n" + item["input"]
            messages.append({"role": "user", "content": prompt})
            if item.get("output"):
                messages.append({"role": "assistant", "content": item["output"]})
            return messages

        # ShareGPT format
        if "conversations" in item:
            messages = []
            for turn in item["conversations"]:
                role = "user" if turn.get("from") in ("human", "user") else "assistant"
                messages.append({"role": role, "content": turn.get("value", "")})
            return messages

        # Direct messages format
        if "messages" in item:
            return item["messages"]

        return []

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        input_ids = torch.tensor(ex["input_ids"], dtype=torch.long)

        # Labels: mask prompt tokens with -100 (ignore in loss)
        labels = input_ids.clone()
        labels[:ex["prompt_len"]] = -100

        return {"input_ids": input_ids, "labels": labels}


def collate_sft(batch: list[dict[str, torch.Tensor]], pad_id: int = 0) -> dict[str, torch.Tensor]:
    """Collate and pad SFT batch."""
    max_len = max(b["input_ids"].size(0) for b in batch)

    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = b["input_ids"].size(0)
        input_ids[i, :seq_len] = b["input_ids"]
        labels[i, :seq_len] = b["labels"]

    return {"input_ids": input_ids, "labels": labels}


# ─── Dataset Loaders ─────────────────────────────────────────────────────────

def load_sft_data(name: str, max_samples: int | None = None) -> list[dict]:
    """Load instruction-following data from HuggingFace or local files."""

    if name == "alpaca":
        return _load_alpaca(max_samples)
    elif name == "sharegpt":
        return _load_sharegpt(max_samples)
    elif Path(name).exists():
        return _load_local(name, max_samples)
    else:
        raise ValueError(f"Unknown SFT dataset: {name}. Use 'alpaca', 'sharegpt', or a file path.")


def _load_alpaca(max_samples: int | None = None) -> list[dict]:
    """Load Stanford Alpaca dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        data = list(ds)
        if max_samples:
            data = data[:max_samples]
        logger.info("Loaded %d Alpaca examples", len(data))
        return data
    except Exception as e:
        logger.error("Failed to load Alpaca: %s", e)
        return []


def _load_sharegpt(max_samples: int | None = None) -> list[dict]:
    """Load ShareGPT dataset."""
    try:
        from datasets import load_dataset
        ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")
        data = list(ds)
        if max_samples:
            data = data[:max_samples]
        logger.info("Loaded %d ShareGPT examples", len(data))
        return data
    except Exception as e:
        logger.error("Failed to load ShareGPT: %s", e)
        return []


def _load_local(path: str, max_samples: int | None = None) -> list[dict]:
    """Load from a local JSONL or JSON file."""
    file_path = Path(path)
    data = []
    if file_path.suffix == ".jsonl":
        with open(file_path) as f:
            for line in f:
                data.append(json.loads(line))
    else:
        with open(file_path) as f:
            data = json.load(f)
    if max_samples:
        data = data[:max_samples]
    logger.info("Loaded %d examples from %s", len(data), path)
    return data


# ─── SFT Training Runner ────────────────────────────────────────────────────

def run_sft_training(
    model_size: str = "125m",
    dataset_name: str = "alpaca",
    use_lora: bool = True,
    lora_r: int = 8,
    max_steps: int = 1000,
    batch_size: int = 4,
    lr: float = 2e-5,
    checkpoint_dir: str = "./checkpoints/sft",
    resume: str | None = None,
    device: torch.device | None = None,
    max_samples: int | None = None,
) -> None:
    """
    Run supervised fine-tuning.

    Args:
        model_size: Model preset (125m, 500m, 1.5b)
        dataset_name: Dataset name or path
        use_lora: Use LoRA adapters (recommended)
        lora_r: LoRA rank
        max_steps: Training steps
        batch_size: Batch size
        lr: Learning rate
        checkpoint_dir: Checkpoint save directory
        resume: Resume from checkpoint path
        device: Training device
        max_samples: Max dataset samples
    """
    from hbllm.model.config import get_config
    from hbllm.model.transformer import HBLLMForCausalLM
    from hbllm.model.tokenizer import HBLLMTokenizer
    from hbllm.training.trainer import Trainer, TrainingConfig

    device = device or torch.device("cpu")

    # Load model
    logger.info("Loading model (%s)...", model_size)
    config = get_config(model_size)
    model = HBLLMForCausalLM(config)

    # Load pre-trained weights if available
    ckpt_dir = Path(checkpoint_dir).parent / "pretrained"
    if ckpt_dir.exists():
        from hbllm.training.trainer import CheckpointManager
        mgr = CheckpointManager(ckpt_dir)
        ckpt = mgr.load_latest()
        if ckpt:
            model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"], strict=False)
            logger.info("Loaded pre-trained weights from %s", ckpt)

    # Inject LoRA
    if use_lora:
        from hbllm.modules.lora import LoRAManager
        injected = LoRAManager.inject(model, r=lora_r)
        logger.info("LoRA injected into %d modules (r=%d)", len(injected), lora_r)

        # Freeze base, train only LoRA
        for name, param in model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info("Trainable: %s / %s (%.1f%%)", f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    # Load data
    logger.info("Loading SFT data (%s)...", dataset_name)
    tokenizer = HBLLMTokenizer.from_tiktoken()
    raw_data = load_sft_data(dataset_name, max_samples=max_samples)
    dataset = InstructionDataset(raw_data, tokenizer, max_length=config.max_position_embeddings)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_sft(b, pad_id=tokenizer.pad_id),
        num_workers=2,
        pin_memory=True,
    )

    # Train
    logger.info("Starting SFT (LoRA=%s, lr=%.1e, steps=%d)...", use_lora, lr, max_steps)
    train_config = TrainingConfig(
        learning_rate=lr,
        max_steps=max_steps,
        batch_size=batch_size,
        gradient_accumulation_steps=1,
        checkpoint_dir=checkpoint_dir,
    )
    trainer = Trainer(model, train_config, device=device)

    data_iter = iter(dataloader)
    for step in range(max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        metrics = trainer.train_step(batch)
        step_metrics = trainer.step()

        if (step + 1) % 10 == 0:
            logger.info(
                "SFT step %d/%d | loss=%.4f lr=%.2e",
                step + 1, max_steps, metrics["loss"], step_metrics["lr"],
            )

        if (step + 1) % 200 == 0:
            trainer.save_checkpoint(loss=metrics["loss"])

    # Final save
    trainer.save_checkpoint(loss=metrics.get("loss", 0))

    # Save LoRA weights separately
    if use_lora:
        from hbllm.modules.lora import LoRAManager
        lora_state = LoRAManager.get_lora_state_dict(model)
        lora_path = Path(checkpoint_dir) / "lora_adapter.pt"
        torch.save(lora_state, lora_path)
        logger.info("LoRA adapter saved to %s (%d params)", lora_path, len(lora_state))

    logger.info("SFT complete!")
