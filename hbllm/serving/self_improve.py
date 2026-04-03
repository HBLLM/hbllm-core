"""
Offline Self-Improvement Worker.

Scans the reflection directory for datasets dumped by the MetaReasoningNode,
loads the base model with domain-specific LoRA adapters, runs actual SFT
training on the reflection data, and saves the updated adapter weights.

Usage:
    python -m hbllm.serving.self_improve
    python -m hbllm.serving.self_improve --reflection-dir workspace/reflection --model-size 125m
"""

from __future__ import annotations

import glob
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("self_improve")


def _load_reflection_data(ds_path: str) -> tuple[str, list[dict]]:
    """Load a reflection JSONL file and return (domain, samples)."""
    domain = "unknown"
    samples = []
    with open(ds_path) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            domain = data.get("domain", domain)
            samples.append(data)
    return domain, samples


def _convert_to_sft_format(samples: list[dict]) -> list[dict]:
    """Convert reflection data to instruction-tuning format."""
    sft_data = []
    for s in samples:
        instruction = s.get("query", s.get("instruction", s.get("input", "")))
        response = s.get("improved_response", s.get("response", s.get("output", "")))
        context = s.get("context", "")

        if instruction and response:
            sft_data.append(
                {
                    "instruction": instruction,
                    "input": context,
                    "output": response,
                }
            )
    return sft_data


def run_improvement_cycle(
    reflection_dir: str = "workspace/reflection",
    model_size: str = "125m",
    checkpoint_dir: str = "./checkpoints/self_improve",
    use_lora: bool = True,
    lora_r: int = 8,
    max_steps: int = 100,
    lr: float = 2e-5,
    batch_size: int = 2,
) -> dict[str, Any]:
    """
    Scan for weakness datasets and perform real offline learning.

    Args:
        reflection_dir: Directory containing reflection JSONL datasets.
        model_size: Model preset size (125m, 500m, 1.5b).
        checkpoint_dir: Where to save trained checkpoints.
        use_lora: Whether to use LoRA adapters (recommended).
        lora_r: LoRA rank.
        max_steps: Max training steps per dataset.
        lr: Learning rate.
        batch_size: Training batch size.

    Returns:
        Dict with training results per domain.
    """
    if not os.path.exists(reflection_dir):
        logger.info("No reflection directory found at %s. Nothing to improve.", reflection_dir)
        return {}

    datasets = glob.glob(os.path.join(reflection_dir, "*.jsonl"))
    if not datasets:
        logger.info("No reflection datasets pending. System is optimal.")
        return {}

    archive_dir = os.path.join(reflection_dir, "archive")
    os.makedirs(archive_dir, exist_ok=True)

    results: dict[str, Any] = {}

    for ds_path in datasets:
        logger.info("=" * 60)
        logger.info("Found reflection dataset: %s", ds_path)

        # 1. Load and parse the reflection data
        domain, raw_samples = _load_reflection_data(ds_path)
        sft_data = _convert_to_sft_format(raw_samples)

        if not sft_data:
            logger.warning("No valid training samples in %s — skipping", ds_path)
            continue

        logger.info("Domain: %s | Training samples: %d", domain.upper(), len(sft_data))

        # 2. Run actual training
        try:
            domain_result = _train_on_domain(
                domain=domain,
                sft_data=sft_data,
                model_size=model_size,
                checkpoint_dir=checkpoint_dir,
                use_lora=use_lora,
                lora_r=lora_r,
                max_steps=min(max_steps, len(sft_data) * 3),  # Scale to data size
                lr=lr,
                batch_size=batch_size,
            )
            results[domain] = domain_result
        except Exception as e:
            logger.error("Training failed for domain '%s': %s", domain, e)
            results[domain] = {"error": str(e)}

        # 3. Archive the processed dataset
        archived_path = os.path.join(archive_dir, os.path.basename(ds_path))
        shutil.move(ds_path, archived_path)
        logger.info("Archived dataset to %s", archived_path)
        logger.info("=" * 60 + "\n")

    return results


def _train_on_domain(
    domain: str,
    sft_data: list[dict],
    model_size: str,
    checkpoint_dir: str,
    use_lora: bool,
    lora_r: int,
    max_steps: int,
    lr: float,
    batch_size: int,
) -> dict[str, Any]:
    """Run SFT training for a single domain."""
    import torch
    from torch.utils.data import DataLoader

    from hbllm.model.config import get_config
    from hbllm.model.tokenizer import Tokenizer
    from hbllm.model.transformer import HBLLMForCausalLM
    from hbllm.training.sft import InstructionDataset, collate_sft
    from hbllm.training.trainer import Trainer, TrainingConfig

    # Load model
    config = get_config(model_size)
    logger.info("Loading %s model (%s params)...", config.name, f"{config.num_params_estimate:,}")

    model = HBLLMForCausalLM(config)
    tokenizer = Tokenizer()

    # Check for existing checkpoint
    domain_ckpt_dir = os.path.join(checkpoint_dir, domain)
    base_ckpt = Path(checkpoint_dir) / "base"
    if base_ckpt.exists():
        latest = sorted(base_ckpt.glob("step_*.pt"))
        if latest:
            logger.info("Loading base checkpoint: %s", latest[-1])
            from hbllm.utils.checkpoint import extract_model_state, load_checkpoint

            ckpt = load_checkpoint(latest[-1])
            model.load_state_dict(extract_model_state(ckpt), strict=False)

    # Inject LoRA if requested
    if use_lora:
        from hbllm.modules.lora import LoRAManager

        injected = LoRAManager.inject(model, r=lora_r)
        logger.info("Injected LoRA (r=%d) into %d modules", lora_r, len(injected))

        # Load existing domain adapter if available
        adapter_path = Path(domain_ckpt_dir) / "lora_adapter.pt"
        if adapter_path.exists():
            logger.info("Loading existing LoRA adapter for domain '%s'", domain)
            state = torch.load(adapter_path, map_location="cpu", weights_only=True)
            LoRAManager.load_lora_state_dict(model, state)

    # Build dataset and dataloader
    dataset = InstructionDataset(sft_data, tokenizer, max_length=512)
    pad_id = getattr(tokenizer, "pad_id", 0)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_sft(b, pad_id=pad_id),
    )

    # Configure training
    train_config = TrainingConfig(
        learning_rate=lr,
        max_steps=max_steps,
        batch_size=batch_size,
        gradient_accumulation_steps=1,
        checkpoint_dir=domain_ckpt_dir,
        log_interval_steps=5,
        eval_interval_steps=max_steps + 1,  # Skip eval during self-improve
    )

    # Train
    trainer = Trainer(model, train_config)
    logger.info("Starting SFT training for domain '%s' (%d steps)...", domain, max_steps)

    step = 0
    total_loss = 0.0
    losses = []

    for epoch in range(max(1, max_steps // max(len(loader), 1) + 1)):
        for batch in loader:
            if step >= max_steps:
                break

            result = trainer.train_step(batch)
            loss = result.get("loss", 0.0)
            total_loss += loss
            losses.append(loss)
            step += 1

            if step % train_config.log_interval_steps == 0:
                avg = total_loss / step
                logger.info(
                    "  [%s] Step %d/%d  loss=%.4f  avg=%.4f  lr=%.2e",
                    domain,
                    step,
                    max_steps,
                    loss,
                    avg,
                    result.get("lr", lr),
                )

            # Optimizer step after accumulation
            if step % train_config.gradient_accumulation_steps == 0:
                trainer.step()

        if step >= max_steps:
            break

    # Save
    os.makedirs(domain_ckpt_dir, exist_ok=True)

    if use_lora:
        from hbllm.modules.lora import LoRAManager

        adapter_state = LoRAManager.get_lora_state_dict(model)
        save_path = Path(domain_ckpt_dir) / "lora_adapter.pt"
        torch.save(adapter_state, save_path)
        logger.info("Saved LoRA adapter to %s (%d params)", save_path, len(adapter_state))
    else:
        trainer.save_checkpoint(loss=losses[-1] if losses else 0.0)

    avg_loss = total_loss / max(step, 1)
    logger.info(
        "Training complete for domain '%s': %d steps, avg_loss=%.4f",
        domain,
        step,
        avg_loss,
    )

    return {
        "domain": domain,
        "steps": step,
        "avg_loss": round(avg_loss, 4),
        "final_loss": round(losses[-1], 4) if losses else 0.0,
        "samples": len(sft_data),
        "lora": use_lora,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HBLLM Self-Improvement Worker")
    parser.add_argument("--reflection-dir", default="workspace/reflection")
    parser.add_argument("--model-size", default="125m", choices=["125m", "500m", "1.5b"])
    parser.add_argument("--checkpoint-dir", default="./checkpoints/self_improve")
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    logger.info("Starting HBLLM Self-Improvement Worker...")
    results = run_improvement_cycle(
        reflection_dir=args.reflection_dir,
        model_size=args.model_size,
        checkpoint_dir=args.checkpoint_dir,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        max_steps=args.max_steps,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    if results:
        logger.info("Self-improvement results:")
        for domain, r in results.items():
            logger.info("  %s: %s", domain, r)
    else:
        logger.info("No improvements were needed.")
