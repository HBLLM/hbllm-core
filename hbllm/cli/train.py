"""
HBLLM Training CLI — one-command training entry point.

Usage:
    python -m hbllm.cli.train --size 125m --data fineweb --max-samples 100000
    python -m hbllm.cli.train --size 125m --resume checkpoints/
    python -m hbllm.cli.train --size 125m --sft --sft-data alpaca
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("hbllm.train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HBLLM Training")

    # Model
    p.add_argument("--size", default="125m", choices=["125m", "500m", "1.5b"],
                   help="Model size preset")

    # Data
    p.add_argument("--data", default="fineweb",
                   choices=["fineweb", "wikipedia", "the_stack_v2"],
                   help="Dataset to use")
    p.add_argument("--max-samples", type=int, default=100_000,
                   help="Max training samples to download")
    p.add_argument("--data-dir", default="./data/training",
                   help="Working directory for data pipeline")

    # Training
    p.add_argument("--max-steps", type=int, default=10_000,
                   help="Max training steps")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Micro batch size")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Learning rate")
    p.add_argument("--grad-accum", type=int, default=4,
                   help="Gradient accumulation steps")
    p.add_argument("--checkpoint-dir", default="./checkpoints",
                   help="Checkpoint save directory")
    p.add_argument("--resume", default=None,
                   help="Resume from checkpoint directory")

    # SFT / LoRA
    p.add_argument("--sft", action="store_true",
                   help="Run supervised fine-tuning instead of pre-training")
    p.add_argument("--sft-data", default="alpaca",
                   help="SFT dataset name")
    p.add_argument("--lora", action="store_true",
                   help="Use LoRA for fine-tuning")
    p.add_argument("--lora-r", type=int, default=8,
                   help="LoRA rank")

    # Eval
    p.add_argument("--eval-interval", type=int, default=500,
                   help="Steps between evaluations")
    p.add_argument("--no-eval", action="store_true",
                   help="Skip evaluation")

    # Device
    p.add_argument("--device", default="auto",
                   help="Device: auto, cpu, cuda, mps")

    return p.parse_args()


def get_device(name: str) -> torch.device:
    """Resolve device string to torch.device."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def run_pretrain(args: argparse.Namespace) -> None:
    """Run pre-training pipeline."""
    from hbllm.model.config import get_config
    from hbllm.model.transformer import HBLLMForCausalLM
    from hbllm.model.tokenizer import HBLLMTokenizer
    from hbllm.training.trainer import Trainer, TrainingConfig
    from hbllm.data.dataloader import create_dataloader

    device = get_device(args.device)

    # 1. Prepare data
    logger.info("=== Step 1: Data Pipeline ===")
    data_dir = Path(args.data_dir)
    shard_dir = data_dir / "shards"

    if not list(shard_dir.glob("shard_*.bin")) if shard_dir.exists() else True:
        logger.info("No shards found. Running data pipeline...")
        try:
            from hbllm.data.pipeline import DataPipeline
            pipeline = DataPipeline(data_dir)
            pipeline.run_all(
                dataset_name=args.data,
                max_samples=args.max_samples,
            )
        except RuntimeError as e:
            logger.error("Data pipeline failed: %s", e)
            logger.info("Tip: Install Rust extensions or provide pre-tokenized shards in %s", shard_dir)
            sys.exit(1)
    else:
        logger.info("Using existing shards in %s", shard_dir)

    # 2. Create model
    logger.info("=== Step 2: Model (%s) ===", args.size)
    config = get_config(args.size)
    model = HBLLMForCausalLM(config)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Parameters: %s", f"{param_count:,}")

    # 3. Create dataloader
    train_loader = create_dataloader(
        shard_dir=shard_dir,
        sequence_length=config.max_position_embeddings,
        batch_size=args.batch_size,
    )

    # 4. Train
    logger.info("=== Step 3: Training ===")
    train_config = TrainingConfig(
        learning_rate=args.lr,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        checkpoint_dir=args.checkpoint_dir,
        eval_interval_steps=args.eval_interval,
    )
    trainer = Trainer(model, train_config, device=device)

    if args.resume:
        ckpt = trainer.ckpt_manager.load_latest()
        if ckpt:
            trainer.load_checkpoint(ckpt)

    # Training loop
    data_iter = iter(train_loader)
    for step in range(trainer.global_step, args.max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        metrics = trainer.train_step(batch)

        if (step + 1) % trainer.config.gradient_accumulation_steps == 0:
            step_metrics = trainer.step()

            if (step + 1) % train_config.log_interval_steps == 0:
                logger.info(
                    "Step %d | loss=%.4f lr=%.2e grad_norm=%.2f",
                    step + 1,
                    metrics["loss"],
                    step_metrics["lr"],
                    step_metrics["grad_norm"],
                )

            if (step + 1) % train_config.eval_interval_steps == 0 and not args.no_eval:
                logger.info("Running evaluation...")
                from hbllm.training.evaluator import ModelEvaluator
                tokenizer = HBLLMTokenizer.from_tiktoken()
                evaluator = ModelEvaluator(model, tokenizer, device)
                eval_results = evaluator.evaluate_all(
                    hellaswag=False,
                    generate=True,
                )
                if "samples" in eval_results:
                    for s in eval_results["samples"][:2]:
                        logger.info("  >> %s... -> %s...", s["prompt"][:30], s["generated"][:60])

            if (step + 1) % (train_config.eval_interval_steps * 2) == 0:
                trainer.save_checkpoint(loss=metrics["loss"])

    # Final save
    trainer.save_checkpoint(loss=metrics.get("loss", 0))
    logger.info("Training complete! Checkpoint saved to %s", args.checkpoint_dir)


def run_sft(args: argparse.Namespace) -> None:
    """Run supervised fine-tuning."""
    from hbllm.training.sft import run_sft_training
    device = get_device(args.device)
    run_sft_training(
        model_size=args.size,
        dataset_name=args.sft_data,
        use_lora=args.lora,
        lora_r=args.lora_r,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        device=device,
    )


def main() -> None:
    args = parse_args()

    logger.info("HBLLM Training")
    logger.info("  Model: %s", args.size)
    logger.info("  Device: %s", args.device)
    logger.info("  Mode: %s", "SFT" if args.sft else "Pre-training")

    if args.sft:
        run_sft(args)
    else:
        run_pretrain(args)


if __name__ == "__main__":
    main()
