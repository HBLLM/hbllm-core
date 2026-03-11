"""
HBLLM Training CLI — one-command training entry point.

Usage:
    python -m hbllm.cli.train --size 125m --data fineweb --max-samples 100000
    python -m hbllm.cli.train --size 125m --sft --sft-data alpaca
    python -m hbllm.cli.train --dpo --reflection-dir workspace/reflection
    python -m hbllm.cli.train --eval --checkpoint checkpoints/step_1000.pt
    python -m hbllm.cli.train --export onnx --checkpoint checkpoints/step_1000.pt
    python -m hbllm.cli.train --serve-local
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
    p = argparse.ArgumentParser(description="HBLLM Training & Tools")

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
    p.add_argument("--checkpoint", default=None,
                   help="Path to a specific checkpoint .pt file")
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

    # DPO (self-improvement)
    p.add_argument("--dpo", action="store_true",
                   help="Run DPO self-improvement on reflection data")
    p.add_argument("--reflection-dir", default="workspace/reflection",
                   help="Directory with reflection JSONL datasets")

    # Evaluation
    p.add_argument("--eval", action="store_true", dest="run_eval",
                   help="Run model evaluation (perplexity, HellaSwag, samples)")
    p.add_argument("--eval-interval", type=int, default=500,
                   help="Steps between evaluations")
    p.add_argument("--no-eval", action="store_true",
                   help="Skip evaluation during training")

    # Export
    p.add_argument("--export", default=None,
                   choices=["onnx", "gguf", "fp16", "int8"],
                   help="Export trained model to specified format")
    p.add_argument("--export-output", default=None,
                   help="Output path for exported model")

    # Embedding training
    p.add_argument("--embed", action="store_true",
                   help="Train custom embedding model (InfoNCE contrastive)")
    p.add_argument("--embed-dim", type=int, default=256,
                   help="Embedding dimension")
    p.add_argument("--embed-data", default=None,
                   help="Path to embedding training data (JSONL with anchor/positive/negative)")

    # Local serving
    p.add_argument("--serve-local", action="store_true",
                   help="Start a local brain with the trained model")

    # Auto-eval gating
    p.add_argument("--auto-eval", action="store_true", default=True,
                   help="Auto-evaluate after training and gate adapter saving on improvement")

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
    tiktoken_mode = False

    if not list(shard_dir.glob("shard_*.bin")) if shard_dir.exists() else True:
        logger.info("No shards found. Running data pipeline...")
        try:
            from hbllm.data.pipeline import DataPipeline
            pipeline = DataPipeline(data_dir)
            pipeline.run_all(
                dataset_name=args.data,
                max_samples=args.max_samples,
            )
        except RuntimeError:
            logger.info("Rust extensions not available. Using Pure-Python pipeline (tiktoken)...")
            from hbllm.data.pipeline import PurePythonPipeline
            pipeline = PurePythonPipeline(data_dir)
            pipeline.run_all(
                dataset_name=args.data,
                max_samples=args.max_samples,
            )
            tiktoken_mode = True
    else:
        logger.info("Using existing shards in %s", shard_dir)

    # 2. Create model
    logger.info("=== Step 2: Model (%s) ===", args.size)
    config = get_config(args.size)
    if tiktoken_mode:
        # Override vocab size for tiktoken cl100k_base
        config.vocab_size = 100277
        logger.info("Using tiktoken vocab_size=%d", config.vocab_size)
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
        micro_batch_size=args.batch_size,
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


def run_dpo(args: argparse.Namespace) -> None:
    """Run DPO self-improvement on reflection data."""
    from hbllm.serving.self_improve import run_improvement_cycle

    logger.info("=== DPO Self-Improvement ===")
    results = run_improvement_cycle(
        reflection_dir=args.reflection_dir,
        model_size=args.size,
        checkpoint_dir=args.checkpoint_dir,
        use_lora=True,
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
        logger.info("No reflection data found. Nothing to improve.")

    # Auto-evaluate after DPO
    if args.auto_eval and results:
        logger.info("Running post-DPO evaluation...")
        _run_post_training_eval(args)


def run_eval(args: argparse.Namespace) -> None:
    """Run model evaluation."""
    from hbllm.model.config import get_config
    from hbllm.model.transformer import HBLLMForCausalLM
    from hbllm.model.tokenizer import Tokenizer
    from hbllm.training.evaluator import ModelEvaluator

    device = get_device(args.device)

    logger.info("=== Model Evaluation ===")
    config = get_config(args.size)
    model = HBLLMForCausalLM(config)

    # Load checkpoint
    if args.checkpoint:
        logger.info("Loading checkpoint: %s", args.checkpoint)
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    else:
        logger.warning("No checkpoint specified, evaluating random model.")

    model = model.to(device)
    tokenizer = Tokenizer()

    evaluator = ModelEvaluator(model, tokenizer, device)
    results = evaluator.evaluate_all(
        hellaswag=True,
        generate=True,
    )

    logger.info("Evaluation Results:")
    for key, value in results.items():
        if key != "samples":
            logger.info("  %s: %s", key, value)
    if "samples" in results:
        for s in results["samples"]:
            logger.info("  Prompt: %s...", s["prompt"][:50])
            logger.info("  Output: %s...", s["generated"][:100])


def run_export(args: argparse.Namespace) -> None:
    """Export trained model to various formats."""
    from hbllm.model.config import get_config
    from hbllm.model.transformer import HBLLMForCausalLM
    from hbllm.model.tokenizer import Tokenizer
    from hbllm.model.export import ModelExporter

    logger.info("=== Model Export (%s) ===", args.export)
    config = get_config(args.size)
    model = HBLLMForCausalLM(config)

    # Load checkpoint
    if args.checkpoint:
        logger.info("Loading checkpoint: %s", args.checkpoint)
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    else:
        logger.warning("No checkpoint specified, exporting random model.")

    tokenizer = Tokenizer()
    exporter = ModelExporter(model, tokenizer, config)

    # Print summary
    summary = exporter.summary()
    logger.info("Model: %s (%s params)", summary["model_name"], f"{summary['total_params']:,}")
    logger.info("Est. sizes: FP32=%.1f MB, FP16=%.1f MB, INT8=%.1f MB",
                summary["estimated_fp32_mb"], summary["estimated_fp16_mb"], summary["estimated_int8_mb"])

    # Default output paths
    export_dir = Path("./exports")
    output = args.export_output

    if args.export == "onnx":
        path = output or str(export_dir / f"{config.name}.onnx")
        exporter.export_onnx(path)
    elif args.export == "gguf":
        path = output or str(export_dir / f"{config.name}.gguf")
        exporter.export_gguf(path)
    elif args.export == "fp16":
        path = output or str(export_dir / f"{config.name}_fp16.pt")
        exporter.export_fp16(path)
    elif args.export == "int8":
        path = output or str(export_dir / f"{config.name}_int8.pt")
        exporter.quantize_dynamic(path)


def run_serve_local(args: argparse.Namespace) -> None:
    """Start a local brain for interactive use."""
    import asyncio

    async def _serve():
        from hbllm.brain.factory import BrainFactory
        brain = await BrainFactory.create_local(
            checkpoint_path=args.checkpoint,
            model_size=args.size,
            device=args.device,
        )
        logger.info("Local brain ready! Type queries (Ctrl+C to exit):")
        try:
            while True:
                query = input("\n> ")
                if not query.strip():
                    continue
                result = await brain.process(query)
                print(f"\n{result.text}")
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            await brain.shutdown()

    asyncio.run(_serve())


def run_embed(args: argparse.Namespace) -> None:
    """Train custom embedding model with InfoNCE contrastive loss."""
    from hbllm.training.embeddings import EmbeddingTrainer
    from hbllm.model.config import get_config
    from hbllm.model.transformer import HBLLMForCausalLM

    device = get_device(args.device)

    logger.info("=== Embedding Training ===")
    config = get_config(args.size)
    model = HBLLMForCausalLM(config)

    if args.checkpoint:
        logger.info("Loading base checkpoint: %s", args.checkpoint)
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)

    model = model.to(device)

    trainer = EmbeddingTrainer(
        model=model,
        embedding_dim=args.embed_dim,
        device=device,
    )

    if args.embed_data:
        logger.info("Loading embedding data from: %s", args.embed_data)
        import json
        triplets = []
        with open(args.embed_data) as f:
            for line in f:
                triplets.append(json.loads(line))
        logger.info("Loaded %d triplets", len(triplets))
    else:
        logger.warning("No --embed-data provided. Using synthetic examples.")
        triplets = []

    if triplets:
        trainer.train(triplets, max_steps=args.max_steps, lr=args.lr)

    # Save embedding head
    save_path = Path(args.checkpoint_dir) / "embedding_head.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.projection.state_dict(), str(save_path))
    logger.info("Embedding head saved to %s", save_path)


def _run_post_training_eval(args: argparse.Namespace) -> None:
    """Auto-evaluate after training completion."""
    try:
        from hbllm.model.config import get_config
        from hbllm.model.transformer import HBLLMForCausalLM
        from hbllm.model.tokenizer import HBLLMTokenizer
        from hbllm.training.evaluator import ModelEvaluator

        device = get_device(args.device)
        config = get_config(args.size)
        model = HBLLMForCausalLM(config)

        # Load the just-trained checkpoint
        ckpt_dir = Path(args.checkpoint_dir)
        latest = sorted(ckpt_dir.glob("*.pt"))[-1] if ckpt_dir.exists() else None
        if latest:
            ckpt = torch.load(str(latest), map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)

        model = model.to(device)
        tokenizer = HBLLMTokenizer.from_tiktoken()
        evaluator = ModelEvaluator(model, tokenizer, device)

        results = evaluator.evaluate_all(hellaswag=False, generate=True)
        logger.info("Post-training eval: %s", {
            k: v for k, v in results.items() if k != "samples"
        })
        if "samples" in results:
            for s in results["samples"][:2]:
                logger.info("  >> %s... -> %s...", s["prompt"][:30], s["generated"][:60])
    except Exception as e:
        logger.warning("Post-training evaluation failed (non-critical): %s", e)


def main() -> None:
    args = parse_args()

    logger.info("HBLLM Training")
    logger.info("  Model: %s", args.size)
    logger.info("  Device: %s", args.device)

    if args.export:
        run_export(args)
    elif args.run_eval:
        run_eval(args)
    elif args.dpo:
        logger.info("  Mode: DPO Self-Improvement")
        run_dpo(args)
    elif args.sft:
        logger.info("  Mode: SFT")
        run_sft(args)
    elif args.embed:
        logger.info("  Mode: Embedding Training")
        run_embed(args)
    elif args.serve_local:
        logger.info("  Mode: Local Serve")
        run_serve_local(args)
    else:
        logger.info("  Mode: Pre-training")
        run_pretrain(args)


if __name__ == "__main__":
    main()
