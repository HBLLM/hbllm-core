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
from pathlib import Path
from typing import Any

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("hbllm.train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HBLLM Training & Tools")

    # Model
    p.add_argument(
        "--size", default="125m", choices=["125m", "500m", "1.5b"], help="Model size preset"
    )

    # Data
    p.add_argument(
        "--data",
        default="fineweb",
        help="Dataset(s) to use. Available: fineweb, wikipedia, "
        "the_stack_v2, starcoderdata, codeparrot, openwebmath, "
        "metamath, pes2o, openhermes, slimorca. "
        "Mix with +: --data fineweb+starcoderdata+openwebmath",
    )
    p.add_argument(
        "--max-samples", type=int, default=100_000, help="Max training samples to download"
    )
    p.add_argument(
        "--data-weights",
        default=None,
        help="Proportional weights for each dataset when mixing. "
        "Comma-separated floats matching --data order. "
        "E.g.: --data fineweb+starcoderdata --data-weights 0.7,0.3. "
        "Defaults to equal weights.",
    )
    p.add_argument(
        "--data-dir", default="./data/training", help="Working directory for data pipeline"
    )

    # Training
    p.add_argument("--max-steps", type=int, default=10_000, help="Max training steps")
    p.add_argument("--batch-size", type=int, default=8, help="Micro batch size")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--checkpoint-dir", default="./checkpoints", help="Checkpoint save directory")
    p.add_argument("--checkpoint", default=None, help="Path to a specific checkpoint .pt file")
    p.add_argument("--resume", default=None, help="Resume from checkpoint directory")

    # SFT / LoRA
    p.add_argument(
        "--sft", action="store_true", help="Run supervised fine-tuning instead of pre-training"
    )
    p.add_argument("--sft-data", default="alpaca", help="SFT dataset name")
    p.add_argument("--lora", action="store_true", help="Use LoRA for fine-tuning")
    p.add_argument("--lora-r", type=int, default=8, help="LoRA rank")

    # DPO (self-improvement)
    p.add_argument("--dpo", action="store_true", help="Run DPO self-improvement on reflection data")
    p.add_argument(
        "--reflection-dir",
        default="workspace/reflection",
        help="Directory with reflection JSONL datasets",
    )

    # Evaluation
    p.add_argument(
        "--eval",
        action="store_true",
        dest="run_eval",
        help="Run model evaluation (perplexity, HellaSwag, samples)",
    )
    p.add_argument("--eval-interval", type=int, default=500, help="Steps between evaluations")
    p.add_argument("--no-eval", action="store_true", help="Skip evaluation during training")

    # Export
    p.add_argument(
        "--export",
        default=None,
        choices=["onnx", "gguf", "fp16", "int8"],
        help="Export trained model to specified format",
    )
    p.add_argument("--export-output", default=None, help="Output path for exported model")

    # Embedding training
    p.add_argument(
        "--embed", action="store_true", help="Train custom embedding model (InfoNCE contrastive)"
    )
    p.add_argument("--embed-dim", type=int, default=256, help="Embedding dimension")
    p.add_argument(
        "--embed-data",
        default=None,
        help="Path to embedding training data (JSONL with anchor/positive/negative)",
    )

    # Cognitive training
    p.add_argument(
        "--cognitive",
        action="store_true",
        help="Enable cognitive training (build knowledge graph, skills, memory during training)",
    )
    p.add_argument(
        "--cognitive-interval", type=int, default=10, help="Run cognitive processing every N steps"
    )
    p.add_argument(
        "--cognitive-dir",
        default="./cognitive_checkpoints",
        help="Output directory for cognitive artifacts",
    )

    # Local serving
    p.add_argument(
        "--serve-local", action="store_true", help="Start a local brain with the trained model"
    )

    # Auto-eval gating
    p.add_argument(
        "--auto-eval",
        action="store_true",
        default=True,
        help="Auto-evaluate after training and gate adapter saving on improvement",
    )

    # Device
    p.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps")

    args = p.parse_args()

    # Validate --data dataset names against registry
    from hbllm.data.downloader import PREDEFINED_SOURCES

    dataset_names = [n.strip() for n in args.data.split("+") if n.strip()]
    unknown = [n for n in dataset_names if n not in PREDEFINED_SOURCES]
    if unknown:
        available = ", ".join(sorted(PREDEFINED_SOURCES.keys()))
        p.error(
            f"Unknown dataset(s): {', '.join(unknown)}\n"
            f"Available: {available}\n"
            f"Mix with +: --data fineweb+starcoderdata+openwebmath"
        )

    return args


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
    from hbllm.data.dataloader import create_dataloader
    from hbllm.model.config import get_config
    from hbllm.model.tokenizer import HBLLMTokenizer
    from hbllm.model.transformer import HBLLMForCausalLM
    from hbllm.training.trainer import Trainer, TrainingConfig

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

            pipeline: Any = DataPipeline(data_dir)
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

    # 3. Create dataloader (num_workers=0 to avoid multiprocessing spawn issues)
    train_loader = create_dataloader(
        shard_dir=shard_dir,
        sequence_length=config.max_position_embeddings,
        batch_size=args.batch_size,
        num_workers=0,
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
        ckpt = trainer.ckpt_manager.load_latest()  # type: ignore[attr-defined]
        if ckpt:
            trainer.load_checkpoint(ckpt)

    # Training loop with verbose logging
    import time as _time

    logger.info("=" * 60)
    logger.info("Starting training loop")
    logger.info("  Steps       : %d", args.max_steps)
    logger.info(
        "  Batch size  : %d micro x %d accum = %d effective",
        args.batch_size,
        args.grad_accum,
        args.batch_size * args.grad_accum,
    )
    logger.info("  Seq length  : %d", config.max_position_embeddings)
    logger.info("  Device      : %s", device)
    logger.info("  Precision   : %s", train_config.precision)
    logger.info("=" * 60)

    data_iter = iter(train_loader)
    train_start = _time.time()
    step_start = _time.time()
    total_tokens_processed = 0
    running_loss = 0.0
    loss_count = 0

    for step in range(trainer.global_step, args.max_steps):
        # Load batch
        try:
            batch = next(data_iter)
        except StopIteration:
            logger.info("  [Epoch boundary] Restarting data iterator...")
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # Track tokens
        batch_tokens = batch["input_ids"].numel()
        total_tokens_processed += batch_tokens

        # Forward + backward
        logger.info(
            "  Step %d/%d: forward+backward pass (%d tokens)...",
            step + 1,
            args.max_steps,
            batch_tokens,
        )
        metrics = trainer.train_step(batch)
        running_loss += metrics["loss"]
        loss_count += 1

        if (step + 1) % trainer.config.gradient_accumulation_steps == 0:
            step_metrics = trainer.step()

            # Timing
            step_elapsed = _time.time() - step_start
            total_elapsed = _time.time() - train_start
            steps_done = step + 1
            steps_remaining = args.max_steps - steps_done
            avg_step_time = total_elapsed / max(1, steps_done)
            eta_seconds = steps_remaining * avg_step_time
            tok_per_sec = total_tokens_processed / max(1, total_elapsed)

            # ETA formatting
            if eta_seconds < 60:
                eta_str = f"{eta_seconds:.0f}s"
            elif eta_seconds < 3600:
                eta_str = f"{eta_seconds / 60:.1f}min"
            else:
                eta_str = f"{eta_seconds / 3600:.1f}hr"

            avg_loss = running_loss / max(1, loss_count)

            logger.info(
                "  ✓ Step %d/%d | loss=%.4f (avg=%.4f) | lr=%.2e | "
                "grad_norm=%.2f | %.1f tok/s | %.1fs/step | ETA: %s",
                steps_done,
                args.max_steps,
                metrics["loss"],
                avg_loss,
                step_metrics["lr"],
                step_metrics["grad_norm"],
                tok_per_sec,
                step_elapsed,
                eta_str,
            )

            step_start = _time.time()

            # Eval
            if (step + 1) % train_config.eval_interval_steps == 0 and not args.no_eval:
                logger.info("  [Eval] Running evaluation at step %d...", steps_done)
                from hbllm.training.evaluator import ModelEvaluator

                tokenizer = HBLLMTokenizer.from_tiktoken()
                evaluator = ModelEvaluator(model, tokenizer, device)
                eval_results = evaluator.evaluate_all(
                    hellaswag=False,
                    generate=True,
                )
                if "samples" in eval_results:
                    for s in eval_results["samples"][:2]:
                        logger.info("    >> %s... -> %s...", s["prompt"][:30], s["generated"][:60])

            # Checkpoint
            if (step + 1) % (train_config.eval_interval_steps * 2) == 0:
                logger.info("  [Checkpoint] Saving at step %d...", steps_done)
                trainer.save_checkpoint(loss=metrics["loss"])
                logger.info("  [Checkpoint] Saved!")

    # Final save
    total_elapsed = _time.time() - train_start
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("  Total steps  : %d", args.max_steps)
    logger.info("  Total tokens : %d", total_tokens_processed)
    logger.info("  Final loss   : %.4f", metrics.get("loss", 0))
    logger.info("  Avg loss     : %.4f", running_loss / max(1, loss_count))
    logger.info("  Total time   : %.1fs", total_elapsed)
    logger.info("  Avg tok/s    : %.0f", total_tokens_processed / max(1, total_elapsed))
    logger.info("=" * 60)
    trainer.save_checkpoint(loss=metrics.get("loss", 0))
    logger.info("Final checkpoint saved to %s", args.checkpoint_dir)


def run_cognitive_pretrain(args: argparse.Namespace) -> None:
    """Run cognitive pre-training: base model + knowledge graph + skills + memory + LoRA."""
    import os
    import time as _time

    from hbllm.data.dataloader import create_dataloader
    from hbllm.model.config import get_config
    from hbllm.model.transformer import HBLLMForCausalLM
    from hbllm.training.cognitive_trainer import CognitiveConfig, CognitiveTrainer
    from hbllm.training.trainer import TrainingConfig

    device = get_device(args.device)

    # 1. Data pipeline (reuse standard pipeline)
    logger.info("=== Step 1: Data Pipeline ===")
    shard_dir = os.path.join(args.data_dir, "shards")
    raw_dir = os.path.join(args.data_dir, "raw")

    tiktoken_mode = False
    if not os.path.exists(shard_dir) or not os.listdir(shard_dir):
        logger.info("No shards found. Running data pipeline...")
        try:
            from hbllm.data.pipeline import DataPipeline

            pipeline: Any = DataPipeline(args.data_dir)
            pipeline.run(args.data, max_samples=args.max_samples)
        except RuntimeError:
            logger.info("Rust pipeline unavailable, using PurePythonPipeline...")
            from hbllm.data.pipeline import PurePythonPipeline

            pipeline = PurePythonPipeline(args.data_dir)
            pipeline.run(args.data, max_samples=args.max_samples)
            tiktoken_mode = True
    else:
        logger.info("Using existing shards in %s", shard_dir)

    # 2. Create model
    logger.info("=== Step 2: Model (%s) ===", args.size)
    config = get_config(args.size)
    if tiktoken_mode:
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
        num_workers=0,
    )

    # 4. Load raw texts for cognitive processing
    raw_texts_cache: list[str] = []
    try:
        import glob

        raw_files = sorted(glob.glob(os.path.join(raw_dir, "**", "*.jsonl"), recursive=True))
        if raw_files:
            import json as _json

            for rf in raw_files:
                with open(rf) as f:
                    for line in f:
                        try:
                            doc = _json.loads(line)
                            text = doc.get("text", doc.get("content", ""))
                            if text:
                                raw_texts_cache.append(text)
                        except _json.JSONDecodeError:
                            continue
                        if len(raw_texts_cache) >= args.max_samples:
                            break
            logger.info("Loaded %d raw texts for cognitive processing", len(raw_texts_cache))
    except Exception as e:
        logger.warning("Could not load raw texts: %s (cognitive processing will be limited)", e)

    # 5. Configure and create CognitiveTrainer
    logger.info("=== Step 3: Cognitive Training ===")
    train_config = TrainingConfig(
        learning_rate=args.lr,
        max_steps=args.max_steps,
        micro_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        checkpoint_dir=args.checkpoint_dir,
        eval_interval_steps=getattr(args, "eval_interval", 500),
    )
    cognitive_config = CognitiveConfig(
        output_dir=getattr(args, "cognitive_dir", "./cognitive_checkpoints"),
        cognitive_interval=getattr(args, "cognitive_interval", 10),
        use_lora=args.lora,
        lora_r=args.lora_r,
        build_knowledge_graph=True,
        track_training_memory=True,
        detect_skills=True,
        extract_concepts=True,
    )
    cog_trainer = CognitiveTrainer(model, train_config, cognitive_config, device)

    logger.info("=" * 60)
    logger.info("Starting COGNITIVE training loop")
    logger.info("  Steps        : %d", args.max_steps)
    logger.info(
        "  Batch size   : %d micro x %d accum = %d effective",
        args.batch_size,
        args.grad_accum,
        args.batch_size * args.grad_accum,
    )
    logger.info("  Seq length   : %d", config.max_position_embeddings)
    logger.info("  Device       : %s", device)
    logger.info("  LoRA         : %s (r=%d)", "ON" if args.lora else "OFF", args.lora_r)
    logger.info("  Knowledge Graph : ON")
    logger.info("  Training Memory : ON")
    logger.info("  Skill Detection : ON")
    logger.info("  Cognitive interval: every %d steps", cognitive_config.cognitive_interval)
    logger.info("  Raw texts    : %d available", len(raw_texts_cache))
    logger.info("=" * 60)

    # 6. Training loop
    data_iter = iter(train_loader)
    train_start = _time.time()
    step_start = _time.time()
    total_tokens_processed = 0
    running_loss = 0.0
    loss_count = 0
    raw_text_idx = 0

    for step in range(cog_trainer.global_step, args.max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            logger.info("  [Epoch boundary] Restarting data iterator...")
            data_iter = iter(train_loader)
            batch = next(data_iter)

        batch_tokens = batch["input_ids"].numel()
        total_tokens_processed += batch_tokens

        # Get raw texts for this batch (if available)
        batch_raw_texts = None
        if raw_texts_cache:
            batch_raw_texts = []
            for _ in range(args.batch_size):
                if raw_text_idx < len(raw_texts_cache):
                    batch_raw_texts.append(raw_texts_cache[raw_text_idx])
                    raw_text_idx += 1
                else:
                    raw_text_idx = 0
                    batch_raw_texts.append(raw_texts_cache[raw_text_idx])
                    raw_text_idx += 1

        logger.info(
            "  Step %d/%d: forward+backward (%d tokens)...", step + 1, args.max_steps, batch_tokens
        )

        metrics = cog_trainer.cognitive_train_step(batch, batch_raw_texts)
        running_loss += metrics["loss"]
        loss_count += 1

        if (step + 1) % cog_trainer.config.gradient_accumulation_steps == 0:
            step_metrics = cog_trainer.step()

            step_elapsed = _time.time() - step_start
            total_elapsed = _time.time() - train_start
            steps_done = step + 1
            steps_remaining = args.max_steps - steps_done
            avg_step_time = total_elapsed / max(1, steps_done)
            eta_seconds = steps_remaining * avg_step_time
            tok_per_sec = total_tokens_processed / max(1, total_elapsed)

            if eta_seconds < 60:
                eta_str = f"{eta_seconds:.0f}s"
            elif eta_seconds < 3600:
                eta_str = f"{eta_seconds / 60:.1f}min"
            else:
                eta_str = f"{eta_seconds / 3600:.1f}hr"

            avg_loss = running_loss / max(1, loss_count)

            # Core training metrics
            log_parts = [
                f"  ✓ Step {steps_done}/{args.max_steps}",
                f"loss={metrics['loss']:.4f} (avg={avg_loss:.4f})",
                f"lr={step_metrics['lr']:.2e}",
                f"grad_norm={step_metrics['grad_norm']:.2f}",
                f"{tok_per_sec:.0f} tok/s",
                f"{step_elapsed:.1f}s/step",
                f"ETA: {eta_str}",
            ]
            logger.info(" | ".join(log_parts))

            # Cognitive metrics (if processed this step)
            if "kg_total_entities" in metrics:
                logger.info(
                    "    🧠 KG: %d entities | Memory: %d records | Domains: %d",
                    metrics.get("kg_total_entities", 0),
                    metrics.get("memory_records", 0),
                    metrics.get("domains_seen", 0),
                )

            step_start = _time.time()

            # Periodic cognitive status
            if steps_done % 50 == 0:
                cog_trainer.log_cognitive_status()

            # Eval
            if (step + 1) % train_config.eval_interval_steps == 0 and not args.no_eval:
                logger.info("  [Eval] Running evaluation at step %d...", steps_done)
                from hbllm.model.tokenizer import HBLLMTokenizer
                from hbllm.training.evaluator import ModelEvaluator

                tokenizer = HBLLMTokenizer.from_tiktoken()
                evaluator = ModelEvaluator(model, tokenizer, device)
                eval_results = evaluator.evaluate_all(hellaswag=False, generate=True)
                if "samples" in eval_results:
                    for s in eval_results["samples"][:2]:
                        logger.info("    >> %s... -> %s...", s["prompt"][:30], s["generated"][:60])

            # Checkpoint
            if (step + 1) % (train_config.eval_interval_steps * 2) == 0:
                logger.info("  [Cognitive Checkpoint] Saving at step %d...", steps_done)
                cog_trainer.save_cognitive_checkpoint(loss=metrics["loss"])

    # Final save
    total_elapsed = _time.time() - train_start
    logger.info("=" * 60)
    logger.info("Cognitive training complete!")
    logger.info("  Total steps  : %d", args.max_steps)
    logger.info("  Total tokens : %d", total_tokens_processed)
    logger.info("  Final loss   : %.4f", metrics.get("loss", 0))
    logger.info("  Avg loss     : %.4f", running_loss / max(1, loss_count))
    logger.info("  Total time   : %.1fs", total_elapsed)
    logger.info("  Avg tok/s    : %.0f", total_tokens_processed / max(1, total_elapsed))
    logger.info("=" * 60)

    ckpt_dir = cog_trainer.save_cognitive_checkpoint(loss=metrics.get("loss", 0))
    logger.info("Cognitive checkpoint saved to %s", ckpt_dir)


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
    from hbllm.model.tokenizer import HBLLMTokenizer as Tokenizer
    from hbllm.model.transformer import HBLLMForCausalLM
    from hbllm.training.evaluator import ModelEvaluator

    device = get_device(args.device)

    logger.info("=== Model Evaluation ===")
    config = get_config(args.size)
    model = HBLLMForCausalLM(config)

    # Load checkpoint
    if args.checkpoint:
        logger.info("Loading checkpoint: %s", args.checkpoint)
        from hbllm.utils.checkpoint import extract_model_state, load_checkpoint

        ckpt = load_checkpoint(args.checkpoint)
        model.load_state_dict(extract_model_state(ckpt), strict=False)
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
    from hbllm.model.export import ModelExporter
    from hbllm.model.tokenizer import HBLLMTokenizer as Tokenizer
    from hbllm.model.transformer import HBLLMForCausalLM

    logger.info("=== Model Export (%s) ===", args.export)
    config = get_config(args.size)
    model = HBLLMForCausalLM(config)

    # Load checkpoint
    if args.checkpoint:
        logger.info("Loading checkpoint: %s", args.checkpoint)
        from hbllm.utils.checkpoint import extract_model_state, load_checkpoint

        ckpt = load_checkpoint(args.checkpoint)
        model.load_state_dict(extract_model_state(ckpt), strict=False)
    else:
        logger.warning("No checkpoint specified, exporting random model.")

    tokenizer = Tokenizer()
    exporter = ModelExporter(model, tokenizer, config)

    # Print summary
    summary = exporter.summary()
    logger.info("Model: %s (%s params)", summary["model_name"], f"{summary['total_params']:,}")
    logger.info(
        "Est. sizes: FP32=%.1f MB, FP16=%.1f MB, INT8=%.1f MB",
        summary["estimated_fp32_mb"],
        summary["estimated_fp16_mb"],
        summary["estimated_int8_mb"],
    )

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

    async def _serve() -> None:
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
    from hbllm.model.config import get_config
    from hbllm.model.transformer import HBLLMForCausalLM
    from hbllm.training.embeddings import EmbeddingTrainer

    device = get_device(args.device)

    logger.info("=== Embedding Training ===")
    config = get_config(args.size)
    model = HBLLMForCausalLM(config)

    if args.checkpoint:
        logger.info("Loading base checkpoint: %s", args.checkpoint)
        from hbllm.utils.checkpoint import extract_model_state, load_checkpoint

        ckpt = load_checkpoint(args.checkpoint)
        model.load_state_dict(extract_model_state(ckpt), strict=False)

    model = model.to(device)

    trainer = EmbeddingTrainer(
        embedding_dim=args.embed_dim,
        device=str(device),
        vocab_size=config.vocab_size,
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
        trainer.train(triplets, epochs=args.max_steps, lr=args.lr)

    # Save embedding head
    save_path = Path(args.checkpoint_dir) / "embedding_head.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.model.proj.state_dict(), str(save_path))
    logger.info("Embedding head saved to %s", save_path)


def _run_post_training_eval(args: argparse.Namespace) -> None:
    """Auto-evaluate after training completion."""
    try:
        from hbllm.model.config import get_config
        from hbllm.model.tokenizer import HBLLMTokenizer
        from hbllm.model.transformer import HBLLMForCausalLM
        from hbllm.training.evaluator import ModelEvaluator

        device = get_device(args.device)
        config = get_config(args.size)
        model = HBLLMForCausalLM(config)

        # Load the just-trained checkpoint
        ckpt_dir = Path(args.checkpoint_dir)
        latest = sorted(ckpt_dir.glob("*.pt"))[-1] if ckpt_dir.exists() else None
        if latest:
            from hbllm.utils.checkpoint import extract_model_state, load_checkpoint

            ckpt = load_checkpoint(str(latest))
            model.load_state_dict(extract_model_state(ckpt), strict=False)

        model = model.to(device)
        tokenizer = HBLLMTokenizer.from_tiktoken()
        evaluator = ModelEvaluator(model, tokenizer, device)

        results = evaluator.evaluate_all(hellaswag=False, generate=True)
        logger.info("Post-training eval: %s", {k: v for k, v in results.items() if k != "samples"})
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
    elif getattr(args, "cognitive", False):
        logger.info("  Mode: Cognitive Pre-training")
        run_cognitive_pretrain(args)
    else:
        logger.info("  Mode: Pre-training")
        run_pretrain(args)


if __name__ == "__main__":
    main()
