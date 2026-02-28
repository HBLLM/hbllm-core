"""
CLI for running data pipeline and training tasks.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

# Setup simple logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def run_pipeline(args):
    """Run the data pipeline."""
    from hbllm.data.pipeline import DataPipeline

    pipeline = DataPipeline(args.work_dir)
    pipeline.run_all(
        dataset_name=args.dataset,
        max_samples=args.samples,
        vocab_size=args.vocab_size,
        sequence_length=args.seq_len,
    )


def run_train(args):
    """Run the pre-training loop."""
    import torch
    from hbllm.model.config import get_config
    from hbllm.model.transformer import HBLLMForCausalLM
    from hbllm.training.trainer import Trainer, TrainingConfig
    from hbllm.data.dataloader import create_dataloader
    
    # 1. Configuration
    model_config = get_config(args.model_size)
    train_config = TrainingConfig(wandb_project=args.wandb_project)
    
    logging.info("Initializing %s model with %d parameters...", args.model_size, model_config.num_params_estimate)
    model = HBLLMForCausalLM(model_config)
    
    trainer = Trainer(model, train_config)
    
    shard_dir = Path(args.work_dir) / "shards"
    dataloader = create_dataloader(
        shard_dir, 
        sequence_length=train_config.sequence_length if hasattr(train_config, 'sequence_length') else 2048,
        batch_size=train_config.micro_batch_size,
        num_workers=0
    )
    
    logging.info("Starting training loop...")
    for step, batch in enumerate(dataloader):
        if step >= train_config.max_steps:
            break
            
        metrics = trainer.train_step(batch)
        
        if (step + 1) % train_config.gradient_accumulation_steps == 0:
            step_metrics = trainer.step()
            metrics.update(step_metrics)
            
            if trainer.global_step % train_config.log_interval_steps == 0:
                logging.info(
                    "Step %d | Loss: %.4f | LR: %.2e | GradNorm: %.2f",
                    trainer.global_step,
                    metrics["loss"],
                    metrics["lr"],
                    metrics["grad_norm"],
                )
                if trainer._wandb_run is not None:
                    trainer._wandb_run.log({"loss": metrics["loss"]}, step=trainer.global_step)
                
            if trainer.global_step % train_config.save_interval_steps == 0:
                trainer.save_checkpoint(metrics["loss"])


def main():
    parser = argparse.ArgumentParser(description="HBLLM CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Data Pipeline Command
    data_parser = subparsers.add_parser("data", help="Run the data preparation pipeline")
    data_parser.add_argument("--work-dir", type=str, default="./workspace", help="Working directory for data")
    data_parser.add_argument("--dataset", type=str, default="fineweb", help="Dataset preset to download")
    data_parser.add_argument("--samples", type=int, default=100_000, help="Max samples to download")
    data_parser.add_argument("--vocab-size", type=int, default=32768, help="Target vocabulary size")
    data_parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length for sharding")

    # Train Command
    train_parser = subparsers.add_parser("train", help="Run the pre-training loop")
    train_parser.add_argument("--work-dir", type=str, default="./workspace", help="Working directory with shards")
    train_parser.add_argument("--model-size", type=str, default="125m", choices=["125m", "500m", "1.5b"], help="Model size preset")
    train_parser.add_argument("--wandb-project", type=str, default=None, help="Weights & Biases project name")

    args = parser.parse_args()

    if args.command == "data":
        run_pipeline(args)
    elif args.command == "train":
        run_train(args)


if __name__ == "__main__":
    main()
