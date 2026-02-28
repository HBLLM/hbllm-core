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


def run_serve(args):
    """Start the HBLLM API server."""
    logging.info("Starting HBLLM server on %s:%d...", args.host, args.port)
    try:
        import uvicorn
        uvicorn.run(
            "hbllm.serving.api:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level="info",
        )
    except ImportError:
        logging.error("uvicorn not installed. Run: pip install uvicorn")


def run_info(args):
    """Show system architecture info."""
    from hbllm import __version__
    print(f"""
🧠 HBLLM Core v{__version__}
{"=" * 50}

Architecture: Human-Brain Inspired Cognitive Architecture
Nodes:        23 specialized brain nodes
Memory:       5 systems (Episodic, Semantic, Procedural, Value, Working)
Model:        Zoning — shared base + LoRA domain adapters
Sizes:        125M / 500M / 1.5B parameters

Zones:
  ├── Perception:  Vision, Audio In, Audio Out
  ├── Brain:       Router, Planner, Decision, Critic, Learner,
  │                World Model, Curiosity, Identity, Meta, Workspace,
  │                Collective, Sleep, Spawner
  ├── Memory:      Episodic, Semantic, Procedural, Value
  └── Actions:     Execution, API, Browser, Logic, Fuzzy, MCP, IoT/MQTT

Features:
  ✅ Self-expanding zones (SpawnerNode)
  ✅ Async message bus with circuit breaker
  ✅ Policy engine (YAML governance)
  ✅ LoRA hot-swapping per domain
  ✅ Home automation via MQTT
  ✅ 100% local — runs on Raspberry Pi
""")


def run_nodes(args):
    """List all brain nodes."""
    nodes = [
        ("🔀 Router", "brain", "Thalamus — routes inputs to specialists"),
        ("📋 Planner", "brain", "Prefrontal Cortex — breaks tasks into steps"),
        ("⚖️ Decision", "brain", "Gatekeeper — safety + output routing"),
        ("🔍 Critic", "brain", "Quality Assurance — evaluates responses"),
        ("🎓 Learner", "brain", "Continuous Learning — DPO from feedback"),
        ("🌍 World Model", "brain", "Internal Simulation — predicts outcomes"),
        ("🔭 Curiosity", "brain", "Exploration Drive — seeks knowledge gaps"),
        ("🛡️ Identity", "brain", "Ethics Engine — value alignment"),
        ("🧠 Meta", "brain", "Self-Awareness — monitors performance"),
        ("📝 Workspace", "brain", "Global Blackboard — thought integration"),
        ("📊 Collective", "brain", "Swarm Intelligence — consensus"),
        ("💤 Sleep", "brain", "Memory Consolidation — optimization"),
        ("🧬 Spawner", "brain", "Neurogenesis — grows new specialists"),
        ("👁️ Vision", "perception", "Image Understanding — OCR + analysis"),
        ("🎤 Audio In", "perception", "Speech Recognition"),
        ("🔊 Audio Out", "perception", "Speech Synthesis"),
        ("⚡ Execution", "actions", "Code Sandbox — safe Python execution"),
        ("🌐 API", "actions", "Tool Synthesis — API schemas"),
        ("🖥️ Browser", "actions", "Web Agent — browses + extracts"),
        ("🔧 Logic", "actions", "Formal Reasoning — proofs"),
        ("🌀 Fuzzy", "actions", "Fuzzy Logic — uncertainty"),
        ("🔌 MCP", "actions", "Model Context Protocol — tools"),
        ("📡 IoT/MQTT", "actions", "Home Automation — MQTT devices"),
    ]

    print(f"\n🧠 HBLLM Brain Nodes ({len(nodes)} cognitive modules)\n")
    print(f"{'Node':<18} {'Zone':<12} {'Description'}")
    print("─" * 65)
    for name, zone, desc in nodes:
        print(f"{name:<18} {zone:<12} {desc}")
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="hbllm",
        description="🧠 HBLLM — Human-Brain Inspired Language Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Data pipeline
    data_parser = subparsers.add_parser("data", help="Run data preparation pipeline")
    data_parser.add_argument("--work-dir", type=str, default="./workspace")
    data_parser.add_argument("--dataset", type=str, default="fineweb")
    data_parser.add_argument("--samples", type=int, default=100_000)
    data_parser.add_argument("--vocab-size", type=int, default=32768)
    data_parser.add_argument("--seq-len", type=int, default=2048)

    # Training
    train_parser = subparsers.add_parser("train", help="Run pre-training loop")
    train_parser.add_argument("--work-dir", type=str, default="./workspace")
    train_parser.add_argument("--model-size", type=str, default="125m", choices=["125m", "500m", "1.5b"])
    train_parser.add_argument("--wandb-project", type=str, default=None)

    # Serve
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--workers", type=int, default=1)

    # Info / Nodes
    subparsers.add_parser("info", help="Show system architecture info")
    subparsers.add_parser("nodes", help="List all brain nodes")

    args = parser.parse_args()

    dispatch = {
        "data": run_pipeline,
        "train": run_train,
        "serve": run_serve,
        "info": run_info,
        "nodes": run_nodes,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()

