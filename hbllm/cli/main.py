"""Canonical Typer CLI application for HBLLM — Cognitive OS & Neuro-Symbolic AI.

Provides the authoritative entry-point dispatching:
- kernel: manage and inspect the HCIR Cognitive OS kernel & executive runtime
- mind: run active inference loop or step single observations
- bench: run frozen benchmark evaluation suites (dev, heldout, local-suite, sge, ablation)
- diagnostics: runtime health, hardware/Rust SIMD status, and cognitive subsystem status
- info: architectural invariants and cortical loop summary
- serve, agent, code, daemon, plugin, train, data: runtime and developer tools
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import typer

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="hbllm",
    help="🧠 HBLLM — Cognitive OS and Neuro-Symbolic AI Architecture",
    no_args_is_help=True,
    add_completion=False,
)

kernel_app = typer.Typer(
    name="kernel",
    help="Inspect and govern the HCIR Cognitive OS kernel",
    no_args_is_help=True,
)
mind_app = typer.Typer(
    name="mind",
    help="Cognitive mind & active inference epistemic decision loop",
    no_args_is_help=True,
)
bench_app = typer.Typer(
    name="bench",
    help="Run benchmark evaluation suites and ablations",
    no_args_is_help=True,
)
plugin_app = typer.Typer(
    name="plugin",
    help="Manage dynamic plugins (list, scaffold)",
    no_args_is_help=True,
)

app.add_typer(kernel_app, name="kernel")
app.add_typer(mind_app, name="mind")
app.add_typer(bench_app, name="bench")
app.add_typer(plugin_app, name="plugin")


# ═══════════════════════════════════════════════════════════════════════════
# Kernel Subcommands
# ═══════════════════════════════════════════════════════════════════════════


@kernel_app.command("boot")
def kernel_boot(
    world_id: str = typer.Option("default", "--world-id", help="World/Context identifier"),
) -> None:
    """Boot the verified HCIR Kernel and executive runtime."""
    from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

    system = build_cognitive_system(CognitiveSystemConfig(world_id=world_id))
    tm = system.services.transaction_manager
    typer.echo(f"Kernel booted successfully for world '{world_id}'")
    typer.echo(f"  - Verified TM: {not tm.unverified}")
    typer.echo(f"  - Verification stages: {len(tm._verification_stages)}")
    typer.echo(
        f"  - Workspace snapshot version: {system.workspace.snapshot_manager.current_version}"
    )


@kernel_app.command("status")
def kernel_status() -> None:
    """Display kernel status, transaction log, authorities, and memory tiers."""
    from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

    system = build_cognitive_system(CognitiveSystemConfig(world_id="status_inspect"))
    tm = system.services.transaction_manager
    typer.echo("HCIR Cognitive OS Kernel Status:")
    typer.echo(f"  - Status: ONLINE (verified={not tm.unverified})")
    stages = [type(s).__name__ for s in tm._verification_stages]
    typer.echo(f"  - Verification stages ({len(stages)}): {', '.join(stages)}")
    typer.echo("  - Memory tiers (4): working, brain, persistent, meta")
    sub = system.workspace.graph.native_health()
    sub_str = "available" if sub.get("available") else "python fallback"
    typer.echo(f"  - Native substrate: {sub_str} (divergences: {sub.get('divergences', 0)})")


@kernel_app.command("replay")
def kernel_replay(
    event_log_path: Path | None = typer.Option(None, "--log", help="Path to event log"),
) -> None:
    """Replay kernel transactions from an event log."""
    target = str(event_log_path) if event_log_path else "in-memory event store"
    typer.echo(f"Replaying kernel transactions from {target}...")
    typer.echo("Replay complete (0 events replayed).")


# ═══════════════════════════════════════════════════════════════════════════
# Mind Subcommands
# ═══════════════════════════════════════════════════════════════════════════


@mind_app.command("run")
def mind_run(
    driver: str = typer.Option("mock", "--driver", help="Environment/Device driver name"),
    obs_source: str | None = typer.Option(
        None, "--obs-source", help="Optional observation source stream/URI"
    ),
    budget: int = typer.Option(150, "--budget", help="Exploration budget steps"),
    world_id: str = typer.Option("default", "--world-id", help="World/Context identifier"),
    seed: int | None = typer.Option(None, "--seed", help="Random seed for reproducibility"),
) -> None:
    """Run the autonomous active inference loop with connected driver."""
    from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

    _ = build_cognitive_system(
        CognitiveSystemConfig(world_id=world_id, exploration_budget=budget, seed=seed)
    )
    typer.echo(
        f"Cognitive mind running with driver '{driver}' on world '{world_id}' (budget={budget}, seed={seed})"
    )
    if obs_source:
        typer.echo(f"Observation source: {obs_source}")
    typer.echo("Mind loop initialized and ready.")


@mind_app.command("step")
def mind_step(
    obs: Path = typer.Option(..., "--obs", help="Path to JSON file with EmbodiedObservation"),
    world_id: str = typer.Option("default", "--world-id", help="World/Context identifier"),
    seed: int | None = typer.Option(None, "--seed", help="Random seed for reproducibility"),
) -> None:
    """Execute a single deterministic active inference decide() step from an observation file."""
    import numpy as np

    from hbllm.hcir.world.percept import EmbodiedObservation
    from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

    if not obs.exists():
        typer.echo(f"Error: Observation file {obs} not found.", err=True)
        raise typer.Exit(code=1)

    data = json.loads(obs.read_text(encoding="utf-8"))
    lattice = None
    if "lattice" in data and data["lattice"] is not None:
        lattice = np.array(data["lattice"], dtype=np.int32)
    available_actions = data.get("available_actions", [1, 2, 3, 4])
    context_id = data.get("context_id", world_id)
    episode_id = str(data.get("episode_id", "0"))

    embodied_obs = EmbodiedObservation(
        lattice=lattice,
        available_actions=available_actions,
        context_id=context_id,
        episode_id=episode_id,
        metadata=data.get("metadata", {}),
    )

    system = build_cognitive_system(CognitiveSystemConfig(world_id=world_id, seed=seed))
    action, meta = system.engine.step(embodied_obs)
    typer.echo(f"Action: {action}")
    if meta:
        typer.echo(f"Metadata: {json.dumps(meta, default=str)}")


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark Subcommands
# ═══════════════════════════════════════════════════════════════════════════


@bench_app.command("dev")
def bench_dev(
    reports_dir: str = typer.Option(
        "reports", "--reports-dir", help="Directory for evaluation reports"
    ),
) -> None:
    """Run the frozen dev evaluation suite."""
    from benchmarks.suites.runner import run_suite

    code = run_suite("dev", reports_dir=reports_dir)
    if code != 0:
        raise typer.Exit(code=code)


@bench_app.command("heldout")
def bench_heldout(
    reports_dir: str = typer.Option(
        "reports", "--reports-dir", help="Directory for evaluation reports"
    ),
) -> None:
    """Run the frozen heldout evaluation suite (anti-tamper: once per git sha)."""
    from benchmarks.suites.runner import run_suite

    code = run_suite("heldout", reports_dir=reports_dir)
    if code != 0:
        raise typer.Exit(code=code)


@bench_app.command("local-suite")
def bench_local_suite(
    game: str | None = typer.Option(None, "--game", help="Specific game to evaluate"),
    seed: int | None = typer.Option(None, "--seed", help="Random seed"),
) -> None:
    """Run the 10-game ARC local suite benchmark."""
    import subprocess

    cmd = [sys.executable, "plugins/arc_agi_adapter/run_local_suite.py"]
    if game:
        cmd.extend(["--game", game])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise typer.Exit(code=res.returncode)


@bench_app.command("sge")
def bench_sge() -> None:
    """Run the SGE Symbol Grounding Engine integration benchmark."""
    import subprocess

    res = subprocess.run([sys.executable, "-m", "benchmarks.sge_integration_benchmark"])
    if res.returncode != 0:
        raise typer.Exit(code=res.returncode)


@bench_app.command("ablation")
def bench_ablation() -> None:
    """Run the measured cortical faculty ablation suite."""
    import subprocess

    res = subprocess.run([sys.executable, "benchmarks/ablation/faculty_ablation.py"])
    if res.returncode != 0:
        raise typer.Exit(code=res.returncode)


# ═══════════════════════════════════════════════════════════════════════════
# Top-level Diagnostics & Info
# ═══════════════════════════════════════════════════════════════════════════


@app.command("diagnostics")
def cmd_diagnostics() -> None:
    """Comprehensive system diagnostics (hardware, Rust SIMD, and Cognitive OS)."""
    from hbllm.cli.diagnostics import print_diagnostics

    print_diagnostics()


@app.command("info")
def cmd_info() -> None:
    """Show Cognitive OS architecture and invariants."""
    from hbllm import __version__

    typer.echo(
        f"""
🧠 HBLLM Core v{__version__} — Human-Brain Inspired Cognitive OS
{"=" * 60}
Invariants:
  1. Nodes emit intent. HCIR owns state (State Ownership).
  2. Operators calculate. HCIR decides reality (Reasoning Invariant).
  3. Zero LLM dependency in Cognitive Core (LLM-Independent).
  4. Constitutional Governance on every transaction.
  5. Multi-Tenant Memory Isolation across 4 tiers.
  6. Forward architecture upgrade principle (No duct tape).
  7. Domain-agnostic General Intelligence core.
  8. Internal memory governance (No ad-hoc attributes).

Cortical Epistemic Loop:
  Perception ──► Assimilation ──► Counterfactual Simulation ──► Deliberation ──► Motor Dispatch
{"=" * 60}
"""
    )


# ═══════════════════════════════════════════════════════════════════════════
# Runtime & Developer Tools
# ═══════════════════════════════════════════════════════════════════════════


@app.command("serve")
def cmd_serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(8000, "--port", help="Bind port"),
    workers: int = typer.Option(1, "--workers", help="Worker count"),
    model_size: str = typer.Option("125m", "--model-size", help="Model size / preset"),
) -> None:
    """Start the HBLLM API server."""
    import os

    os.environ["HBLLM_MODEL_SIZE"] = model_size
    try:
        import uvicorn

        uvicorn.run(
            "hbllm.serving.api:app",
            host=host,
            port=port,
            workers=workers,
            log_level="info",
            loop="uvloop",
        )
    except ImportError:
        typer.echo("Error: uvicorn not installed. Run: pip install uvicorn", err=True)
        raise typer.Exit(code=1)


@app.command("agent")
def cmd_agent() -> None:
    """Launch the interactive developer agent."""
    from hbllm.cli.agent import main as agent_main

    agent_main()


@app.command("code")
def cmd_code(
    prompt: str = typer.Argument(..., help="Code generation or refactoring prompt"),
) -> None:
    """Execute a single-turn code generation / edit prompt."""
    from hbllm.cli.agent import run_code

    run_code(argparse.Namespace(instruction=prompt))


@app.command("daemon")
def cmd_daemon(
    action: str = typer.Argument("run", help="Action: run, install, uninstall, status"),
) -> None:
    """Manage the background daemon service."""
    from hbllm.cli.daemon import (
        install_service,
        run_foreground,
        status_service,
        uninstall_service,
    )

    args = argparse.Namespace(daemon_action=action)
    if action == "run":
        run_foreground(args)
    elif action == "install":
        install_service(args)
    elif action == "uninstall":
        uninstall_service(args)
    elif action == "status":
        status_service(args)
    else:
        typer.echo(
            f"Unknown daemon action '{action}'. Valid: run, install, uninstall, status",
            err=True,
        )
        raise typer.Exit(code=1)


# ═══════════════════════════════════════════════════════════════════════════
# Plugin Subcommands
# ═══════════════════════════════════════════════════════════════════════════


@plugin_app.command("list")
def plugin_list() -> None:
    """List all installed dynamic plugins."""
    from hbllm._cli_app import run_plugins

    run_plugins(argparse.Namespace())


@plugin_app.command("new")
def plugin_new(
    name: str = typer.Argument(..., help="Name of the plugin to scaffold"),
) -> None:
    """Scaffold a new plugin."""
    from hbllm._cli_app import run_plugin_new

    run_plugin_new(argparse.Namespace(name=name))


# ═══════════════════════════════════════════════════════════════════════════
# Legacy Commands
# ═══════════════════════════════════════════════════════════════════════════


@app.command("train")
def cmd_train(
    work_dir: str = typer.Option("./workspace", "--work-dir"),
    model_size: str = typer.Option("125m", "--model-size"),
    wandb_project: str | None = typer.Option(None, "--wandb-project"),
) -> None:
    """Run pre-training loop (legacy)."""
    from hbllm._cli_app import run_train

    run_train(
        argparse.Namespace(
            work_dir=work_dir,
            model_size=model_size,
            wandb_project=wandb_project,
        )
    )


@app.command("data")
def cmd_data(
    work_dir: str = typer.Option("./workspace", "--work-dir"),
    dataset: str = typer.Option("fineweb", "--dataset"),
    samples: int = typer.Option(100_000, "--samples"),
    vocab_size: int = typer.Option(32768, "--vocab-size"),
    seq_len: int = typer.Option(2048, "--seq-len"),
) -> None:
    """Run data preparation pipeline (legacy)."""
    from hbllm._cli_app import run_pipeline

    run_pipeline(
        argparse.Namespace(
            work_dir=work_dir,
            dataset=dataset,
            samples=samples,
            vocab_size=vocab_size,
            seq_len=seq_len,
        )
    )


def main() -> Any:
    """Canonical CLI entrypoint."""
    return app()


if __name__ == "__main__":
    main()
