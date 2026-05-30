"""
HBLLM Interactive Developer Agent — ``hbllm agent``

Runs the HBLLM cognitive loop interactively in the workspace directory.
Hooks into git, local compiler diagnostics, language servers, and
executes local shell commands (with configurable manual safety prompts).

Usage::

    hbllm agent
    hbllm agent --model anthropic/claude-3-5-sonnet
    hbllm agent --model ollama/llama3 --no-approval
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace

    from rich.console import Console

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────


async def _listen_to_thoughts(bus: object, console: Console) -> None:
    """Subscribes to system thoughts and prints them in real-time."""

    async def thought_handler(message: object) -> None:
        thought = message.payload.get("text", "")  # type: ignore[attr-defined]
        if thought:
            console.print(f"[dim cyan]🧠 [Thinking] {thought}[/]")

    await bus.subscribe("system.thought", thought_handler)  # type: ignore[attr-defined]


# ── Core REPL loop ──────────────────────────────────────────────────────────


async def _run_agent(
    model: str,
    require_approval: bool,
    data_dir: str,
    task: str | None = None,
    index: bool = False,
    no_index: bool = False,
    tenant_id: str = "default",
) -> None:
    """Boot the cognitive brain and run an interactive or one-shot loop."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    from hbllm.brain.factory import BrainConfig, BrainFactory

    console = Console()

    console.print(
        Panel(
            f"[bold green]Initializing Sovereign Developer Agent[/]\n"
            f"🤖 LLM Backend: [bold cyan]{model}[/]\n"
            f"🛡️ Safety Approval: [bold {'yellow' if require_approval else 'red'}]"
            f"{'ENABLED' if require_approval else 'DISABLED'}[/]\n"
            f"📂 Workspace: [bold blue]{os.getcwd()}[/]",
            title="🧠 HBLLM Agent",
            border_style="green",
        )
    )

    config = BrainConfig(
        data_dir=data_dir,
        system_prompt=(
            "You are a sovereign developer agent capable of writing, compiling, testing, "
            "debugging, and committing code in the local workspace. Use your tools (shell execution, "
            "git operations, LSP queries, and compiler checks) to solve the user's tasks.\n\n"
            "### Active Workspace Scoping\n"
            f"- Active Codebase Root: {os.getcwd()}\n"
            f"- Local Project Databases (.hbllm/): {data_dir}\n"
            f"- Tenant Context: {tenant_id}\n"
            "All localized episodic logs, session graphs, and vector database indices are stored "
            "in the local project path, while your global identity context remains cleanly bound."
        ),
        inject_knowledge=True,
        inject_persistence=True,
        inject_awareness=True,
        inject_plugins=True,
        inject_evaluation=True,
        inject_reflection=True,
        inject_shell=True,
        require_shell_approval=require_approval,
    )

    with console.status("[bold yellow]Booting cognitive brain & scanning plugins...[/]"):
        brain = await BrainFactory.create(provider=model, config=config)
        await _listen_to_thoughts(brain.bus, console)

    # Show loaded plugins
    plugins_table = Table(title="🔌 Active Developer Plugins", border_style="cyan")
    plugins_table.add_column("Plugin Name", style="bold green")
    plugins_table.add_column("Namespace", style="dim yellow")
    plugins_table.add_column("Version", style="dim")

    if brain.plugin_manager and brain.plugin_manager.bundles:
        for name, bundle in brain.plugin_manager.bundles.items():
            plugins_table.add_row(name, bundle.namespace, bundle.bundle.manifest.version)
        console.print(plugins_table)
    else:
        console.print("[yellow]⚠️ No developer plugins active. Standard reasoning only.[/]")

    # ── Workspace Codebase Indexing / Onboarding ────────────────────
    if brain.knowledge_base and not no_index:
        workspace_path = os.path.abspath(os.getcwd())
        existing_source = None
        for source_info in brain.knowledge_base.list_sources():
            if os.path.abspath(source_info["path"]) == workspace_path:
                existing_source = source_info
                break

        should_index = index or (not existing_source)

        if should_index:
            if existing_source:
                console.print(
                    f"[bold yellow]🔄 Updating workspace index: [blue]{workspace_path}[/]...[/]"
                )
                source_id = existing_source["source_id"]
            else:
                console.print(
                    f"[bold yellow]🔍 First-time run: Indexing workspace [blue]{workspace_path}[/]...[/]"
                )
                try:
                    source = brain.knowledge_base.add_source(workspace_path, source_type="folder")
                    source_id = source.source_id
                except Exception as e:
                    console.print(f"[red]⚠️ Failed to register workspace: {e}[/]")
                    source_id = None

            if source_id:
                with console.status("[bold yellow]Ingesting and indexing codebase...[/]"):
                    try:
                        num_chunks = await asyncio.to_thread(
                            brain.knowledge_base.ingest_source, source_id
                        )
                        console.print(
                            f"[bold green]✅ Codebase indexed: Ingested {num_chunks} chunks successfully![/]"
                        )
                    except Exception as e:
                        console.print(f"[red]⚠️ Failed to index codebase: {e}[/]")
        else:
            total_chunks = existing_source.get("chunk_count", 0) if existing_source else 0
            console.print(
                f"[dim]ℹ️ Workspace codebase already indexed ({total_chunks} chunks). "
                f"Use --index to force re-indexing.[/]"
            )

    # ── One-shot mode ────────────────────────────────────────────────
    if task:
        console.print(f"\n[bold yellow]🚀 Running task: {task}[/]\n")
        result = await brain.process(text=task, tenant_id=tenant_id, session_id="oneshot")
        console.print(Panel(result.text, border_style="green", title="Output"))
        await brain.stop()
        return

    # ── Interactive REPL ─────────────────────────────────────────────
    console.print("\n[bold green]Ready! Type your instruction or 'exit'/'quit' to leave.[/]\n")

    session_id = "repl_session"
    while True:
        try:
            user_input = input("hbllm> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break

            console.print("\n[bold yellow]🚀 Sending task to agent...[/]")
            _start_time = asyncio.get_event_loop().time()

            result = await brain.process(
                text=user_input, tenant_id=tenant_id, session_id=session_id
            )
            _duration = asyncio.get_event_loop().time() - _start_time

            console.print("\n[bold green]✨ Response from Agent:[/]")
            console.print(Panel(result.text, border_style="green", title="Output"))

            usage = brain.usage
            if usage:
                console.print(
                    f"[dim]Tokens used: {usage.get('total_tokens', 0)} | "
                    f"Latency: {_duration:.2f}s[/]"
                )
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️ Task aborted by user. Ready for next query.[/]\n")
        except Exception as e:
            console.print(f"\n[bold red]❌ Error executing task: {e}[/]\n")

    console.print("[bold yellow]Stopping agent nodes and saving state...[/]")
    await brain.stop()
    console.print("[bold green]Goodbye![/]")


# ── CLI wiring ──────────────────────────────────────────────────────────────


def register_subcommands(subparsers: object) -> None:
    """Register both the ``agent`` and ``code`` subcommands on the CLI parser."""
    # ── Subcommand: agent ──
    agent_parser = subparsers.add_parser(  # type: ignore[attr-defined]
        "agent",
        help="Launch interactive developer agent in current directory",
    )
    agent_parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="Model provider identifier (e.g. openai/gpt-4o, anthropic/claude-3-5-sonnet, ollama/llama3)",
    )
    agent_parser.add_argument(
        "--no-approval",
        action="store_true",
        help="Disable manual confirmation before executing local shell commands",
    )
    agent_parser.add_argument(
        "--data-dir",
        default="./workspace/agent_data",
        help="Local persistence directory for knowledge and memory",
    )
    agent_parser.add_argument(
        "--index",
        action="store_true",
        help="Force full re-indexing of the workspace codebase",
    )
    agent_parser.add_argument(
        "--no-index",
        action="store_true",
        help="Skip automatic workspace codebase indexing",
    )
    agent_parser.add_argument(
        "task",
        nargs="?",
        default=None,
        help="Optional one-shot task string. If omitted, starts interactive REPL.",
    )

    # ── Subcommand: code ──
    code_parser = subparsers.add_parser(  # type: ignore[attr-defined]
        "code",
        help="Start developer agent session isolated for a specific project path (like 'hbllm code .')",
    )
    code_parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to the project directory to work on (defaults to '.')",
    )
    code_parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="Model provider identifier (e.g. openai/gpt-4o, anthropic/claude-3-5-sonnet, ollama/llama3)",
    )
    code_parser.add_argument(
        "--no-approval",
        action="store_true",
        help="Disable manual confirmation before executing local shell commands",
    )
    code_parser.add_argument(
        "--index",
        action="store_true",
        help="Force full re-indexing of the workspace codebase",
    )
    code_parser.add_argument(
        "--no-index",
        action="store_true",
        help="Skip automatic workspace codebase indexing",
    )
    code_parser.add_argument(
        "--task",
        default=None,
        help="Optional one-shot task string. If omitted, starts interactive REPL.",
    )


def run_agent(args: Namespace) -> None:
    """Entry point called by ``_cli_app.dispatch`` for agent subcommand."""
    logging.getLogger("hbllm").setLevel(logging.WARNING)

    asyncio.run(
        _run_agent(
            model=args.model,
            require_approval=not args.no_approval,
            data_dir=args.data_dir,
            task=args.task,
            index=args.index,
            no_index=args.no_index,
            tenant_id="default",
        )
    )


def run_code(args: Namespace) -> None:
    """Entry point called by ``_cli_app.dispatch`` for code subcommand."""
    import sys

    logging.getLogger("hbllm").setLevel(logging.WARNING)

    # 1. Resolve target path
    target_path = os.path.abspath(os.path.expanduser(args.path))
    if not os.path.isdir(target_path):
        print(f"❌ Error: Path is not a directory: {target_path}")
        sys.exit(1)

    # 2. Shift process directory to the target path
    os.chdir(target_path)

    # 3. Isolate project database local storage under .hbllm/ in the project root path
    data_dir = os.path.join(target_path, ".hbllm")

    # 4. Execute the agent REPL loop under the default tenant identity
    asyncio.run(
        _run_agent(
            model=args.model,
            require_approval=not args.no_approval,
            data_dir=data_dir,
            task=args.task,
            index=args.index,
            no_index=args.no_index,
            tenant_id="default",
        )
    )
