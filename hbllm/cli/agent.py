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
    from argparse import Namespace

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
    from pathlib import Path

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    from hbllm.network.bus import InProcessBus
    from hbllm.network.plugin_manager import PluginManager
    from hbllm.network.registry import ServiceRegistry
    from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

    console = Console()

    console.print(
        Panel(
            f"[bold green]Initializing Sovereign HCIR Developer Agent[/]\n"
            f"🤖 Fallback Model: [bold cyan]{model}[/]\n"
            f"🛡️ Safety Approval: [bold {'yellow' if require_approval else 'red'}]"
            f"{'ENABLED' if require_approval else 'DISABLED'}[/]\n"
            f"📂 Workspace: [bold blue]{os.getcwd()}[/]",
            title="🧠 HBLLM Agent",
            border_style="green",
        )
    )

    bus = InProcessBus()
    await bus.start()
    registry = ServiceRegistry()

    with console.status("[bold yellow]Booting HCIR CognitiveSystem & scanning plugins...[/]"):
        cog_sys = build_cognitive_system(
            CognitiveSystemConfig(
                world_id="dev_agent",
                memory_dir=Path(data_dir) / "hcir",
                tenant_id=tenant_id,
                enable_autonomous_curiosity=True,
            )
        )
        if cog_sys.runtime is not None:
            await cog_sys.runtime.start()
        plugin_dir = Path(__file__).resolve().parent.parent.parent / "plugins"
        pm = PluginManager(plugin_dirs=[plugin_dir], bus=bus, registry=registry, brain=cog_sys)
        pm.discover()
        await pm.load_all()
        await _listen_to_thoughts(bus, console)

    plugins_table = Table(title="🔌 Active Developer Plugins", border_style="cyan")
    plugins_table.add_column("Plugin Name", style="bold green")
    plugins_table.add_column("Namespace", style="dim yellow")
    plugins_table.add_column("Version", style="dim")

    if pm.bundles:
        for name, bundle in pm.bundles.items():
            plugins_table.add_row(name, bundle.namespace, bundle.bundle.manifest.version)
        console.print(plugins_table)
    else:
        console.print("[yellow]⚠️ No developer plugins active. Pure-HCIR reasoning only.[/]")

    provider = None
    try:
        from hbllm.serving.provider import get_provider

        provider = get_provider(model)
    except Exception:
        provider = None

    async def _execute_query(prompt_text: str) -> str:
        turn_result = await asyncio.to_thread(
            cog_sys.process_dialogue, prompt_text, speaker_id="developer"
        )
        resp_text = str(getattr(turn_result, "response_text", "") or "")
        if bool(getattr(turn_result, "is_success", False)) and resp_text:
            return resp_text
        if provider is not None:
            llm_resp = await provider.generate(
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=1024,
            )
            return str(getattr(llm_resp, "content", "") or str(llm_resp))
        return resp_text or "[HCIR] Instruction parsed; no action output returned."

    if task:
        console.print(f"\n[bold yellow]🚀 Running task: {task}[/]\n")
        output = await _execute_query(task)
        console.print(Panel(output, border_style="green", title="Output"))
        if cog_sys.runtime is not None:
            await cog_sys.runtime.stop()
        await bus.stop()
        return

    console.print("\n[bold green]Ready! Type your instruction or 'exit'/'quit' to leave.[/]\n")

    while True:
        try:
            user_input = input("hbllm> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break

            console.print("\n[bold yellow]🚀 Sending task to agent...[/]")
            _start_time = asyncio.get_running_loop().time()
            output = await _execute_query(user_input)
            _duration = asyncio.get_running_loop().time() - _start_time

            console.print("\n[bold green]✨ Response from Agent:[/]")
            console.print(Panel(output, border_style="green", title="Output"))
            console.print(f"[dim]Latency: {_duration:.2f}s[/]\n")

        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️ Task aborted by user. Ready for next query.[/]\n")
        except Exception as e:
            console.print(f"\n[bold red]❌ Error executing task: {e}[/]\n")

    console.print("[bold yellow]Stopping HCIR agent and saving state...[/]")
    if cog_sys.runtime is not None:
        await cog_sys.runtime.stop()
    await bus.stop()
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
        default=os.environ.get(
            "HBLLM_DEFAULT_MODEL", os.environ.get("HBLLM_MODEL", "openai/gpt-4o-mini")
        ),
        help="Model provider identifier (e.g. openai/gpt-4o, anthropic/claude-3-5-sonnet, ollama/llama3, groq/llama3-8b-8192, nvidia/meta/llama-3.1-70b-instruct)",
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
        default=os.environ.get(
            "HBLLM_DEFAULT_MODEL", os.environ.get("HBLLM_MODEL", "openai/gpt-4o-mini")
        ),
        help="Model provider identifier (e.g. openai/gpt-4o, anthropic/claude-3-5-sonnet, ollama/llama3, groq/llama3-8b-8192, nvidia/meta/llama-3.1-70b-instruct)",
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

    from hbllm.security.identity_resolver import resolve_sovereign_identity
    from hbllm.security.tenant_guard import TenantContext
    from hbllm.security.tenant_registry import get_tenant_registry

    dev_tenant_id, device_id = resolve_sovereign_identity()

    # Register standard developer profile centrally
    registry = get_tenant_registry()
    registry.register_tenant(dev_tenant_id, parent_id=None, name="Developer Profile")

    with TenantContext(dev_tenant_id, device_id=device_id):
        asyncio.run(
            _run_agent(
                model=args.model,
                require_approval=not args.no_approval,
                data_dir=args.data_dir,
                task=args.task,
                index=args.index,
                no_index=args.no_index,
                tenant_id=dev_tenant_id,
            )
        )


def run_code(args: Namespace) -> None:
    """Entry point called by ``_cli_app.dispatch`` for code subcommand."""
    import hashlib
    import re
    import sys

    logging.getLogger("hbllm").setLevel(logging.WARNING)

    from hbllm.security.identity_resolver import resolve_sovereign_identity
    from hbllm.security.tenant_guard import TenantContext
    from hbllm.security.tenant_registry import get_tenant_registry

    dev_tenant_id, device_id = resolve_sovereign_identity()

    # 1. Resolve target path
    target_path = os.path.abspath(os.path.expanduser(args.path))
    if not os.path.isdir(target_path):
        print(f"❌ Error: Path is not a directory: {target_path}")
        sys.exit(1)

    # 2. Shift process directory to the target path
    os.chdir(target_path)

    # 3. Compute unique project identifier and register as a child of the developer centrally
    path_hash = hashlib.sha256(target_path.encode("utf-8")).hexdigest()[:8]
    folder_name = re.sub(r"[^a-zA-Z0-9_-]", "_", os.path.basename(target_path))
    project_tenant_id = f"project_{folder_name}_{path_hash}"

    registry = get_tenant_registry()
    registry.register_tenant(dev_tenant_id, parent_id=None, name="Developer Profile")
    registry.register_tenant(
        project_tenant_id,
        parent_id=dev_tenant_id,
        name=f"Workspace Project: {folder_name} ({target_path})",
    )

    # 4. Isolate project database local storage under .hbllm/ in the project root path
    data_dir = os.path.join(target_path, ".hbllm")

    # 5. Execute the agent REPL loop under the project tenant identity context
    with TenantContext(project_tenant_id, device_id=device_id):
        asyncio.run(
            _run_agent(
                model=args.model,
                require_approval=not args.no_approval,
                data_dir=data_dir,
                task=args.task,
                index=args.index,
                no_index=args.no_index,
                tenant_id=project_tenant_id,
            )
        )
