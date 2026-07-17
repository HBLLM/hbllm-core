"""
CLI Transport — Interactive terminal interface for HBLLM.

A thin transport adapter (~250 lines) that connects a Rich-powered
terminal UI to the Gateway. Contains zero cognitive logic.

Architecture::

    User types in terminal
        ↓
    CLITransport  (this module)
        ↓
    Gateway.handle_inbound()
        ↓
    ConversationBus → Executive

Usage::

    from hbllm.network.transports.cli import CLITransport
    from hbllm.network.gateway import Gateway

    gateway = Gateway(bus)
    cli = CLITransport(gateway)
    await cli.run()
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from hbllm.network.session import (
    SessionMessage,
    TransportType,
)

logger = logging.getLogger(__name__)

# Transport ID constant
CLI_TRANSPORT_ID = "cli-interactive"

# ═══════════════════════════════════════════════════════════════════════════
# ANSI Helpers (fallback when Rich is not installed)
# ═══════════════════════════════════════════════════════════════════════════

_CYAN = "\033[36m"
_GREEN = "\033[32m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"
_YELLOW = "\033[33m"
_RED = "\033[31m"


# ═══════════════════════════════════════════════════════════════════════════
# Rich Renderer (optional dependency)
# ═══════════════════════════════════════════════════════════════════════════


class _RichRenderer:
    """Wrapper around Rich console for styled output.

    Falls back to plain ANSI if Rich is not installed.
    """

    def __init__(self) -> None:
        self._console: Any = None
        self._has_rich = False
        try:
            from rich.console import Console
            from rich.markdown import Markdown
            from rich.panel import Panel

            self._console = Console()
            self._has_rich = True
        except ImportError:
            pass

    def print_banner(self) -> None:
        if self._has_rich:
            from rich.panel import Panel

            self._console.print(
                Panel.fit(
                    "[bold cyan]🧠 HBLLM Cognitive OS[/bold cyan]\n"
                    "[dim]Type your message. Use /quit to exit, /help for commands.[/dim]",
                    border_style="cyan",
                )
            )
        else:
            print(f"\n{_BOLD}{_CYAN}🧠 HBLLM Cognitive OS{_RESET}")
            print(f"{_DIM}Type your message. Use /quit to exit, /help for commands.{_RESET}\n")

    def print_user_prompt(self) -> None:
        if self._has_rich:
            self._console.print("\n[bold green]You ›[/bold green] ", end="")
        else:
            print(f"\n{_BOLD}{_GREEN}You ›{_RESET} ", end="")

    def print_response(self, text: str) -> None:
        if self._has_rich:
            from rich.markdown import Markdown

            self._console.print()
            self._console.print(
                "[bold cyan]HBLLM ›[/bold cyan]",
            )
            self._console.print(Markdown(text))
        else:
            print(f"\n{_BOLD}{_CYAN}HBLLM ›{_RESET}")
            print(text)

    def print_system(self, text: str) -> None:
        if self._has_rich:
            self._console.print(f"[dim]{text}[/dim]")
        else:
            print(f"{_DIM}{text}{_RESET}")

    def print_error(self, text: str) -> None:
        if self._has_rich:
            self._console.print(f"[bold red]Error:[/bold red] {text}")
        else:
            print(f"{_BOLD}{_RED}Error:{_RESET} {text}")


# ═══════════════════════════════════════════════════════════════════════════
# Slash Commands
# ═══════════════════════════════════════════════════════════════════════════


class _SlashCommands:
    """Handles CLI slash commands."""

    COMMANDS = {
        "/quit": "Exit the CLI",
        "/exit": "Exit the CLI",
        "/help": "Show available commands",
        "/clear": "Clear conversation history",
        "/session": "Show current session info",
        "/workspace": "Show or switch workspace",
        "/memory": "Query memory systems",
        "/status": "Show system status",
    }

    @staticmethod
    def is_command(text: str) -> bool:
        return text.strip().startswith("/")

    @staticmethod
    def parse(text: str) -> tuple[str, str]:
        """Parse a slash command into (command, args)."""
        parts = text.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        return cmd, args


# ═══════════════════════════════════════════════════════════════════════════
# CLITransport
# ═══════════════════════════════════════════════════════════════════════════


class CLITransport:
    """Interactive terminal transport for HBLLM.

    This is a thin adapter. It:
      1. Reads user input from stdin.
      2. Sends it to the Gateway via handle_inbound().
      3. Receives responses via a registered callback.
      4. Renders responses to the terminal.

    Contains NO cognitive logic whatsoever.
    """

    def __init__(
        self,
        gateway: Any,  # Gateway
        *,
        tenant_id: str = "default",
        user_id: str = "default",
        device_id: str = "local",
        workspace_id: str = "default",
    ) -> None:
        self._gateway = gateway
        self._tenant_id = tenant_id
        self._user_id = user_id
        self._device_id = device_id
        self._workspace_id = workspace_id

        self._renderer = _RichRenderer()
        self._response_event = asyncio.Event()
        self._pending_response: str = ""
        self._running = False
        self._session_id: str | None = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Register with the Gateway and start the interactive loop."""
        self._gateway.register_transport(CLI_TRANSPORT_ID, self._on_response)
        self._running = True
        logger.info("CLI transport registered with Gateway")

    async def stop(self) -> None:
        """Unregister and stop."""
        self._running = False
        self._gateway.unregister_transport(CLI_TRANSPORT_ID)
        logger.info("CLI transport stopped")

    async def run(self) -> None:
        """Main interactive loop.

        Reads from stdin, dispatches to Gateway, waits for response.
        """
        await self.start()
        self._renderer.print_banner()

        loop = asyncio.get_event_loop()

        try:
            while self._running:
                self._renderer.print_user_prompt()
                sys.stdout.flush()

                # Read input asynchronously
                try:
                    user_input = await loop.run_in_executor(None, self._read_line)
                except (EOFError, KeyboardInterrupt):
                    break

                if not user_input or not user_input.strip():
                    continue

                text = user_input.strip()

                # Handle slash commands
                if _SlashCommands.is_command(text):
                    should_exit = await self._handle_command(text)
                    if should_exit:
                        break
                    continue

                # Dispatch to Gateway
                self._response_event.clear()
                self._pending_response = ""

                self._session_id = await self._gateway.handle_inbound(
                    transport_type=TransportType.CLI,
                    transport_id=CLI_TRANSPORT_ID,
                    tenant_id=self._tenant_id,
                    user_id=self._user_id,
                    device_id=self._device_id,
                    workspace_id=self._workspace_id,
                    text=text,
                )

                # Wait for response from Brain (via Gateway callback)
                try:
                    await asyncio.wait_for(self._response_event.wait(), timeout=120.0)
                    self._renderer.print_response(self._pending_response)
                except asyncio.TimeoutError:
                    self._renderer.print_error("Response timed out after 120s")

        finally:
            await self.stop()
            self._renderer.print_system("\nGoodbye! 👋")

    # ── Response Callback ────────────────────────────────────────────────

    async def _on_response(self, session_id: str, message: SessionMessage) -> None:
        """Called by the Gateway when the Brain responds."""
        self._pending_response = message.text
        self._response_event.set()

    # ── Input Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _read_line() -> str:
        """Blocking readline (runs in executor)."""
        return sys.stdin.readline()

    # ── Slash Command Handlers ───────────────────────────────────────────

    async def _handle_command(self, text: str) -> bool:
        """Handle a slash command.

        Returns True if the CLI should exit.
        """
        cmd, args = _SlashCommands.parse(text)

        if cmd in ("/quit", "/exit"):
            return True

        if cmd == "/help":
            self._renderer.print_system("\nAvailable commands:")
            for c, desc in _SlashCommands.COMMANDS.items():
                self._renderer.print_system(f"  {c:15s} {desc}")
            return False

        if cmd == "/clear":
            # Clear the session by closing it (a new one will be created)
            session = self._gateway.get_session(self._tenant_id, self._user_id, CLI_TRANSPORT_ID)
            if session:
                session.close()
            self._renderer.print_system("Conversation cleared.")
            return False

        if cmd == "/session":
            session = self._gateway.get_session(self._tenant_id, self._user_id, CLI_TRANSPORT_ID)
            if session:
                self._renderer.print_system(f"  Session:   {session.id[:12]}...")
                self._renderer.print_system(f"  State:     {session.state}")
                self._renderer.print_system(f"  Messages:  {session.message_count}")
                self._renderer.print_system(f"  Workspace: {session.workspace_id}")
            else:
                self._renderer.print_system("  No active session.")
            return False

        if cmd == "/workspace":
            if args:
                self._workspace_id = args.strip()
                self._renderer.print_system(f"Switched to workspace: {self._workspace_id}")
            else:
                self._renderer.print_system(f"Current workspace: {self._workspace_id}")
            return False

        if cmd == "/status":
            active = self._gateway.active_session_count
            self._renderer.print_system(f"  Active sessions: {active}")
            self._renderer.print_system("  Transport:       CLI")
            self._renderer.print_system(f"  Workspace:       {self._workspace_id}")
            return False

        self._renderer.print_error(f"Unknown command: {cmd}")
        return False
