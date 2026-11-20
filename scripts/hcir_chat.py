#!/usr/bin/env python3
"""HCIR Interactive Chat — Pure non-LLM cognitive chat with optional NVIDIA NIM shadow.

Usage:
    python scripts/hcir_chat.py                   # Auto mode (HCIR first, NVIDIA fallback)
    python scripts/hcir_chat.py --mode pure        # Pure HCIR only (no LLM fallback)
    python scripts/hcir_chat.py --mode nvidia      # Always shadow with NVIDIA NIM

Commands inside the REPL:
    /metrics    — Show autonomy metrics
    /verbs      — List known command verbs
    /caps       — List registered capabilities
    /status     — System status
    /reset      — Reset conversation state
    /quit       — Exit
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_dotenv() -> None:
    """Load .env from repo root if present."""
    env_file = Path(_REPO_ROOT) / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


def build_system(enable_curiosity: bool = False):
    """Boot the CognitiveSystem from the canonical bootstrap."""
    from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

    cfg = CognitiveSystemConfig(
        world_id="chat",
        enable_grounding=False,
        enable_brain_llm=False,
        enable_autonomous_curiosity=enable_curiosity,
    )
    return build_cognitive_system(cfg)


def attach_nvidia_scaffold(system):
    """Optionally attach NVIDIA NIM scaffold if API key is available."""
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        return None

    try:
        from plugins.llm_scaffold.teacher import LLMTeacherScaffold

        scaffold = LLMTeacherScaffold.from_nvidia_endpoint(api_key=api_key)
        system.services.extensions["llm_scaffold"] = scaffold
        return scaffold
    except Exception as exc:
        print(f"⚠  NVIDIA scaffold attach failed: {exc}")
        return None


def format_metrics(system) -> str:
    """Format autonomy metrics for display."""
    metrics = system.services.extensions.get("autonomy_metrics", {})
    parse_att = metrics.get("parse_attempts", 0)
    parse_ok = metrics.get("parse_hcir_success", 0)
    shadow = metrics.get("shadow_invocations", 0)
    distilled = metrics.get("distilled_lessons", 0)

    parse_rate = (parse_ok / parse_att * 100) if parse_att > 0 else 0.0
    autonomy_rate = ((parse_att - shadow) / parse_att * 100) if parse_att > 0 else 0.0

    lines = [
        "╔═══════════════ Autonomy Metrics ═══════════════╗",
        f"  Parse attempts:     {parse_att}",
        f"  HCIR success:       {parse_ok} ({parse_rate:.1f}%)",
        f"  Shadow invocations: {shadow}",
        f"  Distilled lessons:  {distilled}",
        f"  Autonomy rate:      {autonomy_rate:.1f}%",
        "╚════════════════════════════════════════════════╝",
    ]
    return "\n".join(lines)


def handle_meta_command(cmd: str, system) -> str | None:
    """Handle REPL meta-commands (starting with /). Returns response or None."""
    cmd = cmd.strip().lower()

    if cmd == "/metrics":
        return format_metrics(system)

    if cmd == "/verbs":
        dispatcher = system.services.extensions.get("command_dispatcher")
        if dispatcher is None:
            from hbllm.hcir.kernel.command_dispatcher import CommandDispatcher

            dispatcher = CommandDispatcher(
                system.services.capability_resolver,
                workspace=system.workspace,
            )
            system.services.extensions["command_dispatcher"] = dispatcher
        verbs = dispatcher.known_verbs()
        return f"Known verbs ({len(verbs)}): {', '.join(verbs)}"

    if cmd == "/caps":
        caps = system.services.capability_resolver.list_capabilities()
        return f"Registered capabilities ({len(caps)}):\n" + "\n".join(
            f"  • {c}" for c in sorted(caps)
        )

    if cmd == "/status":
        graph = system.workspace.graph
        node_count = graph.node_count
        edge_count = graph.edge_count
        scaffold = system.services.extensions.get("llm_scaffold")
        scaffold_status = "attached" if scaffold else "none"
        lines = [
            "╔═══════════════ System Status ══════════════════╗",
            f"  Graph nodes:  {node_count}",
            f"  Graph edges:  {edge_count}",
            f"  LLM scaffold: {scaffold_status}",
            f"  World ID:     {system.config.world_id}",
            "╚════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)

    if cmd == "/reset":
        system.services.extensions["autonomy_metrics"] = {
            "parse_attempts": 0,
            "parse_hcir_success": 0,
            "realize_attempts": 0,
            "realize_hcir_success": 0,
            "shadow_invocations": 0,
            "distilled_lessons": 0,
        }
        return "Metrics reset."

    if cmd in ("/quit", "/exit", "/q"):
        return "__QUIT__"

    if cmd == "/help":
        return (
            "Available commands:\n"
            "  /metrics  — Show autonomy metrics\n"
            "  /verbs    — List known command verbs\n"
            "  /caps     — List registered capabilities\n"
            "  /status   — System status\n"
            "  /reset    — Reset metrics\n"
            "  /help     — This help\n"
            "  /quit     — Exit"
        )

    return None


def run_repl(system, mode: str = "auto") -> None:
    """Run the interactive REPL loop."""
    scaffold = system.services.extensions.get("llm_scaffold")

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          HCIR Interactive Chat (Pure Non-LLM Core)         ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Mode: {mode:<10}  NVIDIA scaffold: {'yes' if scaffold else 'no':<20}  ║")
    print("║  Type /help for commands, /quit to exit                    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    while True:
        try:
            user_input = input("\033[1;36mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Meta commands
        if user_input.startswith("/"):
            result = handle_meta_command(user_input, system)
            if result == "__QUIT__":
                print("Goodbye!")
                break
            if result is not None:
                print(f"\033[90m{result}\033[0m")
            else:
                print(f"\033[33mUnknown command: {user_input}. Try /help\033[0m")
            continue

        # Process through CognitiveSystem
        t0 = time.monotonic()

        if mode == "pure":
            # Pure HCIR only — no scaffold fallback
            original_scaffold = system.services.extensions.pop("llm_scaffold", None)
            try:
                turn = system.process_dialogue(user_input, speaker_id="user")
            finally:
                if original_scaffold is not None:
                    system.services.extensions["llm_scaffold"] = original_scaffold
        elif mode == "nvidia":
            # Always invoke scaffold (even if HCIR succeeds, also shadow for learning)
            turn = system.process_dialogue(user_input, speaker_id="user")
        else:
            # Auto: HCIR first, scaffold fallback on failure
            turn = system.process_dialogue(user_input, speaker_id="user")

        latency = (time.monotonic() - t0) * 1000

        # Display response
        if isinstance(turn, dict):
            # Scaffold result
            response = turn.get("response", turn.get("text", str(turn)))
            source = "NVIDIA"
            print(f"\033[1;33mHCIR [{source}] ({latency:.0f}ms):\033[0m {response}")
        else:
            # LanguageTurnResult
            source = "Pure-HCIR"
            frame_type = turn.semantic_frame.frame_type if hasattr(turn, "semantic_frame") else "?"
            response = getattr(turn, "response_text", "") or "No response generated."

            if getattr(turn, "is_success", False):
                color = "\033[1;32m"  # Green
            else:
                color = "\033[1;31m"  # Red

            print(f"{color}HCIR [{source}|{frame_type}] ({latency:.0f}ms):\033[0m {response}")

        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HCIR Interactive Chat — Pure non-LLM cognitive chat"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "pure", "nvidia"],
        default="auto",
        help="Chat mode: auto (HCIR+fallback), pure (HCIR only), nvidia (always shadow)",
    )
    parser.add_argument(
        "--curiosity",
        action="store_true",
        help="Enable autonomous epistemic curiosity (governed web research on knowledge gaps)",
    )
    args = parser.parse_args()

    _load_dotenv()

    print("Booting cognitive system...")
    system = build_system(enable_curiosity=args.curiosity)
    print(f"  ✓ System booted (graph: {system.workspace.graph.node_count} nodes)")
    if args.curiosity:
        print("  ✓ Autonomous epistemic curiosity enabled (governed web.search/web.fetch)")

    scaffold = attach_nvidia_scaffold(system)
    if scaffold:
        print("  ✓ NVIDIA NIM scaffold attached")
    else:
        if args.mode == "nvidia":
            print("  ⚠ NVIDIA API key not found. Set NVIDIA_API_KEY in .env")
            print("    Falling back to pure HCIR mode.")
            args.mode = "pure"
        print("  ℹ Running without LLM scaffold (pure HCIR)")

    print()
    run_repl(system, mode=args.mode)

    # Print final metrics
    print()
    print(format_metrics(system))


if __name__ == "__main__":
    main()
