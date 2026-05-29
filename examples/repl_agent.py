#!/usr/bin/env python3
"""
HBLLM Interactive Developer Agent (Claude Code / Codex Alternative)
===================================================================

This is a convenience wrapper for development.
The canonical entry point is::

    hbllm agent                               # interactive REPL
    hbllm agent --model ollama/llama3          # use a local model
    hbllm agent "fix the failing test"         # one-shot mode

Or equivalently::

    python -m hbllm.cli agent
"""

import argparse
import sys

from hbllm.cli.agent import run_agent


def main():
    # Build the same argument parser the CLI subcommand uses, so this
    # script can be run standalone without `hbllm` being installed.
    parser = argparse.ArgumentParser(description="HBLLM Developer Agent REPL")
    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="Model provider identifier (e.g. openai/gpt-4o, anthropic/claude-3-5-sonnet, ollama/llama3)",
    )
    parser.add_argument(
        "--no-approval",
        action="store_true",
        help="Disable manual confirmation step before executing local shell commands",
    )
    parser.add_argument(
        "--data-dir",
        default="./workspace/repl_data",
        help="Local database and knowledge persistence directory",
    )
    parser.add_argument(
        "task",
        nargs="?",
        default=None,
        help="Optional one-shot task string. If omitted, starts interactive REPL.",
    )

    args = parser.parse_args()
    run_agent(args)


if __name__ == "__main__":
    main()
