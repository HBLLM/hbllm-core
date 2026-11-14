"""HBLLM CLI — top-level command dispatcher.

Re-exports ``main`` and ``app`` from the canonical Typer application in ``main``.
"""

from __future__ import annotations

from hbllm.cli.main import app, main

__all__ = ["app", "main"]
