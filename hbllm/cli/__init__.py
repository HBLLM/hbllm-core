"""
HBLLM CLI — top-level command dispatcher.

Re-exports ``main`` from the sibling ``_cli_app`` module so that
``pyproject.toml``'s ``hbllm.cli:main`` entry-point works even though
``hbllm/cli/`` is a package (which shadows the old ``hbllm/cli.py``).
"""

from hbllm._cli_app import main

__all__ = ["main"]
