"""Structured logging for HBLLM."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO", rich_output: bool = True) -> None:
    """
    Configure logging for HBLLM.

    Uses Rich handler for beautiful console output if available,
    falls back to standard logging otherwise.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    if rich_output:
        try:
            from rich.logging import RichHandler

            logging.basicConfig(
                level=log_level,
                format="%(message)s",
                datefmt="[%X]",
                handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
            )
            return
        except ImportError:
            pass

    # Fallback to standard logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name, prefixed with hbllm."""
    return logging.getLogger(f"hbllm.{name}")
