"""
Structured logging for HBLLM.

Supports:
  - JSON structured output (for production log aggregation)
  - Rich console output (for development)
  - Per-module log level configuration
  - Correlation ID context for request tracing
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from typing import Any

# Context variable for request-scoped correlation IDs
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production environments."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Add correlation ID if set
        corr_id = _correlation_id.get("")
        if corr_id:
            log_entry["correlation_id"] = corr_id

        # Add exception info
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        for key in ("node_id", "tenant_id", "session_id", "duration_ms"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        return json.dumps(log_entry, default=str)


def setup_logging(
    level: str = "INFO",
    rich_output: bool = True,
    json_output: bool = False,
    module_levels: dict[str, str] | None = None,
) -> None:
    """
    Configure logging for HBLLM.

    Args:
        level: Default log level
        rich_output: Use Rich handler for colorful console output (dev mode)
        json_output: Use JSON structured format (production mode)
        module_levels: Per-module log levels, e.g. {"hbllm.brain": "DEBUG"}
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # JSON mode — structured logs for production
    if json_output:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logging.basicConfig(
            level=log_level,
            handlers=[handler],
        )
    elif rich_output:
        # Rich console output for development
        try:
            from rich.logging import RichHandler

            logging.basicConfig(
                level=log_level,
                format="%(message)s",
                datefmt="[%X]",
                handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
            )
        except ImportError:
            # Fallback to standard logging
            logging.basicConfig(
                level=log_level,
                format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                stream=sys.stdout,
            )
    else:
        # Standard formatted output
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            stream=sys.stdout,
        )

    # Apply per-module log levels
    if module_levels:
        for module_name, mod_level in module_levels.items():
            mod_log_level = getattr(logging, mod_level.upper(), None)
            if mod_log_level is not None:
                logging.getLogger(module_name).setLevel(mod_log_level)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name, prefixed with hbllm."""
    return logging.getLogger(f"hbllm.{name}")


def set_correlation_id(corr_id: str) -> None:
    """Set the correlation ID for the current async context."""
    _correlation_id.set(corr_id)


def get_correlation_id() -> str:
    """Get the current correlation ID."""
    return _correlation_id.get("")
