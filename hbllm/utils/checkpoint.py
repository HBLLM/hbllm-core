"""
Safe checkpoint loading utilities for HBLLM.

Provides ``load_checkpoint()`` which wraps ``torch.load()`` with
``weights_only=True`` by default, preventing arbitrary code execution
from untrusted checkpoint files (CVE-2025-XXXX class vulnerabilities).

For legacy checkpoints that contain non-tensor objects (e.g., optimizer
state with custom LR schedulers), falls back to ``weights_only=False``
with a logged security warning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import torch

logger = logging.getLogger(__name__)

# Allowlist of safe classes that torch.load may need to unpickle
# when loading HBLLM checkpoints with weights_only=True.
_SAFE_GLOBALS: list[Any] = []


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict_security: bool = True,
) -> dict[str, Any]:
    """
    Load a PyTorch checkpoint safely.

    Attempts ``weights_only=True`` first. If that fails (e.g., checkpoint
    contains optimizer state with custom classes), retries with
    ``weights_only=False`` and logs a security warning — unless
    ``strict_security=True``, in which case it raises the error.

    Args:
        path: Path to the ``.pt`` checkpoint file.
        map_location: Device to map tensors to (default: ``"cpu"``).
        strict_security: If True, never fall back to ``weights_only=False``.

    Returns:
        The loaded checkpoint dict.

    Raises:
        FileNotFoundError: If the checkpoint file doesn't exist.
        RuntimeError: If strict_security=True and the checkpoint
            requires ``weights_only=False``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Try safe loading first
    try:
        ckpt = cast(
            dict[str, Any],
            torch.load(
                str(path),
                map_location=map_location,
                weights_only=True,
            ),
        )
        logger.debug("Loaded checkpoint safely (weights_only=True): %s", path.name)
        return ckpt
    except Exception as e:
        if strict_security:
            raise RuntimeError(
                f"Checkpoint {path.name} requires weights_only=False but "
                f"strict_security=True. Error: {e}"
            ) from e

        logger.warning(
            "Checkpoint '%s' requires weights_only=False (contains non-tensor "
            "objects). Loading in compatibility mode. Only load checkpoints "
            "you trust! Error was: %s",
            path.name,
            type(e).__name__,
        )
        return cast(
            dict[str, Any],
            torch.load(
                str(path),
                map_location=map_location,
                weights_only=False,
            ),
        )


def extract_model_state(ckpt: dict[str, Any]) -> dict[str, Any]:
    """
    Extract the model state dict from a checkpoint, handling both
    ``{"model_state_dict": ...}`` and bare state dict formats.
    """
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return cast(dict[str, Any], ckpt["model_state_dict"])
    return ckpt
