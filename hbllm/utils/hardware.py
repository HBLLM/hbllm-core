"""
Hardware Detection Utilities.

Provides a single, canonical source of truth for hardware capability detection.
Used by the pipeline, brain factory, and router to adjust behavior for
CPU-only vs GPU-accelerated environments.
"""

from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def is_slow_cpu() -> bool:
    """Detect whether the system lacks GPU acceleration (CUDA or Apple MPS).

    Returns True on CPU-only hardware, False when a GPU is available.
    The result is cached for the lifetime of the process.
    """
    try:
        import torch

        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return not (has_cuda or has_mps)
    except ImportError:
        return True
