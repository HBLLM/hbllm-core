"""
Kernel Executor — Centralized execution service for CPU and I/O tasks.

All cognitive subsystems route blocking operations (embedding calculation,
tokenization, heavy mathematical transformations, file/SQLite I/O) through
this executor to prevent event loop stalls.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class KernelExecutor:
    """
    Centralized thread pool executor managed by the Cognitive OS kernel.

    Provides priority-isolated thread pools for CPU-bound and I/O-bound tasks.
    """

    def __init__(self, max_cpu_workers: int = 4, max_io_workers: int = 8) -> None:
        self._cpu_pool = ThreadPoolExecutor(
            max_workers=max_cpu_workers,
            thread_name_prefix="kernel-cpu",
        )
        self._io_pool = ThreadPoolExecutor(
            max_workers=max_io_workers,
            thread_name_prefix="kernel-io",
        )
        self._shutdown = False

    async def run_cpu_bound(self, fn: Callable[..., T], *args: Any, name: str = "") -> T:
        """
        Offload a CPU-intensive synchronous function to the CPU thread pool.

        Args:
            fn: Callable to execute.
            *args: Arguments for the callable.
            name: Diagnostic name for telemetry logging.

        Returns:
            Result of the callable execution.
        """
        if self._shutdown:
            raise RuntimeError("KernelExecutor is shut down")

        loop = asyncio.get_running_loop()
        if name:
            logger.debug("[KernelExecutor] Scheduling CPU task: %s", name)
        return await loop.run_in_executor(self._cpu_pool, fn, *args)

    async def run_io_bound(self, fn: Callable[..., T], *args: Any, name: str = "") -> T:
        """
        Offload an I/O-blocking synchronous function to the I/O thread pool.

        Args:
            fn: Callable to execute.
            *args: Arguments for the callable.
            name: Diagnostic name for telemetry logging.

        Returns:
            Result of the callable execution.
        """
        if self._shutdown:
            raise RuntimeError("KernelExecutor is shut down")

        loop = asyncio.get_running_loop()
        if name:
            logger.debug("[KernelExecutor] Scheduling I/O task: %s", name)
        return await loop.run_in_executor(self._io_pool, fn, *args)

    def shutdown(self, wait: bool = True) -> None:
        """Shut down thread pools gracefully."""
        self._shutdown = True
        self._cpu_pool.shutdown(wait=wait)
        self._io_pool.shutdown(wait=wait)
        logger.info("[KernelExecutor] Thread pools shut down")
