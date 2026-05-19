"""In-Process Async Cognitive Watchdog and Recursion Detector.

This layer provides fine-grained, fast-reaction oversight of the AutonomyCore.
It monitors cognitive budgets, cancels runaway tasks, and detects A->B->A
planning loops. It is designed to be the first line of defense before the
External Process Supervisor is needed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any

logger = logging.getLogger(__name__)


class RecursionDetector:
    """Detects infinite planning loops or cyclic task generation."""

    def __init__(self, history_size: int = 50) -> None:
        self.history_size = history_size
        self._execution_hashes: list[str] = []

    def record_execution(self, task_name: str, args_hash: str) -> bool:
        """Record a task execution.

        Returns:
            True if a recursion cycle is detected, False otherwise.
        """
        exec_id = f"{task_name}:{args_hash}"
        self._execution_hashes.append(exec_id)

        if len(self._execution_hashes) > self.history_size:
            self._execution_hashes.pop(0)

        return self._detect_cycle()

    def _detect_cycle(self) -> bool:
        """Heuristic detection of A->B->A or A->A->A loops."""
        n = len(self._execution_hashes)
        if n < 4:
            return False

        # Check for immediate repetition (A -> A -> A)
        if self._execution_hashes[-1] == self._execution_hashes[-2] == self._execution_hashes[-3]:
            logger.warning(
                "RecursionDetector: Immediate repetition cycle detected (%s)",
                self._execution_hashes[-1],
            )
            return True

        # Check for oscillating repetition (A -> B -> A -> B)
        if n >= 4:
            if (
                self._execution_hashes[-1] == self._execution_hashes[-3]
                and self._execution_hashes[-2] == self._execution_hashes[-4]
            ):
                logger.warning(
                    "RecursionDetector: Oscillating cycle detected (%s -> %s)",
                    self._execution_hashes[-2],
                    self._execution_hashes[-1],
                )
                return True

        return False


class CognitiveWatchdog:
    """Async guardian for the cognitive runtime."""

    def __init__(self, max_deliberation_ms: int = 2000) -> None:
        self.max_deliberation_seconds = max_deliberation_ms / 1000.0
        self.recursion_detector = RecursionDetector()

    async def execute_with_guard(
        self,
        name: str,
        args_hash: str,
        coro: Coroutine[Any, Any, Any],
        timeout_override: float | None = None,
    ) -> Any:
        """Execute a coroutine under watchdog supervision.

        Raises:
            asyncio.TimeoutError if the execution exceeds the deliberation budget.
            RuntimeError if a recursion cycle is detected.
        """
        # 1. Check Recursion
        if self.recursion_detector.record_execution(name, args_hash):
            raise RuntimeError(
                f"CognitiveWatchdog: Recursion cycle detected in '{name}'. Hard stop."
            )

        # 2. Enforce Timeouts
        timeout = (
            timeout_override if timeout_override is not None else self.max_deliberation_seconds
        )

        start_time = time.monotonic()
        try:
            result = await asyncio.wait_for(coro, timeout=timeout)

            elapsed = time.monotonic() - start_time
            if elapsed > (timeout * 0.8):
                logger.warning(
                    "CognitiveWatchdog: '%s' nearly exhausted its time budget (%.2fs / %.2fs)",
                    name,
                    elapsed,
                    timeout,
                )

            return result
        except asyncio.TimeoutError:
            logger.error(
                "CognitiveWatchdog: '%s' exceeded strict timeout of %.2fs and was killed.",
                name,
                timeout,
            )
            raise
