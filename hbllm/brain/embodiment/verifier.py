"""Execution Verification Engine.

Answers the question: "Did the real world actually change?"
Verifies tool execution by querying OS sensors asynchronously, rather
than blindly trusting HTTP 200 or tool return codes.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from hbllm.brain.embodiment.os_adapter import OSAdapter

logger = logging.getLogger(__name__)


class ExecutionVerifier:
    """Asynchronously polls the OS to confirm physical state changes."""

    def __init__(self, adapter: OSAdapter) -> None:
        self.adapter = adapter

    async def verify_file_creation(self, filepath: str, max_wait_s: float = 5.0) -> bool:
        """Poll the filesystem to verify a file was actually created."""
        return await self._poll_condition(
            condition=lambda: self.adapter.check_file_exists(filepath),
            expected=True,
            max_wait_s=max_wait_s,
            interval_s=0.5,
        )

    async def verify_file_deletion(self, filepath: str, max_wait_s: float = 5.0) -> bool:
        """Poll the filesystem to verify a file was actually deleted."""
        return await self._poll_condition(
            condition=lambda: self.adapter.check_file_exists(filepath),
            expected=False,
            max_wait_s=max_wait_s,
            interval_s=0.5,
        )

    async def _poll_condition(
        self, condition: Callable[[], bool], expected: bool, max_wait_s: float, interval_s: float
    ) -> bool:
        """Generic non-blocking polling loop."""
        attempts = int(max_wait_s / interval_s)
        for i in range(attempts):
            if condition() == expected:
                logger.info("Physical verification succeeded after %d attempts.", i + 1)
                return True
            await asyncio.sleep(interval_s)

        logger.warning("Physical verification FAILED. Condition not met after %.1fs", max_wait_s)
        return False
