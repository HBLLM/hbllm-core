"""Execution Verification Engine.

Answers the question: "Did the real world actually change?"
Verifies tool execution by querying OS sensors asynchronously, rather
than blindly trusting HTTP 200 or tool return codes.

Supports verification of:
    - File operations (create, delete)
    - Process operations (launch, kill)
    - Volume/brightness changes
    - Network connectivity
    - Generic sensor-based assertions
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from hbllm.brain.embodiment.os_adapter import OSAdapter

logger = logging.getLogger(__name__)


class ExecutionVerifier:
    """Asynchronously polls the OS to confirm physical state changes.

    The verifier is the "reality check" layer — after the brain executes
    an action, the verifier confirms the real world actually changed.
    """

    def __init__(self, adapter: OSAdapter) -> None:
        self.adapter = adapter
        self._verifications_run = 0
        self._verifications_passed = 0
        self._verifications_failed = 0

    async def verify_file_creation(self, filepath: str, max_wait_s: float = 5.0) -> bool:
        """Poll the filesystem to verify a file was actually created."""
        result = await self._poll_condition(
            condition=lambda: self.adapter.check_file_exists(filepath),
            expected=True,
            max_wait_s=max_wait_s,
            interval_s=0.5,
            description=f"file_creation({filepath})",
        )
        return result

    async def verify_file_deletion(self, filepath: str, max_wait_s: float = 5.0) -> bool:
        """Poll the filesystem to verify a file was actually deleted."""
        result = await self._poll_condition(
            condition=lambda: self.adapter.check_file_exists(filepath),
            expected=False,
            max_wait_s=max_wait_s,
            interval_s=0.5,
            description=f"file_deletion({filepath})",
        )
        return result

    async def verify_process_running(
        self,
        process_name: str,
        max_wait_s: float = 10.0,
    ) -> bool:
        """Verify that a process with the given name is running."""

        def _check() -> bool:
            procs = self.adapter.read_active_processes(top_n=50)
            return any(p.get("name", "").lower() == process_name.lower() for p in procs)

        return await self._poll_condition(
            condition=_check,
            expected=True,
            max_wait_s=max_wait_s,
            interval_s=1.0,
            description=f"process_running({process_name})",
        )

    async def verify_volume_level(
        self,
        expected_level: float,
        tolerance: float = 0.05,
        max_wait_s: float = 3.0,
    ) -> bool:
        """Verify that system volume matches expected level."""

        def _check() -> bool:
            current = self.adapter.read_volume()
            if current is None:
                return False
            return abs(current - expected_level) <= tolerance

        return await self._poll_condition(
            condition=_check,
            expected=True,
            max_wait_s=max_wait_s,
            interval_s=0.5,
            description=f"volume_level({expected_level})",
        )

    async def verify_network_connectivity(self, max_wait_s: float = 10.0) -> bool:
        """Verify that network is connected."""

        def _check() -> bool:
            status = self.adapter.read_network_status()
            return status.get("connected", False)

        return await self._poll_condition(
            condition=_check,
            expected=True,
            max_wait_s=max_wait_s,
            interval_s=1.0,
            description="network_connectivity",
        )

    async def verify_sensor_matches(
        self,
        sensor_fn: Callable[[], Any],
        expected: Any,
        comparator: Callable[[Any, Any], bool] | None = None,
        max_wait_s: float = 5.0,
        interval_s: float = 0.5,
        description: str = "sensor_match",
    ) -> bool:
        """Generic sensor verification.

        Args:
            sensor_fn: Callable that reads the sensor value.
            expected: The expected value.
            comparator: Custom comparison function. Defaults to equality.
            max_wait_s: Maximum wait time.
            interval_s: Polling interval.
            description: Human-readable description for logging.
        """
        cmp = comparator or (lambda a, b: a == b)

        def _check() -> bool:
            actual = sensor_fn()
            return cmp(actual, expected)

        return await self._poll_condition(
            condition=_check,
            expected=True,
            max_wait_s=max_wait_s,
            interval_s=interval_s,
            description=description,
        )

    async def _poll_condition(
        self,
        condition: Callable[[], bool],
        expected: bool,
        max_wait_s: float,
        interval_s: float,
        description: str = "condition",
    ) -> bool:
        """Generic non-blocking polling loop with telemetry."""
        self._verifications_run += 1
        attempts = max(1, int(max_wait_s / interval_s))

        for i in range(attempts):
            try:
                if condition() == expected:
                    self._verifications_passed += 1
                    logger.info(
                        "Physical verification [%s] succeeded after %d attempt(s).",
                        description,
                        i + 1,
                    )
                    return True
            except Exception as e:
                logger.debug("Verification check error for [%s]: %s", description, e)

            if i < attempts - 1:  # Don't sleep after last attempt
                await asyncio.sleep(interval_s)

        self._verifications_failed += 1
        logger.warning(
            "Physical verification [%s] FAILED. Condition not met after %.1fs (%d attempts)",
            description,
            max_wait_s,
            attempts,
        )
        return False

    def stats(self) -> dict[str, int]:
        """Verification telemetry."""
        return {
            "total": self._verifications_run,
            "passed": self._verifications_passed,
            "failed": self._verifications_failed,
        }
