"""External Process Supervisor for Hard Recovery.

This is the OS-level guardian of cognition. It runs in a separate thread
or process to ensure it cannot be starved if the main asyncio event loop
is completely blocked by a synchronous task (e.g., an LLM hanging).

It monitors a shared heartbeat. If the heartbeat stops, it triggers
a hard recovery sequence.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


class ProcessSupervisor:
    """External monitor to detect system hangs and enforce hard recovery."""

    def __init__(
        self, heartbeat_timeout_s: float = 10.0, recovery_callback: Callable[[], None] | None = None
    ) -> None:
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.recovery_callback = recovery_callback

        self._last_heartbeat = time.monotonic()
        self._running = False
        self._monitor_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the supervisor in a separate OS thread."""
        if self._running:
            return

        self._running = True
        self._last_heartbeat = time.monotonic()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="ProcessSupervisor"
        )
        self._monitor_thread.start()
        logger.info("ProcessSupervisor started (timeout=%.1fs)", self.heartbeat_timeout_s)

    def stop(self) -> None:
        """Stop the supervisor."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

    def heartbeat(self) -> None:
        """Update the last seen heartbeat from the main AutonomyCore."""
        self._last_heartbeat = time.monotonic()

    def _monitor_loop(self) -> None:
        """The loop running in the external thread."""
        while self._running:
            time.sleep(1.0)  # Check every second

            elapsed = time.monotonic() - self._last_heartbeat
            if elapsed > self.heartbeat_timeout_s:
                logger.critical(
                    "ProcessSupervisor: System HANG detected! No heartbeat in %.1f seconds. "
                    "Event loop is completely blocked.",
                    elapsed,
                )
                self._trigger_recovery()

                # Reset heartbeat to prevent spamming recovery while it restarts
                self._last_heartbeat = time.monotonic()

    def _trigger_recovery(self) -> None:
        """Execute the hard recovery callback."""
        if self.recovery_callback:
            try:
                self.recovery_callback()
            except Exception:
                logger.exception("ProcessSupervisor: Failed to execute hard recovery callback.")
        else:
            logger.error("ProcessSupervisor: No recovery callback configured. System remains hung.")
