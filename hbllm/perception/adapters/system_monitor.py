"""System Activity Monitor adapter for RealityEventBus.

This is a high-signal, low-noise adapter that emits events for:
- Active window tracking (app switching)
- Idle detection
- System sleep/wake states
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from hbllm.perception.reality_bus import (
    EventOrigin,
    PerceptionEvent,
    PerceptionModality,
    RealityEventBus,
)

logger = logging.getLogger(__name__)


class SystemActivityMonitor:
    """Mock adapter simulating OS-level activity events."""

    def __init__(self, bus: RealityEventBus, device_id: str = "local_device") -> None:
        self.bus = bus
        self.device_id = device_id
        self._running = False
        self._task: asyncio.Task[Any] | None = None

    def start(self) -> None:
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self) -> None:
        """Simulate polling the OS for active window and idle state."""
        # For a real implementation, this would use pyobjc (macOS) or X11/Wayland
        # APIs to get the current frontmost application and user idle time.

        while self._running:
            try:
                # In a real scenario, we would only emit when the state *changes*.
                # For now, we simulate an occasional "app_switch" event.
                await asyncio.sleep(60.0)  # Check every minute

                event = PerceptionEvent(
                    entity_id=self.device_id,
                    event_type="os_activity",
                    sub_type="app_switch",
                    modality=PerceptionModality.SYSTEM,
                    origin=EventOrigin.SYSTEM,
                    confidence=1.0,  # OS APIs are highly reliable
                    source_trust=1.0,
                    payload={"active_app": "VSCode", "idle_time_s": 0},
                )

                await self.bus.ingest(event)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in SystemActivityMonitor: %s", e)
                await asyncio.sleep(5.0)
