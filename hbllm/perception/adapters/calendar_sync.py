"""Calendar Sync adapter for RealityEventBus.

This adapter monitors scheduled events (meetings, reminders) and
emits start/end events into the perception layer. This provides
the system with temporal awareness of the user's obligations.
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


class CalendarSync:
    """Mock adapter simulating calendar meeting events."""

    def __init__(self, bus: RealityEventBus, user_id: str = "local_user") -> None:
        self.bus = bus
        self.user_id = user_id
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
        """Simulate polling a calendar API (e.g., Google Calendar)."""
        while self._running:
            try:
                # In a real implementation, this would poll the Google Calendar API
                # or listen to webhooks for impending meetings.
                await asyncio.sleep(120.0)  # Check every 2 minutes

                event = PerceptionEvent(
                    entity_id=self.user_id,
                    event_type="schedule",
                    sub_type="meeting_start",
                    modality=PerceptionModality.APP,  # Modality tier: APP
                    origin=EventOrigin.EXTERNAL,
                    confidence=0.9,
                    source_trust=0.9,
                    payload={
                        "meeting_id": "meet_123",
                        "title": "Weekly Sync",
                        "attendees": ["alice@example.com"],
                    },
                )

                await self.bus.ingest(event)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in CalendarSync adapter: %s", e)
                await asyncio.sleep(10.0)
