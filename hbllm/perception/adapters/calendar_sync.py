"""Calendar Sync adapter for RealityEventBus.

This adapter monitors scheduled events (meetings, reminders) and
emits start/end events into the perception layer. This provides
the system with temporal awareness of the user's obligations.

Supports multiple calendar providers:
- Google Calendar (via Google Calendar API)
- Microsoft Outlook (via Microsoft Graph API)
- iCal/WebCal feeds (via icalendar library)
- Mock provider (for testing/fallback)
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from hbllm.perception.reality_bus import (
    EventOrigin,
    PerceptionEvent,
    PerceptionModality,
    RealityEventBus,
)

logger = logging.getLogger(__name__)


@dataclass
class CalendarEvent:
    """A calendar event from any provider."""

    event_id: str
    title: str
    start_time: datetime
    end_time: datetime
    attendees: list[str]
    location: str = ""
    description: str = ""
    provider: str = "unknown"


class CalendarProvider(ABC):
    """Abstract base class for calendar providers."""

    @abstractmethod
    async def get_upcoming_events(self, lookahead_hours: int = 24) -> list[CalendarEvent]:
        """Get upcoming calendar events."""
        pass

    @abstractmethod
    async def get_events_in_range(self, start: datetime, end: datetime) -> list[CalendarEvent]:
        """Get events within a time range."""
        pass


class MockCalendarProvider(CalendarProvider):
    """Mock calendar provider for testing and fallback."""

    def __init__(self) -> None:
        self._events: list[CalendarEvent] = []

    async def get_upcoming_events(self, lookahead_hours: int = 24) -> list[CalendarEvent]:
        """Return mock upcoming events."""
        now = datetime.now()
        return [
            CalendarEvent(
                event_id="mock_1",
                title="Weekly Sync",
                start_time=now + timedelta(hours=1),
                end_time=now + timedelta(hours=2),
                attendees=["alice@example.com"],
                location="Conference Room A",
                provider="mock",
            ),
            CalendarEvent(
                event_id="mock_2",
                title="Team Standup",
                start_time=now + timedelta(hours=4),
                end_time=now + timedelta(hours=4, minutes=30),
                attendees=["team@example.com"],
                location="Virtual",
                provider="mock",
            ),
        ]

    async def get_events_in_range(self, start: datetime, end: datetime) -> list[CalendarEvent]:
        """Return mock events in range."""
        return await self.get_upcoming_events()


class GoogleCalendarProvider(CalendarProvider):
    """Google Calendar API provider."""

    def __init__(self, credentials_path: str | None = None, calendar_id: str = "primary") -> None:
        self.credentials_path = credentials_path
        self.calendar_id = calendar_id
        self._service: Any = None

    async def _initialize(self) -> None:
        """Initialize Google Calendar service."""
        try:
            from google.oauth2.credentials import Credentials  # type: ignore[import-not-found]
            from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore[import-not-found]
            from googleapiclient.discovery import build  # type: ignore[import-not-found]

            SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

            if self.credentials_path:
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
            else:
                # Use default credentials
                creds = Credentials.from_authorized_user_file("token.json", SCOPES)

            self._service = build("calendar", "v3", credentials=creds)
            logger.info("Google Calendar provider initialized")

        except ImportError:
            logger.warning(
                "google-api-python-client not installed. Install with: pip install google-api-python-client google-auth-oauthlib"
            )
            raise
        except Exception as e:
            logger.error("Failed to initialize Google Calendar provider: %s", e)
            raise

    async def get_upcoming_events(self, lookahead_hours: int = 24) -> list[CalendarEvent]:
        """Get upcoming events from Google Calendar."""
        if not self._service:
            await self._initialize()

        try:
            now = datetime.utcnow().isoformat() + "Z"
            time_max = (datetime.utcnow() + timedelta(hours=lookahead_hours)).isoformat() + "Z"

            events_result = (
                self._service.events()
                .list(
                    calendarId=self.calendar_id,
                    timeMin=now,
                    timeMax=time_max,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )

            events = events_result.get("items", [])
            return [self._parse_event(e) for e in events]

        except Exception as e:
            logger.error("Failed to fetch Google Calendar events: %s", e)
            return []

    async def get_events_in_range(self, start: datetime, end: datetime) -> list[CalendarEvent]:
        """Get events from Google Calendar within range."""
        if not self._service:
            await self._initialize()

        try:
            events_result = (
                self._service.events()
                .list(
                    calendarId=self.calendar_id,
                    timeMin=start.isoformat() + "Z",
                    timeMax=end.isoformat() + "Z",
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )

            events = events_result.get("items", [])
            return [self._parse_event(e) for e in events]

        except Exception as e:
            logger.error("Failed to fetch Google Calendar events in range: %s", e)
            return []

    def _parse_event(self, event: dict[str, Any]) -> CalendarEvent:
        """Parse Google Calendar event to CalendarEvent."""
        start = event.get("start", {}).get("dateTime", event.get("start", {}).get("date"))
        end = event.get("end", {}).get("dateTime", event.get("end", {}).get("date"))

        # Parse ISO format dates
        if start:
            start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        else:
            start_dt = datetime.now()

        if end:
            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        else:
            end_dt = start_dt + timedelta(hours=1)

        attendees = []
        for attendee in event.get("attendees", []):
            email = attendee.get("email", "")
            if email:
                attendees.append(email)

        return CalendarEvent(
            event_id=event.get("id", ""),
            title=event.get("summary", "Untitled"),
            start_time=start_dt,
            end_time=end_dt,
            attendees=attendees,
            location=event.get("location", ""),
            description=event.get("description", ""),
            provider="google",
        )


class ICalProvider(CalendarProvider):
    """iCal/WebCal feed provider."""

    def __init__(self, url: str) -> None:
        self.url = url

    async def get_upcoming_events(self, lookahead_hours: int = 24) -> list[CalendarEvent]:
        """Get upcoming events from iCal feed."""
        try:
            import httpx
            from icalendar import Calendar  # type: ignore[import-not-found]

            async with httpx.AsyncClient() as client:
                response = await client.get(self.url)
                response.raise_for_status()

            cal = Calendar.from_ical(response.text)
            events = []

            now = datetime.now()
            end_time = now + timedelta(hours=lookahead_hours)

            for component in cal.walk():
                if component.name == "VEVENT":
                    event = self._parse_ical_event(component)
                    if event and now <= event.start_time <= end_time:
                        events.append(event)

            return events

        except ImportError:
            logger.warning("icalendar not installed. Install with: pip install icalendar")
            return []
        except Exception as e:
            logger.error("Failed to fetch iCal events: %s", e)
            return []

    async def get_events_in_range(self, start: datetime, end: datetime) -> list[CalendarEvent]:
        """Get events from iCal feed within range."""
        try:
            import httpx
            from icalendar import Calendar  # type: ignore[import-not-found]

            async with httpx.AsyncClient() as client:
                response = await client.get(self.url)
                response.raise_for_status()

            cal = Calendar.from_ical(response.text)
            events = []

            for component in cal.walk():
                if component.name == "VEVENT":
                    event = self._parse_ical_event(component)
                    if event and start <= event.start_time <= end:
                        events.append(event)

            return events

        except ImportError:
            logger.warning("icalendar not installed. Install with: pip install icalendar")
            return []
        except Exception as e:
            logger.error("Failed to fetch iCal events in range: %s", e)
            return []

    def _parse_ical_event(self, component: Any) -> CalendarEvent | None:
        """Parse iCal event component to CalendarEvent."""
        try:
            start = component.get("dtstart").dt
            end = component.get("dtend").dt

            if isinstance(start, datetime) and isinstance(end, datetime):
                return CalendarEvent(
                    event_id=str(component.get("uid", "")),
                    title=str(component.get("summary", "Untitled")),
                    start_time=start,
                    end_time=end,
                    attendees=[],
                    location=str(component.get("location", "")),
                    description=str(component.get("description", "")),
                    provider="ical",
                )
        except Exception as e:
            logger.debug("Failed to parse iCal event: %s", e)

        return None


class CalendarSync:
    """Calendar sync adapter with pluggable providers."""

    def __init__(
        self,
        bus: RealityEventBus,
        user_id: str = "local_user",
        provider: CalendarProvider | None = None,
        poll_interval_seconds: int = 120,
    ) -> None:
        self.bus = bus
        self.user_id = user_id
        self._provider = provider or MockCalendarProvider()
        self._poll_interval = poll_interval_seconds
        self._running = False
        self._task: asyncio.Task[Any] | None = None
        self._notified_events: set[str] = set()

    def start(self) -> None:
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._monitor_loop())
            logger.info("CalendarSync started with provider: %s", type(self._provider).__name__)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("CalendarSync stopped")

    async def _monitor_loop(self) -> None:
        """Poll calendar provider for upcoming events."""
        while self._running:
            try:
                events = await self._provider.get_upcoming_events(lookahead_hours=24)

                for event in events:
                    # Emit event if not already notified
                    if event.event_id not in self._notified_events:
                        await self._emit_event(event)
                        self._notified_events.add(event.event_id)

                # Clean up old notifications
                # We'd need to track event times to clean up properly
                # For now, just keep the set manageable
                if len(self._notified_events) > 1000:
                    self._notified_events = set(list(self._notified_events)[-500:])

                await asyncio.sleep(self._poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in CalendarSync adapter: %s", e)
                await asyncio.sleep(10.0)

    async def _emit_event(self, event: CalendarEvent) -> None:
        """Emit a calendar event to the reality bus."""
        perception_event = PerceptionEvent(
            entity_id=self.user_id,
            event_type="schedule",
            sub_type="meeting_start",
            modality=PerceptionModality.APP,
            origin=EventOrigin.EXTERNAL,
            confidence=0.9,
            source_trust=0.9,
            payload={
                "meeting_id": event.event_id,
                "title": event.title,
                "start_time": event.start_time.isoformat(),
                "end_time": event.end_time.isoformat(),
                "attendees": event.attendees,
                "location": event.location,
                "description": event.description,
                "provider": event.provider,
            },
        )

        await self.bus.ingest(perception_event)
        logger.info("Emitted calendar event: %s at %s", event.title, event.start_time)
