"""World State Engine — unified environmental awareness.

Aggregates all perception sources into a single coherent state snapshot:
    - System hardware (battery, CPU, RAM, thermal)
    - IoT devices (lights, locks, sensors, appliances)
    - Audio environment (ambient sounds, speech activity)
    - Temporal context (time of day, day of week, upcoming events)
    - User engagement (active/idle/deep_idle)

The WorldState is injected into the LLM's system prompt so the AI
"knows" its physical environment without being asked.

Architecture:
    1. Subscribes to all perception topics
    2. Maintains a live state dict with TTL-based expiry
    3. Generates a concise natural language summary for prompt injection
    4. Publishes state diffs to `world.state.updated`

Usage::

    engine = WorldStateEngine(bus=message_bus)
    await engine.start()

    # Get current world state
    state = engine.get_state()
    summary = engine.get_summary()  # "Battery 85%, WiFi connected, kitchen light on..."
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from hbllm.network.messages import Message

logger = logging.getLogger(__name__)


class WorldStateEngine:
    """Maintains a live snapshot of the physical world.

    All perception events flow through here and are aggregated into
    a coherent state model that the LLM can reason about.
    """

    def __init__(
        self,
        bus: Any | None = None,
        state_ttl_s: float = 300.0,  # How long a state entry stays valid
    ) -> None:
        self.bus = bus
        self.state_ttl_s = state_ttl_s

        # Live state sections
        self._hardware: dict[str, Any] = {}
        self._iot_devices: dict[str, dict[str, Any]] = {}
        self._audio_env: dict[str, Any] = {}
        self._user_state: dict[str, Any] = {}
        self._weather: dict[str, Any] = {}
        self._calendar: dict[str, Any] = {}
        self._fused_events: list[dict[str, Any]] = []

        # Timestamps for TTL tracking
        self._timestamps: dict[str, float] = {}
        self._max_fused = 10  # Keep last 10 fused events

        # Telemetry
        self._updates_received = 0
        self._summaries_generated = 0

    async def start(self) -> None:
        """Subscribe to all perception topics."""
        if not self.bus:
            return

        # Hardware sensors
        await self.bus.subscribe("perception.system.*", self._on_hardware)
        await self.bus.subscribe("system.hardware.*", self._on_hardware)

        # IoT device events
        await self.bus.subscribe("iot.event", self._on_iot_event)
        await self.bus.subscribe("iot.discovery", self._on_iot_event)

        # Audio environment
        await self.bus.subscribe("perception.audio.ambient", self._on_audio)
        await self.bus.subscribe("perception.audio.ambient.critical", self._on_audio)

        # User engagement state
        await self.bus.subscribe("autonomy.user.state", self._on_user_state)

        # Weather
        await self.bus.subscribe("perception.weather", self._on_weather)

        # Calendar
        await self.bus.subscribe("calendar.*", self._on_calendar)
        await self.bus.subscribe("autonomy.watcher.calendar", self._on_calendar)

        # Fused temporal sequences
        await self.bus.subscribe("perception.fused.sequence", self._on_fused)

        logger.info("WorldStateEngine started — subscribing to all perception topics")

    async def _on_hardware(self, msg: Message) -> None:
        self._updates_received += 1
        self._hardware.update(msg.payload)
        self._timestamps["hardware"] = time.time()

    async def _on_iot_event(self, msg: Message) -> None:
        self._updates_received += 1
        device_id = msg.payload.get("device_id", msg.payload.get("id", "unknown"))
        self._iot_devices[device_id] = {
            **msg.payload,
            "_updated_at": time.time(),
        }
        self._timestamps["iot"] = time.time()

    async def _on_audio(self, msg: Message) -> None:
        self._updates_received += 1
        self._audio_env = {
            "sound_class": msg.payload.get("sound_class", "unknown"),
            "confidence": msg.payload.get("confidence", 0),
            "energy_db": msg.payload.get("energy_db", -60),
            "is_critical": msg.payload.get("is_critical", False),
            "_updated_at": time.time(),
        }
        self._timestamps["audio"] = time.time()

    async def _on_user_state(self, msg: Message) -> None:
        self._updates_received += 1
        self._user_state = msg.payload
        self._timestamps["user_state"] = time.time()

    async def _on_weather(self, msg: Message) -> None:
        self._updates_received += 1
        self._weather = msg.payload
        self._timestamps["weather"] = time.time()

    async def _on_calendar(self, msg: Message) -> None:
        self._updates_received += 1
        self._calendar = msg.payload
        self._timestamps["calendar"] = time.time()

    async def _on_fused(self, msg: Message) -> None:
        self._updates_received += 1
        self._fused_events.append(msg.payload)
        if len(self._fused_events) > self._max_fused:
            self._fused_events = self._fused_events[-self._max_fused :]
        self._timestamps["fused"] = time.time()

    def get_state(self) -> dict[str, Any]:
        """Get the full world state as a structured dict."""
        now = time.time()
        dt = datetime.now(timezone.utc)

        return {
            "timestamp": now,
            "datetime_utc": dt.isoformat(),
            "temporal": {
                "hour": dt.hour,
                "day_of_week": dt.strftime("%A"),
                "period": self._get_time_period(dt.hour),
            },
            "hardware": self._filter_fresh("hardware", self._hardware),
            "iot_devices": {
                k: v
                for k, v in self._iot_devices.items()
                if now - v.get("_updated_at", 0) < self.state_ttl_s
            },
            "audio_environment": self._filter_fresh("audio", self._audio_env),
            "user_engagement": self._filter_fresh("user_state", self._user_state),
            "weather": self._filter_fresh("weather", self._weather),
            "calendar": self._filter_fresh("calendar", self._calendar),
            "recent_events": self._fused_events[-5:],
        }

    def get_summary(self) -> str:
        """Generate a concise natural language summary for LLM prompt injection.

        Returns a 2-5 line summary of the current world state.
        """
        self._summaries_generated += 1
        now = time.time()
        dt = datetime.now(timezone.utc)

        lines: list[str] = []

        # Time context
        period = self._get_time_period(dt.hour)
        lines.append(f"⏰ {dt.strftime('%A')} {period} ({dt.strftime('%H:%M')} UTC)")

        # Hardware
        hw = self._hardware
        if hw and self._is_fresh("hardware"):
            parts: list[str] = []
            battery = hw.get("battery_level") or hw.get("battery")
            if battery is not None:
                pct = battery if battery > 1 else battery * 100
                emoji = "🔋" if pct > 20 else "🪫"
                parts.append(f"{emoji} Battery {pct:.0f}%")
            if hw.get("cpu_load"):
                parts.append(f"CPU {hw['cpu_load']:.0%}")
            if parts:
                lines.append(" | ".join(parts))

        # IoT devices (active only)
        active_devices: list[str] = []
        for dev_id, dev in self._iot_devices.items():
            if now - dev.get("_updated_at", 0) < self.state_ttl_s:
                state = dev.get("state", {})
                name = dev.get("name", dev_id)
                if isinstance(state, dict) and state.get("on"):
                    active_devices.append(f"{name}: on")
                elif isinstance(state, str):
                    active_devices.append(f"{name}: {state}")

        if active_devices:
            lines.append(f"🏠 {', '.join(active_devices[:5])}")

        # Audio
        if self._audio_env and self._is_fresh("audio"):
            sound = self._audio_env.get("sound_class", "")
            if sound and sound != "silence":
                conf = self._audio_env.get("confidence", 0)
                lines.append(f"🔊 Ambient: {sound} ({conf:.0%})")

        # User state
        if self._user_state and self._is_fresh("user_state"):
            user_s = self._user_state.get("state", "unknown")
            lines.append(f"👤 User: {user_s}")

        # Weather
        if self._weather and self._is_fresh("weather"):
            temp = self._weather.get("temperature")
            condition = self._weather.get("condition", "")
            if temp is not None:
                lines.append(f"🌤️ {condition} {temp}°C")

        # Recent notable events
        if self._fused_events:
            latest = self._fused_events[-1]
            age = now - latest.get("timestamp", now)
            if age < 300:  # Within last 5 minutes
                lines.append(f"📡 {latest.get('narrative', 'Event detected')}")

        return "\n".join(lines) if lines else "🌐 No environmental data available."

    def _is_fresh(self, section: str) -> bool:
        """Check if a section's data is still within TTL."""
        ts = self._timestamps.get(section, 0)
        return time.time() - ts < self.state_ttl_s

    def _filter_fresh(self, section: str, data: dict[str, Any]) -> dict[str, Any]:
        """Return data only if fresh, otherwise empty dict."""
        if self._is_fresh(section):
            return {k: v for k, v in data.items() if not k.startswith("_")}
        return {}

    def _get_time_period(self, hour: int) -> str:
        """Get human-readable time period."""
        if 5 <= hour < 9:
            return "early morning"
        elif 9 <= hour < 12:
            return "morning"
        elif 12 <= hour < 14:
            return "midday"
        elif 14 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 20:
            return "evening"
        elif 20 <= hour < 23:
            return "night"
        else:
            return "late night"

    def stats(self) -> dict[str, Any]:
        """Engine statistics."""
        return {
            "updates_received": self._updates_received,
            "summaries_generated": self._summaries_generated,
            "iot_device_count": len(self._iot_devices),
            "fused_event_count": len(self._fused_events),
            "sections_fresh": {
                section: self._is_fresh(section)
                for section in ["hardware", "iot", "audio", "user_state", "weather", "calendar"]
            },
        }
