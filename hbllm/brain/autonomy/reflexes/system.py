"""System Health Reflexes — device-level monitoring rules.

7 reflexes for monitoring the host device:
    1. battery_critical   — battery < 10%
    2. battery_low        — battery < 20%
    3. memory_pressure    — RAM > 90%
    4. disk_full          — free disk < 2 GB
    5. cpu_sustained_high — CPU > 90% sustained
    6. network_down       — connectivity lost
    7. thermal_throttling — thermal pressure critical

All reflexes are deterministic (Tier 1) — zero LLM cost.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from hbllm.brain.autonomy.attention import AttentionEvent
from hbllm.brain.autonomy.reflexes import make_push_message
from hbllm.network.messages import Message

logger = logging.getLogger(__name__)

ReflexRule = Callable[[AttentionEvent], Message | None]


def _battery_critical(event: AttentionEvent) -> Message | None:
    """Battery below 10% — critical alert."""
    if event.source in ("system.hardware.critical", "perception.system.health_alert"):
        battery = event.payload.get("battery_level") or event.payload.get("level")
        if battery is not None and battery < 0.10:
            return make_push_message(
                title="🔋 Battery Critical",
                body=f"Battery at {battery:.0%}. Connect charger immediately.",
                priority="critical",
            )
    return None


def _battery_low(event: AttentionEvent) -> Message | None:
    """Battery below 20% — high alert."""
    if event.source in ("system.hardware.critical", "perception.system.health_alert"):
        battery = event.payload.get("battery_level") or event.payload.get("level")
        if battery is not None and 0.10 <= battery < 0.20:
            return make_push_message(
                title="🔋 Battery Low",
                body=f"Battery at {battery:.0%}. Consider plugging in.",
                priority="high",
            )
    return None


def _memory_pressure(event: AttentionEvent) -> Message | None:
    """RAM usage above 90%."""
    if event.source in ("system.hardware.critical", "perception.system.health_alert"):
        ram = event.payload.get("ram_percent", 0)
        if isinstance(ram, (int, float)) and ram > 90:
            return make_push_message(
                title="💾 Memory Pressure",
                body=f"System RAM at {ram:.0f}%. Consider closing applications.",
                priority="high",
            )
    return None


def _disk_full(event: AttentionEvent) -> Message | None:
    """Free disk space below 2 GB."""
    if event.source in ("perception.system.health_alert",):
        alert_type = event.payload.get("type")
        if alert_type == "disk_low":
            free_gb = event.payload.get("free_gb", 999)
            if free_gb < 2.0:
                return make_push_message(
                    title="💿 Disk Almost Full",
                    body=f"Only {free_gb:.1f} GB free. Free up space to avoid issues.",
                    priority="critical" if free_gb < 1.0 else "high",
                )
    return None


def _cpu_sustained_high(event: AttentionEvent) -> Message | None:
    """CPU load sustained above 90%."""
    if event.source in ("perception.system.health_alert",):
        alert_type = event.payload.get("type")
        if alert_type == "cpu_high":
            load = event.payload.get("load_1min", 0)
            return make_push_message(
                title="🖥️ CPU Load High",
                body=f"CPU load average: {load:.1f}. System may be sluggish.",
                priority="high",
            )
    return None


def _network_down(event: AttentionEvent) -> Message | None:
    """Network connectivity lost."""
    if event.source == "system.network":
        connected = event.payload.get("connected", True)
        if not connected:
            return make_push_message(
                title="📡 Network Disconnected",
                body="Internet connectivity lost. Operating in offline mode.",
                priority="high",
            )
    return None


def _thermal_throttling(event: AttentionEvent) -> Message | None:
    """Thermal pressure is critical — device may throttle."""
    if event.source in ("perception.system.health_alert", "system.hardware.critical"):
        thermal = event.payload.get("thermal_status") or event.payload.get("thermal")
        if thermal in ("heavy", "critical"):
            return make_push_message(
                title="🌡️ Thermal Warning",
                body=f"Device thermal pressure is {thermal}. Performance may be reduced.",
                priority="critical" if thermal == "critical" else "high",
            )
    return None


def get_system_reflexes() -> dict[str, ReflexRule]:
    """Return all system health reflexes."""
    return {
        "battery_critical": _battery_critical,
        "battery_low": _battery_low,
        "memory_pressure": _memory_pressure,
        "disk_full": _disk_full,
        "cpu_sustained_high": _cpu_sustained_high,
        "network_down": _network_down,
        "thermal_throttling": _thermal_throttling,
    }
