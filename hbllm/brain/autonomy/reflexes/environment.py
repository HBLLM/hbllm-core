"""Environment Reflexes — IoT and smart home awareness rules.

8 reflexes for physical environment monitoring:
    1. temperature_alert       — thermostat outside comfort range
    2. humidity_alert          — abnormal humidity levels
    3. motion_at_night         — motion detected during sleep hours
    4. door_left_open          — door sensor open > 5 minutes
    5. window_open_rain        — window open + rain detected
    6. lights_left_on          — lights on + no motion for 30 min
    7. appliance_energy_spike  — unusual power consumption
    8. smoke_co_detected       — smoke/CO alarm (from ambient audio or sensor)

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


def _temperature_alert(event: AttentionEvent) -> Message | None:
    """Temperature outside comfort range (18-28°C)."""
    if event.source.startswith("iot.") or event.source.startswith("perception."):
        temp = event.payload.get("temperature")
        if temp is not None and isinstance(temp, (int, float)):
            room = event.payload.get("room", "unknown")
            if temp < 16:
                return make_push_message(
                    title="🥶 Temperature Low",
                    body=f"{room} is at {temp:.1f}°C. Consider turning on heating.",
                    priority="high",
                )
            elif temp > 30:
                return make_push_message(
                    title="🔥 Temperature High",
                    body=f"{room} is at {temp:.1f}°C. Consider turning on cooling.",
                    priority="high",
                )
    return None


def _humidity_alert(event: AttentionEvent) -> Message | None:
    """Humidity outside healthy range (30-60%)."""
    if event.source.startswith("iot.") or event.source.startswith("perception."):
        humidity = event.payload.get("humidity")
        if humidity is not None and isinstance(humidity, (int, float)):
            room = event.payload.get("room", "unknown")
            if humidity < 25:
                return make_push_message(
                    title="💧 Low Humidity",
                    body=f"{room} humidity at {humidity:.0f}%. Consider using a humidifier.",
                    priority="info",
                )
            elif humidity > 70:
                return make_push_message(
                    title="💧 High Humidity",
                    body=f"{room} humidity at {humidity:.0f}%. Risk of mold — improve ventilation.",
                    priority="high",
                )
    return None


def _motion_at_night(event: AttentionEvent) -> Message | None:
    """Motion detected during nighttime hours (23:00-05:00)."""
    if event.source.startswith("iot.") and event.payload.get("motion_detected"):
        import datetime

        hour = datetime.datetime.now().hour
        if hour >= 23 or hour < 5:
            room = event.payload.get("room", "unknown")
            return make_push_message(
                title="🌙 Night Motion",
                body=f"Motion detected in {room} at {hour:02d}:{datetime.datetime.now().minute:02d}.",
                priority="high",
                category="security",
            )
    return None


def _door_left_open(event: AttentionEvent) -> Message | None:
    """Door sensor open for too long."""
    if event.source.startswith("iot."):
        door_state = event.payload.get("door_state") or event.payload.get("contact")
        duration_s = event.payload.get("open_duration_s", 0)
        if door_state == "open" and duration_s > 300:  # 5 minutes
            door_name = event.payload.get("device_name", "A door")
            mins = duration_s // 60
            return make_push_message(
                title="🚪 Door Left Open",
                body=f"{door_name} has been open for {mins} minutes.",
                priority="info",
            )
    return None


def _window_open_rain(event: AttentionEvent) -> Message | None:
    """Window open while raining."""
    if event.source.startswith("perception.weather") or event.source.startswith("iot."):
        raining = event.payload.get("raining", False)
        window_open = event.payload.get("window_open", False)
        if raining and window_open:
            room = event.payload.get("room", "unknown")
            return make_push_message(
                title="🌧️ Close Window",
                body=f"It's raining and a window in {room} is open.",
                priority="high",
            )
    return None


def _lights_left_on(event: AttentionEvent) -> Message | None:
    """Lights on with no motion for extended period."""
    if event.source.startswith("iot."):
        lights_on = event.payload.get("lights_on", False)
        no_motion_s = event.payload.get("no_motion_duration_s", 0)
        if lights_on and no_motion_s > 1800:  # 30 minutes
            room = event.payload.get("room", "unknown")
            mins = no_motion_s // 60
            return make_push_message(
                title="💡 Lights Still On",
                body=f"Lights in {room} have been on with no motion for {mins} minutes.",
                priority="suggestion",
            )
    return None


def _appliance_energy_spike(event: AttentionEvent) -> Message | None:
    """Unusual power consumption from smart plug."""
    if event.source.startswith("iot."):
        power_w = event.payload.get("power_watts") or event.payload.get("power")
        baseline_w = event.payload.get("baseline_watts", 0)
        if power_w is not None and baseline_w > 0:
            if power_w > baseline_w * 2 and power_w > 100:  # 2x baseline and > 100W
                device = event.payload.get("device_name", "An appliance")
                return make_push_message(
                    title="⚡ Energy Spike",
                    body=f"{device} is drawing {power_w:.0f}W (usual: {baseline_w:.0f}W).",
                    priority="info",
                )
    return None


def _smoke_co_detected(event: AttentionEvent) -> Message | None:
    """Smoke or CO alarm detected (from sensor or ambient audio)."""
    # From IoT smoke/CO sensor
    if event.source.startswith("iot."):
        alarm = event.payload.get("smoke_detected") or event.payload.get("co_detected")
        if alarm:
            alarm_type = "Smoke" if event.payload.get("smoke_detected") else "CO"
            room = event.payload.get("room", "unknown")
            return make_push_message(
                title=f"🚨 {alarm_type} Alarm!",
                body=f"{alarm_type} detected in {room}. Check immediately!",
                priority="critical",
                category="security",
            )

    # From ambient audio classifier
    if event.source.startswith("perception.audio.ambient"):
        sound_class = event.payload.get("sound_class")
        if sound_class in ("smoke_detector", "alarm"):
            confidence = event.payload.get("confidence", 0)
            if confidence > 0.5:
                return make_push_message(
                    title="🚨 Alarm Sound Detected!",
                    body=f"Audio classifier detected {sound_class} sound (confidence: {confidence:.0%}).",
                    priority="critical",
                    category="security",
                )
    return None


def get_environment_reflexes() -> dict[str, ReflexRule]:
    """Return all environment reflexes."""
    return {
        "temperature_alert": _temperature_alert,
        "humidity_alert": _humidity_alert,
        "motion_at_night": _motion_at_night,
        "door_left_open": _door_left_open,
        "window_open_rain": _window_open_rain,
        "lights_left_on": _lights_left_on,
        "appliance_energy_spike": _appliance_energy_spike,
        "smoke_co_detected": _smoke_co_detected,
    }
