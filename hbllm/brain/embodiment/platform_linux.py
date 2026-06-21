"""Linux Platform Adapters — native sensor and actuator backends.

Uses Linux-specific tools:
    upower       — battery status
    amixer/pactl — volume control
    xrandr       — brightness control
    nmcli        — WiFi SSID
    notify-send  — desktop notifications

All functions are designed to fail gracefully, returning None or
default values when platform tools are unavailable.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


# ── Sensors ──────────────────────────────────────────────────────────────────


def read_battery_level() -> dict[str, Any] | None:
    """Read battery status via upower.

    Returns:
        Dict with keys: level (0.0-1.0), charging (bool), ac_powered (bool),
        time_remaining_minutes (int | None). None if no battery.
    """
    try:
        # Find battery device
        result = subprocess.run(
            ["upower", "-e"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        battery_path = None
        for line in result.stdout.splitlines():
            if "battery" in line.lower() and "BAT" in line:
                battery_path = line.strip()
                break

        if not battery_path:
            return None

        # Get battery details
        result = subprocess.run(
            ["upower", "-i", battery_path],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        info: dict[str, Any] = {
            "level": None,
            "charging": False,
            "ac_powered": False,
            "time_remaining_minutes": None,
        }

        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("percentage:"):
                try:
                    pct = line.split(":")[-1].strip().rstrip("%")
                    info["level"] = float(pct) / 100.0
                except ValueError:
                    pass
            elif line.startswith("state:"):
                state = line.split(":")[-1].strip().lower()
                info["charging"] = state in ("charging", "fully-charged")
                info["ac_powered"] = state in ("charging", "fully-charged")
            elif "time to empty" in line.lower():
                # "time to empty: 3.5 hours"
                try:
                    val_str = line.split(":")[-1].strip()
                    parts = val_str.split()
                    if len(parts) >= 2:
                        val = float(parts[0])
                        unit = parts[1].lower()
                        if "hour" in unit:
                            info["time_remaining_minutes"] = int(val * 60)
                        elif "minute" in unit:
                            info["time_remaining_minutes"] = int(val)
                except (ValueError, IndexError):
                    pass

        return info if info["level"] is not None else None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.debug("Linux battery read failed: %s", e)
        return None


def read_volume() -> float | None:
    """Read current system output volume (0.0-1.0) via pactl or amixer."""
    # Try pactl first (PulseAudio/PipeWire)
    try:
        result = subprocess.run(
            ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # "Volume: front-left: 32768 /  50% / -18.06 dB, ..."
            for part in result.stdout.split("/"):
                part = part.strip()
                if "%" in part:
                    try:
                        return int(part.rstrip("%").strip()) / 100.0
                    except ValueError:
                        pass
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Fallback: amixer (ALSA)
    try:
        result = subprocess.run(
            ["amixer", "get", "Master"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "%" in line:
                    start = line.find("[")
                    end = line.find("%")
                    if start != -1 and end != -1:
                        try:
                            return int(line[start + 1 : end]) / 100.0
                        except ValueError:
                            pass
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return None


def read_brightness() -> float | None:
    """Read display brightness (0.0-1.0) via sysfs or xrandr."""
    # Try sysfs backlight first (most reliable on laptops)
    import pathlib

    backlight_dir = pathlib.Path("/sys/class/backlight")
    if backlight_dir.exists():
        for device in backlight_dir.iterdir():
            try:
                current = int((device / "brightness").read_text().strip())
                maximum = int((device / "max_brightness").read_text().strip())
                if maximum > 0:
                    return current / maximum
            except (ValueError, OSError):
                continue

    # Fallback: xrandr
    try:
        result = subprocess.run(
            ["xrandr", "--verbose"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "Brightness:" in line:
                    try:
                        return float(line.split(":")[-1].strip())
                    except ValueError:
                        pass
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return None


def read_wifi_ssid() -> str | None:
    """Read current WiFi SSID via nmcli or iwgetid."""
    # Try nmcli first
    try:
        result = subprocess.run(
            ["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith("yes:"):
                    ssid = line.split(":", 1)[1].strip()
                    if ssid:
                        return ssid
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Fallback: iwgetid
    try:
        result = subprocess.run(
            ["iwgetid", "-r"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            ssid = result.stdout.strip()
            if ssid:
                return ssid
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return None


def read_thermal_pressure() -> str | None:
    """Read thermal pressure from sysfs thermal zones.

    Returns: "nominal", "moderate", "heavy", "critical", or None.
    """
    import pathlib

    thermal_dir = pathlib.Path("/sys/class/thermal")
    if not thermal_dir.exists():
        return None

    max_temp = 0
    for zone in thermal_dir.glob("thermal_zone*"):
        try:
            temp_str = (zone / "temp").read_text().strip()
            temp_mc = int(temp_str)  # millidegrees Celsius
            temp_c = temp_mc / 1000.0
            max_temp = max(max_temp, temp_c)
        except (ValueError, OSError):
            continue

    if max_temp == 0:
        return None
    elif max_temp < 60:
        return "nominal"
    elif max_temp < 75:
        return "moderate"
    elif max_temp < 90:
        return "heavy"
    else:
        return "critical"


# ── Actuators ────────────────────────────────────────────────────────────────


async def set_volume(level: float) -> bool:
    """Set system output volume (0.0-1.0) via pactl or amixer."""
    vol_pct = max(0, min(100, int(level * 100)))

    # Try pactl first
    try:
        proc = await asyncio.create_subprocess_exec(
            "pactl",
            "set-sink-volume",
            "@DEFAULT_SINK@",
            f"{vol_pct}%",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=5)
        if proc.returncode == 0:
            return True
    except (asyncio.TimeoutError, FileNotFoundError, OSError):
        pass

    # Fallback: amixer
    try:
        proc = await asyncio.create_subprocess_exec(
            "amixer",
            "set",
            "Master",
            f"{vol_pct}%",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=5)
        return proc.returncode == 0
    except (asyncio.TimeoutError, FileNotFoundError, OSError) as e:
        logger.warning("Linux set volume failed: %s", e)
        return False


async def set_mute(muted: bool) -> bool:
    """Mute or unmute system audio."""
    state = "1" if muted else "0"
    try:
        proc = await asyncio.create_subprocess_exec(
            "pactl",
            "set-sink-mute",
            "@DEFAULT_SINK@",
            state,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=5)
        if proc.returncode == 0:
            return True
    except (asyncio.TimeoutError, FileNotFoundError, OSError):
        pass

    # Fallback: amixer
    toggle = "mute" if muted else "unmute"
    try:
        proc = await asyncio.create_subprocess_exec(
            "amixer",
            "set",
            "Master",
            toggle,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=5)
        return proc.returncode == 0
    except (asyncio.TimeoutError, FileNotFoundError, OSError) as e:
        logger.warning("Linux set mute failed: %s", e)
        return False


async def send_notification(title: str, body: str, sound: bool = True) -> bool:
    """Send a desktop notification via notify-send."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "notify-send",
            title,
            body,
            "--urgency=normal",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=5)
        return proc.returncode == 0
    except (asyncio.TimeoutError, FileNotFoundError, OSError) as e:
        logger.warning("Linux notification failed: %s", e)
        return False


async def open_application(app_name: str) -> bool:
    """Open an application via xdg-open or by name."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "xdg-open",
            app_name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=10)
        return proc.returncode == 0
    except (asyncio.TimeoutError, FileNotFoundError, OSError) as e:
        logger.warning("Linux open application failed: %s", e)
        return False


async def set_brightness(level: float) -> bool:
    """Set display brightness (0.0-1.0) via xrandr or sysfs."""
    import pathlib

    clamped = max(0.0, min(1.0, level))

    # Try sysfs first (most direct)
    backlight_dir = pathlib.Path("/sys/class/backlight")
    if backlight_dir.exists():
        for device in backlight_dir.iterdir():
            try:
                maximum = int((device / "max_brightness").read_text().strip())
                target = int(clamped * maximum)
                (device / "brightness").write_text(str(target))
                return True
            except (ValueError, OSError, PermissionError):
                continue

    # Fallback: xrandr
    try:
        proc = await asyncio.create_subprocess_exec(
            "xrandr",
            "--output",
            "eDP-1",
            "--brightness",
            str(clamped),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=5)
        return proc.returncode == 0
    except (asyncio.TimeoutError, FileNotFoundError, OSError) as e:
        logger.warning("Linux set brightness failed: %s", e)
        return False
