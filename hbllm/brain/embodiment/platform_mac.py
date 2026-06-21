"""macOS Platform Adapters — native sensor and actuator backends.

Uses macOS-specific tools:
    pmset       — battery status
    osascript   — volume, brightness, notifications, app control
    networksetup — WiFi SSID
    powermetrics — thermal data (requires sudo, best-effort)

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
    """Read battery status via pmset.

    Returns:
        Dict with keys: level (0.0-1.0), charging (bool), ac_powered (bool),
        time_remaining_minutes (int | None). None if no battery.
    """
    try:
        result = subprocess.run(
            ["pmset", "-g", "batt"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        output = result.stdout
        # Parse: "Now drawing from 'Battery Power'" or "'AC Power'"
        ac_powered = "'AC Power'" in output

        # Parse: "InternalBattery-0 (id=...)  85%; charging; 1:30 remaining"
        for line in output.splitlines():
            if "InternalBattery" in line or "%" in line:
                parts = line.strip()
                # Extract percentage
                pct_start = parts.find("\t")
                if pct_start == -1:
                    pct_start = 0
                pct_str = parts[pct_start:]

                level = None
                for segment in pct_str.split(";"):
                    segment = segment.strip()
                    if "%" in segment:
                        try:
                            level = int(segment.replace("%", "").strip()) / 100.0
                        except ValueError:
                            pass

                charging = "charging" in pct_str.lower() and "not charging" not in pct_str.lower()

                time_remaining = None
                for segment in pct_str.split(";"):
                    segment = segment.strip()
                    if "remaining" in segment.lower():
                        # "1:30 remaining"
                        time_part = segment.split()[0] if segment.split() else ""
                        if ":" in time_part:
                            try:
                                h, m = time_part.split(":")
                                time_remaining = int(h) * 60 + int(m)
                            except ValueError:
                                pass

                if level is not None:
                    return {
                        "level": level,
                        "charging": charging,
                        "ac_powered": ac_powered,
                        "time_remaining_minutes": time_remaining,
                    }

        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.debug("macOS battery read failed: %s", e)
        return None


def read_volume() -> float | None:
    """Read current system output volume (0.0-1.0) via osascript."""
    try:
        result = subprocess.run(
            ["osascript", "-e", "output volume of (get volume settings)"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) / 100.0
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, OSError) as e:
        logger.debug("macOS volume read failed: %s", e)
    return None


def read_brightness() -> float | None:
    """Read display brightness (0.0-1.0) via osascript/CoreBrightness.

    Falls back to AppleScript if brightness CLI is unavailable.
    """
    try:
        # Try brightness command first
        result = subprocess.run(
            ["brightness", "-l"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "brightness" in line.lower():
                    parts = line.split()
                    for part in parts:
                        try:
                            val = float(part)
                            if 0.0 <= val <= 1.0:
                                return val
                        except ValueError:
                            continue
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Fallback: AppleScript (requires accessibility permissions)
    try:
        result = subprocess.run(
            [
                "osascript",
                "-e",
                'tell application "System Events" to get value of slider 1 '
                'of group 1 of group 2 of window "Control Centre" of '
                'application process "ControlCentre"',
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return float(result.stdout.strip()) / 100.0
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, OSError):
        pass

    return None


def read_wifi_ssid() -> str | None:
    """Read current WiFi SSID via networksetup."""
    try:
        # macOS 14+ uses "Wi-Fi", older uses the device name
        for device in ["Wi-Fi", "en0"]:
            result = subprocess.run(
                ["networksetup", "-getairportnetwork", device],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and "Current Wi-Fi Network" in result.stdout:
                # "Current Wi-Fi Network: MyNetwork"
                parts = result.stdout.split(":", 1)
                if len(parts) == 2:
                    ssid = parts[1].strip()
                    if ssid and ssid != "You are not associated with an AirPort network.":
                        return ssid
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.debug("macOS WiFi SSID read failed: %s", e)
    return None


def read_thermal_pressure() -> str | None:
    """Read thermal pressure level.

    Returns: "nominal", "moderate", "heavy", "critical", or None.
    Uses the macOS `thermal` command if available.
    """
    try:
        result = subprocess.run(
            ["pmset", "-g", "therm"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            output = result.stdout.lower()
            if "nominal" in output:
                return "nominal"
            elif "moderate" in output or "fair" in output:
                return "moderate"
            elif "heavy" in output or "serious" in output:
                return "heavy"
            elif "critical" in output:
                return "critical"
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


# ── Actuators ────────────────────────────────────────────────────────────────


async def set_volume(level: float) -> bool:
    """Set system output volume (0.0-1.0) via osascript."""
    vol_int = max(0, min(100, int(level * 100)))
    try:
        proc = await asyncio.create_subprocess_exec(
            "osascript",
            "-e",
            f"set volume output volume {vol_int}",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=5)
        return proc.returncode == 0
    except (asyncio.TimeoutError, FileNotFoundError, OSError) as e:
        logger.warning("macOS set volume failed: %s", e)
        return False


async def set_mute(muted: bool) -> bool:
    """Mute or unmute system audio via osascript."""
    try:
        cmd = "set volume with output muted" if muted else "set volume without output muted"
        proc = await asyncio.create_subprocess_exec(
            "osascript",
            "-e",
            cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=5)
        return proc.returncode == 0
    except (asyncio.TimeoutError, FileNotFoundError, OSError) as e:
        logger.warning("macOS set mute failed: %s", e)
        return False


async def send_notification(title: str, body: str, sound: bool = True) -> bool:
    """Send a native macOS notification via osascript."""
    sound_clause = 'sound name "Glass"' if sound else ""
    script = f'display notification "{body}" with title "{title}" {sound_clause}'
    try:
        proc = await asyncio.create_subprocess_exec(
            "osascript",
            "-e",
            script,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=5)
        return proc.returncode == 0
    except (asyncio.TimeoutError, FileNotFoundError, OSError) as e:
        logger.warning("macOS notification failed: %s", e)
        return False


async def open_application(app_name: str) -> bool:
    """Open an application by name via the 'open' command."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "open",
            "-a",
            app_name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.wait(), timeout=10)
        return proc.returncode == 0
    except (asyncio.TimeoutError, FileNotFoundError, OSError) as e:
        logger.warning("macOS open application failed: %s", e)
        return False


async def set_brightness(level: float) -> bool:
    """Set display brightness (0.0-1.0) via brightness CLI."""
    clamped = max(0.0, min(1.0, level))
    try:
        proc = await asyncio.create_subprocess_exec(
            "brightness",
            str(clamped),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=5)
        return proc.returncode == 0
    except (asyncio.TimeoutError, FileNotFoundError, OSError):
        # Fallback: AppleScript (less reliable)
        return False
