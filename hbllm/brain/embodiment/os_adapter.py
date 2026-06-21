"""Device Reality Integration — Real OS Sensors and Actuators.

Provides standardized OS adapters for the cognitive mesh, bridging
the gap between AI plans and actual host OS hardware execution.

Uses:
    - ``psutil`` for cross-platform CPU, memory, disk, battery, network
    - Platform-specific backends for volume, brightness, WiFi, notifications
    - Cached sensor reads to avoid hammering the OS

The OSAdapter is the "body" of the cognitive system — it gives the brain
real sensory data about the physical device it runs on.
"""

from __future__ import annotations

import logging
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Cached Sensor Read Helper ────────────────────────────────────────────────


@dataclass
class _CachedValue:
    """A sensor reading with a time-based cache."""

    value: Any = None
    timestamp: float = 0.0
    ttl: float = 5.0  # seconds

    def is_fresh(self) -> bool:
        return self.value is not None and (time.monotonic() - self.timestamp) < self.ttl

    def set(self, value: Any) -> Any:
        self.value = value
        self.timestamp = time.monotonic()
        return value


# ── Actuator Safety ──────────────────────────────────────────────────────────


@dataclass
class ActuatorConfig:
    """Safety configuration for OS actuators."""

    # Directories where file operations are allowed
    safe_write_dirs: list[str] = field(
        default_factory=lambda: [
            os.path.expanduser("~/Documents"),
            os.path.expanduser("~/Downloads"),
            "/tmp",
        ]
    )

    # Applications allowed to be opened
    app_allowlist: list[str] = field(
        default_factory=lambda: [
            "Safari",
            "Firefox",
            "Chrome",
            "Terminal",
            "Calculator",
            "Notes",
            "TextEdit",
            "Calendar",
            "Music",
            "Finder",
        ]
    )

    # Whether file mutations are enabled
    enable_file_writes: bool = True
    enable_file_deletes: bool = False  # Extra caution — off by default

    def is_path_safe(self, filepath: str) -> bool:
        """Check if a filepath is within an allowed directory."""
        resolved = os.path.realpath(filepath)
        return any(resolved.startswith(os.path.realpath(d)) for d in self.safe_write_dirs)

    def is_app_allowed(self, app_name: str) -> bool:
        """Check if an application is on the allowlist (case-insensitive)."""
        return app_name.lower() in [a.lower() for a in self.app_allowlist]


# ── Platform Backend Loader ──────────────────────────────────────────────────


def _detect_platform() -> str:
    """Detect the current platform."""
    system = platform.system().lower()
    if system == "darwin":
        return "mac"
    elif system == "linux":
        return "linux"
    elif system == "windows":
        return "windows"
    return "unknown"


def _load_platform_module() -> Any:
    """Dynamically load the platform-specific backend."""
    plat = _detect_platform()
    if plat == "mac":
        from hbllm.brain.embodiment import platform_mac

        return platform_mac
    elif plat == "linux":
        from hbllm.brain.embodiment import platform_linux

        return platform_linux
    else:
        logger.warning("No platform backend for '%s' — actuators disabled", plat)
        return None


# ── OSAdapter ────────────────────────────────────────────────────────────────


class OSAdapter:
    """Safe wrapper for interacting with the Host OS sensors and actuators.

    Sensors (read-only, always safe):
        - Battery level, charging status
        - CPU load, memory usage, disk usage
        - Network connectivity, WiFi SSID
        - Thermal pressure, display brightness, volume
        - Active processes, system uptime

    Actuators (guarded by ActuatorConfig):
        - File create/delete (path-whitelisted)
        - Volume/brightness control
        - Application launching (allowlisted)
        - System notifications

    All sensor reads are cached with configurable TTL (default 5s) to
    avoid hammering the OS. Actuators require explicit configuration.
    """

    def __init__(
        self,
        actuator_config: ActuatorConfig | None = None,
        cache_ttl: float = 5.0,
    ) -> None:
        self.config = actuator_config or ActuatorConfig()
        self._platform = _load_platform_module()
        self._system = _detect_platform()

        # Sensor caches
        self._cache_battery = _CachedValue(ttl=cache_ttl)
        self._cache_cpu = _CachedValue(ttl=max(1.0, cache_ttl))  # CPU needs more frequent reads
        self._cache_memory = _CachedValue(ttl=cache_ttl)
        self._cache_disk: dict[str, _CachedValue] = {}
        self._cache_network = _CachedValue(ttl=cache_ttl * 2)  # Network is slow to query
        self._cache_volume = _CachedValue(ttl=cache_ttl)
        self._cache_brightness = _CachedValue(ttl=cache_ttl)
        self._cache_wifi = _CachedValue(ttl=cache_ttl * 6)  # WiFi rarely changes
        self._cache_thermal = _CachedValue(ttl=cache_ttl * 2)
        self._cache_processes = _CachedValue(ttl=cache_ttl)
        self._cache_uptime = _CachedValue(ttl=60.0)  # Uptime rarely changes

        # Try to import psutil once
        self._psutil: Any = None
        try:
            import psutil  # type: ignore[import]

            self._psutil = psutil
        except ImportError:
            logger.warning(
                "psutil not installed — sensor reads will use platform-specific fallbacks. "
                "Install with: pip install psutil"
            )

        logger.info(
            "OSAdapter initialized (platform=%s, psutil=%s, actuators=%s)",
            self._system,
            "available" if self._psutil else "unavailable",
            "enabled" if self.config.enable_file_writes else "read-only",
        )

    # ── Sensors ──────────────────────────────────────────────────────────

    def read_battery_level(self) -> float:
        """Read device battery level (0.0 to 1.0).

        Returns 1.0 if no battery is present (desktop/server).
        """
        info = self.read_battery_info()
        if info is None:
            return 1.0  # No battery = effectively "full" (plugged in desktop)
        return info.get("level", 1.0)

    def read_battery_info(self) -> dict[str, Any] | None:
        """Read detailed battery information.

        Returns:
            Dict with level, charging, ac_powered, time_remaining_minutes.
            None if no battery is present.
        """
        if self._cache_battery.is_fresh():
            return self._cache_battery.value

        # Try psutil first (cross-platform)
        if self._psutil:
            try:
                batt = self._psutil.sensors_battery()
                if batt is not None:
                    return self._cache_battery.set(
                        {
                            "level": batt.percent / 100.0,
                            "charging": batt.power_plugged is True,
                            "ac_powered": batt.power_plugged is True,
                            "time_remaining_minutes": int(batt.secsleft / 60)
                            if batt.secsleft > 0
                            else None,
                        }
                    )
                return self._cache_battery.set(None)
            except Exception as e:
                logger.debug("psutil battery read failed: %s", e)

        # Fallback: platform-specific
        if self._platform and hasattr(self._platform, "read_battery_level"):
            result = self._platform.read_battery_level()
            return self._cache_battery.set(result)

        return self._cache_battery.set(None)

    def read_cpu_load(self) -> float:
        """Read CPU utilization (0.0 to 1.0)."""
        if self._cache_cpu.is_fresh():
            return self._cache_cpu.value

        if self._psutil:
            try:
                # Non-blocking call with 0.1s interval
                pct = self._psutil.cpu_percent(interval=0.1)
                return self._cache_cpu.set(pct / 100.0)
            except Exception as e:
                logger.debug("psutil CPU read failed: %s", e)

        # Fallback: os.getloadavg (Unix only)
        try:
            load1, _, _ = os.getloadavg()
            cpu_count = os.cpu_count() or 1
            return self._cache_cpu.set(min(1.0, load1 / cpu_count))
        except (OSError, AttributeError):
            return self._cache_cpu.set(0.0)

    def read_memory_usage(self) -> dict[str, Any]:
        """Read memory usage information.

        Returns:
            Dict with total_gb, available_gb, used_gb, percent (0.0-1.0).
        """
        if self._cache_memory.is_fresh():
            return self._cache_memory.value

        if self._psutil:
            try:
                mem = self._psutil.virtual_memory()
                return self._cache_memory.set(
                    {
                        "total_gb": round(mem.total / (1024**3), 2),
                        "available_gb": round(mem.available / (1024**3), 2),
                        "used_gb": round(mem.used / (1024**3), 2),
                        "percent": mem.percent / 100.0,
                    }
                )
            except Exception as e:
                logger.debug("psutil memory read failed: %s", e)

        return self._cache_memory.set(
            {"total_gb": 0, "available_gb": 0, "used_gb": 0, "percent": 0.0}
        )

    def read_disk_usage(self, path: str = "/") -> dict[str, Any]:
        """Read disk usage for a given path.

        Returns:
            Dict with total_gb, free_gb, used_gb, percent (0.0-1.0).
        """
        if path not in self._cache_disk:
            self._cache_disk[path] = _CachedValue(ttl=30.0)  # Disk changes slowly

        cache = self._cache_disk[path]
        if cache.is_fresh():
            return cache.value

        if self._psutil:
            try:
                usage = self._psutil.disk_usage(path)
                return cache.set(
                    {
                        "total_gb": round(usage.total / (1024**3), 2),
                        "free_gb": round(usage.free / (1024**3), 2),
                        "used_gb": round(usage.used / (1024**3), 2),
                        "percent": usage.percent / 100.0,
                    }
                )
            except Exception as e:
                logger.debug("psutil disk read failed: %s", e)

        # Fallback: shutil
        import shutil

        try:
            usage = shutil.disk_usage(path)
            return cache.set(
                {
                    "total_gb": round(usage.total / (1024**3), 2),
                    "free_gb": round(usage.free / (1024**3), 2),
                    "used_gb": round(usage.used / (1024**3), 2),
                    "percent": round(usage.used / usage.total, 2) if usage.total > 0 else 0.0,
                }
            )
        except OSError:
            return cache.set({"total_gb": 0, "free_gb": 0, "used_gb": 0, "percent": 0.0})

    def read_network_status(self) -> dict[str, Any]:
        """Read network interface and connectivity status.

        Returns:
            Dict with connected (bool), interfaces (list), wifi_ssid (str|None).
        """
        if self._cache_network.is_fresh():
            return self._cache_network.value

        result: dict[str, Any] = {
            "connected": False,
            "interfaces": [],
            "wifi_ssid": None,
        }

        if self._psutil:
            try:
                addrs = self._psutil.net_if_addrs()
                stats = self._psutil.net_if_stats()

                interfaces = []
                for name, stat in stats.items():
                    if stat.isup and name != "lo" and not name.startswith("lo"):
                        interfaces.append(
                            {
                                "name": name,
                                "speed_mbps": stat.speed,
                                "mtu": stat.mtu,
                                "addresses": [
                                    addr.address
                                    for addr in addrs.get(name, [])
                                    if addr.family.name in ("AF_INET", "AF_INET6")
                                ],
                            }
                        )

                result["interfaces"] = interfaces
                result["connected"] = len(interfaces) > 0
            except Exception as e:
                logger.debug("psutil network read failed: %s", e)

        # WiFi SSID from platform
        if self._platform and hasattr(self._platform, "read_wifi_ssid"):
            result["wifi_ssid"] = self._platform.read_wifi_ssid()

        return self._cache_network.set(result)

    def read_volume(self) -> float | None:
        """Read system output volume (0.0 to 1.0). None if unavailable."""
        if self._cache_volume.is_fresh():
            return self._cache_volume.value

        if self._platform and hasattr(self._platform, "read_volume"):
            return self._cache_volume.set(self._platform.read_volume())
        return self._cache_volume.set(None)

    def read_brightness(self) -> float | None:
        """Read display brightness (0.0 to 1.0). None if unavailable."""
        if self._cache_brightness.is_fresh():
            return self._cache_brightness.value

        if self._platform and hasattr(self._platform, "read_brightness"):
            return self._cache_brightness.set(self._platform.read_brightness())
        return self._cache_brightness.set(None)

    def read_thermal_status(self) -> str | None:
        """Read thermal pressure level.

        Returns: "nominal", "moderate", "heavy", "critical", or None.
        """
        if self._cache_thermal.is_fresh():
            return self._cache_thermal.value

        # psutil temperature sensors
        if self._psutil and hasattr(self._psutil, "sensors_temperatures"):
            try:
                temps = self._psutil.sensors_temperatures()
                if temps:
                    max_temp = 0.0
                    for entries in temps.values():
                        for entry in entries:
                            if entry.current > max_temp:
                                max_temp = entry.current

                    if max_temp > 0:
                        if max_temp < 60:
                            return self._cache_thermal.set("nominal")
                        elif max_temp < 75:
                            return self._cache_thermal.set("moderate")
                        elif max_temp < 90:
                            return self._cache_thermal.set("heavy")
                        else:
                            return self._cache_thermal.set("critical")
            except Exception:
                pass

        # Platform fallback
        if self._platform and hasattr(self._platform, "read_thermal_pressure"):
            return self._cache_thermal.set(self._platform.read_thermal_pressure())

        return self._cache_thermal.set(None)

    def read_active_processes(self, top_n: int = 10) -> list[dict[str, Any]]:
        """Read top N processes by CPU usage.

        Returns:
            List of dicts with pid, name, cpu_percent, memory_percent.
        """
        if self._cache_processes.is_fresh():
            return self._cache_processes.value[:top_n]

        if not self._psutil:
            return self._cache_processes.set([])

        try:
            procs = []
            for proc in self._psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
                try:
                    info = proc.info
                    if info.get("cpu_percent", 0) > 0:
                        procs.append(
                            {
                                "pid": info["pid"],
                                "name": info["name"],
                                "cpu_percent": info["cpu_percent"],
                                "memory_percent": round(info.get("memory_percent", 0), 2),
                            }
                        )
                except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                    continue

            procs.sort(key=lambda p: p["cpu_percent"], reverse=True)
            return self._cache_processes.set(procs[:50])[:top_n]  # Cache top 50, return top N
        except Exception as e:
            logger.debug("psutil process read failed: %s", e)
            return self._cache_processes.set([])

    def read_uptime(self) -> float:
        """Read system uptime in seconds."""
        if self._cache_uptime.is_fresh():
            return self._cache_uptime.value

        if self._psutil:
            try:
                boot_time = self._psutil.boot_time()
                uptime = time.time() - boot_time
                return self._cache_uptime.set(uptime)
            except Exception:
                pass

        return self._cache_uptime.set(0.0)

    def read_wifi_ssid(self) -> str | None:
        """Read current WiFi SSID. None if not connected or unavailable."""
        if self._cache_wifi.is_fresh():
            return self._cache_wifi.value

        if self._platform and hasattr(self._platform, "read_wifi_ssid"):
            return self._cache_wifi.set(self._platform.read_wifi_ssid())
        return self._cache_wifi.set(None)

    def check_file_exists(self, filepath: str) -> bool:
        """Sensor: Verify if a physical file exists on disk."""
        return os.path.exists(filepath)

    def read_all_sensors(self) -> dict[str, Any]:
        """Read all sensors in a single call. Useful for dashboards.

        Returns a comprehensive snapshot of the host device state.
        """
        return {
            "platform": self._system,
            "battery": self.read_battery_info(),
            "cpu_load": self.read_cpu_load(),
            "memory": self.read_memory_usage(),
            "disk": self.read_disk_usage("/"),
            "network": self.read_network_status(),
            "volume": self.read_volume(),
            "brightness": self.read_brightness(),
            "thermal": self.read_thermal_status(),
            "uptime_seconds": self.read_uptime(),
            "wifi_ssid": self.read_wifi_ssid(),
            "timestamp": time.time(),
        }

    # ── Actuators ────────────────────────────────────────────────────────

    async def create_file(self, filepath: str, content: str) -> bool:
        """Actuator: Create a file safely.

        Only writes to whitelisted directories. Logs all operations.
        """
        if not self.config.enable_file_writes:
            logger.warning("File writes disabled — rejecting create_file(%s)", filepath)
            return False

        if not self.config.is_path_safe(filepath):
            logger.warning(
                "Path not in safe directories — rejecting create_file(%s). Allowed: %s",
                filepath,
                self.config.safe_write_dirs,
            )
            return False

        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            logger.info("OS Actuator: Created file at %s (%d bytes)", filepath, len(content))
            return True
        except Exception:
            logger.exception("Failed to execute create_file on host OS: %s", filepath)
            return False

    async def delete_file(self, filepath: str) -> bool:
        """Actuator: Delete a file safely.

        Only deletes from whitelisted directories. Logs all operations.
        """
        if not self.config.enable_file_deletes:
            logger.warning("File deletes disabled — rejecting delete_file(%s)", filepath)
            return False

        if not self.config.is_path_safe(filepath):
            logger.warning(
                "Path not in safe directories — rejecting delete_file(%s)",
                filepath,
            )
            return False

        try:
            path = Path(filepath)
            if path.exists():
                path.unlink()
                logger.info("OS Actuator: Deleted file at %s", filepath)
                return True
            else:
                logger.warning("OS Actuator: File does not exist: %s", filepath)
                return False
        except Exception:
            logger.exception("Failed to execute delete_file on host OS: %s", filepath)
            return False

    async def set_volume(self, level: float) -> bool:
        """Actuator: Set system output volume (0.0-1.0)."""
        if self._platform and hasattr(self._platform, "set_volume"):
            result = await self._platform.set_volume(level)
            if result:
                self._cache_volume.set(level)  # Optimistically update cache
            return result
        logger.warning("Volume control not available on platform '%s'", self._system)
        return False

    async def set_mute(self, muted: bool) -> bool:
        """Actuator: Mute or unmute system audio."""
        if self._platform and hasattr(self._platform, "set_mute"):
            return await self._platform.set_mute(muted)
        logger.warning("Mute control not available on platform '%s'", self._system)
        return False

    async def set_brightness(self, level: float) -> bool:
        """Actuator: Set display brightness (0.0-1.0)."""
        if self._platform and hasattr(self._platform, "set_brightness"):
            result = await self._platform.set_brightness(level)
            if result:
                self._cache_brightness.set(level)
            return result
        logger.warning("Brightness control not available on platform '%s'", self._system)
        return False

    async def open_application(self, app_name: str) -> bool:
        """Actuator: Open an application by name.

        Only opens applications on the allowlist.
        """
        if not self.config.is_app_allowed(app_name):
            logger.warning(
                "Application '%s' not on allowlist — rejecting. Allowed: %s",
                app_name,
                self.config.app_allowlist,
            )
            return False

        if self._platform and hasattr(self._platform, "open_application"):
            return await self._platform.open_application(app_name)
        logger.warning("App launching not available on platform '%s'", self._system)
        return False

    async def send_system_notification(self, title: str, body: str) -> bool:
        """Actuator: Send a native OS notification."""
        if self._platform and hasattr(self._platform, "send_notification"):
            return await self._platform.send_notification(title, body)
        logger.warning("Notifications not available on platform '%s'", self._system)
        return False

    # ── Introspection ────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Quick introspection snapshot for the autonomy dashboard."""
        return {
            "platform": self._system,
            "psutil_available": self._psutil is not None,
            "platform_backend": self._platform.__name__ if self._platform else None,
            "actuator_config": {
                "file_writes": self.config.enable_file_writes,
                "file_deletes": self.config.enable_file_deletes,
                "safe_dirs": self.config.safe_write_dirs,
                "app_allowlist_count": len(self.config.app_allowlist),
            },
        }
