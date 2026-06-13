"""System Health Watcher — monitors disk, memory, CPU.

Uses only stdlib (``shutil.disk_usage``, ``os.getloadavg``).
Emits alerts when system resource thresholds are exceeded.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import time
from dataclasses import dataclass
from typing import Any

from hbllm.network.messages import Message, MessageType

logger = logging.getLogger(__name__)


@dataclass
class HealthThresholds:
    """Configurable alert thresholds."""

    disk_free_min_gb: float = 5.0  # Alert if free disk < 5GB
    load_avg_max: float = 8.0  # Alert if 1-min load avg > 8.0
    check_interval: float = 60.0  # Minimum seconds between checks


class SystemHealthWatcher:
    """Proactive handler that monitors system health.

    Checks disk space and CPU load average on each tick.
    Emits alert messages when thresholds are exceeded.

    Usage::

        watcher = SystemHealthWatcher()
        autonomy_core.add_proactive_handler("system_health", watcher.check)
    """

    def __init__(
        self,
        thresholds: HealthThresholds | None = None,
        disk_paths: list[str] | None = None,
    ) -> None:
        self.thresholds = thresholds or HealthThresholds()
        self.disk_paths = disk_paths or ["/"]
        self._last_check: float = 0.0
        self._alert_cooldowns: dict[str, float] = {}
        self._alert_cooldown_s = 300.0  # Don't repeat same alert within 5 min

    def _check_disk(self) -> list[dict[str, Any]]:
        """Check disk space on monitored paths."""
        alerts: list[dict[str, Any]] = []
        for path in self.disk_paths:
            try:
                usage = shutil.disk_usage(path)
                free_gb = usage.free / (1024**3)
                total_gb = usage.total / (1024**3)
                used_pct = (usage.used / usage.total) * 100 if usage.total > 0 else 0

                if free_gb < self.thresholds.disk_free_min_gb:
                    alerts.append(
                        {
                            "type": "disk_low",
                            "path": path,
                            "free_gb": round(free_gb, 2),
                            "total_gb": round(total_gb, 2),
                            "used_pct": round(used_pct, 1),
                            "severity": "critical" if free_gb < 1.0 else "warning",
                        }
                    )
            except OSError as e:
                logger.debug("[SystemHealthWatcher] Cannot check disk at %s: %s", path, e)
        return alerts

    def _check_cpu_load(self) -> list[dict[str, Any]]:
        """Check CPU load average (macOS/Linux only)."""
        alerts: list[dict[str, Any]] = []
        if platform.system() == "Windows":
            return alerts

        try:
            load1, load5, load15 = os.getloadavg()
            if load1 > self.thresholds.load_avg_max:
                alerts.append(
                    {
                        "type": "cpu_high",
                        "load_1min": round(load1, 2),
                        "load_5min": round(load5, 2),
                        "load_15min": round(load15, 2),
                        "severity": "critical"
                        if load1 > self.thresholds.load_avg_max * 1.5
                        else "warning",
                    }
                )
        except OSError:
            pass
        return alerts

    def _is_on_cooldown(self, alert_key: str) -> bool:
        """Check if an alert type is on cooldown."""
        now = time.monotonic()
        last = self._alert_cooldowns.get(alert_key, 0.0)
        if now - last < self._alert_cooldown_s:
            return True
        self._alert_cooldowns[alert_key] = now
        return False

    async def check(self) -> list[Message] | None:
        """Proactive handler callback — check system health.

        Returns alert Messages when thresholds are exceeded, None otherwise.
        """
        now = time.monotonic()
        if now - self._last_check < self.thresholds.check_interval:
            return None
        self._last_check = now

        all_alerts: list[dict[str, Any]] = []
        all_alerts.extend(self._check_disk())
        all_alerts.extend(self._check_cpu_load())

        if not all_alerts:
            return None

        messages: list[Message] = []
        for alert in all_alerts:
            alert_key = f"{alert['type']}:{alert.get('path', 'cpu')}"
            if self._is_on_cooldown(alert_key):
                continue

            urgency = 0.8 if alert.get("severity") == "critical" else 0.5

            messages.append(
                Message(
                    type=MessageType.EVENT,
                    source_node_id="autonomy.watcher.system_health",
                    topic="perception.system.health_alert",
                    payload={
                        **alert,
                        "_urgency": urgency,
                    },
                )
            )

            logger.warning(
                "[SystemHealthWatcher] %s alert: %s",
                alert.get("severity", "warning").upper(),
                alert,
            )

        return messages if messages else None

    def get_current_status(self) -> dict[str, Any]:
        """Get current system health metrics (non-alerting, informational)."""
        status: dict[str, Any] = {"platform": platform.system()}

        for path in self.disk_paths:
            try:
                usage = shutil.disk_usage(path)
                status[f"disk_{path}"] = {
                    "free_gb": round(usage.free / (1024**3), 2),
                    "total_gb": round(usage.total / (1024**3), 2),
                    "used_pct": round((usage.used / usage.total) * 100, 1)
                    if usage.total > 0
                    else 0,
                }
            except OSError:
                pass

        if platform.system() != "Windows":
            try:
                load1, load5, load15 = os.getloadavg()
                status["cpu_load"] = {
                    "1min": round(load1, 2),
                    "5min": round(load5, 2),
                    "15min": round(load15, 2),
                }
            except OSError:
                pass

        return status

    def snapshot(self) -> dict[str, Any]:
        """Introspection snapshot."""
        return {
            "disk_paths": self.disk_paths,
            "thresholds": {
                "disk_free_min_gb": self.thresholds.disk_free_min_gb,
                "load_avg_max": self.thresholds.load_avg_max,
            },
            "active_cooldowns": len(self._alert_cooldowns),
        }
