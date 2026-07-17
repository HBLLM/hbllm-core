"""
Digital Twin — ephemeral operational runtime state manager.

ADR 002 §7: Decouples persistent identity (``SelfModel``) from live
operational runtime telemetry.

The DigitalTwin represents the current hardware, task, resource, and
cluster state of the cognitive system.  Unlike ``SelfModel`` (which
stores enduring identity, ethics, and personality), the DigitalTwin is:

    - **Ephemeral**: Reconstructed on every system startup.
    - **Not persisted**: Excluded from long-term episodic memory.
    - **Observable**: Exposes live metrics for diagnostics and scheduling.
    - **Rebuildable**: Can be reconstructed from active subsystem queries.

Use cases:
    - Runtime monitoring and scheduling decisions.
    - Zero-downtime failover and state migration.
    - Swarm diagnostics and distributed execution.
    - DigitalTwin feeds CognitiveScheduler resource budget checks.

Usage::

    from hbllm.brain.self_model.digital_twin import DigitalTwin

    twin = DigitalTwin(system_id="hbllm-node-01")
    twin.update_hardware(cpu_percent=45.0, ram_mb_used=1200)
    twin.register_active_goal("goal_123", "Answer user query")
    twin.register_plugin("temporal-reasoning", version="1.2.0")

    snapshot = twin.snapshot()
"""

from __future__ import annotations

import logging
import platform
import time
import uuid
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HardwareState:
    """Current hardware resource utilization.

    Attributes:
        cpu_percent: CPU utilization [0.0, 100.0].
        ram_mb_used: RAM used in megabytes.
        ram_mb_total: Total RAM in megabytes.
        vram_mb_used: VRAM used in megabytes (0 if CPU-only).
        vram_mb_total: Total VRAM in megabytes.
        disk_gb_free: Free disk space in gigabytes.
        temperature_c: CPU temperature in Celsius (None if unavailable).
        battery_percent: Battery level [0.0, 100.0] (None if plugged in).
    """

    cpu_percent: float = 0.0
    ram_mb_used: float = 0.0
    ram_mb_total: float = 0.0
    vram_mb_used: float = 0.0
    vram_mb_total: float = 0.0
    disk_gb_free: float = 0.0
    temperature_c: float | None = None
    battery_percent: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_percent": self.cpu_percent,
            "ram_mb_used": self.ram_mb_used,
            "ram_mb_total": self.ram_mb_total,
            "vram_mb_used": self.vram_mb_used,
            "vram_mb_total": self.vram_mb_total,
            "disk_gb_free": self.disk_gb_free,
            "temperature_c": self.temperature_c,
            "battery_percent": self.battery_percent,
        }


class DigitalTwin:
    """Ephemeral operational state manager for the cognitive runtime.

    Tracks live hardware metrics, active goals, loaded plugins,
    memory consumption, scheduler queues, connected devices, and
    cluster nodes.

    Invariants (ADR 002):
        - NOT persisted as identity or long-term memory.
        - Rebuilt from scratch on every system startup.
        - Excluded from episodic memory consolidation.
        - Primarily serves runtime monitoring, scheduling, and diagnostics.

    Args:
        system_id: Identifier for this HBLLM node instance.
    """

    def __init__(self, system_id: str = "") -> None:
        self.system_id = system_id or f"hbllm-{uuid.uuid4().hex[:8]}"
        self.started_at = time.time()

        # Hardware state
        self._hardware = HardwareState()

        # Active goals
        self._active_goals: dict[str, dict[str, Any]] = {}

        # Loaded plugins
        self._plugins: dict[str, dict[str, Any]] = {}

        # Connected devices (IoT, robotics)
        self._connected_devices: dict[str, dict[str, Any]] = {}

        # Cluster peers (for distributed mode)
        self._cluster_peers: dict[str, dict[str, Any]] = {}

        # Running tasks (fed by CognitiveScheduler)
        self._running_tasks: dict[str, dict[str, Any]] = {}

        # Memory subsystem stats
        self._memory_stats: dict[str, Any] = {}

        # Last snapshot timestamp
        self._last_snapshot_at = 0.0

        logger.info("DigitalTwin initialized: %s", self.system_id)

    # ── Hardware updates ─────────────────────────────────────────────

    def update_hardware(self, **kwargs: Any) -> None:
        """Update hardware state fields.

        Args:
            **kwargs: Any field of ``HardwareState`` (e.g., cpu_percent=45.0).
        """
        for key, value in kwargs.items():
            if hasattr(self._hardware, key):
                setattr(self._hardware, key, value)

    @property
    def hardware(self) -> HardwareState:
        """Current hardware state (read-only reference)."""
        return self._hardware

    # ── Goals ────────────────────────────────────────────────────────

    def register_active_goal(self, goal_id: str, description: str) -> None:
        """Register an active goal."""
        self._active_goals[goal_id] = {
            "description": description,
            "started_at": time.time(),
        }

    def complete_goal(self, goal_id: str) -> None:
        """Remove a goal from the active set."""
        self._active_goals.pop(goal_id, None)

    @property
    def active_goal_count(self) -> int:
        return len(self._active_goals)

    # ── Plugins ──────────────────────────────────────────────────────

    def register_plugin(
        self,
        plugin_name: str,
        version: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a loaded plugin."""
        self._plugins[plugin_name] = {
            "version": version,
            "loaded_at": time.time(),
            **(metadata or {}),
        }

    def unregister_plugin(self, plugin_name: str) -> None:
        """Unregister a plugin."""
        self._plugins.pop(plugin_name, None)

    # ── Connected devices ────────────────────────────────────────────

    def register_device(self, device_id: str, device_info: dict[str, Any]) -> None:
        """Register a connected IoT/robotics device."""
        self._connected_devices[device_id] = {
            **device_info,
            "connected_at": time.time(),
        }

    def disconnect_device(self, device_id: str) -> None:
        """Remove a disconnected device."""
        self._connected_devices.pop(device_id, None)

    # ── Cluster peers ────────────────────────────────────────────────

    def register_peer(self, peer_id: str, peer_info: dict[str, Any]) -> None:
        """Register a cluster peer node."""
        self._cluster_peers[peer_id] = {
            **peer_info,
            "registered_at": time.time(),
        }

    def remove_peer(self, peer_id: str) -> None:
        """Remove a cluster peer."""
        self._cluster_peers.pop(peer_id, None)

    # ── Task tracking ────────────────────────────────────────────────

    def register_task(self, task_id: str, task_info: dict[str, Any]) -> None:
        """Register a running task (fed by CognitiveScheduler)."""
        self._running_tasks[task_id] = {
            **task_info,
            "started_at": time.time(),
        }

    def complete_task(self, task_id: str) -> None:
        """Remove a completed task."""
        self._running_tasks.pop(task_id, None)

    # ── Memory stats ─────────────────────────────────────────────────

    def update_memory_stats(self, stats: dict[str, Any]) -> None:
        """Update memory subsystem statistics."""
        self._memory_stats = stats

    # ── Full snapshot ────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Generate a complete operational state snapshot.

        This is the primary output consumed by diagnostics, scheduling,
        and distributed coordination systems.

        Returns:
            Complete operational state as a serializable dict.
        """
        self._last_snapshot_at = time.time()
        return {
            "system_id": self.system_id,
            "platform": {
                "os": platform.system(),
                "arch": platform.machine(),
                "python": platform.python_version(),
            },
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "hardware": self._hardware.to_dict(),
            "active_goals": self._active_goals,
            "active_goal_count": len(self._active_goals),
            "plugins": self._plugins,
            "plugin_count": len(self._plugins),
            "connected_devices": self._connected_devices,
            "device_count": len(self._connected_devices),
            "cluster_peers": self._cluster_peers,
            "peer_count": len(self._cluster_peers),
            "running_tasks": self._running_tasks,
            "task_count": len(self._running_tasks),
            "memory_stats": self._memory_stats,
            "snapshot_at": self._last_snapshot_at,
        }

    def reset(self) -> None:
        """Reset the DigitalTwin to a clean startup state.

        Called on system restart to ensure the twin is fully ephemeral.
        """
        self.started_at = time.time()
        self._hardware = HardwareState()
        self._active_goals.clear()
        self._plugins.clear()
        self._connected_devices.clear()
        self._cluster_peers.clear()
        self._running_tasks.clear()
        self._memory_stats.clear()
        logger.info("DigitalTwin reset: %s", self.system_id)
