"""
Device Bridge — cross-device session continuity and presence tracking.

Enables seamless conversation handoff between devices:
    - Start a conversation on your phone, continue on your laptop
    - Receive push notifications on the nearest active device
    - Automatic presence detection via heartbeat

Architecture:
    1. Each client device registers with a unique device_id and capabilities
    2. Devices send periodic heartbeats to maintain presence
    3. When a session migrates, context is transferred to the new device
    4. Notifications route to the most appropriate active device
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.network.messages import Message, MessageType

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """Registered device with capabilities and presence state."""

    device_id: str
    tenant_id: str
    device_type: str = "unknown"  # "mobile", "desktop", "speaker", "browser"
    capabilities: list[str] = field(default_factory=list)  # ["audio", "display", "keyboard"]
    last_heartbeat: float = field(default_factory=time.time)
    is_active: bool = True
    current_session_id: str | None = None
    push_token: str | None = None  # FCM/APNs token for push notifications
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.last_heartbeat

    @property
    def is_stale(self) -> bool:
        """Device is considered stale after 5 minutes without heartbeat."""
        return self.age_seconds > 300


@dataclass
class SessionHandoff:
    """Record of a session migrating between devices."""

    session_id: str
    from_device: str
    to_device: str
    timestamp: float = field(default_factory=time.time)
    context_transferred: bool = False


class DeviceBridge:
    """
    Manages cross-device session continuity and presence tracking.

    Usage:
        bridge = DeviceBridge(bus=message_bus)
        await bridge.start()
        bridge.register_device(DeviceInfo(device_id="phone-1", tenant_id="user1", ...))
    """

    HEARTBEAT_TIMEOUT_S = 300.0  # 5 minutes
    PRESENCE_CHECK_INTERVAL_S = 60.0

    def __init__(self, bus: Any | None = None) -> None:
        self.bus = bus
        self._devices: dict[str, DeviceInfo] = {}  # device_id → DeviceInfo
        self._tenant_devices: dict[str, set[str]] = {}  # tenant_id → {device_ids}
        self._handoff_log: list[SessionHandoff] = []
        self._running = False
        self._presence_task: asyncio.Task[Any] | None = None

    async def start(self) -> None:
        """Start the device bridge."""
        self._running = True
        self._presence_task = asyncio.create_task(self._presence_loop())

        if self.bus:
            await self.bus.subscribe("device.register", self._on_device_register)
            await self.bus.subscribe("device.heartbeat", self._on_heartbeat)
            await self.bus.subscribe("device.handoff", self._on_handoff_request)
            await self.bus.subscribe("device.unregister", self._on_device_unregister)

        logger.info("DeviceBridge started")

    async def stop(self) -> None:
        """Stop the device bridge."""
        self._running = False
        if self._presence_task:
            self._presence_task.cancel()
            try:
                await self._presence_task
            except asyncio.CancelledError:
                pass
        logger.info("DeviceBridge stopped (%d devices tracked)", len(self._devices))

    # ── Device Management ────────────────────────────────────────────────

    def register_device(self, device: DeviceInfo) -> None:
        """Register a new device or update an existing one."""
        self._devices[device.device_id] = device

        if device.tenant_id not in self._tenant_devices:
            self._tenant_devices[device.tenant_id] = set()
        self._tenant_devices[device.tenant_id].add(device.device_id)

        logger.info(
            "Device registered: %s (type=%s, tenant=%s, caps=%s)",
            device.device_id,
            device.device_type,
            device.tenant_id,
            device.capabilities,
        )

    def unregister_device(self, device_id: str) -> None:
        """Remove a device from tracking."""
        device = self._devices.pop(device_id, None)
        if device and device.tenant_id in self._tenant_devices:
            self._tenant_devices[device.tenant_id].discard(device_id)

    def heartbeat(self, device_id: str) -> None:
        """Update device presence."""
        device = self._devices.get(device_id)
        if device:
            device.last_heartbeat = time.time()
            device.is_active = True

    # ── Session Handoff ──────────────────────────────────────────────────

    async def handoff_session(
        self,
        session_id: str,
        from_device: str,
        to_device: str,
    ) -> bool:
        """Transfer a session from one device to another."""
        from_dev = self._devices.get(from_device)
        to_dev = self._devices.get(to_device)

        if not from_dev or not to_dev:
            logger.warning("Handoff failed: device not found (%s → %s)", from_device, to_device)
            return False

        if from_dev.tenant_id != to_dev.tenant_id:
            logger.warning("Handoff denied: cross-tenant handoff not allowed")
            return False

        # Update session ownership
        from_dev.current_session_id = None
        to_dev.current_session_id = session_id

        handoff = SessionHandoff(
            session_id=session_id,
            from_device=from_device,
            to_device=to_device,
        )

        # Notify both devices
        if self.bus:
            await self.bus.publish(
                "device.handoff.complete",
                Message(
                    type=MessageType.EVENT,
                    source_node_id="device_bridge",
                    topic="device.handoff.complete",
                    payload={
                        "session_id": session_id,
                        "from_device": from_device,
                        "to_device": to_device,
                        "to_device_type": to_dev.device_type,
                        "to_device_capabilities": to_dev.capabilities,
                    },
                ),
            )

        handoff.context_transferred = True
        self._handoff_log.append(handoff)

        logger.info(
            "Session %s handed off: %s (%s) → %s (%s)",
            session_id,
            from_device,
            from_dev.device_type,
            to_device,
            to_dev.device_type,
        )
        return True

    # ── Device Selection ─────────────────────────────────────────────────

    def get_best_device(
        self,
        tenant_id: str,
        required_capabilities: list[str] | None = None,
    ) -> DeviceInfo | None:
        """Select the best active device for a tenant.

        Priority:
            1. Most recently active device
            2. Device with required capabilities
            3. Desktop > Mobile > Speaker > Browser
        """
        device_ids = self._tenant_devices.get(tenant_id, set())
        if not device_ids:
            return None

        active = [
            self._devices[did]
            for did in device_ids
            if did in self._devices and self._devices[did].is_active
        ]

        if not active:
            return None

        # Filter by capabilities
        if required_capabilities:
            filtered = [
                d for d in active if all(cap in d.capabilities for cap in required_capabilities)
            ]
            if filtered:
                active = filtered

        # Sort by recency
        active.sort(key=lambda d: d.last_heartbeat, reverse=True)

        return active[0]

    def get_active_devices(self, tenant_id: str) -> list[DeviceInfo]:
        """Get all active devices for a tenant."""
        device_ids = self._tenant_devices.get(tenant_id, set())
        return [
            self._devices[did]
            for did in device_ids
            if did in self._devices and self._devices[did].is_active
        ]

    # ── Bus Event Handlers ───────────────────────────────────────────────

    async def _on_device_register(self, msg: Message) -> None:
        payload = msg.payload
        device = DeviceInfo(
            device_id=payload.get("device_id", ""),
            tenant_id=msg.tenant_id or payload.get("tenant_id", "default"),
            device_type=payload.get("device_type", "unknown"),
            capabilities=payload.get("capabilities", []),
            push_token=payload.get("push_token"),
            metadata=payload.get("metadata", {}),
        )
        if device.device_id:
            self.register_device(device)

    async def _on_heartbeat(self, msg: Message) -> None:
        device_id = msg.payload.get("device_id", "")
        if device_id:
            self.heartbeat(device_id)

    async def _on_handoff_request(self, msg: Message) -> None:
        payload = msg.payload
        await self.handoff_session(
            session_id=payload.get("session_id", ""),
            from_device=payload.get("from_device", ""),
            to_device=payload.get("to_device", ""),
        )

    async def _on_device_unregister(self, msg: Message) -> None:
        device_id = msg.payload.get("device_id", "")
        if device_id:
            self.unregister_device(device_id)

    # ── Presence Loop ────────────────────────────────────────────────────

    async def _presence_loop(self) -> None:
        """Periodically check device presence and mark stale devices."""
        while self._running:
            try:
                await asyncio.sleep(self.PRESENCE_CHECK_INTERVAL_S)

                stale_ids = []
                for device_id, device in self._devices.items():
                    if device.is_stale and device.is_active:
                        device.is_active = False
                        stale_ids.append(device_id)

                if stale_ids:
                    logger.info("Devices went inactive: %s", stale_ids)

                    if self.bus:
                        for device_id in stale_ids:
                            await self.bus.publish(
                                "device.inactive",
                                Message(
                                    type=MessageType.EVENT,
                                    source_node_id="device_bridge",
                                    topic="device.inactive",
                                    payload={"device_id": device_id},
                                ),
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Presence loop error: %s", e)

    # ── Stats ────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "total_devices": len(self._devices),
            "active_devices": sum(1 for d in self._devices.values() if d.is_active),
            "tenants": len(self._tenant_devices),
            "handoffs": len(self._handoff_log),
        }
