"""Device Discovery & Hot-Plug Engine for HBLLM.

Monitors physical USB/Serial ports and network endpoints for smart peripherals,
automatically enumerating them, performing descriptor handshakes, and mounting them
into the DriverManager and Cognitive Bus without restarting the Brain.
"""

from __future__ import annotations

import asyncio
import glob
import logging
import platform
from typing import Any

from hbllm.drivers.base import (
    DriverModality,
    DriverStreamType,
    SynapticDeviceDescriptor,
)
from hbllm.drivers.manager import DriverManager
from hbllm.drivers.network_driver import NetworkDeviceDriver
from hbllm.drivers.serial_driver import SerialDeviceDriver

logger = logging.getLogger(__name__)


class DeviceDiscoveryEngine:
    """Discovers, enumerates, and hot-plugs physical and virtual devices."""

    def __init__(
        self,
        driver_manager: DriverManager,
        scan_interval_s: float = 3.0,
        enable_physical_scan: bool = True,
    ) -> None:
        self.driver_manager = driver_manager
        self.scan_interval_s = scan_interval_s
        self.enable_physical_scan = enable_physical_scan
        self._running = False
        self._scan_task: asyncio.Task[None] | None = None
        self._known_ports: set[str] = set()

    async def start(self) -> None:
        """Start the background hot-plug discovery loop."""
        self._running = True
        self._known_ports = set(self._scan_available_ports())
        self._scan_task = asyncio.create_task(self._discovery_loop(), name="device_discovery_loop")
        logger.info(
            "DeviceDiscoveryEngine started (Scan interval: %.1fs, Initial ports: %s)",
            self.scan_interval_s,
            list(self._known_ports),
        )

    async def stop(self) -> None:
        """Stop discovery loop."""
        self._running = False
        if self._scan_task and not self._scan_task.done():
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
        logger.info("DeviceDiscoveryEngine stopped.")

    def _scan_available_ports(self) -> list[str]:
        """Cross-platform scan for serial/USB ports."""
        if not self.enable_physical_scan:
            return []

        try:
            import serial.tools.list_ports  # type: ignore

            ports = [p.device for p in serial.tools.list_ports.comports()]
            if ports:
                return ports
        except Exception:
            pass

        # Fallback to globbing device nodes on POSIX
        system = platform.system()
        found: list[str] = []
        if system == "Darwin":
            found.extend(glob.glob("/dev/cu.usb*"))
            found.extend(glob.glob("/dev/tty.usb*"))
        elif system == "Linux":
            found.extend(glob.glob("/dev/ttyUSB*"))
            found.extend(glob.glob("/dev/ttyACM*"))

        return sorted(found)

    async def _discovery_loop(self) -> None:
        """Periodic background loop detecting plug/unplug events."""
        while self._running:
            try:
                current_ports = set(self._scan_available_ports())

                # Newly attached ports
                new_ports = current_ports - self._known_ports
                for port in new_ports:
                    await self._handle_port_attached(port)

                # Detached ports
                removed_ports = self._known_ports - current_ports
                for port in removed_ports:
                    await self._handle_port_detached(port)

                self._known_ports = current_ports
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in device discovery loop: %s", e)

            await asyncio.sleep(self.scan_interval_s)

    async def _handle_port_attached(self, port: str) -> None:
        """Create and attach driver when a new physical port is detected."""
        device_id = f"usb_{port.split('/')[-1].replace('.', '_')}"
        logger.info("Detected new physical device at '%s' -> device_id: '%s'", port, device_id)

        descriptor = SynapticDeviceDescriptor(
            device_id=device_id,
            device_type="bidirectional",
            modalities=[DriverModality.STRUCTURED],
            stream_type=DriverStreamType.CONTINUOUS_STREAM,
            metadata={"port": port},
        )
        driver = SerialDeviceDriver(name=device_id, port=port, descriptor=descriptor)
        await self.driver_manager.attach_driver(driver, target=port)

    async def _handle_port_detached(self, port: str) -> None:
        """Detach driver when a physical port is removed."""
        device_id = f"usb_{port.split('/')[-1].replace('.', '_')}"
        logger.info("Physical device disconnected from '%s' -> device_id: '%s'", port, device_id)
        if device_id in self.driver_manager.get_active_drivers():
            await self.driver_manager.detach_driver(device_id)

    # ── Network Device PnP Registration ───────────────────────────────────

    async def register_network_device(
        self,
        descriptor: SynapticDeviceDescriptor,
        target_info: Any = None,
    ) -> NetworkDeviceDriver:
        """Hot-plug an external network device (e.g. phone, ESP32, web client)."""
        driver = NetworkDeviceDriver(
            name=descriptor.device_id,
            descriptor=descriptor,
        )
        await self.driver_manager.attach_driver(driver, target=target_info)
        logger.info("Registered network device '%s' into DriverManager", descriptor.device_id)
        return driver

    async def unregister_network_device(self, device_id: str) -> bool:
        """Hot-unplug an external network device."""
        return await self.driver_manager.detach_driver(device_id)
