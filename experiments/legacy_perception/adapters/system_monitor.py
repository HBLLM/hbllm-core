"""System Activity Monitor adapter for RealityEventBus.

This is a high-signal, low-noise adapter that emits events for:
- Active window tracking (app switching)
- Idle detection
- System sleep/wake states

Supports multiple platforms:
- macOS (via pyobjc)
- Linux (via X11/Wayland)
- Windows (via win32gui)
- Fallback (simulated events)
"""

from __future__ import annotations

import asyncio
import logging
import platform
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from hbllm.perception.reality_bus import (
    EventOrigin,
    PerceptionEvent,
    PerceptionModality,
    RealityEventBus,
)

logger = logging.getLogger(__name__)


@dataclass
class SystemState:
    """Current system state snapshot."""

    active_app: str = ""
    active_window_title: str = ""
    idle_time_seconds: float = 0.0
    is_screen_locked: bool = False
    is_sleeping: bool = False
    platform: str = platform.system()


class SystemMonitorProvider(ABC):
    """Abstract base class for system monitoring providers."""

    @abstractmethod
    async def get_system_state(self) -> SystemState:
        """Get current system state."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available on the current platform."""
        pass


class MockSystemProvider(SystemMonitorProvider):
    """Mock system provider for testing and fallback."""

    def __init__(self) -> None:
        self._last_app = ""
        self._apps = ["VSCode", "Terminal", "Browser", "Slack", "Notes"]
        self._app_index = 0

    async def get_system_state(self) -> SystemState:
        """Return mock system state."""
        # Simulate app switching occasionally
        if time.time() % 60 < 5:  # Every minute, switch apps
            self._app_index = (self._app_index + 1) % len(self._apps)
            self._last_app = self._apps[self._app_index]

        return SystemState(
            active_app=self._last_app or "VSCode",
            active_window_title=f"{self._last_app} Window",
            idle_time_seconds=0.0,
            is_screen_locked=False,
            is_sleeping=False,
            platform="mock",
        )

    def is_available(self) -> bool:
        return True


class MacOSSystemProvider(SystemMonitorProvider):
    """macOS system provider using pyobjc."""

    def __init__(self) -> None:
        self._workspace: Any = None
        self._initialized = False

    def _initialize(self) -> None:
        """Initialize macOS workspace access."""
        try:
            from AppKit import NSWorkspace  # type: ignore[import-not-found]

            self._workspace = NSWorkspace.sharedWorkspace()
            self._initialized = True
            logger.info("macOS system provider initialized")
        except ImportError:
            logger.warning("pyobjc not installed. Install with: pip install pyobjc")
        except Exception as e:
            logger.error("Failed to initialize macOS provider: %s", e)

    async def get_system_state(self) -> SystemState:
        """Get current macOS system state."""
        if not self._initialized:
            self._initialize()

        if not self._workspace:
            # Fallback to mock
            mock = MockSystemProvider()
            return await mock.get_system_state()

        try:
            # Get active application
            active_app = self._workspace.frontmostApplication()
            app_name = active_app.localizedName() if active_app else "Unknown"

            # Get idle time using IOKit (simplified)
            idle_time = 0.0
            try:
                from Quartz import (
                    CGEventSourceSecondsSinceLastEventType,  # type: ignore[import-not-found]
                    kCGEventSourceStateHIDSystemState,  # type: ignore[import-not-found]
                )

                idle_time = CGEventSourceSecondsSinceLastEventType(
                    kCGEventSourceStateHIDSystemState,  # type: ignore[arg-type]
                    1,  # Any event type
                )
            except Exception:
                pass  # Idle time not available

            return SystemState(
                active_app=app_name,
                active_window_title=app_name,
                idle_time_seconds=idle_time,
                is_screen_locked=False,  # Would need additional API
                is_sleeping=False,
                platform="macOS",
            )

        except Exception as e:
            logger.error("Failed to get macOS system state: %s", e)
            mock = MockSystemProvider()
            return await mock.get_system_state()

    def is_available(self) -> bool:
        return platform.system() == "Darwin"


class LinuxSystemProvider(SystemMonitorProvider):
    """Linux system provider using X11/Wayland."""

    def __init__(self) -> None:
        self._display: Any = None
        self._initialized = False

    def _initialize(self) -> None:
        """Initialize X11 display access."""
        try:
            import Xlib.display  # type: ignore[import-not-found]

            self._display = Xlib.display.Display()
            self._initialized = True
            logger.info("Linux system provider initialized")
        except ImportError:
            logger.warning("python-xlib not installed. Install with: pip install python-xlib")
        except Exception as e:
            logger.error("Failed to initialize Linux provider: %s", e)

    async def get_system_state(self) -> SystemState:
        """Get current Linux system state."""
        if not self._initialized:
            self._initialize()

        if not self._display:
            mock = MockSystemProvider()
            return await mock.get_system_state()

        try:
            # Get active window
            window = self._display.get_input_focus().focus
            window_name = window.get_wm_name() if window else "Unknown"

            # Get idle time from XScreenSaver
            idle_time = 0.0
            try:
                import Xlib.ext.xss  # type: ignore[import-not-found]

                idle_info = Xlib.ext.xss.query_info(self._display, self._display.screen().root)
                idle_time = idle_info.idle / 1000.0  # Convert milliseconds to seconds
            except Exception:
                pass

            return SystemState(
                active_app=window_name,
                active_window_title=window_name,
                idle_time_seconds=idle_time,
                is_screen_locked=False,
                is_sleeping=False,
                platform="Linux",
            )

        except Exception as e:
            logger.error("Failed to get Linux system state: %s", e)
            mock = MockSystemProvider()
            return await mock.get_system_state()

    def is_available(self) -> bool:
        return platform.system() == "Linux"


class WindowsSystemProvider(SystemMonitorProvider):
    """Windows system provider using win32gui."""

    def __init__(self) -> None:
        self._initialized = False

    def _initialize(self) -> None:
        """Initialize Windows API access."""
        try:
            import importlib.util

            if importlib.util.find_spec("win32gui") is None:
                raise ImportError("win32gui not available")

            self._initialized = True
            logger.info("Windows system provider initialized")
        except ImportError:
            logger.warning("pywin32 not installed. Install with: pip install pywin32")
        except Exception as e:
            logger.error("Failed to initialize Windows provider: %s", e)

    async def get_system_state(self) -> SystemState:
        """Get current Windows system state."""
        if not self._initialized:
            self._initialize()

        if not self._initialized:
            mock = MockSystemProvider()
            return await mock.get_system_state()

        try:
            import win32gui  # type: ignore[import-not-found]
            import win32process  # type: ignore[import-not-found]

            # Get active window
            hwnd = win32gui.GetForegroundWindow()
            window_title = win32gui.GetWindowText(hwnd)

            # Get process name
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            try:
                import psutil  # type: ignore[import-not-found]

                process = psutil.Process(pid)
                app_name = process.name()
            except ImportError:
                app_name = window_title

            # Get idle time (simplified)
            idle_time = 0.0
            try:
                import ctypes

                class LASTINPUTINFO(ctypes.Structure):
                    _fields_ = [
                        ("cbSize", ctypes.c_uint32),
                        ("dwTime", ctypes.c_uint32),
                    ]

                lastInputInfo = LASTINPUTINFO()
                lastInputInfo.cbSize = ctypes.sizeof(lastInputInfo)
                ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lastInputInfo))
                millis = ctypes.windll.kernel32.GetTickCount() - lastInputInfo.dwTime
                idle_time = millis / 1000.0
            except Exception:
                pass

            return SystemState(
                active_app=app_name,
                active_window_title=window_title,
                idle_time_seconds=idle_time,
                is_screen_locked=False,
                is_sleeping=False,
                platform="Windows",
            )

        except Exception as e:
            logger.error("Failed to get Windows system state: %s", e)
            mock = MockSystemProvider()
            return await mock.get_system_state()

    def is_available(self) -> bool:
        return platform.system() == "Windows"


class SystemActivityMonitor:
    """System activity monitor with pluggable providers."""

    def __init__(
        self,
        bus: RealityEventBus,
        device_id: str = "local_device",
        provider: SystemMonitorProvider | None = None,
        poll_interval_seconds: int = 30,
        idle_threshold_seconds: int = 300,
    ) -> None:
        self.bus = bus
        self.device_id = device_id
        self._provider = provider or self._get_default_provider()
        self._poll_interval = poll_interval_seconds
        self._idle_threshold = idle_threshold_seconds
        self._running = False
        self._task: asyncio.Task[Any] | None = None
        self._last_state: SystemState | None = None

    def _get_default_provider(self) -> SystemMonitorProvider:
        """Get the default provider for the current platform."""
        providers = [
            MacOSSystemProvider(),
            LinuxSystemProvider(),
            WindowsSystemProvider(),
        ]

        for provider in providers:
            if provider.is_available():
                return provider

        logger.warning("No native system provider available, using mock")
        return MockSystemProvider()

    def start(self) -> None:
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._monitor_loop())
            logger.info(
                "SystemActivityMonitor started with provider: %s", type(self._provider).__name__
            )

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("SystemActivityMonitor stopped")

    async def _monitor_loop(self) -> None:
        """Poll system provider for state changes."""
        while self._running:
            try:
                current_state = await self._provider.get_system_state()

                # Detect and emit state changes
                if self._last_state is None:
                    # First state, emit initial state
                    await self._emit_state_change("initial", current_state)
                else:
                    # Check for changes
                    if current_state.active_app != self._last_state.active_app:
                        await self._emit_state_change("app_switch", current_state)

                    if current_state.is_screen_locked != self._last_state.is_screen_locked:
                        if current_state.is_screen_locked:
                            await self._emit_state_change("screen_locked", current_state)
                        else:
                            await self._emit_state_change("screen_unlocked", current_state)

                    if current_state.is_sleeping != self._last_state.is_sleeping:
                        if current_state.is_sleeping:
                            await self._emit_state_change("system_sleep", current_state)
                        else:
                            await self._emit_state_change("system_wake", current_state)

                    # Check idle threshold
                    if current_state.idle_time_seconds >= self._idle_threshold and (
                        self._last_state.idle_time_seconds < self._idle_threshold
                    ):
                        await self._emit_state_change("idle", current_state)

                self._last_state = current_state
                await asyncio.sleep(self._poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in SystemActivityMonitor: %s", e)
                await asyncio.sleep(5.0)

    async def _emit_state_change(self, sub_type: str, state: SystemState) -> None:
        """Emit a system state change event to the reality bus."""
        event = PerceptionEvent(
            entity_id=self.device_id,
            event_type="os_activity",
            sub_type=sub_type,
            modality=PerceptionModality.SYSTEM,
            origin=EventOrigin.SYSTEM,
            confidence=1.0,
            source_trust=1.0,
            payload={
                "active_app": state.active_app,
                "active_window_title": state.active_window_title,
                "idle_time_seconds": state.idle_time_seconds,
                "is_screen_locked": state.is_screen_locked,
                "is_sleeping": state.is_sleeping,
                "platform": state.platform,
            },
        )

        await self.bus.ingest(event)
        logger.info("Emitted system event: %s (app=%s)", sub_type, state.active_app)
