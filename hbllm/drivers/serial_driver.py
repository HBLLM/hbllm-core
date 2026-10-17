"""Serial/USB Hardware Device Driver for physical microcontrollers and sensors.

Connects to physical USB/Serial ports (Arduino, ESP32, STM32, USB sensors)
and streams sensor observations into the HBLLM Brain.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from hbllm.drivers.base import (
    BaseDriver,
    DriverAction,
    DriverCapability,
    DriverFeedback,
    DriverInput,
    DriverModality,
    DriverStreamType,
    SynapticDeviceDescriptor,
)

logger = logging.getLogger(__name__)


class SerialDeviceDriver(BaseDriver):
    """Driver for physical microcontrollers communicating via USB/Serial."""

    def __init__(
        self,
        name: str,
        port: str = "",
        baudrate: int = 115200,
        descriptor: SynapticDeviceDescriptor | None = None,
    ) -> None:
        desc = descriptor or SynapticDeviceDescriptor(
            device_id=name,
            device_type="bidirectional",
            modalities=[DriverModality.STRUCTURED],
            stream_type=DriverStreamType.CONTINUOUS_STREAM,
        )
        super().__init__(
            name=name,
            capabilities={
                DriverCapability.STREAMING_OBSERVATIONS,
                DriverCapability.ASYNC_EVENT_DRIVEN,
                DriverCapability.DISCRETE_ACTIONS,
            },
            descriptor=desc,
        )
        self.port = port
        self.baudrate = baudrate
        self._serial: Any = None
        self._input_queue: asyncio.Queue[DriverInput] = asyncio.Queue()
        self._read_task: asyncio.Task[None] | None = None
        self._last_observation: Any = None

    def connect(self, target: Any) -> bool:
        """Connect to the specified serial port or mock target."""
        self._target = target or self.port
        if isinstance(target, str):
            self.port = target

        try:
            # Try importing serial if available
            import serial  # type: ignore

            self._serial = serial.Serial(self.port, self.baudrate, timeout=1.0)
            logger.info(
                "Connected to physical serial port '%s' at %d baud", self.port, self.baudrate
            )
        except Exception:
            # Mock / virtual fallback for testing and platforms without hardware attached
            self._serial = None
            logger.info("Serial port '%s' connected via virtual serial loopback", self.port)

        self.is_connected = True
        return True

    def disconnect(self) -> None:
        """Disconnect and close serial port."""
        if self._read_task and not self._read_task.done():
            self._read_task.cancel()
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None
        self.is_connected = False
        logger.info("Serial device '%s' disconnected", self.name)

    async def simulate_incoming_line(self, line: str) -> None:
        """Simulate an incoming serial line (useful for virtual devices & testing)."""
        if not self.is_connected:
            return
        parsed_data = line
        try:
            parsed_data = json.loads(line)
        except Exception:
            pass

        self._last_observation = parsed_data
        inp = DriverInput(
            raw_data=parsed_data,
            timestamp=time.time(),
            source_id=self.name,
            modality=DriverModality.STRUCTURED,
        )
        await self._input_queue.put(inp)

    def get_inputs(self) -> DriverInput:
        return DriverInput(
            raw_data=self._last_observation,
            timestamp=time.time(),
            source_id=self.name,
            modality=DriverModality.STRUCTURED,
        )

    async def stream_inputs(self) -> Any:
        """Yield observations from queue or physical serial line."""
        while self.is_connected:
            try:
                # If physical serial is active and has bytes waiting
                if self._serial is not None and getattr(self._serial, "in_waiting", 0) > 0:

                    def _read_line() -> str:
                        try:
                            return self._serial.readline().decode("utf-8", errors="replace").strip()
                        except Exception:
                            return ""

                    line = await asyncio.to_thread(_read_line)
                    if line:
                        await self.simulate_incoming_line(line)

                inp = await asyncio.wait_for(self._input_queue.get(), timeout=0.5)
                yield inp
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def get_action_list(self, inputs: DriverInput) -> list[DriverAction]:
        if not self.descriptor or not self.descriptor.action_schema:
            return []
        return [
            DriverAction(
                action_id=s.get("action_id", s.get("name")),
                semantic_intent=s.get("name", ""),
                parameters=s.get("parameters", {}),
            )
            for s in self.descriptor.action_schema
        ]

    def send_output(self, action: DriverAction) -> Any:
        cmd_str = json.dumps({"action": action.action_id, "params": action.parameters}) + "\n"
        if self._serial is not None:
            try:
                self._serial.write(cmd_str.encode("utf-8"))
            except Exception as e:
                logger.error("Failed to write to serial port: %s", e)
        return {"status": "sent", "command": cmd_str.strip()}

    def process_feedback(self, raw_result: Any) -> DriverFeedback:
        return DriverFeedback(
            success=True,
            raw_response=raw_result,
            info={"port": self.port},
        )
