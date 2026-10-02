"""Network Device Driver for external smart peripherals.

Provides a plug-and-play network bridge for IoT microcontrollers (ESP32, Raspberry Pi),
mobile apps, or remote agents connecting over WebSocket / TCP / HTTP.
"""

from __future__ import annotations

import asyncio
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


class NetworkDeviceDriver(BaseDriver):
    """Driver representing a remote network-connected device or sensor."""

    def __init__(
        self,
        name: str,
        descriptor: SynapticDeviceDescriptor | None = None,
        capabilities: set[DriverCapability] | None = None,
    ) -> None:
        caps = capabilities or {
            DriverCapability.STREAMING_OBSERVATIONS,
            DriverCapability.ASYNC_EVENT_DRIVEN,
            DriverCapability.DISCRETE_ACTIONS,
        }
        desc = descriptor or SynapticDeviceDescriptor(
            device_id=name,
            device_type="bidirectional",
            modalities=[DriverModality.STRUCTURED],
            stream_type=DriverStreamType.CONTINUOUS_STREAM,
        )
        super().__init__(name=name, capabilities=caps, descriptor=desc)
        self._target: Any = None
        self._input_queue: asyncio.Queue[DriverInput | None] = asyncio.Queue()
        self._action_sink: Any = None
        self._last_observation: Any = None

    def connect(self, target: Any) -> bool:
        """Synchronous connect fallback."""
        self._target = target
        self.is_connected = True
        logger.info("NetworkDeviceDriver '%s' connected to %s", self.name, target)
        return True

    def disconnect(self) -> None:
        """Tear down network connection."""
        self.is_connected = False
        self._target = None
        try:
            self._input_queue.put_nowait(None)
        except Exception:
            pass
        logger.info("NetworkDeviceDriver '%s' disconnected", self.name)

    def set_action_sink(self, sink: Any) -> None:
        """Register a callable or websocket sender to transmit outbound actions to device."""
        self._action_sink = sink

    async def ingest_network_payload(
        self, raw_payload: Any, metadata: dict[str, Any] | None = None
    ) -> None:
        """Feed an incoming network observation packet directly into the driver's queue."""
        if not self.is_connected:
            return

        now = time.time()
        self._last_observation = raw_payload
        modality = (
            self.descriptor.modalities[0]
            if self.descriptor and self.descriptor.modalities
            else DriverModality.STRUCTURED
        )
        inp = DriverInput(
            raw_data=raw_payload,
            timestamp=now,
            metadata=metadata or {},
            source_id=self.name,
            modality=modality,
        )
        await self._input_queue.put(inp)

    def get_inputs(self) -> DriverInput:
        """Return the latest cached observation."""
        modality = (
            self.descriptor.modalities[0]
            if self.descriptor and self.descriptor.modalities
            else DriverModality.STRUCTURED
        )
        return DriverInput(
            raw_data=self._last_observation,
            timestamp=time.time(),
            source_id=self.name,
            modality=modality,
        )

    async def stream_inputs(self) -> Any:
        """Asynchronously stream incoming packets from the network queue."""
        while self.is_connected:
            try:
                inp = await asyncio.wait_for(self._input_queue.get(), timeout=0.2)
                if inp is None:
                    break
                yield inp
            except asyncio.TimeoutError:
                continue

    def get_action_list(self, inputs: DriverInput) -> list[DriverAction]:
        """Return declared actions from the SynapticDeviceDescriptor schema."""
        if not self.descriptor or not self.descriptor.action_schema:
            return []

        actions: list[DriverAction] = []
        for schema in self.descriptor.action_schema:
            act_id = schema.get("action_id") or schema.get("name")
            actions.append(
                DriverAction(
                    action_id=act_id,
                    semantic_intent=schema.get("name", str(act_id)),
                    parameters=schema.get("parameters", {}),
                )
            )
        return actions

    def send_output(self, action: DriverAction) -> Any:
        """Dispatch action to the registered action sink."""
        if self._action_sink is not None:
            if callable(self._action_sink):
                return self._action_sink(action)
            elif hasattr(self._action_sink, "send"):
                return self._action_sink.send(action)
        return {"status": "dispatched", "action": action.action_id}

    async def handle_action(self, action: DriverAction) -> DriverFeedback:
        """Asynchronously dispatch action across network socket."""
        raw_res = None
        if self._action_sink is not None:
            if asyncio.iscoroutinefunction(self._action_sink):
                raw_res = await self._action_sink(action)
            elif callable(self._action_sink):
                raw_res = self._action_sink(action)
            elif hasattr(self._action_sink, "send"):
                send_method = self._action_sink.send
                if asyncio.iscoroutinefunction(send_method):
                    raw_res = await send_method(action)
                else:
                    raw_res = send_method(action)

        return self.process_feedback(raw_res or {"status": "ok", "action_id": action.action_id})

    def process_feedback(self, raw_result: Any) -> DriverFeedback:
        """Normalize response from network target."""
        success = True
        if isinstance(raw_result, dict):
            success = raw_result.get("success", True) and raw_result.get("status") != "error"

        return DriverFeedback(
            success=success,
            raw_response=raw_result,
            info={"network_target": str(self._target)},
        )
