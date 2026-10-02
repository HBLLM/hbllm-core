"""Driver Manager Cognitive Node.

Integrates DriverManager directly into the HBLLM MessageBus:
- Subscribes to actions targeted at drivers/peripherals.
- Ingests streaming driver inputs and broadcasts them to the Perception/Reality bus and AttentionSystem.
- Emits device attachment and detachment events for dynamic tool mounting.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.drivers.base import (
    BaseDriver,
    DriverAction,
    DriverFeedback,
    DriverInput,
)
from hbllm.drivers.manager import DriverManager
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import DeviceTier, Node, NodeType

logger = logging.getLogger(__name__)


class DriverManagerNode(Node):
    """MessageBus Node wrapping DriverManager.

    Acts as the 'Thalamic Routing Hub' / 'USB Host Controller' for the Brain:
    - Auto-announces connected drivers via 'device.attached'
    - Broadcasts observations via 'perception.driver.input' & 'reality.event'
    - Routes incoming brain commands ('action.driver.dispatch') to physical/virtual devices
    """

    def __init__(
        self,
        node_id: str = "driver_manager",
        driver_manager: DriverManager | None = None,
        device_tier: DeviceTier = DeviceTier.SERVER,
        tool_registry: Any | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.GATEWAY,
            capabilities=[
                "driver_management",
                "hardware_io",
                "mimo_perception",
                "device_hotplug",
            ],
            device_tier=device_tier,
        )
        self.driver_manager = driver_manager or DriverManager()
        self.tool_registry = tool_registry
        self._mounted_tools: dict[str, list[str]] = {}

    async def on_start(self) -> None:
        """Hook DriverManager callbacks to bus publications and subscribe to action topics."""
        # Connect callbacks from DriverManager
        self.driver_manager.add_on_attached_callback(self._on_driver_attached)
        self.driver_manager.add_on_detached_callback(self._on_driver_detached)
        self.driver_manager.add_on_input_callback(self._on_driver_input)

        # Bus subscriptions
        await self.bus.subscribe("action.driver.dispatch", self._handle_action_dispatch)
        await self.bus.subscribe("device.hotplug.attach", self._handle_hotplug_attach)
        await self.bus.subscribe("device.hotplug.detach", self._handle_hotplug_detach)

        # Start streaming observations for any active drivers
        await self.driver_manager.start_streaming()
        logger.info("DriverManagerNode started and listening on bus.")

    async def on_stop(self) -> None:
        """Stop background streaming tasks when node shuts down."""
        await self.driver_manager.stop_streaming()
        logger.info("DriverManagerNode stopped.")

    async def handle_message(self, message: Message) -> Message | None:
        """Direct message handler required by Node ABC."""
        if message.topic == "action.driver.dispatch":
            await self._handle_action_dispatch(message)
        return None

    # ── Internal Event Handlers ───────────────────────────────────────────

    async def _on_driver_attached(self, driver: BaseDriver) -> None:
        """Broadcast device attachment and dynamically mount its actions into ToolRegistry."""
        if not self._running or not self._bus:
            return
        desc = driver.descriptor.to_dict() if driver.descriptor else {}
        msg = Message(
            type=MessageType.EVENT,
            source_node_id=self.node_id,
            target_node_id="*",
            topic="device.attached",
            payload={
                "device_id": driver.name,
                "descriptor": desc,
            },
        )
        await self.publish("device.attached", msg)
        logger.info("Broadcasted device.attached for '%s'", driver.name)

        # Dynamically mount tools into ToolRegistry if present
        if self.tool_registry is not None and driver.descriptor and driver.descriptor.action_schema:
            mounted = []
            for schema in driver.descriptor.action_schema:
                action_name = schema.get("name") or str(schema.get("action_id", "act"))
                clean_name = f"device_{driver.name}_{action_name}".replace(".", "_").replace(
                    "-", "_"
                )
                desc_text = (
                    schema.get("description")
                    or f"Execute action '{action_name}' on device '{driver.name}'"
                )
                params = schema.get("parameters") or {}

                act_id = schema.get("action_id") or action_name
                target_dev = driver.name

                def _build_handler(dev_name: str, a_id: Any, t_name: str):
                    async def _handler(**kwargs: Any) -> Any:
                        import time

                        from hbllm.actions.tool_registry import ToolResult

                        start_t = time.monotonic()
                        act = DriverAction(
                            action_id=a_id, semantic_intent=t_name, parameters=kwargs
                        )
                        fb = await self.driver_manager.dispatch_action_async(dev_name, act)
                        duration_ms = (time.monotonic() - start_t) * 1000
                        return ToolResult(
                            tool=t_name,
                            success=fb.success,
                            output=str(fb.raw_response or fb.info or "Action completed"),
                            duration_ms=duration_ms,
                        )

                    return _handler

                handler = _build_handler(target_dev, act_id, clean_name)
                self.tool_registry.register(clean_name, desc_text, handler, parameters=params)
                mounted.append(clean_name)

            self._mounted_tools[driver.name] = mounted
            logger.info(
                "Dynamically mounted %d tools for device '%s': %s",
                len(mounted),
                driver.name,
                mounted,
            )

    async def _on_driver_detached(self, name: str) -> None:
        """Broadcast device detachment and unmount its tools from ToolRegistry."""
        if not self._running or not self._bus:
            return
        msg = Message(
            type=MessageType.EVENT,
            source_node_id=self.node_id,
            target_node_id="*",
            topic="device.detached",
            payload={"device_id": name},
        )
        await self.publish("device.detached", msg)
        logger.info("Broadcasted device.detached for '%s'", name)

        if self.tool_registry is not None and name in self._mounted_tools:
            for tool_name in self._mounted_tools.pop(name, []):
                self.tool_registry.unregister(tool_name)
            logger.info("Unmounted tools for detached device '%s'", name)

    async def _on_driver_input(self, driver_input: DriverInput) -> None:
        """Route raw sensor observation into perception and world-state topics."""
        if not self._running or not self._bus:
            return

        # 1. Perception topic for cognitive sensory ingestion
        msg = Message(
            type=MessageType.EVENT,
            source_node_id=self.node_id,
            target_node_id="*",
            topic="perception.driver.input",
            payload={
                "source_id": driver_input.source_id,
                "modality": (
                    driver_input.modality.value
                    if hasattr(driver_input.modality, "value")
                    else str(driver_input.modality)
                ),
                "raw_data": driver_input.raw_data,
                "timestamp": driver_input.timestamp,
                "metadata": driver_input.metadata,
            },
        )
        await self.publish("perception.driver.input", msg)

        # 2. Reality bus / world state update
        reality_msg = Message(
            type=MessageType.EVENT,
            source_node_id=self.node_id,
            target_node_id="world_state",
            topic="reality.event",
            payload={
                "source": driver_input.source_id,
                "category": "driver",
                "state": driver_input.raw_data,
                "timestamp": driver_input.timestamp,
            },
        )
        await self.publish("reality.event", reality_msg)

    async def _handle_action_dispatch(self, message: Message) -> None:
        """Execute motor command from cognitive brain to physical/virtual device."""
        driver_name = message.payload.get("driver_name") or message.payload.get("device_id")
        action_payload = message.payload.get("action")
        if not driver_name or not action_payload:
            logger.warning("Invalid action.driver.dispatch payload: %s", message.payload)
            return

        action = DriverAction(
            action_id=action_payload.get("action_id"),
            semantic_intent=action_payload.get("semantic_intent", ""),
            parameters=action_payload.get("parameters", {}),
            confidence=action_payload.get("confidence", 1.0),
        )

        try:
            feedback: DriverFeedback = await self.driver_manager.dispatch_action_async(
                driver_name, action
            )
            reply_topic = message.payload.get("reply_topic") or "action.driver.feedback"
            feedback_msg = Message(
                type=MessageType.FEEDBACK,
                source_node_id=self.node_id,
                target_node_id=message.source_node_id,
                topic=reply_topic,
                correlation_id=message.id,
                payload={
                    "driver_name": driver_name,
                    "action_id": action.action_id,
                    "success": feedback.success,
                    "reward": feedback.reward,
                    "terminated": feedback.terminated,
                    "causal_delta": feedback.causal_delta,
                    "info": feedback.info,
                },
            )
            await self.publish(reply_topic, feedback_msg)
        except Exception as e:
            logger.error("Failed to execute action on driver '%s': %s", driver_name, e)

    async def _handle_hotplug_attach(self, message: Message) -> None:
        """Handle request to dynamically attach a registered driver."""
        driver_name = message.payload.get("name")
        target = message.payload.get("target")
        if driver_name and driver_name in self.driver_manager.list_drivers():
            driver = self.driver_manager.get_driver(driver_name)
            await self.driver_manager.attach_driver(driver, target)

    async def _handle_hotplug_detach(self, message: Message) -> None:
        """Handle request to dynamically detach a registered driver."""
        driver_name = message.payload.get("name")
        if driver_name:
            await self.driver_manager.detach_driver(driver_name)
