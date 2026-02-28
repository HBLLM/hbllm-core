"""
IoT/MQTT Action Node — Home Automation Bridge.

Bridges the HBLLM cognitive architecture to IoT devices via MQTT.
Enables the brain to:
  - Control smart home devices (lights, thermostats, locks, sensors)
  - Process sensor readings and trigger automated responses
  - Learn user routines and adapt behavior over time
  - Integrate with Home Assistant, Zigbee2MQTT, Tuya, etc.

Requires: pip install paho-mqtt (optional dependency)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)

# ── Device Categories ────────────────────────────────────────────────────────

DEVICE_TYPES = {
    "light": {"actions": ["on", "off", "brightness", "color", "toggle"]},
    "thermostat": {"actions": ["set_temp", "mode", "schedule"]},
    "lock": {"actions": ["lock", "unlock", "status"]},
    "sensor": {"actions": ["read", "subscribe"]},
    "switch": {"actions": ["on", "off", "toggle"]},
    "camera": {"actions": ["snapshot", "stream", "motion_detect"]},
    "speaker": {"actions": ["play", "stop", "volume", "tts"]},
    "blinds": {"actions": ["open", "close", "position"]},
    "fan": {"actions": ["on", "off", "speed"]},
    "plug": {"actions": ["on", "off", "power_reading"]},
}


class DeviceState:
    """Tracks the state of a discovered IoT device."""

    __slots__ = ("id", "name", "type", "room", "state", "last_seen", "attributes")

    def __init__(
        self, id: str, name: str, type: str = "unknown",
        room: str = "unknown", state: dict | None = None,
    ):
        self.id = id
        self.name = name
        self.type = type
        self.room = room
        self.state = state or {}
        self.last_seen = time.time()
        self.attributes: dict[str, Any] = {}

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "type": self.type,
            "room": self.room, "state": self.state,
            "last_seen": self.last_seen, "attributes": self.attributes,
        }


class MqttIoTNode(Node):
    """
    Action node that bridges HBLLM to IoT devices via MQTT.

    Subscribes to:
      - iot.command      → Execute device commands (from brain decisions)
      - iot.query        → Query device states
      - iot.scene        → Activate scenes (multi-device)
      - iot.routine      → Trigger learned routines

    Publishes:
      - iot.event        → Sensor readings, state changes
      - iot.discovery    → New devices found
      - sensory.input    → Contextual awareness for brain

    MQTT Topics:
      - hbllm/devices/+/state     → Subscribe to device states
      - hbllm/devices/+/command   → Publish device commands
      - hbllm/sensors/+/data      → Subscribe to sensor data
      - homeassistant/#           → Home Assistant auto-discovery
      - zigbee2mqtt/#             → Zigbee2MQTT integration
    """

    def __init__(
        self,
        node_id: str = "iot_mqtt",
        broker_host: str = "localhost",
        broker_port: int = 1883,
        username: str | None = None,
        password: str | None = None,
        topic_prefix: str = "hbllm",
        ha_discovery: bool = True,
    ):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.DOMAIN_MODULE,
            capabilities=["iot", "mqtt", "home_automation", "sensors"],
        )
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.username = username
        self.password = password
        self.topic_prefix = topic_prefix
        self.ha_discovery = ha_discovery

        # Device registry
        self.devices: dict[str, DeviceState] = {}
        self.scenes: dict[str, list[dict]] = {}
        self.sensor_callbacks: dict[str, list[Callable]] = {}

        # MQTT client (lazy loaded)
        self._mqtt_client = None
        self._mqtt_loop_task = None
        self._connected = False

    async def on_start(self) -> None:
        """Subscribe to brain messages and connect to MQTT broker."""
        logger.info("Starting MqttIoTNode — Home Automation Bridge")

        # Subscribe to brain messages
        await self.bus.subscribe("iot.command", self._handle_command)
        await self.bus.subscribe("iot.query", self._handle_query)
        await self.bus.subscribe("iot.scene", self._handle_scene)

        # Connect to MQTT broker
        await self._connect_mqtt()

    async def on_stop(self) -> None:
        """Disconnect from MQTT broker."""
        logger.info("Stopping MqttIoTNode")
        if self._mqtt_client and self._connected:
            self._mqtt_client.disconnect()
        if self._mqtt_loop_task:
            self._mqtt_loop_task.cancel()

    async def handle_message(self, message: Message) -> Message | None:
        """Route messages to appropriate handlers."""
        topic = message.topic
        if topic == "iot.command":
            return await self._handle_command(message)
        elif topic == "iot.query":
            return await self._handle_query(message)
        return None

    # ── MQTT Connection ──────────────────────────────────────────────────

    async def _connect_mqtt(self) -> None:
        """Connect to MQTT broker with auto-reconnect."""
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            logger.warning(
                "paho-mqtt not installed. IoT node running in simulation mode. "
                "Install: pip install paho-mqtt"
            )
            return

        client = mqtt.Client(
            client_id=f"hbllm-{self.node_id}",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )

        if self.username:
            client.username_pw_set(self.username, self.password)

        client.on_connect = self._on_mqtt_connect
        client.on_message = self._on_mqtt_message
        client.on_disconnect = self._on_mqtt_disconnect

        try:
            client.connect_async(self.broker_host, self.broker_port)
            self._mqtt_client = client
            self._mqtt_loop_task = asyncio.create_task(
                asyncio.to_thread(client.loop_forever)
            )
            logger.info("MQTT connecting to %s:%d", self.broker_host, self.broker_port)
        except Exception as e:
            logger.error("MQTT connection failed: %s", e)

    def _on_mqtt_connect(self, client, userdata, flags, rc, properties=None) -> None:
        """Called when MQTT connection is established."""
        self._connected = True
        logger.info("MQTT connected to broker (rc=%s)", rc)

        # Subscribe to device state topics
        client.subscribe(f"{self.topic_prefix}/devices/+/state")
        client.subscribe(f"{self.topic_prefix}/sensors/+/data")

        # Home Assistant auto-discovery
        if self.ha_discovery:
            client.subscribe("homeassistant/#")

        # Zigbee2MQTT
        client.subscribe("zigbee2mqtt/+")

    def _on_mqtt_message(self, client, userdata, msg) -> None:
        """Process incoming MQTT messages."""
        topic = msg.topic
        try:
            payload = json.loads(msg.payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {"raw": msg.payload.decode(errors="replace")}

        # Route based on topic pattern
        if "/devices/" in topic and "/state" in topic:
            device_id = topic.split("/")[-2]
            self._update_device_state(device_id, payload)
        elif "/sensors/" in topic:
            sensor_id = topic.split("/")[-2]
            self._process_sensor_data(sensor_id, payload)
        elif topic.startswith("homeassistant/"):
            self._handle_ha_discovery(topic, payload)
        elif topic.startswith("zigbee2mqtt/"):
            self._handle_zigbee_message(topic, payload)

    def _on_mqtt_disconnect(self, client, userdata, flags, rc, properties=None) -> None:
        self._connected = False
        logger.warning("MQTT disconnected (rc=%s). Will auto-reconnect.", rc)

    # ── Device Management ────────────────────────────────────────────────

    def _update_device_state(self, device_id: str, state: dict) -> None:
        """Update local device registry with new state."""
        if device_id in self.devices:
            self.devices[device_id].state.update(state)
            self.devices[device_id].last_seen = time.time()
        else:
            # Auto-discover new device
            dev = DeviceState(
                id=device_id,
                name=state.get("friendly_name", device_id),
                type=state.get("type", "unknown"),
                room=state.get("room", "unknown"),
                state=state,
            )
            self.devices[device_id] = dev
            logger.info("Discovered new IoT device: %s (%s)", dev.name, dev.type)

            # Notify brain of new device
            asyncio.create_task(self._publish_discovery(dev))

    def _process_sensor_data(self, sensor_id: str, data: dict) -> None:
        """Process sensor readings and forward to brain."""
        self._update_device_state(sensor_id, {**data, "type": "sensor"})

        # Forward significant sensor events to the brain
        asyncio.create_task(self._publish_sensor_event(sensor_id, data))

    def _handle_ha_discovery(self, topic: str, payload: dict) -> None:
        """Process Home Assistant MQTT auto-discovery messages."""
        parts = topic.split("/")
        if len(parts) >= 4 and parts[-1] == "config":
            component = parts[1]  # e.g., light, switch, sensor
            device_id = parts[2]
            name = payload.get("name", device_id)
            dev = DeviceState(
                id=device_id, name=name, type=component,
                room=payload.get("room", "unknown"),
            )
            self.devices[device_id] = dev
            logger.info("HA discovery: %s (%s)", name, component)

    def _handle_zigbee_message(self, topic: str, payload: dict) -> None:
        """Process Zigbee2MQTT device messages."""
        device_name = topic.split("/")[-1]
        if device_name not in ("bridge", ""):
            self._update_device_state(device_name, payload)

    # ── Command Execution ────────────────────────────────────────────────

    async def _handle_command(self, message: Message) -> Message | None:
        """Execute an IoT device command from the brain."""
        payload = message.payload
        device_id = payload.get("device_id", "")
        action = payload.get("action", "")
        params = payload.get("params", {})

        logger.info("IoT command: %s → %s %s", device_id, action, params)

        # Validate device exists
        device = self.devices.get(device_id)
        if not device:
            return self._error_response(message, f"Unknown device: {device_id}")

        # Validate action
        valid_actions = DEVICE_TYPES.get(device.type, {}).get("actions", [])
        if valid_actions and action not in valid_actions:
            return self._error_response(
                message, f"Invalid action '{action}' for {device.type}. Valid: {valid_actions}"
            )

        # Send MQTT command
        mqtt_payload = {"action": action, **params}

        if self._mqtt_client and self._connected:
            topic = f"{self.topic_prefix}/devices/{device_id}/command"
            self._mqtt_client.publish(topic, json.dumps(mqtt_payload))
            logger.info("MQTT published: %s → %s", topic, mqtt_payload)
        else:
            logger.info("IoT command (simulated): %s → %s", device_id, mqtt_payload)

        # Update local state optimistically
        device.state.update(params)
        device.last_seen = time.time()

        # Acknowledge to brain
        ack = Message(
            type=MessageType.EVENT,
            source_node_id=self.node_id,
            tenant_id=message.tenant_id,
            session_id=message.session_id,
            topic="iot.event",
            payload={
                "event": "command_executed",
                "device_id": device_id,
                "action": action,
                "params": params,
                "success": True,
            },
            correlation_id=message.correlation_id,
        )
        await self.bus.publish("iot.event", ack)
        return None

    async def _handle_query(self, message: Message) -> Message | None:
        """Query device states for the brain."""
        payload = message.payload
        query_type = payload.get("type", "all")

        if query_type == "all":
            devices = {k: v.to_dict() for k, v in self.devices.items()}
        elif query_type == "room":
            room = payload.get("room", "")
            devices = {
                k: v.to_dict() for k, v in self.devices.items()
                if v.room == room
            }
        elif query_type == "device":
            device_id = payload.get("device_id", "")
            dev = self.devices.get(device_id)
            devices = {device_id: dev.to_dict()} if dev else {}
        elif query_type == "type":
            dev_type = payload.get("device_type", "")
            devices = {
                k: v.to_dict() for k, v in self.devices.items()
                if v.type == dev_type
            }
        else:
            devices = {}

        response = Message(
            type=MessageType.RESPONSE,
            source_node_id=self.node_id,
            tenant_id=message.tenant_id,
            session_id=message.session_id,
            topic="iot.query.response",
            payload={"devices": devices, "count": len(devices)},
            correlation_id=message.correlation_id,
        )
        await self.bus.publish("iot.query.response", response)
        return None

    async def _handle_scene(self, message: Message) -> Message | None:
        """Activate a multi-device scene."""
        payload = message.payload
        scene_name = payload.get("scene", "")
        commands = payload.get("commands", [])

        if scene_name and scene_name in self.scenes:
            commands = self.scenes[scene_name]

        if not commands:
            return self._error_response(message, f"Unknown scene: {scene_name}")

        logger.info("Activating scene '%s' with %d commands", scene_name, len(commands))

        for cmd in commands:
            cmd_msg = Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                tenant_id=message.tenant_id,
                session_id=message.session_id,
                topic="iot.command",
                payload=cmd,
                correlation_id=message.correlation_id,
            )
            await self._handle_command(cmd_msg)

        return None

    # ── Scene Management ─────────────────────────────────────────────────

    def register_scene(self, name: str, commands: list[dict]) -> None:
        """Register a multi-device scene."""
        self.scenes[name] = commands
        logger.info("Registered scene '%s' with %d commands", name, len(commands))

    def register_device(
        self, device_id: str, name: str, device_type: str = "unknown",
        room: str = "unknown",
    ) -> DeviceState:
        """Manually register a device."""
        dev = DeviceState(id=device_id, name=name, type=device_type, room=room)
        self.devices[device_id] = dev
        return dev

    # ── Internal Helpers ─────────────────────────────────────────────────

    async def _publish_discovery(self, device: DeviceState) -> None:
        """Notify brain of newly discovered device."""
        msg = Message(
            type=MessageType.EVENT,
            source_node_id=self.node_id,
            topic="iot.discovery",
            payload={"event": "new_device", "device": device.to_dict()},
        )
        await self.bus.publish("iot.discovery", msg)

    async def _publish_sensor_event(self, sensor_id: str, data: dict) -> None:
        """Forward sensor data to brain for contextual awareness."""
        msg = Message(
            type=MessageType.EVENT,
            source_node_id=self.node_id,
            topic="sensory.input",
            payload={
                "source": "iot_sensor",
                "sensor_id": sensor_id,
                "data": data,
            },
        )
        await self.bus.publish("sensory.input", msg)

    def _error_response(self, original: Message, error: str) -> Message:
        """Create an error response message."""
        return Message(
            type=MessageType.ERROR,
            source_node_id=self.node_id,
            tenant_id=original.tenant_id,
            session_id=original.session_id,
            topic="iot.error",
            payload={"error": error},
            correlation_id=original.correlation_id,
        )
