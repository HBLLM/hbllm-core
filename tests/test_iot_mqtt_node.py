"""Tests for MqttIoTNode — Home Automation Bridge."""

import pytest
import asyncio

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.actions.iot_mqtt_node import MqttIoTNode, DeviceState, DEVICE_TYPES


# ── Device State Tests ───────────────────────────────────────────────────────

def test_device_state_creation():
    dev = DeviceState(id="light_1", name="Living Room Light", type="light", room="living_room")
    assert dev.id == "light_1"
    assert dev.type == "light"
    assert dev.room == "living_room"


def test_device_state_to_dict():
    dev = DeviceState(id="sensor_1", name="Temp Sensor", type="sensor")
    d = dev.to_dict()
    assert d["id"] == "sensor_1"
    assert "last_seen" in d


def test_device_types_defined():
    assert "light" in DEVICE_TYPES
    assert "thermostat" in DEVICE_TYPES
    assert "on" in DEVICE_TYPES["light"]["actions"]


# ── Node Lifecycle Tests ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_iot_node_starts_and_stops():
    bus = InProcessBus()
    await bus.start()

    node = MqttIoTNode(node_id="iot_test")
    await node.start(bus)
    assert node.node_id == "iot_test"
    assert len(node.devices) == 0

    await node.stop()
    await bus.stop()


# ── Device Registration ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_register_device():
    bus = InProcessBus()
    await bus.start()

    node = MqttIoTNode(node_id="iot_reg")
    await node.start(bus)

    dev = node.register_device("light_1", "Living Room", device_type="light", room="living_room")
    assert dev.id == "light_1"
    assert "light_1" in node.devices

    await node.stop()
    await bus.stop()


# ── Command Execution ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_command():
    bus = InProcessBus()
    await bus.start()

    events = []
    await bus.subscribe("iot.event", lambda msg: events.append(msg))

    node = MqttIoTNode(node_id="iot_cmd")
    await node.start(bus)

    node.register_device("light_1", "Test Light", device_type="light", room="bedroom")

    msg = Message(
        type=MessageType.QUERY,
        source_node_id="brain",
        topic="iot.command",
        payload={"device_id": "light_1", "action": "on", "params": {"brightness": 80}},
    )
    await node._handle_command(msg)
    await asyncio.sleep(0.2)

    assert len(events) == 1
    assert events[0].payload["success"] is True
    assert events[0].payload["action"] == "on"

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_command_unknown_device():
    bus = InProcessBus()
    await bus.start()

    node = MqttIoTNode(node_id="iot_unknown")
    await node.start(bus)

    msg = Message(
        type=MessageType.QUERY,
        source_node_id="brain",
        topic="iot.command",
        payload={"device_id": "nonexistent", "action": "on"},
    )
    result = await node._handle_command(msg)
    assert result is not None
    assert result.type == MessageType.ERROR

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_command_invalid_action():
    bus = InProcessBus()
    await bus.start()

    node = MqttIoTNode(node_id="iot_invalid_action")
    await node.start(bus)
    node.register_device("light_1", "Test", device_type="light")

    msg = Message(
        type=MessageType.QUERY,
        source_node_id="brain",
        topic="iot.command",
        payload={"device_id": "light_1", "action": "explode"},
    )
    result = await node._handle_command(msg)
    assert result is not None
    assert result.type == MessageType.ERROR

    await node.stop()
    await bus.stop()


# ── Device Query ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_query_all_devices():
    bus = InProcessBus()
    await bus.start()

    responses = []
    await bus.subscribe("iot.query.response", lambda msg: responses.append(msg))

    node = MqttIoTNode(node_id="iot_query")
    await node.start(bus)
    node.register_device("light_1", "Light", device_type="light", room="bedroom")
    node.register_device("thermo_1", "Thermostat", device_type="thermostat", room="living")

    msg = Message(
        type=MessageType.QUERY,
        source_node_id="brain",
        topic="iot.query",
        payload={"type": "all"},
    )
    await node._handle_query(msg)
    await asyncio.sleep(0.2)

    assert len(responses) == 1
    assert responses[0].payload["count"] == 2

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_query_by_room():
    bus = InProcessBus()
    await bus.start()

    responses = []
    await bus.subscribe("iot.query.response", lambda msg: responses.append(msg))

    node = MqttIoTNode(node_id="iot_room_q")
    await node.start(bus)
    node.register_device("l1", "L1", device_type="light", room="bedroom")
    node.register_device("l2", "L2", device_type="light", room="kitchen")
    node.register_device("l3", "L3", device_type="light", room="bedroom")

    msg = Message(
        type=MessageType.QUERY,
        source_node_id="brain",
        topic="iot.query",
        payload={"type": "room", "room": "bedroom"},
    )
    await node._handle_query(msg)
    await asyncio.sleep(0.2)

    assert responses[0].payload["count"] == 2

    await node.stop()
    await bus.stop()


# ── Scene Management ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_register_and_activate_scene():
    bus = InProcessBus()
    await bus.start()

    events = []
    await bus.subscribe("iot.event", lambda msg: events.append(msg))

    node = MqttIoTNode(node_id="iot_scene")
    await node.start(bus)
    node.register_device("light_1", "Light", device_type="light")
    node.register_device("blinds_1", "Blinds", device_type="blinds")

    node.register_scene("goodnight", [
        {"device_id": "light_1", "action": "off"},
        {"device_id": "blinds_1", "action": "close"},
    ])

    msg = Message(
        type=MessageType.QUERY,
        source_node_id="brain",
        topic="iot.scene",
        payload={"scene": "goodnight"},
    )
    await node._handle_scene(msg)
    await asyncio.sleep(0.3)

    assert len(events) == 2  # Two commands executed

    await node.stop()
    await bus.stop()
