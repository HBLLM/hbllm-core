"""Tests for Ros2Node — runs entirely in simulation mode (no ROS2 needed)."""

import pytest
import asyncio

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.actions.ros2_node import Ros2Node, RobotState, ROBOT_COMMANDS


# ── Robot State Tests ────────────────────────────────────────────────────────

def test_robot_state_creation():
    r = RobotState(id="bot_1", name="TurtleBot", type="mobile")
    assert r.id == "bot_1"
    assert r.type == "mobile"
    assert r.battery == 100.0


def test_robot_state_to_dict():
    r = RobotState(id="arm_1", name="MyArm", type="arm")
    d = r.to_dict()
    assert d["id"] == "arm_1"
    assert "position" in d
    assert "battery" in d


def test_robot_commands_defined():
    assert "mobile" in ROBOT_COMMANDS
    assert "arm" in ROBOT_COMMANDS
    assert "drone" in ROBOT_COMMANDS
    assert "move" in ROBOT_COMMANDS["mobile"]
    assert "gripper" in ROBOT_COMMANDS["arm"]


# ── Node Lifecycle (no ROS2 needed) ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_ros2_node_starts_in_simulation():
    bus = InProcessBus()
    await bus.start()

    node = Ros2Node(node_id="ros2_test", ros2_enabled=False)
    await node.start(bus)

    assert node.is_simulation is True
    assert node.is_real is False
    assert len(node.robots) == 0

    await node.stop()
    await bus.stop()


# ── Robot Registration ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_register_robot():
    bus = InProcessBus()
    await bus.start()

    node = Ros2Node(node_id="ros2_reg", ros2_enabled=False)
    await node.start(bus)

    robot = node.register_robot("bot_1", "TurtleBot", "mobile")
    assert robot.id == "bot_1"
    assert "bot_1" in node.robots

    await node.stop()
    await bus.stop()


# ── Command Execution (Simulation) ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_move_command():
    bus = InProcessBus()
    await bus.start()

    events = []
    await bus.subscribe("ros2.event", lambda msg: events.append(msg))

    node = Ros2Node(node_id="ros2_move", ros2_enabled=False)
    await node.start(bus)
    node.register_robot("bot_1", "Bot", "mobile")

    msg = Message(
        type=MessageType.QUERY,
        source_node_id="brain",
        topic="ros2.command",
        payload={"robot_id": "bot_1", "command": "move", "params": {"linear_x": 0.5}},
    )
    await node._handle_command(msg)
    await asyncio.sleep(0.2)

    assert len(events) == 1
    assert events[0].payload["success"] is True
    assert events[0].payload["mode"] == "simulation"

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_execute_stop_command():
    bus = InProcessBus()
    await bus.start()

    events = []
    await bus.subscribe("ros2.event", lambda msg: events.append(msg))

    node = Ros2Node(node_id="ros2_stop", ros2_enabled=False)
    await node.start(bus)
    node.register_robot("bot_1", "Bot", "mobile")

    msg = Message(
        type=MessageType.QUERY,
        source_node_id="brain",
        topic="ros2.command",
        payload={"robot_id": "bot_1", "command": "stop"},
    )
    await node._handle_command(msg)
    await asyncio.sleep(0.2)

    assert len(events) == 1
    assert node.robots["bot_1"].status == "idle"

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_command_unknown_robot():
    bus = InProcessBus()
    await bus.start()

    node = Ros2Node(node_id="ros2_unknown", ros2_enabled=False)
    await node.start(bus)

    msg = Message(
        type=MessageType.QUERY,
        source_node_id="brain",
        topic="ros2.command",
        payload={"robot_id": "nonexistent", "command": "move"},
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

    node = Ros2Node(node_id="ros2_invalid", ros2_enabled=False)
    await node.start(bus)
    node.register_robot("bot_1", "Bot", "mobile")

    msg = Message(
        type=MessageType.QUERY,
        source_node_id="brain",
        topic="ros2.command",
        payload={"robot_id": "bot_1", "command": "fly"},
    )
    result = await node._handle_command(msg)
    assert result is not None
    assert result.type == MessageType.ERROR

    await node.stop()
    await bus.stop()


# ── Navigation ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_navigate_command():
    bus = InProcessBus()
    await bus.start()

    events = []
    await bus.subscribe("ros2.event", lambda msg: events.append(msg))

    node = Ros2Node(node_id="ros2_nav", ros2_enabled=False)
    await node.start(bus)
    node.register_robot("bot_1", "NavBot", "mobile")

    msg = Message(
        type=MessageType.QUERY,
        source_node_id="brain",
        topic="ros2.navigate",
        payload={"robot_id": "bot_1", "x": 5.0, "y": 3.0, "yaw": 1.57},
    )
    await node._handle_navigate(msg)
    await asyncio.sleep(0.2)

    assert len(events) == 1
    assert events[0].payload["event"] == "navigation_started"
    assert node.robots["bot_1"].position["x"] == 5.0

    await node.stop()
    await bus.stop()


# ── Query ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_query_all_robots():
    bus = InProcessBus()
    await bus.start()

    responses = []
    await bus.subscribe("ros2.query.response", lambda msg: responses.append(msg))

    node = Ros2Node(node_id="ros2_query", ros2_enabled=False)
    await node.start(bus)
    node.register_robot("bot_1", "Bot1", "mobile")
    node.register_robot("arm_1", "Arm1", "arm")

    msg = Message(
        type=MessageType.QUERY,
        source_node_id="brain",
        topic="ros2.query",
        payload={"type": "all"},
    )
    await node._handle_query(msg)
    await asyncio.sleep(0.2)

    assert len(responses) == 1
    assert responses[0].payload["mode"] == "simulation"
    assert len(responses[0].payload["data"]) == 2

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_query_available_commands():
    bus = InProcessBus()
    await bus.start()

    responses = []
    await bus.subscribe("ros2.query.response", lambda msg: responses.append(msg))

    node = Ros2Node(node_id="ros2_cmds", ros2_enabled=False)
    await node.start(bus)

    msg = Message(
        type=MessageType.QUERY,
        source_node_id="brain",
        topic="ros2.query",
        payload={"type": "commands"},
    )
    await node._handle_query(msg)
    await asyncio.sleep(0.2)

    commands = responses[0].payload["data"]["commands"]
    assert "mobile" in commands
    assert "arm" in commands

    await node.stop()
    await bus.stop()


# ── Behaviors ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_register_and_execute_behavior():
    bus = InProcessBus()
    await bus.start()

    events = []
    await bus.subscribe("ros2.event", lambda msg: events.append(msg))

    node = Ros2Node(node_id="ros2_behavior", ros2_enabled=False)
    await node.start(bus)
    node.register_robot("bot_1", "Bot", "mobile")

    node.register_behavior("patrol", [
        {"command": "navigate", "params": {"x": 1.0, "y": 0.0}},
        {"command": "navigate", "params": {"x": 0.0, "y": 1.0}},
    ])

    msg = Message(
        type=MessageType.QUERY,
        source_node_id="brain",
        topic="ros2.behavior",
        payload={"behavior": "patrol", "robot_id": "bot_1"},
    )
    await node._handle_behavior(msg)
    await asyncio.sleep(0.3)

    assert len(events) == 2  # Two commands executed

    await node.stop()
    await bus.stop()


# ── Command Log ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_command_log():
    bus = InProcessBus()
    await bus.start()

    node = Ros2Node(node_id="ros2_log", ros2_enabled=False)
    await node.start(bus)
    node.register_robot("bot_1", "Bot", "mobile")

    for i in range(3):
        msg = Message(
            type=MessageType.QUERY,
            source_node_id="brain",
            topic="ros2.command",
            payload={"robot_id": "bot_1", "command": "move", "params": {"linear_x": float(i)}},
        )
        await node._handle_command(msg)

    assert len(node.command_log) == 3
    assert all(c["mode"] == "simulation" for c in node.command_log)

    await node.stop()
    await bus.stop()
