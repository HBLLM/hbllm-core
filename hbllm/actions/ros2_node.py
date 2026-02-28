"""
ROS2 Robotics Action Node — Optional Brain-to-Robot Bridge.

Bridges the HBLLM cognitive architecture to ROS2-based robots.
Enables the brain to:
  - Send navigation goals (move_base, Nav2)
  - Control robot arms (MoveIt2)
  - Process sensor data (LiDAR, cameras, IMU)
  - Trigger predefined robot behaviors
  - Monitor robot state and diagnostics

OPTIONAL DEPENDENCY:
  This node does NOT require ROS2 to be installed.
  Without rclpy, it runs in **simulation mode** — all commands
  are logged and bus messages still flow for testing/dev.

  To enable real ROS2:
    1. Install ROS2 Humble/Iron: https://docs.ros.org/en/humble/Installation.html
    2. pip install rclpy  (or source /opt/ros/humble/setup.bash)
    3. Set HBLLM_ROS2_ENABLED=1

Usage without ROS2:
    node = Ros2Node(node_id="ros2")
    # Works fine — all commands simulated, bus messages flow normally

Usage with ROS2:
    export HBLLM_ROS2_ENABLED=1
    node = Ros2Node(node_id="ros2", ros2_enabled=True)
    # Now actually publishes to ROS2 topics
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)

# ── Lazy ROS2 Import ─────────────────────────────────────────────────────────
# Only import rclpy when actually needed. The node works fine without it.

_rclpy = None
_rclpy_checked = False


def _get_rclpy():
    """Lazy-load rclpy. Returns None if not installed."""
    global _rclpy, _rclpy_checked
    if not _rclpy_checked:
        try:
            import rclpy
            import rclpy.node
            from geometry_msgs.msg import Twist, PoseStamped
            from std_msgs.msg import String
            _rclpy = rclpy
            logger.info("ROS2 (rclpy) available — real robot mode enabled")
        except ImportError:
            _rclpy = None
            logger.info("ROS2 (rclpy) not installed — running in simulation mode")
        _rclpy_checked = True
    return _rclpy


# ── Robot State ──────────────────────────────────────────────────────────────

@dataclass
class RobotState:
    """Tracks state of a connected robot."""
    id: str
    name: str
    type: str = "mobile"  # mobile, arm, drone, humanoid
    status: str = "idle"  # idle, moving, executing, error
    position: dict = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0})
    orientation: dict = field(default_factory=lambda: {"roll": 0.0, "pitch": 0.0, "yaw": 0.0})
    battery: float = 100.0
    sensors: dict = field(default_factory=dict)
    last_seen: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "type": self.type,
            "status": self.status, "position": self.position,
            "orientation": self.orientation, "battery": self.battery,
            "sensors": self.sensors, "last_seen": self.last_seen,
        }


# ── Robot Commands ───────────────────────────────────────────────────────────

ROBOT_COMMANDS = {
    "mobile": {
        "move": "Send velocity command (linear_x, angular_z)",
        "navigate": "Navigate to goal pose (x, y, yaw)",
        "stop": "Emergency stop",
        "rotate": "Rotate by angle (degrees)",
        "dock": "Return to charging dock",
    },
    "arm": {
        "move_joint": "Move to joint positions",
        "move_pose": "Move end-effector to pose",
        "gripper": "Open/close gripper",
        "home": "Return to home position",
        "pick": "Pick object at location",
        "place": "Place object at location",
    },
    "drone": {
        "takeoff": "Take off to altitude",
        "land": "Land at current position",
        "move": "Fly to coordinates",
        "hover": "Hover at current position",
        "return_home": "Return to launch point",
    },
    "humanoid": {
        "walk": "Walk to position",
        "gesture": "Perform gesture",
        "speak": "Text-to-speech output",
        "look": "Look at target",
    },
}


class Ros2Node(Node):
    """
    Action node that bridges HBLLM to ROS2 robots.

    Works in two modes:
      1. **Simulation** (default) — no ROS2 needed, commands logged
      2. **Real** — publishes to actual ROS2 topics when rclpy available

    Subscribes to (HBLLM bus):
      - ros2.command     → Execute robot commands
      - ros2.navigate    → Send navigation goals
      - ros2.query       → Query robot state
      - ros2.behavior    → Trigger predefined behaviors

    Publishes to (HBLLM bus):
      - ros2.event       → Robot state changes, command completions
      - ros2.sensor      → Sensor data forwarded to brain
      - sensory.input    → Robot context for brain awareness
    """

    def __init__(
        self,
        node_id: str = "ros2",
        ros2_enabled: bool | None = None,
        ros2_node_name: str = "hbllm_brain",
        cmd_vel_topic: str = "/cmd_vel",
        nav_goal_topic: str = "/goal_pose",
    ):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.DOMAIN_MODULE,
            capabilities=["robotics", "ros2", "navigation", "manipulation"],
        )

        # Auto-detect or use explicit flag
        if ros2_enabled is None:
            ros2_enabled = os.getenv("HBLLM_ROS2_ENABLED", "0") == "1"

        self.ros2_enabled = ros2_enabled
        self.ros2_node_name = ros2_node_name
        self.cmd_vel_topic = cmd_vel_topic
        self.nav_goal_topic = nav_goal_topic

        # Robot registry
        self.robots: dict[str, RobotState] = {}
        self.behaviors: dict[str, list[dict]] = {}
        self.command_log: list[dict] = []

        # ROS2 internals (only if enabled)
        self._ros2_node = None
        self._ros2_executor = None
        self._ros2_spin_task = None
        self._publishers: dict[str, Any] = {}

    @property
    def is_real(self) -> bool:
        """Whether connected to a real ROS2 system."""
        return self._ros2_node is not None

    @property
    def is_simulation(self) -> bool:
        return not self.is_real

    async def on_start(self) -> None:
        """Subscribe to brain messages and optionally init ROS2."""
        logger.info(
            "Starting Ros2Node — %s mode",
            "REAL ROS2" if self.ros2_enabled else "SIMULATION"
        )

        # Subscribe to brain messages (always works)
        await self.bus.subscribe("ros2.command", self._handle_command)
        await self.bus.subscribe("ros2.navigate", self._handle_navigate)
        await self.bus.subscribe("ros2.query", self._handle_query)
        await self.bus.subscribe("ros2.behavior", self._handle_behavior)

        # Only init ROS2 if explicitly enabled AND rclpy is available
        if self.ros2_enabled:
            await self._init_ros2()

    async def on_stop(self) -> None:
        """Shutdown ROS2 node if running."""
        logger.info("Stopping Ros2Node")
        if self._ros2_spin_task:
            self._ros2_spin_task.cancel()
        if self._ros2_node:
            self._ros2_node.destroy_node()
        rclpy = _get_rclpy()
        if rclpy:
            try:
                rclpy.shutdown()
            except Exception:
                pass

    async def handle_message(self, message: Message) -> Message | None:
        return None

    # ── ROS2 Initialization ──────────────────────────────────────────────

    async def _init_ros2(self) -> None:
        """Initialize ROS2 node (only if rclpy available)."""
        rclpy = _get_rclpy()
        if not rclpy:
            logger.warning(
                "ROS2 enabled but rclpy not installed. "
                "Install: pip install rclpy (requires ROS2 Humble/Iron). "
                "Falling back to simulation mode."
            )
            self.ros2_enabled = False
            return

        try:
            rclpy.init()
            self._ros2_node = rclpy.node.Node(self.ros2_node_name)

            # Create publishers
            from geometry_msgs.msg import Twist, PoseStamped
            from std_msgs.msg import String

            self._publishers["cmd_vel"] = self._ros2_node.create_publisher(
                Twist, self.cmd_vel_topic, 10
            )
            self._publishers["nav_goal"] = self._ros2_node.create_publisher(
                PoseStamped, self.nav_goal_topic, 10
            )
            self._publishers["brain_cmd"] = self._ros2_node.create_publisher(
                String, "/hbllm/commands", 10
            )

            # Spin in background thread
            self._ros2_spin_task = asyncio.create_task(
                asyncio.to_thread(rclpy.spin, self._ros2_node)
            )

            logger.info("ROS2 node '%s' initialized with publishers", self.ros2_node_name)

        except Exception as e:
            logger.error("ROS2 init failed: %s. Falling back to simulation.", e)
            self.ros2_enabled = False
            self._ros2_node = None

    # ── Command Execution ────────────────────────────────────────────────

    async def _handle_command(self, message: Message) -> Message | None:
        """Execute a robot command."""
        payload = message.payload
        robot_id = payload.get("robot_id", "default")
        command = payload.get("command", "")
        params = payload.get("params", {})

        logger.info("Robot command: %s → %s %s", robot_id, command, params)

        # Get or create robot
        robot = self.robots.get(robot_id)
        if not robot:
            return self._error(message, f"Unknown robot: {robot_id}. Register first.")

        # Validate command
        valid_cmds = ROBOT_COMMANDS.get(robot.type, {})
        if valid_cmds and command not in valid_cmds:
            return self._error(
                message,
                f"Invalid command '{command}' for {robot.type}. "
                f"Valid: {list(valid_cmds.keys())}"
            )

        # Log command
        cmd_entry = {
            "robot_id": robot_id, "command": command,
            "params": params, "timestamp": time.time(),
            "mode": "real" if self.is_real else "simulation",
        }
        self.command_log.append(cmd_entry)

        # Execute
        if command == "move":
            await self._execute_move(robot, params)
        elif command == "stop":
            await self._execute_stop(robot)
        elif command == "navigate":
            await self._execute_navigate(robot, params, message)
        elif command in ("gripper", "pick", "place"):
            await self._execute_manipulation(robot, command, params)
        else:
            # Generic command — log and simulate
            robot.status = "executing"
            logger.info("[SIM] %s executing '%s' with %s", robot_id, command, params)
            robot.status = "idle"

        # Acknowledge
        ack = Message(
            type=MessageType.EVENT,
            source_node_id=self.node_id,
            tenant_id=message.tenant_id,
            session_id=message.session_id,
            topic="ros2.event",
            payload={
                "event": "command_executed",
                "robot_id": robot_id,
                "command": command,
                "mode": "real" if self.is_real else "simulation",
                "success": True,
            },
            correlation_id=message.correlation_id,
        )
        await self.bus.publish("ros2.event", ack)
        return None

    async def _execute_move(self, robot: RobotState, params: dict) -> None:
        """Send velocity command."""
        linear_x = params.get("linear_x", 0.0)
        angular_z = params.get("angular_z", 0.0)

        if self.is_real and "cmd_vel" in self._publishers:
            from geometry_msgs.msg import Twist
            twist = Twist()
            twist.linear.x = float(linear_x)
            twist.angular.z = float(angular_z)
            self._publishers["cmd_vel"].publish(twist)
            logger.info("ROS2 published Twist: linear=%.2f, angular=%.2f", linear_x, angular_z)
        else:
            logger.info("[SIM] Move: linear=%.2f, angular=%.2f", linear_x, angular_z)

        robot.status = "moving"

    async def _execute_stop(self, robot: RobotState) -> None:
        """Emergency stop."""
        await self._execute_move(robot, {"linear_x": 0.0, "angular_z": 0.0})
        robot.status = "idle"

    async def _execute_navigate(
        self, robot: RobotState, params: dict, message: Message
    ) -> None:
        """Send Nav2 goal."""
        x = params.get("x", 0.0)
        y = params.get("y", 0.0)
        yaw = params.get("yaw", 0.0)

        if self.is_real and "nav_goal" in self._publishers:
            from geometry_msgs.msg import PoseStamped
            import math
            goal = PoseStamped()
            goal.header.frame_id = "map"
            goal.pose.position.x = float(x)
            goal.pose.position.y = float(y)
            goal.pose.orientation.z = math.sin(float(yaw) / 2)
            goal.pose.orientation.w = math.cos(float(yaw) / 2)
            self._publishers["nav_goal"].publish(goal)
            logger.info("ROS2 navigation goal: (%.2f, %.2f, yaw=%.2f)", x, y, yaw)
        else:
            logger.info("[SIM] Navigate to (%.2f, %.2f, yaw=%.2f)", x, y, yaw)

        robot.status = "navigating"
        robot.position = {"x": x, "y": y, "z": 0.0}

    async def _execute_manipulation(
        self, robot: RobotState, command: str, params: dict
    ) -> None:
        """Handle arm/gripper commands."""
        if self.is_real:
            from std_msgs.msg import String
            cmd_msg = String()
            cmd_msg.data = json.dumps({"command": command, "params": params})
            if "brain_cmd" in self._publishers:
                self._publishers["brain_cmd"].publish(cmd_msg)
        else:
            logger.info("[SIM] %s: %s %s", robot.name, command, params)

        robot.status = "executing"

    # ── Navigation ───────────────────────────────────────────────────────

    async def _handle_navigate(self, message: Message) -> Message | None:
        """Shortcut for navigation goals."""
        payload = message.payload
        robot_id = payload.get("robot_id", "default")
        robot = self.robots.get(robot_id)
        if not robot:
            return self._error(message, f"Unknown robot: {robot_id}")

        await self._execute_navigate(robot, payload, message)

        ack = Message(
            type=MessageType.EVENT,
            source_node_id=self.node_id,
            tenant_id=message.tenant_id,
            session_id=message.session_id,
            topic="ros2.event",
            payload={
                "event": "navigation_started",
                "robot_id": robot_id,
                "target": {"x": payload.get("x"), "y": payload.get("y")},
            },
            correlation_id=message.correlation_id,
        )
        await self.bus.publish("ros2.event", ack)
        return None

    # ── Query ────────────────────────────────────────────────────────────

    async def _handle_query(self, message: Message) -> Message | None:
        """Query robot states."""
        payload = message.payload
        query_type = payload.get("type", "all")

        if query_type == "all":
            data = {k: v.to_dict() for k, v in self.robots.items()}
        elif query_type == "robot":
            rid = payload.get("robot_id", "")
            r = self.robots.get(rid)
            data = {rid: r.to_dict()} if r else {}
        elif query_type == "commands":
            data = {"commands": ROBOT_COMMANDS}
        elif query_type == "log":
            data = {"log": self.command_log[-20:]}  # Last 20 commands
        else:
            data = {}

        response = Message(
            type=MessageType.RESPONSE,
            source_node_id=self.node_id,
            tenant_id=message.tenant_id,
            session_id=message.session_id,
            topic="ros2.query.response",
            payload={"data": data, "mode": "real" if self.is_real else "simulation"},
            correlation_id=message.correlation_id,
        )
        await self.bus.publish("ros2.query.response", response)
        return None

    # ── Behaviors ────────────────────────────────────────────────────────

    async def _handle_behavior(self, message: Message) -> Message | None:
        """Trigger a predefined multi-step behavior."""
        payload = message.payload
        behavior_name = payload.get("behavior", "")
        robot_id = payload.get("robot_id", "default")

        commands = payload.get("commands", [])
        if behavior_name and behavior_name in self.behaviors:
            commands = self.behaviors[behavior_name]

        if not commands:
            return self._error(message, f"Unknown behavior: {behavior_name}")

        logger.info("Executing behavior '%s' (%d steps)", behavior_name, len(commands))
        for cmd in commands:
            cmd["robot_id"] = cmd.get("robot_id", robot_id)
            cmd_msg = Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                tenant_id=message.tenant_id,
                session_id=message.session_id,
                topic="ros2.command",
                payload=cmd,
                correlation_id=message.correlation_id,
            )
            await self._handle_command(cmd_msg)

        return None

    # ── Registration ─────────────────────────────────────────────────────

    def register_robot(
        self, robot_id: str, name: str, robot_type: str = "mobile",
    ) -> RobotState:
        """Register a robot in the brain's registry."""
        robot = RobotState(id=robot_id, name=name, type=robot_type)
        self.robots[robot_id] = robot
        logger.info("Registered robot: %s (%s) [%s]", name, robot_type,
                     "real" if self.is_real else "simulation")
        return robot

    def register_behavior(self, name: str, commands: list[dict]) -> None:
        """Register a multi-step behavior sequence."""
        self.behaviors[name] = commands

    # ── Helpers ───────────────────────────────────────────────────────────

    def _error(self, original: Message, error: str) -> Message:
        return Message(
            type=MessageType.ERROR,
            source_node_id=self.node_id,
            tenant_id=original.tenant_id,
            session_id=original.session_id,
            topic="ros2.error",
            payload={"error": error},
            correlation_id=original.correlation_id,
        )
