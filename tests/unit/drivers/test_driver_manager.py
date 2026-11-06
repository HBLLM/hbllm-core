"""Unit tests for HBLLM Unified Driver Management Layer."""

from __future__ import annotations

from typing import Any

import pytest

from hbllm.drivers.base import (
    BaseDriver,
    DriverAction,
    DriverCapability,
    DriverFeedback,
    DriverInput,
)
from hbllm.drivers.manager import DriverManager


class MockDeviceDriver(BaseDriver):
    """Mock driver simulating a connected device or environment.

    Drivers are pure I/O adapters — they have NO knowledge of HCIR internals.
    They provide raw observations and accept action commands.
    """

    def __init__(self, name: str = "mock_device") -> None:
        super().__init__(
            name=name,
            capabilities={
                DriverCapability.DISCRETE_ACTIONS,
                DriverCapability.STEP_BASED_EXECUTION,
            },
        )
        self.dispatched_actions: list[DriverAction] = []
        self.current_state = {"counter": 0}

    def connect(self, target: Any) -> bool:
        self._target = target
        self.is_connected = True
        return True

    def disconnect(self) -> None:
        self.is_connected = False
        self._target = None

    def get_inputs(self) -> DriverInput:
        return DriverInput(
            raw_data=dict(self.current_state),
            metadata={"status": "online"},
        )

    def get_action_list(self, inputs: DriverInput) -> list[DriverAction]:
        return [
            DriverAction(action_id=1, semantic_intent="INCREMENT"),
            DriverAction(action_id=2, semantic_intent="RESET"),
        ]

    def send_output(self, action: DriverAction) -> Any:
        self.dispatched_actions.append(action)
        if action.action_id == 1:
            self.current_state["counter"] += 1
        elif action.action_id == 2:
            self.current_state["counter"] = 0
        return {"status": "ok", "new_counter": self.current_state["counter"]}

    def process_feedback(self, raw_result: Any) -> DriverFeedback:
        return DriverFeedback(
            success=True,
            reward=1.0 if raw_result["new_counter"] > 0 else 0.0,
            terminated=False,
            causal_delta=raw_result["new_counter"],
            raw_response=raw_result,
        )

    def get_perception_data(self, inputs: DriverInput) -> dict[str, Any]:
        """Return raw structured data — no HCIR types, just dicts.

        The core blackbox interprets this data using its registered lifters.
        The driver has NO knowledge of SpatialEntity, EntityRole, etc.
        """
        return {
            "counter_value": inputs.raw_data.get("counter", 0),
            "status": "online",
        }


class MockCognitiveAgent:
    """Mock cognitive agent interfacing via DriverProtocol.

    Uses the new perception_data dict interface instead of SpatialEntity.
    """

    def __init__(self) -> None:
        self.causal_history: list[tuple[DriverAction, DriverFeedback]] = []

    def plan_next_driver_action(
        self,
        perception_data: dict[str, Any],
        available_actions: list[DriverAction],
        driver_input: DriverInput,
    ) -> DriverAction:
        # Choose INCREMENT
        return available_actions[0]

    def update_causal_feedback(self, action: DriverAction, feedback: DriverFeedback) -> None:
        self.causal_history.append((action, feedback))


def test_driver_registration_and_binding() -> None:
    manager = DriverManager()
    driver = MockDeviceDriver(name="test_device")

    manager.register(driver)
    assert manager.get_driver("test_device") == driver

    # Binding
    target = {"device_port": 8080}
    bound = manager.bind("test_device", target)
    assert bound == driver
    assert driver.is_connected is True
    assert manager.active_driver == driver

    # Unbinding
    manager.unbind()
    assert manager.active_driver is None
    assert driver.is_connected is False


def test_driver_canonical_cognitive_loop() -> None:
    manager = DriverManager()
    driver = MockDeviceDriver(name="mock_env")
    manager.register(driver)
    manager.bind("mock_env", target="dummy_hardware")

    agent = MockCognitiveAgent()

    # Step 1
    fb1 = manager.execute_cognitive_step(agent)
    assert fb1.success is True
    assert fb1.causal_delta == 1
    assert fb1.reward == 1.0
    assert len(driver.dispatched_actions) == 1
    assert driver.dispatched_actions[0].semantic_intent == "INCREMENT"
    assert len(agent.causal_history) == 1

    # Step 2
    fb2 = manager.execute_cognitive_step(agent)
    assert fb2.causal_delta == 2
    assert len(driver.dispatched_actions) == 2
    assert len(agent.causal_history) == 2


def test_unbound_driver_execution_error() -> None:
    manager = DriverManager()
    agent = MockCognitiveAgent()

    with pytest.raises(RuntimeError, match="Cannot execute step: no active connected driver"):
        manager.execute_cognitive_step(agent)


def test_synaptic_device_descriptor_defaults() -> None:
    driver = MockDeviceDriver(name="sensor_cam")
    assert driver.descriptor is not None
    assert driver.descriptor.device_id == "sensor_cam"
    desc_dict = driver.descriptor.to_dict()
    assert desc_dict["device_id"] == "sensor_cam"
    assert "stream_type" in desc_dict


@pytest.mark.asyncio
async def test_driver_manager_async_hotplug() -> None:
    manager = DriverManager()
    driver = MockDeviceDriver(name="hotplug_device")

    attached_names = []
    detached_names = []

    manager.add_on_attached_callback(lambda d: attached_names.append(d.name))
    manager.add_on_detached_callback(lambda n: detached_names.append(n))

    # Attach
    ok = await manager.attach_driver(driver, target={"port": 1234})
    assert ok is True
    assert driver.is_connected is True
    assert "hotplug_device" in manager.get_active_drivers()
    assert attached_names == ["hotplug_device"]

    # Detach
    detached = await manager.detach_driver("hotplug_device")
    assert detached is True
    assert driver.is_connected is False
    assert "hotplug_device" not in manager.get_active_drivers()
    assert detached_names == ["hotplug_device"]


@pytest.mark.asyncio
async def test_driver_manager_async_action_dispatch() -> None:
    manager = DriverManager()
    driver = MockDeviceDriver(name="actuator_device")
    await manager.attach_driver(driver, target={"pin": 5})

    action = DriverAction(action_id=1, semantic_intent="INCREMENT")
    fb = await manager.dispatch_action_async("actuator_device", action)

    assert fb.success is True
    assert fb.causal_delta == 1
    assert len(driver.dispatched_actions) == 1

    await manager.detach_driver("actuator_device")


@pytest.mark.asyncio
async def test_driver_manager_node_bus_integration() -> None:
    import asyncio

    from hbllm.drivers.node import DriverManagerNode
    from hbllm.network.bus import InProcessBus
    from hbllm.network.messages import Message, MessageType

    bus = InProcessBus()
    await bus.start()
    node = DriverManagerNode(node_id="driver_manager_test")
    await node.start(bus)

    attached_events = []
    driver_inputs = []
    feedbacks = []

    async def on_device_attached(msg: Message) -> None:
        attached_events.append(msg.payload)

    async def on_driver_input(msg: Message) -> None:
        driver_inputs.append(msg.payload)

    async def on_feedback(msg: Message) -> None:
        feedbacks.append(msg.payload)

    await bus.subscribe("device.attached", on_device_attached)
    await bus.subscribe("perception.driver.input", on_driver_input)
    await bus.subscribe("action.test.reply", on_feedback)

    # 1. Attach driver
    driver = MockDeviceDriver(name="pnp_sensor")
    await node.driver_manager.attach_driver(driver, target="pnp_target")

    await asyncio.sleep(0.05)
    assert len(attached_events) == 1
    assert attached_events[0]["device_id"] == "pnp_sensor"

    # 2. Dispatch action through bus
    action_msg = Message(
        type=MessageType.COMMAND,
        source_node_id="planner",
        target_node_id=node.node_id,
        topic="action.driver.dispatch",
        payload={
            "driver_name": "pnp_sensor",
            "action": {"action_id": 1, "semantic_intent": "INCREMENT"},
            "reply_topic": "action.test.reply",
        },
    )
    await bus.publish("action.driver.dispatch", action_msg)
    await asyncio.sleep(0.05)

    assert len(feedbacks) == 1
    assert feedbacks[0]["driver_name"] == "pnp_sensor"
    assert feedbacks[0]["success"] is True

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_network_device_driver() -> None:
    from hbllm.drivers.base import SynapticDeviceDescriptor
    from hbllm.drivers.network_driver import NetworkDeviceDriver

    desc = SynapticDeviceDescriptor(
        device_id="esp32_sensor",
        action_schema=[{"name": "turn_on_led", "action_id": 101}],
    )
    driver = NetworkDeviceDriver(name="esp32_sensor", descriptor=desc)
    assert driver.connect("ws://192.168.1.50:8080") is True

    # Ingest network packet
    await driver.ingest_network_payload({"temperature": 23.4, "humidity": 55})
    inp = driver.get_inputs()
    assert inp.raw_data == {"temperature": 23.4, "humidity": 55}

    # Action sink
    dispatched = []

    def _sink(act: DriverAction) -> dict[str, str]:
        dispatched.append(act)
        return {"status": "ok"}

    driver.set_action_sink(_sink)
    fb = await driver.handle_action(DriverAction(action_id=101, semantic_intent="turn_on_led"))
    assert fb.success is True
    assert len(dispatched) == 1
    assert dispatched[0].action_id == 101

    driver.disconnect()
    assert driver.is_connected is False


@pytest.mark.asyncio
async def test_serial_device_driver() -> None:
    from hbllm.drivers.serial_driver import SerialDeviceDriver

    driver = SerialDeviceDriver(name="arduino_uno", port="/dev/ttyUSB0")
    assert driver.connect("/dev/ttyUSB0") is True

    await driver.simulate_incoming_line('{"light_level": 450}')
    inp = driver.get_inputs()
    assert inp.raw_data == {"light_level": 450}

    res = driver.send_output(DriverAction(action_id=1, parameters={"led": "HIGH"}))
    assert res["status"] == "sent"

    driver.disconnect()
    assert driver.is_connected is False


@pytest.mark.asyncio
async def test_device_discovery_engine() -> None:
    from hbllm.drivers.base import SynapticDeviceDescriptor
    from hbllm.drivers.discovery import DeviceDiscoveryEngine

    manager = DriverManager()
    engine = DeviceDiscoveryEngine(driver_manager=manager, enable_physical_scan=False)
    await engine.start()

    desc = SynapticDeviceDescriptor(device_id="phone_sensors")
    driver = await engine.register_network_device(desc, target_info="client_123")
    assert driver.name == "phone_sensors"
    assert "phone_sensors" in manager.get_active_drivers()

    unregistered = await engine.unregister_network_device("phone_sensors")
    assert unregistered is True
    assert "phone_sensors" not in manager.get_active_drivers()

    await engine.stop()


@pytest.mark.asyncio
async def test_dynamic_tool_mounting_with_tool_registry() -> None:
    import asyncio

    from hbllm.actions.tool_registry import ToolRegistry
    from hbllm.drivers.base import SynapticDeviceDescriptor
    from hbllm.drivers.network_driver import NetworkDeviceDriver
    from hbllm.drivers.node import DriverManagerNode
    from hbllm.network.bus import InProcessBus

    bus = InProcessBus()
    await bus.start()

    tool_reg = ToolRegistry(bus=bus)
    node = DriverManagerNode(node_id="driver_manager_tools", tool_registry=tool_reg)
    await node.start(bus)

    desc = SynapticDeviceDescriptor(
        device_id="smart_switch",
        action_schema=[
            {
                "name": "toggle",
                "action_id": "act_toggle",
                "description": "Toggle the smart switch on or off",
                "parameters": {"state": "string"},
            }
        ],
    )
    driver = NetworkDeviceDriver(name="smart_switch", descriptor=desc)
    driver.set_action_sink(lambda act: {"status": "ok", "state": act.parameters.get("state")})

    # Attach driver -> Should dynamically mount tool into ToolRegistry
    await node.driver_manager.attach_driver(driver, target="local_test")
    await asyncio.sleep(0.05)

    expected_tool_name = "device_smart_switch_toggle"
    tools = [t["name"] for t in tool_reg.list_tools()]
    assert expected_tool_name in tools

    # Invoke tool directly via ToolRegistry
    result = await tool_reg.invoke(expected_tool_name, state="ON")
    assert result.success is True
    assert "ON" in result.output

    # Detach driver -> Should cleanly unmount tool
    await node.driver_manager.detach_driver("smart_switch")
    await asyncio.sleep(0.05)
    tools_after = [t["name"] for t in tool_reg.list_tools()]
    assert expected_tool_name not in tools_after

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_driver_manager_lifecycle_shutdown() -> None:
    manager = DriverManager()
    driver1 = MockDeviceDriver(name="dev_1")
    driver2 = MockDeviceDriver(name="dev_2")

    await manager.attach_driver(driver1, target={"port": 1})
    await manager.attach_driver(driver2, target={"port": 2})

    assert manager.is_driver_active("dev_1") is True
    assert manager.is_driver_active("dev_2") is True
    assert set(manager.list_active_drivers()) == {"dev_1", "dev_2"}

    await manager.start_streaming()
    assert manager.is_streaming is True

    await manager.shutdown()
    assert manager.is_streaming is False
    assert len(manager.list_active_drivers()) == 0
    assert driver1.is_connected is False
    assert driver2.is_connected is False


@pytest.mark.asyncio
async def test_driver_manager_hcir_blackbox_integration() -> None:
    """Verify DriverManager properly drives HCIR CognitiveBlackbox observation, decision, and learning."""
    from hbllm.drivers.cognitive_blackbox import CognitiveBlackbox
    from hbllm.hcir.workspace import HCIRWorkspaceState

    hcir_ws = HCIRWorkspaceState()
    blackbox = CognitiveBlackbox(workspace=hcir_ws)

    manager = DriverManager()
    manager.set_cognitive_engine(blackbox)
    assert manager.cognitive_engine is blackbox

    driver = MockDeviceDriver(name="hcir_test_dev")
    await manager.attach_driver(driver, target={"init": True})

    # Execute cognitive step through DriverManager -> CognitiveBlackbox
    feedback = manager.execute_cognitive_step()
    assert feedback.success is True
    assert len(driver.dispatched_actions) == 1

    # Verify HCIR blackbox state was updated
    state = blackbox.get_state("hcir_test_dev")
    assert state.step_count >= 1
    assert state.last_action_id is not None
    assert state.total_reward >= 0.0

    await manager.shutdown()
