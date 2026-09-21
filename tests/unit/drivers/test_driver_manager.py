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
from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity


class MockDeviceDriver(BaseDriver):
    """Mock driver simulating a connected device or environment."""

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

    def lift_to_hcir(self, inputs: DriverInput) -> tuple[list[SpatialEntity], set[tuple[int, int]]]:
        # Lift device state into a domain-neutral entity
        ent = SpatialEntity(
            id="device_counter",
            role=EntityRole.RESOURCE,
            centroid=(0.0, 0.0),
            grid_pos=(0, 0),
            area=1,
            bounding_box=(0, 0, 0, 0),
            properties={"value": inputs.raw_data.get("counter", 0)},
        )
        return [ent], set()


class MockCognitiveAgent:
    """Mock cognitive agent interfacing via DriverProtocol."""

    def __init__(self) -> None:
        self.causal_history: list[tuple[DriverAction, DriverFeedback]] = []

    def plan_next_driver_action(
        self,
        entities: list[SpatialEntity],
        barriers: set[tuple[int, int]],
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
