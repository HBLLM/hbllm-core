"""Driver Manager for orchestrating lifecycle and standardized execution loops across connected devices."""

from __future__ import annotations

import logging
from typing import Any

from hbllm.drivers.base import BaseDriver, DriverAction, DriverFeedback, DriverInput

logger = logging.getLogger(__name__)


class DriverManager:
    """Central registry and lifecycle manager for HBLLM drivers.

    Orchestrates the canonical Input -> Cognition -> Output -> Feedback loop,
    ensuring that internal cognitive and planning models remain completely decoupled
    from the specifics of any connected hardware, game environment, mobile device, or web interface.
    """

    def __init__(self) -> None:
        self._drivers: dict[str, BaseDriver] = {}
        self._active_driver: BaseDriver | None = None

    @property
    def active_driver(self) -> BaseDriver | None:
        """The currently bound active driver."""
        return self._active_driver

    def register(self, driver: BaseDriver) -> None:
        """Register a driver instance in the manager."""
        self._drivers[driver.name] = driver
        logger.info("Registered driver: '%s'", driver.name)

    def get_driver(self, name: str) -> BaseDriver:
        """Retrieve a registered driver by name."""
        if name not in self._drivers:
            raise KeyError(f"Driver '{name}' not found. Available: {list(self._drivers.keys())}")
        return self._drivers[name]

    def bind(self, name: str, target: Any) -> BaseDriver:
        """Bind and connect a driver to a specific environment or device target."""
        driver = self.get_driver(name)
        if (
            self._active_driver
            and self._active_driver != driver
            and self._active_driver.is_connected
        ):
            self._active_driver.disconnect()

        success = driver.connect(target)
        if not success:
            raise RuntimeError(f"Failed to connect driver '{name}' to target {target}")

        self._active_driver = driver
        logger.info("Bound and connected active driver: '%s'", name)
        return driver

    def unbind(self) -> None:
        """Disconnect and unbind the active driver."""
        if self._active_driver and self._active_driver.is_connected:
            self._active_driver.disconnect()
        self._active_driver = None

    def execute_cognitive_step(self, cognitive_agent: Any) -> DriverFeedback:
        """Execute a single canonical cognitive-driver step.

        1. Ingest input: raw observation from active driver
        2. Lift to HCIR: convert raw input into domain-neutral entities & barriers
        3. Query action space: available action options from driver
        4. Plan next action: internal HCIR cognition & subgoal reasoning
        5. Dispatch output: transmit chosen action command to device
        6. Process feedback: normalize device response
        7. Update causal dynamics & memory in cognitive agent
        """
        driver = self._active_driver
        if driver is None or not driver.is_connected:
            raise RuntimeError("Cannot execute step: no active connected driver.")

        # 1. Ingest input
        driver_input: DriverInput = driver.get_inputs()

        # 2. Lift to HCIR (sensory abstraction)
        entities, barriers = driver.lift_to_hcir(driver_input)

        # 3. Available action capabilities
        available_actions: list[DriverAction] = driver.get_action_list(driver_input)

        # 4. Cognitive decision / Subgoal planning inside HBLLM
        # Agent plan_next_action accepts lifted HCIR entities or raw inputs according to agent design
        if hasattr(cognitive_agent, "plan_next_driver_action"):
            selected_action = cognitive_agent.plan_next_driver_action(
                entities=entities,
                barriers=barriers,
                available_actions=available_actions,
                driver_input=driver_input,
            )
        elif hasattr(cognitive_agent, "plan_next_action"):
            # Fallback for existing ARC agent format
            avail_ids = [a.action_id for a in available_actions]
            act_id, conf = cognitive_agent.plan_next_action(driver_input.raw_data, avail_ids)
            selected_action = next(
                (a for a in available_actions if a.action_id == act_id),
                DriverAction(action_id=act_id, confidence=conf),
            )
        else:
            raise AttributeError(
                "Cognitive agent must implement plan_next_driver_action or plan_next_action"
            )

        # 5. Output dispatch
        raw_result = driver.send_output(selected_action)

        # 6. Feedback normalization
        feedback: DriverFeedback = driver.process_feedback(raw_result)

        # 7. Causal dynamics & memory update
        if hasattr(cognitive_agent, "update_causal_feedback"):
            cognitive_agent.update_causal_feedback(selected_action, feedback)
        elif hasattr(cognitive_agent, "update_causal_dynamics"):
            prev_obs = driver_input.raw_data
            curr_obs = (
                feedback.causal_delta
                if feedback.causal_delta is not None
                else driver.get_inputs().raw_data
            )
            cognitive_agent.update_causal_dynamics(selected_action.action_id, prev_obs, curr_obs)

        return feedback
