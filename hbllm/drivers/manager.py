"""Driver Manager for orchestrating lifecycle and standardized execution loops across connected devices.

Supports both single-driver (legacy) and concurrent multi-driver (MIMO) operation modes.
When a CognitiveBlackbox is embedded, all cognitive state is managed internally —
no external cognitive_agent parameter is needed.

The DriverManager orchestrates the canonical loop:
  Input (raw) → Perception (raw) → Core Cognition → Action → Feedback → Learn

Drivers never touch HCIR internals. All interpretation happens in the core blackbox.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.drivers.base import BaseDriver, DriverAction, DriverFeedback, DriverInput

logger = logging.getLogger(__name__)


class DriverManager:
    """Central registry and lifecycle manager for HBLLM drivers.

    Orchestrates the canonical Input → Cognition → Output → Feedback loop,
    ensuring that internal cognitive and planning models remain completely decoupled
    from the specifics of any connected hardware, game environment, mobile device, or web interface.

    Supports two operation modes:
        1. Single-driver (legacy): bind() + execute_cognitive_step(agent)
        2. MIMO multi-driver: bind_concurrent() + execute_mimo_step()
    """

    def __init__(self) -> None:
        self._drivers: dict[str, BaseDriver] = {}
        self._active_driver: BaseDriver | None = None

        # MIMO extensions
        self._active_drivers: dict[str, BaseDriver] = {}  # Concurrent active drivers
        self._primary_driver: str | None = None  # Default source for single-output
        self._cognitive_engine: Any | None = None  # Embedded CognitiveBlackbox

    @property
    def active_driver(self) -> BaseDriver | None:
        """The currently bound active driver (legacy single-driver mode)."""
        return self._active_driver

    @property
    def cognitive_engine(self) -> Any | None:
        """The embedded CognitiveBlackbox, if set."""
        return self._cognitive_engine

    def set_cognitive_engine(self, engine: Any) -> None:
        """Embed a CognitiveBlackbox for self-contained cognitive operation."""
        self._cognitive_engine = engine
        if hasattr(engine, "register_driver"):
            for driver in self._drivers.values():
                engine.register_driver(driver)
        logger.info("Embedded CognitiveBlackbox into DriverManager")

    # ── Driver Registration & Lifecycle ───────────────────────────────────

    def register(self, driver: BaseDriver) -> None:
        """Register a driver instance in the manager."""
        self._drivers[driver.name] = driver
        if self._cognitive_engine is not None and hasattr(
            self._cognitive_engine, "register_driver"
        ):
            self._cognitive_engine.register_driver(driver)
        logger.info("Registered driver: '%s'", driver.name)

    def get_driver(self, name: str) -> BaseDriver:
        """Retrieve a registered driver by name."""
        if name not in self._drivers:
            raise KeyError(f"Driver '{name}' not found. Available: {list(self._drivers.keys())}")
        return self._drivers[name]

    def list_drivers(self) -> list[str]:
        """List all registered driver names."""
        return list(self._drivers.keys())

    # ── Single-Driver Mode (Legacy) ──────────────────────────────────────

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
        self._active_drivers[name] = driver
        if self._primary_driver is None:
            self._primary_driver = name
        logger.info("Bound and connected active driver: '%s'", name)
        return driver

    def unbind(self) -> None:
        """Disconnect and unbind the active driver."""
        if self._active_driver and self._active_driver.is_connected:
            self._active_driver.disconnect()
            # Remove from concurrent active set
            for name, drv in list(self._active_drivers.items()):
                if drv is self._active_driver:
                    del self._active_drivers[name]
                    break
        self._active_driver = None

    # ── MIMO Multi-Driver Mode ────────────────────────────────────────────

    def bind_concurrent(self, name: str, target: Any) -> BaseDriver:
        """Bind a driver without disconnecting others (MIMO mode)."""
        driver = self.get_driver(name)
        success = driver.connect(target)
        if not success:
            raise RuntimeError(f"Failed to connect driver '{name}' to target {target}")

        self._active_drivers[name] = driver
        if self._primary_driver is None:
            self._primary_driver = name
        logger.info(
            "Bound concurrent driver: '%s' (total active: %d)", name, len(self._active_drivers)
        )
        return driver

    def unbind_concurrent(self, name: str) -> None:
        """Disconnect a single driver without affecting others."""
        driver = self._active_drivers.pop(name, None)
        if driver and driver.is_connected:
            driver.disconnect()
        if self._primary_driver == name:
            self._primary_driver = next(iter(self._active_drivers), None)

    def get_active_drivers(self) -> dict[str, BaseDriver]:
        """Get all currently active concurrent drivers."""
        return dict(self._active_drivers)

    # ── Cognitive Step Execution ──────────────────────────────────────────

    def execute_cognitive_step(self, cognitive_agent: Any = None) -> DriverFeedback:
        """Execute a single canonical cognitive-driver step.

        1. Ingest input: raw observation from active driver
        2. Get perception data: raw structured data from driver (no HCIR types)
        3. Query action space: available action options from driver
        4. Plan next action: core cognition decides action from raw perception
        5. Dispatch output: transmit chosen action command to device
        6. Process feedback: normalize device response
        7. Update: core learns from action-outcome pair

        If cognitive_agent is None, uses the embedded CognitiveBlackbox.
        """
        driver = self._active_driver
        if driver is None or not driver.is_connected:
            raise RuntimeError("Cannot execute step: no active connected driver.")

        # Use embedded engine if no external agent provided
        agent = cognitive_agent or self._cognitive_engine
        if agent is None:
            raise RuntimeError(
                "Cannot execute step: no cognitive_agent provided and no embedded CognitiveBlackbox. "
                "Call set_cognitive_engine() or pass a cognitive_agent."
            )

        # 1. Ingest input (raw)
        driver_input: DriverInput = driver.get_inputs()
        driver_input.source_id = driver_input.source_id or driver.name

        # 2. Get raw perception data (driver provides structured data, core interprets)
        perception_data = driver.get_perception_data(driver_input)

        # 3. Available action capabilities
        available_actions: list[DriverAction] = driver.get_action_list(driver_input)

        # 4. Cognitive decision — core blackbox handles ALL interpretation and planning
        if hasattr(agent, "observe") and hasattr(agent, "decide"):
            # CognitiveBlackbox interface — pure blackbox, no HCIR types exposed
            agent.observe(driver_input, perception_data=perception_data)
            selected_action = agent.decide(available_actions, source_id=driver.name)
        elif hasattr(agent, "plan_next_driver_action"):
            selected_action = agent.plan_next_driver_action(
                perception_data=perception_data,
                available_actions=available_actions,
                driver_input=driver_input,
            )
        elif hasattr(agent, "plan_next_action"):
            # Legacy fallback for existing ARC agent format
            avail_ids = [a.action_id for a in available_actions]
            act_id, conf = agent.plan_next_action(driver_input.raw_data, avail_ids)
            selected_action = next(
                (a for a in available_actions if a.action_id == act_id),
                DriverAction(action_id=act_id, confidence=conf),
            )
        else:
            raise AttributeError(
                "Cognitive agent must implement observe/decide (CognitiveBlackbox), "
                "plan_next_driver_action, or plan_next_action"
            )

        # 5. Output dispatch
        raw_result = driver.send_output(selected_action)

        # 6. Feedback normalization
        feedback: DriverFeedback = driver.process_feedback(raw_result)

        # 7. Core learns from action-outcome pair
        if hasattr(agent, "update") and hasattr(agent, "observe"):
            # CognitiveBlackbox interface
            agent.update(selected_action, feedback, source_id=driver.name)
        elif hasattr(agent, "update_causal_feedback"):
            agent.update_causal_feedback(selected_action, feedback)
        elif hasattr(agent, "update_causal_dynamics"):
            prev_obs = driver_input.raw_data
            curr_obs = (
                feedback.causal_delta
                if feedback.causal_delta is not None
                else driver.get_inputs().raw_data
            )
            agent.update_causal_dynamics(selected_action.action_id, prev_obs, curr_obs)

        return feedback

    def execute_mimo_step(self) -> dict[str, DriverFeedback]:
        """Execute cognitive step across ALL active drivers simultaneously (MIMO mode).

        1. Gather raw observations from all active drivers
        2. Feed raw perception into CognitiveBlackbox (core interprets internally)
        3. Decide actions per driver
        4. Dispatch actions and collect feedback
        5. Core learns from all action-outcome pairs

        Requires embedded CognitiveBlackbox (set_cognitive_engine).
        """
        if self._cognitive_engine is None:
            raise RuntimeError(
                "MIMO mode requires an embedded CognitiveBlackbox. Call set_cognitive_engine()."
            )

        if not self._active_drivers:
            raise RuntimeError("No active drivers for MIMO step.")

        agent = self._cognitive_engine
        results: dict[str, DriverFeedback] = {}

        # 1. Gather observations from ALL drivers
        observations: dict[str, tuple[DriverInput, dict]] = {}
        for name, driver in self._active_drivers.items():
            if not driver.is_connected:
                continue
            driver_input = driver.get_inputs()
            driver_input.source_id = driver_input.source_id or name
            perception_data = driver.get_perception_data(driver_input)
            observations[name] = (driver_input, perception_data)

        # 2. Feed observations into blackbox (core does all lifting/interpretation)
        for name, (driver_input, perception_data) in observations.items():
            agent.observe(driver_input, perception_data=perception_data)

        # 3. Decide and dispatch per driver
        for name, driver in self._active_drivers.items():
            if name not in observations or not driver.is_connected:
                continue
            driver_input, _ = observations[name]
            available_actions = driver.get_action_list(driver_input)
            selected_action = agent.decide(available_actions, source_id=name)

            raw_result = driver.send_output(selected_action)
            feedback = driver.process_feedback(raw_result)
            agent.update(selected_action, feedback, source_id=name)
            results[name] = feedback

        return results

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return driver manager statistics."""
        return {
            "registered_drivers": len(self._drivers),
            "active_drivers": len(self._active_drivers),
            "primary_driver": self._primary_driver,
            "has_cognitive_engine": self._cognitive_engine is not None,
            "driver_names": list(self._drivers.keys()),
        }
