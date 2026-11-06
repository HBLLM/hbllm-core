"""Driver Manager for orchestrating lifecycle and standardized execution loops across connected devices.

Supports both single-driver (legacy) and concurrent multi-driver (MIMO) operation modes.
When a CognitiveBlackbox is embedded, all cognitive state is managed internally —
no external cognitive_agent parameter is needed.

The DriverManager orchestrates the canonical loop:
  Input (raw) → Perception (raw) → Core Cognition → Action → Feedback → Learn

Drivers never touch HCIR internals. All interpretation happens in the core blackbox.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

from hbllm.drivers.base import (
    BaseDriver,
    DriverAction,
    DriverFeedback,
    DriverInput,
)

logger = logging.getLogger(__name__)


class DriverManager:
    """Central registry and lifecycle manager for HBLLM drivers.

    Orchestrates the canonical Input → Cognition → Output → Feedback loop,
    ensuring that internal cognitive and planning models remain completely decoupled
    from the specifics of any connected hardware, game environment, mobile device, or web interface.

    Supports three operation modes:
        1. Single-driver (legacy): bind() + execute_cognitive_step(agent)
        2. MIMO multi-driver: bind_concurrent() + execute_mimo_step()
        3. Event-driven streaming & hot-plug: attach_driver(), start_streaming(), dispatch_action_async()
    """

    def __init__(self) -> None:
        self._drivers: dict[str, BaseDriver] = {}
        self._active_driver: BaseDriver | None = None

        # MIMO extensions
        self._active_drivers: dict[str, BaseDriver] = {}  # Concurrent active drivers
        self._primary_driver: str | None = None  # Default source for single-output
        self._cognitive_engine: Any | None = None  # Embedded CognitiveBlackbox

        # Hot-plug & streaming event hooks
        self._on_device_attached_callbacks: list[Callable[[BaseDriver], Any]] = []
        self._on_device_detached_callbacks: list[Callable[[str], Any]] = []
        self._on_driver_input_callbacks: list[Callable[[DriverInput], Any]] = []
        self._streaming_tasks: dict[str, asyncio.Task[None]] = {}
        self._streaming_active: bool = False

    @property
    def is_streaming(self) -> bool:
        """Whether background streaming observation loops are currently active."""
        return self._streaming_active

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

    # ── Callback Management ───────────────────────────────────────────────

    def add_on_attached_callback(self, callback: Callable[[BaseDriver], Any]) -> None:
        """Register a callback invoked when a driver/device is attached."""
        self._on_device_attached_callbacks.append(callback)

    def add_on_detached_callback(self, callback: Callable[[str], Any]) -> None:
        """Register a callback invoked when a driver/device is detached."""
        self._on_device_detached_callbacks.append(callback)

    def add_on_input_callback(self, callback: Callable[[DriverInput], Any]) -> None:
        """Register a callback invoked when an active driver produces an observation."""
        self._on_driver_input_callbacks.append(callback)

    # ── Async Hot-Plug & Streaming ────────────────────────────────────────

    async def attach_driver(self, driver: BaseDriver, target: Any = None) -> bool:
        """Asynchronously connect, register, and activate a driver with hot-plug notifications."""
        success = await driver.connect_async(target)
        if not success:
            logger.warning("Failed to connect driver '%s'", driver.name)
            return False

        self.register(driver)
        self._active_drivers[driver.name] = driver
        if self._primary_driver is None:
            self._primary_driver = driver.name
        if self._active_driver is None:
            self._active_driver = driver

        # Trigger hot-plug callbacks
        for cb in self._on_device_attached_callbacks:
            try:
                res = cb(driver)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as e:
                logger.error("Error in on_device_attached callback: %s", e)

        # Launch streaming if streaming engine is active
        if self._streaming_active:
            self._start_driver_stream_task(driver)

        logger.info(
            "Attached driver '%s' (Total active: %d)", driver.name, len(self._active_drivers)
        )
        return True

    async def detach_driver(self, name: str) -> bool:
        """Asynchronously disconnect and unbind a driver with hot-plug notifications."""
        driver = self._active_drivers.pop(name, None)
        if driver and driver.is_connected:
            await driver.disconnect_async()

        task = self._streaming_tasks.pop(name, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        if self._active_driver and self._active_driver.name == name:
            self._active_driver = next(iter(self._active_drivers.values()), None)

        if self._primary_driver == name:
            self._primary_driver = next(iter(self._active_drivers), None)

        # Trigger detach callbacks
        for cb in self._on_device_detached_callbacks:
            try:
                res = cb(name)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as e:
                logger.error("Error in on_device_detached callback: %s", e)

        logger.info("Detached driver '%s'", name)
        return True

    async def start_streaming(self, poll_interval_s: float = 0.05) -> None:
        """Start background streaming observation loops for all active drivers."""
        self._streaming_active = True
        for driver in self._active_drivers.values():
            if (
                driver.name not in self._streaming_tasks
                or self._streaming_tasks[driver.name].done()
            ):
                self._start_driver_stream_task(driver, poll_interval_s=poll_interval_s)

    async def stop_streaming(self) -> None:
        """Stop all background streaming observation loops."""
        self._streaming_active = False
        for name, task in list(self._streaming_tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._streaming_tasks.clear()

    def _start_driver_stream_task(self, driver: BaseDriver, poll_interval_s: float = 0.05) -> None:
        """Spawn background coroutine to stream inputs from driver."""
        old_task = self._streaming_tasks.get(driver.name)
        if old_task and not old_task.done():
            old_task.cancel()
        task = asyncio.create_task(
            self._driver_stream_loop(driver, poll_interval_s=poll_interval_s),
            name=f"stream_{driver.name}",
        )
        self._streaming_tasks[driver.name] = task

    async def _driver_stream_loop(self, driver: BaseDriver, poll_interval_s: float = 0.05) -> None:
        """Continuous stream / event consumer loop for an active driver."""
        logger.debug("Starting stream loop for driver '%s'", driver.name)
        try:
            while self._streaming_active and driver.is_connected:
                try:
                    # Check if driver supports continuous async stream iterator
                    stream_obj = (
                        driver.stream_inputs() if hasattr(driver, "stream_inputs") else None
                    )
                    if stream_obj is not None and hasattr(stream_obj, "__aiter__"):
                        async for inp in stream_obj:
                            if not self._streaming_active or not driver.is_connected:
                                break
                            if inp is None:
                                continue
                            if getattr(inp, "timestamp", 0.0) == 0.0:
                                inp.timestamp = time.time()
                            if hasattr(inp, "source_id"):
                                inp.source_id = inp.source_id or driver.name
                            await self._dispatch_driver_input(inp)
                    else:
                        inp = await driver.get_inputs_async()
                        if inp is not None:
                            if getattr(inp, "timestamp", 0.0) == 0.0:
                                inp.timestamp = time.time()
                            if hasattr(inp, "source_id"):
                                inp.source_id = inp.source_id or driver.name
                            await self._dispatch_driver_input(inp)
                except asyncio.CancelledError:
                    raise
                except Exception as inner_err:
                    logger.warning(
                        "Transient error in driver '%s' observation iteration: %s",
                        driver.name,
                        inner_err,
                    )

                # Backoff / interval
                interval = (
                    1.0 / driver.descriptor.sample_rate_hz
                    if driver.descriptor and driver.descriptor.sample_rate_hz > 0
                    else poll_interval_s
                )
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Fatal error in driver '%s' stream loop: %s", driver.name, e)

    async def _dispatch_driver_input(self, inp: DriverInput) -> None:
        """Notify all registered callbacks of an incoming driver observation."""
        for cb in self._on_driver_input_callbacks:
            try:
                res = cb(inp)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as e:
                logger.error("Error in on_driver_input callback: %s", e)

    async def dispatch_action_async(self, driver_name: str, action: DriverAction) -> DriverFeedback:
        """Asynchronously dispatch an action to a connected driver and record feedback."""
        driver = self.get_driver(driver_name)
        if not driver.is_connected:
            raise RuntimeError(f"Driver '{driver_name}' is not connected.")

        feedback = await driver.handle_action(action)

        if self._cognitive_engine is not None and hasattr(self._cognitive_engine, "update"):
            res = self._cognitive_engine.update(action, feedback, source_id=driver_name)
            if asyncio.iscoroutine(res):
                await res

        return feedback

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
        if self._streaming_active:
            self._start_driver_stream_task(driver)
        logger.info("Bound and connected active driver: '%s'", name)
        return driver

    def unbind(self) -> None:
        """Disconnect and unbind the active driver."""
        if self._active_driver and self._active_driver.is_connected:
            name = self._active_driver.name
            task = self._streaming_tasks.pop(name, None)
            if task and not task.done():
                task.cancel()
            self._active_driver.disconnect()
            # Remove from concurrent active set
            for d_name, drv in list(self._active_drivers.items()):
                if drv is self._active_driver:
                    del self._active_drivers[d_name]
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
        if self._active_driver is None:
            self._active_driver = driver
        if self._streaming_active:
            self._start_driver_stream_task(driver)
        logger.info(
            "Bound concurrent driver: '%s' (total active: %d)", name, len(self._active_drivers)
        )
        return driver

    def unbind_concurrent(self, name: str) -> None:
        """Disconnect a single driver without affecting others."""
        task = self._streaming_tasks.pop(name, None)
        if task and not task.done():
            task.cancel()
        driver = self._active_drivers.pop(name, None)
        if driver and driver.is_connected:
            driver.disconnect()
        if self._active_driver and self._active_driver.name == name:
            self._active_driver = next(iter(self._active_drivers.values()), None)
        if self._primary_driver == name:
            self._primary_driver = next(iter(self._active_drivers), None)

    def get_active_drivers(self) -> dict[str, BaseDriver]:
        """Get all currently active concurrent drivers."""
        return dict(self._active_drivers)

    def list_active_drivers(self) -> list[str]:
        """List all currently active connected driver names."""
        return list(self._active_drivers.keys())

    def is_driver_active(self, name: str) -> bool:
        """Check if a driver is currently active and connected."""
        return name in self._active_drivers and self._active_drivers[name].is_connected

    async def shutdown(self) -> None:
        """Stop all streaming and disconnect all active drivers."""
        await self.stop_streaming()
        for name, driver in list(self._active_drivers.items()):
            try:
                if driver.is_connected:
                    await driver.disconnect_async()
            except Exception as e:
                logger.warning("Error disconnecting driver '%s' during shutdown: %s", name, e)
        self._active_drivers.clear()
        self._active_driver = None

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
