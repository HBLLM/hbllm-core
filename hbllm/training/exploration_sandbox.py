"""Exploration Sandbox — safe environment for experimentation.

Provides an isolated execution environment where the AI can:
    1. Try tool invocations without real-world side effects
    2. Test IoT commands in simulation mode
    3. Experiment with new strategies before deployment
    4. Run "what if" scenarios for goal planning

Architecture:
    - Wraps the real tool chain with a simulated layer
    - Records all simulated actions for review
    - No actual OS/IoT/network commands are executed
    - Results are synthesized from heuristics or cached real data
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SimulatedAction:
    """A recorded simulated action."""

    action: str
    params: dict[str, Any]
    simulated_result: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    would_succeed: bool = True
    risk_tier: int = 0


class ExplorationSandbox:
    """Isolated execution environment for safe experimentation.

    All actions are simulated — nothing touches the real world.

    Usage::

        sandbox = ExplorationSandbox()

        # Simulate an IoT command
        result = sandbox.simulate_action(
            action="light.on",
            params={"device_id": "kitchen_light", "brightness": 80},
        )

        # Review what happened
        history = sandbox.get_history()

        # Check if a sequence of actions would work
        plan_ok = sandbox.validate_plan([
            {"action": "thermostat.set_temp", "params": {"temp": 22}},
            {"action": "blinds.close", "params": {"room": "bedroom"}},
        ])
    """

    def __init__(
        self,
        known_devices: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._history: list[SimulatedAction] = []
        self._simulated_state: dict[str, dict[str, Any]] = {}
        self._known_devices = known_devices or {}
        self._max_history = 500

        # Simulated world state
        self._world: dict[str, Any] = {
            "lights": {},
            "locks": {},
            "thermostat": {"temp": 22, "mode": "auto"},
            "files": set(),
            "processes": [],
        }

    def simulate_action(
        self,
        action: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Simulate an action without real-world effects.

        Returns a synthetic result that approximates what would happen.
        """
        params = params or {}

        # Determine risk tier
        from hbllm.actions.confirmation import ActionRiskClassifier

        classifier = ActionRiskClassifier()
        assessment = classifier.classify(action)

        # Simulate based on action category
        result = self._simulate(action, params)

        # Record
        record = SimulatedAction(
            action=action,
            params=params,
            simulated_result=result,
            would_succeed=result.get("success", True),
            risk_tier=assessment.tier,
        )
        self._history.append(record)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        logger.debug(
            "Sandbox: %s(%s) → %s (tier=%d)",
            action,
            params,
            result.get("status", "unknown"),
            assessment.tier,
        )

        return result

    def validate_plan(
        self,
        plan: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Validate a sequence of actions in the sandbox.

        Returns a report of which steps would succeed/fail.
        """
        results: list[dict[str, Any]] = []
        all_ok = True

        for i, step in enumerate(plan):
            action = step.get("action", "")
            params = step.get("params", {})

            result = self.simulate_action(action, params)
            step_ok = result.get("success", True)

            results.append(
                {
                    "step": i + 1,
                    "action": action,
                    "success": step_ok,
                    "result": result,
                }
            )

            if not step_ok:
                all_ok = False

        return {
            "plan_valid": all_ok,
            "steps": results,
            "total_steps": len(plan),
            "successful_steps": sum(1 for r in results if r["success"]),
        }

    def what_if(
        self,
        action: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a "what if" scenario and describe consequences.

        Returns the simulated result plus predicted side effects.
        """
        result = self.simulate_action(action, params or {})

        # Predict side effects
        side_effects: list[str] = []

        if action.startswith("light."):
            side_effects.append("Room illumination would change")
        elif action.startswith("lock."):
            side_effects.append("Security state would change")
            side_effects.append("Audit entry would be logged")
            side_effects.append("User notification would be sent")
        elif action.startswith("thermostat."):
            side_effects.append("HVAC system would activate")
            side_effects.append("Energy consumption would change")
        elif action.startswith("file.delete"):
            side_effects.append("File would be permanently removed")
            side_effects.append("Undo may not be possible")
        elif action.startswith("shell."):
            side_effects.append("System command would be executed")
            side_effects.append("Process resources would be consumed")

        return {
            **result,
            "side_effects": side_effects,
            "risk_assessment": result.get("risk_tier", 0),
        }

    def get_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent simulation history."""
        return [
            {
                "action": a.action,
                "params": a.params,
                "result": a.simulated_result,
                "would_succeed": a.would_succeed,
                "risk_tier": a.risk_tier,
                "timestamp": a.timestamp,
            }
            for a in self._history[-limit:]
        ]

    def reset(self) -> None:
        """Reset the sandbox to initial state."""
        self._history.clear()
        self._world = {
            "lights": {},
            "locks": {},
            "thermostat": {"temp": 22, "mode": "auto"},
            "files": set(),
            "processes": [],
        }

    def _simulate(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Generate a simulated result for an action."""
        # Light actions
        if action.startswith("light."):
            device = params.get("device_id", "unknown")
            if "on" in action:
                self._world["lights"][device] = {"state": "on", **params}
                return {"success": True, "status": "Light turned on (simulated)"}
            elif "off" in action:
                self._world["lights"][device] = {"state": "off"}
                return {"success": True, "status": "Light turned off (simulated)"}

        # Lock actions
        if action.startswith("lock."):
            device = params.get("device_id", "unknown")
            if "unlock" in action:
                self._world["locks"][device] = "unlocked"
                return {"success": True, "status": "Lock unlocked (simulated)", "risk_tier": 3}
            elif "lock" in action:
                self._world["locks"][device] = "locked"
                return {"success": True, "status": "Lock locked (simulated)", "risk_tier": 3}

        # Thermostat
        if action.startswith("thermostat."):
            temp = params.get("temp", params.get("temperature"))
            if temp is not None:
                self._world["thermostat"]["temp"] = temp
                return {"success": True, "status": f"Thermostat set to {temp}°C (simulated)"}

        # File operations
        if action == "file.create":
            path = params.get("path", "")
            self._world["files"].add(path)
            return {"success": True, "status": f"File {path} created (simulated)"}
        if action == "file.delete":
            path = params.get("path", "")
            self._world["files"].discard(path)
            return {"success": True, "status": f"File {path} deleted (simulated)", "risk_tier": 3}

        # Read actions always succeed
        if action.startswith("read") or action.startswith("query") or action.startswith("get"):
            return {"success": True, "status": "Read operation (simulated)", "data": {}}

        # Default
        return {
            "success": True,
            "status": f"Action '{action}' simulated",
            "note": "No specific simulation handler",
        }

    def stats(self) -> dict[str, Any]:
        return {
            "total_simulations": len(self._history),
            "world_lights": len(self._world.get("lights", {})),
            "world_locks": len(self._world.get("locks", {})),
        }
