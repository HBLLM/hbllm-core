"""
Actuator Capability Adapter — controls physical hardware, relays, and external APIs.

Safely dispatches physical actuation commands:

    ActionNode (e.g., "trigger_irrigation_relay")
                  │
       ActuatorCapabilityAdapter
                  ├── BranchMode.LIVE → Dispatch to physical hardware / API
                  └── BranchMode.SIMULATION / REPLAY → Simulate response (safety block)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.abi import ExecutionMetrics, ExecutionResult, ICognitiveNodeABI
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.transactions import HCIRDelta
from hbllm.hcir.types import BranchMode
from hbllm.hcir.workspace import HCIRWorkspaceState

logger = logging.getLogger(__name__)


@dataclass
class ActuatorCommand:
    """A command to execute on physical hardware or an external system."""

    actuator_id: str
    command_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 5.0


class ActuatorCapabilityAdapter(ICognitiveNodeABI):
    """Safely dispatches physical commands with strict BranchMode isolation.

    Usage::

        adapter = ActuatorCapabilityAdapter(services)
        adapter.register_hardware_handler("relay_01", my_relay_fn)
        cmd = ActuatorCommand(actuator_id="relay_01", command_name="turn_on")
        res = await adapter.dispatch_command(cmd)
    """

    supported_hcir_versions = ["1.0.0"]
    required_kernel_services = ["TransactionManager", "CapabilityResolver"]
    declared_capabilities = ["physical_actuation"]

    def __init__(self, services: KernelServices) -> None:
        self._services = services
        self._handlers: dict[str, Callable[[dict[str, Any]], bool]] = {}

    def register_hardware_handler(
        self, actuator_id: str, handler_fn: Callable[[dict[str, Any]], bool]
    ) -> None:
        """Register a handler callback for a specific actuator ID."""
        self._handlers[actuator_id] = handler_fn

    async def execute(
        self,
        transaction: Any,
        workspace: Any,
        services: Any,
    ) -> ExecutionResult:
        """ABI execution contract."""
        start_time = time.monotonic()
        ops = transaction.operations if hasattr(transaction, "operations") else []

        delta = HCIRDelta()
        for op in ops:
            if op.op.value == "add_node" and op.node_data:
                node_data = op.node_data
                if node_data.get("node_type") == "action":
                    cmd = ActuatorCommand(
                        actuator_id=node_data.get("id", "actuator_default"),
                        command_name=node_data.get("intent", "execute"),
                        parameters=node_data.get("params", {}),
                    )
                    dispatch_res = self.dispatch_command_sync(cmd, workspace)
                    delta.add_nodes.append(
                        {
                            "id": f"evt_actuation_{cmd.actuator_id}",
                            "node_type": "event",
                            "event_kind": "actuation_executed"
                            if dispatch_res
                            else "actuation_simulated",
                            "event_data": {"cmd": cmd.command_name, "success": dispatch_res},
                        }
                    )

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        return ExecutionResult(
            delta=delta,
            metrics=ExecutionMetrics(elapsed_ms=elapsed_ms),
            success=True,
        )

    def dispatch_command_sync(
        self,
        command: ActuatorCommand,
        workspace: HCIRWorkspaceState,
    ) -> bool:
        """Dispatch command to hardware if in LIVE mode; simulate if SIMULATION/REPLAY."""
        mode = workspace.branch_mode

        if mode != BranchMode.LIVE:
            logger.info(
                "BRANCH_MODE_SAFETY: Suppressed physical hardware dispatch for '%s' (mode=%s)",
                command.command_name,
                mode,
            )
            return True  # Simulated success in non-live mode

        # LIVE Mode: Invoke hardware handler if registered
        handler = self._handlers.get(command.actuator_id)
        if handler is None:
            logger.warning("No hardware handler registered for actuator '%s'", command.actuator_id)
            return False

        try:
            success = handler(command.parameters)
            logger.info(
                "Executed physical command '%s' on '%s': %s",
                command.command_name,
                command.actuator_id,
                success,
            )
            return success
        except Exception as exc:
            logger.error("Hardware handler exception for '%s': %s", command.actuator_id, exc)
            return False
