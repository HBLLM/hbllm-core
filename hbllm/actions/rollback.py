"""Action Rollback — transactional multi-step action execution with undo.

Wraps sequences of tool/action calls in a transaction:
    1. Records pre-execution state snapshots
    2. Executes steps in order
    3. On failure, runs compensating actions in reverse order
    4. Logs everything to the audit trail

Components:
    - ActionTransaction: wraps a multi-step execution sequence
    - RollbackRegistry: maps action types → their undo counterparts
    - StateSnapshot: captures pre-execution state for comparison

Integrates with IdempotencyEngine to prevent double-rollback.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── Types ────────────────────────────────────────────────────────────────


class StepStatus(str, Enum):
    """Status of a transaction step."""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    ROLLBACK_FAILED = "rollback_failed"


class TransactionStatus(str, Enum):
    """Overall transaction status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PARTIALLY_FAILED = "partially_failed"
    ROLLED_BACK = "rolled_back"
    ROLLBACK_FAILED = "rollback_failed"


@dataclass
class StateSnapshot:
    """Pre-execution state captured for rollback comparison."""

    key: str  # Identifier for what was captured
    value: Any = None  # The state value
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": str(self.value)[:200] if self.value is not None else None,
            "timestamp": self.timestamp,
        }


@dataclass
class TransactionStep:
    """A single step in a multi-step action transaction."""

    step_id: int
    action_type: str  # e.g., "iot.light.on", "file.create"
    params: dict[str, Any] = field(default_factory=dict)
    pre_state: StateSnapshot | None = None
    result: Any = None
    error: str | None = None
    status: StepStatus = StepStatus.PENDING
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action_type": self.action_type,
            "params": self.params,
            "status": self.status.value,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "pre_state": self.pre_state.to_dict() if self.pre_state else None,
        }


# ── Rollback Registry ───────────────────────────────────────────────────


class RollbackRegistry:
    """Maps action types to their compensating (undo) actions.

    Usage::

        registry = RollbackRegistry()

        # Get undo for a known action
        undo = registry.get_undo("file.create")
        # Returns {"action": "file.delete", "param_map": {"path": "path"}}

        # Register a custom undo
        registry.register("custom.deploy", "custom.rollback", {"version": "previous_version"})
    """

    def __init__(self) -> None:
        # Built-in undo mappings: action → (undo_action, param_mapping)
        self._registry: dict[str, dict[str, Any]] = {
            # File operations
            "file.create": {
                "action": "file.delete",
                "param_map": {"path": "path"},
            },
            "file.delete": {
                "action": "file.create",
                "param_map": {"path": "path"},
                "requires_pre_state": True,  # Need file content from snapshot
            },
            "file.write": {
                "action": "file.write",
                "param_map": {"path": "path"},
                "requires_pre_state": True,  # Restore previous content
            },
            "file.rename": {
                "action": "file.rename",
                "param_map": {"src": "dst", "dst": "src"},  # Swap src/dst
            },
            # IoT device operations
            "iot.light.on": {
                "action": "iot.light.off",
                "param_map": {"device_id": "device_id"},
            },
            "iot.light.off": {
                "action": "iot.light.on",
                "param_map": {"device_id": "device_id"},
            },
            "iot.thermostat.set": {
                "action": "iot.thermostat.set",
                "param_map": {"device_id": "device_id"},
                "requires_pre_state": True,  # Restore previous temperature
            },
            "iot.lock.lock": {
                "action": "iot.lock.unlock",
                "param_map": {"device_id": "device_id"},
            },
            "iot.lock.unlock": {
                "action": "iot.lock.lock",
                "param_map": {"device_id": "device_id"},
            },
            # System operations
            "system.volume.set": {
                "action": "system.volume.set",
                "param_map": {"device_id": "device_id"},
                "requires_pre_state": True,
            },
            "system.brightness.set": {
                "action": "system.brightness.set",
                "param_map": {"device_id": "device_id"},
                "requires_pre_state": True,
            },
            "system.notification.send": {
                "action": None,  # Cannot undo a sent notification
                "param_map": {},
            },
        }

    def register(
        self,
        action_type: str,
        undo_action: str | None,
        param_map: dict[str, str] | None = None,
        requires_pre_state: bool = False,
    ) -> None:
        """Register a custom undo mapping."""
        self._registry[action_type] = {
            "action": undo_action,
            "param_map": param_map or {},
            "requires_pre_state": requires_pre_state,
        }

    def get_undo(self, action_type: str) -> dict[str, Any] | None:
        """Get the undo mapping for an action type."""
        return self._registry.get(action_type)

    def is_reversible(self, action_type: str) -> bool:
        """Check if an action type has a known undo mapping."""
        mapping = self._registry.get(action_type)
        return mapping is not None and mapping.get("action") is not None

    def build_undo_params(
        self,
        action_type: str,
        original_params: dict[str, Any],
        pre_state: StateSnapshot | None = None,
    ) -> dict[str, Any] | None:
        """Build the parameters for an undo action.

        Args:
            action_type: The original action type.
            original_params: The parameters used in the original action.
            pre_state: Pre-execution state snapshot (for state-dependent undos).

        Returns:
            Dictionary of undo parameters, or None if not reversible.
        """
        mapping = self._registry.get(action_type)
        if not mapping or not mapping.get("action"):
            return None

        undo_params: dict[str, Any] = {}

        # Map parameters using the param_map
        for undo_key, orig_key in mapping.get("param_map", {}).items():
            if orig_key in original_params:
                undo_params[undo_key] = original_params[orig_key]

        # If undo requires pre-state, inject it
        if mapping.get("requires_pre_state") and pre_state:
            undo_params["value"] = pre_state.value

        return undo_params

    def list_reversible(self) -> list[str]:
        """List all action types that have registered undo mappings."""
        return [k for k, v in self._registry.items() if v.get("action") is not None]

    def stats(self) -> dict[str, Any]:
        return {
            "total_registered": len(self._registry),
            "reversible": len(self.list_reversible()),
            "irreversible": len(self._registry) - len(self.list_reversible()),
        }


# ── Action Transaction ──────────────────────────────────────────────────


class ActionTransaction:
    """Wraps a multi-step action sequence with rollback support.

    Usage::

        registry = RollbackRegistry()
        tx = ActionTransaction(
            transaction_id="tx-001",
            rollback_registry=registry,
            executor=my_action_executor,
        )

        tx.add_step("iot.light.on", {"device_id": "living_room"})
        tx.add_step("iot.thermostat.set", {"device_id": "hvac", "temp": 22})
        tx.add_step("system.notification.send", {"msg": "Good morning!"})

        result = await tx.execute()
        if result.status == TransactionStatus.ROLLED_BACK:
            print("Transaction failed and was rolled back:", result.error)
    """

    def __init__(
        self,
        transaction_id: str,
        rollback_registry: RollbackRegistry | None = None,
        executor: Callable[..., Awaitable[Any]] | None = None,
        state_reader: Callable[..., Awaitable[StateSnapshot]] | None = None,
        auto_rollback: bool = True,
    ) -> None:
        self.transaction_id = transaction_id
        self.registry = rollback_registry or RollbackRegistry()
        self._executor = executor
        self._state_reader = state_reader
        self.auto_rollback = auto_rollback

        self.steps: list[TransactionStep] = []
        self.status = TransactionStatus.PENDING
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.error: str | None = None

    def add_step(
        self,
        action_type: str,
        params: dict[str, Any] | None = None,
    ) -> int:
        """Add a step to the transaction. Returns step ID."""
        step_id = len(self.steps)
        self.steps.append(
            TransactionStep(
                step_id=step_id,
                action_type=action_type,
                params=params or {},
            )
        )
        return step_id

    async def execute(self) -> ActionTransaction:
        """Execute all steps in order. Rollback on failure if auto_rollback is True."""
        self.status = TransactionStatus.IN_PROGRESS
        self.start_time = time.time()

        completed_steps: list[TransactionStep] = []

        for step in self.steps:
            step.status = StepStatus.EXECUTING
            step_start = time.monotonic()

            try:
                # Capture pre-state if undo requires it
                undo_mapping = self.registry.get_undo(step.action_type)
                if undo_mapping and undo_mapping.get("requires_pre_state") and self._state_reader:
                    step.pre_state = await self._state_reader(step.action_type, step.params)

                # Execute the action
                if self._executor:
                    step.result = await self._executor(step.action_type, step.params)
                else:
                    # Dry-run mode — just record
                    step.result = {"dry_run": True, "action": step.action_type}

                step.status = StepStatus.COMPLETED
                step.duration_ms = (time.monotonic() - step_start) * 1000
                completed_steps.append(step)

                logger.debug(
                    "TX %s step %d completed: %s",
                    self.transaction_id,
                    step.step_id,
                    step.action_type,
                )

            except Exception as e:
                step.status = StepStatus.FAILED
                step.error = str(e)
                step.duration_ms = (time.monotonic() - step_start) * 1000
                self.error = f"Step {step.step_id} ({step.action_type}) failed: {e}"

                logger.warning(
                    "TX %s step %d failed: %s — %s",
                    self.transaction_id,
                    step.step_id,
                    step.action_type,
                    e,
                )

                if self.auto_rollback and completed_steps:
                    await self._rollback(completed_steps)
                else:
                    self.status = TransactionStatus.PARTIALLY_FAILED

                break

        else:
            # All steps completed successfully
            self.status = TransactionStatus.COMPLETED

        self.end_time = time.time()
        return self

    async def _rollback(self, completed_steps: list[TransactionStep]) -> None:
        """Roll back completed steps in reverse order."""
        logger.info(
            "TX %s rolling back %d completed steps",
            self.transaction_id,
            len(completed_steps),
        )

        all_rolled_back = True

        for step in reversed(completed_steps):
            undo_params = self.registry.build_undo_params(
                step.action_type, step.params, step.pre_state
            )

            if undo_params is None:
                logger.debug(
                    "TX %s: step %d (%s) is irreversible, skipping rollback",
                    self.transaction_id,
                    step.step_id,
                    step.action_type,
                )
                continue

            undo_mapping = self.registry.get_undo(step.action_type)
            if not undo_mapping or not undo_mapping.get("action"):
                continue

            try:
                if self._executor:
                    await self._executor(undo_mapping["action"], undo_params)
                step.status = StepStatus.ROLLED_BACK

                logger.debug(
                    "TX %s: step %d rolled back via %s",
                    self.transaction_id,
                    step.step_id,
                    undo_mapping["action"],
                )

            except Exception as e:
                step.status = StepStatus.ROLLBACK_FAILED
                step.error = f"Rollback failed: {e}"
                all_rolled_back = False

                logger.error(
                    "TX %s: rollback failed for step %d: %s",
                    self.transaction_id,
                    step.step_id,
                    e,
                )

        self.status = (
            TransactionStatus.ROLLED_BACK if all_rolled_back else TransactionStatus.ROLLBACK_FAILED
        )

    async def rollback_all(self) -> None:
        """Manually trigger rollback of all completed steps."""
        completed = [s for s in self.steps if s.status == StepStatus.COMPLETED]
        if completed:
            await self._rollback(completed)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the transaction for audit logging."""
        return {
            "transaction_id": self.transaction_id,
            "status": self.status.value,
            "steps": [s.to_dict() for s in self.steps],
            "error": self.error,
            "duration_ms": (self.end_time - self.start_time) * 1000 if self.end_time else 0,
        }

    def stats(self) -> dict[str, Any]:
        """Transaction statistics."""
        return {
            "transaction_id": self.transaction_id,
            "status": self.status.value,
            "total_steps": len(self.steps),
            "completed": sum(1 for s in self.steps if s.status == StepStatus.COMPLETED),
            "failed": sum(1 for s in self.steps if s.status == StepStatus.FAILED),
            "rolled_back": sum(1 for s in self.steps if s.status == StepStatus.ROLLED_BACK),
        }
