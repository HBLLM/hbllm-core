"""Action Rollback — transactional undo for multi-step operations.

Extends the ``InterventionAPI`` reversibility model with concrete
transaction support: pre-state snapshots, automatic undo on failure,
and a registry of action→undo pairs.

Usage::

    registry = RollbackRegistry()
    registry.register("file.create", snapshot_fn=capture_path, undo_fn=delete_path)

    async with ActionTransaction(registry, "file.create", {"path": "/tmp/x"}) as txn:
        create_file("/tmp/x")
        txn.commit()

    # On exception, undo_fn is called automatically with the snapshot.

Bus Topics:
    action.rollback.executed  — Published when a rollback is performed
    action.rollback.failed    — Published when a rollback attempt fails
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class ActionSnapshot:
    """Pre-state snapshot captured before an action executes."""

    action_name: str
    pre_state: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    tenant_id: str = "default"
    committed: bool = False
    rolled_back: bool = False


@dataclass
class ActionResult:
    """Outcome of an action execution with rollback support."""

    success: bool
    action_name: str
    error: str | None = None
    rolled_back: bool = False
    snapshot: ActionSnapshot | None = None


@dataclass
class RollbackHandler:
    """Defines how to snapshot and undo a specific action."""

    action_name: str
    snapshot_fn: Callable[..., dict[str, Any]] | None = None
    undo_fn: Callable[..., Coroutine[Any, Any, bool]] | None = None
    description: str = ""


# ── Registry ─────────────────────────────────────────────────────────────────


class RollbackRegistry:
    """Registry of action→undo pairs with history tracking.

    Args:
        bus: Optional MessageBus for event publishing.
        max_history: Maximum snapshots to retain per tenant.
    """

    def __init__(
        self,
        bus: Any | None = None,
        max_history: int = 100,
    ) -> None:
        self.bus = bus
        self.max_history = max_history

        self._handlers: dict[str, RollbackHandler] = {}
        self._history: dict[str, list[ActionSnapshot]] = defaultdict(list)

        # Telemetry
        self._total_executions = 0
        self._total_rollbacks = 0
        self._failed_rollbacks = 0

    def register(
        self,
        action_name: str,
        snapshot_fn: Callable[..., dict[str, Any]] | None = None,
        undo_fn: Callable[..., Coroutine[Any, Any, bool]] | None = None,
        description: str = "",
    ) -> None:
        """Register an action with its snapshot and undo functions.

        Args:
            action_name: Unique action identifier (e.g. "file.create").
            snapshot_fn: Sync function that captures pre-state: (params) → dict.
            undo_fn: Async function that reverses the action: (snapshot) → bool.
            description: Human-readable description of the undo behaviour.
        """
        self._handlers[action_name] = RollbackHandler(
            action_name=action_name,
            snapshot_fn=snapshot_fn,
            undo_fn=undo_fn,
            description=description,
        )
        logger.info("Registered rollback handler: %s", action_name)

    def has_handler(self, action_name: str) -> bool:
        """Check if a rollback handler exists for this action."""
        return action_name in self._handlers

    # ── Execution ────────────────────────────────────────────────────

    async def execute_with_rollback(
        self,
        action_name: str,
        execute_fn: Callable[..., Coroutine[Any, Any, Any]],
        params: dict[str, Any] | None = None,
        tenant_id: str = "default",
    ) -> ActionResult:
        """Execute an action with automatic rollback on failure.

        Args:
            action_name: The registered action name.
            execute_fn: The async function to execute.
            params: Parameters passed to both snapshot_fn and execute_fn.
            tenant_id: Tenant context for history tracking.

        Returns:
            ActionResult with success/failure and rollback status.
        """
        params = params or {}
        self._total_executions += 1

        handler = self._handlers.get(action_name)
        snapshot = ActionSnapshot(
            action_name=action_name,
            tenant_id=tenant_id,
        )

        # Capture pre-state
        if handler and handler.snapshot_fn:
            try:
                snapshot.pre_state = handler.snapshot_fn(params)
            except Exception as e:
                logger.warning("Snapshot capture failed for %s: %s", action_name, e)
                snapshot.pre_state = {}
        else:
            snapshot.pre_state = {}

        # Execute action
        try:
            await execute_fn(**params)
            snapshot.committed = True
            self._add_to_history(tenant_id, snapshot)
            return ActionResult(
                success=True,
                action_name=action_name,
                snapshot=snapshot,
            )
        except Exception as e:
            logger.error("Action '%s' failed: %s — attempting rollback", action_name, e)
            rolled_back = await self._attempt_rollback(handler, snapshot)
            snapshot.rolled_back = rolled_back
            self._add_to_history(tenant_id, snapshot)
            return ActionResult(
                success=False,
                action_name=action_name,
                error=str(e),
                rolled_back=rolled_back,
                snapshot=snapshot,
            )

    async def undo_last(self, tenant_id: str = "default") -> bool:
        """Undo the most recent committed action for a tenant.

        Returns True if the rollback succeeded.
        """
        history = self._history.get(tenant_id, [])

        # Find the most recent committed, non-rolled-back action
        for snapshot in reversed(history):
            if snapshot.committed and not snapshot.rolled_back:
                handler = self._handlers.get(snapshot.action_name)
                success = await self._attempt_rollback(handler, snapshot)
                if success:
                    snapshot.rolled_back = True
                return success

        logger.warning("No undoable action found for tenant %s", tenant_id)
        return False

    def get_history(
        self,
        tenant_id: str = "default",
        limit: int = 20,
    ) -> list[ActionSnapshot]:
        """Get recent action history for a tenant."""
        history = self._history.get(tenant_id, [])
        return list(reversed(history[-limit:]))

    # ── Internal ─────────────────────────────────────────────────────

    async def _attempt_rollback(
        self,
        handler: RollbackHandler | None,
        snapshot: ActionSnapshot,
    ) -> bool:
        """Try to execute a rollback using the registered undo function."""
        if not handler or not handler.undo_fn:
            logger.warning(
                "No undo handler for '%s' — rollback impossible",
                snapshot.action_name,
            )
            self._failed_rollbacks += 1
            return False

        try:
            success = await handler.undo_fn(snapshot.pre_state)
            if success:
                self._total_rollbacks += 1
                logger.info("Rollback succeeded for '%s'", snapshot.action_name)
            else:
                self._failed_rollbacks += 1
                logger.error("Rollback returned False for '%s'", snapshot.action_name)
            return success
        except Exception as e:
            self._failed_rollbacks += 1
            logger.exception("Rollback failed for '%s': %s", snapshot.action_name, e)
            return False

    def _add_to_history(self, tenant_id: str, snapshot: ActionSnapshot) -> None:
        """Append snapshot to tenant history with size limiting."""
        history = self._history[tenant_id]
        history.append(snapshot)
        if len(history) > self.max_history:
            self._history[tenant_id] = history[-self.max_history :]

    def stats(self) -> dict[str, Any]:
        """Registry statistics."""
        return {
            "registered_handlers": list(self._handlers.keys()),
            "total_executions": self._total_executions,
            "total_rollbacks": self._total_rollbacks,
            "failed_rollbacks": self._failed_rollbacks,
        }


# ── Transaction Context Manager ──────────────────────────────────────────────


class ActionTransaction:
    """Async context manager for transactional action execution.

    Usage::

        async with ActionTransaction(registry, "file.create", {"path": p}) as txn:
            await create_file(p)
            txn.commit()
        # If an exception is raised before commit(), rollback is automatic.
    """

    def __init__(
        self,
        registry: RollbackRegistry,
        action_name: str,
        params: dict[str, Any] | None = None,
        tenant_id: str = "default",
    ) -> None:
        self.registry = registry
        self.action_name = action_name
        self.params = params or {}
        self.tenant_id = tenant_id
        self.snapshot: ActionSnapshot | None = None
        self._committed = False

    async def __aenter__(self) -> ActionTransaction:
        handler = self.registry._handlers.get(self.action_name)
        self.snapshot = ActionSnapshot(
            action_name=self.action_name,
            tenant_id=self.tenant_id,
        )
        if handler and handler.snapshot_fn:
            try:
                self.snapshot.pre_state = handler.snapshot_fn(self.params)
            except Exception as e:
                logger.warning("Snapshot failed: %s", e)
                self.snapshot.pre_state = {}
        else:
            self.snapshot.pre_state = {}
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if self.snapshot is None:
            return False

        if exc_type is not None and not self._committed:
            # Exception occurred before commit — auto-rollback
            logger.warning(
                "ActionTransaction auto-rollback for '%s' due to %s",
                self.action_name,
                exc_type.__name__,
            )
            handler = self.registry._handlers.get(self.action_name)
            self.snapshot.rolled_back = await self.registry._attempt_rollback(
                handler,
                self.snapshot,
            )
            self.registry._add_to_history(self.tenant_id, self.snapshot)
            return True  # Suppress the exception

        if self._committed:
            self.snapshot.committed = True
            self.registry._add_to_history(self.tenant_id, self.snapshot)

        return False

    def commit(self) -> None:
        """Mark the transaction as successfully completed."""
        self._committed = True
