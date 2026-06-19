"""
DelegationChain — Long-running autonomous task execution with progress tracking.

When a user says "handle this", the brain should autonomously plan, execute,
verify, and report back — potentially over hours or days.  DelegationChain
manages the lifecycle of delegated tasks:

    1. User delegates a high-level objective
    2. Brain decomposes into steps (via PlannerNode)
    3. Each step executes autonomously with progress tracking
    4. Sensitive steps pause for user approval
    5. Results are reported via NotificationGateway
    6. State persists across restarts

Bus Topics:
    delegation.create     → New delegation request
    delegation.progress   → Step completed / progress update
    delegation.approval   → User approval for sensitive step
    delegation.complete   → Delegation finished
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Data Types ───────────────────────────────────────────────────────────────


class StepStatus(str, Enum):
    """Status of a delegation step."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"  # Paused for user approval
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class DelegationStatus(str, Enum):
    """Status of the overall delegation."""

    ACTIVE = "active"
    PAUSED = "paused"  # Waiting for user input
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepSensitivity(str, Enum):
    """Whether a step requires user approval before execution."""

    SAFE = "safe"  # Execute automatically
    SENSITIVE = "sensitive"  # Pause and ask for approval
    CRITICAL = "critical"  # Always require explicit approval


@dataclass
class DelegationStep:
    """A single step in a delegation chain."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str = ""
    action: str = ""  # Tool or action to execute
    parameters: dict[str, Any] = field(default_factory=dict)
    sensitivity: StepSensitivity = StepSensitivity.SAFE
    status: StepStatus = StepStatus.PENDING
    # Execution results
    result: str = ""
    error: str = ""
    started_at: float | None = None
    completed_at: float | None = None
    # Dependencies
    depends_on: list[str] = field(default_factory=list)  # Step IDs

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "action": self.action,
            "parameters": self.parameters,
            "sensitivity": self.sensitivity.value,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "depends_on": self.depends_on,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DelegationStep:
        return cls(
            id=data["id"],
            description=data.get("description", ""),
            action=data.get("action", ""),
            parameters=data.get("parameters", {}),
            sensitivity=StepSensitivity(data.get("sensitivity", "safe")),
            status=StepStatus(data.get("status", "pending")),
            result=data.get("result", ""),
            error=data.get("error", ""),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            depends_on=data.get("depends_on", []),
        )


@dataclass
class Delegation:
    """A complete delegation with its step chain."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    tenant_id: str = ""
    objective: str = ""  # User's original request
    steps: list[DelegationStep] = field(default_factory=list)
    status: DelegationStatus = DelegationStatus.ACTIVE
    # Lifecycle
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    # Metadata
    summary: str = ""  # Final summary when completed
    max_retries: int = 2
    retry_count: int = 0

    @property
    def progress(self) -> float:
        """Completion percentage (0.0 to 1.0)."""
        if not self.steps:
            return 0.0
        completed = sum(
            1 for s in self.steps if s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)
        )
        return completed / len(self.steps)

    @property
    def current_step(self) -> DelegationStep | None:
        """The next step to execute."""
        for step in self.steps:
            if step.status in (StepStatus.PENDING, StepStatus.RUNNING, StepStatus.WAITING_APPROVAL):
                return step
        return None

    @property
    def completed_steps(self) -> list[DelegationStep]:
        return [s for s in self.steps if s.status == StepStatus.COMPLETED]

    @property
    def failed_steps(self) -> list[DelegationStep]:
        return [s for s in self.steps if s.status == StepStatus.FAILED]

    def can_execute_step(self, step: DelegationStep) -> bool:
        """Check if a step's dependencies are satisfied."""
        completed_ids = {s.id for s in self.completed_steps}
        return all(dep in completed_ids for dep in step.depends_on)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "objective": self.objective,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "summary": self.summary,
            "progress": round(self.progress, 2),
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Delegation:
        d = cls(
            id=data["id"],
            tenant_id=data.get("tenant_id", ""),
            objective=data.get("objective", ""),
            status=DelegationStatus(data.get("status", "active")),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            completed_at=data.get("completed_at"),
            summary=data.get("summary", ""),
            max_retries=data.get("max_retries", 2),
            retry_count=data.get("retry_count", 0),
        )
        d.steps = [DelegationStep.from_dict(s) for s in data.get("steps", [])]
        return d


# ── DelegationManager ────────────────────────────────────────────────────────


class DelegationManager:
    """
    Manages the lifecycle of delegated tasks.

    Usage:
        manager = DelegationManager(storage_dir="data/delegations")

        # Create a delegation
        delegation = manager.create(
            tenant_id="user_1",
            objective="Deploy the staging environment",
            steps=[
                DelegationStep(description="Run tests", action="shell", ...),
                DelegationStep(description="Build image", action="shell", ...),
                DelegationStep(description="Deploy to k8s", action="shell",
                               sensitivity=StepSensitivity.SENSITIVE, ...),
            ],
        )

        # Execute next step
        step = manager.next_step("user_1", delegation.id)

        # Complete a step
        manager.complete_step("user_1", delegation.id, step.id, result="Tests passed")

        # Approve a sensitive step
        manager.approve_step("user_1", delegation.id, step.id)
    """

    def __init__(self, storage_dir: str | Path = "data/delegations") -> None:
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._delegations: dict[str, dict[str, Delegation]] = {}
        self._load_all()

        logger.info(
            "DelegationManager initialized with %d delegations from %s",
            sum(len(v) for v in self._delegations.values()),
            self._storage_dir,
        )

    def _tenant_dir(self, tenant_id: str) -> Path:
        d = self._storage_dir / tenant_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load_all(self) -> None:
        """Load all delegations from disk."""
        for tenant_dir in self._storage_dir.iterdir():
            if not tenant_dir.is_dir():
                continue
            tenant_id = tenant_dir.name
            self._delegations[tenant_id] = {}
            for path in tenant_dir.glob("*.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    delegation = Delegation.from_dict(data)
                    self._delegations[tenant_id][delegation.id] = delegation
                except Exception as e:
                    logger.warning("Failed to load delegation from %s: %s", path, e)

    def _save(self, delegation: Delegation) -> None:
        """Persist a delegation to disk."""
        path = self._tenant_dir(delegation.tenant_id) / f"{delegation.id}.json"
        try:
            path.write_text(json.dumps(delegation.to_dict(), indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save delegation %s: %s", delegation.id, e)

    def create(
        self,
        tenant_id: str,
        objective: str,
        steps: list[DelegationStep] | None = None,
    ) -> Delegation:
        """Create a new delegation."""
        delegation = Delegation(
            tenant_id=tenant_id,
            objective=objective,
            steps=steps or [],
        )
        if tenant_id not in self._delegations:
            self._delegations[tenant_id] = {}
        self._delegations[tenant_id][delegation.id] = delegation
        self._save(delegation)
        logger.info(
            "Created delegation '%s' (%d steps) for tenant '%s'",
            objective[:50],
            len(delegation.steps),
            tenant_id,
        )
        return delegation

    def get(self, tenant_id: str, delegation_id: str) -> Delegation | None:
        """Get a delegation by ID."""
        return self._delegations.get(tenant_id, {}).get(delegation_id)

    def list_active(self, tenant_id: str) -> list[Delegation]:
        """List active delegations for a tenant."""
        return [
            d
            for d in self._delegations.get(tenant_id, {}).values()
            if d.status == DelegationStatus.ACTIVE
        ]

    def list_all(self, tenant_id: str, limit: int = 50) -> list[Delegation]:
        """List all delegations for a tenant, most recent first."""
        delegations = list(self._delegations.get(tenant_id, {}).values())
        delegations.sort(key=lambda d: d.updated_at, reverse=True)
        return delegations[:limit]

    def next_step(self, tenant_id: str, delegation_id: str) -> DelegationStep | None:
        """Get the next executable step for a delegation."""
        delegation = self.get(tenant_id, delegation_id)
        if not delegation or delegation.status != DelegationStatus.ACTIVE:
            return None

        for step in delegation.steps:
            if step.status != StepStatus.PENDING:
                continue
            if not delegation.can_execute_step(step):
                continue

            # If step is sensitive, pause for approval
            if step.sensitivity != StepSensitivity.SAFE:
                step.status = StepStatus.WAITING_APPROVAL
                delegation.status = DelegationStatus.PAUSED
                delegation.updated_at = time.time()
                self._save(delegation)
                return step

            # Mark as running
            step.status = StepStatus.RUNNING
            step.started_at = time.time()
            delegation.updated_at = time.time()
            self._save(delegation)
            return step

        return None

    def approve_step(self, tenant_id: str, delegation_id: str, step_id: str) -> bool:
        """Approve a step that's waiting for user approval."""
        delegation = self.get(tenant_id, delegation_id)
        if not delegation:
            return False

        for step in delegation.steps:
            if step.id == step_id and step.status == StepStatus.WAITING_APPROVAL:
                step.status = StepStatus.RUNNING
                step.started_at = time.time()
                delegation.status = DelegationStatus.ACTIVE
                delegation.updated_at = time.time()
                self._save(delegation)
                logger.info("Step '%s' approved in delegation %s", step.description, delegation_id)
                return True
        return False

    def complete_step(
        self,
        tenant_id: str,
        delegation_id: str,
        step_id: str,
        result: str = "",
    ) -> bool:
        """Mark a step as completed."""
        delegation = self.get(tenant_id, delegation_id)
        if not delegation:
            return False

        for step in delegation.steps:
            if step.id == step_id:
                step.status = StepStatus.COMPLETED
                step.result = result
                step.completed_at = time.time()
                delegation.updated_at = time.time()

                # Check if all steps are done
                if delegation.progress >= 1.0:
                    delegation.status = DelegationStatus.COMPLETED
                    delegation.completed_at = time.time()
                    logger.info(
                        "Delegation completed: '%s' for tenant '%s'",
                        delegation.objective[:50],
                        tenant_id,
                    )

                self._save(delegation)
                return True
        return False

    def fail_step(
        self,
        tenant_id: str,
        delegation_id: str,
        step_id: str,
        error: str = "",
    ) -> bool:
        """Mark a step as failed."""
        delegation = self.get(tenant_id, delegation_id)
        if not delegation:
            return False

        for step in delegation.steps:
            if step.id == step_id:
                step.status = StepStatus.FAILED
                step.error = error
                step.completed_at = time.time()
                delegation.updated_at = time.time()

                # Check if we should retry or fail the whole delegation
                if delegation.retry_count < delegation.max_retries:
                    delegation.retry_count += 1
                    step.status = StepStatus.PENDING  # Reset for retry
                    step.started_at = None
                    step.completed_at = None
                    logger.warning(
                        "Step '%s' failed (retry %d/%d): %s",
                        step.description,
                        delegation.retry_count,
                        delegation.max_retries,
                        error,
                    )
                else:
                    delegation.status = DelegationStatus.FAILED
                    logger.error(
                        "Delegation failed: '%s' (step: %s, error: %s)",
                        delegation.objective[:50],
                        step.description,
                        error,
                    )

                self._save(delegation)
                return True
        return False

    def cancel(self, tenant_id: str, delegation_id: str) -> bool:
        """Cancel an active delegation."""
        delegation = self.get(tenant_id, delegation_id)
        if delegation and delegation.status in (DelegationStatus.ACTIVE, DelegationStatus.PAUSED):
            delegation.status = DelegationStatus.CANCELLED
            delegation.updated_at = time.time()
            self._save(delegation)
            logger.info("Delegation cancelled: '%s'", delegation.objective[:50])
            return True
        return False

    def stats(self, tenant_id: str) -> dict[str, Any]:
        """Stats for a tenant's delegations."""
        delegations = list(self._delegations.get(tenant_id, {}).values())
        return {
            "total": len(delegations),
            "active": sum(1 for d in delegations if d.status == DelegationStatus.ACTIVE),
            "paused": sum(1 for d in delegations if d.status == DelegationStatus.PAUSED),
            "completed": sum(1 for d in delegations if d.status == DelegationStatus.COMPLETED),
            "failed": sum(1 for d in delegations if d.status == DelegationStatus.FAILED),
        }
