"""Parallel Workspace — multiple reasoning threads.

Enables the brain to work on multiple tasks simultaneously:
    - Background research while conversing
    - Parallel goal evaluation
    - Speculative pre-computation
    - Multi-hypothesis reasoning

Architecture:
    Each workspace is an isolated reasoning context with:
    - Its own message history
    - Its own working memory
    - Access to shared long-term memory
    - Priority-based CPU scheduling
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class WorkspaceStatus(str, Enum):
    """Workspace lifecycle states."""

    IDLE = "idle"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Workspace:
    """An isolated reasoning context."""

    workspace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""
    task_description: str = ""
    status: WorkspaceStatus = WorkspaceStatus.IDLE
    priority: int = 0  # 0=background, 1=normal, 2=high, 3=foreground
    messages: list[dict[str, str]] = field(default_factory=list)
    working_memory: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    parent_workspace_id: str | None = None  # For sub-workspaces

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "name": self.name,
            "task": self.task_description[:100],
            "status": self.status.value,
            "priority": self.priority,
            "message_count": len(self.messages),
            "created_at": self.created_at,
            "duration_s": (
                (self.completed_at or time.time()) - (self.started_at or self.created_at)
            ),
        }


class ParallelWorkspaceManager:
    """Manages multiple concurrent reasoning workspaces.

    Usage::

        manager = ParallelWorkspaceManager(provider=llm_provider, max_concurrent=3)

        # Create a background workspace
        ws_id = manager.create_workspace(
            name="Research",
            task="Find the best restaurants near the office",
            priority=0,  # Background
        )

        # Run it
        result = await manager.run_workspace(ws_id)

        # Check all workspaces
        statuses = manager.get_all_statuses()
    """

    def __init__(
        self,
        provider: Any | None = None,
        max_concurrent: int = 3,
    ) -> None:
        self.provider = provider
        self.max_concurrent = max_concurrent

        self._workspaces: dict[str, Workspace] = {}
        self._running_tasks: dict[str, asyncio.Task[Any]] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Telemetry
        self._total_created = 0
        self._total_completed = 0
        self._total_failed = 0

    def create_workspace(
        self,
        name: str,
        task: str,
        priority: int = 1,
        parent_id: str | None = None,
        initial_context: dict[str, Any] | None = None,
    ) -> str:
        """Create a new reasoning workspace.

        Returns the workspace ID.
        """
        ws = Workspace(
            name=name,
            task_description=task,
            priority=priority,
            parent_workspace_id=parent_id,
            working_memory=initial_context or {},
        )
        self._workspaces[ws.workspace_id] = ws
        self._total_created += 1
        logger.info("Created workspace '%s' (id=%s, priority=%d)", name, ws.workspace_id, priority)
        return ws.workspace_id

    async def run_workspace(
        self,
        workspace_id: str,
        timeout_s: float = 300.0,
    ) -> Any:
        """Run a workspace's task using the LLM provider.

        Blocks until the workspace completes or times out.
        """
        ws = self._workspaces.get(workspace_id)
        if not ws:
            raise ValueError(f"Workspace {workspace_id} not found")

        async with self._semaphore:
            ws.status = WorkspaceStatus.ACTIVE
            ws.started_at = time.time()

            try:
                if self.provider:
                    result = await asyncio.wait_for(
                        self._execute_with_provider(ws),
                        timeout=timeout_s,
                    )
                else:
                    result = {"status": "no_provider", "task": ws.task_description}

                ws.result = result
                ws.status = WorkspaceStatus.COMPLETED
                ws.completed_at = time.time()
                self._total_completed += 1
                return result

            except asyncio.TimeoutError:
                ws.status = WorkspaceStatus.FAILED
                ws.completed_at = time.time()
                self._total_failed += 1
                raise

            except Exception as e:
                ws.status = WorkspaceStatus.FAILED
                ws.completed_at = time.time()
                ws.result = {"error": str(e)}
                self._total_failed += 1
                raise

    async def run_in_background(self, workspace_id: str) -> None:
        """Start a workspace running in the background."""
        task = asyncio.create_task(self.run_workspace(workspace_id))
        self._running_tasks[workspace_id] = task

        def _cleanup(t: asyncio.Task[Any]) -> None:
            self._running_tasks.pop(workspace_id, None)

        task.add_done_callback(_cleanup)

    async def _execute_with_provider(self, ws: Workspace) -> Any:
        """Execute the workspace task using the LLM provider."""
        # Build the workspace prompt
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are working in a parallel workspace named '{ws.name}'. "
                    f"Your task: {ws.task_description}\n"
                    f"Context: {ws.working_memory}"
                ),
            },
            {"role": "user", "content": ws.task_description},
        ]
        messages.extend(ws.messages)

        response = await self.provider.generate(
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
        )

        content = response.get("content", "") if isinstance(response, dict) else str(response)
        return {"content": content, "workspace": ws.name}

    def suspend_workspace(self, workspace_id: str) -> None:
        """Suspend a running workspace."""
        ws = self._workspaces.get(workspace_id)
        if ws and ws.status == WorkspaceStatus.ACTIVE:
            ws.status = WorkspaceStatus.SUSPENDED
            task = self._running_tasks.get(workspace_id)
            if task:
                task.cancel()

    def resume_workspace(self, workspace_id: str) -> None:
        """Resume a suspended workspace."""
        ws = self._workspaces.get(workspace_id)
        if ws and ws.status == WorkspaceStatus.SUSPENDED:
            ws.status = WorkspaceStatus.IDLE  # Ready to be run again

    def get_workspace(self, workspace_id: str) -> Workspace | None:
        return self._workspaces.get(workspace_id)

    def get_all_statuses(self) -> list[dict[str, Any]]:
        """Get status of all workspaces."""
        return [ws.to_dict() for ws in self._workspaces.values()]

    def cleanup_completed(self, max_age_s: float = 3600) -> int:
        """Remove completed workspaces older than max_age_s."""
        now = time.time()
        to_remove = [
            ws_id
            for ws_id, ws in self._workspaces.items()
            if ws.status in (WorkspaceStatus.COMPLETED, WorkspaceStatus.FAILED)
            and ws.completed_at
            and now - ws.completed_at > max_age_s
        ]
        for ws_id in to_remove:
            del self._workspaces[ws_id]
        return len(to_remove)

    def stats(self) -> dict[str, Any]:
        statuses = {}
        for ws in self._workspaces.values():
            statuses[ws.status.value] = statuses.get(ws.status.value, 0) + 1
        return {
            "total_created": self._total_created,
            "total_completed": self._total_completed,
            "total_failed": self._total_failed,
            "active_workspaces": len(self._workspaces),
            "running_tasks": len(self._running_tasks),
            "max_concurrent": self.max_concurrent,
            "by_status": statuses,
        }
