"""
Executive Runtime — Lifecycle Owner & Daemon for HCIR Cognitive OS Kernel.

Manages cycle scheduling, execution modes (LIVE, SIMULATION, REPLAY, TRAINING),
background event bus subscriptions, and HCIR workspace-driven cognitive cycles.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.hcir.kernel.cognitive_kernel import CognitiveKernel
from hbllm.hcir.kernel.executive_controller import CognitiveCycleResult, KernelExecutiveController
from hbllm.hcir.kernel.runtime_state import RuntimeState
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.workspace import BranchMode

logger = logging.getLogger(__name__)


class ExecutiveRuntime:
    """The lifecycle runtime daemon managing Cognitive OS execution.

    Extended to be workspace-driven: each cycle logs to the
    CognitiveJournal, creates task frames for goals, and notifies
    the tiered workspace of commits.
    """

    def __init__(
        self,
        services: KernelServices,
        kernel: CognitiveKernel | None = None,
    ) -> None:
        self._services = services
        self._kernel = kernel or CognitiveKernel(services.workspace)
        self._controller = KernelExecutiveController(services)
        self._state = RuntimeState()
        self._is_running = False
        self._session_id: str = ""

    @property
    def services(self) -> KernelServices:
        return self._services

    @property
    def kernel(self) -> CognitiveKernel:
        return self._kernel

    @property
    def controller(self) -> KernelExecutiveController:
        return self._controller

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def session_id(self) -> str:
        return self._session_id

    async def start(self) -> None:
        """Start the executive runtime lifecycle daemon."""
        if self._is_running:
            return
        self._is_running = True

        # Start the bus bridge if available
        if self._services.bus_bridge is not None:
            try:
                await self._services.bus_bridge.start()
            except Exception as exc:
                logger.warning("Failed to start bus bridge: %s", exc)

        logger.info(
            "ExecutiveRuntime started [mode: %s, tenant: %s]",
            self._state.branch_mode,
            self._state.tenant_id,
        )

    async def stop(self) -> None:
        """Stop the executive runtime lifecycle daemon."""
        # Stop the bus bridge if available
        if self._services.bus_bridge is not None:
            try:
                await self._services.bus_bridge.stop()
            except Exception as exc:
                logger.warning("Failed to stop bus bridge: %s", exc)

        self._is_running = False
        logger.info("ExecutiveRuntime stopped")

    async def run_cycle(
        self,
        candidate_actions: list[Any] | None = None,
        branch_mode: BranchMode = BranchMode.LIVE,
    ) -> CognitiveCycleResult:
        """Execute a cognitive cycle through the runtime daemon."""
        self._state.cycle_count += 1
        self._state.active_cycle_id = f"cycle_{self._state.cycle_count}"
        self._state.branch_mode = branch_mode

        # Log cycle start to journal if available
        journal = self._services.cognitive_journal
        if journal is not None:
            from hbllm.hcir.cognitive_journal import CognitiveEvent
            from hbllm.hcir.semantic_normalizer import CognitiveEventKind

            journal.record(
                CognitiveEvent(
                    kind=CognitiveEventKind.COGNITIVE_STATE_CHANGED,
                    author="executive_runtime",
                    session_id=self._session_id,
                    data={
                        "cycle_id": self._state.active_cycle_id,
                        "cycle_count": self._state.cycle_count,
                        "branch_mode": branch_mode.value,
                    },
                )
            )

        result = await self._controller.run_cycle(candidate_actions=candidate_actions)

        # Notify tiered workspace of commit
        tiered = self._services.tiered_workspace
        if tiered is not None:
            tiered.notify_commit()

        # Record cycle execution to MigrationMetrics if available
        metrics = self._services.migration_metrics
        if metrics is not None:
            policy = self._services.migration_policy
            mode = getattr(policy, "mode", None)
            from hbllm.hcir.kernel.governance.policies.migration_policy import MigrationMode

            migration_mode = mode if isinstance(mode, MigrationMode) else MigrationMode.HCIR
            metrics.record_execution(
                capability_name="executive_runtime.cycle",
                mode=migration_mode,
                backend="hcir",
                elapsed_ms=result.elapsed_ms,
            )

        logger.debug(
            "ExecutiveRuntime finished cycle #%d in %d ms", result.cycle_index, result.elapsed_ms
        )
        return result

    # ── Session Lifecycle ────────────────────────────────────────────

    async def start_session(self, session_id: str) -> None:
        """Begin a new cognitive session.

        Creates a task frame in the working tier and logs the session
        start event.
        """
        self._session_id = session_id

        tiered = self._services.tiered_workspace
        if tiered is not None:
            tiered.create_task_frame(f"session_{session_id}")

        journal = self._services.cognitive_journal
        if journal is not None:
            from hbllm.hcir.cognitive_journal import CognitiveEvent
            from hbllm.hcir.semantic_normalizer import CognitiveEventKind

            journal.record(
                CognitiveEvent(
                    kind=CognitiveEventKind.COGNITIVE_STATE_CHANGED,
                    author="executive_runtime",
                    session_id=session_id,
                    data={"event": "session_started"},
                )
            )

        logger.info("Session started: %s", session_id)

    async def end_session(self) -> int:
        """End the current cognitive session.

        Archives brain workspace contents to persistent storage
        and closes active task frames.

        Returns:
            Number of nodes archived from brain to persistent.
        """
        archived = 0
        tiered = self._services.tiered_workspace
        if tiered is not None:
            archived = tiered.archive_brain()

        journal = self._services.cognitive_journal
        if journal is not None:
            from hbllm.hcir.cognitive_journal import CognitiveEvent
            from hbllm.hcir.semantic_normalizer import CognitiveEventKind

            journal.record(
                CognitiveEvent(
                    kind=CognitiveEventKind.COGNITIVE_STATE_CHANGED,
                    author="executive_runtime",
                    session_id=self._session_id,
                    data={"event": "session_ended", "archived_count": archived},
                )
            )

        logger.info("Session ended: %s (archived %d nodes)", self._session_id, archived)
        self._session_id = ""
        return archived
