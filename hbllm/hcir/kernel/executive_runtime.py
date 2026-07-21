"""
Executive Runtime — Lifecycle Owner & Daemon for HCIR Cognitive OS Kernel.

Manages cycle scheduling, execution modes (LIVE, SIMULATION, REPLAY, TRAINING),
and background event bus subscriptions.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.hcir.kernel.cognitive_kernel import CognitiveKernel
from hbllm.hcir.kernel.executive_controller import CognitiveCycleResult, ExecutiveController
from hbllm.hcir.kernel.runtime_state import RuntimeState
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.workspace import BranchMode

logger = logging.getLogger(__name__)


class ExecutiveRuntime:
    """The lifecycle runtime daemon managing Cognitive OS execution."""

    def __init__(
        self,
        services: KernelServices,
        kernel: CognitiveKernel | None = None,
    ) -> None:
        self._services = services
        self._kernel = kernel or CognitiveKernel(services.workspace)
        self._controller = ExecutiveController(services)
        self._state = RuntimeState()
        self._is_running = False

    @property
    def services(self) -> KernelServices:
        return self._services

    @property
    def kernel(self) -> CognitiveKernel:
        return self._kernel

    @property
    def controller(self) -> ExecutiveController:
        return self._controller

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._is_running

    async def start(self) -> None:
        """Start the executive runtime lifecycle daemon."""
        if self._is_running:
            return
        self._is_running = True
        logger.info(
            "ExecutiveRuntime started [mode: %s, tenant: %s]",
            self._state.branch_mode,
            self._state.tenant_id,
        )

    async def stop(self) -> None:
        """Stop the executive runtime lifecycle daemon."""
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

        result = await self._controller.run_cycle(candidate_actions=candidate_actions)
        logger.debug(
            "ExecutiveRuntime finished cycle #%d in %d ms", result.cycle_index, result.elapsed_ms
        )
        return result
