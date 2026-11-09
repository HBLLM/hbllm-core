"""
Executive Runtime — Lifecycle Owner & Daemon for HCIR Cognitive OS Kernel.

Manages cycle scheduling, execution modes (LIVE, SIMULATION, REPLAY, TRAINING),
background event bus subscriptions, and HCIR workspace-driven cognitive cycles.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from hbllm.hcir.kernel.cognitive_kernel import CognitiveKernel
from hbllm.hcir.kernel.executive_controller import CognitiveCycleResult, KernelExecutiveController
from hbllm.hcir.kernel.runtime_state import RuntimeState
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.workspace import BranchMode

logger = logging.getLogger(__name__)


class ExecutiveRuntime:
    """The lifecycle runtime daemon managing Cognitive OS execution.

    Operates on a dual-rhythm cognitive architecture:
    1. Online Reactive: Cycle triggered strictly by incoming observation envelopes
       from the ObservationBus.
    2. Offline Idle Consolidation: Triggered when the sensory bus is quiet for
       Δt > idle_threshold. Runs hippocampal replay, concept maintenance,
       and theory pruning committed under author='offline_consolidation'.
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
        self._services.runtime_state = self._state
        self._is_running = False
        self._session_id: str = ""

        # Dual-Rhythm Autonomy
        self._observation_bus: Any | None = None
        self._idle_threshold: float = 2.0
        self._last_observation_time: float = 0.0
        self._is_idle_consolidated: bool = False
        self._consolidation_count: int = 0
        self._autonomous_task: asyncio.Task[None] | None = None
        self._engine: Any | None = None
        self._concept_learner: Any | None = None
        self._episodic_cortex: Any | None = None

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

    @property
    def last_observation_time(self) -> float:
        return self._last_observation_time

    @property
    def idle_threshold(self) -> float:
        return self._idle_threshold

    @property
    def consolidation_count(self) -> int:
        return self._consolidation_count

    @property
    def is_idle_consolidated(self) -> bool:
        return self._is_idle_consolidated

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
        await self.stop_autonomous_loop()
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

    # ── Dual-Rhythm Autonomy ──────────────────────────────────────────

    def attach_observation_bus(self, bus: Any) -> None:
        """Bind an ObservationBus to receive incoming multimodal envelopes for reactive execution."""
        self._observation_bus = bus
        if hasattr(bus, "subscribe"):
            bus.subscribe(self._on_observation)
        self._last_observation_time = time.monotonic()

    def _on_observation(self, envelope: Any) -> None:
        """Handle incoming observation: update timestamps and reset idle consolidation flag."""
        self._last_observation_time = time.monotonic()
        self._is_idle_consolidated = False

    def attach_engine(self, engine: Any) -> None:
        """Bind the AutonomousEpistemicEngine instance."""
        self._engine = engine
        if hasattr(engine, "episodic_cortex") and engine.episodic_cortex is not None:
            self._episodic_cortex = engine.episodic_cortex

    def attach_concept_learner(self, learner: Any) -> None:
        """Bind the SGE ConceptLearner instance."""
        self._concept_learner = learner

    def attach_episodic_cortex(self, cortex: Any) -> None:
        """Bind the HippocampalEpisodicCortex instance."""
        self._episodic_cortex = cortex

    async def run_offline_consolidation(self) -> Any | None:
        """Execute an offline idle consolidation cycle.

        Executes:
        1. Hippocampal replay of recent episodic traces
        2. Concept maintenance (splits/merges) in ConceptLearner
        3. Schema and hypothesis pruning in HCIRSymbolicWorldTheory
        4. State snapshotting and persistent tier commit under author='offline_consolidation'
        """
        # 1. Hippocampal replay
        if self._episodic_cortex is not None:
            try:
                trajectories = getattr(self._episodic_cortex, "successful_trajectories", [])
                if not trajectories:
                    trajectories = getattr(self._episodic_cortex, "past_episodes", [])
                for traj in trajectories[-3:]:
                    if traj and hasattr(self._episodic_cortex, "replay_forward_and_reverse"):
                        self._episodic_cortex.replay_forward_and_reverse(traj, terminal_reward=1.0)
            except Exception as exc:
                logger.debug("Hippocampal replay error during offline consolidation: %s", exc)

        # 2. Concept maintenance
        if self._concept_learner is not None and hasattr(self._concept_learner, "maintain"):
            try:
                self._concept_learner.maintain()
            except Exception as exc:
                logger.debug("Concept maintenance error during offline consolidation: %s", exc)

        # 3. Schema pruning
        if self._engine is not None and hasattr(self._engine, "symbolic_theory"):
            theory = self._engine.symbolic_theory
            if hasattr(theory, "prune_stale_hypotheses"):
                try:
                    theory.prune_stale_hypotheses()
                except Exception as exc:
                    logger.debug("Schema pruning error during offline consolidation: %s", exc)

        # 4. Propose and commit consolidation state
        dest_ws = None
        grant = None
        tiered = self._services.tiered_workspace
        if tiered is not None:
            dest_ws = tiered.persistent
            grant = getattr(tiered, "_system_grant", None)
        else:
            dest_ws = self._services.workspace

        tx = None
        if dest_ws is not None:
            ops: list[Any] = []
            if self._engine is not None and hasattr(self._engine, "to_transaction_operations"):
                ops = self._engine.to_transaction_operations()

            from hbllm.hcir.nodes import ConceptNode
            from hbllm.hcir.transactions import TransactionOp, TransactionOperation

            marker = ConceptNode(
                id=f"consolidation_marker_{self._consolidation_count + 1}",
                label=f"consolidation_{self._consolidation_count + 1}",
                definition="Offline idle consolidation snapshot",
                domain="cognitive_maintenance",
                properties={
                    "timestamp": time.time(),
                    "consolidation_index": self._consolidation_count + 1,
                    "cycle_count": self._state.cycle_count,
                },
            )
            ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=marker.id,
                    node_data=marker.model_dump(),
                )
            )

            tx = dest_ws.commit_operations(ops, author="offline_consolidation", grant=grant)
            if tiered is not None:
                tiered.notify_commit()

        self._consolidation_count += 1
        self._is_idle_consolidated = True

        # Log to journal
        journal = self._services.cognitive_journal
        if journal is not None:
            from hbllm.hcir.cognitive_journal import CognitiveEvent
            from hbllm.hcir.semantic_normalizer import CognitiveEventKind

            journal.record(
                CognitiveEvent(
                    kind=CognitiveEventKind.COGNITIVE_STATE_CHANGED,
                    author="offline_consolidation",
                    session_id=self._session_id,
                    data={
                        "event": "offline_consolidation_completed",
                        "consolidation_count": self._consolidation_count,
                        "transaction_id": getattr(tx, "id", ""),
                    },
                )
            )

        logger.info(
            "Offline consolidation #%d committed (author='offline_consolidation')",
            self._consolidation_count,
        )
        return tx

    async def start_autonomous_loop(
        self,
        idle_threshold: float = 2.0,
        check_interval: float = 0.05,
    ) -> asyncio.Task[None]:
        """Start the background autonomous worker monitoring idle threshold for offline consolidation."""
        await self.stop_autonomous_loop()
        self._idle_threshold = idle_threshold
        if self._last_observation_time <= 0:
            self._last_observation_time = time.monotonic()
        self._is_running = True
        self._autonomous_task = asyncio.create_task(
            self._autonomous_worker(check_interval=check_interval)
        )
        return self._autonomous_task

    async def stop_autonomous_loop(self) -> None:
        """Stop the background autonomous consolidation worker."""
        if self._autonomous_task is not None:
            self._autonomous_task.cancel()
            try:
                await self._autonomous_task
            except (asyncio.CancelledError, Exception):
                pass
            self._autonomous_task = None

    async def _autonomous_worker(self, check_interval: float) -> None:
        """Background worker loop monitoring sensory inactivity."""
        try:
            while self._is_running:
                await asyncio.sleep(check_interval)
                now = time.monotonic()
                if self._last_observation_time > 0:
                    idle_duration = now - self._last_observation_time
                    if idle_duration >= self._idle_threshold and not self._is_idle_consolidated:
                        await self.run_offline_consolidation()
        except asyncio.CancelledError:
            pass
