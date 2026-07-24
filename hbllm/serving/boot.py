"""
Boot Orchestrator — Unified boot sequence for HBLLM Cognitive OS.

Wires together the Phase A–C subsystems into a coherent startup:

    1. Load BrainProfile → determines which subsystems activate
    2. Create Brain (existing BrainFactory)
    3. Initialize Gateway + Transports
    4. Initialize CognitiveScheduler
    5. Initialize PermissionEngine
    6. Initialize IdentityStateManager → restore identity
    7. Initialize WorkspaceManager
    8. Start AutonomyCore + ProactiveProcessor (existing daemon logic)
    9. Register recurring maintenance tasks

The Boot Orchestrator does NOT replace CognitiveDaemon — it provides
a composable ``boot()`` function that the daemon (or any other
entry point) can call.

Usage::

    from hbllm.serving.boot import BootOrchestrator, BootConfig

    config = BootConfig(
        profile_name="full",
        data_dir="data",
        serve_http=True,
    )

    orchestrator = BootOrchestrator(config)
    context = await orchestrator.boot()

    # context.brain, context.gateway, context.scheduler, etc.

    await orchestrator.shutdown()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Boot Config
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class BootConfig:
    """Configuration for the boot sequence."""

    # Brain profile
    profile_name: str = "full"  # full | lite | edge | research | robot

    # Paths
    data_dir: str = "data"
    checkpoints_dir: str = "./checkpoints"

    # Model
    provider: str = "openai"
    model_size: str = "medium"
    local: bool = False
    checkpoint_path: str | None = None
    provider_kwargs: dict[str, Any] = field(default_factory=dict)

    # Features
    enable_perception: bool = False
    enable_autonomy: bool = True
    enable_http: bool = False
    enable_gateway: bool = True
    enable_scheduler: bool = True
    enable_permissions: bool = True
    enable_identity: bool = True
    enable_workspaces: bool = True

    # HTTP
    host: str = "127.0.0.1"
    port: int = 8420

    # Scheduler
    max_concurrent_llm: int = 3
    scheduler_workers: int = 3


# ═══════════════════════════════════════════════════════════════════════════
# Boot Context (what boot() returns)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class BootContext:
    """All subsystems created during boot, available for wiring."""

    brain: Any = None
    profile: Any = None  # BrainProfile
    gateway: Any = None  # Gateway
    scheduler: Any = None  # CognitiveScheduler
    permission_engine: Any = None  # PermissionEngine
    identity: Any = None  # IdentityStateManager
    workspace_manager: Any = None  # WorkspaceManager
    autonomy: Any = None  # AutonomyCore
    boot_time_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Boot Orchestrator
# ═══════════════════════════════════════════════════════════════════════════


class BootOrchestrator:
    """Unified boot sequence that wires Phase A–C subsystems.

    Composes:
      - BrainProfile (Phase A) — determines which subsystems activate
      - Gateway + Transports (Phase A) — transport → bus bridge
      - CognitiveScheduler (Phase B) — background task arbitration
      - PermissionEngine (Phase B) — plugin sandbox enforcement
      - IdentityStateManager (Phase C) — cross-session continuity
      - WorkspaceManager (Phase A) — multi-domain isolation
      - Brain + AutonomyCore (existing) — cognitive core

    The orchestrator does not replace CognitiveDaemon; it provides
    a composable boot() that the daemon or any entry point can call.
    """

    def __init__(self, config: BootConfig | None = None) -> None:
        self._config = config or BootConfig()
        self._context = BootContext()
        self._started = False

    @property
    def context(self) -> BootContext:
        return self._context

    async def boot(self) -> BootContext:
        """Execute the full boot sequence.

        Returns:
            BootContext with all initialized subsystems.
        """
        start = time.monotonic()
        cfg = self._config
        ctx = self._context

        logger.info("═" * 60)
        logger.info("  HBLLM Cognitive OS — Boot Sequence")
        logger.info("  Profile: %s", cfg.profile_name)
        logger.info("═" * 60)

        # ── 1. Load BrainProfile ─────────────────────────────────────
        from hbllm.config.brain_profile import load_profile

        ctx.profile = load_profile(cfg.profile_name)
        logger.info(
            "Profile loaded: %s (RAM=%sMB, local_inference=%s)",
            ctx.profile.name,
            ctx.profile.max_ram_mb or "∞",
            ctx.profile.features.local_inference,
        )

        # ── 2. Create Brain ──────────────────────────────────────────
        from hbllm.brain.core.factory import BrainConfig, BrainFactory

        brain_config = BrainConfig(
            data_dir=cfg.data_dir,
            inject_awareness=True,
            inject_perception=cfg.enable_perception,
        )

        if cfg.local:
            ctx.brain = await BrainFactory.create_local(
                checkpoint_path=cfg.checkpoint_path,
                model_size=cfg.model_size,
                config=brain_config,
            )
        else:
            ctx.brain = await BrainFactory.create(
                provider=cfg.provider,
                config=brain_config,
                **cfg.provider_kwargs,
            )

        logger.info("Brain created with %d nodes", len(ctx.brain.nodes))

        # ── 3. Initialize Gateway ────────────────────────────────────
        if cfg.enable_gateway:
            from hbllm.network.gateway import Gateway, GatewayConfig

            gateway_config = GatewayConfig(
                max_sessions=100 if ctx.profile.features.multi_tenant else 10,
            )
            ctx.gateway = Gateway(config=gateway_config, bus=ctx.brain.bus)
            await ctx.gateway.start()
            logger.info("Gateway started (max_sessions=%d)", gateway_config.max_sessions)

        # ── 4. Initialize CognitiveScheduler ─────────────────────────
        if cfg.enable_scheduler:
            from hbllm.serving.cognitive_scheduler import CognitiveScheduler

            ctx.scheduler = CognitiveScheduler(
                max_concurrent_llm=cfg.max_concurrent_llm,
            )
            await ctx.scheduler.start(num_workers=cfg.scheduler_workers)
            logger.info("CognitiveScheduler started (%d workers)", cfg.scheduler_workers)

        # ── 5. Initialize PermissionEngine ───────────────────────────
        if cfg.enable_permissions:
            from hbllm.security.permission_engine import PermissionEngine

            ctx.permission_engine = PermissionEngine(data_dir=cfg.data_dir)
            await ctx.permission_engine.init_db()
            logger.info("PermissionEngine initialized (default-deny)")

        # ── 6. Initialize IdentityStateManager ───────────────────────
        if cfg.enable_identity:
            from hbllm.brain.self_model.identity import IdentityStateManager

            ctx.identity = IdentityStateManager(data_dir=cfg.data_dir)
            await ctx.identity.init_db()
            core_id = await ctx.identity.restore()
            await ctx.identity.begin_session()
            logger.info(
                "Identity restored: %s (session #%d)",
                core_id.name,
                core_id.total_sessions,
            )

        # ── 7. Initialize WorkspaceManager ───────────────────────────
        if cfg.enable_workspaces:
            from hbllm.workspace.workspace_manager import WorkspaceManager

            ctx.workspace_manager = WorkspaceManager(data_dir=cfg.data_dir)
            await ctx.workspace_manager.init_db()
            logger.info("WorkspaceManager initialized")

        # ── 8. Start AutonomyCore ────────────────────────────────────
        if cfg.enable_autonomy and ctx.profile.features.autonomous_loops:
            try:
                from hbllm.brain.autonomy.loop import AutonomyCore

                ctx.autonomy = AutonomyCore()
                await ctx.autonomy.start(ctx.brain.bus)
                ctx.brain.autonomy_core = ctx.autonomy
                logger.info("AutonomyCore started — cognitive heartbeat active")
            except Exception as e:
                logger.warning("AutonomyCore failed to start: %s", e)

        # ── 9. Schedule recurring maintenance ────────────────────────
        if ctx.scheduler and ctx.profile.features.memory_consolidation:
            self._schedule_maintenance(ctx)

        # ── Done ─────────────────────────────────────────────────────
        ctx.boot_time_ms = (time.monotonic() - start) * 1000

        logger.info("═" * 60)
        logger.info("  Cognitive OS ready (%.0fms boot)", ctx.boot_time_ms)
        logger.info(
            "  Profile: %s | Gateway: %s | Scheduler: %s",
            cfg.profile_name,
            "ON" if ctx.gateway else "OFF",
            "ON" if ctx.scheduler else "OFF",
        )
        logger.info(
            "  Identity: %s | Workspaces: %s | Permissions: %s",
            "ON" if ctx.identity else "OFF",
            "ON" if ctx.workspace_manager else "OFF",
            "ON" if ctx.permission_engine else "OFF",
        )
        logger.info("═" * 60)

        self._started = True
        return ctx

    async def shutdown(self) -> None:
        """Graceful shutdown of all subsystems."""
        if not self._started:
            return

        logger.info("Shutting down Cognitive OS...")
        ctx = self._context

        # Persist identity before shutdown
        if ctx.identity:
            await ctx.identity.persist()

        # Stop scheduler
        if ctx.scheduler:
            await ctx.scheduler.stop()

        # Stop gateway
        if ctx.gateway:
            await ctx.gateway.stop()

        # Stop autonomy
        if ctx.autonomy:
            await ctx.autonomy.stop()

        # Shutdown brain
        if ctx.brain:
            await ctx.brain.shutdown()

        self._started = False
        logger.info("Cognitive OS shutdown complete")

    # ── Maintenance Scheduling ───────────────────────────────────────────

    def _schedule_maintenance(self, ctx: BootContext) -> None:
        """Register recurring maintenance tasks."""
        if not ctx.scheduler:
            return

        from hbllm.serving.cognitive_scheduler import TaskPriority

        # Memory consolidation every 5 minutes
        if hasattr(ctx.brain, "memory_consolidator"):

            async def _consolidate() -> None:
                if ctx.brain.memory_consolidator:
                    await ctx.brain.memory_consolidator.consolidate()

            ctx.scheduler.schedule_recurring(
                lambda: _consolidate(),
                interval_s=300,
                name="memory-consolidation",
                priority=TaskPriority.MAINTENANCE,
            )

        # Identity persistence every 2 minutes
        if ctx.identity:

            async def _persist_identity() -> None:
                if ctx.identity:
                    await ctx.identity.persist()

            ctx.scheduler.schedule_recurring(
                lambda: _persist_identity(),
                interval_s=120,
                name="identity-persist",
                priority=TaskPriority.IDLE,
            )

        logger.info("Recurring maintenance tasks scheduled")
