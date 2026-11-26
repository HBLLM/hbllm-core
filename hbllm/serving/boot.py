"""
Boot Orchestrator — Unified boot sequence for HBLLM HCIR Cognitive OS.

Wires together the canonical HCIR runtime and serving infrastructure:

    1. Load BrainProfile → determines which subsystems activate
    2. Boot CognitiveSystem via `build_cognitive_system()` (TieredWorkspace + ExecutiveRuntime)
    3. Initialize MessageBus + Gateway
    4. Initialize ServingTaskScheduler
    5. Initialize PermissionEngine
    6. Restore persistent identity ConceptNode in TieredWorkspace
    7. Initialize WorkspaceManager
    8. Schedule recurring HCIR consolidation maintenance
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hbllm.config.brain_profile import BrainProfile
    from hbllm.hcir.kernel.executive_runtime import ExecutiveRuntime
    from hbllm.hcir.kernel.services import KernelServices
    from hbllm.network.bus import InProcessBus
    from hbllm.network.gateway import Gateway
    from hbllm.runtime.bootstrap import CognitiveSystem
    from hbllm.security.permission_engine import PermissionEngine
    from hbllm.serving.cognitive_scheduler import ServingTaskScheduler
    from hbllm.workspace.workspace_manager import WorkspaceManager

logger = logging.getLogger(__name__)


@dataclass
class BootConfig:
    """Configuration for the boot sequence."""

    profile_name: str = "full"  # full | lite | edge | research | robot
    data_dir: str = "data"
    checkpoints_dir: str = "./checkpoints"
    provider: str = "openai"
    model_size: str = "medium"
    local: bool = False
    checkpoint_path: str | None = None
    provider_kwargs: dict[str, Any] = field(default_factory=dict)

    enable_perception: bool = False
    enable_autonomy: bool = True
    enable_http: bool = False
    enable_gateway: bool = True
    enable_scheduler: bool = True
    enable_permissions: bool = True
    enable_identity: bool = True
    enable_workspaces: bool = True

    host: str = "127.0.0.1"
    port: int = 8420
    max_concurrent_llm: int = 3
    scheduler_workers: int = 3


@dataclass
class BootContext:
    """All subsystems created during boot, available for wiring."""

    cognitive_system: CognitiveSystem | None = None
    brain: Any | None = None
    profile: BrainProfile | None = None
    bus: InProcessBus | None = None
    gateway: Gateway | None = None
    scheduler: ServingTaskScheduler | None = None
    permission_engine: PermissionEngine | None = None
    workspace_manager: WorkspaceManager | None = None
    kernel_services: KernelServices | None = None
    executive_runtime: ExecutiveRuntime | None = None
    boot_time_ms: float = 0.0


class BootOrchestrator:
    """Unified boot sequence that wires the Pure-HCIR Cognitive OS and serving layers."""

    def __init__(self, config: BootConfig | None = None) -> None:
        self._config = config or BootConfig()
        self._context = BootContext()
        self._started = False

    @property
    def context(self) -> BootContext:
        return self._context

    async def boot(self) -> BootContext:
        """Execute the full HCIR boot sequence."""
        start = time.monotonic()
        cfg = self._config
        ctx = self._context

        logger.info("═" * 60)
        logger.info("  HBLLM HCIR Cognitive OS — Boot Sequence")
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

        # ── 1.5 Initialize Tracing ───────────────────────────────────
        from hbllm.observability.tracing import init_tracing

        if init_tracing():
            logger.info("OpenTelemetry tracing initialized")

        # ── 2. Boot HCIR CognitiveSystem & Executive Runtime ─────────
        from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

        cog_cfg = CognitiveSystemConfig(
            world_id=cfg.profile_name,
            memory_dir=Path(cfg.data_dir) / "worlds" / cfg.profile_name,
            enable_grounding=cfg.enable_perception,
            enable_autonomous_curiosity=cfg.enable_autonomy and ctx.profile.features.autonomy_core,
        )
        cog_sys = build_cognitive_system(cog_cfg)
        ctx.cognitive_system = cog_sys
        ctx.brain = cog_sys
        ctx.kernel_services = cog_sys.services
        ctx.executive_runtime = cog_sys.runtime

        if ctx.executive_runtime:
            await ctx.executive_runtime.start()

        logger.info("HCIR CognitiveSystem & ExecutiveRuntime online")

        # ── 3. Initialize Bus & Gateway ──────────────────────────────
        if cfg.enable_gateway:
            from hbllm.network.bus import InProcessBus
            from hbllm.network.gateway import Gateway, GatewayConfig

            ctx.bus = InProcessBus()
            await ctx.bus.start()
            gateway_config = GatewayConfig(
                max_sessions=100 if ctx.profile.features.multi_tenant else 10,
            )
            ctx.gateway = Gateway(config=gateway_config, bus=ctx.bus)
            await ctx.gateway.start()
            logger.info("Gateway started (max_sessions=%d)", gateway_config.max_sessions)

        # ── 4. Initialize ServingTaskScheduler ───────────────────────
        if cfg.enable_scheduler:
            from hbllm.serving.cognitive_scheduler import ServingTaskScheduler

            ctx.scheduler = ServingTaskScheduler(
                max_concurrent_llm=cfg.max_concurrent_llm,
                max_concurrent_background=cfg.scheduler_workers,
            )
            await ctx.scheduler.start()
            logger.info(
                "ServingTaskScheduler online (%d workers, max %d LLM concurrent)",
                cfg.scheduler_workers,
                cfg.max_concurrent_llm,
            )

        # ── 5. Initialize PermissionEngine ─────────────────────────
        if cfg.enable_permissions:
            from hbllm.security.permission_engine import PermissionEngine

            ctx.permission_engine = PermissionEngine(data_dir=cfg.data_dir)
            await ctx.permission_engine.init_db()
            logger.info("PermissionEngine initialized (default-deny)")

        # ── 6. Restore HCIR Identity ConceptNode ─────────────────────
        if cfg.enable_identity and cog_sys.tiered_workspace is not None:
            from hbllm.hcir.graph import ConceptNode
            from hbllm.hcir.transactions import TransactionOp, TransactionOperation
            from hbllm.hcir.types import Provenance, Scope

            identity_node = ConceptNode(
                id="core_identity",
                label="Sentra",
                definition="HBLLM HCIR Cognitive OS identity anchor",
                domain="identity",
                provenance=Provenance(created_by="boot_orchestrator"),
                scope=Scope(tenant_id="default"),
                tags=["identity", "system"],
                properties={"profile": cfg.profile_name, "last_boot_ts": time.time()},
            )
            cog_sys.tiered_workspace.persistent.commit_operations(
                [
                    TransactionOperation(
                        op=TransactionOp.UPSERT_NODE,
                        node_id=identity_node.id,
                        node_data=identity_node.model_dump(),
                    )
                ],
                author="boot_orchestrator",
            )

        # ── 7. Initialize WorkspaceManager ─────────────────────────
        if cfg.enable_workspaces:
            from hbllm.workspace.workspace_manager import WorkspaceManager

            ctx.workspace_manager = WorkspaceManager(data_dir=cfg.data_dir)
            await ctx.workspace_manager.init_db()
            logger.info("WorkspaceManager initialized")

        # ── 8. Schedule recurring maintenance ────────────────────────
        if ctx.scheduler and ctx.profile.features.memory_consolidation:
            self._schedule_maintenance(ctx)

        ctx.boot_time_ms = (time.monotonic() - start) * 1000
        logger.info("  HCIR Cognitive OS ready (%.0fms boot)", ctx.boot_time_ms)
        self._started = True
        return ctx

    async def shutdown(self) -> None:
        """Graceful shutdown of all subsystems."""
        if not self._started:
            return

        logger.info("Shutting down Cognitive OS...")
        ctx = self._context

        if ctx.scheduler:
            await ctx.scheduler.stop()

        if ctx.gateway:
            await ctx.gateway.stop()

        if ctx.bus:
            await ctx.bus.stop()

        if ctx.executive_runtime:
            await ctx.executive_runtime.stop()

        from hbllm.observability.tracing import shutdown_tracing

        shutdown_tracing()

        self._started = False
        logger.info("Cognitive OS shutdown complete")

    def _schedule_maintenance(self, ctx: BootContext) -> None:
        """Register recurring HCIR memory consolidation tasks."""
        if not ctx.scheduler:
            return

        from hbllm.serving.cognitive_scheduler import TaskPriority

        if ctx.cognitive_system is not None:

            async def _consolidate() -> None:
                if ctx.cognitive_system is not None:
                    ctx.cognitive_system.tiered_workspace.consolidate_node_budget()

            ctx.scheduler.schedule_recurring(
                _consolidate,
                interval_s=300,
                name="hcir-memory-consolidation",
                priority=TaskPriority.MAINTENANCE,
            )

        logger.info("Recurring HCIR maintenance tasks scheduled")
