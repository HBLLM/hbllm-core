"""
Kernel Services Container — dependency injection for the Cognitive OS.

Provides the ``KernelServices`` container that holds references
to all kernel service implementations.  Cognitive nodes receive
this container through the ABI ``execute()`` contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hbllm.hcir.kernel.capability_resolver import (
    CapabilityImplementation,
    CapabilityResolver,
)
from hbllm.hcir.kernel.capability_sandboxing import CapabilitySandboxManager
from hbllm.hcir.kernel.event_bus import KernelEventBus
from hbllm.hcir.kernel.executor import KernelExecutor
from hbllm.hcir.kernel.governance.governance_engine import GovernanceEngine
from hbllm.hcir.kernel.runtime_state import RuntimeState
from hbllm.hcir.kernel.scheduler import KernelInstructionScheduler
from hbllm.hcir.kernel.telemetry import KernelTelemetry
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.stores import IEventStore
from hbllm.hcir.workspace import HCIRWorkspaceState

if TYPE_CHECKING:
    from hbllm.persistence.service import PersistenceService


class DefaultWorldPredictionExecutor:
    """Default in-memory executor for world_prediction capability in simulation branches."""

    @property
    def is_available(self) -> bool:
        return True

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"status": "simulated", "predicted": True, **params}


@dataclass
class KernelServices:
    """Dependency container for all kernel services.

    Passed to cognitive nodes via the ABI ``execute()`` contract.
    Nodes interact with the kernel exclusively through this container.

    Usage::

        services = KernelServices(
            workspace=workspace,
            transaction_manager=tx_mgr,
            capability_resolver=resolver,
            scheduler=scheduler,
        )
        # Nodes receive this in execute()
    """

    workspace: HCIRWorkspaceState
    transaction_manager: TransactionManager
    capability_resolver: CapabilityResolver
    scheduler: KernelInstructionScheduler
    event_bus: KernelEventBus = field(default_factory=KernelEventBus)
    event_store: IEventStore | None = None
    governance: GovernanceEngine | None = None
    runtime_state: RuntimeState | None = None
    executor: KernelExecutor | None = None
    persistence: PersistenceService | None = None
    telemetry: KernelTelemetry | None = None
    sandbox_manager: CapabilitySandboxManager | None = None

    # ── HCIR Cognitive OS extensions ──
    tiered_workspace: Any = None  # TieredWorkspace (lazy import to avoid circular)
    world_kernel: Any = None  # WorldKernel
    cognitive_journal: Any = None  # CognitiveJournal
    cognitive_event_log: Any = None  # CognitiveEventLog
    semantic_normalizer: Any = None  # SemanticNormalizer
    constitutional_verifier: Any = None  # ConstitutionalVerifier
    bus_bridge: Any = None  # HCIRBusBridge
    migration_policy: Any = None  # MigrationPolicy
    migration_metrics: Any = None  # MigrationMetrics

    # Extension point for future services
    extensions: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create_default(
        cls,
        workspace: HCIRWorkspaceState,
        verification_pipeline: list[Any] | None = None,
    ) -> KernelServices:
        """Create a fully provisioned KernelServices instance with verified pipeline, executor, and telemetry."""
        from hbllm.hcir.capabilities.planning_capabilities import PlanningCapabilityExecutor
        from hbllm.hcir.capabilities.routing_capabilities import RoutingCapabilityExecutor
        from hbllm.hcir.capabilities.world_capabilities import WorldKernelCapabilityExecutor
        from hbllm.hcir.kernel.capability_sandboxing import (
            CapabilityPermissions,
            CapabilitySandboxManager,
            SandboxedCapabilityPolicy,
            TrustLevel,
        )
        from hbllm.hcir.kernel.verification import create_default_pipeline
        from hbllm.hcir.world_kernel import WorldKernel

        pipeline = verification_pipeline or create_default_pipeline()
        tx_mgr = TransactionManager(workspace, verification_stages=pipeline)
        sandbox_mgr = CapabilitySandboxManager()
        resolver = CapabilityResolver(sandbox_manager=sandbox_mgr)
        world_k = WorldKernel(workspace)

        system_perms = CapabilityPermissions(
            allow_filesystem=True,
            allow_network=True,
            allow_subprocess=True,
            allow_db_write=True,
        )

        resolver.register(
            CapabilityImplementation(
                capability_name="world_prediction",
                implementation_id="default_world_prediction",
                executor=DefaultWorldPredictionExecutor(),
                priority=1,
                description="Default in-memory forward simulation executor",
                tags=["safe", "internal"],
            )
        )
        sandbox_mgr.register_policy(
            SandboxedCapabilityPolicy(
                capability_name="world_prediction",
                provider_id="default_world_prediction",
                trust_level=TrustLevel.SYSTEM,
                permissions=system_perms,
            )
        )

        for w_cap in [
            "world.predict_outcome",
            "world.query_state",
            "world.evaluate_prediction_error",
            "world.sync_digital_twin",
            "world.select_action",
            "world.predict",
            "world.evaluate_error",
            "world.sync_twin",
        ]:
            impl_id = f"kernel_{w_cap}"
            world_exec = WorldKernelCapabilityExecutor(world_k, default_capability=w_cap)
            resolver.register(
                CapabilityImplementation(
                    capability_name=w_cap,
                    implementation_id=impl_id,
                    executor=world_exec,
                    priority=5,
                    description=f"Kernel-governed live {w_cap} capability",
                    tags=["safe", "internal"],
                )
            )
            sandbox_mgr.register_policy(
                SandboxedCapabilityPolicy(
                    capability_name=w_cap,
                    provider_id=impl_id,
                    trust_level=TrustLevel.SYSTEM,
                    permissions=system_perms,
                )
            )

        for p_cap in [
            "planning.generate_candidates",
            "planning.validate",
            "planning.expand",
        ]:
            impl_id = f"kernel_{p_cap}"
            plan_exec = PlanningCapabilityExecutor(workspace=workspace, default_capability=p_cap)
            resolver.register(
                CapabilityImplementation(
                    capability_name=p_cap,
                    implementation_id=impl_id,
                    executor=plan_exec,
                    priority=5,
                    description=f"Kernel-governed live {p_cap} capability",
                    tags=["safe", "internal"],
                )
            )
            sandbox_mgr.register_policy(
                SandboxedCapabilityPolicy(
                    capability_name=p_cap,
                    provider_id=impl_id,
                    trust_level=TrustLevel.SYSTEM,
                    permissions=system_perms,
                )
            )

        for r_cap in [
            "routing.resolve_route",
            "execution.sandbox_run",
        ]:
            impl_id = f"kernel_{r_cap}"
            route_exec = RoutingCapabilityExecutor(
                resolver=resolver,
                sandbox_manager=sandbox_mgr,
                default_capability=r_cap,
            )
            resolver.register(
                CapabilityImplementation(
                    capability_name=r_cap,
                    implementation_id=impl_id,
                    executor=route_exec,
                    priority=5,
                    description=f"Kernel-governed live {r_cap} capability",
                    tags=["safe", "internal"],
                )
            )
            sandbox_mgr.register_policy(
                SandboxedCapabilityPolicy(
                    capability_name=r_cap,
                    provider_id=impl_id,
                    trust_level=TrustLevel.SYSTEM,
                    permissions=system_perms,
                )
            )

        return cls(
            workspace=workspace,
            transaction_manager=tx_mgr,
            capability_resolver=resolver,
            scheduler=KernelInstructionScheduler(),
            event_bus=KernelEventBus(),
            executor=KernelExecutor(),
            telemetry=KernelTelemetry(),
            sandbox_manager=sandbox_mgr,
            runtime_state=RuntimeState(),
            world_kernel=world_k,
        )
