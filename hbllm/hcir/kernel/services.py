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
    from hbllm.hcir.adapters.hcir_bus_bridge import HCIRBusBridge
    from hbllm.hcir.cognitive_event_log import CognitiveEventLog
    from hbllm.hcir.cognitive_journal import CognitiveJournal
    from hbllm.hcir.kernel.commit_authority import CommitAuthorityRegistry
    from hbllm.hcir.kernel.governance.constitutional_verifier import ConstitutionalVerifier
    from hbllm.hcir.kernel.governance.policies.migration_policy import MigrationPolicy
    from hbllm.hcir.kernel.migration_metrics import MigrationMetrics
    from hbllm.hcir.semantic_normalizer import SemanticNormalizer
    from hbllm.hcir.workspace_tiers import TieredWorkspace
    from hbllm.hcir.world_kernel import WorldKernel
    from hbllm.persistence.service import PersistenceService


# Canonical world-kernel capabilities exposed through the resolver.
WORLD_CAPABILITIES: tuple[str, ...] = (
    "world.predict_outcome",
    "world.query_state",
    "world.evaluate_prediction_error",
    "world.sync_digital_twin",
    "world.select_action",
    "world.predictive_coding_cycle",
    "world.path_integrate_odometry",
    "world.assess_metacognitive_agency",
    "world.synthesize_skills",
    "world.superposition_rollout",
)

# Short-form aliases retained for the public manifest; they resolve to the canonical name.
WORLD_CAPABILITY_ALIASES: dict[str, str] = {
    "world.predict": "world.predict_outcome",
    "world.evaluate_error": "world.evaluate_prediction_error",
    "world.sync_twin": "world.sync_digital_twin",
}


MOTOR_CAPABILITIES: tuple[str, ...] = (
    "motor.navigate",
    "motor.interact",
    "motor.activate",
    "motor.pickup",
    "motor.drop",
    "motor.manipulate",
    "motor.align",
)


class DefaultWorldPredictionExecutor:
    """Default in-memory executor for world_prediction capability in simulation branches."""

    @property
    def is_available(self) -> bool:
        return True

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"status": "simulated", "predicted": True, **params}


class DefaultMotorExecutor:
    """Governed Layer-0 motor intent dispatcher for symbolic `motor.*` syscalls."""

    def __init__(self, default_capability: str = "motor.interact") -> None:
        self._default_capability = default_capability

    @property
    def is_available(self) -> bool:
        return True

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        cap = str(params.get("capability", self._default_capability))
        return {
            "ok": True,
            "status": "dispatched",
            "capability": cap,
            "target": params.get("target"),
            **params,
        }


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
    tiered_workspace: TieredWorkspace | None = None
    world_kernel: WorldKernel | None = None
    cognitive_journal: CognitiveJournal | None = None
    cognitive_event_log: CognitiveEventLog | None = None
    semantic_normalizer: SemanticNormalizer | None = None
    constitutional_verifier: ConstitutionalVerifier | None = None
    bus_bridge: HCIRBusBridge | None = None
    migration_policy: MigrationPolicy | None = None
    migration_metrics: MigrationMetrics | None = None
    commit_authority_registry: CommitAuthorityRegistry | None = None

    # ── Typed Cognitive / Language Runtime Subsystems (M-4) ──
    language_runtime: Any | None = None
    ontology_store: Any | None = None
    command_dispatcher: Any | None = None
    concept_explainer: Any | None = None
    curiosity_controller: Any | None = None
    epistemic_curiosity: Any | None = None
    autonomy_metrics: Any | None = None
    reasoning_runtime: Any | None = None
    neuromodulation: Any | None = None
    lexicon_acquisition: Any | None = None

    # Extension point for future services
    extensions: dict[str, Any] = field(default_factory=dict)
    require_verified: bool = False

    def __post_init__(self) -> None:
        import os

        if getattr(self.transaction_manager, "unverified", False):
            if self.require_verified or not os.environ.get("PYTEST_CURRENT_TEST"):
                raise AssertionError(
                    "KernelServices cannot be initialized with an unverified TransactionManager"
                )

        # Guarantee the TM publishes lifecycle events on the shared bus so that
        # ``WAIT event_type="transaction.*"`` can resolve.
        if self.transaction_manager.event_bus is None:
            self.transaction_manager.event_bus = self.event_bus
        # Per-tier commit authorities (WP6.4): when a TieredWorkspace is
        # provisioned, ensure a registry exists (injecting it into the
        # constitutional verifier if needed) and attach per-tier TMs.
        if self.tiered_workspace is not None:
            from hbllm.hcir.kernel.commit_authority import CommitAuthorityRegistry
            from hbllm.hcir.kernel.governance.constitutional_verifier import (
                ConstitutionalVerifier,
            )

            if self.commit_authority_registry is None:
                self.commit_authority_registry = CommitAuthorityRegistry()
            stages = list(getattr(self.transaction_manager, "_verification_stages", []))
            for stage in stages:
                if isinstance(stage, ConstitutionalVerifier):
                    stage.authority_registry = self.commit_authority_registry
            self.tiered_workspace.attach_authorities(
                stages, self.event_bus, self.commit_authority_registry
            )
        # Surface graph substrate health via telemetry (WP4).
        ws = getattr(self, "workspace", None)
        if ws is None and self.tiered_workspace is not None:
            ws = getattr(self.tiered_workspace, "working", None)
        graph = getattr(ws, "graph", None)
        if self.telemetry is not None and hasattr(graph, "native_health"):
            self.telemetry.health_probe = graph.native_health

    @classmethod
    def create_default(
        cls,
        workspace: HCIRWorkspaceState,
        verification_pipeline: list[Any] | None = None,
        tiered_workspace: TieredWorkspace | None = None,
        commit_authority_registry: CommitAuthorityRegistry | None = None,
    ) -> KernelServices:
        """Create a fully provisioned KernelServices instance with verified pipeline, executor, and telemetry."""
        from hbllm.hcir.capabilities.memory_capabilities import (
            MEMORY_CAPABILITIES,
            MemoryCapabilityExecutor,
        )
        from hbllm.hcir.capabilities.planning_capabilities import PlanningCapabilityExecutor
        from hbllm.hcir.capabilities.routing_capabilities import RoutingCapabilityExecutor
        from hbllm.hcir.capabilities.tool_capabilities import (
            TOOL_CAPABILITIES,
            ToolCapabilityExecutor,
        )
        from hbllm.hcir.capabilities.world_capabilities import WorldKernelCapabilityExecutor
        from hbllm.hcir.kernel.capability_sandboxing import (
            CapabilityPermissions,
            CapabilitySandboxManager,
            IsolationMode,
            SandboxedCapabilityPolicy,
            TrustLevel,
        )
        from hbllm.hcir.kernel.commit_authority import CommitAuthorityRegistry
        from hbllm.hcir.kernel.governance.constitutional_verifier import (
            ConstitutionalVerifier,
        )
        from hbllm.hcir.kernel.verification import create_default_pipeline
        from hbllm.hcir.world_kernel import WorldKernel

        registry = commit_authority_registry
        verifier: ConstitutionalVerifier | None = None
        if verification_pipeline:
            for s in verification_pipeline:
                if isinstance(s, ConstitutionalVerifier):
                    verifier = s
                    if registry is None and s.authority_registry:
                        registry = s.authority_registry
                    break
        if registry is None:
            registry = CommitAuthorityRegistry()
        if verifier is None:
            verifier = ConstitutionalVerifier(authority_registry=registry)

        pipeline = verification_pipeline or create_default_pipeline(
            constitutional_verifier=verifier
        )
        event_bus = KernelEventBus()
        tx_mgr = TransactionManager(workspace, verification_stages=pipeline, event_bus=event_bus)
        sandbox_mgr = CapabilitySandboxManager()
        resolver = CapabilityResolver(sandbox_manager=sandbox_mgr)
        world_k = WorldKernel(workspace)

        least_privilege_in_memory = CapabilityPermissions(
            allow_filesystem=False,
            allow_network=False,
            allow_subprocess=False,
            allow_db_write=False,
        )

        def _reg(
            cap: str,
            executor: Any,
            *,
            desc: str | None = None,
            tags: list[str] | None = None,
            req: set[str] | None = None,
            perms: CapabilityPermissions = least_privilege_in_memory,
            iso: IsolationMode = IsolationMode.IN_PROCESS,
            prio: int = 5,
            impl_id: str | None = None,
        ) -> None:
            pid = impl_id or f"kernel_{cap}"
            r_perms = set(req or ())
            resolver.register(
                CapabilityImplementation(
                    capability_name=cap,
                    implementation_id=pid,
                    executor=executor,
                    priority=prio,
                    description=desc or f"Kernel-governed live {cap} capability",
                    tags=tags or ["safe", "internal"],
                    required_permissions=r_perms,
                )
            )
            sandbox_mgr.register_policy(
                SandboxedCapabilityPolicy(
                    capability_name=cap,
                    provider_id=pid,
                    trust_level=TrustLevel.SYSTEM,
                    isolation_mode=iso,
                    permissions=perms,
                    required_permissions=r_perms,
                )
            )

        _reg(
            "world_prediction",
            DefaultWorldPredictionExecutor(),
            desc="Default in-memory forward simulation executor",
            prio=1,
            impl_id="default_world_prediction",
        )
        for w_cap in [*WORLD_CAPABILITIES, *WORLD_CAPABILITY_ALIASES]:
            canonical = WORLD_CAPABILITY_ALIASES.get(w_cap, w_cap)
            _reg(
                w_cap,
                WorldKernelCapabilityExecutor(world_k, default_capability=canonical),
                desc=f"Kernel-governed live {canonical} capability"
                if canonical == w_cap
                else f"Alias of {canonical}",
                tags=["safe", "internal"] + (["alias"] if canonical != w_cap else []),
            )
        for p_cap in ("planning.generate_candidates", "planning.validate", "planning.expand"):
            _reg(p_cap, PlanningCapabilityExecutor(workspace=workspace, default_capability=p_cap))
        for r_cap in ("routing.resolve_route", "execution.sandbox_run"):
            is_sbx = r_cap == "execution.sandbox_run"
            _reg(
                r_cap,
                RoutingCapabilityExecutor(
                    resolver=resolver, sandbox_manager=sandbox_mgr, default_capability=r_cap
                ),
                req={"subprocess"} if is_sbx else set(),
                iso=IsolationMode.SUBPROCESS if is_sbx else IsolationMode.IN_PROCESS,
                perms=(
                    CapabilityPermissions(
                        allow_filesystem=False,
                        allow_network=False,
                        allow_subprocess=True,
                        allow_db_write=False,
                    )
                    if is_sbx
                    else least_privilege_in_memory
                ),
            )
        for t_cap in TOOL_CAPABILITIES:
            needs_sub = t_cap in ("git.status", "shell.exec")
            _reg(
                t_cap,
                ToolCapabilityExecutor(default_capability=t_cap),
                desc=f"Kernel-governed sandboxed {t_cap} capability",
                tags=["safe", "internal", "tool"],
                req={"filesystem", "subprocess"} if needs_sub else {"filesystem"},
                perms=CapabilityPermissions(
                    allow_filesystem=True,
                    allow_network=False,
                    allow_subprocess=needs_sub,
                    allow_db_write=False,
                ),
            )

        from hbllm.hcir.language.ontology_store import DEFAULT_ONTOLOGY_DB, LexicalOntologyStore

        ont_store = (
            LexicalOntologyStore(DEFAULT_ONTOLOGY_DB) if DEFAULT_ONTOLOGY_DB.exists() else None
        )
        for m_cap in MEMORY_CAPABILITIES:
            _reg(
                m_cap,
                MemoryCapabilityExecutor(
                    default_capability=m_cap,
                    tiered_workspace=tiered_workspace,
                    ontology_store=ont_store,
                ),
                desc=f"Kernel-governed HCIR-native {m_cap} capability",
                tags=["safe", "internal", "memory"],
            )

        from hbllm.hcir.capabilities.math_capabilities import (
            MATH_CAPABILITIES,
            MathCapabilityExecutor,
        )
        from hbllm.hcir.capabilities.web_capabilities import WEB_CAPABILITIES, WebCapabilityExecutor

        web_perms = CapabilityPermissions(
            allow_filesystem=False, allow_network=True, allow_subprocess=False, allow_db_write=False
        )
        for w_cap in WEB_CAPABILITIES:
            _reg(
                w_cap,
                WebCapabilityExecutor(
                    default_capability=w_cap,
                    tiered_workspace=tiered_workspace,
                    ontology_store=ont_store,
                ),
                desc=f"Kernel-governed sandboxed {w_cap} capability",
                tags=["safe", "internal", "web"],
                req={"network"},
                perms=web_perms,
            )
        for sys_cap, sys_exec in [
            *[(c, MathCapabilityExecutor(default_capability=c)) for c in MATH_CAPABILITIES],
            *[(c, DefaultMotorExecutor(default_capability=c)) for c in MOTOR_CAPABILITIES],
        ]:
            _reg(
                sys_cap,
                sys_exec,
                desc=f"Kernel-governed Layer-0 {sys_cap} syscall",
                tags=["safe", "internal", "layer0"],
            )

        return cls(
            workspace=workspace,
            transaction_manager=tx_mgr,
            capability_resolver=resolver,
            scheduler=KernelInstructionScheduler(),
            event_bus=event_bus,
            executor=KernelExecutor(),
            telemetry=KernelTelemetry(),
            tiered_workspace=tiered_workspace,
            commit_authority_registry=registry,
            sandbox_manager=sandbox_mgr,
            runtime_state=RuntimeState(),
            world_kernel=world_k,
            constitutional_verifier=verifier,
        )
