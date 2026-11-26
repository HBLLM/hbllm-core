"""Phase 2: Blank Brain Cognitive Initialization Profile.

Enforces strict separation between innate cognitive architecture (frozen)
and acquired developmental knowledge (initialized strictly empty),
backed natively by HCIR ``TieredWorkspace`` and copy-on-write ``HCIRInterpreter``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.bytecode import InstructionStream
from hbllm.hcir.interpreter import ExecutionResult, HCIRInterpreter
from hbllm.hcir.kernel.commit_authority import CommitAuthorityRegistry
from hbllm.hcir.kernel.governance.constitutional_verifier import ConstitutionalVerifier
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.kernel.verification import create_default_pipeline
from hbllm.hcir.reasoning.operators.active_inference import ActiveInferenceOperator
from hbllm.hcir.reasoning.operators.registry import create_default_operator_registry
from hbllm.hcir.reasoning.unified_runtime import UnifiedReasoningRuntime
from hbllm.hcir.types import Scope
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.workspace_tiers import TieredWorkspace

from .types import DevelopmentalProfile

logger = logging.getLogger(__name__)


class HCIRMentalSandbox:
    """Copy-on-write counterfactual mental sandbox over HCIR workspaces."""

    def __init__(self, services: KernelServices | None = None) -> None:
        self._services = services

    def fork_workspace(self, workspace: HCIRWorkspaceState) -> HCIRWorkspaceState:
        """Create an isolated copy-on-write workspace fork without mutating reality."""
        forked_graph = workspace.graph.fork()
        authority = CommitAuthorityRegistry()
        verifier = ConstitutionalVerifier(authority_registry=authority)
        pipeline = create_default_pipeline(constitutional_verifier=verifier)
        fork_ws = HCIRWorkspaceState(graph=forked_graph)
        fork_ws.attach_transaction_manager(
            TransactionManager(workspace=fork_ws, verification_stages=pipeline)
        )
        return fork_ws

    def simulate_stream(
        self,
        workspace: HCIRWorkspaceState,
        stream: InstructionStream,
        *,
        tenant_id: str = "developmental",
    ) -> ExecutionResult:
        """Execute an HCIR instruction stream on an isolated copy-on-write fork."""
        fork_ws = self.fork_workspace(workspace)
        svc = self._services or KernelServices.create_default(workspace=fork_ws)
        interp = HCIRInterpreter(services=svc, workspace=fork_ws)
        return interp.execute_sync(stream, tenant_id=tenant_id)


@dataclass
class BlankBrainSubstrate:
    """Pre-existing, innate cognitive architecture with empty learned knowledge."""

    # --- Layer 1: Innate Cognitive Machinery ---
    tiered_workspace: TieredWorkspace = field(
        default_factory=lambda: TieredWorkspace(persistent_store=None, tenant_id="developmental")
    )
    workspace: HCIRWorkspaceState | None = None
    transaction_mgr: TransactionManager | None = None
    memory: TieredWorkspace | None = None
    mental_sandbox: HCIRMentalSandbox = field(default_factory=HCIRMentalSandbox)
    runtime: UnifiedReasoningRuntime | None = None
    active_inference: ActiveInferenceOperator = field(default_factory=ActiveInferenceOperator)

    # --- Layer 2: Developmentally Learned Stores (Initialized Empty) ---
    semantic_concepts: dict[str, Any] = field(default_factory=dict)
    object_categories: dict[str, Any] = field(default_factory=dict)
    causal_rules: list[dict[str, Any]] = field(default_factory=list)
    affordances: dict[str, list[str]] = field(default_factory=dict)
    spatial_schemas: list[dict[str, Any]] = field(default_factory=list)
    procedural_skills: dict[str, Any] = field(default_factory=dict)
    lexical_mapping: dict[str, str] = field(default_factory=dict)

    profile: DevelopmentalProfile = field(default_factory=DevelopmentalProfile)

    def __post_init__(self) -> None:
        if self.workspace is None:
            self.workspace = self.tiered_workspace.working_state
        if self.memory is None:
            self.memory = self.tiered_workspace
        if self.transaction_mgr is None:
            authority = CommitAuthorityRegistry()
            verifier = ConstitutionalVerifier(authority_registry=authority)
            pipeline = create_default_pipeline(
                default_scope=Scope(tenant_id="developmental"),
                constitutional_verifier=verifier,
            )
            self.transaction_mgr = TransactionManager(
                workspace=self.workspace,
                verification_stages=pipeline,
            )
            self.workspace.attach_transaction_manager(self.transaction_mgr)
        if self.runtime is None:
            registry = create_default_operator_registry()
            self.runtime = UnifiedReasoningRuntime(registry)

    def verify_isolation(self) -> bool:
        """Scientifically assert that all learned knowledge layers are strictly empty."""
        assert len(self.semantic_concepts) == 0, "Learned concepts must be empty initially!"
        assert len(self.object_categories) == 0, (
            "Learned object categories must be empty initially!"
        )
        assert len(self.causal_rules) == 0, "Learned causal rules must be empty initially!"
        assert len(self.affordances) == 0, "Learned affordances must be empty initially!"
        assert len(self.spatial_schemas) == 0, "Learned spatial schemas must be empty initially!"
        assert len(self.procedural_skills) == 0, (
            "Learned procedural skills must be empty initially!"
        )
        assert len(self.lexical_mapping) == 0, "Learned lexicon must be empty initially!"
        if self.workspace is not None:
            assert self.workspace.graph.node_count == 0, (
                "Blank brain workspace must contain zero learned nodes initially!"
            )
        return self.profile.verify_blank_brain()


def create_blank_brain_substrate() -> BlankBrainSubstrate:
    """Instantiate a pristine developmental Blank Brain substrate."""
    profile = DevelopmentalProfile()
    substrate = BlankBrainSubstrate(profile=profile)
    substrate.verify_isolation()
    return substrate
