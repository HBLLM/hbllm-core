"""HCIR Kernel — Cognitive OS execution services."""

from __future__ import annotations

from hbllm.hcir.kernel.attention_graph import AttentionManager
from hbllm.hcir.kernel.capability_health import CapabilityHealth, CapabilityHealthRegistry
from hbllm.hcir.kernel.capability_registry import (
    CapabilityRegistry,
    CapabilitySpec,
    KernelCapabilityRegistry,
)
from hbllm.hcir.kernel.capability_resolver import (
    CapabilityImplementation,
    CapabilityResolver,
    ICapabilityExecutor,
    PerformanceProfile,
    ResourceRequirements,
)
from hbllm.hcir.kernel.capability_sandboxing import CapabilitySandboxManager
from hbllm.hcir.kernel.cognitive_kernel import CognitiveKernel, KernelExecutionReceipt
from hbllm.hcir.kernel.consensus_engine import (
    CandidateBelief,
    CognitiveConsensusEngine,
)
from hbllm.hcir.kernel.event_bus import KernelEvent, KernelEventBus
from hbllm.hcir.kernel.executive_controller import CognitiveCycleResult, KernelExecutiveController
from hbllm.hcir.kernel.executive_runtime import ExecutiveRuntime
from hbllm.hcir.kernel.executor import KernelExecutor
from hbllm.hcir.kernel.governance.constitutional_verifier import (
    ConstitutionalTransactionVerifier,
    ConstitutionalVerifier,
)
from hbllm.hcir.kernel.identity_bridge import IdentityBridge
from hbllm.hcir.kernel.runtime_state import RuntimeState
from hbllm.hcir.kernel.scheduler import KernelInstructionScheduler, ScheduledTask
from hbllm.hcir.kernel.scheduler_policy import CognitiveScoreCalculator
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.telemetry import KernelMetric, KernelTelemetry
from hbllm.hcir.kernel.transaction_envelope import (
    CognitiveAuthorityChain,
    KernelTransactionEnvelope,
)
from hbllm.hcir.kernel.transaction_manager import (
    IVerificationStage,
    TransactionBuilder,
    TransactionManager,
    TransactionResult,
)
from hbllm.hcir.kernel.verification import (
    BranchModeVerifier,
    PolicyVerifier,
    ResourceVerifier,
    SchemaVerifier,
    ScopeVerifier,
    create_default_pipeline,
)

__all__ = [
    "AttentionManager",
    "BranchModeVerifier",
    "CandidateBelief",
    "CapabilityHealth",
    "CapabilityHealthRegistry",
    "CapabilityImplementation",
    "CapabilityRegistry",
    "CapabilityResolver",
    "CapabilitySandboxManager",
    "CapabilitySpec",
    "CognitiveAuthorityChain",
    "CognitiveConsensusEngine",
    "CognitiveCycleResult",
    "CognitiveKernel",
    "CognitiveScoreCalculator",
    "ConstitutionalTransactionVerifier",
    "ConstitutionalVerifier",
    "ExecutiveRuntime",
    "ICapabilityExecutor",
    "IVerificationStage",
    "IdentityBridge",
    "KernelEvent",
    "KernelEventBus",
    "KernelExecutionReceipt",
    "KernelExecutor",
    "KernelExecutiveController",
    "KernelInstructionScheduler",
    "KernelMetric",
    "KernelServices",
    "KernelTelemetry",
    "KernelTransactionEnvelope",
    "PerformanceProfile",
    "PolicyVerifier",
    "ResourceRequirements",
    "ResourceVerifier",
    "RuntimeState",
    "ScheduledTask",
    "SchemaVerifier",
    "ScopeVerifier",
    "TransactionBuilder",
    "TransactionManager",
    "TransactionResult",
    "create_default_pipeline",
]
