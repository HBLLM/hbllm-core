"""
Execution Plan — immutable execution intent with identity.

The ExecutionPlan is the frozen contract between the Orchestrator
and the Runtime. Once created, it cannot be modified. Mutable
runtime state lives in ExecutionState.

Key types:
    TaskType — what kind of execution
    ExecutionConstraints — hard limits the orchestrator must respect
    ExecutionRequest — what the cognitive layer sends (zero cognitive metadata)
    ExecutionPlan — frozen, identified execution intent
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hbllm.execution.payload import ExecutionPayload


class TaskType(str, Enum):
    """Enumeration of all execution task types."""

    # Inference
    TEXT_GENERATION = "text_generation"
    TEXT_COMPLETION = "text_completion"
    JSON_GENERATION = "json_generation"
    EMBEDDING = "embedding"

    # Multimodal (future)
    VISION = "vision"
    AUDIO_SYNTHESIS = "audio_synthesis"

    # Actions (future)
    TOOL_CALL = "tool_call"
    CODE_EXECUTION = "code_execution"
    BROWSER_ACTION = "browser_action"

    # Training (dispatched via ExecutionBus)
    LORA_TRAINING = "lora_training"
    DPO_TRAINING = "dpo_training"
    SFT_TRAINING = "sft_training"


@dataclass(frozen=True)
class ExecutionConstraints:
    """
    Hard constraints the orchestrator must respect.

    These are translated from cognitive outputs at the GenerationNode
    boundary — the execution layer never sees cognitive metadata.
    """

    max_tokens: int = 4096
    max_latency_ms: int | None = None
    max_cost_usd: float | None = None
    require_streaming: bool = False
    require_json: bool = False
    required_capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionRequest:
    """
    What the cognitive layer sends to the Execution Orchestrator.

    Contains ONLY execution concerns:
      - task_type: what kind of execution
      - payload: the rendered content (rich multimodal)
      - constraints: hard limits
      - tenant/session: identity

    NO cognitive_metadata. NO domain. NO style. NO persona.
    Those were already translated into constraints and payload
    before this request was constructed.
    """

    task_type: TaskType
    payload: ExecutionPayload
    constraints: ExecutionConstraints = field(default_factory=ExecutionConstraints)
    tenant_id: str | None = None
    session_id: str | None = None


def _make_plan_id() -> str:
    return str(uuid.uuid4())


def _make_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ExecutionPlan:
    """
    Immutable execution intent with identity.

    Created by the ExecutionOrchestrator, dispatched via ExecutionBus,
    executed by a Runtime. The pipeline CANNOT mutate the plan.
    Mutable runtime state lives in ExecutionState.

    Identity fields enable:
      - Retries (same plan_id, incremented version)
      - Forks (new plan_id, parent_plan set)
      - Distributed execution (trace_id)
      - Deterministic replay (plan_id + version)
      - Parent/child executions (parent_plan)
    """

    # ── Identity ──────────────────────────────────────────────
    plan_id: str = field(default_factory=_make_plan_id)
    parent_plan: str | None = None
    trace_id: str | None = None
    created_at: str = field(default_factory=_make_timestamp)
    version: int = 1

    # ── Execution Target ──────────────────────────────────────
    task_type: TaskType = TaskType.TEXT_GENERATION
    runtime: str = "text"
    provider: str = "local"
    model_id: str | None = None

    # ── Payload ───────────────────────────────────────────────
    # Stored as a reference to the original ExecutionPayload.
    # We use Any here because ExecutionPayload is in a separate module
    # and frozen dataclasses can't easily cross-reference with defaults.
    payload_messages: tuple[tuple[str, str], ...] = ()  # (role, content) pairs

    # ── Generation Parameters ─────────────────────────────────
    temperature: float = 0.7
    max_tokens: int = 256
    top_p: float = 0.95
    streaming: bool = False

    # ── Modifier Pipeline (ordered descriptors) ───────────────
    modifiers: tuple[Any, ...] = ()  # tuple[ModifierDescriptor, ...]

    # ── Capabilities Resolved ─────────────────────────────────
    capabilities_used: tuple[str, ...] = ()

    # ── Cache ─────────────────────────────────────────────────
    cache_key: str | None = None

    # ── Extensible metadata (runtime-specific key-value) ──────
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_retry(self) -> ExecutionPlan:
        """Create a retry of this plan (same plan_id, incremented version)."""
        # We need to create a new instance since frozen
        return ExecutionPlan(
            plan_id=self.plan_id,
            parent_plan=self.parent_plan,
            trace_id=self.trace_id,
            created_at=_make_timestamp(),
            version=self.version + 1,
            task_type=self.task_type,
            runtime=self.runtime,
            provider=self.provider,
            model_id=self.model_id,
            payload_messages=self.payload_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            streaming=self.streaming,
            modifiers=self.modifiers,
            capabilities_used=self.capabilities_used,
            cache_key=self.cache_key,
            metadata=dict(self.metadata),
        )

    def with_fork(self, **overrides: Any) -> ExecutionPlan:
        """Create a child plan (new plan_id, this plan as parent)."""
        import dataclasses

        defaults = dataclasses.asdict(self)
        defaults.update(overrides)
        defaults["plan_id"] = _make_plan_id()
        defaults["parent_plan"] = self.plan_id
        defaults["created_at"] = _make_timestamp()
        defaults["version"] = 1
        return ExecutionPlan(**defaults)
