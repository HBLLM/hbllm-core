"""
Execution Manifest — declarative execution specification.

The orchestrator resolves Manifest → Plan.
Enables reproducibility, debugging, and configuration-as-code.

Example manifest (conceptual YAML):
    task: text_generation
    runtime: text
    provider: local
    preferred_models:
      - qwen3
      - gemma3
    requirements:
      streaming: true
      json: false
    constraints:
      max_latency_ms: 500
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hbllm.execution.plan import ExecutionConstraints, TaskType


@dataclass
class ExecutionManifest:
    """
    Declarative execution specification.

    The orchestrator resolves this into an ExecutionPlan by:
    1. Evaluating policy against system state
    2. Resolving runtime (or auto-selecting)
    3. Resolving provider (or auto-selecting)
    4. Selecting from preferred models
    5. Negotiating capabilities
    6. Selecting modifiers

    This gives reproducibility and easier debugging compared to
    building plans programmatically.
    """

    task: TaskType = TaskType.TEXT_GENERATION
    runtime: str | None = None  # None = auto-resolve
    provider: str | None = None  # None = auto-resolve
    preferred_models: list[str] = field(default_factory=list)
    requirements: dict[str, bool] = field(default_factory=dict)
    constraints: ExecutionConstraints = field(default_factory=ExecutionConstraints)
    modifier_hints: list[str] = field(default_factory=list)

    def requires_streaming(self) -> bool:
        return self.requirements.get("streaming", False) or self.constraints.require_streaming

    def requires_json(self) -> bool:
        return self.requirements.get("json", False) or self.constraints.require_json
