"""
Execution Result — minimal result with separated provider metadata.

ExecutionResult stays minimal: content, artifacts, metrics, usage, status.
Provider-specific data (finish_reason, raw response, model details) lives
in ProviderMetadata, preventing provider API pollution of the core types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenUsage:
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ExecutionMetrics:
    """Execution performance metrics."""

    latency_ms: float = 0.0
    provider_latency_ms: float = 0.0
    modifier_latency_ms: float = 0.0
    serialization_latency_ms: float = 0.0
    modifiers_applied: list[str] = field(default_factory=list)
    cached: bool = False
    retries: int = 0


@dataclass
class ProviderMetadata:
    """
    Provider-specific data. Never leaks into the core API.

    Each provider can put whatever it needs here without
    polluting ExecutionResult.
    """

    provider: str = ""
    model: str = ""
    finish_reason: str = "stop"
    raw: Any = None
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """
    Minimal execution result.

    Keeps the core API clean. Provider-specific details live
    in ProviderMetadata.
    """

    content: str = ""
    artifacts: list[Any] = field(default_factory=list)
    metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)
    usage: TokenUsage = field(default_factory=TokenUsage)
    status: str = "completed"
    plan_id: str | None = None
    provider_metadata: ProviderMetadata | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
