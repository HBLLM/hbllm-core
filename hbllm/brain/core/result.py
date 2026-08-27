"""Pipeline execution result data structures for HBLLM Brain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineResult:
    """Result of a cognitive pipeline execution."""

    text: str
    correlation_id: str
    source_node: str = "decision"
    confidence: float = 0.0
    tenant_id: str = "default"
    session_id: str = "default"
    latency_ms: float = 0.0
    stages_completed: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "correlation_id": self.correlation_id,
            "source_node": self.source_node,
            "confidence": self.confidence,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "latency_ms": self.latency_ms,
            "stages_completed": self.stages_completed,
            "metadata": self.metadata,
            "error": self.error,
        }
