"""
HCIR Execution Receipt — execution auditing and verification certificates.

Every cognitive instruction stream execution produces an ``ExecutionReceipt``
that serves as an immutable certificate of what was planned, executed,
verified, and committed.

Receipt Anatomy::

    ExecutionReceipt
    ├── execution_id            (unique run ID)
    ├── process_id / thread_id  (scheduler process binding)
    ├── instruction_stream_hash (SHA256 of bytecode)
    ├── input_snapshot_version  (graph version before run)
    ├── final_snapshot_version  (graph version after run)
    ├── transactions_committed  (list of committed TX IDs)
    ├── transactions_rejected   (list of rejected TX IDs + reasons)
    ├── capabilities_used       (list of capability implementations invoked)
    ├── resources_consumed      (tokens, elapsed_ms, api_calls)
    ├── verification_results    (outcomes of Scope/Schema/Resource/Policy stages)
    └── timestamp               (execution completion time)
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.abi import ExecutionMetrics


@dataclass
class VerificationStageSummary:
    """Summary of a verification stage outcome within a receipt."""

    stage_name: str
    passed: bool
    annotations: list[str] = field(default_factory=list)


@dataclass
class ExecutionReceipt:
    """An immutable certificate of cognitive execution."""

    execution_id: str = field(default_factory=lambda: f"rcpt_{uuid.uuid4().hex[:12]}")
    process_id: str = ""
    thread_id: str = ""
    author: str = ""
    instruction_stream_hash: str = ""
    input_snapshot_version: int = 0
    final_snapshot_version: int = 0
    transactions_committed: list[str] = field(default_factory=list)
    transactions_rejected: list[dict[str, str]] = field(default_factory=list)
    capabilities_used: list[str] = field(default_factory=list)
    metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)
    verification_results: list[VerificationStageSummary] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None
    timestamp: float = field(default_factory=time.time)

    def compute_certificate_hash(self) -> str:
        """Compute SHA256 checksum of the receipt fields for audit verification."""
        payload = json.dumps(
            {
                "execution_id": self.execution_id,
                "author": self.author,
                "stream_hash": self.instruction_stream_hash,
                "input_ver": self.input_snapshot_version,
                "final_ver": self.final_snapshot_version,
                "tx_committed": sorted(self.transactions_committed),
                "capabilities": sorted(self.capabilities_used),
                "success": self.success,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Serialize receipt to a dictionary."""
        return {
            "execution_id": self.execution_id,
            "process_id": self.process_id,
            "thread_id": self.thread_id,
            "author": self.author,
            "instruction_stream_hash": self.instruction_stream_hash,
            "input_snapshot_version": self.input_snapshot_version,
            "final_snapshot_version": self.final_snapshot_version,
            "transactions_committed": self.transactions_committed,
            "transactions_rejected": self.transactions_rejected,
            "capabilities_used": self.capabilities_used,
            "metrics": {
                "elapsed_ms": self.metrics.elapsed_ms,
                "tokens_consumed": self.metrics.tokens_consumed,
                "api_calls": self.metrics.api_calls,
                "memory_bytes": self.metrics.memory_bytes,
            },
            "verification_results": [
                {
                    "stage": v.stage_name,
                    "passed": v.passed,
                    "annotations": v.annotations,
                }
                for v in self.verification_results
            ],
            "outputs": self.outputs,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp,
            "certificate_hash": self.compute_certificate_hash(),
        }


class ReceiptStore:
    """In-memory or persistent store for ExecutionReceipt certificates."""

    def __init__(self) -> None:
        self._receipts: dict[str, ExecutionReceipt] = {}

    def store(self, receipt: ExecutionReceipt) -> None:
        """Store an execution receipt."""
        self._receipts[receipt.execution_id] = receipt

    def get(self, execution_id: str) -> ExecutionReceipt | None:
        """Retrieve a receipt by ID."""
        return self._receipts.get(execution_id)

    def list_by_author(self, author: str) -> list[ExecutionReceipt]:
        """List all receipts generated by a specific author."""
        return [r for r in self._receipts.values() if r.author == author]

    def list_by_process(self, process_id: str) -> list[ExecutionReceipt]:
        """List receipts bound to a cognitive process."""
        return [r for r in self._receipts.values() if r.process_id == process_id]

    @property
    def count(self) -> int:
        return len(self._receipts)
