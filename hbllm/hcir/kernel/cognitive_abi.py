"""
Cognitive ABI Contract — HCIR §10.

Defines standard CapabilityCall, CapabilityResult, ABI_VERSION, and CapabilityManifest
metadata interfaces for versioned capability execution across local, cloud, and swarm kernels.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.workspace import BranchMode

ABI_VERSION = "1.0"


@dataclass
class CapabilityCall:
    """Standard capability call invocation structure."""

    capability_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    abi_version: str = ABI_VERSION
    context_id: str = "default_ctx"
    timestamp: float = field(default_factory=time.time)


@dataclass
class CapabilityResult:
    """Standardized result returned by capability executions."""

    capability_name: str
    status: str = "SUCCESS"
    output: Any = None
    abi_version: str = ABI_VERSION
    receipt_id: str = "rcpt_abi_0"
    mutations: list[dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class CapabilityManifest:
    """Executable metadata describing capability authorization, owner, and sandboxing rules."""

    name: str
    owner: str = "HCIR"
    version: str = "1.0"
    authority: str = "cognitive"
    fallback_backend: str | None = None
    tenant_required: bool = True
    allowed_modes: list[BranchMode] = field(
        default_factory=lambda: [
            BranchMode.LIVE,
            BranchMode.SIMULATION,
            BranchMode.REPLAY,
            BranchMode.TRAINING,
        ]
    )
    sandbox: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)
