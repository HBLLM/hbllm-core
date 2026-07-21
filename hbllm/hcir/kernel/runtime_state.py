"""
Runtime State — Tracks active cycle metadata, execution mode, and tenant state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.workspace import BranchMode


@dataclass
class RuntimeState:
    """Active cognitive runtime state container."""

    active_cycle_id: str = "cycle_0"
    cycle_count: int = 0
    branch_mode: BranchMode = BranchMode.LIVE
    tenant_id: str = "default"
    user_id: str = "system"
    current_goal_id: str | None = None
    started_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
