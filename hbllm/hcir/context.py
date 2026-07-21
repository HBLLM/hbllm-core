"""
HCIR Execution Context — thread and process execution state.

Analogous to CPU registers / process context in operating systems,
``HCIRExecutionContext`` encapsulates the runtime identity, tenant scope,
resource budgets, simulation state, and causal links for a cognitive
execution thread.

Context Anatomy::

    HCIRExecutionContext
    ├── process_id         (HCIRObjectID of parent process)
    ├── thread_id          (HCIRObjectID of current thread)
    ├── tenant_scope       (Scope identity - tenant, user, security level)
    ├── attention_budget   (Remaining token / compute cost budget)
    ├── causal_parent      (Optional parent CausalEvent ID)
    ├── simulation_branch  (Optional branch name if running in simulation)
    └── temporal_context   (Time bounds, timestamp, circadian state)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.identity import HCIRObjectID
from hbllm.hcir.types import CostMetric, Scope


@dataclass
class HCIRExecutionContext:
    """Runtime context for an active cognitive execution thread."""

    process_id: HCIRObjectID = field(default_factory=HCIRObjectID)
    thread_id: HCIRObjectID = field(default_factory=HCIRObjectID)
    tenant_scope: Scope = field(default_factory=Scope)
    attention_budget: CostMetric = 1000
    causal_parent_id: str | None = None
    simulation_branch: str | None = None
    temporal_context: dict[str, Any] = field(
        default_factory=lambda: {"timestamp": time.time(), "phase": "active"}
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_simulation(self) -> bool:
        """True if executing within a simulation branch."""
        return self.simulation_branch is not None and self.simulation_branch != "main"

    def fork_for_simulation(self, branch_name: str) -> HCIRExecutionContext:
        """Fork context into a simulation branch."""
        return HCIRExecutionContext(
            process_id=self.process_id.next_version(),
            thread_id=self.thread_id.next_version(),
            tenant_scope=self.tenant_scope,
            attention_budget=self.attention_budget // 2,
            causal_parent_id=self.causal_parent_id,
            simulation_branch=branch_name,
            temporal_context=dict(self.temporal_context),
            metadata=dict(self.metadata),
        )

    def consume_budget(self, cost: CostMetric) -> bool:
        """Deduct execution cost. Returns False if budget exhausted."""
        if self.attention_budget < cost:
            return False
        self.attention_budget -= cost
        return True
