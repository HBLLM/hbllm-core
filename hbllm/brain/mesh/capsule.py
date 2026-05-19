"""Contextual Cognitive Capsules.

Defines the TaskCapsule which carries minimal executable context
between sovereign nodes, preventing full-state sync explosions.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from hbllm.brain.mesh.registry import TaskPriorityClass


@dataclass
class CognitiveOwnership:
    """Tracks provenance and authority of a delegated task."""

    origin_node: str  # The node that originally created the goal
    authority_node: str  # The sovereign coordinator (e.g., Phone)
    execution_node: str  # The node currently processing the capsule
    verification_node: str  # The node that will verify the result


@dataclass
class TaskCapsule:
    """A minimal, self-contained payload for distributed cognition."""

    capsule_id: str = field(default_factory=lambda: f"cap_{uuid.uuid4().hex[:8]}")
    goal_id: str = ""

    # Ownership & QoS
    ownership: CognitiveOwnership = field(
        default_factory=lambda: CognitiveOwnership("", "", "", "")
    )
    priority: TaskPriorityClass = TaskPriorityClass.BACKGROUND

    # Delegation Safety
    delegation_depth: int = 0
    max_delegation_depth: int = 3
    delegation_trace: list[str] = field(default_factory=list)  # List of node IDs
    expires_at: float = field(default_factory=lambda: time.time() + 300.0)

    # Contextual Cognitive Payload (Minimal State)
    required_entities: dict[str, Any] = field(default_factory=dict)
    causal_dependencies: list[dict[str, Any]] = field(default_factory=list)
    utility_constraints: dict[str, float] = field(default_factory=dict)
    permissions_scope: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if the capsule is safe to process (prevents loops and staleness)."""
        if time.time() > self.expires_at:
            return False
        if self.delegation_depth >= self.max_delegation_depth:
            return False
        return True

    def add_hop(self, node_id: str) -> None:
        """Record a delegation hop to prevent loops."""
        self.delegation_trace.append(node_id)
        self.delegation_depth += 1
