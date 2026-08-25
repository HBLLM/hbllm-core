"""Simulation Events and Deterministic State Hashing for A18.

Defines the formal SimulationEvent structure and event-sourced provenance tracking
for deterministic simulation rollouts and replay.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SimulationEvent:
    """An atomic state transition event in a mental simulation branch."""

    event_id: str = field(default_factory=lambda: f"sev_{uuid.uuid4().hex[:8]}")
    branch_id: str = "main"
    step: int = 0
    operator: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    pre_state_hash: str = ""
    post_state_hash: str = ""
    consequences: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    confidence: float = 1.0
    risk: float = 0.0
    timestamp: float = 0.0


def compute_state_hash(
    nodes: list[tuple[str, str, dict[str, Any]]],
    edges: list[tuple[str, str, str]],  # (source_id, edge_type, target_id)
) -> str:
    """Compute a deterministic SHA-256 hash of a world state snapshot."""
    sorted_nodes = sorted(
        [(nid, ntype, json.dumps(props, sort_keys=True)) for nid, ntype, props in nodes],
        key=lambda x: x[0],
    )
    sorted_edges = sorted(edges, key=lambda e: (e[0], e[1], e[2]))

    payload = json.dumps({"nodes": sorted_nodes, "edges": sorted_edges}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
