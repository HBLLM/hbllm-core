"""Object Identity Assignment and Causal Identity Graph (W003, W019).

Re-exports core HCIR identity faculties for world model domain integration.
"""

from __future__ import annotations

from hbllm.hcir.identity import (
    CausalEvent,
    CausalLink,
    CauseRelation,
    HCIRNamespace,
    HCIRObjectID,
    IdentityCausalGraph,
    IDFactory,
)

__all__ = [
    "CausalEvent",
    "CausalLink",
    "CauseRelation",
    "HCIRNamespace",
    "HCIRObjectID",
    "IDFactory",
    "IdentityCausalGraph",
]
