"""Deprecated re-export shim — import from ``hbllm.hcir.identity`` instead.

Kept only for backward compatibility; emits a ``DeprecationWarning`` on import and
will be removed once no external callers remain.
"""

from __future__ import annotations

import warnings

from hbllm.hcir.identity import (
    CausalEvent,
    CausalLink,
    CauseRelation,
    HCIRNamespace,
    HCIRObjectID,
    IdentityCausalGraph,
    IDFactory,
)

warnings.warn(
    "hbllm.hcir.world.identity is deprecated; import from hbllm.hcir.identity",
    DeprecationWarning,
    stacklevel=2,
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
