"""Cortical Grounding Bridge — Integrates SGE Grounding into the Cortical Epistemic Loop (Facade).

The authoritative bridge integrating SGE into the Cortical Epistemic Loop.
This facade re-exports all symbols from `hbllm.hcir.world.grounding` for backward compatibility.
"""

from __future__ import annotations

from hbllm.hcir.world.grounding import (
    CorticalGroundingBridge,
    CorticalGroundingState,
    _gen_id,
)

__all__ = [
    "CorticalGroundingBridge",
    "CorticalGroundingState",
    "_gen_id",
]
