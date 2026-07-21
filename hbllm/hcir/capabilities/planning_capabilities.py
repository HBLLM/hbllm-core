"""
Planning Capabilities — HCIR §10.

Registers planning.generate_candidates, planning.validate, and planning.expand
capability backends for CounterfactualPlanner.
"""

from __future__ import annotations

import logging

from hbllm.hcir.kernel.cognitive_abi import CapabilityCall, CapabilityManifest, CapabilityResult

logger = logging.getLogger(__name__)

PLANNING_MANIFESTS = [
    CapabilityManifest(name="planning.generate_candidates", owner="HCIR.Planner"),
    CapabilityManifest(name="planning.validate", owner="HCIR.Planner"),
    CapabilityManifest(name="planning.expand", owner="HCIR.Planner"),
]


def execute_planning_capability(call: CapabilityCall) -> CapabilityResult:
    """Execute a planning capability call."""
    logger.debug("Executing planning capability: %s", call.capability_name)
    output = {
        "capability": call.capability_name,
        "candidates_generated": 2,
        "selected_plan": "candidate_alpha",
    }
    return CapabilityResult(
        capability_name=call.capability_name,
        status="SUCCESS",
        output=output,
    )
