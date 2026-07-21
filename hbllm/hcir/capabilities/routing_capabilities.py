"""
Routing & Execution Capabilities — HCIR §10.

Registers routing.resolve_route and execution.sandbox_run capabilities.
"""

from __future__ import annotations

import logging

from hbllm.hcir.kernel.cognitive_abi import CapabilityCall, CapabilityManifest, CapabilityResult

logger = logging.getLogger(__name__)

ROUTING_MANIFESTS = [
    CapabilityManifest(name="routing.resolve_route", owner="HCIR.Router"),
    CapabilityManifest(name="execution.sandbox_run", owner="HCIR.Executor"),
]


def execute_routing_capability(call: CapabilityCall) -> CapabilityResult:
    """Execute a routing or execution capability call."""
    logger.debug("Executing routing capability: %s", call.capability_name)
    output = {
        "capability": call.capability_name,
        "resolved_route": "module.executor.default",
        "sandboxed": True,
    }
    return CapabilityResult(
        capability_name=call.capability_name,
        status="SUCCESS",
        output=output,
    )
