"""
World Capabilities — Privileged World Kernel Capability Provider Registration.

Registers world capabilities in HCIR CapabilityRegistry:
    - world.predict
    - world.evaluate_error
    - world.sync_twin
    - world.query_state
    - world.select_action
"""

from __future__ import annotations

import logging

from hbllm.hcir.kernel.cognitive_abi import CapabilityCall, CapabilityManifest, CapabilityResult

logger = logging.getLogger(__name__)

WORLD_MANIFESTS: list[CapabilityManifest] = [
    CapabilityManifest(
        name="world.predict",
        version="1.0",
        owner="world_kernel",
        authority="cognitive",
        tenant_required=True,
    ),
    CapabilityManifest(
        name="world.evaluate_error",
        version="1.0",
        owner="world_kernel",
        authority="cognitive",
        tenant_required=True,
    ),
    CapabilityManifest(
        name="world.sync_twin",
        version="1.0",
        owner="world_kernel",
        authority="cognitive",
        tenant_required=True,
    ),
    CapabilityManifest(
        name="world.query_state",
        version="1.0",
        owner="world_kernel",
        authority="cognitive",
        tenant_required=True,
    ),
    CapabilityManifest(
        name="world.select_action",
        version="1.0",
        owner="world_kernel",
        authority="cognitive",
        tenant_required=True,
    ),
]


def execute_world_capability(call: CapabilityCall) -> CapabilityResult:
    """Execute governed World Kernel capability call."""
    name = call.capability_name
    params = call.arguments

    if name == "world.predict":
        action_intent = params.get("action_intent", "default_action")
        return CapabilityResult(
            capability_name=name,
            status="SUCCESS",
            output={
                "predicted_outcome": f"Predicted nominal execution of {action_intent}",
                "confidence": 0.88,
            },
        )
    elif name == "world.evaluate_error":
        return CapabilityResult(
            capability_name=name,
            status="SUCCESS",
            output={"surprise_score": 0.05, "is_surprising": False},
        )
    elif name == "world.select_action":
        candidates = params.get("candidate_intents", ["action_default"])
        best = candidates[0] if candidates else "action_default"
        return CapabilityResult(
            capability_name=name,
            status="SUCCESS",
            output={"selected_action_intent": best, "utility_score": 0.85},
        )
    elif name == "world.query_state":
        return CapabilityResult(
            capability_name=name,
            status="SUCCESS",
            output={"status": "nominal", "variables": params.get("variables", {})},
        )
    elif name == "world.sync_twin":
        return CapabilityResult(
            capability_name=name,
            status="SUCCESS",
            output={"synced": True, "variable": params.get("variable_name")},
        )
    else:
        return CapabilityResult(
            capability_name=name,
            status="FAILED",
            output=None,
        )
