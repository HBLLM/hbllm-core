"""
World Capabilities — Privileged World Kernel Capability Provider Registration.

Registers world capabilities in HCIR KernelCapabilityRegistry:
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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hbllm.hcir.world_kernel import WorldKernel

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

EXTENDED_WORLD_MANIFESTS: list[CapabilityManifest] = [
    *WORLD_MANIFESTS,
    CapabilityManifest(
        name="world.predict_outcome",
        version="1.0",
        owner="world_kernel",
        authority="cognitive",
        tenant_required=True,
    ),
    CapabilityManifest(
        name="world.evaluate_prediction_error",
        version="1.0",
        owner="world_kernel",
        authority="cognitive",
        tenant_required=True,
    ),
    CapabilityManifest(
        name="world.sync_digital_twin",
        version="1.0",
        owner="world_kernel",
        authority="cognitive",
        tenant_required=True,
    ),
]


def execute_world_capability(
    call: CapabilityCall, kernel: WorldKernel | None = None
) -> CapabilityResult:
    """Execute governed World Kernel capability call."""
    name = call.capability_name
    params = getattr(call, "arguments", None) or getattr(call, "parameters", None) or {}

    if name in ("world.predict", "world.predict_outcome", "world_prediction"):
        action_intent = params.get("action_intent", "default_action")
        if kernel is not None:
            from hbllm.hcir.graph import ActionNode

            pred = kernel.predict(ActionNode(intent=action_intent))
            return CapabilityResult(
                capability_name=name,
                status="SUCCESS",
                output={
                    "predicted_outcome": pred.claim,
                    "confidence": pred.uncertainty.confidence if pred.uncertainty else 0.88,
                    "prediction_id": pred.id,
                },
            )
        return CapabilityResult(
            capability_name=name,
            status="SUCCESS",
            output={
                "predicted_outcome": f"Predicted nominal execution of {action_intent}",
                "confidence": 0.88,
            },
        )
    elif name in ("world.evaluate_error", "world.evaluate_prediction_error"):
        if kernel is not None:
            expected = params.get("expected", {})
            actual = params.get("actual", {})
            eval_res = kernel.surprise_engine.evaluate(expected, actual)
            score = getattr(eval_res, "surprise_score", getattr(eval_res, "raw_surprise", 0.0))
            return CapabilityResult(
                capability_name=name,
                status="SUCCESS",
                output={
                    "surprise_score": score,
                    "is_surprising": eval_res.is_surprising,
                },
            )
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
        if kernel is not None:
            st = kernel.get_current_world_state()
            return CapabilityResult(
                capability_name=name,
                status="SUCCESS",
                output={
                    "environment_name": st.environment_name,
                    "overall_status": st.overall_status,
                    "state_status": st.overall_status,
                    "variables": st.variables,
                    "entities": st.entities,
                },
            )
        return CapabilityResult(
            capability_name=name,
            status="SUCCESS",
            output={"overall_status": "nominal", "variables": params.get("variables", {})},
        )
    elif name in ("world.sync_twin", "world.sync_digital_twin"):
        var_name = params.get("variable_name")
        val = params.get("value")
        updates = params.get("updates")
        if kernel is not None:
            if hasattr(kernel, "sync_digital_twin"):
                kernel.sync_digital_twin(variable_name=var_name, value=val, updates=updates)
            elif var_name:
                kernel.digital_twin.sync_sensor_telemetry(var_name, val)
        return CapabilityResult(
            capability_name=name,
            status="SUCCESS",
            output={"synced": True, "variable": var_name, "updates": updates},
        )
    else:
        return CapabilityResult(
            capability_name=name,
            status="FAILED",
            output=None,
        )


class WorldKernelCapabilityExecutor:
    """ICapabilityExecutor adapter binding WorldKernel to CapabilityResolver."""

    def __init__(self, kernel: WorldKernel, default_capability: str | None = None) -> None:
        self.kernel = kernel
        self.default_capability = default_capability

    @property
    def is_available(self) -> bool:
        return True

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        cap = params.get("capability") or self.default_capability or "world.predict_outcome"
        call = CapabilityCall(capability_name=cap, arguments=params)
        res = execute_world_capability(call, self.kernel)
        if res.status == "SUCCESS":
            out = dict(res.output) if isinstance(res.output, dict) else {"result": res.output}
            if "action_intent" in params or cap in (
                "world.predict",
                "world.predict_outcome",
                "world_prediction",
            ):
                out["status"] = "simulated"
                out["predicted"] = True
                out["claim"] = res.output.get("predicted_outcome", "")
            else:
                out["status"] = "success"
            return out
        return {"status": "failed", "error": f"Failed capability execution for {cap}"}
